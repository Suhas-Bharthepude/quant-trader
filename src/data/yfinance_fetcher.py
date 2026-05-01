# src/data/yfinance_fetcher.py

"""
Thin adapter around the yfinance library for fetching daily OHLCV bars.

yfinance hits Yahoo Finance's unofficial API — it is unauthenticated and free,
but rate-limited to roughly 2 000 requests per hour.  Stay well under that
ceiling by batching symbols and caching results rather than re-fetching.

Design principle: this fetcher is pure read — it fetches, converts, and returns.
It does NOT write to DuckDB, cache to disk, or do anything stateful.
Caching and persistence are the responsibility of the layer above this one.
Any code that needs historical bars should call the store layer, which calls
this fetcher only on a cache miss.
"""

# timezone.utc is the sentinel we attach to naive timestamps and the target
# we convert timezone-aware timestamps into.  Every datetime in this project
# is UTC; mixing timezones downstream is a bug.
from datetime import timezone

# yfinance is the third-party library that wraps Yahoo Finance's market data.
# Install with: uv add yfinance
import yfinance

# OHLCVBar is the project-wide schema contract.  This fetcher's only job is
# to convert yfinance rows into OHLCVBar instances — nothing else.
from src.data.schema import OHLCVBar


class YFinanceFetcher:
    """Fetch daily OHLCV bars from Yahoo Finance via the yfinance library."""

    # Class-level constant so every bar produced here carries consistent provenance.
    # Using a class attribute (not instance attribute) means it cannot drift per-instance.
    SOURCE: str = "yfinance"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_daily(self, symbol: str, start: str, end: str) -> list[OHLCVBar]:
        """Fetch daily OHLCV bars for symbol between start and end (inclusive).

        Parameters
        ----------
        symbol : str
            Ticker symbol, e.g. "SPY" or "AAPL".
        start : str
            ISO date string for the first bar, e.g. "2024-01-01".
        end : str
            ISO date string for the last bar, e.g. "2024-12-31".
            yfinance treats this as inclusive when interval="1d".

        Returns
        -------
        list[OHLCVBar]
            Bars sorted by timestamp ascending, one per trading day.

        Raises
        ------
        ValueError
            If yfinance returns an empty DataFrame — typically a bad ticker,
            a future date range, or a period with no trading activity.
        """
        # Ticker wraps all yfinance API calls for a single symbol.
        ticker = yfinance.Ticker(symbol)

        # history() fetches OHLCV data from Yahoo Finance.
        # auto_adjust=False: keep raw OHLC prices unchanged; we store both
        #   the raw close and the split/dividend-adjusted close separately,
        #   so we need the unadjusted data to populate the close field.
        # actions=False: skip the Dividends and Stock Splits columns — we
        #   don't need them here and they clutter the DataFrame.
        df = ticker.history(
            start=start,
            end=end,
            interval="1d",        # one row per trading day
            auto_adjust=False,    # keep raw OHLC; adj_close comes from "Adj Close" column
            actions=False,        # exclude dividend and split event columns
        )

        # An empty DataFrame means yfinance found nothing — bad ticker,
        # holiday-only range, or Yahoo's API is misbehaving.  Fail loudly
        # so callers know immediately rather than silently returning [].
        if df.empty:
            raise ValueError(
                f"No data returned for {symbol!r} between {start!r} and {end!r}. "
                "Check the ticker symbol and that the date range contains trading days."
            )

        # iterrows() yields (index_value, Series) pairs.  The index is a
        # DatetimeIndex, so each index_value is a pd.Timestamp representing
        # the bar date.  We pass both to _row_to_bar so the helper is pure
        # (no DataFrame access) and therefore easy to test in isolation.
        bars: list[OHLCVBar] = [
            self._row_to_bar(symbol, ts, row)
            for ts, row in df.iterrows()
        ]

        # Sort ascending by timestamp so callers always get chronological order
        # regardless of what yfinance returns (usually already sorted, but
        # defensive sorting is cheap and makes the contract explicit).
        return sorted(bars, key=lambda b: b.timestamp)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _row_to_bar(self, symbol: str, timestamp, row) -> OHLCVBar:
        """Convert one DataFrame row into an OHLCVBar.

        Parameters
        ----------
        symbol : str
            Ticker — passed through unchanged from fetch_daily.
        timestamp : pd.Timestamp
            The DatetimeIndex entry for this row (the bar date).
        row : pd.Series
            One row from the yfinance history DataFrame.
        """
        # pd.Timestamp.to_pydatetime() converts to a standard Python datetime.
        # We need a plain datetime (not a pandas type) because OHLCVBar is a
        # pure Python dataclass — no pandas dependency in the schema layer.
        ts = timestamp.to_pydatetime()

        # yfinance may return timezone-aware or timezone-naive timestamps
        # depending on the version and the exchange.  We normalise to UTC
        # either way so the rest of the project never has to think about it.
        if ts.tzinfo is None:
            # Naive timestamp — assume UTC.  Daily bars from Yahoo are
            # midnight UTC dates, so this assumption is correct for equities.
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            # Already timezone-aware — convert to UTC in case it is localised
            # to a market timezone such as America/New_York.
            ts = ts.astimezone(timezone.utc)

        # float() guards against numpy scalar types (np.float64, etc.) that
        # yfinance returns; OHLCVBar fields are typed as plain Python float.
        # int() does the same for Volume, which arrives as np.int64.
        return OHLCVBar(
            symbol=symbol,
            timestamp=ts,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            adj_close=float(row["Adj Close"]),  # "Adj Close" is the yfinance column name
            volume=int(row["Volume"]),
            timeframe="1d",        # this fetcher only supports daily bars
            source=self.SOURCE,    # "yfinance" — provenance for the persistence layer
        )
