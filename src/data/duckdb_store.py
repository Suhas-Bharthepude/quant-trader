# src/data/duckdb_store.py

"""
Persistence layer for OHLCV bars using DuckDB.

DuckDB is an embedded analytical SQL database — think SQLite, but designed
for column-oriented analytics rather than OLTP row operations.  It lives in
a single file on disk, requires no server process, and is extremely fast for
aggregation and range queries over time-series data.

Concurrency constraint: DuckDB allows only one writer at a time.  If two
processes open the same .duckdb file simultaneously, the second will raise a
connection error.  For this project — single-process daily ingest — that is
fine.  We never run concurrent ingest jobs against the same file.

Typical usage (context manager is preferred — guarantees the file lock is
released even if an exception occurs mid-ingest):

    from src.data.duckdb_store import DuckDBStore
    from src.data.yfinance_fetcher import YFinanceFetcher

    bars = YFinanceFetcher().fetch_daily("SPY", "2024-01-01", "2024-12-31")

    with DuckDBStore() as store:
        inserted = store.write_bars(bars)
        print(f"Inserted {inserted} new bars")
        result = store.read_bars("SPY", "2024-01-01", "2024-03-31")
"""

# timezone.utc is used when re-attaching timezone info to timestamps that
# DuckDB may return as naive datetimes depending on the driver version.
from datetime import datetime, timezone

# logging lets write_bars warn (not raise) when it skips a corrupt bar — same
# stdlib logging the rest of the data layer uses (yfinance_fetcher, cli_common).
import logging

# math.isfinite() is the canonical "is this a real finite number?" test — False
# for NaN and +/- infinity, True otherwise.  Used to screen close/adj_close on write.
import math

# Path makes cross-platform directory creation one concise call.
from pathlib import Path

# duckdb is the embedded analytical database.  Install with: uv add duckdb
import duckdb

# Import the three schema artefacts this module needs:
#   OHLCVBar         — the in-memory bar type we read/write
#   DUCKDB_TABLE_NAME — the canonical table name string ("ohlcv_bars")
#   CREATE_TABLE_SQL  — the idempotent CREATE TABLE IF NOT EXISTS statement
from src.data.schema import CREATE_TABLE_SQL, DUCKDB_TABLE_NAME, OHLCVBar

# Module-level logger.  getLogger(__name__) ties records to this module so they
# inherit whatever handler/level the application configures — matching the
# logger setup in yfinance_fetcher and research/cli_common.
log = logging.getLogger(__name__)


class DuckDBStore:
    """Read/write OHLCVBar objects to a DuckDB file on disk."""

    def __init__(self, db_path: str = "data/quant_trader.duckdb") -> None:
        """Open (or create) the DuckDB file and ensure the table exists.

        Parameters
        ----------
        db_path : str
            Path to the .duckdb file, relative to the working directory.
            Defaults to data/quant_trader.duckdb so `uv run` from the repo
            root writes into the existing data/ scaffold folder.
        """
        # Remember the path for diagnostics and __repr__.
        self._db_path: str = db_path

        # Create the parent directory before duckdb.connect() tries to create
        # the file inside it.  parents=True handles nested paths; exist_ok=True
        # is a no-op when the directory already exists — safe to call every run.
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # duckdb.connect() opens an existing file or creates a new empty one.
        # The connection holds an exclusive write lock on the file until close()
        # is called, so other processes cannot write while this store is open.
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(db_path)

        # Always run schema setup at startup — CREATE TABLE IF NOT EXISTS is
        # idempotent, so this is safe on every open regardless of state.
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create the ohlcv_bars table if it does not already exist.

        Runs CREATE_TABLE_SQL which uses IF NOT EXISTS — calling this on an
        already-initialised database is a harmless no-op.
        """
        # No parameters needed: CREATE_TABLE_SQL contains only constant
        # identifiers from our own schema module, not user-supplied data.
        self._conn.execute(CREATE_TABLE_SQL)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_bars(self, bars: list[OHLCVBar]) -> int:
        """Insert bars, skipping any that already exist (idempotent).

        Duplicates are detected via the PRIMARY KEY (symbol, timestamp,
        timeframe).  INSERT OR IGNORE silently discards a row when its key
        already exists rather than raising an error.  This makes the ingest
        pipeline idempotent: running fetch + write twice produces no
        duplicates and no exceptions.

        Parameters
        ----------
        bars : list[OHLCVBar]
            Bars to persist.  An empty list returns 0 immediately.

        Returns
        -------
        int
            Number of rows actually written to disk (0 if all duplicates).
        """
        if not bars:
            return 0  # fast exit — skip the round-trip to DuckDB

        # Screen out bars with a non-finite close OR adj_close before they reach
        # the insert.  A NaN/+-inf price clears the NOT NULL / PRIMARY KEY
        # constraints (NaN is a valid DOUBLE), so without this guard a corrupt
        # trailing bar is written straight through and later poisons the
        # backtester's log-return math.  We DROP the bad bar and keep the good
        # ones — a multi-symbol ingest must not abort because one symbol's
        # trailing bar is garbage; we warn instead of raising.
        clean_bars: list[OHLCVBar] = []  # the surviving finite bars, to be written
        for b in bars:
            # math.isfinite is False for NaN and both infinities; a bar is bad
            # if EITHER price field fails the test.
            if not math.isfinite(b.close) or not math.isfinite(b.adj_close):
                # Warn (do not raise) naming the symbol and the bar's date so the
                # operator can see exactly which row was skipped during ingest.
                log.warning(
                    "Skipping bar with non-finite price: symbol=%s date=%s "
                    "close=%r adj_close=%r",
                    b.symbol,
                    b.timestamp.date().isoformat(),
                    b.close,
                    b.adj_close,
                )
                continue  # drop this bar; do not append it to clean_bars
            clean_bars.append(b)  # finite on both fields — keep it

        # If every incoming bar was bad there is nothing to write; return 0 so
        # the count reflects rows actually written and we skip the DB round-trip.
        if not clean_bars:
            return 0

        # Snapshot row count before the insert.  The difference after gives
        # us the number of rows that were not duplicates.
        # f-string is safe here: DUCKDB_TABLE_NAME is a module constant, not
        # user input.  We never interpolate user data into SQL strings.
        before: int = self._conn.execute(
            f"SELECT COUNT(*) FROM {DUCKDB_TABLE_NAME}"
        ).fetchone()[0]

        # Parameterized INSERT — ? placeholders are bound by DuckDB's driver,
        # which handles quoting, escaping, and type coercion correctly.
        # Never build VALUES with f-strings: that opens SQL-injection paths
        # even for non-malicious data (e.g. a ticker like "O'Reilly").
        sql = (
            f"INSERT OR IGNORE INTO {DUCKDB_TABLE_NAME} "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        # executemany() binds each tuple to the ? placeholders and runs one
        # INSERT per tuple — significantly faster than calling execute() in
        # a Python loop because DuckDB batches the work internally.  We pass
        # clean_bars (not bars) so only the finite-price rows are written.
        self._conn.executemany(sql, [self._bar_to_tuple(b) for b in clean_bars])

        # Row count after insert; delta = new rows added.
        after: int = self._conn.execute(
            f"SELECT COUNT(*) FROM {DUCKDB_TABLE_NAME}"
        ).fetchone()[0]

        return after - before

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1d",
    ) -> list[OHLCVBar]:
        """Return stored bars for a symbol/timeframe within a date range.

        Parameters
        ----------
        symbol : str
            Ticker symbol, e.g. "SPY".
        start : str
            Inclusive start date in ISO format, e.g. "2024-01-01".
        end : str
            Inclusive end date in ISO format, e.g. "2024-12-31".
        timeframe : str
            Bar granularity.  Defaults to "1d" (daily).

        Returns
        -------
        list[OHLCVBar]
            Matching bars sorted by timestamp ascending.  Empty list if none.
        """
        # We compare CAST(timestamp AS DATE) rather than the raw TIMESTAMP so
        # that bars stored at non-midnight UTC offsets (e.g. 05:00 UTC for a
        # NYSE daily bar) still match their calendar date correctly.
        # CAST(? AS DATE) lets DuckDB parse the ISO string "2024-01-01" into
        # a DATE value, which then compares cleanly against the truncated ts.
        sql = f"""
            SELECT
                symbol, timestamp, open, high, low, close,
                adj_close, volume, timeframe, source
            FROM {DUCKDB_TABLE_NAME}
            WHERE symbol    = ?
              AND timeframe = ?
              AND CAST(timestamp AS DATE)
                  BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ORDER BY timestamp ASC
        """

        # Positional binding: symbol → first ?, timeframe → second ?,
        # start → third ?, end → fourth ?.
        rows = self._conn.execute(sql, [symbol, timeframe, start, end]).fetchall()

        # Convert each raw tuple back to an OHLCVBar using the helper below.
        return [self._tuple_to_bar(row) for row in rows]

    def last_timestamp(self, symbol: str, timeframe: str = "1d") -> datetime | None:
        """Return the timestamp of the most recent stored bar for symbol, or None if no bars exist.

        Used by the incremental updater to compute the start date for the next fetch.
        """
        # Aggregate query: MAX(timestamp) returns the single latest timestamp for
        # this symbol+timeframe pair, or SQL NULL when no matching rows exist.
        sql = (
            f"SELECT MAX(timestamp) FROM {DUCKDB_TABLE_NAME} "
            "WHERE symbol = ? AND timeframe = ?"
        )

        # Bind symbol and timeframe as parameters — never interpolate user data
        # into SQL strings.  fetchone() always returns a one-element tuple even
        # for scalar aggregates.
        result = self._conn.execute(sql, [symbol, timeframe]).fetchone()

        # result[0] is None when no rows matched (MAX of an empty set is SQL NULL).
        if result[0] is None:
            # No bars stored for this symbol+timeframe yet; caller interprets as
            # "fetch from the beginning of history".
            return None

        # Pull the datetime value out of the single-element tuple.
        ts = result[0]

        # Re-attach UTC if the driver returns a naive datetime — same guard used
        # in _tuple_to_bar(); our contract requires all timestamps to be UTC-aware.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Return the timezone-aware datetime of the most recent stored bar.
        return ts

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the connection and release the exclusive file lock.

        Always call this when finished — other processes cannot open the
        file for writing while this connection is held.  The context manager
        calls close() automatically via __exit__.
        """
        self._conn.close()

    def __enter__(self) -> "DuckDBStore":
        """Support `with DuckDBStore() as store:` — returns self."""
        return self  # connection was opened in __init__; nothing extra needed

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the connection on context-manager exit, success or failure.

        Returning None (implicitly False) means exceptions are not suppressed —
        any error raised inside the `with` block propagates normally.
        """
        self.close()

    # ------------------------------------------------------------------
    # Private serialization helpers
    # ------------------------------------------------------------------

    def _bar_to_tuple(self, bar: OHLCVBar) -> tuple:
        """Serialize one OHLCVBar to a plain tuple in table-column order.

        Column order mirrors CREATE_TABLE_SQL:
        symbol, timestamp, open, high, low, close,
        adj_close, volume, timeframe, source.

        Timezone normalization: DuckDB's TIMESTAMP column is timezone-naive,
        and the Python driver implicitly converts a tz-aware datetime to the
        host machine's LOCAL time before stripping the tzinfo.  That means a
        bar at 2024-01-05 00:00 UTC written from a US/Eastern machine would
        be stored as 2024-01-04 19:00 — silently shifting the calendar date.
        We defend against that by converting to UTC and dropping tzinfo
        ourselves, so the value DuckDB stores is always the UTC wall-clock
        time regardless of the host's local timezone.
        """
        # Normalize to UTC wall-clock time, then strip tzinfo so the driver
        # treats the value as already-naive and skips its local-time conversion.
        # astimezone() on an aware datetime is a pure timezone shift; replace()
        # then drops the tzinfo without further altering the wall-clock time.
        ts = bar.timestamp.astimezone(timezone.utc).replace(tzinfo=None)

        # Floor daily bars to midnight UTC of their UTC date.
        # Why: a 1d bar is identified by its trading date, not its time-of-day.
        # yfinance has historically returned both naive timestamps (stored as
        # 00:00 UTC via the replace() branch in the fetcher) and timezone-aware
        # America/New_York timestamps (stored as 05:00 UTC in winter, 04:00 in
        # summer via the astimezone() branch).  Without this floor, both arrive
        # here as distinct naive UTC times, both pass the PRIMARY KEY constraint
        # on (symbol, timestamp, timeframe), and INSERT OR IGNORE lets both
        # through — producing one duplicate row per date per tz-convention
        # change.  Flooring to midnight UTC collapses any same-date bar to a
        # single canonical key so the PK enforces one row per (symbol, date),
        # and a double-ingest under a different tz convention is silently
        # deduplicated at the write choke point.
        # For US equity daily bars the UTC date always equals the trading date
        # (NYSE opens at 09:30 ET; midnight UTC is well before that), so no
        # date shift occurs.  Intraday bars are left untouched: their sub-day
        # time-of-day is meaningful and must not be floored.
        if bar.timeframe == "1d":
            ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)

        return (
            bar.symbol,
            ts,               # UTC-normalized naive datetime — see docstring above
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.adj_close,
            bar.volume,
            bar.timeframe,
            bar.source,
        )

    def _tuple_to_bar(self, row: tuple) -> OHLCVBar:
        """Deserialize one raw database row tuple back into an OHLCVBar.

        Index positions match the SELECT column list in read_bars():
        0=symbol  1=timestamp  2=open   3=high  4=low    5=close
        6=adj_close  7=volume  8=timeframe  9=source
        """
        ts = row[1]  # DuckDB returns TIMESTAMP as a Python datetime

        # Re-attach UTC if the driver returns a naive datetime.  This guards
        # against driver version differences — our contract requires UTC-aware.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return OHLCVBar(
            symbol=row[0],
            timestamp=ts,
            open=row[2],
            high=row[3],
            low=row[4],
            close=row[5],
            adj_close=row[6],
            volume=row[7],
            timeframe=row[8],
            source=row[9],
        )
