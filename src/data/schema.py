# src/data/schema.py

"""
Schema contract for OHLCV price bars.

This file is the single source of truth for bar data structures.
Every module that reads or writes bars — fetchers, the DuckDB store,
strategies, backtests — must import from here rather than define its
own field names or types.  Keeping the contract in one place means a
field rename or type change propagates everywhere automatically.

Two artefacts are defined:
  OHLCVBar         — the in-memory representation of a single bar.
  CREATE_TABLE_SQL — the DuckDB DDL that mirrors OHLCVBar exactly.
  DUCKDB_TABLE_NAME — the canonical table name used throughout the project.
"""

# dataclass turns a plain class into a structured data container by
# auto-generating __init__, __repr__, and __eq__ from the field list.
# frozen=True makes instances immutable (fields cannot be reassigned after
# construction) and adds __hash__, so bars can be stored in sets or used as
# dict keys — useful for deduplication and caching.
from dataclasses import dataclass

# datetime is used for the bar's timestamp field.  All timestamps in this
# project are timezone-aware UTC; naive datetimes are rejected at fetch time.
from datetime import datetime

# ---------------------------------------------------------------------------
# Table name constant — one place to change if the table is ever renamed.
# ---------------------------------------------------------------------------

# Every query and migration in the project references this constant rather
# than a hard-coded string, so a rename requires editing exactly one line.
DUCKDB_TABLE_NAME: str = "ohlcv_bars"


# ---------------------------------------------------------------------------
# DDL — the CREATE TABLE statement that mirrors OHLCVBar field-for-field.
# ---------------------------------------------------------------------------

# Triple-quoted string so the SQL stays readable without line-continuation
# hacks.  The column order matches OHLCVBar declaration order below.
#
# Type mapping (DuckDB → Python):
#   VARCHAR   → str      (symbol, timeframe, source)
#   TIMESTAMP → datetime (timezone-aware; DuckDB stores UTC implicitly)
#   DOUBLE    → float    (open, high, low, close, adj_close)
#   BIGINT    → int      (volume — shares can exceed INT range on busy days)
#
# PRIMARY KEY (symbol, timestamp, timeframe) prevents duplicate bars:
#   - symbol    distinguishes SPY from AAPL, etc.
#   - timestamp pins the bar to a specific point in time.
#   - timeframe allows daily and intraday bars to coexist in the same table
#     without colliding (a 09:30 "1h" bar and a 09:30 "5m" bar are distinct).
#
# IF NOT EXISTS makes the statement idempotent — safe to call at startup
# every time without failing if the table already exists.
CREATE_TABLE_SQL: str = f"""
CREATE TABLE IF NOT EXISTS {DUCKDB_TABLE_NAME} (
    symbol     VARCHAR   NOT NULL,  -- ticker, e.g. "SPY"
    timestamp  TIMESTAMP NOT NULL,  -- bar close time, UTC
    open       DOUBLE    NOT NULL,  -- first trade price in the interval
    high       DOUBLE    NOT NULL,  -- highest trade price in the interval
    low        DOUBLE    NOT NULL,  -- lowest trade price in the interval
    close      DOUBLE    NOT NULL,  -- last trade price in the interval
    adj_close  DOUBLE    NOT NULL,  -- split/dividend-adjusted close
    volume     BIGINT    NOT NULL,  -- shares traded during the interval
    timeframe  VARCHAR   NOT NULL,  -- "1d", "1h", "5m", etc.
    source     VARCHAR   NOT NULL,  -- "yfinance" or "alpaca" — provenance
    PRIMARY KEY (symbol, timestamp, timeframe)
);
""".strip()  # strip() removes the leading newline so logging the SQL looks clean


# ---------------------------------------------------------------------------
# OHLCVBar — the in-memory representation of one price bar.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)  # frozen=True → immutable + hashable
class OHLCVBar:
    """
    One OHLCV price bar for a single symbol and timeframe.

    Field order matches CREATE_TABLE_SQL column order so the two stay in sync
    visually.  All monetary values are raw floats; callers are responsible for
    rounding before display.
    """

    symbol: str        # ticker symbol, e.g. "SPY" or "AAPL"
    timestamp: datetime  # bar close time; must be timezone-aware UTC

    open: float        # first trade price in the interval
    high: float        # highest trade price in the interval
    low: float         # lowest trade price in the interval
    close: float       # last trade price in the interval
    adj_close: float   # split/dividend-adjusted close (equals close for futures/crypto)

    volume: int        # shares traded during the interval; BIGINT in DuckDB

    timeframe: str     # granularity: "1d" daily, "1h" hourly, "5m" five-minute, etc.
    source: str        # data provenance: "yfinance" or "alpaca"
