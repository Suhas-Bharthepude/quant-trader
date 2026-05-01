# tests/test_data_layer.py

"""
Integration tests for the src/data/ layer: YFinanceFetcher and DuckDBStore.

These tests hit the real Yahoo Finance API and write to a temporary DuckDB
file on disk, so they are tagged @pytest.mark.integration to let CI pipelines
skip them without modifying this file:

    pytest -m "not integration"   # fast unit-only run
    pytest -m integration -v      # full data-layer run (needs network)

Run everything with:
    uv run pytest tests/test_data_layer.py -v
"""

# pytest is the test runner.  tmp_path is a built-in pytest fixture that
# provides a temporary directory unique to each test invocation; pytest
# cleans it up automatically when the test session ends.
import pytest

# DuckDBStore is the persistence layer under test.
from src.data.duckdb_store import DuckDBStore

# OHLCVBar is the schema type we assert against — we never import
# DuckDB or yfinance types directly in tests; we only use our own contract.
from src.data.schema import OHLCVBar

# YFinanceFetcher is the fetch layer under test and also acts as the
# fixture supplier for tests that focus on DuckDBStore.
from src.data.yfinance_fetcher import YFinanceFetcher

# ---------------------------------------------------------------------------
# Shared test parameters — defined once so all three tests stay in sync if
# the symbol or date range ever needs to change.
# ---------------------------------------------------------------------------

# NYSE-listed ETF with a long uninterrupted price history — safe reference
# symbol for all data-layer tests.
TEST_SYMBOL: str = "SPY"

# Start date: first trading day of 2024 (2024-01-01 is a holiday).
# Using a known-good historical range means tests are stable regardless of
# when they run — we are not asking for "today's" data.
TEST_START: str = "2024-01-02"

# End date: inclusive.  The range 2024-01-02 to 2024-01-10 contains exactly
# 6 trading days (Mon 2, Tue 3, Thu 4 [Wed was a holiday? No: Thu 4, Fri 5,
# Mon 8, Tue 9 — 6 bars total]).  We assert len > 0 rather than == 6 to
# stay robust if Yahoo's holiday calendar changes slightly.
TEST_END: str = "2024-01-10"


# ---------------------------------------------------------------------------
# Test 1 — YFinanceFetcher contract
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_yfinance_fetcher_returns_bars() -> None:
    """YFinanceFetcher.fetch_daily returns well-formed OHLCVBar instances.

    Verifies:
      - At least one bar is returned for a known trading range.
      - Every item is an OHLCVBar (not a raw pandas row or dict).
      - Every bar carries the correct symbol, timeframe, and source strings.
      - Bars are sorted ascending by timestamp (contract from fetch_daily).
    """
    # Instantiate the fetcher — no arguments needed; stateless adapter.
    fetcher = YFinanceFetcher()

    # Fetch daily bars for the test range.  This makes a live network call
    # to Yahoo Finance; the integration mark allows CI to skip it.
    bars = fetcher.fetch_daily(TEST_SYMBOL, TEST_START, TEST_END)

    # Must return at least one bar — an empty list means the date range
    # produced no trading days, which would be a bug in our inputs.
    assert len(bars) > 0, (
        f"Expected at least one bar for {TEST_SYMBOL} {TEST_START}–{TEST_END}, "
        f"got {len(bars)}"
    )

    # Every element must be an OHLCVBar instance — not a raw tuple, dict, or
    # pandas Series.  This guards against silent type regressions.
    for bar in bars:
        assert isinstance(bar, OHLCVBar), (
            f"Expected OHLCVBar, got {type(bar)} for bar: {bar!r}"
        )

    # Every bar must carry the correct symbol provenance string.
    for bar in bars:
        assert bar.symbol == TEST_SYMBOL, (
            f"Expected symbol={TEST_SYMBOL!r}, got {bar.symbol!r}"
        )

    # Every bar must be tagged as a daily bar.
    for bar in bars:
        assert bar.timeframe == "1d", (
            f"Expected timeframe='1d', got {bar.timeframe!r}"
        )

    # Every bar must carry the fetcher's source tag so downstream code
    # can identify where the data came from.
    for bar in bars:
        assert bar.source == "yfinance", (
            f"Expected source='yfinance', got {bar.source!r}"
        )

    # Bars must be sorted ascending — fetch_daily's explicit sort guarantee.
    # zip(bars, bars[1:]) pairs each bar with the next one for comparison.
    for earlier, later in zip(bars, bars[1:]):
        assert earlier.timestamp <= later.timestamp, (
            f"Bars are not sorted: {earlier.timestamp} > {later.timestamp}"
        )


# ---------------------------------------------------------------------------
# Test 2 — DuckDBStore round-trip write then read
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_duckdb_store_roundtrip(tmp_path) -> None:
    """DuckDBStore.write_bars + read_bars returns the same bars that were written.

    Uses pytest's tmp_path fixture so the DuckDB file is isolated to this test
    and cleaned up automatically — no stale state from a previous run.

    Verifies:
      - write_bars() reports inserting exactly len(bars) rows on the first call.
      - read_bars() over the same date range returns the same number of bars.
      - The first and last timestamps round-trip correctly.
    """
    # Build a path to a temp DuckDB file.  tmp_path is a pytest-managed
    # temporary directory; appending "test.duckdb" gives us an isolated file
    # that will not collide with the project's real data/quant_trader.duckdb.
    db_file: str = str(tmp_path / "test.duckdb")

    # Fetch bars to use as write input.  We do this outside the context
    # manager so a network error surfaces as a separate, descriptive failure
    # rather than being swallowed inside the DuckDB context.
    fetcher = YFinanceFetcher()
    bars = fetcher.fetch_daily(TEST_SYMBOL, TEST_START, TEST_END)

    # Expected bar count — captured once so all assertions reference the same
    # value.
    expected_count: int = len(bars)

    # Open the store as a context manager so the file lock is always released,
    # even if an assertion raises mid-test.
    with DuckDBStore(db_file) as store:
        # write_bars() must insert every bar on the first call — no duplicates
        # exist yet, so the before/after count delta should equal len(bars).
        inserted = store.write_bars(bars)
        assert inserted == expected_count, (
            f"Expected {expected_count} inserts, got {inserted}"
        )

        # read_bars() over the same symbol and date range must return all bars.
        result = store.read_bars(TEST_SYMBOL, TEST_START, TEST_END)
        assert len(result) == expected_count, (
            f"Expected {expected_count} bars from read_bars, got {len(result)}"
        )

        # Compare calendar dates, not exact datetimes.  DuckDB's TIMESTAMP type
        # is timezone-naive: it strips timezone info on write, so a yfinance
        # timestamp at 05:00+00:00 (midnight New York) round-trips as
        # 00:00+00:00.  The calendar date — what daily bars actually identify —
        # is preserved correctly, so .date() is the right comparison target.
        assert result[0].timestamp.date() == bars[0].timestamp.date(), (
            f"First date mismatch: wrote {bars[0].timestamp.date()}, "
            f"read back {result[0].timestamp.date()}"
        )

        # Same reasoning for the last bar.
        assert result[-1].timestamp.date() == bars[-1].timestamp.date(), (
            f"Last date mismatch: wrote {bars[-1].timestamp.date()}, "
            f"read back {result[-1].timestamp.date()}"
        )


# ---------------------------------------------------------------------------
# Test 3 — DuckDBStore idempotency (no duplicate rows on repeated writes)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_duckdb_store_is_idempotent(tmp_path) -> None:
    """Writing the same bars twice inserts them only once.

    Relies on INSERT OR IGNORE + the PRIMARY KEY (symbol, timestamp, timeframe)
    constraint in CREATE_TABLE_SQL.  A regression here would mean duplicated
    price data in downstream calculations — a silent but serious data quality bug.

    Verifies:
      - First write inserts N rows.
      - Second write with identical bars inserts 0 rows.
      - read_bars() returns exactly N bars (not 2N).
    """
    # Isolated temp file — same pattern as test_duckdb_store_roundtrip.
    db_file: str = str(tmp_path / "test.duckdb")

    # Fetch bars once; we will write the same list twice.
    fetcher = YFinanceFetcher()
    bars = fetcher.fetch_daily(TEST_SYMBOL, TEST_START, TEST_END)

    # Capture expected count before any writes.
    expected_count: int = len(bars)

    with DuckDBStore(db_file) as store:
        # First write — all rows are new, so inserted == N.
        first_insert = store.write_bars(bars)
        assert first_insert == expected_count, (
            f"First write should insert {expected_count} rows, got {first_insert}"
        )

        # Second write with the exact same bars — all keys already exist,
        # so INSERT OR IGNORE discards every row and returns 0.
        second_insert = store.write_bars(bars)
        assert second_insert == 0, (
            f"Second write should insert 0 rows (duplicates), got {second_insert}"
        )

        # Read back the full range — must be N bars, not 2N.
        result = store.read_bars(TEST_SYMBOL, TEST_START, TEST_END)
        assert len(result) == expected_count, (
            f"After two writes expected {expected_count} bars, "
            f"got {len(result)} (possible duplicates)"
        )
