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

# datetime + timezone + date — needed to construct synthetic UTC-aware
# timestamps for unit tests that do not hit the network.
from datetime import date, datetime, timezone

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
        # is timezone-naive, but our store layer normalizes to UTC on write
        # (see _bar_to_tuple) and re-attaches UTC on read (see _tuple_to_bar),
        # so the round-trip preserves the wall-clock UTC time.  We still
        # compare with .date() because that is what a daily bar fundamentally
        # identifies — the time-of-day component is a source-specific detail.
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


# ---------------------------------------------------------------------------
# Test 4 — DuckDBStore.last_timestamp (unit test, no network)
# ---------------------------------------------------------------------------
# Deliberately NOT marked @pytest.mark.integration: this test constructs its
# OHLCVBars in-memory and writes them to a tmp_path DuckDB file, so it runs
# fast and works in CI environments with no internet access.

def test_last_timestamp_returns_max(tmp_path) -> None:
    """last_timestamp returns the most recent timestamp for a symbol or None if absent."""
    # Isolated temp DuckDB file — same tmp_path pattern as the round-trip tests
    # above, so this test has no dependency on the project's real data file.
    db_file: str = str(tmp_path / "test.duckdb")

    # Build three synthetic bars for symbol "TEST" with timestamps in
    # deliberately non-chronological insert order (Jan 2, then Jan 5, then Jan 3).
    # If last_timestamp ever regressed to "the most recently inserted row" it
    # would return Jan 3; the assertion below catches that by requiring Jan 5
    # (the true MAX).
    bars = [
        # First bar — earliest date (Jan 2).  All non-key fields are dummy
        # values; this test only exercises the timestamp-aggregation logic.
        OHLCVBar(
            symbol="TEST",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=100.0, high=101.0, low=99.0, close=100.5, adj_close=100.5,
            volume=1000, timeframe="1d", source="test",
        ),
        # Second bar — the latest date (Jan 5), inserted in the middle of the
        # sequence so MAX-vs-LAST distinguishes between correct and broken impls.
        OHLCVBar(
            symbol="TEST",
            timestamp=datetime(2024, 1, 5, tzinfo=timezone.utc),
            open=100.0, high=101.0, low=99.0, close=100.5, adj_close=100.5,
            volume=1000, timeframe="1d", source="test",
        ),
        # Third bar — Jan 3, inserted last but NOT the MAX timestamp.
        OHLCVBar(
            symbol="TEST",
            timestamp=datetime(2024, 1, 3, tzinfo=timezone.utc),
            open=100.0, high=101.0, low=99.0, close=100.5, adj_close=100.5,
            volume=1000, timeframe="1d", source="test",
        ),
    ]

    # Open the store as a context manager so the file lock is always released,
    # even if an assertion fails mid-test.
    with DuckDBStore(db_file) as store:
        # Persist all three bars in one batched write.  The return value is
        # not asserted here — the round-trip test already covers write counts.
        store.write_bars(bars)

        # Assertion 1: last_timestamp must return the MAX (Jan 5), not the
        # most-recently-inserted row (Jan 3).  Comparing .date() sidesteps
        # any timezone-fuzziness in the driver's TIMESTAMP round-trip.
        assert store.last_timestamp("TEST").date() == date(2024, 1, 5), (
            "last_timestamp should return MAX(timestamp), not most-recent insert"
        )

        # Assertion 2: a symbol that has no rows in the table must yield None,
        # not an error or a sentinel datetime — callers depend on the None
        # branch to decide between backfill and update.
        assert store.last_timestamp("UNKNOWN_SYMBOL") is None, (
            "last_timestamp must return None for symbols not present in the DB"
        )

        # Assertion 3: the same symbol but a different timeframe must also
        # yield None — the WHERE clause filters on (symbol, timeframe), and a
        # leak across timeframes would corrupt incremental updates that mix
        # daily and intraday bars in the same table.
        assert store.last_timestamp("TEST", timeframe="1h") is None, (
            "last_timestamp must filter by timeframe, not symbol alone"
        )


# ---------------------------------------------------------------------------
# Test 5 — Timezone-duplicate regression (no network)
# ---------------------------------------------------------------------------
# Reproduces the exact production bug: SPY had 505 date-duplicate rows because
# one ingest run stored daily bars at 00:00 UTC (yfinance naive timestamps) and
# a second run stored the same bars at 05:00 UTC (yfinance America/New_York
# timestamps, winter/EST = UTC-5).  The PRIMARY KEY (symbol, timestamp,
# timeframe) did not catch them because 00:00 != 05:00.
# The fix floors daily bars to midnight UTC in _bar_to_tuple so both arrive
# with the same stored timestamp and INSERT OR IGNORE deduplicates them.

def test_daily_bar_tz_duplicate_is_rejected(tmp_path) -> None:
    """Writing the same daily bar under two tz conventions inserts exactly one row.

    Verifies:
      - A bar at 2024-01-02 00:00:00 UTC (naive-timestamp yfinance path) is stored.
      - Writing the same bar at 2024-01-02 05:00:00 UTC (Eastern-midnight yfinance
        path, winter/EST = UTC-5) is silently discarded by INSERT OR IGNORE.
      - read_bars() returns exactly 1 row for the date, timestamped at midnight UTC.

    Without the _bar_to_tuple daily-floor fix, read_bars() returns 2 rows because
    both 00:00 and 05:00 clear the PRIMARY KEY constraint.
    """
    # Isolated temp DuckDB file — no dependency on the real data/quant_trader.duckdb.
    db_file: str = str(tmp_path / "test.duckdb")

    # The shared OHLCV values are identical between the two bars; only the
    # timestamp tzinfo/offset differs — matching the confirmed production finding
    # that the 220 "differing" pairs differ only in adj_close at ~1e-5 float
    # noise, with open/high/low/close/volume byte-identical.  We use exactly
    # equal values here to isolate the timestamp dimension.
    common_fields = dict(
        symbol="SPY",
        open=476.01, high=479.20, low=474.68, close=476.90, adj_close=476.90,
        volume=92_000_000, timeframe="1d", source="test",
    )

    # Bar A: midnight UTC — the "naive timestamp" ingest path.
    # yfinance returned a tz-naive pd.Timestamp; the fetcher called
    # ts.replace(tzinfo=timezone.utc), producing 2024-01-02 00:00:00+00:00.
    bar_midnight_utc = OHLCVBar(
        timestamp=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
        **common_fields,
    )

    # Bar B: 05:00 UTC — the "Eastern-aware timestamp" ingest path.
    # yfinance returned a tz-aware pd.Timestamp('2024-01-02 00:00:00-05:00',
    # tz='America/New_York'); the fetcher called ts.astimezone(timezone.utc),
    # producing 2024-01-02 05:00:00+00:00.  This is winter (EST = UTC-5).
    bar_eastern_midnight = OHLCVBar(
        timestamp=datetime(2024, 1, 2, 5, 0, 0, tzinfo=timezone.utc),
        **common_fields,
    )

    with DuckDBStore(db_file) as store:
        # Write bar A first — must be accepted (new row).
        first_insert = store.write_bars([bar_midnight_utc])
        assert first_insert == 1, (
            f"First write (midnight UTC) should insert 1 row, got {first_insert}"
        )

        # Write bar B — must be silently discarded because _bar_to_tuple floors
        # both bars to 2024-01-02 00:00:00 (naive UTC), and the PK already holds
        # that key.  Without the floor, 05:00 != 00:00 and both get stored.
        second_insert = store.write_bars([bar_eastern_midnight])
        assert second_insert == 0, (
            f"Second write (Eastern midnight = 05:00 UTC) should insert 0 rows "
            f"(duplicate after daily floor), got {second_insert}"
        )

        # Read back the date range — must yield exactly 1 bar, not 2.
        result = store.read_bars("SPY", "2024-01-02", "2024-01-02")
        assert len(result) == 1, (
            f"Expected exactly 1 bar for 2024-01-02, got {len(result)} "
            "(tz-duplicate rows were not deduplicated on write)"
        )

        # The stored timestamp must be midnight UTC — the canonical daily key.
        assert result[0].timestamp == datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc), (
            f"Expected stored timestamp 2024-01-02 00:00:00+00:00, "
            f"got {result[0].timestamp}"
        )


# ---------------------------------------------------------------------------
# Test 6 — write_bars drops bars with non-finite prices (no network)
# ---------------------------------------------------------------------------
# In-memory only (no @pytest.mark.integration): exercises the write-boundary
# guard that screens NaN/+-inf close or adj_close before insert.  The known
# production defect was a corrupt trailing bar with NaN close/adj_close; this
# pins down that write_bars silently skips such rows (warns, does not raise)
# while still persisting the good bars in the same batch.

def test_write_bars_skips_non_finite_prices(tmp_path) -> None:
    """write_bars drops bars with a non-finite close or adj_close and keeps the rest."""
    # Isolated temp DuckDB file — same tmp_path pattern as the tests above, so
    # this test never touches the project's real data file.
    db_file: str = str(tmp_path / "test.duckdb")

    # Build a mixed batch: two clean bars, one with a NaN close, one with a NaN
    # adj_close.  All four share symbol "TEST" and distinct daily dates so the
    # PRIMARY KEY never collapses them — any missing row is a dropped bar, not a
    # dedup artifact.
    bars = [
        # Jan 2 — fully finite, must be stored.
        OHLCVBar(
            symbol="TEST",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=100.0, high=101.0, low=99.0, close=100.5, adj_close=100.5,
            volume=1000, timeframe="1d", source="test",
        ),
        # Jan 3 — NaN close: the bad-price guard must DROP this bar.
        OHLCVBar(
            symbol="TEST",
            timestamp=datetime(2024, 1, 3, tzinfo=timezone.utc),
            open=100.0, high=101.0, low=99.0, close=float("nan"), adj_close=100.5,
            volume=1000, timeframe="1d", source="test",
        ),
        # Jan 4 — NaN adj_close: the guard checks BOTH fields, so this drops too.
        OHLCVBar(
            symbol="TEST",
            timestamp=datetime(2024, 1, 4, tzinfo=timezone.utc),
            open=100.0, high=101.0, low=99.0, close=100.5, adj_close=float("nan"),
            volume=1000, timeframe="1d", source="test",
        ),
        # Jan 5 — fully finite, must be stored.
        OHLCVBar(
            symbol="TEST",
            timestamp=datetime(2024, 1, 5, tzinfo=timezone.utc),
            open=100.0, high=101.0, low=99.0, close=101.5, adj_close=101.5,
            volume=1000, timeframe="1d", source="test",
        ),
    ]

    # Two of the four bars are clean; the other two must be skipped.
    expected_written: int = 2

    # Open the store as a context manager so the file lock is always released,
    # even if an assertion fails mid-test.
    with DuckDBStore(db_file) as store:
        # write_bars must return the count of rows ACTUALLY written — only the
        # two finite bars, never the two NaN bars.
        inserted = store.write_bars(bars)
        assert inserted == expected_written, (
            f"Expected {expected_written} rows written (NaN bars skipped), got {inserted}"
        )

        # Read the full date range back; only the two good bars should be present.
        result = store.read_bars("TEST", "2024-01-02", "2024-01-05")
        assert len(result) == expected_written, (
            f"Expected {expected_written} stored bars, got {len(result)}"
        )

        # The surviving bars must be exactly Jan 2 and Jan 5 — the two NaN
        # dates (Jan 3, Jan 4) must be absent from the store.
        stored_dates = {b.timestamp.date() for b in result}
        assert stored_dates == {date(2024, 1, 2), date(2024, 1, 5)}, (
            f"Expected only the finite-price dates stored, got {sorted(stored_dates)}"
        )
