# tests/test_cli_common.py

"""
Unit tests for src/research/cli_common.py.

Tests 1-6 (build_symbol_list) are pure: no DuckDB, no network — the function
takes plain strings and a limit integer and returns a list.  They run in
milliseconds and can be run in isolation.

Tests 7-10 (load_bars_for_symbols) are hermetic: the hermetic_db fixture
builds a temporary DuckDB seeded with SPY/QQQ/AAPL and monkeypatches
cli_common.DuckDBStore to point at it, so they pass in a clean CI checkout
without any ingested data/quant_trader.duckdb on disk.

Run with:
    uv run pytest tests/test_cli_common.py -v
"""

# pytest is the test runner.  The build_symbol_list tests below are plain
# functions; the load_bars_for_symbols tests share the hermetic_db fixture
# defined further down so the temp-DB-and-monkeypatch setup lives in one place.
import pytest

# functools.partial binds the temp DB path to the real DuckDBStore class so the
# no-argument `DuckDBStore()` call inside load_bars_for_symbols resolves to our
# temporary file — see the hermetic_db fixture for why this is the seam we patch.
import functools

# datetime/timezone build the tz-aware UTC timestamps the synthetic bars require.
# OHLCVBar's contract is UTC-aware timestamps, and DuckDBStore floors 1d bars to
# midnight UTC on write, so naive or non-UTC datetimes would be wrong here.
from datetime import datetime, timezone

# The two functions under test.  Importing by name ties the test module to
# the public API of cli_common rather than to its internal structure.
from src.research.cli_common import build_symbol_list, load_bars_for_symbols

# The cli_common module object itself — needed as the monkeypatch target.
# load_bars_for_symbols looks up DuckDBStore via this module's globals at call
# time, so patching cli_common.DuckDBStore (not the symbol imported into this
# test module) is what actually redirects the function to the temp DB.
from src.research import cli_common

# DuckDBStore is the real persistence class.  The fixture builds the temp DB
# through it (so the schema is created from the real CREATE_TABLE_SQL and cannot
# drift) and partial-binds it as the monkeypatch replacement.
from src.data.duckdb_store import DuckDBStore

# OHLCVBar is the typed bar the synthetic fixture rows are constructed from —
# imported from the schema module so the test uses the exact production type.
from src.data.schema import OHLCVBar

# load_universe is used as the reference oracle in the universe-path tests
# (tests 5 and 6) so we compare against the live config rather than
# hardcoding symbol names — if the "test" universe is ever updated the
# tests adapt automatically without code changes.
from src.data.universe import load_universe


# ---------------------------------------------------------------------------
# Hermetic DuckDB fixture for the load_bars_for_symbols tests
# ---------------------------------------------------------------------------


def _synthetic_bars(symbol: str) -> list[OHLCVBar]:
    """Build a few trivial-but-valid daily bars for one symbol.

    Three DISTINCT calendar dates (2021-01-04/05/06) so that DuckDBStore's
    midnight-UTC floor on 1d bars never collapses two rows onto the same
    (symbol, timestamp, timeframe) primary key and silently drops one.  The
    dates sit inside the 2021-01-01 → 2026-05-14 window the tests query, so
    read_bars returns them.  Prices are flat placeholders — these tests assert
    on dict keys/skip/order, never on bar values, so the OHLC numbers only need
    to be present and valid, not realistic.
    """
    return [
        # timeframe="1d" matches read_bars' default filter; source="test" marks
        # the row as fixture data.  timestamps are tz-aware UTC per OHLCVBar's
        # contract (naive datetimes are rejected upstream at fetch time).
        OHLCVBar(
            symbol=symbol,
            timestamp=datetime(2021, 1, day, tzinfo=timezone.utc),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            adj_close=1.0,
            volume=100,
            timeframe="1d",
            source="test",
        )
        for day in (4, 5, 6)
    ]


@pytest.fixture
def hermetic_db(tmp_path, monkeypatch):
    """Point load_bars_for_symbols at a temp DuckDB containing SPY/QQQ/AAPL.

    Makes the DB-backed tests hermetic: they no longer depend on a real ingested
    data/quant_trader.duckdb, so they run (and pass for the right reason) in a
    clean CI checkout.  Returns nothing — the tests just need the fixture active.
    """

    # A real file (not :memory:) under pytest's per-test tmp_path, so the
    # partial-bound DuckDBStore() opens the same database the fixture populated.
    db_path = tmp_path / "test.duckdb"

    # Build the temp DB through the REAL DuckDBStore: __init__ runs the real
    # CREATE_TABLE_SQL, so the ohlcv_bars schema is created by construction and
    # cannot drift from production.  write_bars inserts the synthetic rows for
    # the three known symbols; ZZZZ_NOT_A_REAL_SYMBOL / FAKE1 / FAKE2 are
    # deliberately never inserted so the skip / all-missing paths are exercised.
    with DuckDBStore(str(db_path)) as store:
        for symbol in ("SPY", "QQQ", "AAPL"):
            store.write_bars(_synthetic_bars(symbol))

    # Patch the DuckDBStore name in cli_common's namespace — NOT load_bars_for_
    # symbols itself.  partial pre-binds the temp path, so the function's
    # intentional no-argument `DuckDBStore()` call now opens our temp file.
    # monkeypatch auto-reverts after the test, restoring the real default path.
    monkeypatch.setattr(
        cli_common, "DuckDBStore", functools.partial(DuckDBStore, str(db_path))
    )


# ---------------------------------------------------------------------------
# Tests for build_symbol_list — pure, no I/O
# ---------------------------------------------------------------------------


def test_build_symbol_list_from_csv():
    """CSV input is split and returned as-is; the universe argument is ignored."""

    # "sp500" is passed as the universe to confirm it is truly ignored.  If the
    # function fell through to the universe path, it would return a completely
    # different list — so this double-checks the CSV-wins branch, not just
    # the parsing logic.
    result = build_symbol_list("SPY,QQQ,AAPL", "sp500", 25)

    assert result == ["SPY", "QQQ", "AAPL"]


def test_build_symbol_list_csv_strips_whitespace():
    """Whitespace around each CSV token is stripped before returning."""

    # Users often copy-paste ticker lists with inconsistent spacing.  The
    # function must normalise "SPY, QQQ , AAPL" to the same output as
    # "SPY,QQQ,AAPL" so the caller never sees whitespace-polluted symbols.
    result = build_symbol_list("SPY, QQQ , AAPL", "sp500", 25)

    assert result == ["SPY", "QQQ", "AAPL"]


def test_build_symbol_list_csv_drops_empty_tokens():
    """Trailing commas and double commas produce no empty strings in the result."""

    # A trailing comma ("SPY,QQQ,") is a common cut-and-paste artefact.  A
    # double comma ("SPY,,QQQ") can come from a spreadsheet export.  Both must
    # be silently dropped rather than producing ["SPY", "", "QQQ"] or
    # ["SPY", "QQQ", ""], which would confuse DuckDB lookups downstream.
    result = build_symbol_list("SPY,,QQQ,", "sp500", 25)

    assert result == ["SPY", "QQQ"]


def test_build_symbol_list_csv_ignores_limit():
    """limit is ignored when an explicit CSV is provided."""

    # limit=2 with 5 symbols in the CSV.  If the function applied the limit
    # to the CSV path, it would silently drop AAPL, MSFT, and NVDA — a
    # confusing outcome when the caller wrote out every symbol explicitly.
    # The contract is: explicit input wins, including over the limit cap.
    result = build_symbol_list("SPY,QQQ,AAPL,MSFT,NVDA", "sp500", 2)

    assert result == ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


def test_build_symbol_list_from_universe_applies_limit():
    """Universe path returns the first `limit` symbols from load_universe()."""

    # We compare against load_universe("test")[:3] rather than a hardcoded
    # list so the test adapts if the "test" universe order ever changes.
    # Hardcoding ["SPY", "QQQ", "AAPL"] would make a universe reorder a
    # silent false-negative (the function is correct but the test disagrees).
    result = build_symbol_list(None, "test", 3)

    assert result == load_universe("test")[:3]


def test_build_symbol_list_from_universe_no_limit_overflow():
    """limit larger than the universe returns the full universe (slice is a no-op)."""

    # limit=1000 on a 5-symbol universe.  Python's [:1000] on a 5-element
    # list returns all 5 elements — we verify the function does not raise,
    # truncate, or pad.  Comparing length against the live load_universe call
    # avoids hardcoding "5" in the assertion so the test survives a universe resize.
    result = build_symbol_list(None, "test", 1000)

    assert len(result) == len(load_universe("test"))


# ---------------------------------------------------------------------------
# Tests for load_bars_for_symbols — hermetic DuckDB via the hermetic_db fixture
# ---------------------------------------------------------------------------


def test_load_bars_returns_dict_for_known_symbols(hermetic_db):
    """Known symbols produce a dict with one non-empty bars list per symbol."""

    # SPY and QQQ are both seeded into the temp DB by hermetic_db, so both keys
    # must appear in the result.  Non-empty bars confirms the read actually
    # found rows rather than returning a placeholder empty list.
    result = load_bars_for_symbols(["SPY", "QQQ"], "2021-01-01", "2026-05-14")

    assert set(result.keys()) == {"SPY", "QQQ"}
    assert len(result["SPY"]) > 0
    assert len(result["QQQ"]) > 0


def test_load_bars_skips_missing_symbol(hermetic_db):
    """A symbol absent from DuckDB is silently skipped; present symbols are unaffected."""

    # ZZZZ_NOT_A_REAL_SYMBOL is never seeded into the temp DB.  If load_bars_for_symbols
    # raised on a missing symbol, a 25-symbol sweep would abort because one
    # ticker was not yet ingested — unacceptable for large universes.
    # The correct behaviour is: skip and log, keep going.
    result = load_bars_for_symbols(
        ["SPY", "ZZZZ_NOT_A_REAL_SYMBOL"], "2021-01-01", "2026-05-14"
    )

    # SPY was present — it must appear in the result.
    assert "SPY" in result

    # The fake symbol must have been dropped, not added as an empty-list value
    # or a None — the contract is "absent from dict", not "present but empty".
    assert "ZZZZ_NOT_A_REAL_SYMBOL" not in result


def test_load_bars_preserves_input_order(hermetic_db):
    """Returned dict keys appear in the same order as the input symbol list."""

    # Python dicts preserve insertion order (3.7+), and the function inserts
    # symbols in the iteration order of the input list.  Downstream pivot
    # builders and compare() calls rely on this to produce stable, reproducible
    # output — a symbol order that varies across runs would produce tables that
    # are hard to diff and compare visually.
    result = load_bars_for_symbols(
        ["QQQ", "SPY", "AAPL"], "2021-01-01", "2026-05-14"
    )

    # list(result.keys()) preserves the Python dict insertion order exactly —
    # if the function had sorted or shuffled keys, this would catch it.
    assert list(result.keys()) == ["QQQ", "SPY", "AAPL"]


def test_load_bars_all_missing_returns_empty_dict(hermetic_db):
    """When every symbol is absent from DuckDB the return value is an empty dict, not an error."""

    # The caller (compare_universe, compare_matrix) owns the "no bars found"
    # error message and the return-1.  This function's job is to collect what
    # data exists and hand it back — raising here would force every caller to
    # wrap the call in a try/except just to print a one-liner error.
    #
    # hermetic_db seeds the temp DB with SPY/QQQ/AAPL but NOT FAKE1/FAKE2, so the
    # empty result here proves "these specific symbols have no rows" — the real
    # behaviour — rather than the accidental pass you'd get from an empty/missing
    # database where every lookup trivially returns nothing.
    result = load_bars_for_symbols(
        ["FAKE1", "FAKE2"], "2021-01-01", "2026-05-14"
    )

    # Empty dict, not None, not an exception — the caller can do `if not result`
    # with standard Python truth-testing and branch on the empty case cleanly.
    assert result == {}
