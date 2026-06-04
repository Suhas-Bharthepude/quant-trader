# tests/test_cli_common.py

"""
Unit tests for src/research/cli_common.py.

Tests 1-6 (build_symbol_list) are pure: no DuckDB, no network — the function
takes plain strings and a limit integer and returns a list.  They run in
milliseconds and can be run in isolation.

Tests 7-10 (load_bars_for_symbols) hit real DuckDB and assume the "test"
universe (SPY, QQQ, AAPL, MSFT, NVDA) has been ingested.  They are
integration tests in the sense of touching the filesystem, but still fast
because DuckDB reads are in-process and the test universe is small.

Run with:
    uv run pytest tests/test_cli_common.py -v
"""

# pytest is the test runner; no fixtures are used here — plain functions
# keep the test bodies readable without fixture-lookup indirection.
import pytest

# The two functions under test.  Importing by name ties the test module to
# the public API of cli_common rather than to its internal structure.
from src.research.cli_common import build_symbol_list, load_bars_for_symbols

# load_universe is used as the reference oracle in the universe-path tests
# (tests 5 and 6) so we compare against the live config rather than
# hardcoding symbol names — if the "test" universe is ever updated the
# tests adapt automatically without code changes.
from src.data.universe import load_universe


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
# Tests for load_bars_for_symbols — real DuckDB I/O (test universe required)
# ---------------------------------------------------------------------------


def test_load_bars_returns_dict_for_known_symbols():
    """Known symbols produce a dict with one non-empty bars list per symbol."""

    # SPY and QQQ are both in the test universe and have been ingested, so
    # both keys must appear in the result.  Non-empty bars confirms the read
    # actually found rows rather than returning a placeholder empty list.
    result = load_bars_for_symbols(["SPY", "QQQ"], "2021-01-01", "2026-05-14")

    assert set(result.keys()) == {"SPY", "QQQ"}
    assert len(result["SPY"]) > 0
    assert len(result["QQQ"]) > 0


def test_load_bars_skips_missing_symbol():
    """A symbol absent from DuckDB is silently skipped; present symbols are unaffected."""

    # ZZZZ_NOT_A_REAL_SYMBOL will never be in DuckDB.  If load_bars_for_symbols
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


def test_load_bars_preserves_input_order():
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


def test_load_bars_all_missing_returns_empty_dict():
    """When every symbol is absent from DuckDB the return value is an empty dict, not an error."""

    # The caller (compare_universe, compare_matrix) owns the "no bars found"
    # error message and the return-1.  This function's job is to collect what
    # data exists and hand it back — raising here would force every caller to
    # wrap the call in a try/except just to print a one-liner error.
    result = load_bars_for_symbols(
        ["FAKE1", "FAKE2"], "2021-01-01", "2026-05-14"
    )

    # Empty dict, not None, not an exception — the caller can do `if not result`
    # with standard Python truth-testing and branch on the empty case cleanly.
    assert result == {}
