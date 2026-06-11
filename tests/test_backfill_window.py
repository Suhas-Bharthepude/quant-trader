# tests/test_backfill_window.py

"""
Unit tests for resolve_window in scripts/backfill_universe.py.

These are pure unit tests — no network, no DuckDB, no filesystem, no argparse.
resolve_window was extracted precisely so the window-resolution logic could be
verified in isolation by feeding it the three already-parsed values
(years, start, end) and asserting the (start_iso, end_iso) pair it returns.
"""

# date lets us build explicit start/end inputs and compute today's expected ISO.
from datetime import date

# pytest.raises is the context manager for asserting that the guard raises.
import pytest

# The functions under test — resolve_window picks the window; compute_date_range
# is the --years fallback it delegates to, which we compare against directly so
# the test stays correct even if the years arithmetic changes.
from scripts.backfill_universe import compute_date_range, resolve_window


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_resolve_window_start_only_defaults_end_to_today() -> None:
    """An explicit --start with no --end ends today (matching the --years path)."""
    start: date = date(2008, 1, 1)

    # end=None → end should default to today's ISO date.
    start_iso, end_iso = resolve_window(None, start, None)

    assert start_iso == start.isoformat()      # start passed straight through
    assert end_iso == date.today().isoformat()  # end filled in with today


def test_resolve_window_start_and_end_returned_as_isoformat() -> None:
    """Both --start and --end given → both returned verbatim as ISO strings."""
    start: date = date(2008, 1, 1)
    end: date = date(2020, 1, 1)

    start_iso, end_iso = resolve_window(None, start, end)

    assert start_iso == start.isoformat()  # explicit start preserved
    assert end_iso == end.isoformat()      # explicit end preserved, not today


def test_resolve_window_years_falls_back_to_compute_date_range() -> None:
    """With no --start, resolve_window defers entirely to compute_date_range(years)."""
    years: int = 5

    # Comparing against compute_date_range directly (rather than a hardcoded
    # date) keeps this test stable across runs and decoupled from the exact
    # day-count arithmetic inside compute_date_range.
    assert resolve_window(years, None, None) == compute_date_range(years)


def test_resolve_window_requires_start_or_years() -> None:
    """Neither start nor years given is a misuse — must fail loudly, not silently."""
    with pytest.raises(ValueError):
        resolve_window(None, None, None)
