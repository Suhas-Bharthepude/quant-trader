# tests/test_freshness.py

"""Hermetic unit tests for src/data/freshness.py (hand-built dates, no DB, no clock)."""

# date builds the per-symbol last-bar dates and the injected reference `today`.
from datetime import date

# The pure units under test: the stale-list computer and the go/no-go predicate.
from src.data.freshness import stale_symbols, all_current


# ---------------------------------------------------------------------------
# 1. ALL FRESH — every symbol current within the threshold.
# ---------------------------------------------------------------------------


def test_all_fresh_returns_empty_and_all_current_true():
    """Every symbol's last bar is today or within threshold -> nothing stale."""
    # Reference "today" for the whole test.
    today = date(2024, 6, 10)
    # A: exactly today (0 days), B: 2 days old -- both within a 3-day threshold.
    last_bar_dates = {
        "A": date(2024, 6, 10),  # 0 days stale
        "B": date(2024, 6, 8),   # 2 days stale
    }
    # No symbol exceeds the 3-day threshold, so the stale list is empty...
    assert stale_symbols(last_bar_dates, today, max_staleness_days=3) == []
    # ...and the go/no-go gate reports everything current.
    assert all_current(last_bar_dates, today, max_staleness_days=3) is True


# ---------------------------------------------------------------------------
# 2. ONE STALE — a single symbol past the threshold is reported.
# ---------------------------------------------------------------------------


def test_one_stale_symbol_reported_with_days_and_all_current_false():
    """One symbol older than the threshold appears with its days_stale; gate is False."""
    # Reference "today".
    today = date(2024, 6, 10)
    # A is current (1 day), STALE_SYM is 6 days old -> exceeds a 3-day threshold.
    last_bar_dates = {
        "A": date(2024, 6, 9),          # 1 day stale -> fine
        "STALE_SYM": date(2024, 6, 4),  # 6 days stale -> stale
    }
    # Exactly the one offender is reported, with the correct calendar-day count.
    assert stale_symbols(last_bar_dates, today, max_staleness_days=3) == [("STALE_SYM", 6)]
    # The gate refuses (something is stale).
    assert all_current(last_bar_dates, today, max_staleness_days=3) is False


# ---------------------------------------------------------------------------
# 3. EXACTLY AT THRESHOLD IS NOT STALE — strictly-greater rule.
# ---------------------------------------------------------------------------


def test_exactly_at_threshold_is_not_stale():
    """A last bar exactly max_staleness_days before today is NOT stale (strictly greater)."""
    # Reference "today".
    today = date(2024, 6, 10)
    # A's last bar is exactly 3 days before today, and the threshold is 3.
    last_bar_dates = {"A": date(2024, 6, 7)}  # (2024-06-10 - 2024-06-07).days == 3
    # 3 is NOT > 3, so A is on the boundary and counts as current -> empty list.
    assert stale_symbols(last_bar_dates, today, max_staleness_days=3) == []
    # The gate passes.
    assert all_current(last_bar_dates, today, max_staleness_days=3) is True


# ---------------------------------------------------------------------------
# 4. SORT ORDER — most-stale first.
# ---------------------------------------------------------------------------


def test_stale_symbols_sorted_most_stale_first():
    """Two stale symbols of different ages come back most-stale first."""
    # Reference "today".
    today = date(2024, 6, 10)
    # OLDER is 9 days stale, NEWER is 5 days stale; both exceed a 3-day threshold.
    last_bar_dates = {
        "NEWER": date(2024, 6, 5),  # 5 days stale
        "OLDER": date(2024, 6, 1),  # 9 days stale
    }
    # Result is descending by days_stale: the 9-day offender precedes the 5-day one,
    # regardless of insertion order (NEWER was inserted first).
    assert stale_symbols(last_bar_dates, today, max_staleness_days=3) == [
        ("OLDER", 9),
        ("NEWER", 5),
    ]


# ---------------------------------------------------------------------------
# 5. WEEKEND/GAP REALISM — the threshold must span non-trading gaps.
# ---------------------------------------------------------------------------


def test_threshold_tolerates_weekend_gap_but_flags_a_full_week():
    """A 4-day threshold tolerates Fri->Mon (3 days) but flags a 7-day gap."""
    # 2024-06-14 is a Friday; 2024-06-17 is the following Monday.
    friday = date(2024, 6, 14)
    monday = date(2024, 6, 17)
    # Last bar Friday, today Monday: (Mon - Fri).days == 3, within a 4-day threshold, so
    # a normal weekend gap does NOT read as stale (the threshold must span weekends).
    assert stale_symbols({"A": friday}, monday, max_staleness_days=4) == []
    assert all_current({"A": friday}, monday, max_staleness_days=4) is True
    # Last bar the PRIOR Monday (2024-06-10), today the next Monday: 7 days > 4 -> stale.
    prior_monday = date(2024, 6, 10)
    assert stale_symbols({"A": prior_monday}, monday, max_staleness_days=4) == [("A", 7)]
    assert all_current({"A": prior_monday}, monday, max_staleness_days=4) is False


# ---------------------------------------------------------------------------
# 6. EMPTY INPUT — nothing to check is trivially current.
# ---------------------------------------------------------------------------


def test_empty_input_is_trivially_current():
    """Empty last_bar_dates -> no stale symbols and the gate passes.

    PRESENCE (are all required symbols there?) is a SEPARATE concern owned by the
    caller/script; this freshness gate only judges the dates it is handed, so an
    empty input has nothing to flag.
    """
    # Reference "today".
    today = date(2024, 6, 10)
    # Nothing to check -> empty stale list...
    assert stale_symbols({}, today, max_staleness_days=3) == []
    # ...and the gate is trivially satisfied.
    assert all_current({}, today, max_staleness_days=3) is True
