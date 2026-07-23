# tests/test_history_window.py

# Tests for src/data/history_window.live_history_start: the pure calendar-month
# rolling-start calculator for the live rotation read window.  Every case is
# hermetic -- dates are hand-built, no clock is read -- so an off-by-one in the
# month arithmetic cannot slip through and silently narrow what the bot ranks on.

# date builds the injected `today` values and the expected first-of-month results.
from datetime import date

# The single function under test.
from src.data.history_window import live_history_start


def test_mid_year_rolls_back_whole_year() -> None:
    # A plain 12-month rollback inside one year: July 2026 minus 12 months -> July 2025,
    # anchored to the 1st.  No year-boundary subtlety here, just the base case.
    assert live_history_start(date(2026, 7, 23), 12) == date(2025, 7, 1)


def test_year_boundary_rollback_exactly_twelve() -> None:
    # 12 months back from February 2026 lands in February 2025 (same month, prior year).
    # Exercises the year rollback for a full-year window.
    assert live_history_start(date(2026, 2, 15), 12) == date(2025, 2, 1)


def test_crosses_year_within_fewer_than_twelve_months() -> None:
    # March 2026 minus 5 months crosses the year boundary backwards: Mar->Feb->Jan->Dec->
    # Nov->Oct, landing in October 2025.  Proves the flat-index arithmetic handles a
    # sub-12-month rollback that still crosses into the prior year.
    assert live_history_start(date(2026, 3, 10), 5) == date(2025, 10, 1)


def test_first_of_month_anchor_is_day_agnostic_first_day() -> None:
    # The returned day must ALWAYS be 1, whether today is the 1st...
    assert live_history_start(date(2026, 7, 1), 12).day == 1
    # ...or the last day of the month.  The result never depends on today's day-of-month.
    assert live_history_start(date(2026, 7, 31), 12).day == 1


def test_months_zero_returns_this_months_first() -> None:
    # months == 0 is the degenerate no-rollback case: it returns the first of today's OWN
    # month, confirming the arithmetic is a clean identity (no accidental -1) at zero.
    assert live_history_start(date(2026, 7, 23), 0) == date(2026, 7, 1)


def test_no_month_zero_bug_january_minus_one() -> None:
    # The classic off-by-one: January minus 1 month must be December of the PRIOR year,
    # not "month 0" of the same year.  This is the case naive (month - months) arithmetic
    # gets wrong; the zero-based index makes it correct.
    assert live_history_start(date(2026, 1, 15), 1) == date(2025, 12, 1)
