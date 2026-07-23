# src/data/history_window.py

"""
Pure calendar-month rolling-start calculator for the LIVE rotation read window.

The daily autopilot (scripts/run_daily.py) must read enough history for the rotation
ranking to be well-defined but no more.  The ranking needs a MINIMUM of ~5 month-ends:
lookback + 1 for the trailing return (LIVE_CONFIG.lookback == 3, so the decision
month-end must be the 4th or later), plus 1 for the incomplete current month that
most_recent_completed_month_end drops.  Callers therefore pass `months` comfortably
above that floor -- the live path uses 12 -- and this module turns that count into a
concrete start date.

Anchoring the start to the FIRST day of the target month guarantees the window spans at
least `months` FULL calendar months regardless of what day `today` is, so weekends,
holidays, and short months never change how many month-ends land in the window.

Pure: `today` is INJECTED by the caller (the impure edge reads the clock); this module
never calls date.today()/datetime.now(), does no I/O, and has no dateutil dependency --
plain stdlib date arithmetic only.
"""

# date is the only dependency: the input `today` and the returned start are stdlib
# datetime.date values.  No datetime/timezone/clock is imported -- purity by construction.
from datetime import date


def live_history_start(today: date, months: int) -> date:
    """Return the first day of the calendar month `months` months before `today`'s month.

    Args:
        today:  the live "as of" date, INJECTED by the caller (never read from a clock here).
        months: how many whole calendar months to roll back; the live rotation path passes 12
                (comfortably above the ~5 month-end minimum the ranking needs).

    Returns:
        A date on the FIRST of the target month.  E.g. live_history_start(date(2026, 7, 23), 12)
        returns date(2025, 7, 1) -- exactly `months` full months of history back, day-agnostic.

    The arithmetic uses a ZERO-BASED absolute month index so year boundaries roll back
    correctly (January minus one month becomes December of the prior year), never producing
    a month 0 or a negative month.
    """
    # Collapse (year, month) into a single zero-based absolute month index.  month-1 makes it
    # zero-based (January -> 0), so the index is a clean integer count of months since year 0.
    # Example: 2026-07 -> 2026*12 + (7-1) == 24318.
    zero_based_month_index = today.year * 12 + (today.month - 1)

    # Roll back by `months` whole months in that flat index space.  Because the index is a
    # single integer, subtraction crosses year boundaries automatically -- no special-casing,
    # and no risk of a naive month-1 that could underflow to 0 or go negative.
    target_index = zero_based_month_index - months

    # Convert the flat index back to a calendar year: integer-divide by 12 months per year.
    # (Python's // floors, which is correct here since target_index is non-negative for any
    # realistic today/months, keeping the year arithmetic exact.)
    target_year = target_index // 12

    # Recover the 1-based calendar month: modulo 12 gives the zero-based month (0..11), and
    # +1 restores the 1..12 convention date() expects.  This is the step that turns a
    # December rollback (zero-based 11) back into month 12, never month 0.
    target_month = target_index % 12 + 1

    # Anchor to the FIRST of the resolved month.  Day 1 guarantees the window includes the
    # whole target month, so the count of month-ends in [start, today] depends only on the
    # month arithmetic above, not on today's day-of-month.
    return date(target_year, target_month, 1)
