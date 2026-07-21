# src/data/freshness.py

"""
Live-trading freshness gate: refuse to trade on stale bars.

The entry-point script calls this BEFORE trading and refuses to proceed (fails
loud, non-zero exit) if any REQUIRED symbol's most recent stored bar is not
current through today.  This is the guard that stops the bot from ranking and
rebalancing on data that is behind - a live run that ranked on last week's closes
because the ingest silently failed would trade on a wrong, stale signal.

It is deliberately PURE: no DuckDB, no clock, no I/O.  The caller supplies the
per-symbol last-bar dates (queried from DuckDB in the impure script), the
reference `today` date (never read from a clock in here), and the staleness
threshold.  That keeps the arithmetic unit-testable without a database or a real
date, mirroring the pure-decision discipline of decide_rebalance and
scripts/data_health.should_alert.

NOTE on scope: this checks FRESHNESS (are the bars we HAVE current?), not PRESENCE
(do we have every required symbol at all?).  A symbol simply absent from
last_bar_dates is NOT reported here - confirming that every required symbol is
present is a separate concern the caller owns before calling this.
"""

# date types both the per-symbol last-bar dates and the injected reference `today`.
# datetime is NOT imported: this module never reads a clock (today is passed in).
from datetime import date


def stale_symbols(
    last_bar_dates: dict[str, date],
    today: date,
    max_staleness_days: int,
) -> list[tuple[str, int]]:
    """Return (symbol, days_stale) for every symbol whose latest bar is too old.

    A symbol is STALE when its most recent stored bar is MORE than
    max_staleness_days calendar days before `today` (strictly greater than, so a
    symbol exactly at the threshold is NOT stale).  The result is sorted
    most-stale first so the worst offender surfaces at the top.

    Args:
        last_bar_dates:     symbol -> its most recent stored bar date (from DuckDB).
        today:              the reference date (injected; never a clock read here).
        max_staleness_days: the staleness threshold in calendar days.  Must be set
                            generously enough to span non-trading gaps (weekends,
                            holidays), or a normal Fri->Mon gap would read as stale.

    Returns:
        A list of (symbol, days_stale) for each stale symbol, most-stale first.
        Empty when nothing is stale (or when last_bar_dates is empty).
    """
    # Accumulate the stale (symbol, days_stale) pairs.  Symbols within the threshold
    # are never appended, so a fully-current basket yields an empty list.
    stale: list[tuple[str, int]] = []

    # Check each symbol independently against its own most recent bar date.
    for symbol, last in last_bar_dates.items():
        # Calendar days between the last bar and today.  This matches
        # scripts/data_health.days_since exactly ((today - last).days), so the live
        # gate and the diagnostic report measure staleness identically.
        days_stale = (today - last).days

        # STRICTLY GREATER than the threshold is stale; exactly at the threshold is
        # still current (a symbol max_staleness_days old is on the boundary, allowed).
        if days_stale > max_staleness_days:
            # Record the offender and how stale it is (for a legible failure message).
            stale.append((symbol, days_stale))

    # Sort most-stale first: descending by days_stale (the tuple's second element).
    # A stable sort keeps insertion order among equal staleness, which is fine here.
    stale.sort(key=lambda item: item[1], reverse=True)

    # The (possibly empty) most-stale-first list.
    return stale


def all_current(
    last_bar_dates: dict[str, date],
    today: date,
    max_staleness_days: int,
) -> bool:
    """True when NOTHING is stale (the convenience predicate over stale_symbols).

    Pure: delegates to stale_symbols and reports whether it found nothing.  The
    caller uses this as the go/no-go freshness gate before trading.
    """
    # Nothing stale -> every symbol we hold data for is current through today.
    return not stale_symbols(last_bar_dates, today, max_staleness_days)
