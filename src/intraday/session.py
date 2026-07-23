# src/intraday/session.py

"""
Pure intraday session mechanics: ET trading-day grouping, the opening-range
window, and the backward-only as-of alignment used to sample confirmation
symbols onto the primary symbol's decision timeline.

This module is PURE — no DuckDB, no clock, no I/O.  It converts UTC bar
timestamps to US/Eastern via zoneinfo to decide session membership, but every
input (the bars) is passed in.  That keeps it deterministic and unit-testable
with hand-built bar lists.

Design split (mirrors the pure-vs-rule discipline elsewhere in the repo):
  * session.py  — the MECHANICS: what session a bar belongs to, what the opening
                  range is, and how to sample another symbol as-of a minute.
  * orb_strategy — the RULE: what "breakout" and "confirmation" mean on top of
                  these mechanics.

Why ET, not UTC: the NYSE session (09:30–16:00 America/New_York) shifts between
13:30–20:00 and 14:30–21:00 UTC across DST.  Keying session membership off the
ET wall clock via zoneinfo makes DST automatic and half-days fall out naturally
(the last real bar before 16:00 ET is the session's last bar, whatever the date).
"""

# bisect_right finds the as-of insertion point in O(log n) for the backward-only
# confirmation sampling — no future bar can ever be selected (see as_of_bar).
from bisect import bisect_right

# dataclass builds the frozen value objects; frozen=True mirrors OHLCVBar /
# BacktestResult — a built session is an immutable historical fact.
from dataclasses import dataclass

# date/time/datetime/timedelta express the ET session boundaries and the OR width.
from datetime import date, datetime, time, timedelta

# ZoneInfo gives DST-correct US/Eastern conversion from stdlib (Python 3.9+),
# no third-party tz dependency.
from zoneinfo import ZoneInfo

# OHLCVBar is the bar schema every symbol's list holds — same type the daily
# stack and the DuckDB store use.
from src.data.schema import OHLCVBar


# ---------------------------------------------------------------------------
# Constants — the regular NYSE cash session in ET wall-clock terms.
# ---------------------------------------------------------------------------

# The America/New_York zone; all session-membership decisions convert to this.
ET = ZoneInfo("America/New_York")

# Regular session open/close in ET.  Bars are assumed START-labelled (the 09:30
# bar covers 09:30:00–09:31:00), so regular-session membership is [09:30, 16:00):
# the last regular bar is the one starting 15:59 and there is no 16:00 bar.
SESSION_OPEN: time = time(9, 30)
SESSION_CLOSE: time = time(16, 0)

# Default opening-range width in minutes: 09:30–09:45 → the 15 one-minute bars
# starting 09:30 through 09:44 inclusive.  Entries begin at the 09:45 bar.
DEFAULT_OR_MINUTES: int = 15


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpeningRange:
    """The high/low of a symbol's opening-range window for one session.

    None is used by callers (not this class) to mean "no OR" — an empty OR
    window.  When present, high/low come from the bars actually in the window;
    n_bars records how many contributed (useful for diagnostics and tests).
    """

    high: float
    low: float
    n_bars: int


@dataclass(frozen=True)
class PrimarySession:
    """The primary (traded) symbol's data for one ET session.

    The sim iterates trading_bars in order as its decision/monitoring timeline;
    opening_range is None when the OR window had no bars (no breakout level → no
    trade that day).
    """

    date_et: date
    opening_range: OpeningRange | None
    # Bars in [or_end, close), sorted ascending — the entry-eligible + exit-
    # monitoring window.  The "no trades 09:30–09:45" rule is enforced simply by
    # EXCLUDING the OR window from this list.
    trading_bars: list[OHLCVBar]


@dataclass(frozen=True)
class ConfirmationSession:
    """A confirmation symbol's data for one ET session, ready for as-of sampling.

    opening_range is None when that symbol had no OR-window bars (→ confirmation
    fails, fail-closed).  session_bars is the full regular-session list sorted
    ascending; as_of_bar samples it backward-only at the primary's breakout minute.
    """

    symbol: str
    opening_range: OpeningRange | None
    session_bars: list[OHLCVBar]


@dataclass(frozen=True)
class AlignedSession:
    """One ET session: the primary session plus every confirmation symbol's session.

    Only dates where the PRIMARY symbol has regular-session bars produce an
    AlignedSession — the primary is the decision clock.  A confirmation symbol
    with no bars that date still appears in `confirmations`, with opening_range
    None and empty session_bars, so downstream confirmation fails closed.
    """

    date_et: date
    primary: PrimarySession
    confirmations: dict[str, ConfirmationSession]


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def to_et(ts: datetime) -> datetime:
    """Convert a (tz-aware) UTC bar timestamp to America/New_York wall time."""
    return ts.astimezone(ET)


def session_date(ts: datetime) -> date:
    """The ET calendar date a bar belongs to (its trading-session identifier)."""
    return to_et(ts).date()


def or_end_time(or_minutes: int = DEFAULT_OR_MINUTES) -> time:
    """The ET time the opening-range window ends (exclusive), e.g. 09:45 for 15."""
    # Combine with an arbitrary date to do time arithmetic, then take the time.
    end = datetime.combine(date(2000, 1, 1), SESSION_OPEN) + timedelta(minutes=or_minutes)
    return end.time()


def in_regular_session(ts: datetime) -> bool:
    """True when the bar's ET time is within the regular cash session [09:30, 16:00)."""
    t = to_et(ts).time()
    return SESSION_OPEN <= t < SESSION_CLOSE


def _in_or_window(ts: datetime, or_end: time) -> bool:
    """True when the bar's ET time is within the opening-range window [09:30, or_end)."""
    t = to_et(ts).time()
    return SESSION_OPEN <= t < or_end


# ---------------------------------------------------------------------------
# Opening range + as-of sampling
# ---------------------------------------------------------------------------


def compute_opening_range(
    session_bars: list[OHLCVBar], or_minutes: int = DEFAULT_OR_MINUTES
) -> OpeningRange | None:
    """Return the OpeningRange over [09:30, or_end) for one session, or None if empty.

    None (rather than a zero-bar OpeningRange) is the fail-closed signal: a symbol
    with no bars in its own OR window cannot define a breakout level or confirm.
    """
    or_end = or_end_time(or_minutes)
    window = [b for b in session_bars if _in_or_window(b.timestamp, or_end)]
    if not window:
        return None
    return OpeningRange(
        high=max(b.high for b in window),
        low=min(b.low for b in window),
        n_bars=len(window),
    )


def as_of_bar(session_bars_sorted: list[OHLCVBar], t: datetime) -> OHLCVBar | None:
    """Return the last bar with timestamp <= t (backward-only as-of), or None.

    This is THE anti-lookahead primitive for confirmation alignment: it can only
    ever return a bar at-or-before t, never a future one, so sampling a
    confirmation symbol at the primary's breakout minute cannot peek ahead.  When
    a confirmation symbol is missing the exact minute t, this forward-fills the
    last actually-printed bar (never interpolates).  Returns None when no bar
    exists at-or-before t (→ confirmation fails closed).

    Requires session_bars_sorted to be ascending by timestamp (build_sessions
    guarantees this).
    """
    # bisect_right on the timestamps gives the count of bars with ts <= t; the
    # one just before that index is the as-of bar.  Build the key list once per
    # call — session bar lists are short (a trading day of minutes), so this is
    # cheap and keeps the function self-contained.
    timestamps = [b.timestamp for b in session_bars_sorted]
    idx = bisect_right(timestamps, t)
    if idx == 0:
        return None  # nothing at-or-before t
    return session_bars_sorted[idx - 1]


# ---------------------------------------------------------------------------
# Session assembly
# ---------------------------------------------------------------------------


def _sessions_by_date(bars: list[OHLCVBar]) -> dict[date, list[OHLCVBar]]:
    """Group regular-session bars by ET date, each list sorted ascending.

    Bars outside [09:30, 16:00) ET (pre/post-market) are dropped so extended-hours
    prints never leak into the opening range or the trading window.
    """
    grouped: dict[date, list[OHLCVBar]] = {}
    for b in bars:
        if not in_regular_session(b.timestamp):
            continue
        grouped.setdefault(session_date(b.timestamp), []).append(b)
    # Sort each day's bars ascending so as_of_bar's bisect and the sim's forward
    # walk both see a monotonic timeline.
    for day_bars in grouped.values():
        day_bars.sort(key=lambda bar: bar.timestamp)
    return grouped


def build_sessions(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    primary_symbol: str,
    confirmation_symbols: list[str],
    or_minutes: int = DEFAULT_OR_MINUTES,
) -> list[AlignedSession]:
    """Assemble per-date AlignedSessions keyed off the PRIMARY symbol's sessions.

    The primary symbol is the decision clock: one AlignedSession per date on which
    the primary has regular-session bars, in ascending date order.  Each
    confirmation symbol is attached for that date with its own opening range and
    full session-bar list; a confirmation symbol absent that date gets an empty
    ConfirmationSession (opening_range None, no bars) so confirmation fails closed.

    Args:
        bars_by_symbol:       symbol -> that symbol's OHLCVBar list (any order).
        primary_symbol:       the traded symbol (e.g. "SOXL").
        confirmation_symbols: symbols whose state gates entry (e.g. ["SPY","QQQ"]).
        or_minutes:           opening-range width in minutes (default 15).

    Returns:
        AlignedSession list in ascending date order.  Empty when the primary has
        no regular-session bars at all.

    Raises:
        ValueError: primary_symbol absent from bars_by_symbol (a wiring error —
                    you cannot trade a symbol you passed no data for).
    """
    if primary_symbol not in bars_by_symbol:
        raise ValueError(
            f"primary_symbol {primary_symbol!r} not in bars_by_symbol "
            f"(keys: {sorted(bars_by_symbol.keys())})"
        )

    or_end = or_end_time(or_minutes)

    # Pre-group every symbol we care about by ET date, once.
    primary_by_date = _sessions_by_date(bars_by_symbol[primary_symbol])
    confirm_by_date: dict[str, dict[date, list[OHLCVBar]]] = {
        sym: _sessions_by_date(bars_by_symbol.get(sym, [])) for sym in confirmation_symbols
    }

    aligned: list[AlignedSession] = []
    # Ascending date order so downstream metrics/equity compound chronologically.
    for day in sorted(primary_by_date.keys()):
        primary_day_bars = primary_by_date[day]

        # Primary trading window = regular-session bars OUTSIDE the OR window.
        # Excluding the OR window here is how "no trades 09:30–09:45" is enforced.
        trading_bars = [
            b for b in primary_day_bars if not _in_or_window(b.timestamp, or_end)
        ]
        primary_session = PrimarySession(
            date_et=day,
            opening_range=compute_opening_range(primary_day_bars, or_minutes),
            trading_bars=trading_bars,
        )

        # Attach each confirmation symbol's session for this date (or an empty one).
        confirmations: dict[str, ConfirmationSession] = {}
        for sym in confirmation_symbols:
            sym_day_bars = confirm_by_date[sym].get(day, [])
            confirmations[sym] = ConfirmationSession(
                symbol=sym,
                opening_range=compute_opening_range(sym_day_bars, or_minutes),
                session_bars=sym_day_bars,
            )

        aligned.append(
            AlignedSession(date_et=day, primary=primary_session, confirmations=confirmations)
        )

    return aligned
