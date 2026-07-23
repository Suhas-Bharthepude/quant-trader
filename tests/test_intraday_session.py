# tests/test_intraday_session.py

"""
Tests for src/intraday/session.py — ET session grouping, the opening range, and
the backward-only as-of confirmation sampler.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from src.data.schema import OHLCVBar
from src.intraday.session import (
    ET,
    OpeningRange,
    as_of_bar,
    build_sessions,
    compute_opening_range,
    in_regular_session,
    or_end_time,
    session_date,
)


# ---------------------------------------------------------------------------
# Helpers — build minute bars from ET wall-clock times.
# ---------------------------------------------------------------------------


def _bar_at_et(
    symbol: str,
    y: int,
    m: int,
    d: int,
    hh: int,
    mm: int,
    *,
    open_=100.0,
    high=101.0,
    low=99.0,
    close=100.0,
    volume=1000,
) -> OHLCVBar:
    """One 1-minute bar timestamped at the given ET wall time (stored as UTC)."""
    ts_et = datetime(y, m, d, hh, mm, tzinfo=ET)
    ts_utc = ts_et.astimezone(ZoneInfo("UTC"))
    return OHLCVBar(
        symbol=symbol,
        timestamp=ts_utc,
        open=open_,
        high=high,
        low=low,
        close=close,
        adj_close=close,
        volume=volume,
        timeframe="1m",
        source="test",
    )


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def test_or_end_time_default_is_0945():
    assert or_end_time() == datetime(2000, 1, 1, 9, 45).time()
    assert or_end_time(30) == datetime(2000, 1, 1, 10, 0).time()


def test_in_regular_session_bounds():
    # 09:30 is in; 09:29 (pre-market) is out; 16:00 (post) is out; 15:59 is in.
    assert in_regular_session(_bar_at_et("X", 2024, 3, 4, 9, 30).timestamp)
    assert not in_regular_session(_bar_at_et("X", 2024, 3, 4, 9, 29).timestamp)
    assert not in_regular_session(_bar_at_et("X", 2024, 3, 4, 16, 0).timestamp)
    assert in_regular_session(_bar_at_et("X", 2024, 3, 4, 15, 59).timestamp)


def test_session_date_uses_et_calendar_across_dst():
    # A summer 09:30 ET bar (EDT, 13:30 UTC) and a winter one (EST, 14:30 UTC)
    # both resolve to their ET calendar date, not the UTC date.
    summer = _bar_at_et("X", 2024, 7, 1, 9, 30)  # EDT
    winter = _bar_at_et("X", 2024, 1, 3, 9, 30)  # EST
    assert session_date(summer.timestamp) == date(2024, 7, 1)
    assert session_date(winter.timestamp) == date(2024, 1, 3)
    # Sanity: the stored UTC hours actually differ across DST.
    assert summer.timestamp.hour == 13
    assert winter.timestamp.hour == 14


# ---------------------------------------------------------------------------
# Opening range
# ---------------------------------------------------------------------------


def test_compute_opening_range_over_first_15_minutes():
    # OR window is [09:30, 09:45): bars 09:30..09:44. Put the max high at 09:40
    # and the min low at 09:33; a 09:45 bar must NOT contribute.
    bars = [
        _bar_at_et("X", 2024, 3, 4, 9, 30, high=101, low=100),
        _bar_at_et("X", 2024, 3, 4, 9, 33, high=101, low=98),   # min low
        _bar_at_et("X", 2024, 3, 4, 9, 40, high=105, low=100),  # max high
        _bar_at_et("X", 2024, 3, 4, 9, 44, high=102, low=100),
        _bar_at_et("X", 2024, 3, 4, 9, 45, high=999, low=1),    # OUTSIDE OR window
    ]
    orange = compute_opening_range(bars)
    assert orange == OpeningRange(high=105.0, low=98.0, n_bars=4)


def test_compute_opening_range_none_when_no_or_bars():
    # Only post-OR bars → no opening range (fail-closed signal).
    bars = [_bar_at_et("X", 2024, 3, 4, 10, 0)]
    assert compute_opening_range(bars) is None


# ---------------------------------------------------------------------------
# as_of_bar — backward-only sampling
# ---------------------------------------------------------------------------


def test_as_of_bar_exact_and_forward_fill():
    bars = [
        _bar_at_et("SPY", 2024, 3, 4, 9, 45, close=10),
        _bar_at_et("SPY", 2024, 3, 4, 9, 47, close=12),  # note: 9:46 missing
    ]
    # Exact hit at 09:45.
    t_open = datetime(2024, 3, 4, 9, 45, tzinfo=ET).astimezone(ZoneInfo("UTC"))
    assert as_of_bar(bars, t_open).close == 10
    # 09:46 is missing → forward-fill the last printed (09:45) bar, never the future 09:47.
    t_gap = datetime(2024, 3, 4, 9, 46, tzinfo=ET).astimezone(ZoneInfo("UTC"))
    assert as_of_bar(bars, t_gap).close == 10
    # 09:47 exact.
    t_late = datetime(2024, 3, 4, 9, 47, tzinfo=ET).astimezone(ZoneInfo("UTC"))
    assert as_of_bar(bars, t_late).close == 12


def test_as_of_bar_none_before_first_bar():
    bars = [_bar_at_et("SPY", 2024, 3, 4, 9, 45, close=10)]
    t_before = datetime(2024, 3, 4, 9, 40, tzinfo=ET).astimezone(ZoneInfo("UTC"))
    assert as_of_bar(bars, t_before) is None


# ---------------------------------------------------------------------------
# build_sessions
# ---------------------------------------------------------------------------


def _full_session(symbol: str, d: date, *, drop_or: bool = False) -> list[OHLCVBar]:
    """A tiny session: two OR bars (unless drop_or) + two trading bars + one pre-market."""
    bars = [
        _bar_at_et(symbol, d.year, d.month, d.day, 9, 15),   # pre-market (dropped)
        _bar_at_et(symbol, d.year, d.month, d.day, 9, 45),   # trading
        _bar_at_et(symbol, d.year, d.month, d.day, 10, 0),   # trading
        _bar_at_et(symbol, d.year, d.month, d.day, 16, 30),  # post-market (dropped)
    ]
    if not drop_or:
        bars += [
            _bar_at_et(symbol, d.year, d.month, d.day, 9, 30, high=110, low=100),
            _bar_at_et(symbol, d.year, d.month, d.day, 9, 40, high=112, low=101),
        ]
    return bars


def test_build_sessions_primary_is_decision_clock_and_drops_extended_hours():
    d = date(2024, 3, 4)
    aligned = build_sessions(
        {"SOXL": _full_session("SOXL", d), "SPY": _full_session("SPY", d)},
        primary_symbol="SOXL",
        confirmation_symbols=["SPY"],
    )
    assert len(aligned) == 1
    sess = aligned[0]
    assert sess.date_et == d
    # OR from the 09:30/09:40 bars: high 112, low 100.
    assert sess.primary.opening_range == OpeningRange(high=112.0, low=100.0, n_bars=2)
    # trading_bars excludes the OR window AND the pre/post-market bars → 09:45, 10:00.
    times = [to := b.timestamp.astimezone(ET).strftime("%H:%M") for b in sess.primary.trading_bars]
    assert times == ["09:45", "10:00"]
    # Confirmation attached with its own OR.
    assert sess.confirmations["SPY"].opening_range == OpeningRange(high=112.0, low=100.0, n_bars=2)


def test_build_sessions_confirmation_absent_that_day_is_failclosed_empty():
    d = date(2024, 3, 4)
    aligned = build_sessions(
        {"SOXL": _full_session("SOXL", d), "SPY": []},  # SPY has NO bars
        primary_symbol="SOXL",
        confirmation_symbols=["SPY"],
    )
    conf = aligned[0].confirmations["SPY"]
    assert conf.opening_range is None      # no OR → confirmation fails closed
    assert conf.session_bars == []


def test_build_sessions_confirmation_missing_or_window_only():
    # SPY trades only AFTER the OR window → its OR is None even though it has bars.
    d = date(2024, 3, 4)
    spy_bars = [
        _bar_at_et("SPY", 2024, 3, 4, 9, 45),
        _bar_at_et("SPY", 2024, 3, 4, 10, 0),
    ]
    aligned = build_sessions(
        {"SOXL": _full_session("SOXL", d), "SPY": spy_bars},
        primary_symbol="SOXL",
        confirmation_symbols=["SPY"],
    )
    conf = aligned[0].confirmations["SPY"]
    assert conf.opening_range is None
    assert len(conf.session_bars) == 2  # bars present, just none in the OR window


def test_build_sessions_raises_when_primary_absent():
    try:
        build_sessions({"SPY": []}, primary_symbol="SOXL", confirmation_symbols=["SPY"])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "SOXL" in str(exc)
