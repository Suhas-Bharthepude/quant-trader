# tests/test_orb_strategy.py

"""
Tests for src/intraday/orb_strategy.py — the ORB ENTRY rule and confirmation.

Exits are the sim's job and are tested in tests/test_intraday_sim.py.  Here we
assert WHERE/WHETHER an entry is opened, and that confirmation fails closed.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.data.schema import OHLCVBar
from src.intraday.orb_strategy import (
    OpeningRangeBreakout,
    OpeningRangeBreakoutConfig,
)
from src.intraday.session import ET, build_sessions


UTC = ZoneInfo("UTC")


def _bar(symbol, hh, mm, *, open_, high, low, close, d=(2024, 3, 4)):
    ts = datetime(d[0], d[1], d[2], hh, mm, tzinfo=ET).astimezone(UTC)
    return OHLCVBar(
        symbol=symbol, timestamp=ts, open=open_, high=high, low=low, close=close,
        adj_close=close, volume=1000, timeframe="1m", source="test",
    )


def _soxl_or(high=100.0, low=98.0):
    """Two OR-window bars for SOXL establishing an OR high (default 100)."""
    return [
        _bar("SOXL", 9, 30, open_=99, high=high - 0.5, low=low, close=99.5),
        _bar("SOXL", 9, 40, open_=99.5, high=high, low=99, close=99.8),
    ]


def _spy_or_and_confirm(confirm_close):
    """SPY: OR high 400, plus a 09:45 bar whose close decides confirmation."""
    return [
        _bar("SPY", 9, 30, open_=399, high=400.0, low=398, close=399.5),
        _bar("SPY", 9, 45, open_=400.2, high=402, low=400, close=confirm_close),
    ]


def _find(bars_by_symbol, *, primary="SOXL", confirm=("SPY",), **cfg_kwargs):
    config = OpeningRangeBreakoutConfig(
        primary_symbol=primary, confirmation_symbols=confirm, **cfg_kwargs
    )
    strat = OpeningRangeBreakout(config)
    sessions = build_sessions(bars_by_symbol, primary, list(confirm), config.or_minutes)
    assert len(sessions) == 1
    return strat.find_entry(sessions[0])


# ---------------------------------------------------------------------------
# No opening range
# ---------------------------------------------------------------------------


def test_no_opening_range_no_entry():
    # SOXL trades only after 09:45 → no OR → no breakout level → no entry.
    soxl = [_bar("SOXL", 9, 45, open_=99, high=101, low=99, close=100)]
    entry = _find({"SOXL": soxl, "SPY": _spy_or_and_confirm(401)})
    assert entry is None


# ---------------------------------------------------------------------------
# Confirmed breakout entries
# ---------------------------------------------------------------------------


def test_confirmed_breakout_fills_at_or_high():
    # OR high 100; breakout bar opens 99 (below level) → buy-stop fills at 100.
    soxl = _soxl_or() + [_bar("SOXL", 9, 45, open_=99, high=101, low=99, close=100.5)]
    entry = _find({"SOXL": soxl, "SPY": _spy_or_and_confirm(401)})
    assert entry is not None
    assert entry.entry_index == 0
    assert entry.entry_price == 100.0


def test_gap_through_fills_at_bar_open():
    # Breakout bar opens 100.5, already ABOVE the OR high 100 → fills at the open
    # (worse than the level — conservative gap handling).
    soxl = _soxl_or() + [_bar("SOXL", 9, 45, open_=100.5, high=101.5, low=100.3, close=101)]
    entry = _find({"SOXL": soxl, "SPY": _spy_or_and_confirm(401)})
    assert entry is not None
    assert entry.entry_price == 100.5


# ---------------------------------------------------------------------------
# Confirmation fails closed
# ---------------------------------------------------------------------------


def test_confirmation_below_or_high_blocks_entry():
    # SPY 09:45 close 399 < its OR high 400 → not confirming → no entry.
    soxl = _soxl_or() + [_bar("SOXL", 9, 45, open_=99, high=101, low=99, close=100)]
    entry = _find({"SOXL": soxl, "SPY": _spy_or_and_confirm(399)})
    assert entry is None


def test_confirmation_symbol_absent_blocks_entry():
    # SPY has NO bars that day → no OR → fail-closed → no entry.
    soxl = _soxl_or() + [_bar("SOXL", 9, 45, open_=99, high=101, low=99, close=100)]
    entry = _find({"SOXL": soxl, "SPY": []})
    assert entry is None


def test_confirmation_no_bar_at_or_before_breakout_blocks_entry():
    # SPY has an OR (from 09:30) but its NEXT bar is 09:50 — at the 09:45 breakout
    # minute there is a bar at-or-before t (the 09:30 one), whose close 399.5 is
    # BELOW the OR high 400 → still fails.  This asserts as-of uses the last prior
    # bar, not a future one.
    soxl = _soxl_or() + [_bar("SOXL", 9, 45, open_=99, high=101, low=99, close=100)]
    spy = [
        _bar("SPY", 9, 30, open_=399, high=400.0, low=398, close=399.5),  # as-of at 09:45
        _bar("SPY", 9, 50, open_=401, high=402, low=400, close=401.5),    # FUTURE — must not be used
    ]
    entry = _find({"SOXL": soxl, "SPY": spy})
    assert entry is None


# ---------------------------------------------------------------------------
# Scan continues past an unconfirmed breakout
# ---------------------------------------------------------------------------


def test_scan_enters_on_first_confirmed_breakout_not_the_first_breakout():
    # 09:45 breaks out but SPY not yet confirming (close 399); 09:46 breaks out and
    # SPY confirms (close 401) → entry on the SECOND trading bar.
    soxl = _soxl_or() + [
        _bar("SOXL", 9, 45, open_=99, high=101, low=99, close=100),
        _bar("SOXL", 9, 46, open_=100, high=101, low=99.5, close=100.5),
    ]
    spy = [
        _bar("SPY", 9, 30, open_=399, high=400.0, low=398, close=399.5),
        _bar("SPY", 9, 45, open_=399.6, high=400.5, low=399, close=399.0),  # fail
        _bar("SPY", 9, 46, open_=400.5, high=402, low=400, close=401.0),    # confirm
    ]
    entry = _find({"SOXL": soxl, "SPY": spy})
    assert entry is not None
    assert entry.entry_index == 1


# ---------------------------------------------------------------------------
# Entry cutoff
# ---------------------------------------------------------------------------


def test_entry_cutoff_blocks_late_breakout():
    # Only breakout is at 15:30 ET; cutoff 15:00 → no entry.
    soxl = _soxl_or() + [_bar("SOXL", 15, 30, open_=99, high=101, low=99, close=100)]
    entry = _find(
        {"SOXL": soxl, "SPY": [
            _bar("SPY", 9, 30, open_=399, high=400.0, low=398, close=399.5),
            _bar("SPY", 15, 30, open_=401, high=402, low=400, close=401.5),
        ]},
        entry_cutoff=time(15, 0),
    )
    assert entry is None


# ---------------------------------------------------------------------------
# Symbol-agnostic + name + validation
# ---------------------------------------------------------------------------


def test_symbol_agnostic_works_for_tqqq():
    # Prove no SOXL hard-coding: the SAME rule runs with TQQQ as the primary.
    tqqq = [
        _bar("TQQQ", 9, 30, open_=49, high=49.5, low=48, close=49),
        _bar("TQQQ", 9, 40, open_=49, high=50.0, low=48.5, close=49.5),  # OR high 50
        _bar("TQQQ", 9, 45, open_=49.5, high=51, low=49.5, close=50.5),  # breakout
    ]
    config = OpeningRangeBreakoutConfig(primary_symbol="TQQQ", confirmation_symbols=("SPY",))
    strat = OpeningRangeBreakout(config)
    assert "TQQQ" in strat.name
    sessions = build_sessions(
        {"TQQQ": tqqq, "SPY": _spy_or_and_confirm(401)}, "TQQQ", ["SPY"], 15
    )
    entry = strat.find_entry(sessions[0])
    assert entry is not None
    assert entry.entry_price == 50.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"or_minutes": 0},
        {"stop_pct": 0.0},
        {"stop_pct": 1.0},
        {"stop_pct": -0.1},
        {"target_r": 0.0},
        {"fee_bps": -1.0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        OpeningRangeBreakout(
            OpeningRangeBreakoutConfig(primary_symbol="SOXL", **kwargs)
        )
