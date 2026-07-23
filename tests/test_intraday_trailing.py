# tests/test_intraday_trailing.py

"""
Tests for the ADDITIVE trailing-stop exit mode (src/intraday/sim.resolve_exit_trailing
and the config.trail_pct branch in simulate_session).

The fixed-stop path (resolve_exit) and its tests in test_intraday_sim.py are
untouched; these only exercise the new trailing behaviour, its conservative
within-bar ordering, and its no-lookahead guarantee.
"""

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.data.schema import OHLCVBar
from src.intraday.orb_strategy import (
    OpeningRangeBreakout,
    OpeningRangeBreakoutConfig,
)
from src.intraday.session import ET, PrimarySession, OpeningRange, build_sessions
from src.intraday.sim import (
    resolve_exit,
    resolve_exit_trailing,
    simulate_session,
    run_orb_backtest,
)


UTC = ZoneInfo("UTC")


def _bar(symbol, hh, mm, *, open_, high, low, close, d=(2024, 3, 4)):
    ts = datetime(d[0], d[1], d[2], hh, mm, tzinfo=ET).astimezone(UTC)
    return OHLCVBar(
        symbol=symbol, timestamp=ts, open=open_, high=high, low=low, close=close,
        adj_close=close, volume=1000, timeframe="1m", source="test",
    )


# ---------------------------------------------------------------------------
# resolve_exit_trailing — core behaviour
# ---------------------------------------------------------------------------


def test_trailing_clean_exit_ratchets_up_then_stops():
    # entry 100, trail 10%. Peak rises to 110 (trail 99), then a pullback whose low
    # 98 breaches the trail from the prior peak 110 (= 99) → exit at 99.
    bars = [
        _bar("X", 9, 45, open_=100, high=100, low=99.5, close=100),   # peak 100, trail 90
        _bar("X", 9, 46, open_=100, high=110, low=105, close=109),    # peak -> 110
        _bar("X", 9, 47, open_=109, high=111, low=98, close=100),     # low 98 <= 99 (trail from 110)
    ]
    idx, price, reason = resolve_exit_trailing(bars, entry_index=0, entry_price=100.0, trail_pct=0.10)
    assert (idx, reason) == (2, "trailing_stop")
    assert price == 99.0  # 110 * (1 - 0.10)


def test_trailing_whipsaw_tight_trail_exits_on_entry_bar():
    # A tight 1% trail: initial trail = 99. The entry bar's own low 98.9 breaches
    # it immediately → early exit on bar 0 (whipsaw).
    bars = [_bar("X", 9, 45, open_=100, high=100, low=98.9, close=99.2)]
    idx, price, reason = resolve_exit_trailing(bars, 0, 100.0, trail_pct=0.01)
    assert (idx, reason) == (0, "trailing_stop")
    assert price == 99.0  # min(trail 99, open 100)


def test_trailing_eod_flat_when_never_triggered():
    # Price only rises; the 10% trail is never breached → EOD-flat at last close.
    bars = [
        _bar("X", 9, 45, open_=100, high=100, low=99.5, close=99.8),
        _bar("X", 15, 59, open_=100, high=101, low=100, close=100.5),
    ]
    idx, price, reason = resolve_exit_trailing(bars, 0, 100.0, trail_pct=0.10)
    assert (idx, reason) == (1, "eod")
    assert price == 100.5


def test_trailing_new_high_and_breach_uses_prior_peak_conservative():
    """REQUIREMENT 2: a bar that BOTH makes a new high AND breaches the prior-peak
    trail stops off the PRIOR peak, not the new high."""
    # entry 100, trail 10%. Prior peak = 100 → trail 90. The bar spikes to a new
    # high of 120 but its low 89 breaches the PRIOR-peak trail (90).
    bars = [
        _bar("X", 9, 45, open_=100, high=100, low=99.5, close=100),   # peak 100
        _bar("X", 9, 46, open_=119, high=120, low=89, close=100),     # new high 120 AND low 89
    ]
    idx, price, reason = resolve_exit_trailing(bars, 0, 100.0, trail_pct=0.10)
    assert (idx, reason) == (1, "trailing_stop")
    # Prior-peak trail is 90 (100 * 0.9). If a bug raised the peak to 120 FIRST,
    # the trail would be 108 and the exit would be 108 — assert it is NOT.
    assert price == 90.0


def test_trailing_gap_down_through_trail_fills_at_open():
    # A bar that opens BELOW the trail (gap down) fills at the open, worse than trail.
    bars = [
        _bar("X", 9, 45, open_=100, high=100, low=99.5, close=100),  # peak 100, trail 95 (5%)
        _bar("X", 9, 46, open_=90, high=91, low=89, close=90.5),     # opened at 90 < trail 95
    ]
    idx, price, reason = resolve_exit_trailing(bars, 0, 100.0, trail_pct=0.05)
    assert reason == "trailing_stop"
    assert price == 90.0  # min(trail 95, open 90)


# ---------------------------------------------------------------------------
# No lookahead — a later bar's high cannot raise the trail for an earlier decision
# ---------------------------------------------------------------------------


def test_trailing_future_high_cannot_raise_stop_for_earlier_bar():
    """REQUIREMENT 3: the peak at bar t uses only bars through t.

    Correct behaviour exits at bar 2. A buggy implementation that used the GLOBAL
    max high (106, only reached later) as the trail for every bar would breach and
    exit one bar EARLY, at bar 1.
    """
    bars = [
        _bar("X", 9, 45, open_=100, high=105, low=100, close=104),   # peak -> 105 (trail 94.5)
        _bar("X", 9, 46, open_=105, high=106, low=95.0, close=105),  # trail 94.5: 95.0 > 94.5 → hold; peak -> 106
        _bar("X", 9, 47, open_=100, high=106, low=95.3, close=100),  # trail 95.4 (from 106): 95.3 <= 95.4 → exit
    ]
    idx, price, reason = resolve_exit_trailing(bars, 0, 100.0, trail_pct=0.10)
    assert idx == 2, "a future high must not retroactively raise the trail for bar 1"
    assert reason == "trailing_stop"
    assert abs(price - 95.4) < 1e-9  # 106 * 0.90


# ---------------------------------------------------------------------------
# Additivity: fixed-stop path is unchanged; trail_pct only diverts when set
# ---------------------------------------------------------------------------


def _primary(bars, or_high=100.0):
    return PrimarySession(
        date_et=bars[0].timestamp.astimezone(ET).date(),
        opening_range=OpeningRange(high=or_high, low=or_high - 2, n_bars=2),
        trading_bars=bars,
    )


def test_same_bars_fixed_vs_trailing_diverge_only_when_trail_set():
    from src.intraday.orb_strategy import EntryEvent

    # Bars that under the FIXED path drift to an EOD-flat (never hit stop 94 or
    # target 112), but under a tight 3% trail get stopped out on the pullback.
    bars = [
        _bar("X", 9, 45, open_=100, high=101, low=99.5, close=100.5),
        _bar("X", 9, 46, open_=101, high=105, low=104, close=104.5),   # peak 105, trail 101.85 (3%)
        _bar("X", 9, 47, open_=104, high=104.5, low=101.0, close=101.5),  # low 101 <= 101.85 → trail exit
        _bar("X", 15, 59, open_=101.5, high=102, low=101, close=101.8),
    ]
    entry = EntryEvent(entry_index=0, entry_time=bars[0].timestamp, entry_price=100.0)

    fixed_cfg = OpeningRangeBreakoutConfig(primary_symbol="X")  # trail_pct None
    fixed = simulate_session(_primary(bars), entry, fixed_cfg)
    assert fixed.exit_reason == "eod"  # fixed path unchanged

    trail_cfg = OpeningRangeBreakoutConfig(primary_symbol="X", trail_pct=0.03)
    trailed = simulate_session(_primary(bars), entry, trail_cfg)
    assert trailed.exit_reason == "trailing_stop"
    assert trailed.trade.exit_price == pytest.approx(105 * 0.97)  # 101.85

    # Direct proof the fixed resolver is untouched: it still returns "eod" here.
    idx, _, reason = resolve_exit(bars, 0, 100.0, 0.06, 2.0)
    assert reason == "eod"


# ---------------------------------------------------------------------------
# Integration through run_orb_backtest, incl. symbol-agnostic
# ---------------------------------------------------------------------------


def _trailing_session(symbol, d):
    """A confirmed breakout that ratchets up then pulls back into a trailing stop."""
    primary = [
        _bar(symbol, 9, 30, open_=99, high=99.5, low=98, close=99, d=d),
        _bar(symbol, 9, 40, open_=99, high=100, low=98.5, close=99.5, d=d),   # OR high 100
        _bar(symbol, 9, 45, open_=99, high=101, low=99, close=100.5, d=d),    # breakout, entry 100
        _bar(symbol, 9, 46, open_=101, high=110, low=105, close=109, d=d),    # peak 110
        _bar(symbol, 9, 47, open_=109, high=111, low=98, close=100, d=d),     # trail (10%) hit @ 99
        _bar(symbol, 15, 59, open_=100, high=100.5, low=99.5, close=100, d=d),
    ]
    spy = [
        _bar("SPY", 9, 30, open_=399, high=400.0, low=398, close=399.5, d=d),
        _bar("SPY", 9, 45, open_=400.2, high=402, low=400, close=401, d=d),
    ]
    return primary, spy


def test_run_backtest_trailing_counts_trailing_stop_reason():
    soxl, spy = _trailing_session("SOXL", (2024, 3, 4))
    config = OpeningRangeBreakoutConfig(
        primary_symbol="SOXL", confirmation_symbols=("SPY",), trail_pct=0.10
    )
    result = run_orb_backtest({"SOXL": soxl, "SPY": spy}, OpeningRangeBreakout(config), config)
    assert result.n_trades == 1
    assert result.exit_reason_counts["trailing_stop"] == 1
    assert result.exit_reason_counts["eod"] == 0
    # Entry 100 → trail exit 99 (110 * 0.9), gross log = ln(99/100), costs 0.
    assert result.trades[0].exit_price == pytest.approx(99.0)


def test_run_backtest_trailing_symbol_agnostic():
    tnat, spy = _trailing_session("TNA", (2024, 3, 4))
    config = OpeningRangeBreakoutConfig(
        primary_symbol="TNA", confirmation_symbols=("SPY",), trail_pct=0.10
    )
    result = run_orb_backtest({"TNA": tnat, "SPY": spy}, OpeningRangeBreakout(config), config)
    assert result.primary_symbol == "TNA"
    assert result.exit_reason_counts["trailing_stop"] == 1


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.05])
def test_trail_pct_out_of_range_rejected(bad):
    with pytest.raises(ValueError):
        OpeningRangeBreakout(
            OpeningRangeBreakoutConfig(primary_symbol="SOXL", trail_pct=bad)
        )


def test_trail_pct_none_is_valid():
    # Default None must always construct (fixed-stop mode).
    OpeningRangeBreakout(OpeningRangeBreakoutConfig(primary_symbol="SOXL"))
