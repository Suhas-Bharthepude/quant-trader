# tests/test_intraday_sim.py

"""
Tests for src/intraday/sim.py — the event-driven exit engine and full backtest.

The load-bearing behaviour here is STOP-FIRST on same-bar conflicts
(test_stop_first_when_both_hit_same_bar): if a bar's range spans both the stop
and the target, the sim must fill the STOP.  Plus: entry-bar monitoring, gap-
through fills, EOD-flat, and the no-overnight-leak guarantee.
"""

import math
from datetime import datetime, time
from zoneinfo import ZoneInfo

from src.data.schema import OHLCVBar
from src.intraday.orb_strategy import (
    EntryEvent,
    OpeningRangeBreakout,
    OpeningRangeBreakoutConfig,
)
from src.intraday.session import ET, PrimarySession, OpeningRange
from src.intraday.sim import (
    resolve_exit,
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
# resolve_exit — the stop-first / EOD-flat core
# ---------------------------------------------------------------------------


def test_target_hit_fills_at_target():
    # entry 100, stop 94 (6%), R=6, target=112. A later bar's high reaches 112.
    bars = [
        _bar("X", 9, 45, open_=100, high=101, low=99.5, close=100.5),  # entry bar, nothing hit
        _bar("X", 9, 46, open_=101, high=112.5, low=100.5, close=112),  # target 112 touched
    ]
    idx, price, reason = resolve_exit(bars, entry_index=0, entry_price=100.0,
                                      stop_pct=0.06, target_r=2.0)
    assert (idx, reason) == (1, "target")
    assert price == 112.0  # exactly the target, not the 112.5 high


def test_stop_hit_fills_at_stop():
    # entry 100, stop 94. A later bar's low pierces 94.
    bars = [
        _bar("X", 9, 45, open_=100, high=101, low=99.5, close=100.5),
        _bar("X", 9, 46, open_=99, high=100, low=93.5, close=95),  # low 93.5 <= 94
    ]
    idx, price, reason = resolve_exit(bars, 0, 100.0, 0.06, 2.0)
    assert (idx, reason) == (1, "stop")
    assert price == 94.0  # stop level (open 99 is above stop, so min(94, 99) == 94)


def test_stop_first_when_both_hit_same_bar():
    """REQUIREMENT 1: a bar spanning BOTH stop and target resolves to the STOP."""
    # entry 100, stop 94, target 112. A single violent bar whose range is
    # [93, 113] touches BOTH. The sim must return the stop, not the target.
    bars = [
        _bar("X", 9, 45, open_=100, high=101, low=99.5, close=100.5),  # entry bar
        _bar("X", 9, 46, open_=100, high=113.0, low=93.0, close=100),  # spans both
    ]
    idx, price, reason = resolve_exit(bars, 0, 100.0, 0.06, 2.0)
    assert reason == "stop", "same-bar stop+target must resolve to the STOP (conservative)"
    assert idx == 1
    assert price == 94.0


def test_entry_bar_stop_is_monitored():
    # The ENTRY bar itself can stop out: its own low pierces the stop → stop on bar 0.
    bars = [
        _bar("X", 9, 45, open_=100, high=101, low=93.0, close=95),  # entry bar low 93 <= 94
    ]
    idx, price, reason = resolve_exit(bars, 0, 100.0, 0.06, 2.0)
    assert (idx, reason) == (0, "stop")
    assert price == 94.0


def test_gap_down_through_stop_fills_at_open():
    # A bar that OPENS below the stop (gap-down) fills at the open, worse than stop.
    bars = [
        _bar("X", 9, 45, open_=100, high=101, low=99.5, close=100.5),
        _bar("X", 9, 46, open_=92.0, high=92.5, low=91.0, close=91.5),  # opened at 92 < stop 94
    ]
    idx, price, reason = resolve_exit(bars, 0, 100.0, 0.06, 2.0)
    assert reason == "stop"
    assert price == 92.0  # min(stop 94, open 92) == 92


def test_eod_flat_when_neither_level_touched():
    # Price drifts, never hitting stop 94 or target 112 → exit at last bar close.
    bars = [
        _bar("X", 9, 45, open_=100, high=101, low=99, close=100.5),
        _bar("X", 15, 59, open_=100.5, high=102, low=99.5, close=101.0),
    ]
    idx, price, reason = resolve_exit(bars, 0, 100.0, 0.06, 2.0)
    assert (idx, reason) == (1, "eod")
    assert price == 101.0


# ---------------------------------------------------------------------------
# simulate_session — cost + Trade assembly
# ---------------------------------------------------------------------------


def _primary(bars, *, or_high=100.0):
    return PrimarySession(
        date_et=bars[0].timestamp.astimezone(ET).date(),
        opening_range=OpeningRange(high=or_high, low=or_high - 2, n_bars=2),
        trading_bars=bars,
    )


def test_no_entry_gives_zero_return_outcome():
    bars = [_bar("X", 9, 45, open_=100, high=101, low=99, close=100)]
    config = OpeningRangeBreakoutConfig(primary_symbol="X")
    out = simulate_session(_primary(bars), None, config)
    assert out.trade is None
    assert out.return_log == 0.0
    assert out.exit_reason is None


def test_costs_reduce_trade_return_and_are_net():
    # target hit at 112 from entry 100 → gross log = ln(1.12). With 10 bps/side,
    # round trip cost = 2 * 0.001 = 0.002 subtracted from the log return.
    bars = [
        _bar("X", 9, 45, open_=100, high=101, low=99.5, close=100.5),
        _bar("X", 9, 46, open_=101, high=112.5, low=100.5, close=112),
    ]
    entry = EntryEvent(entry_index=0, entry_time=bars[0].timestamp, entry_price=100.0)
    config = OpeningRangeBreakoutConfig(primary_symbol="X", fee_bps=5.0, slippage_bps=5.0)
    out = simulate_session(_primary(bars), entry, config)
    assert out.exit_reason == "target"
    expected = math.log(112.0 / 100.0) - 0.002
    assert out.trade.return_pct == out.return_log
    assert abs(out.trade.return_pct - expected) < 1e-12
    assert out.trade.direction == 1


# ---------------------------------------------------------------------------
# run_orb_backtest — end to end, no overnight leak across two sessions
# ---------------------------------------------------------------------------


def _day(symbol, d, *, or_high, breakout_high, breakout_close, spy_confirm_close):
    """A one-breakout session for `symbol` plus matching SPY confirmation bars."""
    soxl = [
        _bar(symbol, 9, 30, open_=or_high - 1, high=or_high - 0.5, low=or_high - 2,
             close=or_high - 1, d=d),
        _bar(symbol, 9, 40, open_=or_high - 1, high=or_high, low=or_high - 1.5,
             close=or_high - 0.5, d=d),
        _bar(symbol, 9, 45, open_=or_high - 1, high=breakout_high, low=or_high - 1.5,
             close=breakout_close, d=d),
        _bar(symbol, 15, 59, open_=breakout_close, high=breakout_close + 0.5,
             low=breakout_close - 0.5, close=breakout_close, d=d),
    ]
    spy = [
        _bar("SPY", 9, 30, open_=399, high=400.0, low=398, close=399.5, d=d),
        _bar("SPY", 9, 45, open_=400.2, high=402, low=400, close=spy_confirm_close, d=d),
    ]
    return soxl, spy


def test_run_backtest_two_sessions_metrics_and_no_overnight_leak():
    soxl1, spy1 = _day("SOXL", (2024, 3, 4), or_high=100, breakout_high=101,
                        breakout_close=100.5, spy_confirm_close=401)
    soxl2, spy2 = _day("SOXL", (2024, 3, 5), or_high=100, breakout_high=101,
                        breakout_close=100.5, spy_confirm_close=401)
    bars = {"SOXL": soxl1 + soxl2, "SPY": spy1 + spy2}

    config = OpeningRangeBreakoutConfig(primary_symbol="SOXL", confirmation_symbols=("SPY",))
    strat = OpeningRangeBreakout(config)
    result = run_orb_backtest(bars, strat, config)

    # Two sessions, each with one EOD-flat trade (neither stop 94 nor target 112 hit).
    assert result.n_sessions == 2
    assert result.n_trades == 2
    assert result.exit_reason_counts["eod"] == 2
    assert result.exit_reason_counts["stop"] == 0
    assert result.exit_reason_counts["target"] == 0
    # returns array is one-per-session, equity curve compounds them.
    assert len(result.returns) == 2
    assert len(result.equity_curve) == 2
    # No-overnight-leak: each session's return is a WITHIN-day close/entry move
    # (100.5/100), never the 15:59->next-09:30 gap. Each ~ ln(100.5/100).
    for r in result.returns:
        assert abs(r - math.log(100.5 / 100.0)) < 1e-9


def test_run_backtest_symbol_agnostic_primary():
    # The runner keys entirely off config.primary_symbol — swap to TQQQ.
    tqqq, spy = _day("TQQQ", (2024, 3, 4), or_high=50, breakout_high=51,
                     breakout_close=50.5, spy_confirm_close=401)
    config = OpeningRangeBreakoutConfig(primary_symbol="TQQQ", confirmation_symbols=("SPY",))
    strat = OpeningRangeBreakout(config)
    result = run_orb_backtest({"TQQQ": tqqq, "SPY": spy}, strat, config)
    assert result.primary_symbol == "TQQQ"
    assert result.n_trades == 1
