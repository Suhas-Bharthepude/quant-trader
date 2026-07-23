# tests/test_obsp_strategy.py

"""
Tests for the Open-Buy, Sell-at-Profit (OBSP) strategy + sim exit mode.

Mirrors the ORB/trailing intraday test files: entry at the open, target-hit exit,
never-positive -> close exit, gap/same-bar handling, no-lookahead, symbol-agnostic,
and — the point of OBSP — HONEST gross-vs-net accounting kept strictly separate.
"""

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.data.schema import OHLCVBar
from src.intraday.obsp_strategy import OpenBuySellProfit, OpenBuySellProfitConfig
from src.intraday.session import ET, build_sessions
from src.intraday.sim import resolve_exit_profit_target, run_obsp_backtest


UTC = ZoneInfo("UTC")


def _bar(symbol, hh, mm, *, open_, high, low, close, d=(2024, 3, 4)):
    ts = datetime(d[0], d[1], d[2], hh, mm, tzinfo=ET).astimezone(UTC)
    return OHLCVBar(
        symbol=symbol, timestamp=ts, open=open_, high=high, low=low, close=close,
        adj_close=close, volume=1000, timeframe="1m", source="test",
    )


def _run(bars_by_symbol, symbol="SOXL", **cfg):
    config = OpenBuySellProfitConfig(primary_symbol=symbol, **cfg)
    strat = OpenBuySellProfit(config)
    return run_obsp_backtest(bars_by_symbol, strat, config)


# ---------------------------------------------------------------------------
# resolve_exit_profit_target — core exit behaviour
# ---------------------------------------------------------------------------


def test_target_hit_fills_at_target():
    # entry 100, +0.5% target = 100.5. Second bar's high reaches it.
    bars = [
        _bar("X", 9, 30, open_=100, high=100.2, low=99.8, close=100.1),  # entry bar, no hit
        _bar("X", 9, 31, open_=100.1, high=100.7, low=100.0, close=100.6),  # high 100.7 >= 100.5
    ]
    idx, price, reason = resolve_exit_profit_target(bars, 0, 100.0, 0.005)
    assert (idx, reason) == (1, "target")
    assert price == pytest.approx(100.5)  # the target, not the 100.7 high


def test_same_bar_target_hit_on_entry_bar():
    # The entry bar's OWN high reaches the target → valid same-bar exit at bar 0.
    bars = [_bar("X", 9, 30, open_=100, high=100.6, low=99.9, close=100.4)]
    idx, price, reason = resolve_exit_profit_target(bars, 0, 100.0, 0.005)
    assert (idx, reason) == (0, "target")
    assert price == pytest.approx(100.5)


def test_gap_above_target_still_fills_at_target():
    # A bar that opens ABOVE the target (gap up) is not credited the gap; fills at target.
    bars = [
        _bar("X", 9, 30, open_=100, high=100.1, low=99.9, close=100.0),
        _bar("X", 9, 31, open_=101.0, high=101.5, low=100.9, close=101.2),  # gapped past 100.5
    ]
    idx, price, reason = resolve_exit_profit_target(bars, 0, 100.0, 0.005)
    assert (idx, reason) == (1, "target")
    assert price == pytest.approx(100.5)  # target, NOT the 101.0 open or 101.5 high


def test_never_reaches_target_exits_at_close():
    # Price falls from the open and never reaches +0.5% → EOD-flat at last close.
    bars = [
        _bar("X", 9, 30, open_=100, high=100.1, low=99.5, close=99.8),
        _bar("X", 15, 59, open_=99.8, high=99.9, low=98.0, close=98.5),  # closes down
    ]
    idx, price, reason = resolve_exit_profit_target(bars, 0, 100.0, 0.005)
    assert (idx, reason) == (1, "eod")
    assert price == 98.5


# ---------------------------------------------------------------------------
# No lookahead (mirrors test_intraday_no_lookahead)
# ---------------------------------------------------------------------------


def test_no_lookahead_later_high_cannot_trigger_earlier_exit():
    # Target 100.5. Only the THIRD bar reaches it. A lookahead bug using the global
    # max high would exit on bar 0; correct behaviour exits on bar 2.
    bars = [
        _bar("X", 9, 30, open_=100, high=100.2, low=99.8, close=100.1),  # below target
        _bar("X", 9, 31, open_=100.1, high=100.3, low=100.0, close=100.2),  # below target
        _bar("X", 9, 32, open_=100.2, high=100.6, low=100.1, close=100.5),  # first touch
    ]
    idx, price, reason = resolve_exit_profit_target(bars, 0, 100.0, 0.005)
    assert idx == 2, "a later bar's high must not trigger an earlier exit"
    assert reason == "target"
    assert price == pytest.approx(100.5)


def test_exit_is_first_touching_bar_not_a_later_one():
    bars = [
        _bar("X", 9, 30, open_=100, high=100.2, low=99.9, close=100.1),
        _bar("X", 9, 31, open_=100.1, high=100.6, low=100.0, close=100.5),  # first touch
        _bar("X", 9, 32, open_=100.5, high=101.0, low=100.4, close=100.9),  # later, ignored
    ]
    idx, _, reason = resolve_exit_profit_target(bars, 0, 100.0, 0.005)
    assert (idx, reason) == (1, "target")


# ---------------------------------------------------------------------------
# Entry at the open (full session, no OR strip)
# ---------------------------------------------------------------------------


def test_entry_is_at_first_session_bar_open():
    # A pre-market bar must be dropped; entry is the 09:30 open bar's OPEN price.
    bars = [
        _bar("SOXL", 9, 15, open_=95, high=95.5, low=94, close=95),      # pre-market (dropped)
        _bar("SOXL", 9, 30, open_=100, high=100.6, low=99.5, close=100.4),  # THE open bar
        _bar("SOXL", 9, 31, open_=100.4, high=100.7, low=100.2, close=100.6),
    ]
    result = _run({"SOXL": bars}, profit_target=0.005)
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.entry_price == 100.0  # the 09:30 OPEN, not the pre-market 95
    assert trade.entry_time == bars[1].timestamp


# ---------------------------------------------------------------------------
# Honest gross-vs-net accounting (requirement 3)
# ---------------------------------------------------------------------------


def test_target_hit_is_gross_win_even_when_net_negative():
    # A single session that hits the +0.5% target, but with costs so large the trade
    # is NET-negative. It must still count as a GROSS win (win_rate == 1.0), while
    # avg_net_win is negative — gross and net are DISTINCT.
    bars = [
        _bar("SOXL", 9, 30, open_=100, high=100.6, low=99.9, close=100.4),  # hits 100.5 same bar
        _bar("SOXL", 15, 59, open_=100.4, high=100.5, low=100.0, close=100.3),
    ]
    # 100 bps/side → round trip 200 bps = 0.02 log drag, dwarfing the +0.5% gross.
    result = _run({"SOXL": bars}, profit_target=0.005, fee_bps=100.0, slippage_bps=0.0)
    assert result.n_trades == 1
    assert result.exit_reason_counts["target"] == 1
    assert result.win_rate == 1.0                 # target reached → gross win
    assert result.avg_gross_win > 0               # ~ +0.5%
    assert result.avg_net_win < 0                 # costs pushed it net-negative
    assert result.avg_gross_win != result.avg_net_win  # distinct numbers


def test_gross_and_net_totals_differ_only_by_costs():
    # One target-hit session, modest costs. Net compounded < gross compounded, and
    # the gap equals the round-trip cost in log space.
    bars = [
        _bar("SOXL", 9, 30, open_=100, high=100.1, low=99.9, close=100.0),
        _bar("SOXL", 9, 31, open_=100.0, high=100.6, low=99.9, close=100.5),  # target 100.5
    ]
    result = _run({"SOXL": bars}, profit_target=0.005, fee_bps=5.0, slippage_bps=5.0)
    gross_log = math.log(100.5 / 100.0)
    net_log = gross_log - 0.002  # 10 bps/side round trip
    assert result.gross_total_return_pct == pytest.approx(math.expm1(gross_log))
    assert result.total_return_pct == pytest.approx(math.expm1(net_log))


def test_win_loss_size_ratio_reported():
    # Two sessions: one hits target (+0.5% gross win), one falls to a big EOD loss.
    day1 = [
        _bar("SOXL", 9, 30, open_=100, high=100.6, low=99.9, close=100.4, d=(2024, 3, 4)),
        _bar("SOXL", 15, 59, open_=100.4, high=100.5, low=100.1, close=100.3, d=(2024, 3, 4)),
    ]
    day2 = [
        _bar("SOXL", 9, 30, open_=100, high=100.2, low=95, close=96, d=(2024, 3, 5)),  # never +0.5%
        _bar("SOXL", 15, 59, open_=96, high=96.5, low=94, close=95, d=(2024, 3, 5)),   # -5% EOD
    ]
    result = _run({"SOXL": day1 + day2}, profit_target=0.005)
    assert result.exit_reason_counts == {"target": 1, "eod": 1}
    assert result.win_rate == 0.5
    assert result.avg_gross_win > 0
    assert result.avg_gross_loss < 0
    # Small win vs large loss → ratio well below 1 (the whole point of the claim).
    assert result.win_loss_size_ratio < 1.0
    assert result.win_loss_size_ratio == pytest.approx(
        abs(result.avg_gross_win) / abs(result.avg_gross_loss)
    )


# ---------------------------------------------------------------------------
# Symbol-agnostic + config validation
# ---------------------------------------------------------------------------


def test_symbol_agnostic_runs_for_tqqq():
    bars = [
        _bar("TQQQ", 9, 30, open_=50, high=50.3, low=49.8, close=50.2),
        _bar("TQQQ", 9, 31, open_=50.2, high=50.3, low=50.1, close=50.25),  # +0.5% = 50.25
    ]
    result = _run({"TQQQ": bars}, symbol="TQQQ", profit_target=0.005)
    assert result.primary_symbol == "TQQQ"
    assert "TQQQ" in result.strategy_name
    assert result.n_trades == 1


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.01])
def test_invalid_profit_target_rejected(bad):
    with pytest.raises(ValueError):
        OpenBuySellProfit(OpenBuySellProfitConfig(primary_symbol="SOXL", profit_target=bad))


def test_negative_costs_rejected():
    with pytest.raises(ValueError):
        OpenBuySellProfit(
            OpenBuySellProfitConfig(primary_symbol="SOXL", fee_bps=-1.0)
        )
