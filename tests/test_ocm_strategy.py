# tests/test_ocm_strategy.py

"""
Tests for the Opening-Candle Momentum (OCM) strategy + sim mode.

Mirrors the ORB/OBSP intraday test files: green-signal entry at candle N+1's OPEN,
red-signal no-trade, no-lookahead, exit modes, the buy-open baseline, the mandatory
chronological train/test split, honest gross-vs-net, and symbol-agnosticism.
"""

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.data.schema import OHLCVBar
from src.intraday.ocm_strategy import OpeningCandleMomentum, OpeningCandleMomentumConfig
from src.intraday.sim import (
    resolve_exit_at_close,
    split_chronological,
    run_ocm_backtest,
)


UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


def _bar(symbol, hh, mm, *, open_, high, low, close, d=(2024, 3, 4)):
    ts = datetime(d[0], d[1], d[2], hh, mm, tzinfo=ET).astimezone(UTC)
    return OHLCVBar(
        symbol=symbol, timestamp=ts, open=open_, high=high, low=low, close=close,
        adj_close=close, volume=1000, timeframe="1m", source="test",
    )


def _green(symbol, hh, mm, base, d=(2024, 3, 4)):
    # close > open → green candle.
    return _bar(symbol, hh, mm, open_=base, high=base + 1, low=base - 0.2, close=base + 0.5, d=d)


def _red(symbol, hh, mm, base, d=(2024, 3, 4)):
    # close < open → red candle.
    return _bar(symbol, hh, mm, open_=base, high=base + 0.2, low=base - 1, close=base - 0.5, d=d)


def _run(bars_by_symbol, symbol="SOXL", **cfg):
    config = OpeningCandleMomentumConfig(primary_symbol=symbol, **cfg)
    strat = OpeningCandleMomentum(config)
    return run_ocm_backtest(bars_by_symbol, strat, config)


def _find(bars, symbol="SOXL", **cfg):
    from src.intraday.session import build_sessions
    config = OpeningCandleMomentumConfig(primary_symbol=symbol, **cfg)
    strat = OpeningCandleMomentum(config)
    sessions = build_sessions({symbol: bars}, symbol, [], or_minutes=0)
    assert len(sessions) == 1
    return strat.find_entry(sessions[0])


# ---------------------------------------------------------------------------
# Entry rule
# ---------------------------------------------------------------------------


def test_green_signal_enters_at_candle_n_plus_1_open():
    # N=2: candles 0,1 green → enter at candle 2's OPEN.
    bars = [
        _green("SOXL", 9, 30, 100),
        _green("SOXL", 9, 31, 101),
        _bar("SOXL", 9, 32, open_=102.0, high=103, low=101.5, close=102.8),  # entry candle
        _bar("SOXL", 9, 33, open_=102.8, high=103, low=102, close=102.5),
    ]
    entry = _find(bars, n_candles=2)
    assert entry is not None
    assert entry.entry_index == 2
    assert entry.entry_price == 102.0            # candle 2 OPEN
    assert entry.entry_time == bars[2].timestamp


def test_red_signal_no_trade():
    # N=2 with candle 1 red → no trade.
    bars = [
        _green("SOXL", 9, 30, 100),
        _red("SOXL", 9, 31, 101),
        _green("SOXL", 9, 32, 100),
        _green("SOXL", 9, 33, 101),
    ]
    assert _find(bars, n_candles=2) is None


def test_doji_first_candle_is_not_green_no_trade():
    # close == open is NOT green (strict >). N=1 doji → no trade.
    bars = [
        _bar("SOXL", 9, 30, open_=100, high=100.5, low=99.5, close=100),  # doji
        _bar("SOXL", 9, 31, open_=100, high=101, low=99, close=100.5),
    ]
    assert _find(bars, n_candles=1) is None


def test_not_enough_candles_no_trade():
    # N=2 but only 2 candles → no candle N+1 to enter on.
    bars = [_green("SOXL", 9, 30, 100), _green("SOXL", 9, 31, 101)]
    assert _find(bars, n_candles=2) is None


# ---------------------------------------------------------------------------
# No lookahead
# ---------------------------------------------------------------------------


def test_no_lookahead_entry_candle_close_and_later_candles_irrelevant():
    # The entry decision uses candles 0..N-1 only; the fill is candle N's OPEN.
    # Make candle N's CLOSE and candle N+1 wildly different — must not change entry.
    good = [
        _green("SOXL", 9, 30, 100),
        _green("SOXL", 9, 31, 101),
        _bar("SOXL", 9, 32, open_=102.0, high=200, low=50, close=180),  # crazy close
        _bar("SOXL", 9, 33, open_=180, high=999, low=1, close=5),       # crazy next candle
    ]
    entry = _find(good, n_candles=2)
    assert entry.entry_index == 2
    assert entry.entry_price == 102.0  # the OPEN, unaffected by this bar's close or later bars

    # And a later candle turning green/red cannot RETROACTIVELY create a signal:
    # candle 1 is red, so no matter what candles 2,3 do, there is no trade.
    bad = [
        _green("SOXL", 9, 30, 100),
        _red("SOXL", 9, 31, 101),      # kills the signal
        _green("SOXL", 9, 32, 100),
        _green("SOXL", 9, 33, 101),
    ]
    assert _find(bad, n_candles=2) is None


# ---------------------------------------------------------------------------
# Exit modes
# ---------------------------------------------------------------------------


def test_resolve_exit_at_close_returns_last_close():
    bars = [
        _bar("X", 9, 32, open_=102, high=103, low=101, close=102.5),
        _bar("X", 15, 59, open_=102.5, high=104, low=101, close=103.2),
    ]
    idx, price, reason = resolve_exit_at_close(bars, entry_index=0)
    assert (idx, price, reason) == (1, 103.2, "eod")


def test_close_mode_holds_to_session_close():
    # N=2 fires; close mode → exit at the last bar's close.
    bars = [
        _green("SOXL", 9, 30, 100),
        _green("SOXL", 9, 31, 101),
        _bar("SOXL", 9, 32, open_=102, high=103, low=101, close=102.5),  # entry @ 102
        _bar("SOXL", 15, 59, open_=102.5, high=104, low=101, close=103.0),  # close 103
    ]
    result = _run({"SOXL": bars}, n_candles=2, exit_mode="close")
    assert result.full.n_signals == 1
    assert result.exit_reason_counts["eod"] == 1
    # gross log = ln(103/102).
    assert result.full.gross_total_return_pct == pytest.approx(math.expm1(math.log(103.0 / 102.0)))


def test_bracket_mode_hits_target_via_reused_resolve_exit():
    # N=1 fires; bracket target 1% from entry 102 → 103.02. A later bar's high reaches it.
    bars = [
        _green("SOXL", 9, 30, 100),
        _bar("SOXL", 9, 31, open_=102, high=102.1, low=101.9, close=102.05),  # entry @ 102
        _bar("SOXL", 9, 32, open_=102.05, high=103.5, low=102, close=103.2),  # hits 103.02
    ]
    result = _run({"SOXL": bars}, n_candles=1, exit_mode="bracket",
                  profit_target=0.01, stop_pct=0.01)
    assert result.exit_reason_counts["target"] == 1


# ---------------------------------------------------------------------------
# Baseline (buy 09:30 open -> sell close, on fired days)
# ---------------------------------------------------------------------------


def test_baseline_is_open_to_close_on_fired_day():
    bars = [
        _bar("SOXL", 9, 30, open_=100.0, high=101, low=99, close=100.5),  # green, day OPEN = 100
        _green("SOXL", 9, 31, 101),
        _bar("SOXL", 9, 32, open_=102, high=103, low=101, close=102.5),   # entry
        _bar("SOXL", 15, 59, open_=102.5, high=104, low=101, close=105.0),  # day CLOSE = 105
    ]
    result = _run({"SOXL": bars}, n_candles=2, exit_mode="close")
    # Baseline buys the 09:30 open (100.0) and sells the close (105.0): +5% gross.
    assert result.full.baseline_gross_return_pct == pytest.approx(math.expm1(math.log(105.0 / 100.0)))


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------


def test_split_chronological_even_and_odd():
    assert split_chronological([1, 2, 3, 4]) == ([1, 2], [3, 4])
    assert split_chronological([1, 2, 3, 4, 5]) == ([1, 2, 3], [4, 5])  # extra to in-sample
    assert split_chronological([]) == ([], [])


def test_end_to_end_split_produces_distinct_windows():
    # Four sessions on four dates, all firing (N=1). The split puts 2 in-sample, 2 OOS.
    days = [(2025, 7, 1), (2025, 7, 2), (2025, 7, 3), (2025, 7, 4)]
    bars = []
    for i, d in enumerate(days):
        bars += [
            _green("SOXL", 9, 30, 100 + i, d=d),
            _bar("SOXL", 9, 31, open_=100 + i, high=101 + i, low=99 + i, close=100.5 + i, d=d),
        ]
    result = _run({"SOXL": bars}, n_candles=1, exit_mode="close")
    assert result.full.n_sessions == 4
    assert result.in_sample.n_sessions == 2
    assert result.out_of_sample.n_sessions == 2
    assert result.in_sample.start_date.day == 1
    assert result.out_of_sample.start_date.day == 3


# ---------------------------------------------------------------------------
# Honest gross vs net
# ---------------------------------------------------------------------------


def test_gross_positive_trade_with_big_costs_still_counts_as_gross_win():
    # A green fired day with a gross-positive close-mode trade, but costs so large the
    # trade is net-negative. It must still count as a GROSS win.
    bars = [
        _green("SOXL", 9, 30, 100),
        _bar("SOXL", 9, 31, open_=100, high=100.1, low=99.9, close=100.05),  # entry @ 100
        _bar("SOXL", 15, 59, open_=100.05, high=100.6, low=99.9, close=100.5),  # +0.5% gross
    ]
    result = _run({"SOXL": bars}, n_candles=1, exit_mode="close",
                  fee_bps=100.0, slippage_bps=0.0)  # 200 bps round trip >> +0.5%
    assert result.full.win_rate == 1.0          # gross win
    assert result.full.avg_gross_win > 0
    assert result.full.total_return_pct < 0      # net negative
    assert result.full.gross_total_return_pct > 0


# ---------------------------------------------------------------------------
# Symbol-agnostic + config validation
# ---------------------------------------------------------------------------


def test_symbol_agnostic_tqqq():
    bars = [
        _green("TQQQ", 9, 30, 50),
        _green("TQQQ", 9, 31, 51),
        _bar("TQQQ", 9, 32, open_=52, high=53, low=51, close=52.5),
        _bar("TQQQ", 15, 59, open_=52.5, high=53, low=52, close=52.8),
    ]
    result = _run({"TQQQ": bars}, symbol="TQQQ", n_candles=2, exit_mode="close")
    assert result.primary_symbol == "TQQQ"
    assert "TQQQ" in result.strategy_name
    assert result.full.n_signals == 1


@pytest.mark.parametrize("kwargs", [
    {"n_candles": 0},
    {"exit_mode": "nonsense"},
    {"exit_mode": "bracket", "profit_target": 0.0},
    {"exit_mode": "bracket", "stop_pct": 1.5},
    {"fee_bps": -1.0},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        OpeningCandleMomentum(OpeningCandleMomentumConfig(primary_symbol="SOXL", **kwargs))
