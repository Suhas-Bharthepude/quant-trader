# tests/test_intraday_no_lookahead.py

"""
THE load-bearing intraday invariant — the analogue of tests/test_backtest.py's
test_no_lookahead_bias for the event-driven ORB sim.  Never weaken these to make
something pass.

The intraday sim makes a decision AT minute t using only data through minute t:
  * a breakout that only occurs at t+1 must NOT enter at t;
  * a confirmation that only becomes true at t+1 must NOT retroactively enable an
    earlier entry (the confirmation as-of sample is backward-only);
  * an exit fills on the FIRST chronological bar that touches a level, never a
    later one.
"""

import math
from datetime import datetime
from zoneinfo import ZoneInfo

from src.data.schema import OHLCVBar
from src.intraday.orb_strategy import OpeningRangeBreakout, OpeningRangeBreakoutConfig
from src.intraday.session import ET, build_sessions
from src.intraday.sim import resolve_exit, run_orb_backtest


UTC = ZoneInfo("UTC")


def _bar(symbol, hh, mm, *, open_, high, low, close, d=(2024, 3, 4)):
    ts = datetime(d[0], d[1], d[2], hh, mm, tzinfo=ET).astimezone(UTC)
    return OHLCVBar(
        symbol=symbol, timestamp=ts, open=open_, high=high, low=low, close=close,
        adj_close=close, volume=1000, timeframe="1m", source="test",
    )


def _soxl_or(high=100.0):
    return [
        _bar("SOXL", 9, 30, open_=99, high=high - 0.5, low=98, close=99.5),
        _bar("SOXL", 9, 40, open_=99.5, high=high, low=99, close=99.8),
    ]


def _run(bars_by_symbol, **cfg):
    config = OpeningRangeBreakoutConfig(
        primary_symbol="SOXL", confirmation_symbols=("SPY",), **cfg
    )
    strat = OpeningRangeBreakout(config)
    return run_orb_backtest(bars_by_symbol, strat, config)


# ---------------------------------------------------------------------------
# Breakout timing: a future breakout must not pull an entry earlier.
# ---------------------------------------------------------------------------


def test_breakout_only_at_next_bar_does_not_enter_early():
    # 09:45 does NOT break out (high 99.5 < OR high 100); 09:46 DOES (high 101).
    # A lookahead bug would enter at 09:45 using the 09:46 breakout.
    soxl = _soxl_or() + [
        _bar("SOXL", 9, 45, open_=99, high=99.5, low=98.5, close=99.2),   # no breakout
        _bar("SOXL", 9, 46, open_=99.3, high=101, low=99, close=100.5),   # breakout here
        _bar("SOXL", 15, 59, open_=100.5, high=100.8, low=100, close=100.4),
    ]
    spy = [
        _bar("SPY", 9, 30, open_=399, high=400.0, low=398, close=399.5),
        _bar("SPY", 9, 45, open_=400.5, high=402, low=400, close=401),  # confirming throughout
        _bar("SPY", 9, 46, open_=401, high=402, low=400.5, close=401.5),
    ]
    result = _run({"SOXL": soxl, "SPY": spy})
    assert result.n_trades == 1
    trade = result.trades[0]
    # Entry is the 09:46 bar (the real breakout), NOT the 09:45 bar.
    assert trade.entry_time == soxl[3].timestamp  # index 3 == 09:46 bar
    assert trade.entry_price == 100.0


# ---------------------------------------------------------------------------
# Confirmation timing: a future confirmation must not enable an earlier entry.
# ---------------------------------------------------------------------------


def test_future_confirmation_does_not_enable_earlier_entry():
    # SOXL breaks out at 09:45 AND 09:46. SPY only starts confirming at 09:46
    # (its 09:45 close is below its OR high). A lookahead bug (sampling SPY's
    # future bar) would confirm and enter at 09:45; correct behaviour enters 09:46.
    soxl = _soxl_or() + [
        _bar("SOXL", 9, 45, open_=99, high=101, low=99, close=100.2),  # breakout
        _bar("SOXL", 9, 46, open_=100.2, high=101, low=99.8, close=100.5),  # still broken out
        _bar("SOXL", 15, 59, open_=100.5, high=100.8, low=100, close=100.4),
    ]
    spy = [
        _bar("SPY", 9, 30, open_=399, high=400.0, low=398, close=399.5),
        _bar("SPY", 9, 45, open_=399.6, high=400.4, low=399, close=399.2),  # NOT confirming
        _bar("SPY", 9, 46, open_=400.5, high=402, low=400, close=401.5),    # confirms
    ]
    result = _run({"SOXL": soxl, "SPY": spy})
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.entry_time == soxl[3].timestamp  # 09:46, not 09:45
    # If a lookahead bug entered at 09:45, entry_time would equal soxl[2].timestamp.
    assert trade.entry_time != soxl[2].timestamp


# ---------------------------------------------------------------------------
# Exit timing: fill on the FIRST bar that touches a level, not a later one.
# ---------------------------------------------------------------------------


def test_exit_fills_on_first_touching_bar_not_later():
    # entry 100, target 112. Two bars both reach the target; the exit must be the
    # FIRST one (index 1), not the second (index 2).
    bars = [
        _bar("X", 9, 45, open_=100, high=101, low=99.5, close=100.5),
        _bar("X", 9, 46, open_=101, high=112.5, low=100.5, close=112),   # first touch
        _bar("X", 9, 47, open_=112, high=120.0, low=111, close=118),     # later, must be ignored
    ]
    idx, price, reason = resolve_exit(bars, 0, 100.0, 0.06, 2.0)
    assert idx == 1
    assert reason == "target"
    assert price == 112.0


def test_stop_before_target_across_bars_is_respected_chronologically():
    # The stop is touched on bar 1; the target only on bar 2. Chronology means the
    # trade is already stopped out and can never reach the later target.
    bars = [
        _bar("X", 9, 45, open_=100, high=101, low=99.5, close=100.5),
        _bar("X", 9, 46, open_=99, high=100, low=93.5, close=95),      # stop (94) touched first
        _bar("X", 9, 47, open_=95, high=113.0, low=94.5, close=112),   # target only now
    ]
    idx, price, reason = resolve_exit(bars, 0, 100.0, 0.06, 2.0)
    assert idx == 1
    assert reason == "stop"
    assert price == 94.0
