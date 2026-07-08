# tests/test_momentum_walkforward.py

"""
Hermetic unit tests for scripts/momentum_walkforward.py.

All tests are pure: NO DuckDB, NO network, NO filesystem.  Synthetic OHLCVBar
lists are built in-memory (make_bars / make_oscillating_bars below), matching the
make_price_bars style in tests/test_walk_forward.py — sequential one-per-day UTC
timestamps and strictly-positive prices, with adj_close set EQUAL to close so a
synthetic run is basis-agnostic and these tests never depend on real dividend
adjustment data.

Run with:
    uv run pytest tests/test_momentum_walkforward.py -v
"""

# datetime + timedelta + timezone build sequential UTC-aware timestamps for the
# synthetic bars, matching the project-wide tz-aware-UTC bar convention.
from datetime import datetime, timedelta, timezone

# math.sin drives the oscillating price path in make_oscillating_bars, which
# needs both rising and falling regimes to force TSMOM into FLAT periods.
import math

# pytest.approx is used for the scalar sign-convention assertions.
import pytest

# WalkForwardResult is the return type run_one_symbol produces (via the validator);
# the end-to-end tests assert the wiring returns one.
from src.research.walk_forward import WalkForwardResult

# OHLCVBar is the bar schema the helpers build and the runner consumes.
from src.data.schema import OHLCVBar

# The two units under test: the pure summarize_verdict helper and the thin
# per-symbol runner that wires splits + engine + strategy + validator.
from scripts.momentum_walkforward import summarize_verdict, run_one_symbol


# ---------------------------------------------------------------------------
# Synthetic bar helpers — in-memory, no I/O.
# ---------------------------------------------------------------------------


def make_bars(n: int, start_price: float = 100.0) -> list[OHLCVBar]:
    """Build n OHLCVBars with one-per-day UTC timestamps, close=adj_close=start_price+i.

    Monotonically increasing and strictly positive, so every log return is
    well-defined.  adj_close is set EQUAL to close so the bars are basis-agnostic:
    a run on "close" and a run on "adj_close" see identical prices.
    """
    # Anchor at 2024-01-01 UTC; the absolute date is irrelevant to the math, but
    # pinning it keeps test output deterministic and easy to eyeball.
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        OHLCVBar(
            symbol="T",
            timestamp=base + timedelta(days=i),  # strictly increasing, one per day
            open=start_price + i,
            high=start_price + i,
            low=start_price + i,
            close=start_price + i,       # monotone up; log returns defined every step
            adj_close=start_price + i,   # EQUAL to close → basis-agnostic synthetic data
            volume=1,
            timeframe="1d",
            source="test",
        )
        for i in range(n)
    ]


def make_oscillating_bars(n: int, base_price: float = 100.0, amp: float = 30.0) -> list[OHLCVBar]:
    """Build n OHLCVBars whose price rises and falls (sine wave) to force FLAT periods.

    TSMOM goes FLAT whenever its trailing-lookback return is <= 0, which only
    happens on a path that DECLINES over the lookback horizon.  A monotone series
    (make_bars) never goes flat after warmup, so the cash-on-flat test needs a
    path with genuine down-legs.  A sine wave around base_price with amplitude amp
    (kept < base_price so prices stay strictly positive) supplies both up-legs
    (long) and down-legs (flat), guaranteeing at least some flat bars.
    """
    # Same UTC anchor + one-per-day cadence as make_bars for consistency.
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # Period ~60 bars (i/10 → 2π every ~63 days) gives several full up/down cycles
    # across a few-hundred-bar series, so flat and long regimes both appear inside
    # the out-of-sample test windows, not just the discarded warmup.
    return [
        OHLCVBar(
            symbol="T",
            timestamp=base + timedelta(days=i),
            open=base_price + amp * math.sin(i / 10.0),
            high=base_price + amp * math.sin(i / 10.0),
            low=base_price + amp * math.sin(i / 10.0),
            close=base_price + amp * math.sin(i / 10.0),       # oscillates 70..130
            adj_close=base_price + amp * math.sin(i / 10.0),   # EQUAL to close
            volume=1,
            timeframe="1d",
            source="test",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. summarize_verdict — pure sign conventions.
# ---------------------------------------------------------------------------


def test_summarize_verdict_sign_conventions():
    """sharpe_delta, sortino_delta and dd_reduction carry documented signs (risk-cut positive)."""
    # Args are POSITIONAL in the order:
    # (symbol, n_folds, oos_sharpe, oos_sortino, bh_sharpe, bh_sortino,
    #  oos_max_dd, bh_max_dd, oos_return, bh_return).
    # Case A — momentum is BETTER: higher Sharpe/Sortino AND a smaller drawdown.
    # oos_sharpe=1.0 vs bh_sharpe=0.5 → sharpe_delta = 1.0 - 0.5 = +0.5.
    # oos_sortino=1.2 vs bh_sortino=0.6 → sortino_delta = 1.2 - 0.6 = +0.6.
    # oos_max_dd=0.10 vs bh_max_dd=0.25 → dd_reduction = 0.25 - 0.10 = +0.15
    # (POSITIVE = momentum drew down LESS, i.e. cut risk).  Returns are arbitrary.
    row = summarize_verdict("X", 3, 1.0, 1.2, 0.5, 0.6, 0.10, 0.25, 0.20, 0.10)
    assert row.sharpe_delta == pytest.approx(0.5)
    assert row.sortino_delta == pytest.approx(0.6)
    assert row.dd_reduction == pytest.approx(0.15)

    # Case B — momentum is WORSE on drawdown: oos_max_dd=0.30 vs bh_max_dd=0.20 →
    # dd_reduction = 0.20 - 0.30 = -0.10 (NEGATIVE = momentum drew down MORE than
    # simply holding).  Sortinos are equal (0.5 vs 0.5) → sortino_delta = 0.0.
    # Confirms both the negative dd case and a zero-delta Sortino are correct.
    worse = summarize_verdict("Y", 3, 0.5, 0.5, 0.5, 0.5, 0.30, 0.20, 0.0, 0.0)
    assert worse.dd_reduction == pytest.approx(-0.10)
    assert worse.sortino_delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. run_one_symbol — end-to-end wiring returns a WalkForwardResult.
# ---------------------------------------------------------------------------


def test_run_one_symbol_returns_walkforwardresult():
    """run_one_symbol wires splits+engine+strategy+validator into a WalkForwardResult.

    SMALL windows (NOT the production 756/252): 400 daily bars, train=150, test=60,
    lookback=2 months.  Daily bars are ~21 trading days/month, so each fold's
    full_bars (train+test = 210 ≈ 10 months) has ~10 month-end observations — well
    above lookback=2, so TSMOM's "> lookback month-ends" guard clears comfortably.
    Folds: starts 0, 60, 120, 180 → 4 folds (start=240 needs 450 > 400 bars).
    """
    bars = make_bars(400)  # monotone up; a clean, valid TSMOM run after warmup
    result = run_one_symbol(
        bars,
        lookback=2,            # small lookback so month-end guard clears on 210-bar folds
        train=150,
        test=60,
        step=None,             # → walk_forward_splits uses test_size (non-overlapping)
        price_field="adj_close",
        cash_yield=0.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )

    # The runner must return the validator's frozen result type.
    assert isinstance(result, WalkForwardResult)
    # At least one fold was scored, and per_fold length tracks n_folds exactly.
    assert result.n_folds >= 1
    assert len(result.per_fold) == result.n_folds


# ---------------------------------------------------------------------------
# 3. run_one_symbol — basis-matched run never trips the basis guard.
# ---------------------------------------------------------------------------


def test_run_one_symbol_basis_matched_does_not_raise_guard():
    """A basis-matched run completes without the Day-32 price-basis guard firing.

    run_one_symbol feeds ONE price_field ("adj_close") to BOTH the Backtester and
    the TimeSeriesMomentumStrategy, so the engine's basis and the strategy's basis
    are equal BY CONSTRUCTION.  The walk_forward_validate basis-consistency guard
    can therefore never fire here — this test pins that by-construction equality:
    the call must complete without raising ValueError.
    """
    bars = make_bars(400)  # same safe windows/lookback as the wiring test above
    # No pytest.raises wrapper: any ValueError (including a spurious guard fire)
    # would propagate and fail the test.  Completing and returning is the pass.
    result = run_one_symbol(
        bars,
        lookback=2,
        train=150,
        test=60,
        step=None,
        price_field="adj_close",   # SAME basis to engine and strategy → guard inert
        cash_yield=0.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    assert isinstance(result, WalkForwardResult)


# ---------------------------------------------------------------------------
# 4. run_one_symbol — cash-on-flat yield lifts OOS return (never lowers it).
# ---------------------------------------------------------------------------


def test_cash_yield_lifts_oos_return():
    """A positive --cash-yield never lowers OOS return, and lifts it when bars go FLAT.

    The oscillating price path has genuine down-legs, so TSMOM's trailing return
    goes <= 0 over parts of the series and the strategy sits FLAT there.  On those
    flat bars a positive annual_cash_yield earns interest, so the cash_yield=0.05
    run's stitched OOS total return must be >= the cash_yield=0.0 run's (strictly
    greater whenever any OOS bar was flat).  The engine's own
    test_all_flat_earns_annualized_yield proves the exact per-bar magnitude; here
    we only pin the monotonic direction through the runner, which is robust whether
    or not a given fold happened to contain a flat bar.
    """
    bars = make_oscillating_bars(400)  # rises and falls → guaranteed flat periods

    # Identical run except for the cash yield, so any difference is attributable
    # solely to interest earned on flat bars.
    no_yield = run_one_symbol(
        bars, lookback=2, train=150, test=60, step=None,
        price_field="adj_close", cash_yield=0.0, fee_bps=0.0, slippage_bps=0.0,
    )
    with_yield = run_one_symbol(
        bars, lookback=2, train=150, test=60, step=None,
        price_field="adj_close", cash_yield=0.05, fee_bps=0.0, slippage_bps=0.0,
    )

    # Both complete and return the validator's result type.
    assert isinstance(no_yield, WalkForwardResult)
    assert isinstance(with_yield, WalkForwardResult)

    # Cash-on-flat can only ADD return (interest is never negative), so the yielded
    # run's OOS total return is >= the no-yield run's — strictly greater if any OOS
    # bar was flat, equal only if the strategy was never flat out-of-sample.
    assert with_yield.oos_total_return >= no_yield.oos_total_return
