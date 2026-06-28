# tests/test_momentum_overfitting_tax.py

"""
Hermetic unit tests for scripts/momentum_overfitting_tax.py.

NO DuckDB, NO network, NO filesystem — synthetic OHLCVBar lists are built
in-memory (make_bars below), matching the style of tests/test_momentum_walkforward.py
(sequential one-per-day UTC timestamps, strictly-positive prices, adj_close set
EQUAL to close so a synthetic run is basis-agnostic).

The pure tax math (summarize_tax / TaxRow) is ALREADY covered by
tests/test_overfitting_tax.py, so it is NOT re-tested here.  These tests cover
only what is NEW in the sibling CLI: that the fixed + fitted momentum
walk-forward orchestration runs end-to-end on the adj_close basis, and that the
deliberate basis MISMATCH the sibling avoids would in fact trip the Day-32 guard.

Run with:
    uv run pytest tests/test_momentum_overfitting_tax.py -v
"""

# Sequential UTC-aware timestamps for the synthetic bars, matching the
# project-wide tz-aware-UTC bar convention.
from datetime import datetime, timedelta, timezone

# pytest.raises asserts the negative-control basis mismatch trips the guard.
import pytest

# Backtester is the scoring engine the CLI constructs; the tests build the same
# adj_close engine to exercise the real end-to-end path.
from src.backtest.engine import Backtester

# OHLCVBar is the bar schema the helper builds and the runner consumes.
from src.data.schema import OHLCVBar

# The fixed/fitted strategy and fitter the CLI wires together.
from src.strategies.time_series_momentum import TimeSeriesMomentumStrategy
from src.research.optuna_fit import make_tsmom_optuna_fit_fn

# The validator + splitter the CLI drives, and the reused pure tax helper/row.
from src.research.walk_forward import walk_forward_splits, walk_forward_validate
from scripts.overfitting_tax import TaxRow, summarize_tax


# ---------------------------------------------------------------------------
# Synthetic bar helper — in-memory, no I/O.
# ---------------------------------------------------------------------------


def make_bars(n: int, start_price: float = 100.0) -> list[OHLCVBar]:
    """Build n OHLCVBars with one-per-day UTC timestamps, close=adj_close=start_price+i.

    Monotonically increasing and strictly positive, so every log return is
    well-defined.  adj_close is set EQUAL to close so the bars are basis-agnostic:
    the run behaves identically on "close" and "adj_close", which is fine here —
    these tests pin orchestration and basis-threading, not adjustment values.
    """
    # Anchor at 2024-01-01 UTC; the absolute date is irrelevant to the math.
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


# ---------------------------------------------------------------------------
# 1. End-to-end: fixed + fitted run on adj_close without tripping the guard.
# ---------------------------------------------------------------------------


def test_momentum_tax_runs_end_to_end_on_synthetic_bars():
    """Fixed + fitted momentum walk-forward run end-to-end on a matched adj_close basis.

    Builds production-sized windows (train=756/test=252) on ~1300 daily bars so at
    least one fold exists, runs BOTH the fixed TSMOM(12) and the per-fold
    Optuna-fitted lookback through ONE adj_close engine, summarizes the tax, and
    asserts a well-formed TaxRow with n_folds >= 1.  Because the engine, the fixed
    strategy, and the fitter all use adj_close, the run completes WITHOUT raising
    the Day-32 price-basis guard — the basis-threading proof.
    """
    # ~1300 daily bars (~43 months) → train=756 (~3y) + test=252 (~1y) leaves room
    # for at least one fold (756 + 252 = 1008 <= 1300).
    bars = make_bars(1300)

    # The shared frictionless adj_close engine the CLI constructs once.
    engine = Backtester(price_field="adj_close")

    # Build splits once, exactly as the CLI does, at the production windows.
    splits = walk_forward_splits(bars, train_size=756, test_size=252)

    # Fixed baseline: static TSMOM(12) on adj_close, scored through the engine.
    fixed = walk_forward_validate(
        splits,
        TimeSeriesMomentumStrategy(12, price_field="adj_close"),
        backtester=engine,
    )

    # Fitted: per-fold Optuna lookback tuning, tiny/narrow so the test is fast.
    # The fitter threads price_field="adj_close" so its internal engine and each
    # trial's strategy match the shared engine's basis.
    record: list[dict] = []
    fitted = walk_forward_validate(
        splits,
        TimeSeriesMomentumStrategy(12, price_field="adj_close"),
        backtester=engine,
        fit_fn=make_tsmom_optuna_fit_fn(
            n_trials=3,
            seed=42,
            lookback_range=(3, 6),
            price_field="adj_close",
            record=record,
        ),
    )

    # Summarize via the reused pure helper exactly as the CLI's per-symbol body does.
    row = summarize_tax(
        "SYN",
        fitted.n_folds,
        fixed.oos_sharpe,
        fitted.oos_sharpe,
        fitted.bh_sharpe,
        [r["in_sample_sharpe"] for r in record],
    )

    # The orchestration produced a well-formed row over at least one real fold.
    assert isinstance(row, TaxRow)
    assert row.n_folds >= 1
    # Sanity: the fitter recorded one in-sample Sharpe per fold (basis threaded OK).
    assert len(record) == fitted.n_folds


# ---------------------------------------------------------------------------
# 2. Negative control: a deliberate basis mismatch DOES trip the Day-32 guard.
# ---------------------------------------------------------------------------


def test_momentum_tax_basis_mismatch_would_trip_guard():
    """A close engine + an adj_close strategy trips the Day-32 basis guard — proving why we thread one basis.

    This documents WHY the sibling CLI threads ONE basis everywhere: if the engine
    basis (close) and the strategy basis (adj_close) disagree, the walk-forward
    validator's price-basis consistency guard must raise rather than silently mix
    price return with total return.  The matched run above is safe precisely
    because this mismatched run is not.
    """
    bars = make_bars(1300)
    splits = walk_forward_splits(bars, train_size=756, test_size=252)

    # DELIBERATE mismatch: a close-basis engine but an adj_close-basis strategy.
    mismatched_engine = Backtester(price_field="close")
    adj_strategy = TimeSeriesMomentumStrategy(12, price_field="adj_close")

    # The Day-32 guard fires on fold 0 (before any signal math), naming the bases.
    with pytest.raises(ValueError):
        walk_forward_validate(splits, adj_strategy, backtester=mismatched_engine)
