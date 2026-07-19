# tests/test_rotation_verdict.py

"""
Hermetic unit tests for src/research/rotation_verdict.py — the fixed-parameter
walk-forward bridge for the cross-sectional rotation backtester.

All tests are pure: NO DuckDB, NO network, NO filesystem.  Bars are hand-built with a
make_month_end_bars helper copied from tests/test_portfolio.py (close == adj_close, so
the default price_field="close" reads the intended value).  Float comparisons use
np.testing.assert_allclose.  Matches the comment-heavy style of the sibling test files.

Run with:
    uv run pytest tests/test_rotation_verdict.py -v
"""

# math.isfinite pins that the verdict's scalar fields are real numbers (not NaN/inf).
import math

# numpy builds the synthetic streams and provides assert_allclose for float comparisons.
import numpy as np

# pytest.raises drives the guard tests.
import pytest

# datetime + timezone build the explicit month-end (and intra-month) bar timestamps.
from datetime import datetime, timezone

# OHLCVBar is the bar schema every test builds.
from src.data.schema import OHLCVBar

# _stitch_oos is imported to exercise the prepend-zero crux directly (test 5).
from src.research.walk_forward import _stitch_oos

# The units under test: the month-end fold splitter, the per-fold rotation and benchmark
# producers, the minimal fold-result wrapper, and the top-level verdict.
from src.research.rotation_verdict import (
    RotationVerdict,
    rotation_month_end_folds,
    _rotation_fold_returns,
    _benchmark_fold_returns,
    _fold_result,
    rotation_walk_forward,
)


# ---------------------------------------------------------------------------
# Synthetic bar helpers.
# ---------------------------------------------------------------------------


def make_month_end_bars(
    dates: list[datetime],
    closes: list[float],
    symbol: str = "T",
) -> list[OHLCVBar]:
    """Build one OHLCVBar per (date, close), in the given (ascending) order.

    Copied from tests/test_portfolio.py so these tests stay hermetic: close == adj_close
    so the default price_field="close" reads the intended value; OHLC fields are dummies.
    """
    # One bar per (date, close) pair, in the given order (caller supplies ascending dates).
    return [
        OHLCVBar(
            symbol=symbol,        # arbitrary; the code reads only timestamp + price
            timestamp=ts,         # the bar's real trading date
            open=c,               # dummy
            high=c,               # dummy
            low=c,                # dummy
            close=c,              # the field the default price_field reads
            adj_close=c,          # equal to close -> basis-agnostic
            volume=1000,          # dummy
            timeframe="1d",       # daily
            source="test",        # provenance marker
        )
        for ts, c in zip(dates, closes)
    ]


def monthly_dates(n: int, start_year: int = 2015, start_month: int = 1) -> list[datetime]:
    """Return n ascending month-end-ish dates, one per consecutive calendar month.

    Uses day 28 (valid in every month) so each bar lands in a distinct calendar month;
    month_end_indices only cares that consecutive bars change month, so with one bar per
    month EVERY bar is a month-end and M == n.
    """
    # Walk consecutive (year, month) pairs, emitting one timestamp each.
    dates: list[datetime] = []
    year, month = start_year, start_month
    for _ in range(n):
        # Day 28 is safe for all 12 months; the calendar month is what matters.
        dates.append(datetime(year, month, 28, tzinfo=timezone.utc))
        # Advance one calendar month, rolling the year over after December.
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


# ---------------------------------------------------------------------------
# 1. rotation_month_end_folds — basic non-overlapping, complete-folds-only.
# ---------------------------------------------------------------------------


def test_month_end_folds_basic():
    """12 month-ends, train=4/test=2/step=None -> 3 non-overlapping complete folds."""
    # 12 monthly bars -> M = 12 month-ends (positions 0..11).
    ref_bars = make_month_end_bars(monthly_dates(12), [100.0 + i for i in range(12)])
    # Default step (None) -> test_months=2, so test spans tile with no overlap.
    folds = rotation_month_end_folds(ref_bars, train_months=4, test_months=2)
    # Hand-derived: closing boundary must be <= M-1 = 11.
    #   start=0 -> (0,4,6);  start=2 -> (2,6,8);  start=4 -> (4,8,10);
    #   start=6 -> test_end=12 > 11 -> dropped.
    assert folds == [(0, 4, 6), (2, 6, 8), (4, 8, 10)]
    # Test spans (decision positions) [4,6),[6,8),[8,10) are disjoint and contiguous.
    for (_, ts_i, te_i), (_, ts_next, _) in zip(folds, folds[1:]):
        # Next fold's test start equals this fold's test end -> no gap, no overlap.
        assert ts_next == te_i


# ---------------------------------------------------------------------------
# 2. rotation_month_end_folds — overlapping step + ragged tail.
# ---------------------------------------------------------------------------


def test_month_end_folds_step_and_tail():
    """step=1 (< test) yields overlapping test spans; the ragged tail fold is dropped."""
    # M = 12 month-ends.
    ref_bars = make_month_end_bars(monthly_dates(12), [100.0 + i for i in range(12)])
    # step_months=1 < test_months=2 -> adjacent test spans overlap by design.
    folds = rotation_month_end_folds(ref_bars, train_months=4, test_months=2, step_months=1)
    # starts 0..5 all fit (closing boundary <= 11); start=6 -> test_end=12 > 11 dropped.
    assert len(folds) == 6
    # First and last folds pin the tail rule and the step advance.
    assert folds[0] == (0, 4, 6)
    assert folds[-1] == (5, 9, 11)
    # The ragged tail fold (would be (6,10,12)) is NOT present.
    assert (6, 10, 12) not in folds
    # Overlap is honored: consecutive test spans share a decision position (step<test).
    assert folds[1][1] < folds[0][2]  # fold 1 test_start < fold 0 test_end -> overlap
    # Every emitted fold's closing boundary stays within the grid.
    for _, _, test_end_me in folds:
        assert test_end_me <= 11


# ---------------------------------------------------------------------------
# 3. rotation_month_end_folds — guards.
# ---------------------------------------------------------------------------


def test_month_end_folds_guards():
    """train_months<1, test_months<1, step_months<1, and too-few month-ends all raise."""
    ref_bars = make_month_end_bars(monthly_dates(12), [100.0 + i for i in range(12)])
    # train_months < 1 is nonsensical.
    with pytest.raises(ValueError):
        rotation_month_end_folds(ref_bars, train_months=0, test_months=2)
    # test_months < 1 is nonsensical.
    with pytest.raises(ValueError):
        rotation_month_end_folds(ref_bars, train_months=4, test_months=0)
    # step_months < 1 (when given) would loop forever / walk backwards.
    with pytest.raises(ValueError):
        rotation_month_end_folds(ref_bars, train_months=4, test_months=2, step_months=0)
    # Too few month-ends: need train+test+1 = 7, a 5-month-end grid has only 5.
    short_bars = make_month_end_bars(monthly_dates(5), [100.0 + i for i in range(5)])
    with pytest.raises(ValueError):
        rotation_month_end_folds(short_bars, train_months=4, test_months=2)


# ---------------------------------------------------------------------------
# 4. Fold boundary is ON the month-end — the no-lookahead pin.
# ---------------------------------------------------------------------------


def test_fold_boundary_is_on_month_end_no_lookahead():
    """The first test return is the winner ranked AS OF the boundary, not a lookahead pick.

    Construct a 2-symbol basket where the momentum winner FLIPS exactly at the train/test
    boundary D2, AND the OTHER symbol is the better performer over the first test holding
    period.  The correct (as-of-D2, no-lookahead) choice is B (higher trailing return at
    D2), earning log(B[D3]/B[D2]).  A lookahead bug that ranked by the realized D2->D3
    return would instead pick A and earn log(A[D3]/A[D2]) — a DIFFERENT, detectable number.
    """
    # 5 month-ends D0..D4, one bar per month.
    dates = monthly_dates(5)
    # A: momentum LOSER at D2 but BEST performer over (D2,D3].
    #   trailing at D2 (D1->D2): 111/110 = +0.9%  (loses to B)
    #   realized over (D2,D3]:   133.2/111 = +20%  (a lookahead bug would chase this)
    # B: momentum WINNER at D2, weaker over (D2,D3].
    #   trailing at D2 (D1->D2): 130/105 = +23.8% (wins the as-of-D2 rank)
    #   realized over (D2,D3]:   143/130 = +10%
    bars_by_symbol = {
        "A": make_month_end_bars(dates, [100.0, 110.0, 111.0, 133.2, 140.0], "A"),
        "B": make_month_end_bars(dates, [100.0, 105.0, 130.0, 143.0, 150.0], "B"),
    }
    # One fold: train=[0,2), test decisions [2,4), closing boundary D4.  train>lookback (2>1).
    folds = rotation_month_end_folds(bars_by_symbol["A"], train_months=2, test_months=2)
    assert folds == [(0, 2, 4)]
    train_start_me, test_start_me, test_end_me = folds[0]
    # Produce the test-span rotation returns for this fold (cost-free so the first return
    # is the pure holding log return, unperturbed by turnover cost).
    stream = _rotation_fold_returns(
        bars_by_symbol,
        reference_symbol="A",
        train_start_me=train_start_me,
        test_start_me=test_start_me,
        test_end_me=test_end_me,
        lookback=1,
        top_n=1,
        cost_rate=0.0,
    )
    # The FIRST test return is period (D2, D3], decided as of D2 -> B held -> log(B3/B2).
    assert stream[0] == pytest.approx(np.log(143.0 / 130.0))
    # And it is NOT the lookahead-buggy value that would have chased A's realized return.
    assert stream[0] != pytest.approx(np.log(133.2 / 111.0))


# ---------------------------------------------------------------------------
# 5. Prepend-zero keeps ALL real test-span returns through _stitch_oos.
# ---------------------------------------------------------------------------


def test_prepend_zero_keeps_all_real_returns():
    """_stitch_oos on a prepended-zero fold yields exactly the real returns, none dropped."""
    # Three real test-span per-bar returns (NO structural leading zero).
    test_returns = np.array([0.01, -0.02, 0.03], dtype=float)
    # Wrap via the module's minimal fold-result (which prepends the single 0.0).
    fold = _fold_result(test_returns)
    # Stitch a single-fold list through the REUSED scoring seam.
    oos_returns, _equity, _tot, _sharpe, _mdd, _sortino = _stitch_oos([fold], 252)
    # The stitched OOS returns must have length 3 (NOT 2) — the prepend protected the
    # first real return from _stitch_oos's [1:] strip.
    assert len(oos_returns) == len(test_returns)
    # And they must be exactly the real returns, in order, untouched.
    np.testing.assert_allclose(oos_returns, test_returns)


# ---------------------------------------------------------------------------
# 6. Benchmark holds the WHOLE basket (equal weight), distinct from rotation.
# ---------------------------------------------------------------------------


def test_benchmark_holds_whole_basket():
    """The benchmark stream is 0.5*rA + 0.5*rB per bar; the top_n=1 rotation is not."""
    # 5 month-ends, one bar per month; both symbols rise so both are always eligible.
    dates = monthly_dates(5)
    bars_by_symbol = {
        "A": make_month_end_bars(dates, [100.0, 110.0, 120.0, 130.0, 140.0], "A"),
        "B": make_month_end_bars(dates, [100.0, 120.0, 140.0, 160.0, 180.0], "B"),
    }
    # One fold (0,2,4): test periods (D2,D3] and (D3,D4].  train>lookback (2>1).
    folds = rotation_month_end_folds(bars_by_symbol["A"], train_months=2, test_months=2)
    train_start_me, test_start_me, test_end_me = folds[0]
    # Benchmark: equal-weight hold of BOTH symbols (top_n=2, no filter), cost-free.
    bench = _benchmark_fold_returns(
        bars_by_symbol,
        reference_symbol="A",
        train_start_me=train_start_me,
        test_start_me=test_start_me,
        test_end_me=test_end_me,
        lookback=1,
        cost_rate=0.0,
    )
    # Hand-computed equal-weight (0.5 each) log-return combination per test bar.
    expected_bench = np.array(
        [
            0.5 * np.log(130.0 / 120.0) + 0.5 * np.log(160.0 / 140.0),  # (D2,D3]
            0.5 * np.log(140.0 / 130.0) + 0.5 * np.log(180.0 / 160.0),  # (D3,D4]
        ],
        dtype=float,
    )
    np.testing.assert_allclose(bench, expected_bench)
    # The top_n=1 rotation holds only the momentum winner (B both periods here), so its
    # stream differs from the equal-weight basket — proving the benchmark is not the
    # momentum-filtered subset.
    rotation = _rotation_fold_returns(
        bars_by_symbol,
        reference_symbol="A",
        train_start_me=train_start_me,
        test_start_me=test_start_me,
        test_end_me=test_end_me,
        lookback=1,
        top_n=1,
        cost_rate=0.0,
    )
    # B is the winner both periods -> rotation is pure B; distinct from the 50/50 blend.
    expected_rotation = np.array(
        [np.log(160.0 / 140.0), np.log(180.0 / 160.0)], dtype=float
    )
    np.testing.assert_allclose(rotation, expected_rotation)
    assert not np.allclose(bench, rotation)


# ---------------------------------------------------------------------------
# Shared multi-fold basket for the verdict-level tests (7 and 8).
# ---------------------------------------------------------------------------


def _make_multi_fold_basket() -> dict[str, list[OHLCVBar]]:
    """A 10-month, 2-symbol basket whose winner rotates, giving real test-span turnover."""
    # 10 month-ends, one bar per month -> M = 10.
    dates = monthly_dates(10)
    # Two crossing series so the top-1 winner flips across rebalances (creates turnover,
    # so a positive cost_rate actually bites the rotation stream).  All prices rise, so
    # both symbols stay eligible for the always-held benchmark.
    return {
        "A": make_month_end_bars(
            dates, [100.0, 120.0, 121.0, 150.0, 151.0, 185.0, 186.0, 225.0, 226.0, 270.0], "A"
        ),
        "B": make_month_end_bars(
            dates, [100.0, 101.0, 135.0, 136.0, 180.0, 181.0, 235.0, 236.0, 300.0, 301.0], "B"
        ),
    }


# ---------------------------------------------------------------------------
# 7. Costs can only reduce return — costed <= zero-cost, for BOTH strategy and B&H.
# ---------------------------------------------------------------------------


def test_zero_cost_vs_costed():
    """A positive cost_rate cannot increase total return, for rotation OR the benchmark."""
    bars_by_symbol = _make_multi_fold_basket()
    # Common config: train=3 > lookback=1, test=2 (step defaults to 2 -> disjoint folds).
    kwargs = dict(
        reference_symbol="A",
        lookback=1,
        top_n=1,
        train_months=3,
        test_months=2,
    )
    # Frictionless verdict.
    free = rotation_walk_forward(bars_by_symbol, cost_rate=0.0, **kwargs)
    # Costed verdict (10 bps == 0.001), same everything else.
    costed = rotation_walk_forward(bars_by_symbol, cost_rate=0.001, **kwargs)
    # Transaction costs are a drag: the costed total return cannot exceed the free one.
    assert costed.oos_total_return <= free.oos_total_return
    # The benchmark is charged the SAME cost convention; its ordering is the same
    # (its test-span turnover is ~zero, so it is equal-or-lower, never higher).
    assert costed.bh_total_return <= free.bh_total_return


# ---------------------------------------------------------------------------
# 8. Verdict fields are all populated and finite; n_folds is correct.
# ---------------------------------------------------------------------------


def test_verdict_fields_populated():
    """A small multi-fold run returns a RotationVerdict with finite fields and right n_folds."""
    bars_by_symbol = _make_multi_fold_basket()
    # train=3, test=2, step=None(->2), M=10.  Folds: (0,3,5),(2,5,7),(4,7,9); (6,9,11)
    # dropped (test_end 11 > M-1=9) -> 3 folds.
    verdict = rotation_walk_forward(
        bars_by_symbol,
        reference_symbol="A",
        lookback=1,
        top_n=1,
        train_months=3,
        test_months=2,
        cost_rate=0.001,
    )
    # It IS a RotationVerdict with the expected fold count.
    assert isinstance(verdict, RotationVerdict)
    assert verdict.n_folds == 3
    # Every scalar metric must be a finite real number (no NaN/inf leaking through).
    for value in (
        verdict.oos_sharpe,
        verdict.oos_sortino,
        verdict.oos_max_drawdown,
        verdict.oos_total_return,
        verdict.bh_sharpe,
        verdict.bh_sortino,
        verdict.bh_max_drawdown,
        verdict.bh_total_return,
    ):
        assert math.isfinite(value)


# ---------------------------------------------------------------------------
# 9. rotation_walk_forward guard — train_months must exceed lookback.
# ---------------------------------------------------------------------------


def test_rotation_walk_forward_train_months_guard():
    """train_months <= lookback raises (warm-up cannot satisfy the lookback)."""
    bars_by_symbol = _make_multi_fold_basket()
    # train_months == lookback violates the strict train_months > lookback guard.
    with pytest.raises(ValueError):
        rotation_walk_forward(
            bars_by_symbol,
            reference_symbol="A",
            lookback=3,
            top_n=1,
            train_months=3,
            test_months=2,
        )
