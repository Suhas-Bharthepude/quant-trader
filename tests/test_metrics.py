# tests/test_metrics.py

"""
Unit tests for src/backtest/metrics.py.

Every test uses hand-built arrays and hand-computed expected values so that a
bug in the function under test cannot silently produce a wrong result that
happens to equal a wrong expected value derived from the same buggy code.

No DuckDB, no network, no fixtures — inputs are small numpy arrays or
minimal Trade dataclasses assembled inline.  The test names encode what
property is being asserted so a failure message names the broken contract.

Run with:
    uv run pytest tests/test_metrics.py -v
"""

# numpy builds the hand-crafted input arrays and provides sqrt for computing
# exact expected values from the same arithmetic the functions use.
import numpy as np

# pytest is the test runner; pytest.approx handles floating-point comparisons
# with a tolerance that is tight enough to catch real bugs but loose enough
# to ignore IEEE 754 rounding at the last bit.
import pytest

# datetime + timezone produce UTC-aware timestamps required by the Trade
# dataclass constructor.  win_rate only reads return_pct, so the actual date
# values are stubs — but the frozen dataclass enforces all fields are supplied.
from datetime import datetime, timezone

# The four pure functions under test, imported by name so the test bodies
# read as plain function calls rather than module-qualified attribute lookups.
from src.backtest.metrics import (
    downside_deviation,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)

# Trade is the frozen dataclass whose return_pct field win_rate inspects.
# Imported here so _make_trade can construct valid instances.
from src.backtest.result import Trade


# ---------------------------------------------------------------------------
# Module-level stubs shared by win_rate tests
# ---------------------------------------------------------------------------

# Sentinel UTC timestamps used wherever Trade requires entry_time / exit_time.
# win_rate only reads return_pct, so specific dates carry no semantic weight;
# we use a fixed pair so every make_trade call is reproducible and readable.
_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2024, 1, 2, tzinfo=timezone.utc)


def _make_trade(return_pct: float) -> Trade:
    """Return a minimal Trade with only return_pct set to a meaningful value.

    Trade is a frozen dataclass — every field must be provided at construction
    even if win_rate ignores most of them.  Stub values (price=100.0,
    direction=1, bars_held=1) are chosen to be self-evidently inert rather
    than meaningful, so a future reader sees at a glance that the test is
    only exercising return_pct logic.
    """
    # entry_price / exit_price are stubs; win_rate never reads them.
    # direction=1 (long) and bars_held=1 satisfy the type annotations but
    # are otherwise irrelevant to all win_rate assertions.
    return Trade(
        entry_time=_T0,
        exit_time=_T1,
        entry_price=100.0,
        exit_price=100.0,
        direction=1,
        return_pct=return_pct,
        bars_held=1,
    )


# ===========================================================================
# sharpe_ratio
# ===========================================================================


def test_sharpe_known_array_hand_computed():
    # Array [1.0, 2.0, 3.0]:
    #   mean   = (1+2+3)/3 = 2.0  (exact integer arithmetic)
    #   std    = sqrt(((1-2)² + (2-2)² + (3-2)²) / (3-1))
    #          = sqrt((1 + 0 + 1) / 2)
    #          = sqrt(1) = 1.0    (exact)
    #   sharpe = mean/std * sqrt(252) = 2.0/1.0 * sqrt(252) = 2.0 * sqrt(252)
    # The integer-exact mean and std make this the cleanest hand-computable
    # case: the only irrational is sqrt(252), which numpy evaluates precisely.
    arr = np.array([1.0, 2.0, 3.0])
    # Expected: 2.0 * sqrt(252) — computed with the same numpy call the
    # function uses so there is no second-rounding-path discrepancy.
    expected = 2.0 * np.sqrt(252)
    assert sharpe_ratio(arr) == pytest.approx(expected)


def test_sharpe_empty_array_returns_zero():
    # len([]) = 0 < 2 → the len-guard returns 0.0 before any arithmetic.
    # Testing the empty case separately from len=1 because the guard is a
    # single condition (< 2) that covers both; exercising each edge confirms
    # the guard fires correctly on both sides of the threshold.
    assert sharpe_ratio(np.array([])) == 0.0


def test_sharpe_single_element_returns_zero():
    # len([x]) = 1 < 2 → len-guard fires.  Sample std with ddof=1 and n=1
    # would give sqrt(0/0) = NaN; the guard prevents that computation entirely.
    assert sharpe_ratio(np.array([0.05])) == 0.0


def test_sharpe_constant_array_returns_zero():
    # All values equal → deviations are all 0 → sample std == 0.0 exactly.
    # The std==0 guard fires before division, returning 0.0 rather than inf.
    # The array has len=3 (passes the len-guard) so this specifically exercises
    # the std==0 branch, not the len-guard.
    arr = np.array([0.05, 0.05, 0.05])
    assert sharpe_ratio(arr) == 0.0


def test_sharpe_nan_std_returns_zero():
    # An array containing NaN propagates NaN through .mean() and .std(ddof=1).
    # len=2 passes the len-guard; std != 0.0 (NaN != 0.0 is True in Python);
    # np.isnan(std) is True, so the second branch of the guard returns 0.0.
    # This confirms the guard checks both conditions in order: == 0.0 first,
    # then isnan, matching the engine's original expression exactly.
    arr = np.array([1.0, np.nan])
    assert sharpe_ratio(arr) == 0.0


def test_sharpe_annualization_scales_by_sqrt_ratio():
    # Sharpe is proportional to sqrt(annualization_factor), so
    #   sharpe(arr, 63) / sharpe(arr, 252) == sqrt(63 / 252) == sqrt(1/4) == 0.5
    # Using [1.0, 2.0, 3.0] again (mean=2.0, std=1.0) so the base ratio is
    # exactly 2.0 and the only variable between the two calls is the sqrt factor.
    # sqrt(63/252) == 0.5 is an exact rational, making the expected ratio clean.
    arr = np.array([1.0, 2.0, 3.0])
    s252 = sharpe_ratio(arr, annualization_factor=252)
    s63  = sharpe_ratio(arr, annualization_factor=63)
    # sqrt(63/252) = sqrt(1/4) = 0.5 — the expected scaling factor.
    expected_ratio = np.sqrt(63 / 252)
    assert s63 == pytest.approx(s252 * expected_ratio)


def test_sharpe_does_not_skip_element_0():
    # CONTRACT: sharpe_ratio must NOT internally slice [1:] off the input.
    #
    # Why this contract exists:
    #   The engine passes strategy_returns[1:] at the call site to skip its
    #   own structural index-0 zero.  That slice is a call-site concern, not
    #   a function concern.  When the walk-forward validator calls this function
    #   on a test-fold's returns, bar 0 of the fold is real data and must be
    #   included in the computation.  Any internal [1:] would silently drop
    #   real return data and produce wrong fold-level Sharpe estimates.
    #
    # Test design:
    #   arr_full = [100.0, 0.1, 0.2].  Element 0 (100.0) is a large outlier
    #   that dominates mean and std; dropping it changes the result drastically:
    #     sharpe([100.0, 0.1, 0.2]) ≈ 9.2   (element 0 used)
    #     sharpe([0.1, 0.2])        ≈ 33.7  (element 0 dropped, wrong answer)
    #   The two values differ by ~24 units, well outside any numerical tolerance.
    arr_full = np.array([100.0, 0.1, 0.2])

    # Compute the expected result from all three elements using the same
    # numpy arithmetic the function applies, so the expected value is exact
    # rather than a hard-coded float constant that could drift with precision.
    m_full = arr_full.mean()
    s_full = arr_full.std(ddof=1)
    expected_all_three = float(m_full / s_full * np.sqrt(252))

    # Compute the wrong answer that an internal [1:] slice would produce,
    # so we can confirm the function's result is observably different from it.
    arr_sliced = np.array([0.1, 0.2])
    wrong_if_sliced = float(
        arr_sliced.mean() / arr_sliced.std(ddof=1) * np.sqrt(252)
    )

    result = sharpe_ratio(arr_full, annualization_factor=252)

    # Primary assertion: the function used all three elements.
    assert result == pytest.approx(expected_all_three)

    # Confirmatory: the would-be wrong answer is far enough away that
    # pytest.approx(wrong_if_sliced) would NOT match result, which means
    # this test would catch a regression that added [1:] inside the function.
    assert result != pytest.approx(wrong_if_sliced)


# ===========================================================================
# downside_deviation
# ===========================================================================


def test_downside_deviation_known_value():
    # Array [0.02, -0.03, 0.01, -0.01] with target=0.0:
    #   below-target deviations (np.minimum(r - 0.0, 0.0)):
    #     0.02 → 0.0     (>= target, clamped)
    #    -0.03 → -0.03   (below target)
    #     0.01 → 0.0     (>= target, clamped)
    #    -0.01 → -0.01   (below target)
    #   squared:        [0.0, 0.0009, 0.0, 0.0001]
    #   mean over N=4:  (0.0 + 0.0009 + 0.0 + 0.0001) / 4 = 0.001 / 4 = 0.00025
    #   per-bar dd:     sqrt(0.00025) = 0.0158113883...
    #   annualized:     0.0158113883... * sqrt(252)
    # The divide-by-N (not by the below-target count of 2) is asserted implicitly:
    # dividing by 2 would give sqrt(0.0005) instead, a materially different value.
    arr = np.array([0.02, -0.03, 0.01, -0.01])
    # Expected computed with the same numpy calls the function uses so there is
    # no second-rounding-path discrepancy; sqrt(0.00025) is the population RMS.
    expected = np.sqrt(0.00025) * np.sqrt(252)
    assert downside_deviation(arr) == pytest.approx(expected)


def test_downside_deviation_all_above_target_is_zero():
    # Every return is strictly positive → all >= target 0.0 → np.minimum clamps
    # every deviation to 0.0 → RMS is exactly 0.0 → annualized dd is 0.0.
    # This is the legitimate zero-downside case (no guard inside the function).
    arr = np.array([0.01, 0.02, 0.03])
    assert downside_deviation(arr) == 0.0


# ===========================================================================
# sortino_ratio
# ===========================================================================


def test_sortino_zero_downside_returns_zero_not_nan():
    # CRITICAL guard test: an all-positive returns array has no below-target
    # returns, so downside_deviation is 0.0 and the naive mean/dd would be inf/nan.
    # The guard (dd < _ZERO_STD_TOLERANCE) must fire and return 0.0 instead.
    arr = np.array([0.01, 0.02, 0.03])
    result = sortino_ratio(arr)
    # Must be exactly 0.0 (the guarded return), mirroring sharpe_ratio's zero-var path.
    assert result == 0.0
    # And explicitly NOT nan — proving the divide-by-zero was guarded, not executed.
    assert not np.isnan(result)


def test_sortino_len_below_2_returns_zero():
    # Small-sample short-circuit (len < 2) mirrors sharpe_ratio: both a 0-length
    # and a 1-length array return 0.0 before any dispersion arithmetic runs.
    assert sortino_ratio(np.array([])) == 0.0
    assert sortino_ratio(np.array([0.05])) == 0.0


def test_sortino_exceeds_sharpe_when_upside_volatile():
    # THE point of the metric.  This array has volatile UPSIDE (several large
    # positive returns) but small controlled DOWNSIDE (a few tiny negatives):
    #   full std (ddof=1) is inflated by the big positive swings, so Sharpe's
    #   denominator is large; downside deviation ignores those upside swings, so
    #   Sortino's denominator is much smaller.  With the same positive mean in the
    #   numerator, Sortino must therefore score HIGHER than Sharpe on this array.
    # This is exactly the behaviour that motivates adding the metric: upside
    # volatility should not be penalised as risk.
    arr = np.array([0.10, 0.12, -0.005, 0.09, -0.004, 0.11])
    sortino = sortino_ratio(arr, annualization_factor=252)
    sharpe = sharpe_ratio(arr, annualization_factor=252)
    assert sortino > sharpe


def test_sortino_matches_hand_computation():
    # Pin the exact annualisation algebra so a future mis-annualisation is caught.
    # Array [0.02, -0.03, 0.01, -0.01], target=0.0, annualization_factor=252:
    #   mean            = (0.02 - 0.03 + 0.01 - 0.01) / 4 = -0.01 / 4 = -0.0025
    #   annualized dd   = sqrt(0.00025) * sqrt(252)   (from the dd test above)
    #   sortino         = mean / annualized_dd * annualization_factor
    #                   = -0.0025 / (sqrt(0.00025) * sqrt(252)) * 252
    # The * annualization_factor (full 252, NOT sqrt) is the derived-and-commented
    # algebra in sortino_ratio; asserting the closed form here would fail loudly if
    # anyone "simplified" it to sqrt(252).
    arr = np.array([0.02, -0.03, 0.01, -0.01])
    mean_return = arr.mean()
    annualized_dd = np.sqrt(0.00025) * np.sqrt(252)
    expected = float(mean_return / annualized_dd * 252)
    assert sortino_ratio(arr, annualization_factor=252) == pytest.approx(expected)


# ===========================================================================
# max_drawdown
# ===========================================================================


def test_max_drawdown_monotonically_rising():
    # When equity strictly increases at every step, equity[i] == running_max[i]
    # at every bar (the running max is always the current value).
    # drawdown[i] = (equity[i] - equity[i]) / equity[i] = 0.0 everywhere.
    # drawdown.min() = 0.0 → max_drawdown = float(-0.0) = 0.0.
    equity = np.array([1.0, 1.1, 1.3, 1.5, 2.0])
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_known_curve_hand_computed():
    # Equity curve [1.0, 2.0, 1.0] — hand computation:
    #   running_max  = [1.0, 2.0, 2.0]   (accumulate: max(1)=1, max(1,2)=2, max(1,2,1)=2)
    #   drawdown     = [(1-1)/1, (2-2)/2, (1-2)/2]
    #                = [0.0,     0.0,     -0.5]
    #   drawdown.min() = -0.5
    #   max_drawdown   = float(-(-0.5)) = 0.5
    # The 50% drop from peak 2.0 to trough 1.0 is the canonical drawdown
    # example; the exact value 0.5 is a rational fraction with no rounding.
    equity = np.array([1.0, 2.0, 1.0])
    assert max_drawdown(equity) == pytest.approx(0.5)


def test_max_drawdown_empty_array():
    # Defensive guard added in metrics.py for walk-forward slice reuse:
    # an empty equity array has no peak or trough, so 0.0 is the only
    # meaningful return.  This case is impossible inside the engine (bars
    # are validated non-empty before equity_curve is built) but reachable
    # when the function is called on a zero-length test fold.
    assert max_drawdown(np.array([])) == 0.0


# ===========================================================================
# total_return
# ===========================================================================


def test_total_return_basic():
    # equity[-1] = 150.0, initial_capital = 100.0:
    #   total_return = 150.0 / 100.0 - 1.0 = 1.5 - 1.0 = 0.5
    # An equity curve that grew 50% should return exactly 0.5 (a simple
    # ratio with no rounding); pytest.approx handles any residual numpy
    # float64 vs Python float difference.
    equity = np.array([100.0, 120.0, 150.0])
    assert total_return(equity, initial_capital=100.0) == pytest.approx(0.5)


def test_total_return_slice_initial_capital():
    # CONTRACT: passing a slice's own first bar as initial_capital computes
    # the within-window return, not the return since some earlier start.
    #
    # equity = [120.0, 150.0], initial_capital = 120.0:
    #   total_return = 150.0 / 120.0 - 1.0 = 1.25 - 1.0 = 0.25
    #
    # This is exactly how the walk-forward validator will score each test fold:
    # pass the fold's equity curve slice and the fold's first equity value as
    # initial_capital so the metric reflects only that fold's performance,
    # independent of any prior fold's starting point.
    equity = np.array([120.0, 150.0])
    assert total_return(equity, initial_capital=120.0) == pytest.approx(0.25)


# ===========================================================================
# win_rate
# ===========================================================================


def test_win_rate_no_trades():
    # Empty list → len == 0 → the no-trades guard returns 0.0 directly.
    # No trades means no evidence of winning; 0.0 is the defined result rather
    # than NaN or a division error.
    assert win_rate([]) == 0.0


def test_win_rate_mix_win_loss_breakeven():
    # Four trades: +0.10 (win), -0.05 (loss), 0.00 (breakeven), +0.20 (win).
    # Expected win_rate = 2 / 4 = 0.5.
    # The breakeven trade (return_pct == 0.0) must NOT be counted as a win —
    # the contract is strictly > 0.  This is tested explicitly here so that
    # a regression changing > to >= would fail with result 3/4 = 0.75, not 0.5.
    trades = [
        _make_trade(0.10),   # win: return_pct > 0 → counted
        _make_trade(-0.05),  # loss: return_pct < 0 → not counted
        _make_trade(0.00),   # breakeven: return_pct == 0 → NOT counted (> 0, not >= 0)
        _make_trade(0.20),   # win: return_pct > 0 → counted
    ]
    assert win_rate(trades) == pytest.approx(0.5)


def test_win_rate_breakeven_not_a_win():
    # Isolated single-trade check for the strictly-greater-than-zero contract.
    # A single breakeven trade must return 0.0 (0 wins / 1 trade), not 1.0.
    # If the comparison were accidentally >= instead of >, this test fails
    # immediately with result 1.0, catching the regression in isolation
    # without any noise from other trade types in the list.
    trades = [_make_trade(0.0)]
    assert win_rate(trades) == 0.0
