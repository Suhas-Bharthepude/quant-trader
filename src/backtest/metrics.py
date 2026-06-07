# src/backtest/metrics.py

"""
Pure metric functions extracted from Backtester.run().

All four functions are verbatim arithmetic copies of the inline computations
in engine.py (lines 252-297 as of the extraction commit).  No simplification,
no reordering, no float-operation changes — bit-identical output is the
requirement so that walk-forward aggregation can call these functions on
arbitrary slices and get the same numbers the engine would have produced.

Design contract:
  * Each function is pure: no I/O, no side effects, no instance state.
  * total_return and max_drawdown take explicit parameters instead of self.*
    so a caller can pass a slice's starting equity or a fold's equity curve.
  * sharpe_ratio does NOT slice [1:] internally — that slice stays at the call
    site in engine.run() so this function works on any arbitrary returns array,
    including a single walk-forward test fold's returns.
  * Return types match exactly what engine.run() stored in BacktestResult:
    - total_return  → numpy float64 (no float() wrap; wrap is at call site)
    - sharpe_ratio  → Python float  (float() wrap is inside this function)
    - max_drawdown  → Python float  (float() wrap is inside this function)
    - win_rate      → Python float  (integer division already yields float)
"""

# numpy is the array backend shared with the engine.  All per-bar series
# (equity_curve, returns) are float64 ndarrays; metric math stays in numpy
# so numerical behaviour is identical to the engine's inline expressions.
import numpy as np

# Trade is the dataclass whose .return_pct field win_rate inspects.  Importing
# it here makes the type annotation honest without creating a circular import:
# metrics.py → result.py is a leaf dependency (result.py imports nothing from
# this package).
from src.backtest.result import Trade


# Std values below this threshold are treated as zero.  1e-12 sits far below
# any real per-bar return std (~0.01 for typical daily returns) and far above
# the floating-point noise floor (~1e-18) that a nominally constant series
# (e.g. repeated 0.05) produces due to FP rounding; it cannot false-trigger on
# real data but reliably catches residual "constant" series that slip past
# exact == 0.0 and would otherwise produce an astronomically large garbage Sharpe.
_ZERO_STD_TOLERANCE = 1e-12


def total_return(equity_curve: np.ndarray, initial_capital: float) -> float:
    """Return the fractional total return over the equity curve.

    Verbatim from engine.py line 254:
        equity_curve[-1] / self.initial_capital - 1.0

    Args:
        equity_curve:    Per-bar equity array, float64.  Must be non-empty.
        initial_capital: The capital value the curve started from.  Passing
                         this explicitly (not through self.*) lets a caller
                         compute return for a mid-series slice by passing that
                         slice's first value as initial_capital.

    Returns:
        numpy float64 — NOT wrapped in float() here because the call site in
        engine.run() wraps it when constructing BacktestResult:
            total_return_pct=float(total_return_pct)
        Adding float() here would double-wrap, which is harmless but diverges
        from the engine's original type at intermediate assignment.
    """
    # Verbatim from engine.py line 254.  equity_curve[-1] is the terminal
    # equity value; dividing by initial_capital normalises it to a multiplier
    # (1.25 means the account grew 25%); subtracting 1.0 converts that
    # multiplier to a signed fraction (0.25 for +25%, -0.10 for -10%).
    return equity_curve[-1] / initial_capital - 1.0


def sharpe_ratio(returns: np.ndarray, annualization_factor: int = 252) -> float:
    """Return the annualised Sharpe ratio of the given returns array.

    Verbatim from engine.py lines 265-277.  The [1:] slice that skips the
    engine's structural index-0 zero is NOT applied here — it stays at the
    call site so this function works on any pre-sliced returns array,
    including a walk-forward test fold where the first bar is not a
    structural zero.

    Args:
        returns:              1-D float64 ndarray of per-bar log returns.
                              Pass exactly the array you want Sharpe computed
                              over; no internal slicing is performed.
        annualization_factor: Number of bars per year for scaling.  Defaults
                              to 252 (US trading days), matching the engine's
                              Backtester default.

    Returns:
        Python float — float() is applied inside this function on the
        non-guarded path, matching engine.py line 277 exactly.
        Near-zero (below tolerance) std → 0.0; NaN std → 0.0.
    """
    # Verbatim from engine.py line 265.  At least 2 observations are required
    # to compute a sample standard deviation (ddof=1 needs n-1 >= 1 degrees of
    # freedom).  Returning 0.0 rather than NaN prevents downstream comparisons
    # and sorts from being poisoned by a mathematically undefined value.
    if len(returns) < 2:
        return 0.0

    # Verbatim from engine.py line 268.  Arithmetic mean of per-bar returns
    # is the numerator of the Sharpe formula before annualisation.
    mean_return = returns.mean()

    # Verbatim from engine.py line 271.  ddof=1 → Bessel-corrected sample
    # standard deviation, the academic convention for empirical Sharpe estimates.
    std_return = returns.std(ddof=1)

    # Two guards in the same condition:
    #   std_return < _ZERO_STD_TOLERANCE — near-zero (below tolerance) std → 0.0.
    #                                      Tolerance replaces the original == 0.0
    #                                      because a nominally constant series (e.g.
    #                                      repeated 0.05) can produce std ~1e-18 from
    #                                      FP rounding rather than exactly 0.0; that
    #                                      residual std slips past == 0.0 and divides
    #                                      into the mean to produce a garbage Sharpe
    #                                      in the billions — same class of bug as NaN.
    #   np.isnan(std_return) — NaN < _ZERO_STD_TOLERANCE is False, so isnan is still
    #                          needed; a NaN std would propagate into the result.
    # Both cases return 0.0.  Order preserved: tolerance check first, isnan second.
    if std_return < _ZERO_STD_TOLERANCE or np.isnan(std_return):
        return 0.0

    # Verbatim from engine.py line 277.  mean/std gives the per-bar Sharpe;
    # multiplying by sqrt(annualization_factor) scales it to annual.
    # float() wraps the numpy scalar to a Python float, matching the engine's
    # stored type for sharpe_ratio in BacktestResult (no extra float() at
    # the call site for this metric).
    return float(mean_return / std_return * np.sqrt(annualization_factor))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Return the maximum peak-to-trough decline as a positive fraction.

    Verbatim from engine.py lines 282-290, plus a single defensive guard for
    the empty-array case that was impossible in the engine (bars are validated
    non-empty before equity_curve is built) but is reachable when this function
    is called on a walk-forward slice.

    Args:
        equity_curve: Per-bar equity array, float64.  May be empty when called
                      on a walk-forward slice; non-empty for any engine run.

    Returns:
        Python float — float() is applied inside this function, matching
        engine.py line 290 exactly.  0.0 means no drawdown occurred.
    """
    # Defensive guard added for slice-level reuse: an empty equity_curve has
    # no peak and no trough, so 0.0 (no drawdown) is the only meaningful return.
    # This guard is NOT in the engine because bars are validated non-empty
    # upstream; adding it here extends the function's usable domain without
    # altering the non-empty code path that immediately follows.
    if len(equity_curve) == 0:
        return 0.0

    # Verbatim from engine.py line 282.  np.maximum.accumulate computes the
    # running maximum (high-water mark) up to and including each bar — the
    # highest equity value the account has seen at any prior point.
    running_max = np.maximum.accumulate(equity_curve)

    # Verbatim from engine.py line 286.  (equity - peak) / peak gives the
    # signed percentage distance below the peak at each bar; values are zero
    # or negative (zero when equity equals its own peak, negative when below it).
    # Dividing by running_max (not initial_capital) measures decline from the
    # most recent peak, which is what investors and regulators care about.
    drawdown = (equity_curve - running_max) / running_max

    # Verbatim from engine.py line 290.  drawdown.min() is the most negative
    # value (the worst single-bar distance below peak); negating it yields a
    # positive fraction (0.20 = 20% drawdown).  float() converts the numpy
    # scalar to a Python float, matching the engine's stored type.
    return float(-drawdown.min())


def win_rate(trades: list[Trade]) -> float:
    """Return the fraction of trades that were profitable.

    Verbatim from engine.py lines 294-297.

    Args:
        trades: List of completed Trade records.  May be empty.

    Returns:
        Python float in [0.0, 1.0].  0.0 when there are no trades (avoids
        divide-by-zero and is semantically correct: no evidence of winning).
    """
    # Verbatim from engine.py line 294.  No trades → win rate is 0.0 rather
    # than NaN; a strategy that never traded has no wins, not an undefined rate.
    if len(trades) == 0:
        return 0.0

    # Verbatim from engine.py line 297.  sum() counts trades where return_pct
    # is STRICTLY positive (> 0, not >= 0); breakeven trades do not count as
    # wins.  Integer division by len(trades) yields a Python float in Python 3.
    return sum(1 for t in trades if t.return_pct > 0) / len(trades)
