# src/features/indicators.py

"""
Vectorized technical indicators that operate on lists of OHLCVBar.

Every public function in this module accepts a `list[OHLCVBar]` and returns a
numpy array aligned to the input bars' timestamps — element i of the output
corresponds to bars[i].  This shape contract lets callers (strategies,
backtests, plotting code) line up indicator values against bars by index
without rebuilding a separate timestamp axis.

NaN values appear at positions where the indicator has insufficient lookback
data.  For example, a 50-period SMA cannot be computed until 50 bars are
available, so the first 49 entries of its output array will be NaN.  Callers
must handle NaN explicitly (e.g. `np.isnan(...)` checks before placing trades)
rather than assume the array is always finite.
"""

# numpy provides the ndarray return type and the NaN sentinel used to mark
# under-warmed positions.  All indicator outputs are np.ndarray of dtype float.
import numpy as np

# pandas supplies the rolling-window machinery (Series.rolling) that powers
# moving averages, RSI, and similar lookback-based indicators.  Using pandas
# under the hood keeps the implementations short and well-tested without
# leaking the Series type out to callers — public outputs are numpy arrays.
import pandas as pd

# OHLCVBar is the schema-level bar type.  Importing it from src.data.schema
# (rather than redefining a local bar struct) keeps this module honest: if the
# schema changes, this file breaks loudly at import time instead of silently
# operating on stale field names.
from src.data.schema import OHLCVBar


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Single conversion point shared by every indicator function — DRY principle.
# If we ever need to switch to adj_close, change the source dtype, or add
# defensive sorting, it happens here once instead of in every indicator.
def _bars_to_close_series(bars: list[OHLCVBar]) -> pd.Series:
    """Return a pandas Series of close prices indexed by bar timestamp."""

    # Extract the timestamp axis in a single pass.  List comprehension is
    # faster than repeatedly appending to a Python list and is the idiomatic
    # way to project a list of dataclasses onto one of their fields.
    index = [b.timestamp for b in bars]

    # Extract the close-price axis the same way, preserving order so values[i]
    # corresponds to index[i].  We use `close` (not `adj_close`) because most
    # technical indicators are conventionally computed on raw close prices;
    # callers wanting adjusted prices can wrap their own variant.
    values = [b.close for b in bars]

    # Wrap the parallel lists into a pandas Series.  Naming the series "close"
    # makes downstream debug prints (and any DataFrame the caller builds) more
    # readable without affecting numerical behaviour.
    return pd.Series(values, index=index, name="close")


# ---------------------------------------------------------------------------
# Public indicators
# ---------------------------------------------------------------------------


# Simple Moving Average — the cornerstone trend indicator.  Implemented on
# top of pandas.rolling because the C-level rolling kernel is both faster
# than a hand-rolled numpy convolution and more correct on edge cases (NaN
# propagation, empty windows, off-by-one boundaries).  The output is then
# unwrapped to a numpy array so callers can do `sma_50 - sma_200` and other
# vectorized arithmetic without dragging pandas types through their code.
def sma(bars: list[OHLCVBar], window: int) -> np.ndarray:
    """Simple moving average of close prices over `window` bars."""

    # Empty input is almost always a programming error upstream (e.g. a
    # symbol with no data was passed through silently).  Failing loudly here
    # surfaces it immediately instead of returning a zero-length array that
    # would later trigger confusing IndexErrors in the caller.
    if not bars:
        # Explicit message names the offending argument so the traceback is
        # actionable without the caller having to read this source.
        raise ValueError("bars must be non-empty")

    # window < 1 has no mathematical meaning for a moving average; pandas
    # would raise its own (less specific) error a few lines later, but
    # catching it up front keeps the failure mode crisp and uniform with
    # the other indicators in this module.
    if window < 1:
        # f-string echoes the actual value so the user sees what they passed.
        raise ValueError(f"window must be >= 1, got {window}")

    # window > len(bars) means the indicator can never produce a non-NaN
    # value over the supplied data.  Rather than silently returning an
    # all-NaN array (which masks the configuration mistake), we reject it.
    if window > len(bars):
        # Including both numbers makes the mismatch obvious at a glance.
        raise ValueError(
            f"window ({window}) cannot exceed number of bars ({len(bars)})"
        )

    # Project the bars onto a close-price Series.  Done via the shared
    # helper so any future change to the conversion (e.g. defensive sort,
    # switch to adj_close) propagates to every indicator automatically.
    close = _bars_to_close_series(bars)

    # rolling(window).mean() computes the arithmetic mean of the most
    # recent `window` values at each position.  Positions with fewer than
    # `window` prior observations are filled with NaN — exactly the
    # warm-up semantics promised in the module docstring.
    rolling_mean = close.rolling(window=window).mean()

    # Drop down to numpy on the way out.  Callers compose indicators with
    # vectorized arithmetic (sma_fast - sma_slow, etc.); ndarray is the
    # lingua franca for that, and shedding the pandas index here means
    # downstream code never has to worry about index alignment surprises.
    return rolling_mean.to_numpy()


# Log returns — the workhorse return measure for quantitative analysis.
# Defined as r_t = ln(close_t / close_{t-1}).  Preferred over simple returns
# because they are time-additive (returns over consecutive periods sum,
# rather than compound), are approximately normal for small magnitudes (a
# baseline assumption of most statistical models), and handle multi-period
# compounding correctly without product-of-(1+r) gymnastics.
def log_returns(bars: list[OHLCVBar]) -> np.ndarray:
    """Per-bar log returns ln(close_t / close_{t-1})."""

    # Empty input has no defined return series — there is nothing to take
    # the ratio of.  Reject loudly so an upstream data-fetch bug does not
    # silently propagate as a zero-length array.
    if not bars:
        # Mirror the message used by sma() for consistency across indicators.
        raise ValueError("bars must be non-empty")

    # Project bars to a close-price Series via the shared helper, so every
    # indicator agrees on which field (close vs adj_close) and which index
    # (timestamps) backs its computation.
    close = _bars_to_close_series(bars)

    # close.shift(1) lags the series by one position: shifted[t] == close[t-1].
    # The first element becomes NaN (no prior bar exists), which is exactly
    # the "first value is NaN" warm-up semantics we promise in the contract.
    # Dividing the unshifted by the shifted series gives close_t / close_{t-1};
    # np.log is applied elementwise, propagating the leading NaN through.
    returns = np.log(close / close.shift(1))

    # Hand back a plain ndarray so callers can compose returns with other
    # numpy-based features (e.g. rolling vol, cumulative sum) without
    # worrying about pandas index alignment.
    return returns.to_numpy()


# Relative Strength Index — Wilder (1978), the canonical momentum oscillator.
# RSI sits in [0, 100]; readings above ~70 are conventionally "overbought"
# and below ~30 "oversold".  We use Wilder's exponential smoothing (not the
# default pandas EWMA weights), because that is what TradingView, MetaTrader,
# and Bloomberg all compute — matching the de-facto industry definition is
# more important than matching any particular textbook variant.
def rsi(bars: list[OHLCVBar], period: int = 14) -> np.ndarray:
    """Wilder's RSI on close prices over `period` bars (default 14)."""

    # Empty input has no defined RSI series; reject to surface upstream
    # data-fetch bugs immediately rather than masking them with an empty
    # array.  Same message text as the other indicators for consistency.
    if not bars:
        raise ValueError("bars must be non-empty")

    # period < 1 is mathematically meaningless for a smoothed average and
    # would only fail later inside pandas with a less specific error.  Guard
    # up front for a uniform validation experience across this module.
    if period < 1:
        # Echo the value back so the caller does not have to inspect locals.
        raise ValueError(f"period must be >= 1, got {period}")

    # We need one extra bar beyond `period` to produce even a single RSI
    # value: .diff() drops one observation as the leading NaN, and ewm with
    # min_periods=period needs `period` non-NaN inputs to emit anything.  So
    # period >= len(bars) guarantees an all-NaN output, which almost
    # certainly indicates a misconfiguration — fail loudly.
    if period >= len(bars):
        raise ValueError(
            f"period ({period}) must be less than number of bars ({len(bars)})"
        )

    # Project bars onto a close-price Series via the shared helper so all
    # indicators agree on which field and index back their math.
    close = _bars_to_close_series(bars)

    # Per-bar price change.  diff() inserts NaN at index 0 (no prior bar to
    # subtract), which propagates through the gain/loss split and the EWM
    # below — giving us the leading NaN warm-up region for free.
    delta = close.diff()

    # Gains: positive deltas as-is, negative deltas clipped to zero, NaN
    # preserved.  clip() leaves NaN untouched, so the leading NaN flows
    # through into the seeded recursion below.
    gains = delta.clip(lower=0)

    # Losses are conventionally expressed as positive magnitudes.  Negate
    # then clip so negative deltas become positive losses, positive deltas
    # become zero, and NaN stays NaN.  Equivalent to abs(delta) where
    # delta < 0 else 0, but vectorized in two steps without a branch.
    losses = (-delta).clip(lower=0)

    # Drop down to numpy for the smoothing step.  We need explicit indexed
    # access for Wilder's two-phase update (simple-mean seed, then
    # recursion), and numpy arrays make the index arithmetic clean.  The
    # leading element of each is NaN — courtesy of .diff() — but we never
    # read those positions because the recursion starts at index `period`.
    gains_np = gains.to_numpy()
    losses_np = losses.to_numpy()

    # n is the bar count, i.e. the length of every output array.  Hoisted
    # to a local for readability in the loop bound below.
    n = len(close)

    # Pre-allocate the smoothed averages as all-NaN arrays.  Anything we
    # do not explicitly write (positions 0 through period-1) stays NaN,
    # which is precisely the contractual warm-up region.
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    # ---- Phase 1: Wilder's simple-mean seed at index `period` ----
    #
    # Wilder (1978) seeds the smoothed averages with the arithmetic mean
    # of the first `period` price changes.  After .diff(), the first valid
    # delta sits at index 1, so the first `period` deltas occupy indices
    # 1..period (inclusive on the left, exclusive on the right in slice
    # notation: [1 : period + 1]).  Their mean lands at index `period` —
    # the position of the bar whose delta closes the seed window — and is
    # the FIRST non-NaN entry in the smoothed-average series.
    #
    # This simple-mean seed is what diverges from pandas.ewm(adjust=False)
    # (which would instead start the recursion from gains_np[1] as its
    # initial value).  The simple-mean seed is what matches TradingView,
    # MetaTrader, Bloomberg, ta-lib, and StockCharts on the very first
    # emitted RSI value — not just after the recursion has had time to
    # converge.
    avg_gain[period] = gains_np[1 : period + 1].mean()
    avg_loss[period] = losses_np[1 : period + 1].mean()

    # ---- Phase 2: Wilder's smoothing recursion for t > period ----
    #
    # The recursion is:
    #     avg[t] = (avg[t-1] * (period - 1) + x[t]) / period
    # Algebraically equivalent to alpha = 1/period exponential smoothing,
    # but with the deliberate simple-mean seed above instead of using x[1]
    # as the initial value.  We loop in Python because `period` is small
    # (typically 14) and the loop runs once over the series — vectorising
    # the recursion is possible (cumulative-product trick) but adds real
    # complexity for no measurable speedup on realistic bar counts.
    for t in range(period + 1, n):
        # Each step blends `(period - 1)` parts of the previous smoothed
        # value with one part of the current observation, then divides by
        # `period` to renormalise — the canonical Wilder update rule.
        avg_gain[t] = (avg_gain[t - 1] * (period - 1) + gains_np[t]) / period
        avg_loss[t] = (avg_loss[t - 1] * (period - 1) + losses_np[t]) / period

    # RS is the ratio of average gain to average loss; RSI maps it onto
    # [0, 100] via 100 - 100/(1+RS).  Division by zero is a real
    # possibility (a stretch where prices only rose), so we silence the
    # numpy warning for this single operation — the np.where below
    # assigns the principled value (100) for that case, making the
    # warning noise rather than signal.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Both operands are numpy arrays now, so this is a pure ndarray
        # division; NaN positions in the warm-up propagate to NaN in rs.
        rs = avg_gain / avg_loss

        # Standard RSI formula.  When avg_loss == 0 → rs == inf → this
        # already evaluates to 100, but we patch that case explicitly on
        # the next line to avoid relying on inf arithmetic.
        rsi_values = 100.0 - (100.0 / (1.0 + rs))

    # avg_loss == 0 means every change in the smoothing window was
    # non-negative — RSI is conventionally 100 (maximum strength).  Using
    # np.where lets us patch this case without rewriting the formula.
    rsi_array = np.where(avg_loss == 0, 100.0, rsi_values)

    # Ensure float dtype on the way out — np.where can return object
    # dtype in odd corner cases, and downstream arithmetic expects float64.
    return rsi_array.astype(np.float64)
