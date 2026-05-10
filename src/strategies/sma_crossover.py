# src/strategies/sma_crossover.py

"""
SMA Crossover strategy — the canonical Golden Cross / Death Cross rule.

Rule:
  - Go LONG  when the fast SMA is above the slow SMA (fast > slow).
  - Go SHORT when the fast SMA is below the slow SMA (fast < slow).
  - Stay FLAT when either SMA hasn't warmed up yet (NaN) or the two are
    exactly equal (rare, but going flat on a tie is the conservative choice).

Default parameters are (50, 200) — the canonical "Golden Cross" used by
essentially every retail charting platform. A bullish crossover of the
50-day above the 200-day SMA is the textbook Golden Cross; the reverse is
the Death Cross. Shorter-horizon variants like (20, 50) or (10, 30) are
also common; the windows are constructor arguments so callers can sweep
parameters without touching the strategy code.

This module produces a *position* signal, not a *trade* signal: each
element of the output array says "I should be long/short/flat on this
bar", not "buy" or "sell". The backtester derives trades by diffing
consecutive positions (FLAT->LONG = buy, LONG->FLAT = sell), which keeps
the same strategy usable for both backtests and live execution.
"""

# numpy is the workhorse for the vectorised crossover comparison and the
# integer signal array we return. Importing as `np` matches the rest of
# the project (indicators, base strategy) so the conventions stay uniform.
import numpy as np

# OHLCVBar is the canonical bar dataclass. We type the input as
# `list[OHLCVBar]` to match the contract declared in src.strategies.base.
from src.data.schema import OHLCVBar

# `sma` does the actual rolling-mean math, including NaN fill for the
# warmup window. We delegate to it so this strategy stays purely about
# the *rule* (cross direction), not the indicator computation.
from src.features.indicators import sma

# Strategy is the ABC we inherit from; the SIGNAL_* constants are the
# only legal output values. Using the named constants (rather than the
# raw ints 1/0/-1) keeps the rule expression self-documenting.
from src.strategies.base import (
    Strategy,
    SIGNAL_LONG,
    SIGNAL_FLAT,
    SIGNAL_SHORT,
)


# ---------------------------------------------------------------------------
# SMA Crossover concrete strategy
# ---------------------------------------------------------------------------

class SMACrossoverStrategy(Strategy):
    """Long when fast SMA > slow SMA, short when fast SMA < slow SMA, else flat."""

    def __init__(self, fast_window: int = 50, slow_window: int = 200) -> None:
        # fast_window < 1 has no mathematical meaning (you can't average
        # zero or fewer prices). Catching it here gives a clearer error
        # than the deeper one `sma()` would raise on first call.
        if fast_window < 1:
            # f-string echoes the actual value so the traceback is actionable.
            raise ValueError(f"fast_window must be >= 1, got {fast_window}")

        # If slow_window <= fast_window, the two SMAs collapse to the same
        # series (when equal) or have inverted roles (when slow < fast),
        # making the "cross" notion meaningless. Reject up front.
        if slow_window <= fast_window:
            # Both numbers in the message so the user sees the mismatch.
            raise ValueError(
                f"slow_window ({slow_window}) must be > fast_window ({fast_window})"
            )

        # Stash the validated parameters on the instance so generate_signals
        # and the `name` property can read them. Plain attributes (no
        # property/dataclass) keep this minimal — they're effectively
        # immutable by convention; callers should construct a new strategy
        # rather than mutate an existing one.
        self.fast_window = fast_window
        self.slow_window = slow_window

    @property
    def name(self) -> str:
        # f-string includes the actual window values so logs and backtest
        # reports identify the exact configuration that produced a result.
        # This is critical when comparing parameter sweeps side by side.
        return f"SMA({self.fast_window}, {self.slow_window})"

    def generate_signals(self, bars: list[OHLCVBar]) -> np.ndarray:
        # Empty input is almost always a programming error upstream (e.g.
        # a symbol with no data slipped through). Fail loudly with the
        # same wording used by the indicators module for consistency.
        if not bars:
            raise ValueError("bars must be non-empty")

        # We need strictly more bars than the slow window so that at least
        # one position emits a non-NaN slow SMA. `len(bars) == slow_window`
        # would technically produce a single valid SMA value, but `>` keeps
        # the contract crisp ("history beyond the slow window is required")
        # and matches how callers typically size their backfill.
        if len(bars) <= self.slow_window:
            # Echo both numbers so the user can see the gap at a glance.
            raise ValueError(
                f"need more than slow_window ({self.slow_window}) bars to compute "
                f"the crossover, got {len(bars)}"
            )

        # Compute both moving averages over the close-price series.
        # `sma()` returns an np.ndarray aligned 1:1 with `bars`, with NaN
        # at the leading positions where the rolling window isn't full.
        sma_fast = sma(bars, self.fast_window)
        sma_slow = sma(bars, self.slow_window)

        # Pre-allocate the output array filled with zeros (== SIGNAL_FLAT).
        # int8 is the smallest signed integer dtype; signals only take
        # values in {-1, 0, 1}, so int8 saves ~8x memory vs the default
        # int64 in long backtests across many symbols. Comparison and
        # arithmetic semantics are identical to int64 for this range.
        signals = np.zeros(len(bars), dtype=np.int8)

        # Mask of positions where BOTH SMAs are warmed up (non-NaN). We
        # need this guard because comparisons involving NaN evaluate to
        # False in both directions (NaN > x and NaN < x are both False),
        # which would silently leave warmup positions as FLAT — correct
        # by accident — but using an explicit mask makes the intent clear
        # and protects against future indicator changes.
        valid = ~np.isnan(sma_fast) & ~np.isnan(sma_slow)

        # Long mask: warmed-up AND fast strictly above slow.
        # Strict inequality (`>`, not `>=`) means equality falls through
        # to FLAT, which is the conservative tie-breaking behaviour.
        long_mask = valid & (sma_fast > sma_slow)

        # Short mask: warmed-up AND fast strictly below slow.
        short_mask = valid & (sma_fast < sma_slow)

        # Boolean-mask assignment: write SIGNAL_LONG only at positions
        # where long_mask is True; everything else stays at the existing
        # zero (SIGNAL_FLAT). Same pattern for SIGNAL_SHORT.
        signals[long_mask] = SIGNAL_LONG
        signals[short_mask] = SIGNAL_SHORT

        # Positions where either SMA was NaN, or where the two were
        # exactly equal, remain at SIGNAL_FLAT from the np.zeros default —
        # exactly the warmup / tie-break behaviour we want.
        return signals
