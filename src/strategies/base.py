# src/strategies/base.py

"""
Strategy abstraction layer.

A Strategy turns a list of OHLCVBar into a numpy array of trading signals.
Signals are integers: +1 (long), 0 (flat), -1 (short).

The output array length equals the input bar count, with leading
SIGNAL_FLAT (0) values during the indicator warmup period — i.e. the
contract is "one signal per bar, in lockstep with the bars you passed in".

Concrete strategies live in their own files under src/strategies/
(e.g. sma_crossover.py, rsi_mean_reversion.py). This base class enforces
the shape contract; concrete strategies provide the logic.

Design rule (mirrors src/brokers/base.py): backtest, risk, and execution
modules program against this Strategy interface, never against a specific
concrete strategy class. That keeps the system pluggable — swapping
strategies is a one-line change at the call site.
"""

# abc = Abstract Base Classes. ABC is the base class; abstractmethod is the
# decorator that marks a method as "must be implemented by any concrete subclass".
# Same import pattern as src/brokers/base.py — we use an ABC (not a Protocol)
# so Python raises TypeError at instantiation time if a subclass forgets a method.
from abc import ABC, abstractmethod

# numpy is the return-type backbone: signals are emitted as np.ndarray of ints.
# Using a numpy array (rather than a plain list) lets the backtester do vectorised
# math — element-wise multiplication with returns, cumulative sums for PnL, etc.
import numpy as np

# OHLCVBar is the canonical in-memory bar representation defined in
# src/data/schema.py. Every module that consumes bars imports it from there
# so a schema change (field rename, type change) propagates in one place.
from src.data.schema import OHLCVBar


# ---------------------------------------------------------------------------
# Signal value constants
# ---------------------------------------------------------------------------
# These three integers are the only legal values a strategy may emit.
# Defining them as named module-level constants makes calling code
# self-documenting: `signals[i] == SIGNAL_LONG` reads better than
# the magic number `signals[i] == 1`, and a typo like SIGNAL_LNG raises
# NameError immediately instead of silently comparing against the wrong int.

SIGNAL_LONG = 1    # take or hold a long position (bet the price will rise)
SIGNAL_FLAT = 0    # hold no position (out of the market — also the warmup default)
SIGNAL_SHORT = -1  # take or hold a short position (bet the price will fall)


# ---------------------------------------------------------------------------
# Abstract base class — the contract every strategy must fulfil
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """Abstract base for all trading strategies."""

    # ------------------------------------------------------------------
    # Abstract methods — concrete subclasses MUST override every one of
    # these. If they don't, Python raises TypeError at instantiation time.
    # That's the whole reason we use an ABC instead of a Protocol or a
    # plain duck-typed function: bugs surface at construction, not later
    # when some downstream caller invokes the missing method.
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(self, bars: list[OHLCVBar]) -> np.ndarray:
        """Return an integer signal per bar: SIGNAL_LONG, SIGNAL_FLAT, or SIGNAL_SHORT.

        The output array MUST be the same length as input bars (alignment contract,
        same as indicators in src/features/). Positions where the strategy can't yet
        emit a signal (warmup period) should return SIGNAL_FLAT (0) — no position
        until the strategy is warmed up.

        Implementations should be pure functions of the input bars — no side effects,
        no I/O, no broker calls. Strategies make decisions; brokers execute them.
        """
        # Ellipsis is the conventional "no body" placeholder for abstract methods —
        # it's a valid statement so Python is happy, and it signals to readers that
        # subclasses are expected to override this entirely.
        ...

    # `@property` makes `name` accessible as an attribute (`strategy.name`)
    # rather than a method call (`strategy.name()`). Combined with @abstractmethod,
    # subclasses must provide a concrete property of the same name. Order matters:
    # @property must be the outer decorator (applied last), @abstractmethod inner.
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name for logging and reports (e.g. 'SMA(50, 200)')."""
        # Subclasses typically return an f-string that includes their parameter
        # values, so logs and backtest reports identify exactly which configuration
        # produced a given result — critical when comparing parameter sweeps.
        ...
