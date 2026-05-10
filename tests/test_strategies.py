# tests/test_strategies.py

"""
Pure unit tests for src/strategies/base.py and src/strategies/sma_crossover.py.

These tests build synthetic OHLCVBar lists in-memory — no network calls,
no DuckDB, no yfinance — so they are fast and deterministic.  Run with:

    uv run pytest tests/test_strategies.py -v

The synthetic-bar approach lets each test pin down a single behaviour
(uptrend, downtrend, warmup, validation) without dragging real market
data into the assertion budget.  Same pattern as tests/test_indicators.py.
"""

# datetime + timezone + timedelta produce sequential UTC-aware timestamps
# for the synthetic bars.  timezone.utc matches the project-wide convention
# that every bar timestamp is timezone-aware UTC.
from datetime import datetime, timedelta, timezone

# numpy supplies the int8 dtype constant we assert on, plus the np.all
# helper used for vectorised mask assertions across the signal array.
import numpy as np

# pytest is the test runner; we use pytest.raises to assert on the
# validation failures emitted by the strategy constructors and methods.
import pytest

# OHLCVBar is the bar schema the strategies consume.  Importing from
# src.data.schema means a future schema rename surfaces here at import
# time rather than silently running against a stale shape.
from src.data.schema import OHLCVBar

# The Strategy ABC and the three signal constants under test.  We import
# the concrete names (not the module) so that calls stay short and a
# typo fails at import time instead of when the test executes.
from src.strategies.base import (
    Strategy,
    SIGNAL_LONG,
    SIGNAL_FLAT,
    SIGNAL_SHORT,
)

# The concrete strategy under test — the SMA crossover (Golden Cross)
# implementation that all of the meaningful behaviour tests target.
from src.strategies.sma_crossover import SMACrossoverStrategy


# ---------------------------------------------------------------------------
# Module-level helper (intentionally not a pytest fixture)
# ---------------------------------------------------------------------------


# A plain function — not a fixture — because every test wants different
# closes.  Fixtures shine when state is shared across tests; here each
# test passes its own list, so a helper keeps the call site explicit:
# `make_bars([1, 2, 3])` reads cleaner than juggling fixture params.
def make_bars(closes: list[float]) -> list[OHLCVBar]:
    """Build OHLCVBar list with given closes and sequential daily UTC timestamps starting 2024-01-01."""

    # Anchor every test series at 2024-01-01 UTC.  The exact date does not
    # matter — strategies never read calendar values — but pinning it
    # keeps tests deterministic and easy to eyeball when debugging.
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Build one bar per close price.  enumerate gives the day offset,
    # added to `base` so timestamps stay strictly increasing — a property
    # both the indicators and the strategy implicitly assume.
    return [
        OHLCVBar(
            symbol="TEST",          # arbitrary; strategies do not read this
            timestamp=base + timedelta(days=i),  # strictly increasing UTC
            open=c,                 # dummy: SMA only reads `close`
            high=c,                 # dummy
            low=c,                  # dummy
            close=c,                # the only field SMACrossover reads
            adj_close=c,            # dummy
            volume=1000,            # dummy
            timeframe="1d",         # daily; matches the timedelta above
            source="test",          # provenance marker for synthetic data
        )
        for i, c in enumerate(closes)
    ]


# ---------------------------------------------------------------------------
# Strategy ABC tests
# ---------------------------------------------------------------------------


def test_strategy_is_abstract():
    """Strategy() should be uninstantiable — it's an ABC with abstract methods."""

    # ABC + @abstractmethod means Python raises TypeError at instantiation
    # time if the class still has unimplemented abstract methods.  This
    # test pins that behaviour — the whole reason we chose ABC over
    # Protocol is to surface forgotten implementations early, so a
    # regression that made Strategy instantiable would silently break
    # the contract.
    with pytest.raises(TypeError):
        # Direct construction of the abstract base must fail; if it ever
        # succeeds, either the abstractmethod decorators were dropped or
        # someone added a concrete default that defeats the contract.
        Strategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# SMACrossoverStrategy constructor validation
# ---------------------------------------------------------------------------


def test_sma_crossover_validates_fast_window():
    """fast_window must be >= 1 — zero or negative values are rejected."""

    # fast_window == 0 has no meaning (no prices to average).  The
    # constructor catches it before any sma() call is made so the error
    # surfaces at object-build time, not later when generate_signals runs.
    with pytest.raises(ValueError):
        SMACrossoverStrategy(fast_window=0, slow_window=200)

    # Negative windows are even more obviously nonsense; same guard
    # catches them.  Using a clearly negative value (-5) makes the test
    # intent obvious to a reader skimming the file.
    with pytest.raises(ValueError):
        SMACrossoverStrategy(fast_window=-5, slow_window=200)


def test_sma_crossover_validates_window_ordering():
    """slow_window must be strictly greater than fast_window."""

    # slow < fast inverts the roles of the two SMAs and turns a
    # "Golden Cross" into nonsense — the constructor must reject it.
    with pytest.raises(ValueError):
        SMACrossoverStrategy(fast_window=200, slow_window=50)

    # Equal windows collapse the two SMAs to the same series, so the
    # crossover comparison is always equality and signals never fire.
    # We require *strict* inequality so this degenerate case is caught
    # at construction rather than producing an all-zero signal array.
    with pytest.raises(ValueError):
        SMACrossoverStrategy(fast_window=50, slow_window=50)


# ---------------------------------------------------------------------------
# Name property
# ---------------------------------------------------------------------------


def test_sma_crossover_name_property():
    """name is a human-readable string identifying the configuration."""

    # Build a strategy with non-default windows so the test would fail
    # if `name` ever hard-coded the default (50, 200) values.
    strategy = SMACrossoverStrategy(fast_window=20, slow_window=50)

    # The exact format is part of the contract — backtest reports and
    # logs grep on this string, so a change here must be a deliberate
    # update rather than an accidental drift.
    assert strategy.name == "SMA(20, 50)"


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_sma_crossover_signal_length_matches_bars():
    """Output array length must equal input bar count (alignment contract)."""

    # 250 bars is enough to comfortably exceed the slow window and is
    # close to a year of trading days — typical real-world sizing.
    bars = make_bars([100.0 + i * 0.1 for i in range(250)])

    # Use small windows (10, 30) so the warmup period is short relative
    # to the series; the lengths and dtype are all this test cares about.
    signals = SMACrossoverStrategy(fast_window=10, slow_window=30).generate_signals(bars)

    # The alignment contract says one signal per bar — same shape as
    # the indicators in src/features/.  This is what callers (the
    # backtester, the live trader) rely on for index-based lookups.
    assert len(signals) == 250

    # int8 is part of the contract: signals only take values -1/0/1, so
    # int8 saves 8x memory vs the numpy default int64 in long backtests.
    # If a future refactor accidentally promotes the dtype, this test
    # surfaces it before the memory regression hits production.
    assert signals.dtype == np.int8


# ---------------------------------------------------------------------------
# Warmup behaviour
# ---------------------------------------------------------------------------


def test_sma_crossover_warmup_is_flat():
    """During warmup (slow SMA still NaN) the strategy must emit SIGNAL_FLAT."""

    # 100 bars with a deterministic-but-non-monotone close pattern.  The
    # values themselves are irrelevant — what matters is that the slow
    # SMA cannot produce a non-NaN value before index `slow_window - 1`,
    # so every prior signal must be FLAT regardless of close values.
    closes = [100.0 + (i % 7) - 3 for i in range(100)]  # pseudo-random walk
    bars = make_bars(closes)

    # fast=10, slow=30 → first valid slow SMA appears at index 29
    # (positions 0..28 inclusive are NaN).  Those 29 leading positions
    # must all be FLAT because the strategy cannot have a view yet.
    strategy = SMACrossoverStrategy(fast_window=10, slow_window=30)
    signals = strategy.generate_signals(bars)

    # Vectorised assertion: every one of the first 29 positions equals
    # SIGNAL_FLAT.  np.all collapses the elementwise comparison to a
    # single bool so a single failed position fails the whole assert.
    assert np.all(signals[:29] == SIGNAL_FLAT)

    # The first emittable position (index 29) must be one of the three
    # legal signal values.  We don't pin which one — that depends on
    # the close pattern — only that the strategy emits a *valid* signal
    # the moment it's warmed up.
    assert signals[29] in (SIGNAL_LONG, SIGNAL_FLAT, SIGNAL_SHORT)


# ---------------------------------------------------------------------------
# Trend-direction behaviour
# ---------------------------------------------------------------------------


def test_sma_crossover_uptrend_goes_long():
    """In a strict monotonic uptrend, fast SMA > slow SMA after warmup → LONG."""

    # A linear ramp [1, 2, 3, ..., 50].  In this regime the most recent
    # `fast_window` prices average higher than the most recent
    # `slow_window` prices at every position past warmup, so the fast
    # SMA strictly exceeds the slow SMA and the strategy must say LONG.
    bars = make_bars([float(i + 1) for i in range(50)])

    # fast=5, slow=10 → first non-NaN slow SMA is at index 9, so
    # everything from index 10 onwards is fully warmed up.  Choosing
    # `signals[10:]` (not `signals[9:]`) gives a one-bar safety buffer
    # against off-by-one warmup edge cases.
    strategy = SMACrossoverStrategy(fast_window=5, slow_window=10)
    signals = strategy.generate_signals(bars)

    # Every post-warmup position must be LONG; if even one is FLAT or
    # SHORT, either the rule comparison flipped or the warmup mask is
    # wrong.  np.all gives us a single-bool assertion across the slice.
    assert np.all(signals[10:] == SIGNAL_LONG)


def test_sma_crossover_downtrend_goes_short():
    """In a strict monotonic downtrend, fast SMA < slow SMA after warmup → SHORT."""

    # The mirror image of the uptrend test: closes ramp from 50 down to
    # 1.  In a monotone downtrend the fast SMA lags below the slow SMA
    # throughout, so the rule must produce SHORT at every warmed-up bar.
    bars = make_bars([float(50 - i) for i in range(50)])

    # Same window choices as the uptrend test for symmetry.
    strategy = SMACrossoverStrategy(fast_window=5, slow_window=10)
    signals = strategy.generate_signals(bars)

    # Every post-warmup position must be SHORT.  This pairs with the
    # uptrend test to give us coverage of both branches of np.where.
    assert np.all(signals[10:] == SIGNAL_SHORT)


# ---------------------------------------------------------------------------
# Crossover detection (qualitative)
# ---------------------------------------------------------------------------


def test_sma_crossover_detects_crossover():
    """A flat-then-rising series must produce at least one LONG signal."""

    # 20 bars of flat price — fast and slow SMA converge to 100, so the
    # comparison is equality (or NaN during warmup) → all FLAT.
    flat_phase = [100.0] * 20

    # 30 bars ramping from 101 up to 130.  The fast SMA reacts faster to
    # the new prices than the slow SMA, so somewhere in this stretch
    # fast crosses above slow and the strategy emits SIGNAL_LONG.
    ramp_phase = [100.0 + i for i in range(1, 31)]

    # Concatenate to get a 50-bar series with a clear regime change.
    bars = make_bars(flat_phase + ramp_phase)

    # fast=5 reacts to the ramp within ~5 bars; slow=15 takes longer.
    # The crossover *must* happen somewhere during the ramp phase given
    # those time constants.
    strategy = SMACrossoverStrategy(fast_window=5, slow_window=15)
    signals = strategy.generate_signals(bars)

    # Qualitative assertion only — we don't pin the exact crossover bar
    # because it depends on the smoothing math, and pinning it would
    # make this test brittle to harmless implementation changes.  All
    # we require is that *some* LONG signal appears in the output.
    assert np.any(signals == SIGNAL_LONG)


# ---------------------------------------------------------------------------
# Empty / insufficient input
# ---------------------------------------------------------------------------


def test_sma_crossover_raises_on_empty_bars():
    """Empty bar list must raise ValueError, matching the indicators module contract."""

    # Default windows (50, 200) — the validation happens before any
    # window-specific math, so the exact values don't matter here.
    strategy = SMACrossoverStrategy(fast_window=50, slow_window=200)

    # Empty input is virtually always a programming error upstream
    # (e.g. a symbol with no data).  Failing loudly here surfaces it
    # immediately rather than returning a zero-length signal array
    # that would later trigger confusing IndexErrors.
    with pytest.raises(ValueError):
        strategy.generate_signals([])


def test_sma_crossover_raises_on_insufficient_bars():
    """len(bars) <= slow_window must raise — not enough history to produce any signal."""

    # 50 bars with slow_window=200 → the slow SMA is NaN at every
    # position, so no signal could ever fire.  Strategy rejects this
    # rather than returning an all-zero array that would silently mask
    # the configuration mistake.
    bars = make_bars([100.0 + i * 0.5 for i in range(50)])

    # Default Golden Cross parameters: 50 bars is not enough history
    # for a 200-day slow SMA to ever warm up.
    strategy = SMACrossoverStrategy(fast_window=50, slow_window=200)

    # ValueError is the chosen exception type — same as the indicators
    # module — so callers can write a single except clause that catches
    # every "bad input shape" failure across the data and signals stack.
    with pytest.raises(ValueError):
        strategy.generate_signals(bars)
