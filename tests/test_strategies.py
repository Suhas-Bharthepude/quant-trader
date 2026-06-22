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

# The second concrete strategy under test — long/flat monthly time-series
# momentum.  Imported alongside the SMA strategy so both live in one test file.
from src.strategies.time_series_momentum import TimeSeriesMomentumStrategy


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


# ===========================================================================
# TimeSeriesMomentumStrategy (TSMOM) — long/flat monthly time-series momentum
# ===========================================================================
#
# These tests reuse the same make_bars() helper as the SMA tests above, so the
# bars are calendar-daily OHLCVBars anchored at 2024-01-01 (only `close` is
# read by the strategy).  Month boundaries are derived from each bar's OWN
# timestamp via the small helpers below, so the tests never re-assume the
# anchor date or the day cadence — they ask the bars where the months fall.


# The lookback every TSMOM test uses.  Fixed at 12 (the canonical 12-month
# formation window) so warmup is exactly the first 12 month-end observations.
TSMOM_LOOKBACK = 12

# Number of daily bars in the "long" series.  ~1200 calendar days is roughly
# 39 calendar months — comfortably past the 12-month warmup with room for the
# post-warmup signal to flip, satisfying the "at least 36 months" requirement.
TSMOM_N_BARS = 1200


def _month_key(bar: OHLCVBar) -> tuple[int, int]:
    """Return (year, month) for a bar — the identity of its calendar month."""

    # A (year, month) tuple uniquely names a calendar month and compares
    # correctly across year boundaries (e.g. (2024, 12) != (2025, 1)).
    return (bar.timestamp.year, bar.timestamp.month)


def _is_month_end_bar(bars: list[OHLCVBar], i: int) -> bool:
    """True if bar i is the LAST bar of its calendar month (the strategy's month-end)."""

    # The final bar of the whole series always counts as a month-end, by the
    # same convention the strategy uses (no "next" bar to compare against).
    if i == len(bars) - 1:
        return True

    # Otherwise bar i is a month-end exactly when the NEXT bar falls in a
    # different calendar month — i.e. the month changes between i and i+1.
    return _month_key(bars[i]) != _month_key(bars[i + 1])


def _final_month_indices(bars: list[OHLCVBar]) -> list[int]:
    """Indices of every bar that shares the calendar month of the LAST bar."""

    # The last bar's month is the "final calendar month" all the trend tests
    # assert on (the fully-warmed-up region the signal has settled into).
    final_key = _month_key(bars[-1])

    # Collect every index whose bar belongs to that same final month.
    return [i for i, b in enumerate(bars) if _month_key(b) == final_key]


def _first_n_month_indices(bars: list[OHLCVBar], n: int) -> list[int]:
    """Indices of every bar belonging to the first `n` DISTINCT calendar months."""

    # Walk the bars in order and record each new calendar month the first time
    # we see it, preserving chronological order of the distinct months.
    distinct_months: list[tuple[int, int]] = []
    for b in bars:
        # _month_key(b) is this bar's month; append only months not seen yet.
        key = _month_key(b)
        if key not in distinct_months:
            distinct_months.append(key)

    # The warmup region is the first `n` distinct months; make a fast lookup set.
    warmup_months = set(distinct_months[:n])

    # Return every bar index whose month is one of those first-n warmup months.
    return [i for i, b in enumerate(bars) if _month_key(b) in warmup_months]


# ---------------------------------------------------------------------------
# Constructor validation and name property
# ---------------------------------------------------------------------------


def test_tsmom_validates_lookback():
    """lookback must be >= 1 — zero or negative values are rejected at construction."""

    # lookback == 0 has no meaning (no months to measure a return over); the
    # constructor must reject it before any generate_signals call is made.
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy(lookback=0)

    # A clearly negative value is even more obviously nonsense; the same guard
    # catches it.  Using -3 keeps the test intent obvious to a reader.
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy(lookback=-3)


def test_tsmom_name_property():
    """name is a human-readable string identifying the lookback configuration."""

    # Build with a non-default lookback so the test would fail if `name` ever
    # hard-coded the default 12 instead of reading self.lookback.
    strategy = TimeSeriesMomentumStrategy(lookback=6)

    # The exact format is part of the contract — backtest reports and logs
    # grep on this string — so a drift here must be a deliberate update.
    assert strategy.name == "TSMOM(6)"


# ---------------------------------------------------------------------------
# 1. Uptrend goes long after warmup
# ---------------------------------------------------------------------------


def test_tsmom_uptrend_goes_long():
    """Strictly increasing closes → every bar in the FINAL calendar month is LONG."""

    # A strictly increasing ramp: every month-end close exceeds the close 12
    # month-ends earlier, so the trailing 12-month return is always positive
    # and the warmed-up signal must be LONG.
    bars = make_bars([100.0 + i for i in range(TSMOM_N_BARS)])

    # Run the strategy at the fixed 12-month lookback.
    signals = TimeSeriesMomentumStrategy(lookback=TSMOM_LOOKBACK).generate_signals(bars)

    # The final calendar month is deep past the 12-month warmup, so in a pure
    # uptrend every bar in it must carry SIGNAL_LONG.  np.all collapses the
    # per-index check to a single bool so one wrong bar fails the assert.
    assert np.all(signals[_final_month_indices(bars)] == SIGNAL_LONG)


# ---------------------------------------------------------------------------
# 2. Downtrend stays flat (long/flat never shorts)
# ---------------------------------------------------------------------------


def test_tsmom_downtrend_stays_flat():
    """Strictly decreasing closes → every bar in the FINAL calendar month is FLAT."""

    # A strictly decreasing ramp that stays comfortably positive across all
    # 1200 bars (10000 - 1199*5 = ~4005).  Every trailing 12-month return is
    # negative, so the rule (strictly > 0) yields FLAT — never SHORT.
    bars = make_bars([10000.0 - i * 5 for i in range(TSMOM_N_BARS)])

    # Run the strategy at the fixed 12-month lookback.
    signals = TimeSeriesMomentumStrategy(lookback=TSMOM_LOOKBACK).generate_signals(bars)

    # In a pure downtrend the warmed-up final month must be entirely FLAT.
    # This is the long/flat guarantee: a negative trend produces no position,
    # not a short.
    assert np.all(signals[_final_month_indices(bars)] == SIGNAL_FLAT)


# ---------------------------------------------------------------------------
# 3. Flat prices map to flat (pins the strict-greater-than boundary)
# ---------------------------------------------------------------------------


def test_tsmom_flat_prices_are_flat():
    """Constant closes → the ENTIRE signal array is FLAT (trailing return == 0)."""

    # A perfectly constant price: every trailing return is EXACTLY 0.  Because
    # the rule is strictly `> 0`, a zero return maps to FLAT — this test pins
    # that boundary so a future `>=` typo would be caught immediately.
    bars = make_bars([100.0] * TSMOM_N_BARS)

    # Run the strategy at the fixed 12-month lookback.
    signals = TimeSeriesMomentumStrategy(lookback=TSMOM_LOOKBACK).generate_signals(bars)

    # Not one bar may be LONG — the whole array must be FLAT.  np.all over the
    # full array is the strongest possible statement of "no position anywhere".
    assert np.all(signals == SIGNAL_FLAT)


# ---------------------------------------------------------------------------
# 4. Warmup is flat (first 12 calendar months emit no position)
# ---------------------------------------------------------------------------


def test_tsmom_warmup_is_flat():
    """Strictly increasing closes → every bar in the FIRST 12 calendar months is FLAT."""

    # An uptrend, so the ONLY thing keeping early bars flat is the warmup, not
    # a negative trend.  This isolates warmup behaviour: the first 12 month-end
    # observations have no 12-months-earlier value, so their return is NaN.
    bars = make_bars([100.0 + i for i in range(TSMOM_N_BARS)])

    # Run the strategy at the fixed 12-month lookback.
    signals = TimeSeriesMomentumStrategy(lookback=TSMOM_LOOKBACK).generate_signals(bars)

    # Every bar in the first 12 distinct calendar months must be FLAT, even
    # though prices are rising — proving warmup emits FLAT (0), not a position.
    assert np.all(signals[_first_n_month_indices(bars, TSMOM_LOOKBACK)] == SIGNAL_FLAT)


# ---------------------------------------------------------------------------
# 5. Output contract (ndarray, int8, length matches bars)
# ---------------------------------------------------------------------------


def test_tsmom_output_contract():
    """Return value is a numpy int8 ndarray with length equal to the bar count."""

    # Any series works for a shape/dtype check; reuse the simple uptrend ramp.
    bars = make_bars([100.0 + i for i in range(TSMOM_N_BARS)])

    # Run the strategy at the fixed 12-month lookback.
    signals = TimeSeriesMomentumStrategy(lookback=TSMOM_LOOKBACK).generate_signals(bars)

    # The contract says the strategy returns a numpy ndarray (not a list or a
    # pandas Series) so the backtester can do vectorised math on it directly.
    assert isinstance(signals, np.ndarray)

    # int8 is part of the contract: values are only ever 0/1 here, so int8 is
    # ample and 8x cheaper than the numpy default int64 in long backtests.
    assert signals.dtype == np.int8

    # The alignment contract: exactly one signal per input bar, so downstream
    # index-based lookups (the backtester, the live trader) stay in lockstep.
    assert len(signals) == len(bars)


# ---------------------------------------------------------------------------
# 6. Long/flat only, never short
# ---------------------------------------------------------------------------


def test_tsmom_long_flat_only_never_short():
    """A rise-then-fall series exercises both 0 and 1, and never emits -1."""

    # Build a clear rise-then-fall path: prices climb for the first half, then
    # fall back for the second half.  The rising stretch (post-warmup) yields
    # LONG and the falling stretch eventually yields FLAT, so both 0 and 1 must
    # appear — while a short (-1) must never appear in a long/flat strategy.
    half = TSMOM_N_BARS // 2
    rise = [100.0 + i for i in range(half)]
    fall = [100.0 + half - 1 - j for j in range(1, TSMOM_N_BARS - half + 1)]
    bars = make_bars(rise + fall)

    # Run the strategy at the fixed 12-month lookback.
    signals = TimeSeriesMomentumStrategy(lookback=TSMOM_LOOKBACK).generate_signals(bars)

    # The set of distinct emitted values must be a subset of {FLAT, LONG}; any
    # other value (most importantly the short -1) fails this assertion.
    assert set(np.unique(signals)).issubset({SIGNAL_FLAT, SIGNAL_LONG})

    # Explicit, separate check that SIGNAL_SHORT never appears — this is the
    # whole reason SIGNAL_SHORT is not even imported into the strategy module.
    assert SIGNAL_SHORT not in set(np.unique(signals))

    # Sanity guard that the test actually exercised BOTH states; otherwise a
    # degenerate all-flat result could pass the subset check vacuously.
    assert SIGNAL_LONG in signals and SIGNAL_FLAT in signals


# ---------------------------------------------------------------------------
# 7. Monthly cadence (signal changes ONLY at month-end bars)
# ---------------------------------------------------------------------------


def test_tsmom_changes_only_at_month_end():
    """The signal may flip ONLY on month-end bars — never mid-month."""

    # Reuse the rise-then-fall path so the signal flips at least once; the test
    # is meaningless if the signal is constant for the whole series.
    half = TSMOM_N_BARS // 2
    rise = [100.0 + i for i in range(half)]
    fall = [100.0 + half - 1 - j for j in range(1, TSMOM_N_BARS - half + 1)]
    bars = make_bars(rise + fall)

    # Run the strategy at the fixed 12-month lookback.
    signals = TimeSeriesMomentumStrategy(lookback=TSMOM_LOOKBACK).generate_signals(bars)

    # Track whether we observed at least one change, so we can assert the test
    # actually tested the cadence rather than passing on a flat signal.
    saw_a_change = False

    # Walk every adjacent pair: a change at index i means a new monthly signal
    # took effect there, which is only legal on a month-end bar.
    for i in range(1, len(signals)):
        # Compare each bar to its predecessor; equal values are mid-month holds
        # (forward-fill) and need no check.
        if signals[i] != signals[i - 1]:
            # Record that a flip happened so the final guard below is meaningful.
            saw_a_change = True
            # The flip MUST land on a month-end bar; a mid-month change would
            # mean the monthly signal was not forward-filled correctly (or a
            # spurious internal lag crept in).
            assert _is_month_end_bar(bars, i), f"signal changed mid-month at index {i}"

    # Guarantee the rise-then-fall series really did flip at least once, so the
    # loop above was not vacuously satisfied by a constant signal.
    assert saw_a_change


# ---------------------------------------------------------------------------
# 8. No lookahead (changing the last bar cannot alter any earlier signal)
# ---------------------------------------------------------------------------


def test_tsmom_no_lookahead():
    """Mutating ONLY the final bar's close must not change any non-final-month signal."""

    # A rise-then-fall base series so the signal has real structure to disturb.
    half = TSMOM_N_BARS // 2
    rise = [100.0 + i for i in range(half)]
    fall = [100.0 + half - 1 - j for j in range(1, TSMOM_N_BARS - half + 1)]
    closes = rise + fall

    # Baseline signal S1 from the unmodified closes.
    strategy = TimeSeriesMomentumStrategy(lookback=TSMOM_LOOKBACK)
    bars1 = make_bars(closes)
    s1 = strategy.generate_signals(bars1)

    # Make an independent copy of the closes and change ONLY the final bar's
    # close (double it).  make_bars rebuilds identical timestamps, so the only
    # difference between bars1 and bars2 is the last bar's price.
    closes2 = closes.copy()
    closes2[-1] = closes2[-1] * 2
    bars2 = make_bars(closes2)
    s2 = strategy.generate_signals(bars2)

    # Every bar OUTSIDE the final calendar month must be byte-for-byte identical
    # between S1 and S2.  Changing the last bar can only affect the final
    # month's own trailing return; touching any earlier signal would prove the
    # strategy is peeking at future data — the cardinal backtest sin.
    final = set(_final_month_indices(bars1))
    non_final = [i for i in range(len(s1)) if i not in final]
    assert np.array_equal(s1[non_final], s2[non_final])


# ---------------------------------------------------------------------------
# 9. Too-short input raises (12 or fewer month-end observations)
# ---------------------------------------------------------------------------


def test_tsmom_raises_on_insufficient_months():
    """Fewer than 13 calendar months (<= 12 month-ends) cannot warm up → ValueError."""

    # 300 calendar days from 2024-01-01 spans only ~10 calendar months, giving
    # at most 10 month-end observations — fewer than the 12 the lookback needs,
    # so not even one trailing return is computable.
    bars = make_bars([100.0 + i for i in range(300)])

    # Sanity-check the construction: this series really does have <= 12 month
    # boundaries, so the strategy's guard (M <= lookback) is what trips.
    distinct = {_month_key(b) for b in bars}
    assert len(distinct) <= TSMOM_LOOKBACK

    # The strategy must reject this with ValueError rather than return an
    # all-NaN/all-flat array that would silently mask the too-short input.
    with pytest.raises(ValueError):
        TimeSeriesMomentumStrategy(lookback=TSMOM_LOOKBACK).generate_signals(bars)


# ---------------------------------------------------------------------------
# 10. Empty input raises
# ---------------------------------------------------------------------------


def test_tsmom_raises_on_empty_bars():
    """Empty bar list must raise ValueError, exactly as the SMA strategy does."""

    # Default lookback; the empty-input guard runs before any month math, so
    # the lookback value is irrelevant here.
    strategy = TimeSeriesMomentumStrategy()

    # Empty input is virtually always an upstream bug; failing loudly surfaces
    # it immediately instead of returning a zero-length array downstream.
    with pytest.raises(ValueError):
        strategy.generate_signals([])
