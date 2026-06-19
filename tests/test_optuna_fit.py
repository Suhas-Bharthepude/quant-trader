# tests/test_optuna_fit.py

"""
Unit tests for src/research/optuna_fit.py.

These tests use SMALL windows, ranges, and trial counts (make_price_bars(40),
fast_range=(3, 8), slow_range=(10, 20), n_trials=10) so the Optuna study runs
in well under a second while still exercising the real search path — there are
no mocks here; every test runs a genuine seeded study.

The fixtures mirror tests/test_walk_forward.py: make_price_bars(n) builds a
monotonically rising close series so every log return is well-defined, and the
seam end-to-end test reuses walk_forward_splits / walk_forward_validate exactly
as a caller would.

Run with:
    uv run pytest tests/test_optuna_fit.py -v
"""

# Sequential UTC-aware timestamps for the synthetic bars; timezone.utc matches
# the project-wide convention that every bar timestamp is tz-aware UTC.
from datetime import datetime, timedelta, timezone

# pytest.approx is used in the warm-only test to compare the recorded Sharpe
# against an independently recomputed value within floating-point tolerance.
import pytest

# Backtester + metrics let the maximize/warm-only tests reconstruct the exact
# warm-only Sharpe the objective computes, for an independent cross-check.
from src.backtest.engine import Backtester
from src.backtest import metrics

# OHLCVBar is the bar schema every helper and the function under test operate on.
from src.data.schema import OHLCVBar

# SMACrossoverStrategy is both the type the factory returns and the reference we
# assert constraints against (fast < slow, slow < len(train_bars)).
from src.strategies.sma_crossover import SMACrossoverStrategy

# The factory under test.
from src.research.optuna_fit import make_sma_optuna_fit_fn

# The seam end-to-end test drives the real validator the same way step 3's CLI
# will; WalkForwardResult is its frozen output type.
from src.research.walk_forward import (
    WalkForwardResult,
    walk_forward_splits,
    walk_forward_validate,
)


# ---------------------------------------------------------------------------
# Module-level helper — same construction as test_walk_forward.make_price_bars
# ---------------------------------------------------------------------------


def make_price_bars(n: int, start: float = 100.0) -> list[OHLCVBar]:
    """Build n OHLCVBar objects with strictly-positive, monotonically rising closes.

    Starting at `start` and incrementing by 1.0 per bar keeps every
    log(close[i]/close[i-1]) well-defined and positive — the same property the
    walk-forward tests rely on, so the Optuna objective always has a real,
    finite Sharpe to maximise rather than NaN from a degenerate price path.
    """
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        OHLCVBar(
            symbol="T",
            timestamp=base + timedelta(days=i),
            open=start + i,
            high=start + i,
            low=start + i,
            close=start + i,
            adj_close=start + i,
            volume=1,
            timeframe="1d",
            source="test",
        )
        for i in range(n)
    ]


def make_bars_from_closes(closes: list[float]) -> list[OHLCVBar]:
    """Build OHLCVBars from an explicit close series (open=high=low=adj_close=close).

    Used by the trend-reversal test, where the price path is hand-designed rather
    than a simple ramp, so close values must be supplied directly.  Flat OHLC
    means no intrabar spread artefacts perturb the SMA signals.
    """
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        OHLCVBar(
            symbol="T",
            timestamp=base + timedelta(days=i),
            open=c, high=c, low=c, close=c, adj_close=c,
            volume=1, timeframe="1d", source="test",
        )
        for i, c in enumerate(closes)
    ]


# Shared small search parameters — kept here so every test uses the identical
# fast/slow ranges and trial count, and a single edit retunes the whole suite.
_SMALL_KW = dict(n_trials=10, fast_range=(3, 8), slow_range=(10, 20))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reproducible_with_same_seed():
    """Two fit_fns with the same seed on the same window return identical (fast, slow).

    TPESampler(seed) makes the entire trial sequence deterministic, so two
    independent studies built with the same seed must explore the same points
    and converge on the same best params.  This is the property the step-3 CLI
    relies on for runs to be repeatable.
    """
    train_bars = make_price_bars(40)

    a = make_sma_optuna_fit_fn(seed=42, **_SMALL_KW)(train_bars)
    b = make_sma_optuna_fit_fn(seed=42, **_SMALL_KW)(train_bars)

    # Same seed → same search → same chosen windows, exactly.
    assert (a.fast_window, a.slow_window) == (b.fast_window, b.slow_window)


def test_constraints_hold_for_any_seed():
    """The returned strategy always satisfies fast < slow AND slow < len(train_bars).

    These are the two invariants the clamps exist to guarantee: the dynamic
    lower bound (fast + 1) forces fast < slow, and the -1 upper clamp forces
    slow < len(train_bars) so SMACrossoverStrategy.generate_signals never raises.
    Sweeping several seeds makes the check robust to which point the search lands
    on rather than passing by luck of a single seed.
    """
    train_bars = make_price_bars(40)

    for seed in (0, 1, 7, 42, 123):
        result = make_sma_optuna_fit_fn(seed=seed, **_SMALL_KW)(train_bars)
        assert result.fast_window < result.slow_window
        assert result.slow_window < len(train_bars)


def test_returns_usable_strategy():
    """The factory returns an SMACrossoverStrategy whose generate_signals runs clean.

    Beyond type, the real contract is that the returned strategy is immediately
    usable on the same window it was fit to — the clamps guarantee the windows
    are valid for this bar count, so generate_signals must not raise.
    """
    train_bars = make_price_bars(40)
    result = make_sma_optuna_fit_fn(seed=42, **_SMALL_KW)(train_bars)

    assert isinstance(result, SMACrossoverStrategy)

    # Must not raise — the whole clamp design exists to keep this call valid.
    signals = result.generate_signals(train_bars)
    assert len(signals) == len(train_bars)


def test_record_side_channel_accumulates_per_call():
    """Passing a record list appends one well-formed dict per fit_fn call, in order.

    record is the side-channel step 3 uses to pair each fold's in-sample Sharpe
    against its OOS Sharpe (the overfitting tax).  Calling the fit_fn twice must
    leave exactly two entries — one per call — each carrying fast/slow and a
    float in_sample_sharpe.
    """
    train_bars = make_price_bars(40)
    record: list[dict] = []
    fit_fn = make_sma_optuna_fit_fn(seed=42, record=record, **_SMALL_KW)

    # Two calls simulate two folds; the validator calls fit_fn once per fold.
    fit_fn(train_bars)
    fit_fn(train_bars)

    assert len(record) == 2
    for entry in record:
        assert set(entry.keys()) == {"fast", "slow", "in_sample_sharpe"}
        # in_sample_sharpe is study.best_value — a Python float.
        assert isinstance(entry["in_sample_sharpe"], float)


def test_plugs_into_walk_forward_seam():
    """The fit_fn drives walk_forward_validate end-to-end without error.

    This is the integration check: the factory's output satisfies the seam's
    Callable[[list[OHLCVBar]], Strategy] contract, so the validator can fit a
    fresh strategy per fold and still return a well-formed WalkForwardResult with
    the expected fold count.  The base strategy is ignored when fit_fn is given,
    so the default SMA is a fine placeholder.
    """
    bars = make_price_bars(40)
    # train=20, test=5 (default step) → folds at start 0,5,10,15 → 4 folds.
    splits = walk_forward_splits(bars, train_size=20, test_size=5)

    fit_fn = make_sma_optuna_fit_fn(seed=42, **_SMALL_KW)
    result = walk_forward_validate(splits, SMACrossoverStrategy(5, 10), fit_fn=fit_fn)

    assert isinstance(result, WalkForwardResult)
    assert result.n_folds == len(splits)
    assert len(result.per_fold) == len(splits)


def test_objective_maximizes_not_minimizes():
    """The recorded Sharpe beats a deliberately-poor param's Sharpe — the study maximizes.

    Every other test passes regardless of optimisation DIRECTION: on the monotonic
    ramp the in-sample spread between (fast, slow) choices is near-zero, so a study
    that minimized — or used a constant objective — would still satisfy them.  This
    test uses a trend REVERSAL (rise then fall) where different windows score
    measurably differently in-sample, then pins that the chosen params strictly beat
    the longest-lag corner (fast=8, slow=20) — the combo that tracks the reversal
    worst.  A direction flip to minimize, or a flat objective, would land on (or
    below) that poor corner and fail here.
    """
    # 25 bars rising +2, then 15 bars falling -3: a sharp reversal that rewards
    # faster windows (they turn before slow ones get whipsawed by the top).
    closes = [100.0 + 2 * i for i in range(25)]              # bars 0..24: uptrend
    closes += [closes[-1] - 3 * (j + 1) for j in range(15)]  # bars 25..39: downtrend
    bars = make_bars_from_closes(closes)

    # Longest-lag corner of the search space — the deliberately-poor baseline.
    poor_fast, poor_slow = 8, 20
    poor_signals = SMACrossoverStrategy(poor_fast, poor_slow).generate_signals(bars)
    poor_result = Backtester(annualization_factor=252).run(bars, poor_signals)
    # Warm-only, the same slice the objective scores, so the comparison is like-for-like.
    poor_warm_sharpe = metrics.sharpe_ratio(poor_result.returns[poor_slow:], 252)

    record: list[dict] = []
    # n_trials=25 is ample to cover the small fast∈[3,8] × slow grid.
    fit_fn = make_sma_optuna_fit_fn(
        n_trials=25, seed=42, fast_range=(3, 8), slow_range=(10, 20), record=record
    )
    fit_fn(bars)

    # The maximiser must find params strictly better than the worst corner.
    assert record[0]["in_sample_sharpe"] > poor_warm_sharpe


def test_recorded_sharpe_is_warm_only():
    """The recorded Sharpe equals the warm-only slice value, NOT the full-window value.

    The objective drops the first `slow` returns (the slow-SMA warm-up, forced
    FLAT/zero) before scoring.  On a small window those dropped bars are a large
    fraction of the series, so the warm-only Sharpe and the full-window Sharpe
    diverge materially.  This pins that the drop actually happens: a regression
    back to full-window scoring would make the recorded value equal the
    full-window Sharpe and fail the inequality below.
    """
    # 25 bars: small enough that dropping `slow` (10–20) bars is a meaningful
    # fraction, making warm-only and full-window Sharpe clearly distinct.
    train_bars = make_price_bars(25)

    record: list[dict] = []
    fit_fn = make_sma_optuna_fit_fn(seed=42, record=record, **_SMALL_KW)
    fit_fn(train_bars)

    fast = record[0]["fast"]
    slow = record[0]["slow"]

    # Independently reconstruct what the objective computed for the winning params.
    signals = SMACrossoverStrategy(fast, slow).generate_signals(train_bars)
    result = Backtester(annualization_factor=252).run(train_bars, signals)
    warm_only = metrics.sharpe_ratio(result.returns[slow:], 252)   # objective's metric
    full_window = result.sharpe_ratio                              # engine's full-window value

    # Recorded value IS the warm-only Sharpe...
    assert record[0]["in_sample_sharpe"] == pytest.approx(warm_only)
    # ...and is NOT the full-window Sharpe (proves the warm-up region was dropped).
    assert record[0]["in_sample_sharpe"] != pytest.approx(full_window)
