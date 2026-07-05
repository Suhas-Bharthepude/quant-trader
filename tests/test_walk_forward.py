# tests/test_walk_forward.py

"""
Unit tests for src/research/walk_forward.py.

All tests are pure: no DuckDB, no network, no filesystem — walk_forward_splits
is a pure list-slicing function.  make_bars(n) builds synthetic OHLCVBar
objects in-memory; each bar's close encodes its zero-based index so that
tests can verify WHICH bars landed in which window by reading the close value
rather than comparing timestamp or object identity.

Run with:
    uv run pytest tests/test_walk_forward.py -v
"""

# datetime + timedelta + timezone build sequential UTC-aware timestamps for
# the synthetic bars.  timezone.utc matches the project-wide convention that
# every bar timestamp is timezone-aware UTC.
from datetime import datetime, timedelta, timezone

# numpy is needed in the validator tests for exact per-element array equality
# checks on oos_returns and oos_equity_curve.
import numpy as np

# pytest is the test runner; pytest.raises is used to assert that the
# function's validation paths emit the right exceptions.
import pytest

# Pure metric functions are imported for cross-checking the validator's
# oos_win_rate and oos_sharpe aggregations against independent computations.
from src.backtest import metrics as _metrics

# Backtester is used in the warm-up linchpin test to compute the expected warm
# and cold results independently, for direct comparison against the validator.
from src.backtest.engine import Backtester

# OHLCVBar is the bar schema that all helpers and functions under test operate on.
from src.data.schema import OHLCVBar

# walk_forward_splits and walk_forward_validate are both under test;
# WalkForwardResult is the frozen output type exercised in the validator tests.
from src.research.walk_forward import (
    WalkForwardResult,
    walk_forward_splits,
    walk_forward_validate,
)

# SMACrossoverStrategy is the reference strategy for the warm-up linchpin test:
# slow_window=10 > test_size=5 means a cold run on test_bars alone (5 bars) yields
# all-flat signals; the validator's warm run over train+test produces SIGNAL_LONG.
from src.strategies.sma_crossover import SMACrossoverStrategy

# Strategy/SIGNAL_LONG power the always-long stub used by the buy-and-hold
# benchmark invariant test: a strategy that emits SIGNAL_LONG on every bar must
# reproduce the benchmark exactly, since the benchmark IS an always-long run.
from src.strategies.base import Strategy, SIGNAL_LONG


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def make_bars(n: int) -> list[OHLCVBar]:
    """Build n OHLCVBar objects with sequential daily UTC timestamps.

    Each bar at index i has close = float(i).  This encoding lets tests verify
    WHICH bars landed in which window by reading the close value rather than
    comparing object identity or timestamp arithmetic — bar 7 always has
    close=7.0, so fold slices can be spot-checked by inspecting close values.
    """

    # Anchor the series at 2024-01-01 UTC.  walk_forward_splits never reads
    # the calendar value — it only slices the list — but pinning the date
    # keeps the test output deterministic and easy to eyeball when a test fails.
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # enumerate gives both the day offset (for the timestamp) and the index
    # value (for close).  All price fields are set to float(i) so the bar is
    # internally consistent; volume=1 and source="test" are minimal placeholders
    # that the slicing function never reads.
    return [
        OHLCVBar(
            symbol="T",
            timestamp=base + timedelta(days=i),  # strictly increasing UTC, one per day
            open=float(i),
            high=float(i),
            low=float(i),
            close=float(i),       # encodes index — the key for fold-window verification
            adj_close=float(i),
            volume=1,
            timeframe="1d",
            source="test",
        )
        for i in range(n)
    ]


def make_price_bars(n: int, start: float = 100.0) -> list[OHLCVBar]:
    """Build n OHLCVBar objects with strictly-positive prices for engine tests.

    Unlike make_bars, which sets close=float(i) (close[0]=0.0, undefined log
    return), this helper starts at `start` (default 100.0) and increments by
    1.0 per bar.  Every log(close[i]/close[i-1]) is well-defined and positive.

    A monotonically increasing series has one key property for SMA crossover
    tests: after the slow-window warmup, fast_SMA > slow_SMA at every bar
    (faster window weights more-recent, higher closes), so the strategy emits
    SIGNAL_LONG at every warm bar — deterministic and hand-verifiable.
    """
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        OHLCVBar(
            symbol="T",
            timestamp=base + timedelta(days=i),
            open=start + i,
            high=start + i,
            low=start + i,
            close=start + i,    # strictly positive; log returns defined at every step
            adj_close=start + i,
            volume=1,
            timeframe="1d",
            source="test",
        )
        for i in range(n)
    ]


class _AlwaysLong(Strategy):
    """Test stub strategy: emits SIGNAL_LONG on every bar (no warmup, no flat).

    The buy-and-hold benchmark IS an always-long run over each test window, so an
    always-long *strategy* must produce a stitched OOS identical to the benchmark.
    This stub exists to pin that invariant: if the strategy and benchmark paths
    ever used different windows or stitching, the equality would break.
    """

    @property
    def name(self) -> str:
        return "always-long"

    def generate_signals(self, bars: list[OHLCVBar]) -> np.ndarray:
        # One SIGNAL_LONG per bar; integer dtype to satisfy the engine's contract.
        return np.full(len(bars), SIGNAL_LONG, dtype=np.int64)


# ---------------------------------------------------------------------------
# Tests — fold count and window sizes
# ---------------------------------------------------------------------------


def test_returns_correct_fold_count_clean_division():
    """1000 bars with train=504, test=126 (default step) produces exactly 3 folds."""

    # The valid start offsets are 0, 126, 252.  At start=378 the fold would
    # need bars[378 : 378+504+126] = bars[378:1008], which overshoots the
    # 1000-bar list, so it is dropped.  Three folds, not four.
    bars = make_bars(1000)
    folds = walk_forward_splits(bars, train_size=504, test_size=126)

    assert len(folds) == 3


def test_fold_window_sizes_are_exact():
    """Every fold from the 1000-bar fixture has exactly 504 train bars and 126 test bars."""

    # Using the same parameters as the fold-count test lets us reuse the same
    # mental model of the fixture without a separate setup comment.  The two
    # tests are deliberately split so that a size regression and a count
    # regression produce different failing test names — easier to diagnose.
    bars = make_bars(1000)
    folds = walk_forward_splits(bars, train_size=504, test_size=126)

    for train_bars, test_bars in folds:
        assert len(train_bars) == 504
        assert len(test_bars) == 126


# ---------------------------------------------------------------------------
# Tests — adjacency, overlap, and step semantics
# ---------------------------------------------------------------------------


def test_train_and_test_are_strictly_adjacent():
    """The first test bar immediately follows the last train bar with no gap and no overlap."""

    # 30 bars, train=10, test=5, default step=5 → 4 folds (start ∈ {0,5,10,15};
    # start=20 needs 35 bars).  Encoding close=float(index) lets us verify the
    # exact boundary: train[-1].close + 1 must equal test[0].close, because
    # consecutive bars differ by exactly 1.0 in their close value.
    bars = make_bars(30)
    folds = walk_forward_splits(bars, train_size=10, test_size=5)

    # Loop over every fold so a regression in any fold fails, not just fold 0.
    for train_bars, test_bars in folds:
        # train[-1] is the last training bar; test[0] is the first test bar.
        # Their closes must differ by exactly 1.0 — they are consecutive bars.
        # A difference > 1.0 would mean there's a gap (bars skipped between
        # windows); a difference < 1.0 or equality would mean overlap (the same
        # bar appears in both windows, leaking training data into evaluation).
        assert test_bars[0].close == train_bars[-1].close + 1.0


def test_test_windows_do_not_overlap():
    """Default step tiles test windows without any bar appearing in two test sets."""

    # step defaults to test_size=5.  With 30 bars and train=10, the four test
    # windows cover indices 10-14, 15-19, 20-24, 25-29 — a clean partition
    # of the post-training bars with no bar appearing twice.
    bars = make_bars(30)
    folds = walk_forward_splits(bars, train_size=10, test_size=5)

    # Flatten all test-window close values into one list, then compare the
    # total count against the unique count.  If any bar appears in two test
    # windows its close value will appear twice and the set will be smaller.
    all_test_closes = [bar.close for _, test_bars in folds for bar in test_bars]
    assert len(all_test_closes) == len(set(all_test_closes))


def test_step_advances_by_step_not_test_size():
    """Fold 1's first training bar is exactly `step` positions ahead of fold 0's."""

    # With step=5 and train=10, fold 0 trains on bars[0:10] (first bar close=0.0)
    # and fold 1 trains on bars[5:15] (first bar close=5.0).  The difference
    # must be exactly step=5.  This separates "step controls window advance"
    # from "default step equals test_size" — they happen to be the same value
    # here (step=5, test_size=5), but the assertion targets the advance property.
    bars = make_bars(30)
    folds = walk_forward_splits(bars, train_size=10, test_size=5, step=5)

    fold0_train, _ = folds[0]
    fold1_train, _ = folds[1]

    # close encodes the bar's position in the original list, so the difference
    # in first-bar closes equals the number of positions the window advanced.
    assert fold1_train[0].close == fold0_train[0].close + 5.0


def test_custom_step_larger_than_test_leaves_gap():
    """step > test_size is legal; gaps between test windows are allowed by the caller."""

    # step=10 with test_size=5 means bars between consecutive test windows are
    # never tested (the "gap" bars).  This is an advanced caller choice — the
    # function must not reject it.  The point of this test is that no exception
    # is raised and at least one complete fold is produced.
    bars = make_bars(40)
    folds = walk_forward_splits(bars, train_size=10, test_size=5, step=10)

    # At least one fold must have been produced.
    assert len(folds) >= 1

    # Every fold that was produced must have full-size windows — no short folds.
    for train_bars, test_bars in folds:
        assert len(train_bars) == 10
        assert len(test_bars) == 5


# ---------------------------------------------------------------------------
# Tests — validation errors
# ---------------------------------------------------------------------------


def test_raises_on_train_size_zero():
    """train_size=0 is rejected before any slicing begins."""

    # A zero-bar training window cannot produce signals — there are no bars to
    # fit on.  This must be caught at the boundary, not inside the loop.
    with pytest.raises(ValueError):
        walk_forward_splits(make_bars(30), train_size=0, test_size=5)


def test_raises_on_test_size_zero():
    """test_size=0 is rejected before any slicing begins."""

    # A zero-bar test window cannot evaluate a strategy — there are no bars to
    # measure out-of-sample performance on.  Caught at the boundary.
    with pytest.raises(ValueError):
        walk_forward_splits(make_bars(30), train_size=10, test_size=0)


def test_raises_on_step_zero_when_provided():
    """step=0 is rejected when provided explicitly — it would cause an infinite loop."""

    # step=0 makes the loop advance by zero each iteration, so start never grows
    # past 0 and the loop runs forever.  Validation must catch this before the
    # loop is entered.  step=None (the default) is still valid — only explicit
    # non-positive values are rejected.
    with pytest.raises(ValueError):
        walk_forward_splits(make_bars(30), train_size=10, test_size=5, step=0)


def test_raises_when_bars_too_short_for_one_fold():
    """Fewer bars than train_size + test_size raises rather than returning an empty list."""

    # 10 bars, train=10, test=5: needs 15, has 10.  Silently returning [] would
    # mask a misconfiguration — the caller might interpret the empty list as
    # "ran OK, just no folds" rather than "structurally impossible inputs".
    # The error message must state the required and actual bar counts so the
    # caller can diagnose the shortfall without re-deriving it from parameters.
    with pytest.raises(ValueError):
        walk_forward_splits(make_bars(10), train_size=10, test_size=5)


# ---------------------------------------------------------------------------
# Tests — boundary (exact-fit) case
# ---------------------------------------------------------------------------


def test_exact_minimum_bars_yields_one_fold():
    """train_size + test_size bars exactly produces one fold — the <= condition is inclusive."""

    # 15 bars = 10 (train) + 5 (test).  The single fold uses all bars: fold 0
    # has start=0 and start+train+test=15 == len(bars)=15, satisfying `<=`.
    # Using `<` instead of `<=` in the loop condition would drop this fold,
    # leaving the caller with an empty list even though the data fits perfectly.
    # This test is the direct regression guard for that off-by-one.
    bars = make_bars(15)
    folds = walk_forward_splits(bars, train_size=10, test_size=5)

    assert len(folds) == 1

    # Also verify the single fold covers the correct bars — first train bar is
    # index 0 (close=0.0) and last test bar is index 14 (close=14.0).
    train_bars, test_bars = folds[0]
    assert train_bars[0].close == 0.0
    assert test_bars[-1].close == 14.0


# ===========================================================================
# walk_forward_validate and WalkForwardResult
# ===========================================================================
#
# Standard parameters used throughout this section:
#   SMACrossoverStrategy(5, 10) — slow_window=10 > test_size=5, so running
#   the strategy cold on just the 5-bar test window (< slow_window) yields
#   all-flat signals; the validator's warm run over train+test (25 bars) yields
#   SIGNAL_LONG throughout the test window (monotonic uptrend ensures fast > slow).
#
#   make_price_bars(30): closes=[100..129].  With train=20, test=5, step=5
#   (default), walk_forward_splits yields exactly 2 folds:
#     fold 0: train=[bar 0..19] (prices 100..119), test=[bar 20..24] (prices 120..124)
#     fold 1: train=[bar 5..24] (prices 105..124), test=[bar 25..29] (prices 125..129)
# ===========================================================================


def test_wfv_fold_count():
    """len(per_fold) and n_folds both equal the number of splits passed in."""
    strategy = SMACrossoverStrategy(5, 10)
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)
    result = walk_forward_validate(splits, strategy)

    # Both the list field and the scalar counter must agree with each other
    # and with the number of splits.  Testing both catches a regression where
    # one is updated but the other is not.
    assert len(result.per_fold) == len(splits)
    assert result.n_folds == len(splits)
    assert result.n_folds == 2
    assert isinstance(result, WalkForwardResult)  # result type is frozen dataclass


def test_wfv_warmup_is_linchpin():
    """The validator uses full_bars (train+test) to warm indicators; cold run is impossible.

    SMACrossoverStrategy.generate_signals raises ValueError when len(bars) <= slow_window.
    With slow_window=10 > test_size=5, calling the strategy directly on the 5-bar
    test window is not merely wrong — it is structurally impossible.  The validator's
    approach of passing full_bars (20 train + 5 test = 25 > slow_window) is therefore
    not just a performance improvement; it is required for correctness.  The warm run
    produces SIGNAL_LONG throughout the test window, yielding a positive total return.
    """
    strategy = SMACrossoverStrategy(5, 10)
    bars = make_price_bars(30)
    splits = walk_forward_splits(bars, train_size=20, test_size=5)
    result = walk_forward_validate(splits, strategy)

    train_bars, test_bars = splits[0]  # first fold for independent cross-check

    # Replicate exactly what the validator does internally for fold 0.
    # full_bars = train + test; generate signals on all 25 bars; take the test slice.
    full_bars = train_bars + test_bars
    full_signals = strategy.generate_signals(full_bars)
    test_signals = full_signals[len(train_bars):]       # signals for bars 20..24
    expected_warm = Backtester().run(test_bars, test_signals, strategy_name="warm")

    # Attempting a cold run — just the 5 test bars, without the training window —
    # raises ValueError because 5 <= slow_window=10.  This is stronger than "cold =
    # all-flat": the training window is not optional, it is structurally required.
    with pytest.raises(ValueError, match="slow_window"):
        strategy.generate_signals(test_bars)

    # The validator's fold 0 must match the warm independent computation exactly.
    assert result.per_fold[0].total_return_pct == pytest.approx(expected_warm.total_return_pct)
    assert result.per_fold[0].sharpe_ratio == pytest.approx(expected_warm.sharpe_ratio)

    # The warm result is profitable (monotonic uptrend + SIGNAL_LONG → positive returns).
    assert result.per_fold[0].total_return_pct > 0.0


def test_wfv_oos_returns_stitching():
    """oos_returns == concat(fold.returns[1:] for each fold), equity curve anchored at 1.0."""
    strategy = SMACrossoverStrategy(5, 10)
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)
    result = walk_forward_validate(splits, strategy)

    # oos_returns must be the concatenation of each fold's returns with the
    # structural index-0 zero removed.  That zero is fold-local (no signal on
    # bar 0 of the backtester run); keeping it would inject spurious zeros at
    # fold seams, depressing mean return and inflating std → wrong aggregate Sharpe.
    expected_oos_returns = np.concatenate([fr.returns[1:] for fr in result.per_fold])
    assert result.oos_returns == pytest.approx(expected_oos_returns)

    # Equity curve is anchored at 1.0: exp(cumsum([0.0] + oos_returns))[0] = exp(0) = 1.0.
    # This makes the curve a direct growth multiplier (1.05 = +5%), consistent with
    # each per-fold equity curve which is also anchored at initial_capital=1.0.
    assert result.oos_equity_curve[0] == pytest.approx(1.0)

    # Every subsequent equity value must equal exp(cumulative sum from 0).
    expected_equity = np.exp(np.cumsum(np.concatenate([[0.0], result.oos_returns])))
    assert result.oos_equity_curve == pytest.approx(expected_equity)


def test_wfv_chained_returns_two_folds():
    """oos_total_return equals the compound product of per-fold terminal multipliers minus 1.

    Log returns are time-additive: sum(fold0.returns[1:]) + sum(fold1.returns[1:])
    = log(fold0_terminal) + log(fold1_terminal) = log(fold0_terminal * fold1_terminal).
    So oos_equity_curve[-1] = fold0_terminal * fold1_terminal = (1+r0)*(1+r1), and
    oos_total_return = (1+r0)*(1+r1) - 1.

    Hand-verifiable values:
      fold 0 (prices 120→124): r0 = 124/120 - 1 = 4/120  (exact rational)
      fold 1 (prices 125→129): r1 = 129/125 - 1 = 4/125  (exact rational)
      chained = (124/120)*(129/125) - 1 = 15996/15000 - 1 ≈ 0.0664
    """
    strategy = SMACrossoverStrategy(5, 10)
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)
    result = walk_forward_validate(splits, strategy)

    assert result.n_folds == 2  # precondition: this test requires exactly 2 folds

    r0 = result.per_fold[0].total_return_pct  # fold 0: prices 120→124
    r1 = result.per_fold[1].total_return_pct  # fold 1: prices 125→129

    # Chained compound return: multiply terminal equity multipliers and subtract 1.
    expected_chained = (1 + r0) * (1 + r1) - 1
    assert result.oos_total_return == pytest.approx(expected_chained)


def test_wfv_oos_sharpe_uses_full_returns():
    """oos_sharpe is sharpe_ratio(oos_returns, 252) — no element-0 skip applied.

    oos_returns has no structural zeros (each fold's index-0 zero was stripped
    before concatenation).  So oos_returns[0] is a real return; dropping it
    would change both mean and std, producing a materially different Sharpe
    (~15% higher in this fixture because the 8-element mean is lower than the
    7-element mean and the std is slightly different).  The two values differ by
    ~95 Sharpe units — well outside any numerical tolerance.
    """
    strategy = SMACrossoverStrategy(5, 10)
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)
    result = walk_forward_validate(splits, strategy)

    # Precondition: oos_returns[0] is a real return (log(121/120) ≈ 0.0083), not zero.
    # If this were zero, the two Sharpe values below could coincide and the
    # confirmatory assertion would be vacuous.
    assert result.oos_returns[0] != pytest.approx(0.0)

    # Correct: Sharpe over the full 8-element oos_returns (no skip).
    expected_full = _metrics.sharpe_ratio(result.oos_returns, 252)
    assert result.oos_sharpe == pytest.approx(expected_full)

    # Confirmatory: Sharpe with element 0 removed would be a different (wrong) value.
    wrong_if_skipped = _metrics.sharpe_ratio(result.oos_returns[1:], 252)
    assert result.oos_sharpe != pytest.approx(wrong_if_skipped)


def test_wfv_trade_aggregation():
    """total_trades, oos_win_rate, and n_folds_positive_sharpe are correctly aggregated."""
    strategy = SMACrossoverStrategy(5, 10)
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)
    result = walk_forward_validate(splits, strategy)

    # total_trades must equal the sum of individual fold trade counts.
    expected_total = sum(len(fr.trades) for fr in result.per_fold)
    assert result.total_trades == expected_total

    # oos_win_rate must equal win_rate computed over all aggregated trades combined.
    all_trades = [t for fr in result.per_fold for t in fr.trades]
    assert result.oos_win_rate == pytest.approx(_metrics.win_rate(all_trades))

    # n_folds_positive_sharpe must equal the count of folds where sharpe > 0.
    expected_pos = sum(1 for fr in result.per_fold if fr.sharpe_ratio > 0)
    assert result.n_folds_positive_sharpe == expected_pos

    # With a monotonic uptrend and warm SIGNAL_LONG signals, every fold is
    # profitable → all folds have positive Sharpe.
    assert result.n_folds_positive_sharpe == result.n_folds


def test_wfv_overlap_guard_raises():
    """walk_forward_validate raises ValueError when test windows overlap in time.

    The OOS aggregate requires strictly disjoint test windows so each bar
    contributes exactly once to oos_returns.  Overlapping windows (step < test_size)
    cause some bars to be counted twice, corrupting the aggregate equity curve
    and Sharpe.  The guard fires before any folding, at entry.
    """
    bars = make_price_bars(30)
    # step=3 < test_size=5 → fold 0 tests bars[20..24], fold 1 tests bars[23..27].
    # Bars 23 and 24 appear in both test windows → timestamp overlap detected.
    overlapping_splits = walk_forward_splits(bars, train_size=20, test_size=5, step=3)

    with pytest.raises(ValueError, match="overlap"):
        walk_forward_validate(overlapping_splits, SMACrossoverStrategy(5, 10))


def test_wfv_empty_splits_raises():
    """walk_forward_validate raises ValueError with a diagnostic message when splits is empty.

    An explicit guard fires at the TOP of the function, before any fold loop or
    numpy aggregation, so the error names the cause ('splits is empty') rather
    than surfacing the cryptic downstream message 'need at least one array to
    concatenate'.  The match= assertion verifies the message text, not just the
    exception type — consistent with how test_wfv_overlap_guard_raises verifies
    the overlap message.
    """
    with pytest.raises(ValueError, match="splits is empty"):
        walk_forward_validate([], SMACrossoverStrategy(5, 10))


def test_wfv_mixed_fold_signs():
    """n_folds_positive_sharpe reads actual fold-level Sharpe signs, not a constant.

    Every other validator test uses a monotonically rising price series, so every
    fold is profitable and n_folds_positive_sharpe always equals n_folds.  A bug
    that returned n_folds unconditionally would pass all of those tests.  This
    test constructs a two-fold scenario with exactly one positive and one negative
    fold, verifying the counter is non-trivially computed.

    Price path (hand-verified against SMA(5, 10) signals):

      Bars  0..19: 119, 118, ..., 100 — downtrend through the training window.
                   At bar 20, fast_SMA(101) < slow_SMA(103.5) → SIGNAL_SHORT.
      Bars 20..24: 99, 98, 97, 96, 95 — fold 0 test, continuing decline.
                   Strategy holds SHORT throughout → all per-bar returns positive.
                   Fold 0 outcome: positive total_return and positive Sharpe.
      Bars 25..29: 100, 105, 110, 115, 120 — fold 1 test, sharp V-recovery.
                   Strategy opens SHORT (inertia from training downtrend), then
                   crossover flips to LONG at fold 1 bar 22 (fast 101.2 > slow 100.1).
                   Net: two SHORT losses, two LONG gains, small net loss (≈ -0.83%).
                   Fold 1 outcome: negative total_return and negative Sharpe.
    """
    # Inline price bars: open=high=low=adj_close=close so every log return is
    # well-defined, no OHLC spread artefacts affect the strategy signals.
    closes = (
        [119 - i for i in range(20)]  # bars  0..19: monotone downtrend 119→100
        + [99, 98, 97, 96, 95]        # bars 20..24: fold 0 test, decline continues
        + [100, 105, 110, 115, 120]   # bars 25..29: fold 1 test, V-recovery
    )
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = [
        OHLCVBar(
            symbol="T",
            timestamp=base + timedelta(days=i),
            open=c, high=c, low=c, close=c, adj_close=c,
            volume=1, timeframe="1d", source="test",
        )
        for i, c in enumerate(closes)
    ]

    strategy = SMACrossoverStrategy(5, 10)
    splits = walk_forward_splits(bars, train_size=20, test_size=5)
    result = walk_forward_validate(splits, strategy)

    assert result.n_folds == 2  # precondition: exactly the two designed folds

    # Core assertion: the counter is strictly between 0 and n_folds, proving it
    # reflects real fold-level Sharpe signs and is not a constant.
    assert 0 < result.n_folds_positive_sharpe < result.n_folds

    # Independent recomputation from per_fold — the stored value must agree.
    expected_pos = sum(1 for fr in result.per_fold if fr.sharpe_ratio > 0)
    assert result.n_folds_positive_sharpe == expected_pos

    # At least one fold must have a genuinely negative total return (a real
    # out-of-sample loss, not just a low-Sharpe positive return).
    assert any(fr.total_return_pct < 0 for fr in result.per_fold)


def test_wfv_fixed_strategy_seam():
    """Each fold's strategy_name encodes the strategy name and the fold index.

    The seam line 'fold_strategy = strategy' reuses the same strategy object for
    every fold.  The fold label format is '<strategy.name> fold N', tying the
    result to the exact fold index for human-readable comparison tables.
    Day 22 replaces this line with 'fold_strategy = fit_fn(train_bars)'; the
    name format will then carry whatever name the fitted strategy returns.
    """
    strategy = SMACrossoverStrategy(5, 10)  # strategy.name == "SMA(5, 10)"
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)
    result = walk_forward_validate(splits, strategy)

    for i, fr in enumerate(result.per_fold):
        # strategy.name must appear in the fold's strategy_name so results
        # can be traced back to the originating strategy after aggregation.
        assert strategy.name in fr.strategy_name
        # Fold index must appear so individual folds are distinguishable in
        # a multi-fold comparison report.
        assert f"fold {i}" in fr.strategy_name


# ---------------------------------------------------------------------------
# Tests — buy-and-hold benchmark
# ---------------------------------------------------------------------------


def test_wfv_buy_and_hold_invariant_always_long_equals_benchmark():
    """An always-long strategy's stitched OOS return AND sharpe equal the benchmark.

    This is the strongest alignment check in the suite. The benchmark is itself an
    always-long run over each test window, so a strategy that is also always-long
    must reproduce it exactly — same windows, same warm-up exclusion, same seam-zero
    stripping, same metrics. If the strategy path and the benchmark path were
    measured over even slightly different windows, these equalities would fail and
    Δ would be non-zero.
    """
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)
    result = walk_forward_validate(splits, _AlwaysLong())

    # Strategy OOS and benchmark must match on every comparable axis.
    assert result.oos_total_return == pytest.approx(result.bh_return)
    assert result.oos_sharpe == pytest.approx(result.bh_sharpe)
    assert result.oos_max_drawdown == pytest.approx(result.bh_max_drawdown)

    # Therefore the deltas the CLI prints are ~0 on both axes.
    assert result.oos_total_return - result.bh_return == pytest.approx(0.0)
    assert result.oos_sharpe - result.bh_sharpe == pytest.approx(0.0)


def test_wfv_buy_and_hold_known_monotonic_series():
    """On a monotonic-up series, B&H return is positive and equals the compounded
    asset return over the stitched test windows.

    make_price_bars(30) → closes 100..129. With train=20/test=5 (default step)
    there are exactly 2 folds: fold 0 test closes 120→124, fold 1 test closes
    125→129. Each fold contributes log(last_test / first_test) to the stitched log
    return — the seam-zero strip drops only the first test bar's structural zero,
    not any real return — so the benchmark equals (124/120)*(129/125) - 1.

    The benchmark is always-long regardless of the strategy passed, so any strategy
    yields the same bh_* values; SMACrossoverStrategy is used here only to exercise
    the realistic code path.
    """
    bars = make_price_bars(30)
    splits = walk_forward_splits(bars, train_size=20, test_size=5)
    result = walk_forward_validate(splits, SMACrossoverStrategy(5, 10))

    assert result.n_folds == 2  # precondition: exactly the two designed folds

    # Independent expected value: compound each test window's first→last close ratio.
    # This is the per-window buy-and-hold return chained across the disjoint folds.
    expected_bh = 1.0
    for _, test_bars in splits:
        expected_bh *= test_bars[-1].close / test_bars[0].close
    expected_bh -= 1.0

    assert result.bh_return > 0.0  # monotonic uptrend → positive buy-and-hold
    assert result.bh_return == pytest.approx(expected_bh)


def test_wfv_sortino_fields_populated_both_paths():
    """oos_sortino and bh_sortino are populated as finite floats end-to-end.

    This does NOT re-test the Sortino math (that lives in test_metrics.py) — it
    only pins that the validator threads the new scalar through _stitch_oos on
    BOTH the strategy path (oos_sortino) and the benchmark path (bh_sortino), so
    a construction that omitted either field would fail here rather than silently.
    """
    strategy = SMACrossoverStrategy(5, 10)
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)
    result = walk_forward_validate(splits, strategy)

    # Both fields must exist and be plain floats (frozen dataclass required fields).
    assert isinstance(result.oos_sortino, float)
    assert isinstance(result.bh_sortino, float)

    # On a monotonic-up run neither is nan — the guard only returns nan-free 0.0,
    # and a real run with genuine downside yields a finite ratio. np.isfinite
    # rejects both nan and inf, catching any mis-annualisation blow-up.
    assert np.isfinite(result.oos_sortino)
    assert np.isfinite(result.bh_sortino)


# ---------------------------------------------------------------------------
# Tests — fit_fn seam (per-fold strategy fitting)
# ---------------------------------------------------------------------------


def test_wfv_fit_fn_sees_only_train_window_each_fold():
    """fit_fn is called once per fold, and the i-th call's bars are splits[i]'s TRAIN bars.

    This is the no-lookahead linchpin for the fitting seam: a fitted strategy may
    only ever see training data.  We record every argument fit_fn receives and
    assert (a) it was called exactly n_folds times, and (b) the i-th argument is
    splits[i][0] — the train bars, NOT the test bars and NOT the full train+test
    window.  If fitting were ever handed the test slice (or the concatenated
    full_bars), the test window would leak into parameter selection and the OOS
    measurement would no longer be out-of-sample.
    """
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)
    assert len(splits) == 2  # precondition: exactly the two designed folds

    # Record the exact list handed to each fit_fn call.  The returned strategy is
    # a fixed instance — this test pins WHAT fit_fn sees, not what it returns.
    received: list[list[OHLCVBar]] = []
    fixed = _AlwaysLong()

    def fit_fn(train_bars: list[OHLCVBar]) -> Strategy:
        received.append(train_bars)
        return fixed

    # The base `strategy` arg must be ignored when fit_fn is given; pass an SMA to
    # prove the seam routes through fit_fn rather than the positional strategy.
    walk_forward_validate(splits, SMACrossoverStrategy(5, 10), fit_fn=fit_fn)

    # Called exactly once per fold — no extra calls, none skipped.
    assert len(received) == len(splits)

    # The i-th call must receive precisely fold i's TRAIN bars.  Comparing the
    # full bar lists (OHLCVBar is a frozen dataclass with value equality) catches
    # any off-by-one or wrong-slice regression: test bars or full_bars would fail.
    for i, train_bars in enumerate(received):
        assert train_bars == splits[i][0]


def test_wfv_fit_fn_returned_strategy_is_the_one_used():
    """The strategy fit_fn returns drives the OOS result — proven via the B&H invariant.

    A fit_fn that returns _AlwaysLong() for every fold must make the strategy path
    identical to the buy-and-hold benchmark (which is itself an always-long run over
    each test window).  We pass SMACrossoverStrategy as the base `strategy` to prove
    the base is ignored: if the seam used the base instead of fit_fn's return value,
    the SMA path would diverge from the always-long benchmark and these equalities
    would fail.  Reuses the always-long-equals-benchmark invariant.
    """
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)
    result = walk_forward_validate(
        splits,
        SMACrossoverStrategy(5, 10),     # base strategy must be ignored
        fit_fn=lambda train_bars: _AlwaysLong(),
    )

    # always-long strategy == always-long benchmark on every comparable axis.
    assert result.oos_total_return == pytest.approx(result.bh_return)
    assert result.oos_sharpe == pytest.approx(result.bh_sharpe)


def test_wfv_fit_fn_none_matches_omitted():
    """Passing fit_fn=None is identical to omitting it (backward-compat default path).

    The existing 132 tests already exercise the default-None code path exhaustively;
    one explicit equality is enough here to pin that the new keyword's default does
    not perturb the established behaviour.
    """
    strategy = SMACrossoverStrategy(5, 10)
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)

    result_omitted = walk_forward_validate(splits, strategy)
    result_none = walk_forward_validate(splits, strategy, fit_fn=None)

    assert result_omitted.oos_total_return == pytest.approx(result_none.oos_total_return)


# ---------------------------------------------------------------------------
# Tests — price-basis consistency guard
# ---------------------------------------------------------------------------
#
# The validator raises if the engine's price basis (Backtester.price_field) and
# the running strategy's price basis (strategy.price_field) disagree, so a caller
# cannot silently mix price-return signals with total-return scoring. The guard
# fires BEFORE generate_signals, so the mismatch tests need only enough bars for
# walk_forward_splits to emit one fold — they do NOT depend on a strategy
# producing valid signals (the all-FLAT dummy below never even reaches scoring on
# the mismatch path).
# ---------------------------------------------------------------------------


class _DummyBasisStrategy(Strategy):
    """Test stub exposing a settable price_field, emitting all-FLAT signals.

    Isolates the guard's behaviour from any real strategy's data requirements:
    generate_signals always returns zeros (SIGNAL_FLAT), so a basis-MATCHED run
    is a clean, valid backtest, while a basis-MISMATCHED run is rejected by the
    guard before generate_signals is ever called.
    """

    def __init__(self, price_field: str) -> None:
        # Store the basis the guard reads via getattr(fold_strategy, "price_field").
        self.price_field = price_field

    @property
    def name(self) -> str:
        # Constant name — the guard tests never compare on the label.
        return "dummy-basis"

    def generate_signals(self, bars: list[OHLCVBar]) -> np.ndarray:
        # All-FLAT, integer dtype, length-aligned to bars — a clean run on the
        # matched path; never reached on the mismatched path (guard fires first).
        return np.zeros(len(bars), dtype=np.int8)


def test_basis_mismatch_raises():
    """fit_fn=None path: engine adj_close vs strategy close raises on fold 0."""
    # Engine scores on total return (adj_close); strategy's basis is price (close).
    engine_adj = Backtester(price_field="adj_close")
    # Dummy strategy carrying the MISMATCHED basis ("close").
    dummy_close = _DummyBasisStrategy(price_field="close")
    # Minimal one-fold splits: 15 bars = train 10 + test 5 (the exact-fit case).
    splits = walk_forward_splits(make_price_bars(15), train_size=10, test_size=5)

    # The guard fires on fold 0 before any signal math, so the bar values are
    # irrelevant — only that one fold exists. Mismatch → ValueError.
    with pytest.raises(ValueError):
        walk_forward_validate(splits, dummy_close, backtester=engine_adj)


def test_basis_match_does_not_raise():
    """Engine adj_close and strategy adj_close (matched) run cleanly to a result."""
    # Engine and strategy share the SAME basis ("adj_close") → guard must NOT fire.
    engine_adj = Backtester(price_field="adj_close")
    dummy_adj = _DummyBasisStrategy(price_field="adj_close")
    splits = walk_forward_splits(make_price_bars(15), train_size=10, test_size=5)

    # The all-FLAT dummy produces a clean run; the guard agreeing on basis lets
    # the validator complete and return a WalkForwardResult.
    result = walk_forward_validate(splits, dummy_adj, backtester=engine_adj)
    assert isinstance(result, WalkForwardResult)


def test_strategy_without_price_field_is_skipped():
    """A strategy without price_field (SMA) is skipped by the getattr-None branch."""
    # Engine on adj_close; SMACrossoverStrategy has NO price_field attribute, so
    # getattr(..., None) returns None and the guard skips it regardless of basis.
    engine_adj = Backtester(price_field="adj_close")
    # Small windows + enough bars (30) for at least one fold with slow_window=10.
    strategy = SMACrossoverStrategy(5, 10)
    splits = walk_forward_splits(make_price_bars(30), train_size=20, test_size=5)

    # Must run to completion and return a result — the guard never raises for a
    # strategy that lacks the price_field knob.
    result = walk_forward_validate(splits, strategy, backtester=engine_adj)
    assert isinstance(result, WalkForwardResult)


def test_fit_fn_path_basis_mismatch_raises():
    """fit_fn path: the guard checks fold_strategy (fit_fn's return), not the base."""
    # Engine on adj_close; fit_fn returns a dummy carrying the MISMATCHED "close".
    engine_adj = Backtester(price_field="adj_close")
    splits = walk_forward_splits(make_price_bars(15), train_size=10, test_size=5)

    # fit_fn supplies the per-fold strategy; the positional `strategy` below is
    # ignored when fit_fn is given, so the SMA placeholder proves the guard reads
    # fold_strategy (the dummy with "close"), not the ignored passed-in strategy.
    def fit_fn(train_bars: list[OHLCVBar]) -> Strategy:
        return _DummyBasisStrategy(price_field="close")

    with pytest.raises(ValueError):
        walk_forward_validate(
            splits,
            SMACrossoverStrategy(5, 10),  # ignored placeholder — fit_fn drives the fold
            backtester=engine_adj,
            fit_fn=fit_fn,
        )
