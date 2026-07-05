# src/research/walk_forward.py

"""
Walk-forward validation helpers.

Walk-forward validation avoids the overfitting trap of fitting a strategy once
on the entire dataset and reporting in-sample metrics as if they were
out-of-sample.  Instead, the timeline is divided into a succession of
(train, test) window pairs so that every test measurement is genuinely
out-of-sample — the strategy sees training bars, then is evaluated on bars it
has never seen.

This module exposes one pure function, walk_forward_splits, which performs the
window slicing.  It has no I/O, no side effects, and no dependencies on the
data layer.  Separation of concerns: the caller is responsible for fetching bars
(DuckDBStore), executing the strategy (BacktestRunner), and aggregating results.
This function's only job is to partition a pre-fetched bar list correctly.
"""

# Callable types the fit_fn seam in walk_forward_validate: a per-fold factory
# that takes that fold's train bars and returns a fitted Strategy.  Imported
# from collections.abc (not typing) per the project's modern-typing convention.
from collections.abc import Callable

# OHLCVBar is the canonical in-memory representation of a single price bar,
# defined in src/data/schema.py.  Importing it here keeps the function's type
# annotations honest and lets callers import the type from one place rather
# than knowing where OHLCVBar lives separately.
from src.data.schema import OHLCVBar

# numpy is needed to concatenate per-fold return arrays and compute the
# stitched OOS equity curve via cumsum + exp.
import numpy as np

# dataclass auto-generates __init__, __repr__, and __eq__ for WalkForwardResult.
# frozen=True makes the result immutable after construction — consistent with
# BacktestResult and Trade in src/backtest/result.py.
from dataclasses import dataclass

# Strategy is the abstract base that walk_forward_validate accepts so the
# validator is not coupled to any specific concrete strategy implementation.
# SIGNAL_LONG is imported for the buy-and-hold benchmark: holding it on every
# test bar makes the engine reproduce the asset's own per-bar returns over that
# window, so the benchmark is computed by the exact same path as the strategy.
from src.strategies.base import Strategy, SIGNAL_LONG

# Backtester is the engine used to score each test fold.  Imported here so
# walk_forward_validate can create a default instance when the caller passes None.
from src.backtest.engine import Backtester

# BacktestResult is the per-fold output type; Trade is used in the type
# annotation for the stitched all_trades list before passing to metrics.win_rate.
from src.backtest.result import BacktestResult, Trade

# Pure metric functions compute the stitched OOS scalar aggregates.  Imported
# as a module (not individual names) so call sites read metrics.sharpe_ratio(...)
# — self-documenting and consistent with how engine.py uses them after extraction.
from src.backtest import metrics


def walk_forward_splits(
    bars: list[OHLCVBar],
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[tuple[list[OHLCVBar], list[OHLCVBar]]]:
    """Slice a time-ordered bar list into successive (train, test) window pairs.

    Each fold is a (train_bars, test_bars) tuple where test_bars immediately
    follows train_bars in time with no gap and no overlap.  The rolling /
    sliding-window construction means consecutive folds share training bars
    (the training window slides forward by `step` bars each fold rather than
    being rebuilt from scratch).

    Assumption: `bars` is sorted ascending by timestamp.  This is guaranteed
    upstream by DuckDBStore.read_bars().  This function does NOT re-sort —
    walk-forward on shuffled bars is meaningless, and re-sorting here would
    mask an upstream bug by silently accepting wrong input.

    Args:
        bars:       Time-ordered list of OHLCVBar objects, ascending timestamp.
        train_size: Number of bars in the training window.  Must be >= 1.
        test_size:  Number of bars in the test window.  Must be >= 1.
        step:       How many bars to advance the window between folds.
                    Defaults to test_size when None (non-overlapping test
                    windows that tile the post-training timeline exactly once).
                    Must be >= 1 when provided.

    Returns:
        A list of (train_bars, test_bars) tuples, always non-empty (validation
        check #4 guarantees at least one complete fold fits in bars).

    Raises:
        ValueError: if train_size < 1, test_size < 1, step < 1 (when given),
                    or len(bars) < train_size + test_size.
    """

    # ------------------------------------------------------------------
    # 1. Validate inputs.  All checks run before any slicing so that a
    #    misconfigured call fails atomically — no partial progress followed
    #    by a silent empty list that masks the real problem.
    # ------------------------------------------------------------------

    # train_size < 1 is nonsensical: a zero-bar or negative-bar training
    # window cannot produce any signals.  Naming the actual value in the
    # message avoids the caller having to add a print statement to debug
    # "why did this raise?".
    if train_size < 1:
        raise ValueError(f"train_size must be >= 1, got {train_size}")

    # test_size < 1 is equally nonsensical: you cannot evaluate a strategy
    # on an empty window.  Same naming convention as the train_size guard.
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1, got {test_size}")

    # step is optional (None means "use test_size"), but if the caller
    # provides it explicitly, a value < 1 is an error: step=0 would cause
    # an infinite loop; a negative step would make the window walk backwards.
    # This check runs only when step is not None — None is valid (default).
    if step is not None and step < 1:
        raise ValueError(f"step must be >= 1 when provided, got {step}")

    # Not enough bars for even one complete fold.  This is the structurally-
    # impossible case: no value of start could ever yield a full train+test
    # window, so no fold can be emitted.  Returning [] silently would be a
    # masked failure — the caller might interpret an empty list as "ran OK,
    # just no folds happened to fit" rather than "the inputs are incompatible".
    # Analogous to run_universe's all-skipped guard in runner.py: never return
    # [] silently when the inputs are structurally impossible.  The message
    # states both how many bars are required and how many were given so the
    # caller can calculate the shortfall without re-deriving it from parameters.
    if len(bars) < train_size + test_size:
        raise ValueError(
            f"bars must contain at least train_size + test_size = "
            f"{train_size + test_size} bars; got {len(bars)}"
        )

    # ------------------------------------------------------------------
    # 2. Resolve the effective step.
    # ------------------------------------------------------------------

    # Why step defaults to test_size:
    #   When effective_step == test_size, consecutive test windows tile the
    #   post-training timeline with no overlap and no gap — every bar after
    #   the first train_size bars is included in exactly one test window.
    #   step < test_size causes adjacent test windows to overlap (a bar tested
    #   twice corrupts aggregated out-of-sample metrics, because the same bar
    #   contributes to multiple folds' averages); step > test_size leaves gaps
    #   (bars that are never included in any test window, silently excluded from
    #   evaluation).  The default is therefore the only value that gives a clean
    #   partition, making it the safe choice; callers may override with full
    #   awareness of the trade-off (e.g. step < test_size for an intentional
    #   overlapping evaluation pattern).
    effective_step = step if step is not None else test_size

    # ------------------------------------------------------------------
    # 3. Generate folds by sliding the window forward by effective_step.
    # ------------------------------------------------------------------

    # Pre-allocate the results list.  We append inside the loop (rather than
    # using a list comprehension) because the loop condition is a while-loop
    # over a mutable integer index, which does not translate cleanly into a
    # range-based comprehension without computing the fold count in advance —
    # a floor-division expression that is less readable than an explicit loop.
    results: list[tuple[list[OHLCVBar], list[OHLCVBar]]] = []

    # Fold index `start` is the offset of the first training bar for this
    # fold.  It increments by effective_step each iteration so consecutive
    # training windows slide forward uniformly.
    start = 0

    # Why the loop condition uses `<=` rather than `<`:
    #   start + train_size + test_size is the exclusive end index of the test
    #   window.  When this value equals len(bars), the test window ends exactly
    #   on the last bar — a complete, valid fold.  Using `<` instead would drop
    #   that final fold (off-by-one, discarding a legitimate fold).  Using a
    #   looser bound (e.g. start + train_size <= len(bars)) would emit a short
    #   final fold whose test window extends past the end of bars, producing
    #   either an IndexError or a silently-truncated test window.  `<=` is the
    #   exact tail-handling rule: keep emitting folds as long as a COMPLETE
    #   fold fits; stop the instant the next fold would reach past the end.
    while start + train_size + test_size <= len(bars):

        # Slice the training window.  Python slice semantics: bars[a:b]
        # returns bars at indices a, a+1, ..., b-1.  The training window
        # occupies [start, start + train_size).
        train_bars = bars[start : start + train_size]

        # Why test starts at exactly start + train_size (not +1, not -1):
        #   test_bars must begin on the bar IMMEDIATELY after the last training
        #   bar.  A one-bar overlap (test starting at start + train_size - 1)
        #   would leak the final training bar into the test set — that is
        #   lookahead bias, the exact failure walk-forward validation exists to
        #   prevent.  A one-bar gap (test starting at start + train_size + 1)
        #   would silently skip a bar from evaluation, hiding a portion of the
        #   timeline from the out-of-sample measurement.  Strict adjacency —
        #   no gap, no overlap — is the core correctness property of walk-
        #   forward validation; this index is where correctness lives.
        test_bars = bars[start + train_size : start + train_size + test_size]

        # Why leftover tail bars are dropped rather than emitted as a short fold:
        #   Out-of-sample metrics (Sharpe ratio, win rate, etc.) are only
        #   comparable across folds if every test window is the same length.
        #   A final 30-bar test fold sitting next to several 126-bar folds
        #   would distort any average computed over folds — the short fold's
        #   metrics are estimated with less precision and a different risk
        #   exposure than the full-length folds, making a mean Sharpe across
        #   folds misleading.  Dropping the tail keeps every fold size-identical,
        #   ensuring that fold-aggregated statistics are meaningful.  The while
        #   condition above enforces this: the loop body only executes when a
        #   COMPLETE fold fits, so every (train_bars, test_bars) appended here
        #   is exactly (train_size, test_size) bars long without exception.
        results.append((train_bars, test_bars))

        # Advance the window by effective_step.  This slides both the training
        # and test windows forward uniformly, setting the next fold's starting bar.
        start += effective_step

    # Validation check #4 above guarantees len(bars) >= train_size + test_size,
    # which means the while condition was true on the first iteration (start=0),
    # so results is always non-empty here.  No empty-list guard is needed.
    return results


# ---------------------------------------------------------------------------
# WalkForwardResult — aggregate output of walk_forward_validate.
# ---------------------------------------------------------------------------

# frozen=True makes the result immutable after construction — backtest results
# are historical facts; mutating them is a bug.  Consistent with BacktestResult
# and Trade in src/backtest/result.py.
@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregate result of a walk-forward validation run.

    per_fold holds one BacktestResult per (train, test) fold so callers can
    inspect fold-level metrics alongside the stitched OOS aggregates.  All
    oos_* fields treat the concatenated test windows as one contiguous timeline.
    """

    # One BacktestResult per fold in chronological order.  Each result covers
    # exactly the test window; the strategy was warm over the training bars that
    # preceded it, so signals at the start of the test window are valid.
    per_fold: list[BacktestResult]

    # Stitched per-bar OOS log returns.  The structural index-0 zero from each
    # fold's returns array has been removed so every element is a real return
    # observation.  Length == sum(len(fr.returns) - 1 for fr in per_fold).
    oos_returns: np.ndarray

    # Stitched OOS equity curve anchored at 1.0.  Length == len(oos_returns) + 1
    # (the +1 is the 1.0 anchor point prepended before cumsum).  A value of 1.25
    # means the strategy grew 25% over the full out-of-sample timeline.
    oos_equity_curve: np.ndarray

    # Scalar aggregates computed over the stitched OOS timeline.
    oos_total_return: float       # (oos_equity_curve[-1] / oos_equity_curve[0]) - 1
    oos_sharpe: float             # annualised Sharpe over oos_returns (no structural zeros)
    oos_sortino: float            # annualised Sortino over oos_returns (downside-only denom)
    oos_max_drawdown: float       # worst peak-to-trough in oos_equity_curve, positive fraction
    oos_win_rate: float           # fraction of all OOS trades with return_pct > 0

    # Buy-and-hold benchmark over the SAME stitched OOS test windows. Computed by
    # running the identical per-fold backtest + stitch path with an always-long
    # position (no signal), so these are directly comparable to the oos_* fields
    # above — the only difference is the position series. This lets callers read
    # edge vs beta: a strategy that merely tracks the asset shows oos_* ≈ bh_*,
    # while a strategy adding real edge shows oos_* above the benchmark.
    bh_return: float              # stitched B&H total return over the OOS test windows
    bh_sharpe: float              # annualised B&H Sharpe over the same stitched returns
    bh_sortino: float             # annualised B&H Sortino over the same stitched returns
    bh_max_drawdown: float        # worst peak-to-trough of the B&H stitched equity curve

    # Fold-level summary counts.
    n_folds: int                  # total number of (train, test) folds run
    n_folds_positive_sharpe: int  # folds where fold_result.sharpe_ratio > 0
    total_trades: int             # total completed trades across all folds


# ---------------------------------------------------------------------------
# _stitch_oos — fold-stitching + return-based metrics, shared by the strategy
# path and the buy-and-hold benchmark.
# ---------------------------------------------------------------------------

def _stitch_oos(
    per_fold: list[BacktestResult],
    annualization_factor: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Stitch per-fold test results into one OOS timeline and score it.

    Returns (oos_returns, oos_equity_curve, total_return, sharpe, max_drawdown,
    sortino).

    Factored out so the strategy path and the buy-and-hold benchmark are stitched
    and scored by byte-identical code — the ONLY thing that differs between them
    is the per-fold position series fed to the engine upstream. Duplicating this
    logic would make the apples-to-apples guarantee rest on two copies never
    drifting; a shared function makes it structural instead.
    """
    # Drop returns[0] from every fold before concatenating.
    # Why: the engine always sets strategy_returns[0] = 0.0 because bar 0 of
    # any run has no preceding signal to act on — it is a structural zero, not
    # a real return observation.  That zero is correct inside a single backtest
    # but becomes spurious noise at every fold seam when stitching: it would
    # depress mean return and inflate std, corrupting the stitched Sharpe ratio.
    # Stripping [0] from each fold leaves only genuine OOS return observations.
    oos_returns = np.concatenate([fr.returns[1:] for fr in per_fold])

    # Prepend 0.0 before cumsum so the equity curve starts at exp(0.0) = 1.0.
    # Why 1.0: each per-fold equity curve also starts at 1.0 (Backtester uses
    # initial_capital=1.0 by default); anchoring the stitched curve the same
    # way makes total_return a comparable multiplier from a neutral base.
    # Log returns are additive — log(A/B) + log(B/C) = log(A/C) — so cumsum
    # over oos_returns gives the total log return up to each bar, and exp
    # recovers the multiplicative equity growth factor at each point in time.
    oos_equity_curve = np.exp(np.cumsum(np.concatenate([[0.0], oos_returns])))

    # total_return: pass oos_equity_curve[0] as initial_capital so the metric
    # measures growth from the curve's own starting value.  With the 0.0 anchor
    # above, oos_equity_curve[0] == 1.0 always; we pass it explicitly rather
    # than hardcoding 1.0 to respect metrics.total_return's contract that the
    # caller supplies the reference capital.
    oos_total_return = float(metrics.total_return(oos_equity_curve, oos_equity_curve[0]))

    # sharpe: oos_returns contains no structural zeros — they were stripped
    # above.  Pass the full array with NO [1:] slice; every element is a real
    # OOS return observation that must enter the mean/std computation.  This is
    # the "does not skip element 0" contract documented in metrics.sharpe_ratio.
    oos_sharpe = metrics.sharpe_ratio(oos_returns, annualization_factor)

    # sortino: computed from the SAME stitched oos_returns as the Sharpe directly
    # above, with the SAME positional annualization_factor, so the two risk-adjusted
    # numbers share units.  Downside deviation penalises only below-target (loss)
    # volatility, so this is the upside-swing-neutral companion to oos_sharpe.
    oos_sortino = metrics.sortino_ratio(oos_returns, annualization_factor)

    # max_drawdown: computed over the stitched equity curve so peak-to-trough
    # declines that span multiple fold boundaries are captured correctly.
    oos_max_drawdown = metrics.max_drawdown(oos_equity_curve)

    # Append oos_sortino as the LAST tuple element so existing positional unpacks
    # only need the new trailing name added; element order is:
    # (returns, equity_curve, total_return, sharpe, max_drawdown, sortino).
    return (
        oos_returns,
        oos_equity_curve,
        oos_total_return,
        oos_sharpe,
        oos_max_drawdown,
        oos_sortino,
    )


# ---------------------------------------------------------------------------
# walk_forward_validate — pure orchestration function.
# ---------------------------------------------------------------------------

def walk_forward_validate(
    splits: list[tuple[list[OHLCVBar], list[OHLCVBar]]],
    strategy: Strategy,
    backtester: Backtester | None = None,
    annualization_factor: int = 252,
    fit_fn: Callable[[list[OHLCVBar]], Strategy] | None = None,
) -> WalkForwardResult:
    """Run a strategy over every test fold and return stitched OOS results.

    Pure function: no I/O, no printing, no side effects — consistent with
    BacktestRunner.run_many and the rest of the research layer.

    The caller is responsible for producing splits via walk_forward_splits and
    for loading bars.  This function's only job is fold-level orchestration and
    OOS aggregation — separation of concerns mirrors the splitter's own design.

    Args:
        splits:               Output of walk_forward_splits — list of
                              (train_bars, test_bars) tuples in chronological
                              order.  Test windows must be disjoint (default
                              step=None satisfies this; explicit step < test_size
                              does not and will raise).
        strategy:             Any concrete Strategy instance.  generate_signals
                              is called over train+test bars per fold so that
                              indicators are warm before the test window begins.
                              When fit_fn is provided, `strategy` is ignored —
                              fit_fn supplies the per-fold strategy instead.
        backtester:           Backtester instance for scoring each fold.  When
                              None, defaults to Backtester(annualization_factor=
                              annualization_factor) so fold-level and stitched
                              Sharpe values use the same scaling factor.
        annualization_factor: Bars per year for Sharpe annualisation.  Defaults
                              to 252 (US trading days), matching Backtester.
        fit_fn:               Optional per-fold strategy factory.  When None
                              (default), the passed `strategy` is used unchanged
                              on every fold — its training window serves only as
                              an indicator warm-up region (current behaviour).
                              When provided, fit_fn is called once per fold with
                              that fold's TRAIN bars and must return a Strategy
                              fitted to that window; the returned strategy is then
                              warmed over train+test and scored on the test window
                              exactly as the default path is.  fit_fn sees TRAIN
                              bars only and signals are generated causally, so
                              fitting introduces no lookahead into the test window.

    Returns:
        WalkForwardResult with per-fold BacktestResults, stitched OOS metrics,
        and a buy-and-hold benchmark (bh_return/bh_sharpe/bh_max_drawdown)
        measured over the identical stitched test windows for edge-vs-beta reads.

    Raises:
        ValueError: splits is empty — no folds to evaluate.
        ValueError: consecutive test windows overlap — stitched OOS aggregate
                    requires disjoint test windows (use step >= test_size).
    """

    # ------------------------------------------------------------------
    # 1. Validate inputs.
    # ------------------------------------------------------------------

    # Explicit guard before any loop: fires early with a named cause so the caller
    # sees "splits is empty" rather than a cryptic downstream numpy error
    # ("need at least one array to concatenate").
    # walk_forward_splits already raises when bars are too short for one fold,
    # so an empty list here means the caller either bypassed the splitter or
    # constructed splits by hand with an insufficient bar window.
    if not splits:
        raise ValueError(
            "splits is empty — no folds to validate. "
            "Pass the output of walk_forward_splits(); check that "
            "train_size + test_size does not exceed the available bar count."
        )

    # Overlap guard: the stitched OOS aggregate (concatenated returns, cumulative
    # equity curve) is only valid when each test bar appears in exactly one fold.
    # If step < test_size was passed to walk_forward_splits, consecutive test
    # windows share bars; stitching them counts shared bars twice, corrupting
    # mean return, std, and therefore Sharpe.
    # We compare timestamps rather than bar indices because this function receives
    # the pre-sliced lists, not the original index offsets.
    for i in range(len(splits) - 1):
        # Unpack only the test lists; training bars are not needed for this check.
        _, test_bars_i    = splits[i]
        _, test_bars_next = splits[i + 1]
        # Bars are ascending by timestamp (guaranteed by DuckDBStore.read_bars
        # and walk_forward_splits, which does not re-sort).  If fold i's last
        # test bar is at the same time as or after fold i+1's first test bar,
        # at least one bar is shared between the two test windows.
        if test_bars_i[-1].timestamp >= test_bars_next[0].timestamp:
            raise ValueError(
                f"Test windows for folds {i} and {i + 1} overlap in time: "
                f"fold {i} ends at {test_bars_i[-1].timestamp}, "
                f"fold {i + 1} starts at {test_bars_next[0].timestamp}. "
                "The stitched OOS aggregate requires disjoint test windows. "
                "Use step >= test_size when calling walk_forward_splits "
                "(the default step=None already satisfies this)."
            )

    # ------------------------------------------------------------------
    # 2. Resolve the backtester.
    # ------------------------------------------------------------------

    # Default Backtester uses annualization_factor from this call so the
    # fold-level sharpe_ratio values inside each BacktestResult use the same
    # annual scaling as the stitched oos_sharpe computed in step 6.
    # initial_capital stays at Backtester's own default (1.0) so per-fold
    # equity curves are unit-normalised multipliers — consistent with the
    # stitched equity curve anchored at 1.0 below.
    if backtester is None:
        backtester = Backtester(annualization_factor=annualization_factor)

    # ------------------------------------------------------------------
    # 3. Per-fold: generate warm signals, backtest on test window only.
    # ------------------------------------------------------------------

    # Accumulates one BacktestResult per fold in chronological (input) order.
    per_fold: list[BacktestResult] = []

    # Parallel accumulator for the buy-and-hold benchmark — one BacktestResult
    # per fold, scored on the identical test_bars but with an always-long
    # position instead of the strategy's signals (see the benchmark block below).
    bh_per_fold: list[BacktestResult] = []

    for i, (train_bars, test_bars) in enumerate(splits):

        # ------------------------------------------------------------------
        # SEAM — per-fold strategy selection.  This is now LIVE (fit_fn), not
        # a future TODO.
        #
        # fit_fn is None:  fold_strategy = strategy
        #   The strategy is used unchanged across every fold.  Training bars
        #   serve only as an indicator warm-up window (a 200-bar SMA needs
        #   ≥200 preceding bars before it emits a non-flat signal).
        #
        # fit_fn provided: fold_strategy = fit_fn(train_bars)
        #   fit_fn is called with THIS fold's train bars only and returns a
        #   Strategy fitted to that window.  fitting sees train bars exclusively;
        #   everything below — full_bars concat, generate_signals, slice,
        #   backtester.run — is byte-identical to the default path, so the
        #   fitted strategy is warmed over train+test yet scored only on the
        #   test window.  Signals are generated causally (signals[i] uses only
        #   bars[0..i]), so there is NO lookahead: fitting cannot see test bars
        #   and the train prefix cannot leak forward into the test suffix.
        # ------------------------------------------------------------------
        fold_strategy = strategy if fit_fn is None else fit_fn(train_bars)

        # ------------------------------------------------------------------
        # BASIS-CONSISTENCY GUARD — engine price basis must match the running
        # strategy's price basis, or the run silently mixes price-return signals
        # with total-return scoring (or vice versa).
        # ------------------------------------------------------------------
        # Read the strategy's price basis if it has one.  getattr(..., None) is the
        # hasattr-safe check: strategies WITHOUT a price_field (e.g.
        # SMACrossoverStrategy) return None here and are intentionally SKIPPED by
        # the guard below, never crashed — the None branch of the condition.
        strat_pf = getattr(fold_strategy, "price_field", None)

        # We check fold_strategy (NOT the passed-in `strategy`) on purpose: this
        # covers BOTH paths — the fit_fn=None path where fold_strategy IS strategy,
        # AND the fit_fn path where fold_strategy is the per-fold fitted strategy.
        # A check on `strategy` alone would miss the fit_fn path entirely, because
        # there the passed-in `strategy` is ignored and the real basis lives on the
        # object fit_fn returned.
        #
        # This sits BEFORE the full_bars concat / generate_signals call below so we
        # fail fast on a misconfigured basis, before any signal math runs.
        #
        # strat_pf is None  → strategy has no basis knob → skip (do not raise).
        # strat_pf set but != backtester.price_field → the bases disagree → raise.
        if strat_pf is not None and strat_pf != backtester.price_field:
            # Name the fold index, the strategy's price_field, and the engine's
            # price_field so the caller can see exactly which side to fix; the
            # message states the bases disagree and that BOTH must be passed the
            # same basis (the engine's price_field and the strategy's price_field).
            raise ValueError(
                f"Price-basis mismatch on fold {i}: strategy price_field "
                f"{strat_pf!r} does not match the engine's price_field "
                f"{backtester.price_field!r}. The engine and strategy price bases "
                "disagree — mixing price return with total return. Pass the SAME "
                "basis to both the Backtester and the strategy."
            )

        # Concatenate train + test into one contiguous list so generate_signals
        # sees the full history needed to warm the indicator.  Without training
        # bars, a slow-window strategy (e.g. SMA-200) would emit SIGNAL_FLAT
        # for its entire warmup period inside the test window, producing
        # misleadingly short signal coverage and distorted metrics.
        # Python list concat; copies references, not OHLCVBar objects.
        full_bars = train_bars + test_bars

        # Exact call pattern from BacktestRunner.run_many (runner.py line 140):
        #   signals = strategy.generate_signals(bars)
        # generate_signals returns an integer ndarray aligned 1:1 with full_bars.
        # Causal indicators ensure signals[i] uses only bars[0..i], so the
        # training prefix cannot introduce lookahead into the test suffix.
        full_signals = fold_strategy.generate_signals(full_bars)

        # Slice off the training prefix.  Indices [0, len(train_bars)) are the
        # warm-up region; indices [len(train_bars), len(full_bars)) are the test
        # signals aligned 1:1 with test_bars.  numpy slice is a view — no copy —
        # and preserves integer dtype so the engine's dtype check passes.
        test_signals = full_signals[len(train_bars):]

        # Run the backtester over the test window only.  test_bars and
        # test_signals are both len == len(test_bars), satisfying the engine's
        # alignment contract.  The fold label embeds the index so reports and
        # plots can identify which fold each BacktestResult came from.
        fold_result = backtester.run(
            test_bars,
            test_signals,
            strategy_name=f"{fold_strategy.name} fold {i}",
        )

        # Append in fold order; enumerate processes splits in input sequence so
        # per_fold is chronological — required for correct OOS stitching below.
        per_fold.append(fold_result)

        # ------------------------------------------------------------------
        # Buy-and-hold benchmark for THIS fold — same window, only the
        # position series differs.
        # ------------------------------------------------------------------
        # SIGNAL_LONG on every test bar means "fully invested, no signal". Run it
        # through the SAME backtester.run on the SAME test_bars, so the engine math
        # and the resulting per-bar returns are byte-identical to the strategy path
        # except for the constant position — which is exactly what makes the OOS-vs-
        # B&H comparison apples-to-apples: same windows, same warm-up exclusion
        # (test_bars only, so each fold's first-bar seam return is dropped for both),
        # same seam-zero stripping at stitch time. The engine validates dtype 'i'
        # and values in {-1,0,1}; np.full with SIGNAL_LONG satisfies both.
        bh_signals = np.full(len(test_bars), SIGNAL_LONG, dtype=np.int64)
        bh_result = backtester.run(
            test_bars,
            bh_signals,
            strategy_name=f"buy-and-hold fold {i}",
        )
        bh_per_fold.append(bh_result)

    # ------------------------------------------------------------------
    # 4. Stitch per-fold returns into the OOS timeline and score it.
    # ------------------------------------------------------------------

    # _stitch_oos concatenates each fold's returns (seam-zero stripped), builds
    # the equity curve anchored at 1.0, and computes total_return / sharpe /
    # max_drawdown.  Same helper is used for the benchmark in step 5 so the two
    # are guaranteed to use identical stitching and metric math.
    (
        oos_returns,
        oos_equity_curve,
        oos_total_return,
        oos_sharpe,
        oos_max_drawdown,
        oos_sortino,
    ) = _stitch_oos(per_fold, annualization_factor)

    # ------------------------------------------------------------------
    # 5. Stitch the buy-and-hold benchmark over the IDENTICAL test windows.
    # ------------------------------------------------------------------

    # Same helper, same annualization, same folds — only bh_per_fold's per-bar
    # returns differ (always-long instead of strategy signals). We keep only the
    # three scalar benchmark metrics; the benchmark's returns/equity arrays are
    # not stored on the result (callers compare scalars, not curves).
    # Positions line up with the strategy unpack above: elements 0 and 1
    # (returns, equity_curve) are discarded; 2,3,4,5 are total_return, sharpe,
    # max_drawdown, sortino — so bh_sortino is element 5, matching oos_sortino.
    _, _, bh_return, bh_sharpe, bh_max_drawdown, bh_sortino = _stitch_oos(
        bh_per_fold, annualization_factor
    )

    # ------------------------------------------------------------------
    # 7. Aggregate trades across all folds.
    # ------------------------------------------------------------------

    # Flatten in fold order — each fold's trades are chronological, and folds
    # are in time order, so all_trades is globally chronological.
    all_trades: list[Trade] = [t for fr in per_fold for t in fr.trades]

    # win_rate over the full OOS trade population; metrics.win_rate handles
    # the empty-list case (no trades → 0.0) so no guard is needed here.
    oos_win_rate = metrics.win_rate(all_trades)

    # Cache the count so callers don't recompute len(); mirrors n_trades on
    # BacktestResult.
    total_trades = len(all_trades)

    # ------------------------------------------------------------------
    # 8. Fold-level summary counts.
    # ------------------------------------------------------------------

    # n_folds == len(splits); stored on the result so callers don't recompute.
    n_folds = len(splits)

    # n_folds_positive_sharpe is a robustness indicator: a strategy with
    # consistently positive fold-level Sharpe ratios is more reliable than one
    # whose high aggregate is driven by a single outlier fold.
    n_folds_positive_sharpe = sum(1 for fr in per_fold if fr.sharpe_ratio > 0)

    # ------------------------------------------------------------------
    # 9. Assemble and return the immutable result.
    # ------------------------------------------------------------------

    return WalkForwardResult(
        per_fold=per_fold,
        oos_returns=oos_returns,
        oos_equity_curve=oos_equity_curve,
        oos_total_return=oos_total_return,
        oos_sharpe=oos_sharpe,
        oos_sortino=oos_sortino,
        oos_max_drawdown=oos_max_drawdown,
        oos_win_rate=oos_win_rate,
        bh_return=bh_return,
        bh_sharpe=bh_sharpe,
        bh_sortino=bh_sortino,
        bh_max_drawdown=bh_max_drawdown,
        n_folds=n_folds,
        n_folds_positive_sharpe=n_folds_positive_sharpe,
        total_trades=total_trades,
    )
