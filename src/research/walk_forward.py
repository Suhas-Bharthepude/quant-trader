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

# OHLCVBar is the canonical in-memory representation of a single price bar,
# defined in src/data/schema.py.  Importing it here keeps the function's type
# annotations honest and lets callers import the type from one place rather
# than knowing where OHLCVBar lives separately.
from src.data.schema import OHLCVBar


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
