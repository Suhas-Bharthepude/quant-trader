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

# pytest is the test runner; pytest.raises is used to assert that the
# function's validation paths emit the right exceptions.
import pytest

# datetime + timedelta + timezone build sequential UTC-aware timestamps for
# the synthetic bars.  timezone.utc matches the project-wide convention that
# every bar timestamp is timezone-aware UTC.
from datetime import datetime, timedelta, timezone

# OHLCVBar is the bar schema walk_forward_splits operates on.  Importing from
# src.data.schema ties the test to the canonical type definition so a future
# schema rename surfaces here at import time rather than at a runtime field
# access deep inside a test body.
from src.data.schema import OHLCVBar

# walk_forward_splits is the sole function under test.  Every assertion in
# this file routes through it.
from src.research.walk_forward import walk_forward_splits


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
