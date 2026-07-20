# tests/test_rotation_search.py

"""
Hermetic unit tests for the Day-44 rotation PER-FOLD SEARCH added to
src/research/rotation_verdict.py — _rotation_train_span_returns, _search_fold,
rotation_walk_forward_search, and the RotationSearchVerdict dataclass.

All tests are pure: NO DuckDB, NO network, NO filesystem.  Bars are hand-built with a
make_month_end_bars + monthly_dates helper copied from tests/test_rotation_verdict.py
(close == adj_close, so the default price_field="close" reads the intended value; one
bar per calendar month, so EVERY bar is a month-end and M == n).

Run with:
    uv run pytest tests/test_rotation_search.py -v
"""

# statistics.mean pins the tax formula (mean in-sample − fitted OOS) in test 6.
from statistics import mean

# numpy builds the synthetic streams and provides assert_array_equal for the no-leak proof.
import numpy as np

# pytest.raises drives the guard tests; pytest.approx tolerates float noise in the tax math.
import pytest

# datetime + timezone build the explicit month-end bar timestamps.
from datetime import datetime, timezone

# OHLCVBar is the bar schema every test builds.
from src.data.schema import OHLCVBar

# The reused pure tax row type — asserted on RotationSearchVerdict.tax_row in test 6.
from scripts.overfitting_tax import TaxRow

# The Day-43 verdict + function, exercised by the additive smoke test (test 8) and the
# benchmark-equivalence test (test 7) to prove the search left the Day-43 path untouched.
from src.research.rotation_verdict import (
    RotationVerdict,
    rotation_walk_forward,
)

# The Day-44 units under test: the train-span in-sample stream, the per-fold grid search,
# the searched verdict, and its dataclass.
from src.research.rotation_verdict import (
    RotationSearchVerdict,
    _rotation_train_span_returns,
    _search_fold,
    rotation_walk_forward_search,
)

# metrics.sharpe_ratio independently recomputes candidate in-sample Sharpes in tests 2/3,
# scoring on the SAME function _search_fold uses so the comparison is exact.
from src.backtest import metrics


# ---------------------------------------------------------------------------
# Synthetic bar helpers — copied verbatim from tests/test_rotation_verdict.py.
# ---------------------------------------------------------------------------


def make_month_end_bars(
    dates: list[datetime],
    closes: list[float],
    symbol: str = "T",
) -> list[OHLCVBar]:
    """Build one OHLCVBar per (date, close), in the given (ascending) order.

    Copied from tests/test_rotation_verdict.py so these tests stay hermetic: close ==
    adj_close so the default price_field="close" reads the intended value; OHLC are dummies.
    """
    # One bar per (date, close) pair, in the given order (caller supplies ascending dates).
    return [
        OHLCVBar(
            symbol=symbol,        # arbitrary; the code reads only timestamp + price
            timestamp=ts,         # the bar's real trading date
            open=c,               # dummy
            high=c,               # dummy
            low=c,                # dummy
            close=c,              # the field the default price_field reads
            adj_close=c,          # equal to close -> basis-agnostic
            volume=1000,          # dummy
            timeframe="1d",       # daily
            source="test",        # provenance marker
        )
        for ts, c in zip(dates, closes)
    ]


def monthly_dates(n: int, start_year: int = 2015, start_month: int = 1) -> list[datetime]:
    """Return n ascending month-end-ish dates, one per consecutive calendar month.

    Copied from tests/test_rotation_verdict.py.  Uses day 28 (valid in every month) so each
    bar lands in a distinct calendar month; with one bar per month EVERY bar is a month-end
    and M == n.
    """
    # Walk consecutive (year, month) pairs, emitting one timestamp each.
    dates: list[datetime] = []
    year, month = start_year, start_month
    for _ in range(n):
        # Day 28 is safe for all 12 months; the calendar month is what matters.
        dates.append(datetime(year, month, 28, tzinfo=timezone.utc))
        # Advance one calendar month, rolling the year over after December.
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


def two_symbol_basket(
    closes_a: list[float],
    closes_b: list[float],
) -> dict[str, list[OHLCVBar]]:
    """Build a 2-symbol basket ("A" reference, "B") on shared monthly month-ends.

    Both symbols share the SAME monthly_dates grid (one bar per month), so every held
    symbol's holding-period bars align to the reference spine — rotation_backtest never
    trips its alignment guard.  M == len(closes_a) == len(closes_b).
    """
    # Both close lists must have equal length so the shared date grid aligns exactly.
    assert len(closes_a) == len(closes_b), "A and B must have equal bar counts"
    # One shared ascending monthly date grid for both symbols.
    dates = monthly_dates(len(closes_a))
    # Return the basket keyed by symbol, with "A" the reference spine.
    return {
        "A": make_month_end_bars(dates, closes_a, symbol="A"),
        "B": make_month_end_bars(dates, closes_b, symbol="B"),
    }


# ---------------------------------------------------------------------------
# 1. The no-leak crux: mutating a TEST-span bar leaves the in-sample stream identical.
# ---------------------------------------------------------------------------


def test_train_span_returns_no_test_leak():
    """Mutating a bar strictly AFTER the split boundary does NOT change the in-sample stream.

    Builds a single fold (train_start=0, test_start=5) and produces a candidate's in-sample
    train-span stream via _rotation_train_span_returns.  Then rebuilds the basket with a
    CHANGED close on month-end position 6 (strictly after D_5, i.e. inside the TEST span) and
    reproduces the in-sample stream.  The two must be byte-identical: the in-sample score can
    see ONLY train-span data [0, 5) plus the shared closing boundary D_5, never a test return.
    """
    # M = 8 month-ends; two symbols with distinct varying (all-up) closes so the stream is
    # non-empty and non-degenerate.
    closes_a = [100.0, 108.0, 119.0, 126.0, 140.0, 152.0, 168.0, 180.0]
    closes_b = [100.0, 104.0, 111.0, 118.0, 121.0, 130.0, 137.0, 149.0]
    basket = two_symbol_basket(closes_a, closes_b)

    # In-sample stream for candidate (top_n=1, lookback=2) on fold (train_start=0, test_start=5).
    # Inner triple is (0, 3, 5): it consumes ONLY month-ends 0..5, so positions 6,7 are untouched.
    stream_before = _rotation_train_span_returns(
        bars_by_symbol=basket,
        reference_symbol="A",
        train_start_me=0,
        test_start_me=5,
        lookback=2,
        top_n=1,
        cost_rate=0.0,
        hold_when_all_negative=True,  # always invested -> deterministic holdings for the proof
    )

    # MUTATE a TEST-span bar: change A's close at month-end position 6 (strictly after D_5).
    mutated_a = list(closes_a)
    mutated_a[6] = 999.0  # a large, obviously-different value on a post-boundary month-end
    mutated_basket = two_symbol_basket(mutated_a, closes_b)

    # Reproduce the in-sample stream on the mutated basket, same fold, same candidate.
    stream_after = _rotation_train_span_returns(
        bars_by_symbol=mutated_basket,
        reference_symbol="A",
        train_start_me=0,
        test_start_me=5,
        lookback=2,
        top_n=1,
        cost_rate=0.0,
        hold_when_all_negative=True,
    )

    # The stream must be non-empty (a real proof, not a vacuous equality of two empty arrays).
    assert stream_before.size > 0
    # NO test-span data leaked into the in-sample score: the two streams are byte-identical.
    np.testing.assert_array_equal(stream_before, stream_after)


# ---------------------------------------------------------------------------
# 2. The search picks the candidate with the best in-sample Sharpe.
# ---------------------------------------------------------------------------


def test_search_picks_best_in_sample():
    """One candidate is clearly best in-sample; _search_fold returns it and its Sharpe.

    A trends strongly up (high in-sample Sharpe held alone at top_n=1); B zig-zags around
    flat, so holding BOTH at top_n=2 dilutes A's edge with B's noise and lowers the Sharpe.
    _search_fold must return (top_n=1, lookback=2) and the in-sample Sharpe of that stream.
    """
    # M = 9; A a varied uptrend, B a ~flat zig-zag (so top_n=2 is strictly worse in-sample).
    closes_a = [100.0, 108.0, 119.0, 126.0, 140.0, 152.0, 168.0, 180.0, 200.0]
    closes_b = [100.0, 96.0, 101.0, 97.0, 102.0, 98.0, 103.0, 99.0, 104.0]
    basket = two_symbol_basket(closes_a, closes_b)

    # Grid: same lookback (2), differing only in top_n, so the winner isolates the top_n effect.
    candidates = [(1, 2), (2, 2)]

    # Search fold (train_start=0, test_start=5, test_end=7) — the sole fold at train=5/test=2 on M=9.
    best_top_n, best_lookback, best_sharpe = _search_fold(
        bars_by_symbol=basket,
        reference_symbol="A",
        train_start_me=0,
        test_start_me=5,
        test_end_me=7,
        candidates=candidates,
        cost_rate=0.0,
        annualization_factor=252,
        hold_when_all_negative=True,
    )

    # Independently recompute each candidate's in-sample Sharpe via the SAME helper + metric.
    def in_sample_sharpe(top_n: int, lookback: int) -> float:
        stream = _rotation_train_span_returns(
            bars_by_symbol=basket,
            reference_symbol="A",
            train_start_me=0,
            test_start_me=5,
            lookback=lookback,
            top_n=top_n,
            cost_rate=0.0,
            hold_when_all_negative=True,
        )
        return metrics.sharpe_ratio(stream, 252)

    sharpe_top1 = in_sample_sharpe(1, 2)
    sharpe_top2 = in_sample_sharpe(2, 2)

    # The dataset genuinely separates: holding A alone beats diluting it with B's noise.
    assert sharpe_top1 > sharpe_top2
    # The search returned the clear winner's params and its exact in-sample Sharpe.
    assert (best_top_n, best_lookback) == (1, 2)
    assert best_sharpe == pytest.approx(sharpe_top1)


# ---------------------------------------------------------------------------
# 3. Deterministic tiebreak: on an exact in-sample tie, smallest (lookback, top_n) wins.
# ---------------------------------------------------------------------------


def test_search_tiebreak_deterministic():
    """Two candidates that tie in-sample resolve to the smaller (lookback, top_n).

    With A and B IDENTICAL, top_n=1 (hold one at weight 1) and top_n=2 (hold both at 1/2)
    produce the byte-identical combined return stream, so their in-sample Sharpes tie exactly.
    The locked tiebreak (sort by (lookback, top_n); first-max-wins on strictly-greater) must
    then pick the smaller top_n — (1, 2).
    """
    # M = 8; A and B IDENTICAL varied uptrend so the two candidates' streams are identical.
    closes = [100.0, 108.0, 119.0, 126.0, 140.0, 152.0, 168.0, 180.0]
    basket = two_symbol_basket(list(closes), list(closes))

    # Same lookback (2); the ONLY difference is top_n — and with identical symbols even that
    # cancels, producing an exact tie the tiebreak must resolve deterministically.
    candidates = [(1, 2), (2, 2)]

    # Prove the tie is REAL: the two candidates' in-sample streams are byte-identical.
    stream_top1 = _rotation_train_span_returns(
        bars_by_symbol=basket, reference_symbol="A",
        train_start_me=0, test_start_me=5, lookback=2, top_n=1, cost_rate=0.0,
        hold_when_all_negative=True,
    )
    stream_top2 = _rotation_train_span_returns(
        bars_by_symbol=basket, reference_symbol="A",
        train_start_me=0, test_start_me=5, lookback=2, top_n=2, cost_rate=0.0,
        hold_when_all_negative=True,
    )
    np.testing.assert_array_equal(stream_top1, stream_top2)

    # Search the fold; on the exact tie the winner must be the smaller top_n at that lookback.
    best_top_n, best_lookback, _best_sharpe = _search_fold(
        bars_by_symbol=basket,
        reference_symbol="A",
        train_start_me=0,
        test_start_me=5,
        test_end_me=7,
        candidates=candidates,
        cost_rate=0.0,
        annualization_factor=252,
        hold_when_all_negative=True,
    )

    # Tiebreak resolves to smallest (lookback, top_n) -> (1, 2).
    assert (best_top_n, best_lookback) == (1, 2)


# ---------------------------------------------------------------------------
# 4. The L_max warm-up guard raises when the grid's largest lookback is too big.
# ---------------------------------------------------------------------------


def test_lmax_guard_raises():
    """train_months <= L_max + 1 leaves no in-sample scoring month-ends -> ValueError.

    Grid L_max = 5 with train_months = 6 fails the guard (6 <= 5 + 1), because the largest
    candidate's inner in-sample span would be train_months − L_max − 1 = 0 month-ends.
    """
    # Any small aligned basket suffices; the guard fires before folds are even built.
    basket = two_symbol_basket(
        [100.0 + i for i in range(12)],
        [100.0 + i for i in range(12)],
    )
    # L_max = 5, train_months = 6 -> 6 <= 6 -> the guard must raise.
    with pytest.raises(ValueError):
        rotation_walk_forward_search(
            bars_by_symbol=basket,
            reference_symbol="A",
            candidates=[(1, 5)],
            train_months=6,
            test_months=2,
        )


# ---------------------------------------------------------------------------
# 5. An empty candidate grid raises.
# ---------------------------------------------------------------------------


def test_candidates_empty_raises():
    """An empty candidates list has nothing to search -> ValueError."""
    # A valid basket; the empty-grid guard fires first regardless of the data.
    basket = two_symbol_basket(
        [100.0 + i for i in range(12)],
        [100.0 + i for i in range(12)],
    )
    with pytest.raises(ValueError):
        rotation_walk_forward_search(
            bars_by_symbol=basket,
            reference_symbol="A",
            candidates=[],
            train_months=5,
            test_months=2,
        )


# ---------------------------------------------------------------------------
# 6. The searched verdict is well-formed and its tax matches summarize_tax exactly.
# ---------------------------------------------------------------------------


def test_search_verdict_fields_populated():
    """A 2-fold searched run yields per-fold lists of length n_folds and the locked tax formula.

    Asserts n_folds == 2, len(chosen_*) == len(in_sample_sharpes) == 2, and that
    tax == mean(in_sample_sharpes) − fitted_oos_sharpe (the exact summarize_tax definition),
    with mean_in_sample_sharpe likewise the mean of the per-fold in-sample Sharpes.
    """
    # M = 9, train=4/test=2/step=2 -> folds (0,4,6) and (2,6,8): exactly 2 folds.
    closes_a = [100.0, 108.0, 119.0, 126.0, 140.0, 152.0, 168.0, 180.0, 200.0]
    closes_b = [100.0, 104.0, 96.0, 111.0, 102.0, 118.0, 108.0, 125.0, 114.0]
    basket = two_symbol_basket(closes_a, closes_b)

    # Grid L_max = 2, train_months = 4 > L_max + 1 = 3 -> guard passes.
    verdict = rotation_walk_forward_search(
        bars_by_symbol=basket,
        reference_symbol="A",
        candidates=[(1, 1), (2, 2)],
        train_months=4,
        test_months=2,
        annualization_factor=252,
        hold_when_all_negative=True,
        symbol_label="ROT",
    )

    # It is the searched dataclass, carrying a reused TaxRow.
    assert isinstance(verdict, RotationSearchVerdict)
    assert isinstance(verdict.tax_row, TaxRow)

    # Two folds ran, and every per-fold list is exactly n_folds long.
    assert verdict.n_folds == 2
    assert len(verdict.chosen_top_n) == 2
    assert len(verdict.chosen_lookback) == 2
    assert len(verdict.in_sample_sharpes) == 2

    # mean_in_sample_sharpe is the mean of the per-fold in-sample Sharpes.
    assert verdict.mean_in_sample_sharpe == pytest.approx(mean(verdict.in_sample_sharpes))
    # tax == mean in-sample − fitted OOS (the exact summarize_tax formula, byte-for-byte reused).
    assert verdict.tax == pytest.approx(
        mean(verdict.in_sample_sharpes) - verdict.fitted_oos_sharpe
    )
    # The TaxRow's label threaded through unchanged.
    assert verdict.tax_row.symbol == "ROT"


# ---------------------------------------------------------------------------
# 7. The benchmark is unchanged by the search (lookback-invariant, Day-43-equal).
# ---------------------------------------------------------------------------


def test_benchmark_matches_day43():
    """The searched run's bh_sharpe equals the Day-43 rotation_walk_forward's bh_sharpe.

    The benchmark holds EVERY symbol equal-weight regardless of rank, so its stream is
    lookback-invariant; the searched run therefore produces the SAME benchmark the Day-43
    fixed-parameter bridge does over the same folds/cost/basis.
    """
    # M = 9; the same basket shape as test 6.
    closes_a = [100.0, 108.0, 119.0, 126.0, 140.0, 152.0, 168.0, 180.0, 200.0]
    closes_b = [100.0, 104.0, 96.0, 111.0, 102.0, 118.0, 108.0, 125.0, 114.0]
    basket = two_symbol_basket(closes_a, closes_b)

    # Searched run: benchmark uses the grid's smallest lookback (1) internally.
    searched = rotation_walk_forward_search(
        bars_by_symbol=basket,
        reference_symbol="A",
        candidates=[(1, 1), (2, 2)],
        train_months=4,
        test_months=2,
        cost_rate=0.0,
        hold_when_all_negative=True,
    )

    # Day-43 fixed run with the SAME smallest lookback (1) over the SAME folds/cost.
    day43 = rotation_walk_forward(
        bars_by_symbol=basket,
        reference_symbol="A",
        lookback=1,
        top_n=1,
        train_months=4,
        test_months=2,
        cost_rate=0.0,
        hold_when_all_negative=True,
    )

    # The benchmark aggregates match: the search left the Day-43 benchmark path untouched.
    assert searched.bh_sharpe == pytest.approx(day43.bh_sharpe)
    assert searched.bh_total_return == pytest.approx(day43.bh_total_return)


# ---------------------------------------------------------------------------
# 8. Additive smoke test: the Day-43 rotation_walk_forward is untouched.
# ---------------------------------------------------------------------------


def test_search_is_additive_day43_unchanged():
    """The Day-43 rotation_walk_forward still returns a well-formed RotationVerdict.

    A pure smoke test that the additive Day-44 code did not disturb the Day-43 path: the
    fixed-parameter bridge still runs and returns a RotationVerdict with finite fields.
    """
    # M = 9; a simple aligned basket.
    closes_a = [100.0, 108.0, 119.0, 126.0, 140.0, 152.0, 168.0, 180.0, 200.0]
    closes_b = [100.0, 104.0, 111.0, 118.0, 121.0, 130.0, 137.0, 149.0, 158.0]
    basket = two_symbol_basket(closes_a, closes_b)

    # The Day-43 fixed-parameter verdict.
    verdict = rotation_walk_forward(
        bars_by_symbol=basket,
        reference_symbol="A",
        lookback=2,
        top_n=1,
        train_months=4,
        test_months=2,
    )

    # It is the Day-43 dataclass with its expected fields present and finite.
    assert isinstance(verdict, RotationVerdict)
    assert verdict.n_folds >= 1
    assert np.isfinite(verdict.oos_sharpe)
    assert np.isfinite(verdict.bh_sharpe)
