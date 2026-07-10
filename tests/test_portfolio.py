# tests/test_portfolio.py

"""
Hermetic unit tests for src/research/portfolio.py.

All tests are pure: NO DuckDB, NO network, NO filesystem, NO bars.  combine_period_returns
takes plain dict[str, np.ndarray] of already-computed per-symbol return streams plus a
weights dict, so every test is just hand-built numpy arrays and exact/approx assertions.
Float array comparisons use np.testing.assert_allclose; all arrays are built with
np.array(..., dtype=float).  Matches the comment-heavy style of tests/test_cross_sectional.py.

Run with:
    uv run pytest tests/test_portfolio.py -v
"""

# numpy builds the synthetic per-symbol return arrays and the hand-computed expected
# arrays, and provides assert_allclose for the float comparisons.
import numpy as np

# pytest.raises is used for the three validation tests (n_bars, length, key-set) and the
# rotation_backtest guard tests.
import pytest

# datetime + timezone build the explicit month-end (and intra-month) timestamps for the
# synthetic bars the rotation_backtest tests hand-build.
from datetime import datetime, timezone

# OHLCVBar is the bar schema the _per_bar_log_returns and rotation_backtest tests build.
from src.data.schema import OHLCVBar

# The units under test: the pure return-combination core (existing), plus the new per-bar
# log-return helper and the monthly rotation rebalance loop.
from src.research.portfolio import (
    combine_period_returns,
    _per_bar_log_returns,
    rotation_backtest,
)


# ---------------------------------------------------------------------------
# 1. Equal-weight, fully invested — pure weighted sum, no cash.
# ---------------------------------------------------------------------------


def test_equal_weight_full_invested_combines_correctly():
    """Two symbols, 3 bars, equal weight 0.5 each, no cash -> 0.5*A + 0.5*B elementwise."""
    # Two held symbols, each a 3-bar per-bar return stream (hand chosen so the weighted
    # sum is easy to verify by eye).
    per_symbol_returns = {
        "A": np.array([0.10, 0.00, -0.05], dtype=float),
        "B": np.array([0.00, 0.20, 0.05], dtype=float),
    }
    # Equal weight 0.5 each; they sum to 1.0 so the cash remainder is 0.0 (fully invested).
    weights = {"A": 0.5, "B": 0.5}
    # 3-bar period; cash rate 0.0 so cash contributes nothing even though remainder is 0.
    result = combine_period_returns(per_symbol_returns, weights, n_bars=3, cash_per_bar_return=0.0)
    # Hand-computed expected: 0.5*A + 0.5*B on each bar.
    expected = 0.5 * per_symbol_returns["A"] + 0.5 * per_symbol_returns["B"]
    # Exact-up-to-float weighted sum.
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# 2. All-cash period — empty holdings earn the cash rate on every bar.
# ---------------------------------------------------------------------------


def test_all_cash_period_earns_cash_rate():
    """Empty holdings, n_bars=4, cash rate 0.001 -> every bar is exactly the cash rate."""
    # Nothing held (the ranker returned an empty selection) — empty returns and weights.
    per_symbol_returns: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    # cash_weight = 1 - sum({}) = 1.0, so all capital earns the cash rate on every bar.
    result = combine_period_returns(per_symbol_returns, weights, n_bars=4, cash_per_bar_return=0.001)
    # Every one of the 4 bars must equal 0.001 (1.0 * 0.001).  Pins that the all-cash
    # case is handled with no crash (length comes from n_bars, not the empty dict).
    np.testing.assert_allclose(result, np.full(4, 0.001))


# ---------------------------------------------------------------------------
# 3. Partial-cash period — the unallocated remainder earns the cash rate.
# ---------------------------------------------------------------------------


def test_partial_cash_remainder_earns_cash():
    """Two symbols each at 1/3 (sum 2/3), n_bars=2 -> (1/3)A + (1/3)B + (1/3)*cash_rate."""
    # Two held symbols, 2-bar streams.
    per_symbol_returns = {
        "A": np.array([0.10, -0.02], dtype=float),
        "B": np.array([0.04, 0.06], dtype=float),
    }
    # Each weighted 1/3, so the invested fraction is 2/3 and the cash remainder is 1/3.
    weights = {"A": 1.0 / 3.0, "B": 1.0 / 3.0}
    # 2-bar period; cash rate 0.001 on the 1/3 cash remainder.
    result = combine_period_returns(per_symbol_returns, weights, n_bars=2, cash_per_bar_return=0.001)
    # Hand-computed expected: (1/3)A + (1/3)B + (1/3)*0.001 on each bar.  Pins the
    # partial-cash case (ranker returned fewer than top_n).
    expected = (
        (1.0 / 3.0) * per_symbol_returns["A"]
        + (1.0 / 3.0) * per_symbol_returns["B"]
        + (1.0 / 3.0) * 0.001
    )
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# 4. Single symbol at full weight — the N=1 identity.
# ---------------------------------------------------------------------------


def test_single_symbol_full_weight_equals_that_symbol():
    """One symbol weighted 1.0 -> result is exactly that symbol's stream; cash never contributes."""
    # A single held symbol over 3 bars.
    per_symbol_returns = {"A": np.array([0.07, -0.01, 0.03], dtype=float)}
    # Weighted 1.0, so the cash remainder is 0.0 no matter what the cash rate is.
    weights = {"A": 1.0}
    # Deliberately pass a NON-zero cash rate to prove it must NOT contribute when
    # cash_weight is 0.0.
    result = combine_period_returns(per_symbol_returns, weights, n_bars=3, cash_per_bar_return=0.001)
    # Result must equal the symbol's own stream exactly — the N=1 identity, the anchor
    # for the later engine-equivalence test.
    np.testing.assert_allclose(result, per_symbol_returns["A"])


# ---------------------------------------------------------------------------
# 5. n_bars < 1 is rejected.
# ---------------------------------------------------------------------------


def test_n_bars_below_one_raises():
    """n_bars=0 -> ValueError (a zero-length period is meaningless)."""
    # Empty inputs are fine; the n_bars guard fires first regardless.
    with pytest.raises(ValueError):
        combine_period_returns({}, {}, n_bars=0)


# ---------------------------------------------------------------------------
# 6. A per-symbol array whose length != n_bars is rejected.
# ---------------------------------------------------------------------------


def test_length_mismatch_raises():
    """A symbol whose array length != n_bars -> ValueError (misalignment is a caller bug)."""
    # "A" has 2 bars but the caller declares n_bars=3 — a misalignment.
    per_symbol_returns = {"A": np.array([0.10, 0.00], dtype=float)}
    weights = {"A": 1.0}
    with pytest.raises(ValueError):
        combine_period_returns(per_symbol_returns, weights, n_bars=3)


# ---------------------------------------------------------------------------
# 7. Key-set equality guard — a weight without returns, or returns without a weight.
# ---------------------------------------------------------------------------


def test_weight_without_returns_raises():
    """Key sets must be EQUAL: a weighted symbol with no returns, or vice versa, raises."""
    # Sub-case 1: "B" has a weight but NO returns array -> its contribution would be
    # silently dropped and the portfolio misweighted, so this must raise.
    per_symbol_returns = {"A": np.array([0.10, 0.00], dtype=float)}
    weights = {"A": 0.5, "B": 0.5}
    with pytest.raises(ValueError):
        combine_period_returns(per_symbol_returns, weights, n_bars=2)

    # Sub-case 2: "B" has a returns array but NO weight -> a held symbol would be
    # silently ignored, so this must also raise.
    per_symbol_returns_2 = {
        "A": np.array([0.10, 0.00], dtype=float),
        "B": np.array([0.01, 0.02], dtype=float),
    }
    weights_2 = {"A": 1.0}
    with pytest.raises(ValueError):
        combine_period_returns(per_symbol_returns_2, weights_2, n_bars=2)


# ---------------------------------------------------------------------------
# 8. Output contract — float64 dtype, length n_bars.
# ---------------------------------------------------------------------------


def test_returns_float64_length_n_bars():
    """A normal 2-symbol case returns a float64 array of length n_bars."""
    # Two symbols over 3 bars, equal weight.
    per_symbol_returns = {
        "A": np.array([0.10, 0.00, -0.05], dtype=float),
        "B": np.array([0.00, 0.20, 0.05], dtype=float),
    }
    weights = {"A": 0.5, "B": 0.5}
    result = combine_period_returns(per_symbol_returns, weights, n_bars=3)
    # dtype must be float64 (the engine/metrics convention).
    assert result.dtype == np.float64
    # length must equal n_bars.
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Synthetic bar helper for the _per_bar_log_returns / rotation_backtest tests.
# Copied from tests/test_cross_sectional.py so these tests stay hermetic (no
# DuckDB): one OHLCVBar per (date, close), with close == adj_close so the default
# price_field="close" reads the intended value.  Unlike the cross_sectional usage,
# the rotation tests pass MULTIPLE dates within one calendar month (intra-month
# daily bars), so month_end_indices — not this helper — decides what a month-end is.
# ---------------------------------------------------------------------------


def make_month_end_bars(
    dates: list[datetime],
    closes: list[float],
    symbol: str = "T",
) -> list[OHLCVBar]:
    """Build one OHLCVBar per (date, close), in the given (ascending) order."""
    # Zip pairs each date with its close; one bar apiece, in the given order (the
    # caller supplies ascending chronological dates, as everywhere else).
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


# ---------------------------------------------------------------------------
# 9. _per_bar_log_returns matches the engine convention (element 0 == 0.0).
# ---------------------------------------------------------------------------


def test_per_bar_log_returns_matches_engine_convention():
    """A 4-bar symbol -> [0.0, log(c1/c0), log(c2/c1), log(c3/c2)] (engine.py:288-293)."""
    # Four bars, one per month (months are irrelevant here — the helper does not look at
    # dates, only prices), with round closes so the log returns are easy to reason about.
    dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
        datetime(2024, 4, 30, tzinfo=timezone.utc),
    ]
    # Closes chosen so each ratio is clean; the last one goes DOWN to prove sign is kept.
    closes = [100.0, 110.0, 121.0, 100.0]
    bars = make_month_end_bars(dates, closes)
    # Compute the per-bar log returns via the unit under test.
    result = _per_bar_log_returns(bars)
    # Hand-computed expected: element 0 is the structural 0.0 (no prior bar), then log of
    # each consecutive close ratio — EXACTLY the engine's asset_returns convention.
    expected = np.array(
        [
            0.0,
            np.log(110.0 / 100.0),
            np.log(121.0 / 110.0),
            np.log(100.0 / 121.0),
        ],
        dtype=float,
    )
    # Element 0 must be a hard 0.0 (structural), and the whole array must match.
    assert result[0] == 0.0
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# 10. _per_bar_log_returns rejects empty bars.
# ---------------------------------------------------------------------------


def test_per_bar_log_returns_empty_raises():
    """Empty bars -> ValueError (no return series can be formed)."""
    # No bars at all; the guard must fire before any array is built.
    with pytest.raises(ValueError):
        _per_bar_log_returns([])


# ---------------------------------------------------------------------------
# 11. rotation_backtest end-to-end, two symbols, hand-computed stitched stream.
# ---------------------------------------------------------------------------


def test_rotation_two_symbols_hand_computed():
    """2 symbols, 5 shared month-ends, lookback=1, top_n=1 -> hand-computed stitched stream."""
    # Five shared month-ends (Jan-May 2024), one bar per month so each holding period is
    # exactly one bar (the next month-end) — trivially hand-computable.
    dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
        datetime(2024, 4, 30, tzinfo=timezone.utc),
        datetime(2024, 5, 31, tzinfo=timezone.utc),
    ]
    # A rises monotonically; B chosen so the monthly winner alternates and B goes negative
    # in April (so the absolute filter drops it there).
    bars_by_symbol = {
        "A": make_month_end_bars(dates, [100.0, 110.0, 120.0, 130.0, 140.0], "A"),
        "B": make_month_end_bars(dates, [100.0, 105.0, 130.0, 120.0, 200.0], "B"),
    }
    # Rank by 1-month trailing return; hold the single strongest; A is the reference spine.
    result = rotation_backtest(
        bars_by_symbol, reference_symbol="A", lookback=1, top_n=1
    )
    # Hand-computed, period by period (each period is one bar, the D_{k+1} month-end):
    #   P0 (Jan->Feb): rank as of Jan31 is WARMUP (lookback=1 -> NaN), held empty -> cash
    #                  -> 0.0 (default cash rate).
    #   P1 (Feb->Mar): as of Feb, A=+0.10 > B=+0.05 -> hold A -> log(120/110).
    #   P2 (Mar->Apr): as of Mar, B=+0.2381 > A=+0.0909 -> hold B -> log(120/130).
    #   P3 (Apr->May): as of Apr, A=+0.0833 > 0, B=-0.0769 filtered -> hold A -> log(140/130).
    expected = np.array(
        [
            0.0,
            np.log(120.0 / 110.0),
            np.log(120.0 / 130.0),
            np.log(140.0 / 130.0),
        ],
        dtype=float,
    )
    # The stitched stream must match exactly (one bar per period, four periods).
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# 12. rotation_backtest all-cash period earns the cash rate on each of its bars.
# ---------------------------------------------------------------------------


def test_rotation_all_cash_period_uses_cash_rate():
    """A rebalance where BOTH symbols are <=0 -> empty held -> that period earns cash_rate."""
    # Four shared month-ends; at Feb BOTH symbols are DOWN over the trailing month, so the
    # default absolute filter selects nothing (empty held -> all cash).
    dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
        datetime(2024, 4, 30, tzinfo=timezone.utc),
    ]
    bars_by_symbol = {
        "A": make_month_end_bars(dates, [100.0, 90.0, 95.0, 100.0], "A"),
        "B": make_month_end_bars(dates, [100.0, 80.0, 85.0, 90.0], "B"),
    }
    # Non-zero cash rate so the all-cash period is visibly the cash rate, not zero.
    result = rotation_backtest(
        bars_by_symbol, reference_symbol="A", lookback=1, top_n=1, cash_per_bar_return=0.001
    )
    # Period 1 = (Feb, Mar]: ranked as of Feb where A=-0.10 and B=-0.20 are both <=0, so
    # held is empty and the whole (single-bar) period earns exactly the cash rate.
    np.testing.assert_allclose(result[1], 0.001)


# ---------------------------------------------------------------------------
# 13. rotation_backtest partial-cash: fewer than top_n qualify -> rule (b) 1/top_n each.
# ---------------------------------------------------------------------------


def test_rotation_partial_cash_fewer_than_top_n():
    """top_n=2 but only 1 symbol clears the filter -> (1/2)*symbol + (1/2)*cash (rule b)."""
    # Three shared month-ends; at Feb only A is positive (B is negative and filtered), so
    # ONE symbol is held while top_n=2 -> rule (b) invests 1/2 and leaves 1/2 in cash.
    dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
    ]
    bars_by_symbol = {
        "A": make_month_end_bars(dates, [100.0, 110.0, 120.0], "A"),
        "B": make_month_end_bars(dates, [100.0, 90.0, 130.0], "B"),
    }
    # top_n=2 with a non-zero cash rate on the un-filled half.
    result = rotation_backtest(
        bars_by_symbol, reference_symbol="A", lookback=1, top_n=2, cash_per_bar_return=0.001
    )
    # Period 1 = (Feb, Mar]: as of Feb, A=+0.10 (kept), B=-0.10 (filtered) -> held = [A].
    # Rule (b): A gets weight 1/top_n = 0.5, and the remaining 0.5 sits in cash.  So the
    # single-bar period return is 0.5*log(120/110) + 0.5*0.001.  Under rule (a) it would
    # instead be 1.0*log(120/110) with NO cash — THIS is the test that distinguishes them.
    expected_period1 = 0.5 * np.log(120.0 / 110.0) + 0.5 * 0.001
    np.testing.assert_allclose(result[1], expected_period1)


# ---------------------------------------------------------------------------
# 14. rotation_backtest no-lookahead: D_k's own bar is EXCLUDED from the period.
# ---------------------------------------------------------------------------


def test_rotation_no_lookahead_excludes_Dk_bar():
    """The holding period is (D_k, D_{k+1}] — D_k's own bar's return never appears."""
    # Intra-month daily bars so a holding period spans MORE than one bar and D_k's own
    # month-end bar is distinct from the next bar.  Months: Jan/Feb/Mar, each with a mid
    # and an end bar.  A is engineered so that INCLUDING Feb29 (D_1) would inject the very
    # distinctive return log(110/200); the correct first period-1 return is log(121/110).
    a_dates = [
        datetime(2024, 1, 15, tzinfo=timezone.utc),
        datetime(2024, 1, 31, tzinfo=timezone.utc),  # D_0 (Jan month-end)
        datetime(2024, 2, 15, tzinfo=timezone.utc),  # close 200 -> big Feb29/Feb15 drop
        datetime(2024, 2, 29, tzinfo=timezone.utc),  # D_1 (Feb month-end, EXCLUDED)
        datetime(2024, 3, 15, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),  # D_2 (Mar month-end)
    ]
    a_closes = [50.0, 100.0, 200.0, 110.0, 121.0, 133.0]
    # B stays below A's trailing return at Feb so A is the one held.
    b_dates = list(a_dates)
    b_closes = [50.0, 100.0, 90.0, 95.0, 96.0, 97.0]
    bars_by_symbol = {
        "A": make_month_end_bars(a_dates, a_closes, "A"),
        "B": make_month_end_bars(b_dates, b_closes, "B"),
    }
    result = rotation_backtest(
        bars_by_symbol, reference_symbol="A", lookback=1, top_n=1
    )
    # Period 0 = (Jan31, Feb29]: rank as of Jan31 is warmup -> empty -> cash -> two 0.0
    # bars (Feb15, Feb29).  Period 1 = (Feb29, Mar31]: as of Feb29, A=110/100-1=+0.10 beats
    # B=95/100-1=-0.05 (filtered) -> hold A over Mar15, Mar31 -> [log(121/110), log(133/121)].
    expected = np.array(
        [
            0.0,
            0.0,
            np.log(121.0 / 110.0),
            np.log(133.0 / 121.0),
        ],
        dtype=float,
    )
    np.testing.assert_allclose(result, expected)
    # The tell-tale D_1-bar return (Feb29/Feb15 = 110/200) must NOT appear anywhere — if
    # the period wrongly included D_k's own bar, this value would be the first return.
    forbidden = np.log(110.0 / 200.0)
    assert not np.any(np.isclose(result, forbidden))


# ---------------------------------------------------------------------------
# 15. rotation_backtest fails loud on an interior date gap (no silent misalign).
# ---------------------------------------------------------------------------


def test_rotation_interior_gap_raises():
    """A held symbol missing an interior spine date -> ValueError (never pad/truncate)."""
    # REF is the spine and has a Mar15 interior bar; B is held (highest trailing return at
    # Feb) but is MISSING Mar15, so its holding-period slice cannot align to the spine.
    ref_dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 15, tzinfo=timezone.utc),  # interior day the spine has
        datetime(2024, 3, 31, tzinfo=timezone.utc),
    ]
    b_dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),  # MISSING Mar15 -> gap vs spine
    ]
    bars_by_symbol = {
        "REF": make_month_end_bars(ref_dates, [100.0, 101.0, 102.0, 103.0], "REF"),
        "B": make_month_end_bars(b_dates, [100.0, 150.0, 160.0], "B"),
    }
    # As of Feb29: B=+0.50 beats REF=+0.01 -> held = [B]; B's (Feb29, Mar31] slice is just
    # Mar31 (no Mar15), while the REF spine is [Mar15, Mar31] -> the alignment guard raises.
    with pytest.raises(ValueError):
        rotation_backtest(bars_by_symbol, reference_symbol="REF", lookback=1, top_n=1)


# ---------------------------------------------------------------------------
# 16. rotation_backtest rejects an absent reference symbol.
# ---------------------------------------------------------------------------


def test_rotation_reference_symbol_absent_raises():
    """reference_symbol not in bars_by_symbol -> ValueError."""
    dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
    ]
    bars_by_symbol = {"A": make_month_end_bars(dates, [100.0, 110.0], "A")}
    # "SPY" is not a key -> the reference guard fires before any scheduling.
    with pytest.raises(ValueError):
        rotation_backtest(bars_by_symbol, reference_symbol="SPY", lookback=1, top_n=1)


# ---------------------------------------------------------------------------
# 17. rotation_backtest rejects a reference with too few month-ends.
# ---------------------------------------------------------------------------


def test_rotation_too_few_month_ends_raises():
    """A reference with only 1 month-end -> ValueError (cannot form a holding-period pair)."""
    # A single bar means a single month-end, so no adjacent (D_k, D_{k+1}) pair exists.
    dates = [datetime(2024, 1, 31, tzinfo=timezone.utc)]
    bars_by_symbol = {"A": make_month_end_bars(dates, [100.0], "A")}
    with pytest.raises(ValueError):
        rotation_backtest(bars_by_symbol, reference_symbol="A", lookback=1, top_n=1)
