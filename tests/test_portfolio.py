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

# pytest.raises is used for the three validation tests (n_bars, length, key-set).
import pytest

# The single unit under test: the pure return-combination core.
from src.research.portfolio import combine_period_returns


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
