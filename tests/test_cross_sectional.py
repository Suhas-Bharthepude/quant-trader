# tests/test_cross_sectional.py

"""
Hermetic unit tests for src/research/cross_sectional.py.

All tests are pure: NO DuckDB, NO network, NO filesystem, NO bars at all — the
function under test takes a plain dict[str, float] of already-computed trailing
returns and returns a selected list[str], so the tests are just hand-built dicts
and exact list-equality assertions.  Order matters in every result, so `==` on
the whole list is the right check.  Matches the comment-heavy style of
tests/test_momentum_walkforward.py.

Run with:
    uv run pytest tests/test_cross_sectional.py -v
"""

# pytest.raises is used for the top_n < 1 validation test.
import pytest

# The single unit under test: the pure cross-sectional ranking/selection helper.
from src.research.cross_sectional import rank_by_trailing_return


# ---------------------------------------------------------------------------
# 1. Basic ranking + top_n.
# ---------------------------------------------------------------------------


def test_basic_ranking_selects_top_n_in_descending_order():
    """Distinct positive returns, top_n=2 -> the two highest in descending order."""
    # Three distinct positive returns; SPY (0.20) > TLT (0.10) > GLD (0.05).
    returns = {"SPY": 0.20, "TLT": 0.10, "GLD": 0.05}
    # top_n=2 selects the two strongest, highest first.
    result = rank_by_trailing_return(returns, top_n=2)
    # Exact list AND order: SPY then TLT, GLD excluded by the top_n=2 cut.
    assert result == ["SPY", "TLT"]


# ---------------------------------------------------------------------------
# 2. Deterministic tie-break: equal returns -> alphabetical symbol order.
# ---------------------------------------------------------------------------


def test_ties_broken_alphabetically_by_symbol():
    """Two symbols with identical returns come out in ascending symbol order."""
    # AAA and BBB tie at 0.10; CCC is lower at 0.05.  Insertion order deliberately
    # puts BBB before AAA to prove the result is NOT relying on dict order.
    returns = {"BBB": 0.10, "AAA": 0.10, "CCC": 0.05}
    # top_n=2 pulls the two 0.10 symbols; the tie must resolve alphabetically.
    result = rank_by_trailing_return(returns, top_n=2)
    # AAA before BBB (alphabetical secondary key), CCC dropped by the cut.
    assert result == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# 3. top_n larger than available -> all returned, no error.
# ---------------------------------------------------------------------------


def test_top_n_larger_than_available_returns_all():
    """top_n=10 on a 3-symbol dict returns all 3, still in ranked order."""
    # Only three symbols available; top_n far exceeds that count.
    returns = {"SPY": 0.20, "TLT": 0.10, "GLD": 0.05}
    # top_n=10 must not raise or pad — the slice past the end yields the whole list.
    result = rank_by_trailing_return(returns, top_n=10)
    # All three, still highest-first.
    assert result == ["SPY", "TLT", "GLD"]


# ---------------------------------------------------------------------------
# 4. top_n < 1 raises ValueError.
# ---------------------------------------------------------------------------


def test_top_n_below_one_raises():
    """top_n < 1 is rejected with a ValueError."""
    # A valid dict but an invalid top_n — the guard must fire before any ranking.
    returns = {"SPY": 0.20}
    # top_n=0 is meaningless; pytest.raises confirms the ValueError.
    with pytest.raises(ValueError):
        rank_by_trailing_return(returns, top_n=0)


# ---------------------------------------------------------------------------
# 5. Empty dict in -> empty list out.
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_list():
    """An empty trailing_returns dict yields an empty selection (nothing to rank)."""
    # No symbols supplied at all.
    result = rank_by_trailing_return({}, top_n=3)
    # Nothing to rank -> empty list (not an error).
    assert result == []


# ---------------------------------------------------------------------------
# 6. Absolute filter (default): negatives are never selectable.
# ---------------------------------------------------------------------------


def test_absolute_filter_excludes_negatives_default():
    """Default filter: a negative-return symbol is never returned, even under top_n reach."""
    # Two positives (SPY 0.15, TLT 0.05) and one negative (GLD -0.10).
    returns = {"SPY": 0.15, "TLT": 0.05, "GLD": -0.10}
    # top_n=3 would reach all three, but the absolute filter drops GLD first.
    result = rank_by_trailing_return(returns, top_n=3)
    # Exactly the two positive symbols, highest first; GLD absent despite top_n=3.
    assert result == ["SPY", "TLT"]


# ---------------------------------------------------------------------------
# 7. All-negative with default filter -> empty (cash).
# ---------------------------------------------------------------------------


def test_all_negative_default_returns_empty_cash():
    """All returns <= 0 under the default filter -> empty list (go to cash)."""
    # Every symbol is down over the lookback.
    returns = {"SPY": -0.05, "TLT": -0.20, "GLD": -0.10}
    # Default hold_when_all_negative=False -> nothing clears the > 0.0 filter.
    result = rank_by_trailing_return(returns, top_n=2)
    # Empty selection: the strategy sits in cash rather than holding the least-bad.
    assert result == []


# ---------------------------------------------------------------------------
# 8. All-negative with hold_when_all_negative=True -> least-bad returned.
# ---------------------------------------------------------------------------


def test_all_negative_hold_mode_returns_least_bad():
    """hold_when_all_negative=True returns the top_n least-bad in descending order."""
    # All negative; least-bad is X (-0.05), then Z (-0.10), then Y (-0.20).
    returns = {"X": -0.05, "Y": -0.20, "Z": -0.10}
    # No absolute filter -> rank all by sign-aware value, least negative first.
    result = rank_by_trailing_return(returns, top_n=2, hold_when_all_negative=True)
    # X (least bad) then Z; Y (worst) dropped by the top_n=2 cut.
    assert result == ["X", "Z"]


# ---------------------------------------------------------------------------
# 9. Exactly-zero return under default filter -> filtered out.
# ---------------------------------------------------------------------------


def test_exactly_zero_return_is_not_positive():
    """A return of exactly 0.0 is NOT positive (strict > 0.0) and is filtered out."""
    # SPY is positive; FLAT is exactly zero (the strict-> 0.0 boundary case).
    returns = {"SPY": 0.10, "FLAT": 0.0}
    # Default filter uses strict > 0.0, so FLAT (== 0.0) does not qualify.
    result = rank_by_trailing_return(returns, top_n=2)
    # Only SPY selected; FLAT excluded despite top_n=2 having room for it.
    assert result == ["SPY"]


# ---------------------------------------------------------------------------
# 10. NaN handling: never selectable in EITHER mode.
# ---------------------------------------------------------------------------


def test_nan_return_never_selected_in_either_mode():
    """A NaN-valued symbol is treated as ineligible in both filter modes."""
    # BAD has no valid return (NaN); SPY is a real positive; DOWN is a real negative.
    returns = {"SPY": 0.10, "BAD": float("nan"), "DOWN": -0.05}

    # Default mode: NaN dropped up front, negative dropped by the absolute filter,
    # leaving only SPY.
    default_result = rank_by_trailing_return(returns, top_n=3)
    assert default_result == ["SPY"]

    # Hold mode: no absolute filter, so DOWN is now rankable — but BAD (NaN) must
    # STILL be dropped up front, so the result is SPY then DOWN and never BAD.
    hold_result = rank_by_trailing_return(returns, top_n=3, hold_when_all_negative=True)
    assert hold_result == ["SPY", "DOWN"]
