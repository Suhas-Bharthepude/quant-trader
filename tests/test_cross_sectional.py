# tests/test_cross_sectional.py

"""
Hermetic unit tests for src/research/cross_sectional.py.

All tests are pure: NO DuckDB, NO network, NO filesystem.  The rank_by_trailing_return
tests take a plain dict[str, float] of already-computed trailing returns and need NO
bars at all — they are just hand-built dicts and exact list-equality assertions.  The
trailing_returns_at tests DO build synthetic bars (one bar per month-end, hand-chosen
closes) so the as-of trailing returns are hand-computable.  Order matters in every
ranked result, so `==` on the whole list is the right check.  Matches the
comment-heavy style of tests/test_momentum_walkforward.py.

Run with:
    uv run pytest tests/test_cross_sectional.py -v
"""

# datetime + timezone build the explicit month-end timestamps for the synthetic
# bars and the rebalance_ts passed to trailing_returns_at.
from datetime import datetime, timezone

# pytest.raises is used for the top_n < 1 validation test; pytest.approx for the
# float trailing-return assertions.
import pytest

# OHLCVBar is the bar schema the trailing_returns_at tests build.
from src.data.schema import OHLCVBar

# The two units under test: the pure ranking/selection helper and the returns-
# computer that produces the dict the ranker consumes.
from src.research.cross_sectional import rank_by_trailing_return, trailing_returns_at


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


# ---------------------------------------------------------------------------
# Synthetic bar helper for the trailing_returns_at tests — one bar per month-end.
# ---------------------------------------------------------------------------


def make_month_end_bars(
    dates: list[datetime],
    closes: list[float],
    symbol: str = "T",
) -> list[OHLCVBar]:
    """Build one OHLCVBar per (date, close), each date in a distinct calendar month.

    Because every date is its own calendar month, month_end_indices treats EVERY
    bar as a month-end, so the trailing-return Series has exactly one entry per
    (date, close) pair — making the as-of returns hand-computable.  close=adj_close
    so the default price_field="close" reads the intended value and the bars are
    basis-agnostic.
    """
    # Zip pairs each date with its close; one bar apiece, in the given order (the
    # caller supplies ascending chronological dates, as everywhere else).
    return [
        OHLCVBar(
            symbol=symbol,        # arbitrary; the helper reads only timestamp + price
            timestamp=ts,         # distinct month per bar → each is a month-end
            open=c,               # dummy
            high=c,               # dummy
            low=c,                # dummy
            close=c,              # the field the default price_field reads
            adj_close=c,          # equal to close → basis-agnostic
            volume=1000,          # dummy
            timeframe="1d",       # daily
            source="test",        # provenance marker
        )
        for ts, c in zip(dates, closes)
    ]


# ---------------------------------------------------------------------------
# 11. trailing_returns_at — as-of returns on a shared month-end.
# ---------------------------------------------------------------------------


def test_trailing_returns_at_computes_as_of_returns():
    """Each symbol maps to its hand-computed trailing return (ratio-minus-1) as of D."""
    # Five shared month-ends (Jan–May 2024); lookback=2 so the May value is real.
    dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
        datetime(2024, 4, 30, tzinfo=timezone.utc),
        datetime(2024, 5, 31, tzinfo=timezone.utc),
    ]
    # Two symbols with distinct closes so their returns differ and are hand-checkable.
    bars_by_symbol = {
        "A": make_month_end_bars(dates, [100.0, 110.0, 120.0, 150.0, 200.0], "A"),
        "B": make_month_end_bars(dates, [50.0, 60.0, 55.0, 66.0, 80.0], "B"),
    }

    # Rebalance on the last shared month-end (May 31): the most recent month-end for
    # both symbols is index 4, whose lookback=2 return is close[4]/close[2] - 1.
    result = trailing_returns_at(bars_by_symbol, datetime(2024, 5, 31, tzinfo=timezone.utc), lookback=2)

    # Both symbols eligible; each maps to its own May-vs-March trailing return.
    assert set(result.keys()) == {"A", "B"}
    assert result["A"] == pytest.approx(200.0 / 120.0 - 1.0)
    assert result["B"] == pytest.approx(80.0 / 55.0 - 1.0)


# ---------------------------------------------------------------------------
# 12. trailing_returns_at — omit a too-short-history symbol (EMPTY guard).
# ---------------------------------------------------------------------------


def test_trailing_returns_at_omits_short_history_symbol():
    """A symbol whose first month-end is AFTER D is absent (empty-slice guard), good symbol present."""
    # GOOD has five month-ends ending May 31; SHORT's first month-end (Jun 30) is
    # after the rebalance date, so its at-or-before slice is empty.
    good_dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
        datetime(2024, 4, 30, tzinfo=timezone.utc),
        datetime(2024, 5, 31, tzinfo=timezone.utc),
    ]
    short_dates = [
        datetime(2024, 6, 30, tzinfo=timezone.utc),  # both AFTER the May-31 rebalance
        datetime(2024, 7, 31, tzinfo=timezone.utc),
    ]
    bars_by_symbol = {
        "GOOD": make_month_end_bars(good_dates, [100.0, 110.0, 120.0, 150.0, 200.0], "GOOD"),
        "SHORT": make_month_end_bars(short_dates, [10.0, 20.0], "SHORT"),
    }

    # As of May 31: SHORT has no month-end at-or-before D → empty slice → omitted.
    result = trailing_returns_at(bars_by_symbol, datetime(2024, 5, 31, tzinfo=timezone.utc), lookback=2)

    # SHORT is ABSENT (not present with NaN); GOOD is present with its real return.
    assert "SHORT" not in result
    assert "GOOD" in result
    assert result["GOOD"] == pytest.approx(200.0 / 120.0 - 1.0)


# ---------------------------------------------------------------------------
# 13. trailing_returns_at — omit a warmup-NaN symbol (NaN guard, distinct path).
# ---------------------------------------------------------------------------


def test_trailing_returns_at_omits_warmup_nan_symbol():
    """A symbol with month-ends at-or-before D but a NaN as-of value is omitted (NaN guard)."""
    # GOOD has five month-ends; WARMUP has only TWO month-ends at-or-before D, so
    # with lookback=2 BOTH are NaN (indices 0 and 1 are the warmup) — the most
    # recent one (May 31) is still NaN, so the slice is non-empty but iloc[-1] is NaN.
    good_dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
        datetime(2024, 4, 30, tzinfo=timezone.utc),
        datetime(2024, 5, 31, tzinfo=timezone.utc),
    ]
    warmup_dates = [
        datetime(2024, 4, 30, tzinfo=timezone.utc),  # index 0 → NaN (lookback=2)
        datetime(2024, 5, 31, tzinfo=timezone.utc),  # index 1 → NaN, and it IS the as-of bar
    ]
    bars_by_symbol = {
        "GOOD": make_month_end_bars(good_dates, [100.0, 110.0, 120.0, 150.0, 200.0], "GOOD"),
        "WARMUP": make_month_end_bars(warmup_dates, [70.0, 90.0], "WARMUP"),
    }

    # As of May 31: WARMUP's slice is non-empty (two month-ends) but iloc[-1] is NaN,
    # so the NaN guard — distinct from the empty guard — omits it.
    result = trailing_returns_at(bars_by_symbol, datetime(2024, 5, 31, tzinfo=timezone.utc), lookback=2)

    # WARMUP absent via the NaN path; GOOD present.
    assert "WARMUP" not in result
    assert "GOOD" in result


# ---------------------------------------------------------------------------
# 14. trailing_returns_at — rebalance before any month-end → empty dict, no crash.
# ---------------------------------------------------------------------------


def test_trailing_returns_at_rebalance_before_any_month_end_omits():
    """A rebalance date earlier than every symbol's first month-end yields {} (no IndexError)."""
    dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
    ]
    bars_by_symbol = {
        "A": make_month_end_bars(dates, [100.0, 110.0, 120.0], "A"),
        "B": make_month_end_bars(dates, [50.0, 60.0, 70.0], "B"),
    }

    # 2024-01-01 precedes every symbol's first month-end (Jan 31), so every slice is
    # empty — the empty-length guard must prevent the .iloc[-1] IndexError.
    result = trailing_returns_at(bars_by_symbol, datetime(2024, 1, 1, tzinfo=timezone.utc), lookback=2)

    # All symbols hit the empty guard → empty dict.
    assert result == {}


# ---------------------------------------------------------------------------
# 15. trailing_returns_at — each symbol resolves on its OWN month-end grid.
# ---------------------------------------------------------------------------


def test_trailing_returns_at_as_of_uses_own_grid():
    """At one rebalance_ts, each symbol uses ITS OWN most-recent month-end at-or-before D."""
    # A has a March month-end (Mar 31 == D); B has NO March bar, so B's most recent
    # month-end at-or-before Mar 31 is Feb 29.  Both have enough history to be real.
    a_dates = [
        datetime(2023, 10, 31, tzinfo=timezone.utc),
        datetime(2023, 11, 30, tzinfo=timezone.utc),
        datetime(2023, 12, 31, tzinfo=timezone.utc),
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),  # A's March month-end == D
    ]
    b_dates = [
        datetime(2023, 10, 31, tzinfo=timezone.utc),
        datetime(2023, 11, 30, tzinfo=timezone.utc),
        datetime(2023, 12, 31, tzinfo=timezone.utc),
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),  # B's most recent ≤ D (no March bar)
    ]
    bars_by_symbol = {
        "A": make_month_end_bars(a_dates, [10.0, 20.0, 30.0, 100.0, 110.0, 132.0], "A"),
        "B": make_month_end_bars(b_dates, [10.0, 20.0, 50.0, 40.0, 60.0], "B"),
    }

    # Rebalance Mar 31: right-inclusive, so A resolves to its Mar-31 month-end
    # (index 5) while B — having no March bar — resolves to its Feb-29 month-end
    # (index 4).  Each is a lookback=2 return on its OWN grid.
    result = trailing_returns_at(bars_by_symbol, datetime(2024, 3, 31, tzinfo=timezone.utc), lookback=2)

    # A: close[5]/close[3] - 1 = 132/100 - 1 = 0.32 (its March return).
    assert result["A"] == pytest.approx(132.0 / 100.0 - 1.0)
    # B: close[4]/close[2] - 1 = 60/50 - 1 = 0.20 (its February return — own grid).
    assert result["B"] == pytest.approx(60.0 / 50.0 - 1.0)


# ---------------------------------------------------------------------------
# 16. trailing_returns_at → rank_by_trailing_return end-to-end seam.
# ---------------------------------------------------------------------------


def test_trailing_returns_at_output_feeds_ranker():
    """The returns-computer's output dict is valid input to the ranker (end-to-end seam)."""
    dates = [
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
        datetime(2024, 4, 30, tzinfo=timezone.utc),
        datetime(2024, 5, 31, tzinfo=timezone.utc),
    ]
    # A's as-of return (200/120 - 1 ≈ 0.667) exceeds B's (80/55 - 1 ≈ 0.455).
    bars_by_symbol = {
        "A": make_month_end_bars(dates, [100.0, 110.0, 120.0, 150.0, 200.0], "A"),
        "B": make_month_end_bars(dates, [50.0, 60.0, 55.0, 66.0, 80.0], "B"),
    }

    # Compute the returns dict, then feed it straight into the ranker.
    returns = trailing_returns_at(bars_by_symbol, datetime(2024, 5, 31, tzinfo=timezone.utc), lookback=2)

    # top_n=1 selects the single strongest symbol — A, whose return is higher.
    assert rank_by_trailing_return(returns, top_n=1) == ["A"]
    # top_n=2 returns both, A first (highest), B second.
    assert rank_by_trailing_return(returns, top_n=2) == ["A", "B"]
