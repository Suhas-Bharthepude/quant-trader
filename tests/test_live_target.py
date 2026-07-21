# tests/test_live_target.py

"""Tests for the pure live-target resolver src/execution/live_target.py."""

# dataclasses.FrozenInstanceError is raised when code tries to mutate a frozen
# dataclass; the CONFIG PINNED test asserts LIVE_CONFIG is immutable via it.
import dataclasses

# datetime + timezone build the tz-aware UTC "today" instants and month-end dates;
# tz-aware matches the bar timestamps (a naive today would raise on comparison).
from datetime import datetime, timezone

# pytest.raises drives the FrozenInstanceError and missing-reference ValueError checks.
import pytest

# OHLCVBar is the bar schema the synthetic fixture builds.
from src.data.schema import OHLCVBar

# The unit under test: the frozen config type, the shipped singleton, and the resolver.
from src.execution.live_target import (
    LiveRotationConfig,
    LIVE_CONFIG,
    live_target_weights,
)

# single_date_weights is the shared seam live_target_weights delegates to; the
# DELEGATION EQUIVALENCE test calls it DIRECTLY and asserts byte-identical output,
# proving live_target_weights is pure delegation with no second logic path.
from src.research.cross_sectional import single_date_weights


# ---------------------------------------------------------------------------
# Synthetic bar helper — copied verbatim from tests/test_portfolio.py so this
# test stays hermetic (no DuckDB): one OHLCVBar per (date, close), with
# close == adj_close so the default price_field="close" reads the intended value.
# ---------------------------------------------------------------------------


def make_month_end_bars(
    dates: list[datetime],
    closes: list[float],
    symbol: str = "T",
) -> list[OHLCVBar]:
    """Build one OHLCVBar per (date, close), in the given (ascending) order."""
    # Zip pairs each date with its close; one bar apiece, in the given order.
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
# Shared deterministic fixture: 8 month-ends (day 28 of Jan..Aug 2024, so each
# bar is its own calendar month => a clean month-end), three symbols with
# distinct, crossing price paths so the top-1 selection actually differentiates.
# ---------------------------------------------------------------------------


# One bar per month on day 28 (valid in every month), Jan..Aug 2024, tz-aware UTC.
_DATES = [
    datetime(2024, 1, 28, tzinfo=timezone.utc),  # month-end index 0
    datetime(2024, 2, 28, tzinfo=timezone.utc),  # index 1
    datetime(2024, 3, 28, tzinfo=timezone.utc),  # index 2
    datetime(2024, 4, 28, tzinfo=timezone.utc),  # index 3
    datetime(2024, 5, 28, tzinfo=timezone.utc),  # index 4
    datetime(2024, 6, 28, tzinfo=timezone.utc),  # index 5
    datetime(2024, 7, 28, tzinfo=timezone.utc),  # index 6 (Jul month-end)
    datetime(2024, 8, 28, tzinfo=timezone.utc),  # index 7 (Aug month-end, last bar)
]

# Per-symbol closes; values are irrelevant to date resolution but make the ranking
# non-degenerate so top-1 picks a definite winner.
_CLOSES = {
    "SPY": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0],  # reference spine
    "A": [100.0, 130.0, 90.0, 140.0, 95.0, 150.0, 100.0, 160.0],      # volatile
    "B": [100.0, 101.0, 150.0, 102.0, 160.0, 103.0, 170.0, 104.0],    # different phase
}


def _bars() -> dict[str, list[OHLCVBar]]:
    """Build the shared bars_by_symbol fixture (fresh dict per test)."""
    # One OHLCVBar list per symbol over the shared monthly date grid.
    return {sym: make_month_end_bars(_DATES, closes, sym) for sym, closes in _CLOSES.items()}


# ---------------------------------------------------------------------------
# 1. CONFIG PINNED — the shipped config is the exact Day-55 tuple and is frozen.
# ---------------------------------------------------------------------------


def test_live_config_is_pinned_and_frozen():
    """LIVE_CONFIG == the Day-55 tuple, and it cannot be mutated (frozen)."""
    # The shipped singleton must be exactly the designated validated config.
    assert LIVE_CONFIG == LiveRotationConfig(
        lookback=3, top_n=1, price_field="close", hold_when_all_negative=False
    )
    # Mutating any field of a frozen dataclass raises FrozenInstanceError, so the
    # shipped parameters cannot silently change at runtime.
    with pytest.raises(dataclasses.FrozenInstanceError):
        LIVE_CONFIG.lookback = 5


# ---------------------------------------------------------------------------
# 2. DELEGATION EQUIVALENCE — anti-drift proof: identical to single_date_weights.
# ---------------------------------------------------------------------------


def test_delegation_matches_single_date_weights_at_resolved_date():
    """A today after the last bar resolves to Aug 28; result == single_date_weights there."""
    # Fresh fixture.
    bars = _bars()
    # "Today" is in September, a month AFTER the last bar (Aug 28), so every month-end
    # is completed and the resolver returns the last one, Aug 28.
    today = datetime(2024, 9, 15, tzinfo=timezone.utc)
    # The known completed decision date the resolver should land on.
    d = datetime(2024, 8, 28, tzinfo=timezone.utc)
    # live_target_weights must equal a DIRECT call to the shared seam at that date,
    # with the LIVE_CONFIG parameters spelled out — proving pure delegation.
    assert live_target_weights(bars, today, reference_symbol="SPY") == single_date_weights(
        bars, d, 3, 1, "close", False
    )


# ---------------------------------------------------------------------------
# 3. MID-MONTH TODAY — a partial current month resolves through the PRIOR month-end.
# ---------------------------------------------------------------------------


def test_mid_month_today_uses_prior_completed_month_end():
    """Mid-August today: August is partial, so weights match single_date_weights at Jul 28."""
    # Fresh fixture.
    bars = _bars()
    # "Today" is mid-August; the Aug 28 bar is in the future relative to it, so August
    # is an in-progress (partial) month with no completed month-end yet.
    today = datetime(2024, 8, 15, tzinfo=timezone.utc)
    # The most recent COMPLETED month-end at-or-before mid-August is Jul 28.
    d_prior = datetime(2024, 7, 28, tzinfo=timezone.utc)
    # The resolver must fall back to the prior completed month-end, NOT the partial
    # current month, so the weights equal single_date_weights AT Jul 28.
    assert live_target_weights(bars, today, reference_symbol="SPY") == single_date_weights(
        bars, d_prior, 3, 1, "close", False
    )


# ---------------------------------------------------------------------------
# 4. NO COMPLETED MONTH-END YET — today before any completed month-end -> {}.
# ---------------------------------------------------------------------------


def test_no_completed_month_end_returns_empty():
    """Today before the first bar: no completed month-end exists -> all-cash {}."""
    # Fresh fixture.
    bars = _bars()
    # "Today" precedes the first bar (2024-01-28), so the visible slice is empty and no
    # month-end can be resolved.
    today = datetime(2024, 1, 15, tzinfo=timezone.utc)
    # With no completed month-end, the target is all-cash: an empty dict (which
    # reconcile_to_target reads as "hold nothing / exit everything to zero").
    assert live_target_weights(bars, today, reference_symbol="SPY") == {}


# ---------------------------------------------------------------------------
# 5. SHAPE CONTRACT — dict[str, float], each value 1/top_n, sum <= 1, absent = unheld.
# ---------------------------------------------------------------------------


def test_shape_contract_matches_reconcile_target_weights():
    """The non-empty result is a dict[str,float] with 1/top_n weights summing to <= 1.0."""
    # Fresh fixture.
    bars = _bars()
    # A today that resolves to Aug 28, where the top-1 winner has a positive trailing
    # return, so the target holds exactly one symbol.
    today = datetime(2024, 9, 15, tzinfo=timezone.utc)
    # Resolve the target weights.
    result = live_target_weights(bars, today, reference_symbol="SPY")
    # It must be a plain dict.
    assert isinstance(result, dict)
    # Keys are symbol strings; values are plain floats (reconcile_to_target's contract).
    assert all(isinstance(sym, str) for sym in result)
    assert all(isinstance(w, float) for w in result.values())
    # top_n == 1, so at most one symbol is held and each held weight is exactly 1/top_n.
    assert len(result) <= LIVE_CONFIG.top_n
    assert all(w == 1.0 / LIVE_CONFIG.top_n for w in result.values())
    # Weights never over-allocate: the sum is <= 1.0 (remainder is implicit cash).
    assert sum(result.values()) <= 1.0 + 1e-9
    # Absent symbols are simply not held (exit-to-zero); the held set is a subset of the
    # basket, and any basket symbol not present is treated as weight 0 downstream.
    assert set(result).issubset(set(bars))


# ---------------------------------------------------------------------------
# 6. MISSING REFERENCE — reference_symbol absent -> ValueError.
# ---------------------------------------------------------------------------


def test_missing_reference_symbol_raises():
    """A reference_symbol not in bars_by_symbol raises ValueError (its month-ends are the schedule)."""
    # Fresh fixture.
    bars = _bars()
    # Any valid today; the guard fires before date resolution.
    today = datetime(2024, 9, 15, tzinfo=timezone.utc)
    # An absent reference spine must fail loud rather than silently return {}.
    with pytest.raises(ValueError):
        live_target_weights(bars, today, reference_symbol="NOPE")
