# tests/test_data_health.py

"""
Unit tests for the pure content-check helpers in scripts/data_health.py.

find_zero_volume_bars and find_extreme_jumps were written as pure functions —
no DB, no network, no argparse — so they can be verified with hand-built bar
series. A bar row here is (date, close, volume), matching what main() builds
from the read-only query before handing it to the helpers.
"""

# date builds the bar timestamps; the helpers return dates, so we assert on dates.
from datetime import date

# pytest.approx compares the float ratios without exact-equality flakiness.
import pytest

# The functions under test.
from scripts.data_health import (
    find_extreme_jumps,
    find_zero_volume_bars,
    should_alert,
)


# ---------------------------------------------------------------------------
# find_zero_volume_bars
# ---------------------------------------------------------------------------

def test_find_zero_volume_bars_returns_only_zero_volume_dates() -> None:
    """A series with two zero-volume bars returns exactly those two dates, in order."""
    bars: list[tuple[date, float, int]] = [
        (date(2020, 1, 2), 100.0, 1_000),  # normal volume — not flagged
        (date(2020, 1, 3), 101.0, 0),      # zero volume — flagged
        (date(2020, 1, 6), 102.0, 1_500),  # normal volume — not flagged
        (date(2020, 1, 7), 103.0, 0),      # zero volume — flagged
    ]

    assert find_zero_volume_bars(bars) == [date(2020, 1, 3), date(2020, 1, 7)]


def test_find_zero_volume_bars_clean_series_returns_empty() -> None:
    """A series where every bar has positive volume returns an empty list."""
    bars: list[tuple[date, float, int]] = [
        (date(2020, 1, 2), 100.0, 1_000),
        (date(2020, 1, 3), 101.0, 1_200),
        (date(2020, 1, 6), 102.0, 900),
    ]

    assert find_zero_volume_bars(bars) == []


# ---------------------------------------------------------------------------
# find_extreme_jumps
# ---------------------------------------------------------------------------

def test_find_extreme_jumps_flags_both_directions_with_correct_fields() -> None:
    """A +296% spike and a 4:1 split (ratio 0.25) are both flagged with full detail."""
    bars: list[tuple[date, float, int]] = [
        (date(2020, 1, 2), 100.0, 1_000),  # baseline
        (date(2020, 1, 3), 101.0, 1_200),  # 101/100 = 1.01 — inside band, not flagged
        (date(2020, 1, 6), 400.0, 1_500),  # 400/101 ≈ 3.96 — upward jump, flagged
        (date(2020, 1, 7), 100.0, 1_400),  # 100/400 = 0.25 — split-sized drop, flagged
    ]

    flags = find_extreme_jumps(bars)

    # Exactly two flags, in chronological order.
    assert len(flags) == 2

    # First flag: the 101 -> 400 upward jump.
    up_date, up_prior, up_current, up_ratio = flags[0]
    assert up_date == date(2020, 1, 6)
    assert up_prior == 101.0
    assert up_current == 400.0
    assert up_ratio == pytest.approx(400.0 / 101.0)  # ≈ 3.9604

    # Second flag: the 400 -> 100 split-sized drop.
    down_date, down_prior, down_current, down_ratio = flags[1]
    assert down_date == date(2020, 1, 7)
    assert down_prior == 400.0
    assert down_current == 100.0
    assert down_ratio == pytest.approx(0.25)


def test_find_extreme_jumps_clean_series_returns_empty() -> None:
    """Modest day-to-day moves all sit inside [0.5, 2.0] and produce no flags."""
    bars: list[tuple[date, float, int]] = [
        (date(2020, 1, 2), 100.0, 1_000),
        (date(2020, 1, 3), 103.0, 1_200),  # +3%
        (date(2020, 1, 6), 99.0, 900),     # -3.9%
        (date(2020, 1, 7), 101.0, 1_100),  # +2%
    ]

    assert find_extreme_jumps(bars) == []


def test_find_extreme_jumps_skips_non_positive_prior_close() -> None:
    """A pair whose prior close is <= 0 is skipped — no flag, no ZeroDivisionError."""
    bars: list[tuple[date, float, int]] = [
        (date(2020, 1, 2), 0.0, 1_000),    # prior close 0 — pair must be skipped
        (date(2020, 1, 3), 100.0, 1_200),  # would be a huge ratio if not guarded
        (date(2020, 1, 6), 101.0, 1_500),  # 101/100 = 1.01 — inside band
    ]

    # The 0.0 -> 100.0 pair is skipped entirely; the 100 -> 101 pair is clean,
    # so the result is empty and nothing raises.
    assert find_extreme_jumps(bars) == []


def test_find_extreme_jumps_flags_drop_to_zero() -> None:
    """A close dropping to 0.0 has ratio 0.0 (< 0.5) and is flagged exactly once."""
    bars: list[tuple[date, float, int]] = [
        (date(2020, 1, 2), 100.0, 1_000),  # positive prior close — guard passes
        (date(2020, 1, 3), 0.0, 1_000),    # ratio 0/100 = 0.0 — flagged
    ]

    flags = find_extreme_jumps(bars)

    assert len(flags) == 1
    jump_date, prior, current, ratio = flags[0]
    assert jump_date == date(2020, 1, 3)
    assert prior == 100.0
    assert current == 0.0
    assert ratio == pytest.approx(0.0)


def test_find_extreme_jumps_boundary_is_strict() -> None:
    """ratio == 2.0 must NOT flag (inclusive band); ratio just above 2.0 must flag."""
    # Exactly on the upper bound: 200/100 == 2.0 sits inside [0.5, 2.0] → no flag.
    on_boundary: list[tuple[date, float, int]] = [
        (date(2020, 1, 2), 100.0, 1_000),
        (date(2020, 1, 3), 200.0, 1_000),
    ]
    assert find_extreme_jumps(on_boundary) == []

    # Just past the bound: 201/100 == 2.01 > 2.0 → flagged. Pins the strict >.
    past_boundary: list[tuple[date, float, int]] = [
        (date(2020, 1, 2), 100.0, 1_000),
        (date(2020, 1, 3), 201.0, 1_000),
    ]
    flags = find_extreme_jumps(past_boundary)
    assert len(flags) == 1
    assert flags[0][0] == date(2020, 1, 3)
    assert flags[0][3] == pytest.approx(2.01)

    # Exactly on the lower bound: 100/200 == 0.5 → no flag.
    on_lower: list[tuple[date, float, int]] = [
        (date(2020, 1, 2), 200.0, 1_000),
        (date(2020, 1, 3), 100.0, 1_000),
    ]
    assert find_extreme_jumps(on_lower) == []

    # Just past it: 99/200 == 0.495 < 0.5 → flagged. Pins the strict <.
    past_lower: list[tuple[date, float, int]] = [
        (date(2020, 1, 2), 200.0, 1_000),
        (date(2020, 1, 3), 99.0, 1_000),
    ]
    low_flags = find_extreme_jumps(past_lower)
    assert len(low_flags) == 1
    assert low_flags[0][3] == pytest.approx(0.495)


# ---------------------------------------------------------------------------
# should_alert
# ---------------------------------------------------------------------------

def test_should_alert_all_empty_is_false() -> None:
    """No stale, missing, or zero-volume findings → no alert (exit 0)."""
    assert should_alert([], [], []) is False


def test_should_alert_each_flag_individually_is_true() -> None:
    """Any one of stale / missing / zero-volume being non-empty triggers an alert."""
    # A single non-empty list in each position, others empty — proves each is
    # independently sufficient. The element values are irrelevant; only emptiness
    # matters, so a bare sentinel per list is enough.
    assert should_alert(["X"], [], []) is True   # stale only
    assert should_alert([], ["X"], []) is True   # missing only
    assert should_alert([], [], ["X"]) is True   # zero-volume only
