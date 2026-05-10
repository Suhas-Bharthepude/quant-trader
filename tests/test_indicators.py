# tests/test_indicators.py

"""
Pure unit tests for src/features/indicators.py.

These tests build synthetic OHLCVBar lists in-memory — no network calls, no
DuckDB, no yfinance — so they are fast and deterministic.  Run with:

    uv run pytest tests/test_indicators.py -v

The synthetic-bar approach lets each test pin down a single behaviour
(constant series, monotone-up, monotone-down, canonical Wilder example)
without dragging real market data into the assertion budget.
"""

# datetime + timezone + timedelta produce sequential UTC-aware timestamps
# for the synthetic bars.  Using timezone.utc explicitly mirrors the
# project-wide convention that all bar timestamps are timezone-aware UTC.
from datetime import datetime, timedelta, timezone

# numpy is needed for NaN sentinel comparisons (np.isnan) and for the
# array-equality assertion helpers used throughout this file.
import numpy as np

# pytest is the test runner; we use pytest.raises to assert on validation
# failures from the indicator functions.
import pytest

# OHLCVBar is the bar schema the indicators consume.  Importing it from
# src.data.schema ensures these tests fail fast if the schema is renamed
# or restructured, rather than silently running against a stale shape.
from src.data.schema import OHLCVBar

# The three public indicators under test.  We import the concrete
# functions (not the module) so the call sites stay short and the
# missing-symbol failure mode is import-time rather than runtime.
from src.features.indicators import log_returns, rsi, sma


# ---------------------------------------------------------------------------
# Module-level helper (intentionally not a pytest fixture)
# ---------------------------------------------------------------------------


# A plain function — not a fixture — because every test wants different
# closes.  Fixtures shine when state is shared across tests; here each
# test passes its own list, so a helper keeps the call site explicit:
# `make_bars([1, 2, 3])` is more legible than juggling fixture params.
def make_bars(closes: list[float]) -> list[OHLCVBar]:
    """Build OHLCVBar list with given closes and sequential daily UTC timestamps."""

    # Anchor every test series at 2024-01-01 UTC.  The exact date does not
    # matter — none of the indicators look at calendar values — but pinning
    # it keeps tests deterministic and easy to eyeball when debugging.
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Build one bar per close price.  enumerate gives the day offset, which
    # is added to `base` so the timestamps stay strictly increasing — a
    # property all indicators implicitly assume.
    return [
        OHLCVBar(
            symbol="TEST",          # arbitrary; indicators do not read this
            timestamp=base + timedelta(days=i),  # strictly increasing UTC
            open=c,                 # dummy: indicators only use `close`
            high=c,                 # dummy
            low=c,                  # dummy
            close=c,                # the only field the indicators read
            adj_close=c,            # dummy
            volume=1000,            # dummy
            timeframe="1d",         # daily; matches the timedelta above
            source="test",          # provenance marker for synthetic data
        )
        for i, c in enumerate(closes)
    ]


# ---------------------------------------------------------------------------
# SMA tests
# ---------------------------------------------------------------------------


def test_sma_constant_series_returns_constant():
    """A flat price series should yield a flat SMA (after warm-up)."""

    # Ten bars all priced at 100.0 — any moving average over a constant
    # series must equal that constant once the window is full.
    bars = make_bars([100.0] * 10)

    # Three-bar window means the first two outputs are NaN (insufficient
    # lookback) and positions 2..9 should all equal 100.0.
    out = sma(bars, window=3)

    # First two positions are warm-up — assert they are NaN explicitly.
    # `np.isnan` returns a bool array; `.all()` collapses to a single bool.
    assert np.isnan(out[:2]).all()

    # All remaining positions should equal 100.0 to within float tolerance.
    # decimal=4 is more than enough precision for an integer constant.
    np.testing.assert_array_almost_equal(out[2:], [100.0] * 8, decimal=4)


def test_sma_known_values():
    """Hand-computed SMA values for an arithmetic series."""

    # Closes = 1, 2, 3, 4, 5.  Mean of any three consecutive integers is
    # the middle one, so SMA(3) is [_, _, 2, 3, 4] after the two-NaN warm-up.
    bars = make_bars([1.0, 2.0, 3.0, 4.0, 5.0])

    # Compute the indicator under test.
    out = sma(bars, window=3)

    # Warm-up region: first two values are NaN by contract.
    assert np.isnan(out[:2]).all()

    # The remaining three values should match the hand-computed means.
    # Splitting the warm-up assertion from the value assertion is cleaner
    # than building a single mixed array with NaN sentinels.
    np.testing.assert_array_almost_equal(out[2:], [2.0, 3.0, 4.0], decimal=4)


def test_sma_raises_on_invalid_window():
    """SMA should reject window < 1 and window > len(bars)."""

    # Three bars — gives us a finite ceiling for the over-large-window check.
    bars = make_bars([1.0, 2.0, 3.0])

    # window=0 is mathematically meaningless; the function must raise.
    # `pytest.raises` as a context manager fails the test if the body
    # exits without raising the expected exception type.
    with pytest.raises(ValueError):
        sma(bars, window=0)

    # window=4 exceeds len(bars)=3, which would produce an all-NaN array;
    # the indicator rejects it instead of silently returning useless data.
    with pytest.raises(ValueError):
        sma(bars, window=4)


def test_sma_raises_on_empty_bars():
    """SMA should reject an empty bar list outright."""

    # Empty input has no defined moving average — the function raises
    # ValueError instead of returning a zero-length array (which would
    # mask upstream data-fetch bugs).
    with pytest.raises(ValueError):
        sma([], window=3)


# ---------------------------------------------------------------------------
# log_returns tests
# ---------------------------------------------------------------------------


def test_log_returns_known_values():
    """Hand-computed log returns for a small series."""

    # 100 → 110 → 99: roughly +10% then ~-10% in log space.
    bars = make_bars([100.0, 110.0, 99.0])

    # Compute the indicator under test.
    out = log_returns(bars)

    # First value is NaN — there is no prior bar to compare against.
    assert np.isnan(out[0])

    # Hand-computed expected values: ln(110/100) ≈ 0.09531, ln(99/110) ≈ -0.10536.
    expected = [np.log(110 / 100), np.log(99 / 110)]

    # decimal=4 tolerates the tiny float-rounding wobble between numpy's
    # internal log and the freshly-computed expected values.
    np.testing.assert_array_almost_equal(out[1:], expected, decimal=4)


def test_log_returns_constant_series_is_zero():
    """A flat price series should produce zero log returns after the leading NaN."""

    # Four identical closes → ratio is 1 every step → log(1) = 0.
    bars = make_bars([100.0, 100.0, 100.0, 100.0])

    # Compute the indicator under test.
    out = log_returns(bars)

    # The leading NaN is by contract (no prior bar).
    assert np.isnan(out[0])

    # Every subsequent return must be exactly zero — log(1) is exact in
    # IEEE 754, so a strict equality would also work; we use almost_equal
    # for symmetry with the rest of the file.
    np.testing.assert_array_almost_equal(out[1:], [0.0, 0.0, 0.0], decimal=4)


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------


def test_rsi_all_gains_returns_100():
    """Monotonically rising prices should pin RSI at 100 after warm-up."""

    # 21 strictly increasing closes — every delta is positive, so avg_loss
    # in Wilder's smoothing is identically zero and the RSI formula
    # collapses to its avg_loss==0 branch (RSI = 100, max strength).
    bars = make_bars([float(10 + i) for i in range(21)])

    # Compute the indicator under test with the canonical 14-bar period.
    out = rsi(bars, period=14)

    # First 14 values are warm-up NaN by contract (period values masked).
    assert np.isnan(out[:14]).all()

    # The remaining 7 values must be exactly 100.0 — the np.where branch
    # writes the literal 100.0, so strict equality is appropriate here.
    np.testing.assert_array_almost_equal(out[14:], [100.0] * 7, decimal=4)


def test_rsi_all_losses_returns_0():
    """Monotonically falling prices should pin RSI at 0 after warm-up."""

    # 21 strictly decreasing closes — every delta is negative, so
    # avg_gain == 0 in Wilder's smoothing.  RS = 0 / avg_loss = 0, and
    # RSI = 100 - 100/(1+0) = 0.
    bars = make_bars([float(30 - i) for i in range(21)])

    # Compute the indicator under test.
    out = rsi(bars, period=14)

    # Warm-up: first 14 values NaN.
    assert np.isnan(out[:14]).all()

    # All emitted values should equal 0.0 to float tolerance.
    np.testing.assert_array_almost_equal(out[14:], [0.0] * 7, decimal=4)


def test_rsi_known_value():
    """RSI(14) on the canonical Wilder/StockCharts reference series."""

    # 21-bar reference series widely published as the "Wilder example".
    # The expected RSI on the final bar (~62.78) is the value reported by
    # TradingView, ta-lib, StockCharts, and MetaTrader for this exact input
    # — so this test simultaneously locks in agreement with all four.
    closes = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
        46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
        46.22, 45.64, 46.21,
    ]

    # Build bars from the reference closes.
    bars = make_bars(closes)

    # Compute the indicator under test.
    out = rsi(bars, period=14)

    # Tolerance of 0.5 absorbs the tiny rounding drift between the
    # reference (the closes are quoted to 2 decimal places, while the
    # original publication carries them to 4) and our double-precision
    # computation; agreement is in fact ~0.1.
    assert abs(out[-1] - 62.78) < 0.5


# ---------------------------------------------------------------------------
# Cross-indicator length contract
# ---------------------------------------------------------------------------


def test_indicators_align_with_input_length():
    """Every indicator must return an array the same length as the input."""

    # 50 flat bars — value content is irrelevant, only the length matters.
    bars = make_bars([1.0] * 50)

    # SMA: one output per input bar.  Validates the alignment promise that
    # callers rely on when zipping indicator arrays back to bars.
    assert len(sma(bars, 10)) == 50

    # RSI: same alignment promise; the warm-up region is NaN, not absent.
    assert len(rsi(bars, 14)) == 50

    # log_returns: leading NaN, but still one output per input bar.
    assert len(log_returns(bars)) == 50
