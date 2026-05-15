# tests/test_research_runner.py

"""
Pure unit tests for src/research/runner.py.

These tests build synthetic OHLCVBar lists in-memory and use small SMA
windows (5/10/20/30) so test data stays tiny.  No network, no DuckDB, no
yfinance — the same fast-and-deterministic pattern as tests/test_backtest.py.

Run with:
    uv run pytest tests/test_research_runner.py -v
"""

# numpy supplies the float64 array constructors used by the synthetic
# BacktestResult factory (equity_curve, returns).  Same dtype as the engine
# emits, so the dataclass accepts the inputs without coercion.
import numpy as np

# pandas is imported even though the tests rarely reach into it — compare()
# returns a DataFrame and we check shape/column names off it.  Keeping the
# import explicit documents the dependency.
import pandas as pd

# pytest is the test runner; pytest.raises is used to assert that the
# runner's validation paths emit the right exceptions.
import pytest

# datetime + timezone + timedelta build sequential UTC-aware timestamps for
# the synthetic bars.  timezone.utc matches the project-wide convention
# that every bar timestamp is timezone-aware UTC.
from datetime import datetime, timedelta, timezone

# OHLCVBar is the bar schema the runner forwards to the engine.  Importing
# from src.data.schema means a future schema rename surfaces here at import
# time rather than silently running against a stale shape.
from src.data.schema import OHLCVBar

# Concrete strategy class used in the run_many tests — small windows let
# us drive end-to-end behaviour with ~50-bar fixtures.
from src.strategies.sma_crossover import SMACrossoverStrategy

# BacktestRunner is the unit under test — every assertion in this file
# routes through one of its two methods (run_many, compare).
from src.research.runner import BacktestRunner

# BacktestResult is constructed directly in the compare() tests so we can
# hand-pick the metric values that drive the sort logic, bypassing the
# backtester entirely.
from src.backtest.result import BacktestResult


# ---------------------------------------------------------------------------
# Module-level helpers (intentionally plain functions, not pytest fixtures)
# ---------------------------------------------------------------------------


# make_bars mirrors the helper in tests/test_backtest.py: a plain function
# rather than a fixture because every test passes its own close list, and
# `make_bars([100.0 + i for i in range(50)])` reads clearer than parameterizing
# a fixture across test cases.
def make_bars(closes: list[float]) -> list[OHLCVBar]:
    """Build OHLCVBar list with given closes and sequential daily UTC timestamps."""

    # Anchor every series at 2024-01-01 UTC.  The runner never reads the
    # calendar values — it only forwards bars to the engine — but pinning
    # the date keeps the tests deterministic and easy to eyeball.
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # One bar per close.  enumerate gives the day offset; timestamps stay
    # strictly increasing so start_date < end_date on the resulting result.
    return [
        OHLCVBar(
            symbol="TEST",                         # arbitrary; runner does not read
            timestamp=base + timedelta(days=i),    # strictly increasing UTC
            open=c, high=c, low=c, close=c,        # only close matters downstream
            adj_close=c,                           # equal to close — no split adjustments
            volume=1000,                           # dummy positive volume
            timeframe="1d",                        # daily bars — matches engine assumption
            source="test",                         # marker so a stray bar in the DB would be obvious
        )
        for i, c in enumerate(closes)
    ]


# make_result builds a synthetic BacktestResult with caller-chosen metric
# values, bypassing the engine entirely.  This is the right tool for testing
# compare()'s sort/format logic in isolation: the engine has its own tests,
# and forcing compare() tests to go through a real backtest run would couple
# them to engine internals (equity curves, signal generation) that the sort
# logic doesn't care about.  Hand-picked metrics make the expected ordering
# self-evident in each test body.
def make_result(
    name: str,
    sharpe: float,
    total_return: float = 0.0,
    max_drawdown: float = 0.0,
    win_rate: float = 0.0,
    n_trades: int = 0,
) -> BacktestResult:
    """Construct a BacktestResult with caller-chosen metrics for compare() tests."""

    # BacktestResult is frozen — every field must be supplied at construction.
    # The time-series fields (equity_curve, returns) are required by the
    # dataclass but unused by compare(), so we hand it length-10 arrays of
    # ones/zeros: shape-valid placeholders that don't influence assertions.
    return BacktestResult(
        strategy_name=name,                                             # appears in df['strategy'] column
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),           # arbitrary; not read by compare
        end_date=datetime(2024, 1, 10, tzinfo=timezone.utc),            # arbitrary; not read by compare
        n_bars=10,                                                      # arbitrary; not read by compare
        equity_curve=np.ones(10, dtype=np.float64),                     # placeholder — compare ignores
        returns=np.zeros(10, dtype=np.float64),                         # placeholder — compare ignores
        trades=[],                                                      # placeholder — compare ignores
        total_return_pct=total_return,                                  # surfaces in df['total_return']
        sharpe_ratio=sharpe,                                            # surfaces in df['sharpe']
        max_drawdown_pct=max_drawdown,                                  # surfaces in df['max_drawdown']
        win_rate=win_rate,                                              # surfaces in df['win_rate']
        n_trades=n_trades,                                              # surfaces in df['n_trades']
    )


# ---------------------------------------------------------------------------
# Tests for run_many — validation and ordering contracts
# ---------------------------------------------------------------------------


def test_runner_validates_empty_bars():
    """run_many rejects an empty bars list before doing any work."""

    # SMACrossoverStrategy(5, 10) is a legal strategy — the failure must come
    # from the empty bars list, not from a bad strategy argument.
    # pytest.raises asserts the expected exception fires; if no exception
    # fires (or a different type fires) the test fails loudly.
    with pytest.raises(ValueError):
        BacktestRunner().run_many([], [SMACrossoverStrategy(5, 10)])


def test_runner_validates_empty_strategies():
    """run_many rejects an empty strategies list — there'd be nothing to run."""

    # 50 monotonically increasing closes — plenty of bars to make this a
    # legitimate test of the strategies-list validation, not a bars-too-short
    # failure.  range(50) keeps the literal short while still passing every
    # warmup window we use elsewhere in the file.
    bars = make_bars([100.0 + i for i in range(50)])

    # The runner should refuse rather than silently return an empty results list.
    with pytest.raises(ValueError):
        BacktestRunner().run_many(bars, [])


def test_runner_validates_strategy_instances():
    """run_many rejects non-Strategy entries in the strategies list."""

    # Same legitimate bars as above; the failure must be the type check, not
    # an upstream guard.
    bars = make_bars([100.0 + i for i in range(50)])

    # A plain string is not a Strategy — the isinstance check in run_many
    # must surface this as TypeError before any generate_signals call.
    with pytest.raises(TypeError):
        BacktestRunner().run_many(bars, ["not_a_strategy"])


def test_runner_returns_results_in_same_order():
    """Results from run_many appear in the same order as the input strategies."""

    # Monotonic-up closes mean every SMA crossover will eventually go long
    # and stay long — but the actual signals don't matter for this test.
    # What matters is the *order* of the returned results list.
    bars = make_bars([100.0 + i for i in range(50)])

    # Three distinct configurations so we can match strategy_name back to
    # the input position by inspection.  SMACrossoverStrategy.name returns
    # f"SMA({fast}, {slow})" — the exact strings used in the assertions below.
    strategies = [
        SMACrossoverStrategy(5, 10),    # expected at results[0]
        SMACrossoverStrategy(5, 20),    # expected at results[1]
        SMACrossoverStrategy(10, 30),   # expected at results[2]
    ]

    # Run them all through a fresh runner — default Backtester is fine
    # because we're testing order, not metric correctness.
    results = BacktestRunner().run_many(bars, strategies)

    # Three strategies in → three results out; any other length would
    # indicate the runner silently dropped or duplicated work.
    assert len(results) == 3

    # Strategy names appear in input order; the f-string format must match
    # SMACrossoverStrategy.name exactly (including the space after the comma).
    assert results[0].strategy_name == "SMA(5, 10)"
    assert results[1].strategy_name == "SMA(5, 20)"
    assert results[2].strategy_name == "SMA(10, 30)"


def test_runner_results_are_backtest_results():
    """Every element returned by run_many is a BacktestResult instance."""

    # Minimal setup — one strategy, enough bars to warm up its 10-bar window.
    bars = make_bars([100.0 + i for i in range(50)])
    strategies = [SMACrossoverStrategy(5, 10)]

    # Run and unwrap.
    results = BacktestRunner().run_many(bars, strategies)

    # isinstance over every element so a partial corruption (e.g. one stray
    # tuple) would still fail.  all() short-circuits on first mismatch.
    assert all(isinstance(r, BacktestResult) for r in results)


# ---------------------------------------------------------------------------
# Tests for compare — DataFrame shape, sort behaviour, validation
# ---------------------------------------------------------------------------


def test_compare_returns_dataframe_with_expected_columns():
    """compare() produces a DataFrame with the documented schema."""

    # Two synthetic results with distinct names so the row count is
    # unambiguous; the metric values don't matter for the schema check.
    results = [
        make_result("A", sharpe=1.0),
        make_result("B", sharpe=2.0),
    ]

    # Default sort (sharpe descending) — schema is independent of order.
    df = BacktestRunner.compare(results)

    # The six documented columns, as a set so column order doesn't matter
    # for this assertion.  A future column addition would require updating
    # this set, which is the right kind of test friction.
    expected = {"strategy", "total_return", "sharpe", "max_drawdown", "win_rate", "n_trades"}
    assert set(df.columns) == expected

    # Two results → two rows.  Catches any silent dedup or filtering.
    assert len(df) == 2


def test_compare_sorts_by_sharpe_descending_by_default():
    """Default sort puts the highest Sharpe at the top."""

    # Two results with hand-picked Sharpe values — B beats A.
    result_a = make_result("A", sharpe=0.5)
    result_b = make_result("B", sharpe=1.5)

    # Default call: no sort_by argument → sharpe; no ascending → False.
    df = BacktestRunner.compare([result_a, result_b])

    # Best Sharpe first → B at iloc[0], A at iloc[1].  iloc is integer
    # positional indexing — robust against the runner having reset the index.
    assert df.iloc[0]["strategy"] == "B"
    assert df.iloc[1]["strategy"] == "A"


def test_compare_sorts_ascending_when_requested():
    """ascending=True inverts the order — useful for max_drawdown."""

    # Same two results as the previous test; only the sort direction differs.
    result_a = make_result("A", sharpe=0.5)
    result_b = make_result("B", sharpe=1.5)

    # Pass ascending=True explicitly — Sharpe ascending means lowest first.
    df = BacktestRunner.compare([result_a, result_b], ascending=True)

    # Lower Sharpe at iloc[0] → A first, B second.  This is the same data
    # as the descending test but reversed, which is exactly the property
    # we want to verify.
    assert df.iloc[0]["strategy"] == "A"
    assert df.iloc[1]["strategy"] == "B"


def test_compare_validates_sort_by():
    """compare() rejects sort_by values outside the whitelist."""

    # One result is enough — the whitelist check fires before any sort work.
    results = [make_result("A", sharpe=0.5)]

    # 'nonsense' is not one of the five whitelisted keys → ValueError.
    with pytest.raises(ValueError):
        BacktestRunner.compare(results, sort_by="nonsense")


def test_compare_validates_empty_results():
    """compare() rejects an empty results list rather than returning an empty frame."""

    # No results → ValueError; silently returning an empty frame would
    # mask a caller bug (e.g. forgetting to capture run_many's output).
    with pytest.raises(ValueError):
        BacktestRunner.compare([])
