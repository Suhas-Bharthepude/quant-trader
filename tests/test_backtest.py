# tests/test_backtest.py

"""
Pure unit tests for src/backtest/engine.py and src/backtest/result.py.

These tests build synthetic OHLCVBar lists in-memory — no network calls,
no DuckDB, no yfinance — so they are fast and deterministic.  Run with:

    uv run pytest tests/test_backtest.py -v

The synthetic-bar approach lets each test pin down a single behaviour
(validation, no-lookahead, drawdown math, immutability) without dragging
real market data into the assertion budget.  Same pattern as
tests/test_strategies.py and tests/test_indicators.py.
"""

# numpy supplies the int8 dtype for signal arrays and the array constructors
# used to hand-craft tiny signal sequences for each test case.
import numpy as np

# pytest is the test runner; we use pytest.raises to assert on the
# validation failures emitted by the Backtester constructor and run() method.
import pytest

# dataclasses provides FrozenInstanceError — the specific exception that
# frozen dataclasses raise on attribute assignment.  We assert on it in the
# immutability test so a future change away from frozen=True fails loudly.
import dataclasses

# datetime + timezone + timedelta produce sequential UTC-aware timestamps
# for the synthetic bars.  timezone.utc matches the project-wide convention
# that every bar timestamp is timezone-aware UTC.
from datetime import datetime, timedelta, timezone

# OHLCVBar is the bar schema the backtester consumes.  Importing from
# src.data.schema means a future schema rename surfaces here at import
# time rather than silently running against a stale shape.
from src.data.schema import OHLCVBar

# The three legal signal values, imported as named constants so the test
# bodies read intent-first ("SIGNAL_LONG") rather than magic-number-first ("1").
from src.strategies.base import SIGNAL_LONG, SIGNAL_FLAT, SIGNAL_SHORT

# The Backtester class under test — entry point for every behaviour assertion.
from src.backtest.engine import Backtester

# Trade and BacktestResult are the immutable result dataclasses we read off
# the Backtester output; importing both lets the type names appear in the
# test bodies without fully-qualified prefixes.
from src.backtest.result import Trade, BacktestResult


# ---------------------------------------------------------------------------
# Module-level helper (intentionally not a pytest fixture)
# ---------------------------------------------------------------------------


# A plain function — not a fixture — because every test wants different
# closes.  Fixtures shine when state is shared across tests; here each test
# passes its own list, so a helper keeps the call site explicit:
# `make_bars([100, 110])` reads cleaner than juggling fixture params.
def make_bars(closes: list[float]) -> list[OHLCVBar]:
    """Build OHLCVBar list with given closes and sequential daily UTC timestamps starting 2024-01-01."""

    # Anchor every test series at 2024-01-01 UTC.  The exact date does not
    # matter — the backtester never reads calendar values — but pinning it
    # keeps tests deterministic and easy to eyeball when debugging.
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Build one bar per close price.  enumerate gives the day offset, added
    # to `base` so timestamps stay strictly increasing — a property the
    # backtester relies on for start_date / end_date assignment.
    return [
        OHLCVBar(
            symbol="TEST",          # arbitrary; backtester does not read this
            timestamp=base + timedelta(days=i),  # strictly increasing UTC
            open=c,                 # dummy: engine only reads `close`
            high=c,                 # dummy
            low=c,                  # dummy
            close=c,                # the only price field the engine reads
            adj_close=c,            # dummy
            volume=1000,            # dummy
            timeframe="1d",         # daily; matches the timedelta above
            source="test",          # provenance marker for synthetic data
        )
        for i, c in enumerate(closes)
    ]


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_backtester_validates_initial_capital():
    """initial_capital must be strictly positive — zero or negative is rejected."""

    # Zero capital would make any percentage-return math undefined (divide
    # by zero in total_return_pct) — the constructor must reject it before
    # run() is ever called.
    with pytest.raises(ValueError):
        Backtester(initial_capital=0)

    # Negative capital is even more obviously nonsense; same guard catches
    # it.  Using a clearly negative value (-100) makes the test intent
    # obvious to a reader skimming the file.
    with pytest.raises(ValueError):
        Backtester(initial_capital=-100)


def test_backtester_validates_annualization_factor():
    """annualization_factor must be strictly positive — sqrt(0) and sqrt(neg) are nonsense."""

    # Zero would zero out the Sharpe scaling factor; negative would attempt
    # sqrt of a negative number.  Both are configuration errors that the
    # constructor catches up front rather than producing NaN deep in run().
    with pytest.raises(ValueError):
        Backtester(annualization_factor=0)

    # -1 is the canonical "obviously wrong" negative for this kind of guard.
    with pytest.raises(ValueError):
        Backtester(annualization_factor=-1)


# ---------------------------------------------------------------------------
# Input validation on run()
# ---------------------------------------------------------------------------


def test_backtester_validates_empty_bars():
    """Empty bars must raise — no data to backtest against."""

    # Empty input is virtually always a programming error upstream (e.g. a
    # symbol with no history).  Failing loudly here surfaces it immediately
    # rather than producing a zero-length equity curve that would later
    # trigger confusing IndexErrors when callers index into [-1].
    with pytest.raises(ValueError):
        # Empty signal array matched against empty bars — both sides empty
        # so the length check would pass; the explicit empty-bars check
        # is what catches this case.
        Backtester().run([], np.array([], dtype=np.int8))


def test_backtester_validates_signal_length_mismatch():
    """signals length must equal bars length — alignment contract."""

    # 5 bars but only 3 signals: the alignment contract from Strategy says
    # one signal per bar, so a mismatch is a programming error somewhere
    # upstream (likely a forgotten warmup pad).  Catch it here, not deep
    # in the math.
    bars = make_bars([100.0, 101.0, 102.0, 103.0, 104.0])

    # int8 dtype matches what Strategy implementations emit; the test isn't
    # about dtype, just the length mismatch, so we use the canonical dtype.
    signals = np.array([0, 1, 0], dtype=np.int8)

    # ValueError matches the constructor and the empty-bars guard — one
    # exception type for the whole "bad input" surface, so callers can
    # write a single except clause.
    with pytest.raises(ValueError):
        Backtester().run(bars, signals)


def test_backtester_validates_signal_values():
    """signals must contain only values in {-1, 0, 1} — anything else is a strategy bug."""

    # 4 bars, 4 signals — length is valid, dtype is valid, but the value 2
    # is illegal.  A stray 2 would silently double-scale the returns, which
    # is exactly the kind of bug the validation is designed to catch.
    bars = make_bars([100.0, 101.0, 102.0, 103.0])
    signals = np.array([0, 1, 2, 0], dtype=np.int8)

    # ValueError, same as the other validation cases — uniform exception
    # surface across all input-shape checks.
    with pytest.raises(ValueError):
        Backtester().run(bars, signals)


# ---------------------------------------------------------------------------
# Behaviour tests — known mathematical answers
# ---------------------------------------------------------------------------


def test_flat_signals_produce_zero_return():
    """All-FLAT signals → no exposure → zero return regardless of underlying volatility."""

    # A deliberately volatile underlying so the test would fail if the
    # engine accidentally accumulated asset returns even while flat.
    bars = make_bars([100.0, 110.0, 95.0, 105.0, 100.0])

    # All-FLAT signals: the strategy is never in a position, so no
    # underlying return can be captured.
    signals = np.zeros(5, dtype=np.int8)

    # Run with default capital (1.0) so the equity curve is a multiplier.
    result = Backtester().run(bars, signals)

    # Total return must be exactly 0 (within float tolerance).  Any non-zero
    # value here means the engine is leaking exposure on flat signals.
    assert abs(result.total_return_pct) < 1e-12

    # No flat→nonzero transitions in the signal array, so no trades should
    # have been recorded — round-trip extraction must respect "flat means flat."
    assert result.n_trades == 0

    # Equity curve should be a flat line at the initial capital (1.0).
    # Vectorised comparison: every element within 1e-12 of 1.0.
    assert np.all(np.abs(result.equity_curve - 1.0) < 1e-12)


def test_long_signal_captures_uptrend():
    """A long position over a 10% up move should earn ~+10% (log-return-based)."""

    # Two bars with a 10% close-to-close rise.  This is the smallest series
    # that exercises the next-bar execution: signal at index 0 is held over
    # bar 1, capturing the 100→110 move.
    bars = make_bars([100.0, 110.0])

    # Both signals long.  signals[0]=LONG is what actually matters under
    # next-bar execution — it earns the return from bar 0 to bar 1.
    # signals[1]=LONG has no return to apply to (no bar 2).
    signals = np.array([SIGNAL_LONG, SIGNAL_LONG], dtype=np.int8)

    result = Backtester().run(bars, signals)

    # Expected math: log(110/100) ≈ 0.0953 cumulative log return.
    # exp(0.0953) - 1 ≈ 0.10 → 10% total return.
    # Tolerance of 1e-3 leaves room for the small log/exp round-trip error.
    assert abs(result.total_return_pct - 0.10) < 1e-3


def test_short_signal_captures_downtrend():
    """A short position over a 10% down move should earn a positive return (you profit when price falls)."""

    # Two bars with a 10% close-to-close fall.  The short signal at index 0
    # is held over bar 1, capturing -1 * log(90/100) ≈ +0.1054 log return.
    bars = make_bars([100.0, 90.0])

    # Both signals short — same next-bar-execution logic as the long test,
    # mirrored to the short side.
    signals = np.array([SIGNAL_SHORT, SIGNAL_SHORT], dtype=np.int8)

    result = Backtester().run(bars, signals)

    # Expected math: log return = -1 * log(90/100) = -log(0.9) ≈ +0.1054.
    # exp(0.1054) - 1 ≈ 0.1111 → +11.11% total return.  This is asymmetric
    # vs the long test (10% up ≠ -10% down in log space) — that asymmetry
    # is real, and any "exactly 10%" assertion here would be wrong.
    assert abs(result.total_return_pct - 0.111) < 1e-3


def test_no_lookahead_bias():
    """THE critical test: a signal at bar i must NOT capture the move INTO bar i."""

    # 100% jump from bar 0 to bar 1.  A buggy engine that uses signals[i]
    # on asset_returns[i] (instead of signals[i-1]) would report ~+100%
    # return when the signal "go long" appears at index 1.
    bars = make_bars([100.0, 200.0])

    # Signal goes long at index 1.  Under correct next-bar execution:
    #   * signals[0] = 0 → no return captured over bar 1 (the 100→200 move)
    #   * signals[1] = 1 → would earn return over bar 2, but there is no bar 2
    # So realized P&L must be exactly zero.
    signals = np.array([SIGNAL_FLAT, SIGNAL_LONG], dtype=np.int8)

    result = Backtester().run(bars, signals)

    # If this assertion fails with ~+1.0 (100%), the engine has lookahead bias.
    # 1e-10 tolerance is well below any legitimate floating-point noise and
    # well above the ~+1.0 a lookahead bug would produce.
    assert abs(result.total_return_pct) < 1e-10


# ---------------------------------------------------------------------------
# Trade extraction
# ---------------------------------------------------------------------------


def test_trade_extraction_simple_round_trip():
    """A run of LONG signals between two FLATs produces exactly one Trade."""

    # Six bars, prices stepping up then down.  The signal pattern below
    # opens a trade at index 1 and closes it at index 4.
    bars = make_bars([100.0, 101.0, 102.0, 103.0, 102.0, 101.0])

    # 0 → 1 transition at index 1 opens a long.
    # 1 → 0 transition at index 4 closes it.
    # No other transitions, so exactly one trade.
    signals = np.array([0, 1, 1, 1, 0, 0], dtype=np.int8)

    result = Backtester().run(bars, signals)

    # Exactly one round-trip should be recorded.  Off-by-one in the
    # transition detection would surface here as 0 or 2 trades.
    assert result.n_trades == 1

    # Pull the single trade for field-by-field assertions.
    trade = result.trades[0]

    # Direction must reflect the run of LONG signals (+1), not e.g. the
    # transition direction or the price-move direction.
    assert trade.direction == SIGNAL_LONG

    # Entry price is bar 1's close (101) — the bar where signals transitioned
    # 0 → 1.  Fills are at the close of the signal bar, per the docstring.
    assert trade.entry_price == 101.0

    # Exit price is bar 4's close (102) — the bar where signals transitioned
    # 1 → 0.  Same close-fill convention as entry.
    assert trade.exit_price == 102.0

    # bars_held = exit_idx - entry_idx = 4 - 1 = 3.  Inclusive of entry,
    # exclusive of exit per the Trade docstring.
    assert trade.bars_held == 3

    # Return is log(102/101) ≈ 0.00985.  Direction is +1 so no sign flip.
    assert abs(trade.return_pct - np.log(102 / 101)) < 1e-12


def test_round_trip_with_no_edge():
    """A trade that exits at the same price as a non-trade reference should net to ~zero strategy return."""

    # Prices: 100 → 110 → 100.  The strategy is long over the 100→110 move
    # AND over the 110→100 move (signals[1]=LONG is held over bar 2's return).
    # Net log return: +log(1.10) + log(100/110) = 0 exactly.
    bars = make_bars([100.0, 110.0, 100.0])

    # signals[0] = LONG → captures bar 1's return (+log(1.10))
    # signals[1] = LONG → captures bar 2's return (log(100/110), negative)
    # signals[2] = FLAT → closes the trade at bar 2's close.
    signals = np.array([1, 1, 0], dtype=np.int8)

    result = Backtester().run(bars, signals)

    # Sum of log returns is 0, so exp(0) = 1, so total return = 0.
    # 1e-10 tolerance covers float64 add/subtract round-trip noise.
    assert abs(result.total_return_pct) < 1e-10

    # The 0 → 1 transition at index 0 opens a trade, and the 1 → 0
    # transition at index 2 closes it.  Exactly one trade.
    assert result.n_trades == 1


def test_max_drawdown_on_known_series():
    """Max drawdown is the worst peak-to-trough decline as a positive fraction."""

    # Equity curve will be: 1.0 → 1.10 → 0.90 (approximately).
    # Peak = 1.10, trough = 0.90, drawdown = (0.90 - 1.10) / 1.10 ≈ -0.1818.
    # So max_drawdown_pct ≈ 0.1818 reported as a positive number.
    bars = make_bars([100.0, 110.0, 90.0])

    # Long the whole time so the equity curve tracks the asset directly.
    signals = np.array([1, 1, 1], dtype=np.int8)

    result = Backtester().run(bars, signals)

    # 0.1818... is the magnitude; the engine reports it as positive even
    # though the underlying drawdown is a negative number.  Tolerance of
    # 1e-3 covers the small log/exp round-trip error in the equity curve.
    assert abs(result.max_drawdown_pct - 0.1818) < 1e-3


def test_win_rate_winning_trade():
    """A single profitable trade → win_rate == 1.0."""

    # 100 → 110 → 105.  Long the first two bars then flat: trade entry at
    # bar 0 (100), exit at bar 2 (105) — net log return positive.
    bars = make_bars([100.0, 110.0, 105.0])

    # signals[0] = LONG opens the trade at bar 0's close.
    # signals[2] = FLAT closes it at bar 2's close.
    signals = np.array([1, 1, 0], dtype=np.int8)

    result = Backtester().run(bars, signals)

    # Exactly one trade so win_rate is just "did it win or not."
    assert result.n_trades == 1

    # log(105/100) > 0 → counted as a win → win_rate = 1.0 exactly.
    # No tolerance needed: this is integer-counting math, not float math.
    assert result.win_rate == 1.0


def test_win_rate_losing_trade():
    """A single losing trade → win_rate == 0.0."""

    # 100 → 90 → 95.  Long the first two bars then flat: trade entry at
    # bar 0 (100), exit at bar 2 (95) — net log return negative.
    bars = make_bars([100.0, 90.0, 95.0])

    # Same signal pattern as the winning-trade test; only the prices differ.
    signals = np.array([1, 1, 0], dtype=np.int8)

    result = Backtester().run(bars, signals)

    # Exactly one trade so win_rate is just "did it win or not."
    assert result.n_trades == 1

    # log(95/100) < 0 → not counted as a win → win_rate = 0.0 exactly.
    # The strict > 0 check (not >= 0) means break-even trades are also losses;
    # that's by design — a flat trade still cost the opportunity to deploy
    # capital elsewhere.
    assert result.win_rate == 0.0


# ---------------------------------------------------------------------------
# Result immutability
# ---------------------------------------------------------------------------


def test_result_is_immutable():
    """BacktestResult is a frozen dataclass — attribute assignment must fail."""

    # Run any small backtest just to obtain a real result object.  The
    # specific values don't matter — we're testing the type's mutability,
    # not the math.
    bars = make_bars([100.0, 101.0])
    signals = np.array([1, 1], dtype=np.int8)
    result = Backtester().run(bars, signals)

    # FrozenInstanceError is the specific exception @dataclass(frozen=True)
    # raises on attribute assignment.  Asserting on the specific subclass
    # (not just any Exception) means a future change to frozen=False would
    # silently pass a broader assertion but fails this one — exactly the
    # regression we want to catch.
    with pytest.raises(dataclasses.FrozenInstanceError):
        # Mutating a backtest result after construction would silently
        # invalidate any analysis that already read it.  Frozen catches
        # that bug at write time.
        result.total_return_pct = 999.0


# ---------------------------------------------------------------------------
# Transaction-cost model
#
# Cost contract (from engine.run, block 4b):
#   held[i]      = signals[i-1], held[0] = 0.0   (next-bar execution lag)
#   turnover[i]  = |held[i] - held[i-1]|, turnover[0] = 0.0
#   cost_returns = turnover * cost_rate          (per-bar log-return drag)
#   net returns  = gross returns - cost_returns
#   total_cost_pct = sum(cost_returns)           (fraction of capital)
#   cost_rate    = (fee_bps + slippage_bps) / 10000.0
# Every expected number below is derived by hand in the comments.
# ---------------------------------------------------------------------------


def test_zero_cost_is_identical_to_before():
    """Backward-compat: default costs and explicit fee=slip=0 must be bit-identical."""

    # A small series with both an entry and an exit so turnover is non-trivial:
    # if zero-cost weren't a true no-op, a costed bar would diverge here.
    bars = make_bars([100.0, 110.0, 105.0, 108.0])

    # held = [0, 1, 1, 0] → turnover = [0, 1, 0, 1]: an entry at bar 1 and an
    # exit at bar 3, so there ARE bars that would be charged if cost_rate > 0.
    signals = np.array([1, 1, 0, 0], dtype=np.int8)

    # Default constructor: fee_bps and slippage_bps default to 0.0, so this is
    # the pre-cost code path — the behaviour every existing test pins down.
    default_result = Backtester().run(bars, signals)

    # Explicit zeros: must travel the same arithmetic and produce the same
    # numbers.  Passing 0.0 explicitly should be indistinguishable from the
    # default — that's the contract that keeps all 156 prior tests green.
    explicit_zero_result = Backtester(fee_bps=0, slippage_bps=0).run(bars, signals)

    # np.array_equal (exact, not approximate): with cost_rate == 0.0 the
    # subtraction strategy_returns - cost_returns subtracts an all-zeros array,
    # which must leave every element bit-for-bit unchanged — no tolerance.
    assert np.array_equal(default_result.returns, explicit_zero_result.returns)

    # And the reported cost must be exactly 0.0 on both runs — no cost was
    # configured, so none can have been charged.
    assert default_result.total_cost_pct == 0.0
    assert explicit_zero_result.total_cost_pct == 0.0


def test_single_round_trip_charges_two_units():
    """flat→long→flat charges exactly two bars: the entry and the exit."""

    # Four bars; prices are irrelevant to the cost arithmetic (cost depends
    # only on turnover and cost_rate), so any strictly increasing series works.
    bars = make_bars([100.0, 101.0, 102.0, 103.0])

    # signals [1, 1, 0, 0] → held = signals shifted by one = [0, 1, 1, 0].
    # turnover = |diff(held)| = [_, |1-0|, |1-1|, |0-1|] = [0, 1, 0, 1].
    # So exactly two bars carry turnover: bar 1 (entry) and bar 3 (exit).
    signals = np.array([1, 1, 0, 0], dtype=np.int8)

    # fee_bps=100 → cost_rate = 100/10000 = 0.01, a round number that makes the
    # hand-derived costs (0.01 per charged bar) trivial to read in the asserts.
    bt = Backtester(fee_bps=100, slippage_bps=0)

    # cost_rate sanity check — the rest of the test's hand math assumes 0.01.
    assert bt.cost_rate == 0.01

    # The costed run, plus a free run to diff against for the per-bar check.
    costed = bt.run(bars, signals)
    free = Backtester().run(bars, signals)

    # total_cost = sum(turnover * cost_rate) = (1 + 1) * 0.01 = 0.02.  Exact
    # float math here (0.01 + 0.01), so a tight 1e-12 tolerance is honest.
    assert abs(costed.total_cost_pct - 2 * bt.cost_rate) < 1e-12

    # Per-bar verification: gross minus net should equal the cost charged at
    # each bar, i.e. turnover * cost_rate = [0, 0.01, 0, 0.01].  Charged only
    # at the entry (bar 1) and exit (bar 3); zero on the flat bars.
    expected_cost_per_bar = np.array([0.0, 0.01, 0.0, 0.01])

    # free.returns - costed.returns isolates exactly the cost drag (the gross
    # component is identical between the two runs), so it must match the hand
    # array element-for-element within float noise.
    assert np.allclose(free.returns - costed.returns, expected_cost_per_bar, atol=1e-12)


def test_holding_charges_nothing_extra():
    """Continuous long charges only the single entry bar — holding is free."""

    # Four bars, prices irrelevant to cost (turnover-only), any series works.
    bars = make_bars([100.0, 101.0, 102.0, 103.0])

    # signals [1, 1, 1, 1] → held = [0, 1, 1, 1].
    # turnover = |diff(held)| = [0, |1-0|, |1-1|, |1-1|] = [0, 1, 0, 0].
    # Only bar 1 (the entry) carries turnover; bars 2 and 3 are pure holding.
    signals = np.array([1, 1, 1, 1], dtype=np.int8)

    # cost_rate = 0.01 again (fee_bps=100), for clean hand math.
    bt = Backtester(fee_bps=100, slippage_bps=0)

    costed = bt.run(bars, signals)
    free = Backtester().run(bars, signals)

    # total_cost = 1 entry * cost_rate = 0.01.  Holding adds nothing, so the
    # total is one unit of turnover regardless of how many bars are held.
    assert abs(costed.total_cost_pct - 1 * bt.cost_rate) < 1e-12

    # The interior holding bars (indices 2 and 3) must show ZERO cost drag:
    # free and net returns must be identical there.  This is the assertion
    # that proves cost is charged on position CHANGES, not on exposure.
    diff = free.returns - costed.returns
    assert diff[2] == 0.0
    assert diff[3] == 0.0


def test_long_to_short_flip_charges_double():
    """A long→short flip charges 2 units on the flip bar (close long + open short)."""

    # Four bars, prices irrelevant to the cost arithmetic.
    bars = make_bars([100.0, 101.0, 102.0, 103.0])

    # signals [1, 1, -1, -1] → held = [0, 1, 1, -1].
    # turnover = |diff(held)| = [0, |1-0|, |1-1|, |-1-1|] = [0, 1, 0, 2].
    # Bar 1 enters long (1 unit); bar 3 flips long→short (2 units: exit the
    # long AND enter the short).  Total turnover = 1 + 2 = 3.
    signals = np.array([1, 1, -1, -1], dtype=np.int8)

    # cost_rate = 0.01 (fee_bps=100) for readable hand math.
    bt = Backtester(fee_bps=100, slippage_bps=0)

    costed = bt.run(bars, signals)

    # total_cost = 3 units * cost_rate = 3 * 0.01 = 0.03.  This is the path an
    # SMA-style strategy hits when it legitimately flips sign without going
    # flat in between, so the doubled flip charge must be exercised.
    assert abs(costed.total_cost_pct - 3 * bt.cost_rate) < 1e-12


def test_costs_lower_sharpe_and_total_return():
    """On a profitable long series, costs strictly lower total return and don't raise Sharpe."""

    # A profitable, varied long series.  Bar 1's return log(100.5/100)≈0.00499
    # is well BELOW the mean of the per-bar returns, so charging cost there
    # pushes that return further from the mean — lowering the mean AND raising
    # the spread, which drives Sharpe down rather than up.
    bars = make_bars([100.0, 100.5, 105.0, 110.0, 112.0])

    # Long throughout → held = [0, 1, 1, 1, 1], turnover = [0, 1, 0, 0, 0]:
    # a single entry charge at bar 1.  No shorts — this is a "long-only series."
    signals = np.array([1, 1, 1, 1, 1], dtype=np.int8)

    # Zero-cost baseline vs a positive-cost run.  fee_bps=10 → cost_rate=0.001,
    # small but strictly positive so the drag is unambiguous.
    free = Backtester().run(bars, signals)
    costed = Backtester(fee_bps=10, slippage_bps=0).run(bars, signals)

    # Strictly lower total return: a positive cost is subtracted from the
    # cumulative log return, so net terminal equity must be below gross.
    assert costed.total_return_pct < free.total_return_pct

    # Sharpe must be no higher with costs.  We deliberately do NOT pin an exact
    # value (Sharpe depends on the whole return distribution) — only the
    # direction of the inequality is a robust, cost-driven guarantee.
    assert costed.sharpe_ratio <= free.sharpe_ratio


def test_negative_bps_raise():
    """fee_bps and slippage_bps must be >= 0 — negatives are rejected at construction."""

    # A negative fee would pay the strategy to trade, inflating returns — the
    # constructor must reject it up front, same as the capital/annualization guards.
    with pytest.raises(ValueError):
        Backtester(fee_bps=-1)

    # Slippage is likewise a cost that can never be negative; its guard mirrors
    # the fee guard exactly, so a negative value must raise here too.
    with pytest.raises(ValueError):
        Backtester(slippage_bps=-1)


def test_fee_and_slippage_add():
    """Only the SUM of fee_bps + slippage_bps matters — the split is irrelevant."""

    # Any series with some turnover so a non-zero cost is actually charged.
    bars = make_bars([100.0, 101.0, 102.0, 103.0])

    # held = [0, 1, 1, 0] → turnover [0, 1, 0, 1]: an entry and an exit, so
    # total turnover is 2 and the run carries a non-zero cost to compare.
    signals = np.array([1, 1, 0, 0], dtype=np.int8)

    # Three different (fee, slippage) splits that all sum to 3 bps, hence all
    # have cost_rate = 3/10000 = 0.0003.  Because the engine sums fee+slippage
    # into a single cost_rate, the per-run total cost must be identical.
    split = Backtester(fee_bps=1, slippage_bps=2).run(bars, signals)
    all_fee = Backtester(fee_bps=3, slippage_bps=0).run(bars, signals)
    all_slip = Backtester(fee_bps=0, slippage_bps=3).run(bars, signals)

    # All three total costs must match exactly: the components enter the model
    # only through their sum, so the split cannot change the charged amount.
    # 1e-12 covers float-add noise (1+2 vs 3+0 vs 0+3 over /10000.0).
    assert abs(split.total_cost_pct - all_fee.total_cost_pct) < 1e-12
    assert abs(split.total_cost_pct - all_slip.total_cost_pct) < 1e-12
