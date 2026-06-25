# src/backtest/engine.py

"""
Vectorized backtester for daily-bar position signals.

Core assumption: signals[i] is the position you HOLD over bar i+1.  That is,
the signal generated at the close of bar i is acted upon at the close of bar
i (entering the position) and earns the return from bar i to bar i+1.

This is the standard "next-bar execution" model used in vectorbt, backtrader,
and academic backtests.  It's intentionally NOT lookahead: a signal at bar i
cannot use information from bar i+1 or later.

Transaction costs (fees + slippage, expressed in basis points and charged
per unit of turnover) are now modeled.  Both default to 0.0 — a default
Backtester is cost-free — and when configured they are applied to the net
per-bar returns, so every downstream metric reflects them.

No position sizing — every signal is "100% of capital, long or short."
Position sizing is also Phase 3.
"""

# numpy is the workhorse: closes, returns, signals, and the equity curve are
# all manipulated as float64 arrays so the per-bar math is vectorized in C
# rather than looped in Python.
import numpy as np

# datetime is only used as a type hint here; the actual timestamps come off
# OHLCVBar.timestamp when we build Trade records.
from datetime import datetime

# OHLCVBar is the canonical bar dataclass.  We only read .close and
# .timestamp off it, but importing the type keeps the run() signature honest.
from src.data.schema import OHLCVBar

# Trade and BacktestResult are the immutable transport objects defined in
# src/backtest/result.py.  This engine produces them; downstream consumers
# (reports, plotting, optimization) read them.
from src.backtest.result import Trade, BacktestResult

# The three legal signal values.  We import the constants rather than
# hard-coding ±1/0 so a future change to the signal scheme propagates here.
from src.strategies.base import SIGNAL_LONG, SIGNAL_FLAT, SIGNAL_SHORT

# Pure metric functions extracted to metrics.py so walk-forward validation
# can call them on arbitrary slices without going through the full Backtester.
from src.backtest import metrics


# ---------------------------------------------------------------------------
# Backtester — converts (bars, signals) into a BacktestResult.
# ---------------------------------------------------------------------------

class Backtester:
    """Run a single backtest from a list of bars and a signal array."""

    def __init__(
        self,
        initial_capital: float = 1.0,
        annualization_factor: int = 252,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ):
        """Configure the run.

        initial_capital defaults to 1.0 so the equity curve is a multiplier —
        a final value of 1.25 means +25%.  This makes percentage thinking the
        default; callers who want dollar amounts can pass their own capital.

        annualization_factor defaults to 252, the conventional number of US
        trading days per year, used to scale Sharpe from per-bar to annualized.

        fee_bps and slippage_bps express transaction costs in basis points of
        traded notional (1 bp = 0.0001).  Both default to 0.0 so a default
        Backtester is cost-free and bit-for-bit identical to the pre-cost
        behaviour; supply positive values to charge fees/slippage per turnover.
        """
        # Capital must be strictly positive — zero or negative capital is
        # nonsensical and would also produce NaNs/infs downstream when used
        # as a denominator.  Fail loudly at construction, not silently later.
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {initial_capital}")

        # Same guard for the annualization factor: Sharpe scaling is
        # mean/std * sqrt(annualization_factor); a non-positive value would
        # break the sqrt or produce zero/negative scaling.
        if annualization_factor <= 0:
            raise ValueError(f"annualization_factor must be > 0, got {annualization_factor}")

        # Fees are a cost, never a credit: a negative fee would pay the strategy
        # to trade, which is nonsensical and would inflate returns.  0.0 is
        # allowed (the cost-free default); only strictly negative is rejected.
        if fee_bps < 0:
            raise ValueError(f"fee_bps must be >= 0, got {fee_bps}")

        # Same guard for slippage: it is a cost charged on turnover and can
        # never be negative.  0.0 stays valid so the default run is cost-free.
        if slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {slippage_bps}")

        # Store both on the instance so run() can read them.  No mutation
        # after construction — Backtester is configured once and reused.
        self.initial_capital = initial_capital
        self.annualization_factor = annualization_factor

        # Keep the raw bps inputs on the instance for introspection/reporting,
        # so a caller can read back exactly what costs were configured.
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

        # cost_rate is the total per-unit-turnover cost as a fraction.  Dividing
        # bps by 10000 converts basis points to a fraction (10 bps -> 0.001).
        # Fees and slippage are both charged per unit of turnover, so they sum
        # into a single rate applied uniformly to each bar's position change.
        self.cost_rate = (fee_bps + slippage_bps) / 10000.0

    def run(
        self,
        bars: list[OHLCVBar],
        signals: np.ndarray,
        strategy_name: str = "unnamed",
    ) -> BacktestResult:
        """Simulate the strategy and return a BacktestResult.

        Pre-conditions enforced at the top of the function:
          * bars is non-empty
          * len(signals) == len(bars)  (alignment contract from Strategy)
          * signals has integer dtype
          * every signal value is in {-1, 0, 1}
        """

        # ------------------------------------------------------------------
        # 1. Validate inputs.  All checks raise ValueError with a message
        #    that identifies the exact failure — easier to debug than a
        #    later NaN or shape mismatch deep in the math.
        # ------------------------------------------------------------------

        # Empty bars would make every subsequent step (closes[0], returns,
        # equity curve) ill-defined.  Reject up front.
        if len(bars) == 0:
            raise ValueError("bars must be non-empty")

        # The Strategy contract guarantees one signal per bar.  If the caller
        # violates that, fail loudly rather than silently truncating one side.
        if len(signals) != len(bars):
            raise ValueError(
                f"signals length ({len(signals)}) must equal bars length ({len(bars)})"
            )

        # signals.dtype.kind == 'i' covers all signed integer widths (int8,
        # int32, int64).  We reject floats even if their values happen to be
        # whole numbers — a float dtype signals a likely bug upstream.
        if signals.dtype.kind != "i":
            raise ValueError(
                f"signals must have integer dtype, got {signals.dtype}"
            )

        # np.isin returns a boolean array; .all() collapses it to a single
        # bool.  Any stray value (e.g. 2 from a buggy strategy) gets caught
        # here before it silently scales returns by the wrong magnitude.
        if not np.isin(signals, [SIGNAL_SHORT, SIGNAL_FLAT, SIGNAL_LONG]).all():
            raise ValueError(
                "signals must contain only values in {-1, 0, 1}"
            )

        # ------------------------------------------------------------------
        # 2. Extract close prices into a numpy array.  Strategies and the
        #    Trade record both use close prices as the reference fill price
        #    — consistent with the "fill at the close of the signal bar"
        #    assumption stated in the module docstring.
        # ------------------------------------------------------------------

        # List comprehension is fine here: bars is already in memory and a
        # single pass is O(n).  dtype=np.float64 guarantees the precision the
        # downstream log/exp math depends on.
        closes = np.array([b.close for b in bars], dtype=np.float64)

        # ------------------------------------------------------------------
        # 2b. Validate close prices.  A close that is non-finite (NaN or
        #     +/- infinity) or not strictly positive (<= 0) cannot produce a
        #     valid log return: np.log(closes[1:] / closes[:-1]) below would
        #     yield NaN (from a NaN or 0/0), -inf (from log of 0), or a
        #     domain error (from log of a negative), silently poisoning every
        #     downstream return, metric, and trade.  Reject the run here so a
        #     corrupt bar surfaces as a clear ValueError instead of a quiet NaN.
        # ------------------------------------------------------------------

        # bad_closes[i] is True when closes[i] is unusable: not finite OR not
        # strictly positive.  ~np.isfinite catches NaN and +/- inf; the
        # `closes <= 0` arm catches zero and negative prices (also non-loggable).
        bad_closes = ~np.isfinite(closes) | (closes <= 0)

        # .any() collapses the mask to a single bool — only build the error
        # message (and pay for np.where) when at least one close is bad.
        if bad_closes.any():
            # np.where returns the indices where the mask is True; [0][0] is the
            # FIRST such index, the most useful one to report for debugging.
            first_bad = int(np.where(bad_closes)[0][0])
            # Mirror the other guards' style: state the count and pinpoint the
            # first offender by index and its bar timestamp so the caller can
            # locate the corrupt row in the source data immediately.
            raise ValueError(
                f"closes must be finite and strictly positive; found "
                f"{int(bad_closes.sum())} bad value(s), first at index {first_bad} "
                f"(timestamp {bars[first_bad].timestamp}, close {closes[first_bad]})"
            )

        # ------------------------------------------------------------------
        # 3. Per-bar log returns of the underlying asset.
        #    asset_returns[i] = log(closes[i] / closes[i-1]) for i >= 1.
        #    asset_returns[0] = 0.0 — no prior bar to compare against.
        # ------------------------------------------------------------------

        # Allocate up front and assign into the [1:] slice so element 0
        # stays the zero we initialised it with.
        asset_returns = np.zeros(len(bars), dtype=np.float64)

        # closes[1:] / closes[:-1] is the per-bar price ratio; np.log makes
        # it a log return.  Log returns are time-additive (cumsum gives total
        # log return) which is exactly what step 5 relies on.
        asset_returns[1:] = np.log(closes[1:] / closes[:-1])

        # ------------------------------------------------------------------
        # 4. Per-bar strategy returns under next-bar execution.
        #    The strategy holds position signals[i-1] over bar i, so
        #    strategy_returns[i] = signals[i-1] * asset_returns[i] for i >= 1.
        #    strategy_returns[0] = 0.0 — no signal is active on the first bar.
        # ------------------------------------------------------------------

        # Same allocate-then-fill pattern as asset_returns so the leading
        # zero is established without a special-case branch.
        strategy_returns = np.zeros(len(bars), dtype=np.float64)

        # signals[:-1] aligned with asset_returns[1:] encodes the "use
        # yesterday's signal on today's return" rule in a single vectorized
        # multiply.  Cast to float64 so the multiply produces float results
        # rather than truncating to int.
        strategy_returns[1:] = signals[:-1].astype(np.float64) * asset_returns[1:]

        # ------------------------------------------------------------------
        # 4b. Transaction costs — fees + slippage charged on position changes.
        #     Computed AFTER the gross returns above and BEFORE the equity
        #     curve below, so every downstream consumer reads NET returns.
        #     With cost_rate == 0.0 (the default) every line here adds exactly
        #     0.0, leaving strategy_returns bit-for-bit unchanged.
        # ------------------------------------------------------------------

        # held[i] is the position actually held over bar i.  Under next-bar
        # execution that is signals[i-1] — the same lag the gross returns use
        # (signals[:-1] aligned to asset_returns[1:]), so costs and returns
        # are consistent by construction with no extra shift of our own.
        held = np.zeros(len(bars), dtype=np.float64)

        # held[1:] = signals[:-1] leaves held[0] = 0.0: we are flat on the
        # first bar, matching strategy_returns[0] = 0.0 (no signal active yet).
        held[1:] = signals[:-1].astype(np.float64)

        # turnover[i] is the size of the position change at bar i, |Δheld|.
        # Allocate-then-fill so turnover[0] stays 0.0 (no prior position to
        # change from on the first bar).
        turnover = np.zeros(len(bars), dtype=np.float64)

        # np.abs(np.diff(held)) gives |held[i] - held[i-1]| for i >= 1.
        # Entering a unit long is turnover 1, exiting is 1, and a long-to-short
        # flip is 2 (close the long, open the short — two units of notional).
        turnover[1:] = np.abs(np.diff(held))

        # cost_returns[i] is the cost charged at bar i, expressed as a
        # log-return drag: turnover scaled by the per-unit cost_rate.  This is
        # a linear approximation of the multiplicative cost — we subtract
        # turnover*cost_rate from the log return rather than multiplying gross
        # by (1 - turnover*cost_rate).  At basis-point magnitudes the gap is
        # negligible (second order in cost_rate), and staying linear makes the
        # per-bar cost exactly additive and the total trivially auditable.
        cost_returns = turnover * self.cost_rate

        # Subtract the cost from the gross returns IN PLACE so everything below
        # — equity curve, Sharpe, total return, max drawdown — consumes the NET
        # array with no other change.  When cost_rate == 0.0, cost_returns is
        # all zeros and this subtraction is a no-op (bit-for-bit identical).
        strategy_returns = strategy_returns - cost_returns

        # total_cost is the cumulative cost charged over the whole run, the sum
        # of the per-bar drags.  total_return_pct is a FRACTION in this codebase
        # (e.g. 0.25 == +25%), so total_cost is left as the raw sum with NO
        # multiply by 100: it is the cumulative trading cost as a fraction of
        # capital.  float() matches the stored-type convention of the metrics.
        total_cost = float(cost_returns.sum())

        # ------------------------------------------------------------------
        # 5. Equity curve.  Working in log space avoids floating-point drift
        #    that would accumulate from repeated (1 + r) multiplications
        #    across thousands of bars.
        # ------------------------------------------------------------------

        # cumsum of log returns is the total log return as of each bar.
        cumulative_log_returns = np.cumsum(strategy_returns)

        # exp(cumulative log return) recovers the multiplicative growth
        # factor; multiply by initial_capital to anchor the curve.  Bar 0's
        # cumsum is 0 → exp(0) = 1 → equity_curve[0] = initial_capital.
        equity_curve = self.initial_capital * np.exp(cumulative_log_returns)

        # ------------------------------------------------------------------
        # 6. Extract trades from signal transitions.
        #    A trade is a run of non-zero signals.  We walk the signals
        #    array once, opening a trade on 0→nonzero transitions and
        #    closing it on nonzero→0 transitions and on sign flips
        #    (which are an exit and an entry back-to-back).
        # ------------------------------------------------------------------

        # The collected Trade records.  Start empty; append in chronological order.
        trades: list[Trade] = []

        # current_position is the signal we believe we're holding right now.
        # 0 means flat.  We update it as we walk and detect transitions.
        current_position: int = SIGNAL_FLAT

        # entry_idx is the bar index where the current (open) trade was
        # entered.  None when no trade is open.  We need the index, not just
        # the price, so we can compute bars_held and pull the entry timestamp.
        entry_idx: int | None = None

        # Single pass over signals.  i is the bar index, sig is the signal
        # value at that bar.  Using a Python loop (not vectorized) is fine
        # here because trades are sparse — a 5-year backtest typically has
        # tens of trades, not thousands.
        for i in range(len(signals)):
            # int() cast: signals[i] is a numpy scalar; converting to a
            # plain Python int makes comparisons and the stored direction
            # field consistent with the Trade dataclass's int type hint.
            sig = int(signals[i])

            # Case A: we're flat and a non-zero signal appears → open a trade.
            if current_position == SIGNAL_FLAT and sig != SIGNAL_FLAT:
                entry_idx = i
                current_position = sig

            # Case B: we're in a position and the signal goes flat → close the trade.
            elif current_position != SIGNAL_FLAT and sig == SIGNAL_FLAT:
                # entry_idx is guaranteed non-None whenever current_position
                # is non-flat — the two are updated together.
                trades.append(self._make_trade(bars, closes, entry_idx, i, current_position))
                # Reset state: no open trade, position is flat.
                current_position = SIGNAL_FLAT
                entry_idx = None

            # Case C: position flips sign (e.g. +1 → -1) without passing
            # through 0.  Treat as a close-then-open on the same bar: the
            # exit and entry both happen at closes[i].
            elif current_position != SIGNAL_FLAT and sig != current_position:
                trades.append(self._make_trade(bars, closes, entry_idx, i, current_position))
                # Immediately open the opposite-direction trade at this bar.
                entry_idx = i
                current_position = sig

            # Case D: signal unchanged → keep holding; nothing to record.
            # (Includes flat→flat and same-direction continuations.)

        # End-of-data: if a position is still open after the loop, close it
        # at the last bar so the trade list reflects realized P&L only.
        if current_position != SIGNAL_FLAT and entry_idx is not None:
            trades.append(
                self._make_trade(bars, closes, entry_idx, len(bars) - 1, current_position)
            )

        # ------------------------------------------------------------------
        # 7. Summary metrics — delegated to src/backtest/metrics.py so the
        #    same functions can be reused on walk-forward test slices without
        #    re-running the full Backtester.
        # ------------------------------------------------------------------

        # total_return: initial_capital passed explicitly (not self.*) so the
        # function can compute return for any slice with its own starting equity.
        # float() wrapping stays at the BacktestResult call site below,
        # matching the original assignment's type behaviour exactly.
        total_return_pct = metrics.total_return(equity_curve, self.initial_capital)

        # sharpe_ratio: strategy_returns[1:] sliced HERE, not inside the
        # function, so the function works on any arbitrary returns array.
        # The [1:] skips the structural index-0 zero (no signal active on
        # bar 0) — identical to the inline active_returns = strategy_returns[1:]
        # that preceded the old if/else block.
        sharpe_ratio = metrics.sharpe_ratio(strategy_returns[1:], self.annualization_factor)

        # max_drawdown: full equity_curve passed unchanged.
        max_drawdown_pct = metrics.max_drawdown(equity_curve)

        # win_rate: full trades list passed; the function handles the no-trades
        # case internally, matching the old inline if/else exactly.
        # NOTE: win_rate stays GROSS — it is computed from signal transitions
        # and close prices with no per-trade cost.  Per-trade cost attribution
        # is a deliberately deferred scope boundary; only the aggregate net
        # metrics (Sharpe, total return, max drawdown, equity) reflect cost.
        win_rate = metrics.win_rate(trades)

        # n_trades is cached on the result so callers don't recompute len().
        n_trades = len(trades)

        # ------------------------------------------------------------------
        # 8. Assemble and return the immutable result.  Once constructed,
        #    BacktestResult is frozen — no one can silently mutate it.
        # ------------------------------------------------------------------
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=bars[0].timestamp,
            end_date=bars[-1].timestamp,
            n_bars=len(bars),
            equity_curve=equity_curve,
            returns=strategy_returns,
            trades=trades,
            total_return_pct=float(total_return_pct),
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            n_trades=n_trades,
            # total_cost_pct carries the cumulative transaction cost as a
            # fraction of capital; 0.0 when no costs are configured.  Passed by
            # keyword (as every field here is) so field order is irrelevant.
            total_cost_pct=total_cost,
        )

    # ------------------------------------------------------------------
    # Internal helper: build one Trade record from entry/exit indices.
    # Pulled out so the trade-extraction loop above stays readable and
    # the entry/exit construction lives in exactly one place.
    # ------------------------------------------------------------------
    @staticmethod
    def _make_trade(
        bars: list[OHLCVBar],
        closes: np.ndarray,
        entry_idx: int,
        exit_idx: int,
        direction: int,
    ) -> Trade:
        # Entry and exit fill prices come from the close array — same source
        # as the strategy_returns computation, so trade P&L is consistent
        # with the equity curve P&L by construction.
        entry_price = float(closes[entry_idx])
        exit_price = float(closes[exit_idx])

        # Log return of the trade, signed by direction so a profitable short
        # (price fell) reads as a positive return_pct just like a winning long.
        return_pct = float(direction * np.log(exit_price / entry_price))

        # bars_held = exit_idx - entry_idx.  Per the Trade docstring this is
        # inclusive of the entry bar and exclusive of the exit bar — so a
        # trade opened at bar 10 and closed at bar 13 has bars_held == 3.
        bars_held = exit_idx - entry_idx

        return Trade(
            entry_time=bars[entry_idx].timestamp,
            exit_time=bars[exit_idx].timestamp,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            return_pct=return_pct,
            bars_held=bars_held,
        )
