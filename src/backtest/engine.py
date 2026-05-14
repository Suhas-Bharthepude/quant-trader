# src/backtest/engine.py

"""
Vectorized backtester for daily-bar position signals.

Core assumption: signals[i] is the position you HOLD over bar i+1.  That is,
the signal generated at the close of bar i is acted upon at the close of bar
i (entering the position) and earns the return from bar i to bar i+1.

This is the standard "next-bar execution" model used in vectorbt, backtrader,
and academic backtests.  It's intentionally NOT lookahead: a signal at bar i
cannot use information from bar i+1 or later.

No transaction costs or slippage in this first version.  That's a Phase 3
addition (risk module) — for now we measure raw strategy edge.

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


# ---------------------------------------------------------------------------
# Backtester — converts (bars, signals) into a BacktestResult.
# ---------------------------------------------------------------------------

class Backtester:
    """Run a single backtest from a list of bars and a signal array."""

    def __init__(self, initial_capital: float = 1.0, annualization_factor: int = 252):
        """Configure the run.

        initial_capital defaults to 1.0 so the equity curve is a multiplier —
        a final value of 1.25 means +25%.  This makes percentage thinking the
        default; callers who want dollar amounts can pass their own capital.

        annualization_factor defaults to 252, the conventional number of US
        trading days per year, used to scale Sharpe from per-bar to annualized.
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

        # Store both on the instance so run() can read them.  No mutation
        # after construction — Backtester is configured once and reused.
        self.initial_capital = initial_capital
        self.annualization_factor = annualization_factor

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
        # 7. Summary metrics.
        # ------------------------------------------------------------------

        # Total return as a fraction: 0.25 means the equity curve ended 25%
        # above where it started.  Subtract 1 so a flat run reads as 0.0.
        total_return_pct = equity_curve[-1] / self.initial_capital - 1.0

        # Sharpe ratio: per-bar mean over per-bar std, scaled to annual by
        # sqrt(annualization_factor).  We skip strategy_returns[0] because
        # it's a structural zero (no signal active) that would bias both
        # mean and std downward.
        # Sharpe needs at least 2 active return observations to compute a sample
        # standard deviation (ddof=1). For shorter inputs, std is mathematically
        # undefined — return 0.0 rather than NaN, which would poison downstream
        # comparisons and sorts.
        active_returns = strategy_returns[1:]
        if len(active_returns) < 2:
            sharpe_ratio = 0.0
        else:
            mean_return = active_returns.mean()
            # ddof=1 → sample standard deviation (Bessel-corrected), the academic
            # convention for empirical Sharpe estimates.
            std_return = active_returns.std(ddof=1)
            # Guard against the all-flat case (std == 0); also catches any residual
            # NaN that slipped through (defensive).
            if std_return == 0.0 or np.isnan(std_return):
                sharpe_ratio = 0.0
            else:
                sharpe_ratio = float(mean_return / std_return * np.sqrt(self.annualization_factor))

        # Max drawdown: largest peak-to-trough decline in the equity curve.
        # np.maximum.accumulate gives the running maximum at each bar — the
        # "high water mark" the curve has reached so far.
        running_max = np.maximum.accumulate(equity_curve)
        # Drawdown at each bar is (current - peak) / peak — zero or negative.
        # Dividing by running_max (not initial_capital) makes it the percentage
        # decline from the most recent peak, which is what investors care about.
        drawdown = (equity_curve - running_max) / running_max
        # drawdown.min() is the most negative value; negate it to report
        # the worst drawdown as a positive fraction (0.20 = 20% drawdown).
        # A monotonically-increasing equity curve has drawdown.min() == 0.
        max_drawdown_pct = float(-drawdown.min())

        # Win rate: fraction of completed trades with strictly positive
        # return.  No trades → 0.0 (defined; avoids a divide-by-zero).
        if len(trades) == 0:
            win_rate = 0.0
        else:
            win_rate = sum(1 for t in trades if t.return_pct > 0) / len(trades)

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
