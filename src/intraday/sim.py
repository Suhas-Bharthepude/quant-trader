# src/intraday/sim.py

"""
Event-driven intraday backtester for the opening-range breakout strategy.

WHY A SEPARATE ENGINE.  The daily vectorized Backtester (src/backtest/engine.py)
earns close-to-close returns over full bars; it structurally cannot model an
intrabar stop, an intrabar profit target, or a forced end-of-day flat.  Those
exits ARE the ORB strategy, so intraday needs a bar-by-bar simulator that walks
each session and resolves exits against each 1-minute bar's high/low.  This file
is that engine.  It does NOT touch, import mutably, or weaken the daily engine —
it only REUSES the read-only Trade record and the pure metrics functions.

Exit model (per session, long-only):
  entry_price  = the strategy's buy-stop fill (OR-high or a gap-through open)
  stop_price   = entry_price * (1 - stop_pct)          [~6% below entry]
  R            = entry_price - stop_price               [risk per share]
  target_price = entry_price + target_r * R            [2R above entry]

For each bar from the entry bar onward:
  * stop touched  ⟺ bar.low  <= stop_price
  * target touched ⟺ bar.high >= target_price
  * STOP-FIRST: if BOTH are touched in the SAME bar, the STOP fills (we cannot
    know the intrabar path from OHLC, so we assume the adverse outcome).
  * gap handling: stop fills at min(stop_price, bar.open) — a gap DOWN through
    the stop fills WORSE; target fills at exactly target_price — a favorable
    gap up is never credited.
If neither level is touched by the last trading bar, the position is force-flat
at that last bar's close (EOD-flat) — the sim NEVER holds overnight, so overnight
gap returns can never leak in.

The entry bar itself is included in exit monitoring: if the entry bar's own range
reaches the stop, we assume the stop (adverse) rather than crediting the entry.
This keeps the sim on the conservative side, per the project's understate-don't-
flatter discipline.

Cost model: fee_bps + slippage_bps charged PER SIDE as a log-return drag; a round
trip (enter + exit) subtracts 2 * (fee_bps + slippage_bps)/10000 from the trade's
log return.  Intraday Trade.return_pct is therefore NET of costs — a deliberate
difference from the daily engine's gross win_rate convention, documented here
because for a leveraged, wide-spread instrument like SOXL costs are material and
a net win-rate is the honest one.
"""

from dataclasses import dataclass
from datetime import date, datetime
import math

import numpy as np

# Reused read-only from the daily stack: the Trade record and the pure metric
# functions.  Neither is modified.
from src.backtest.result import Trade
from src.backtest import metrics

from src.data.schema import OHLCVBar
from src.intraday.orb_strategy import EntryEvent, IntradayStrategy, OpeningRangeBreakoutConfig
from src.intraday.session import PrimarySession, build_sessions, AlignedSession


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionOutcome:
    """What happened on one session: a net log return and, if traded, the Trade.

    exit_reason is "stop", "target", or "eod" when a trade occurred, else None.
    return_log is the NET session log return (0.0 on a no-trade day), the value
    that compounds into the equity curve.
    """

    date_et: date
    return_log: float
    trade: Trade | None
    exit_reason: str | None


@dataclass(frozen=True)
class IntradayBacktestResult:
    """The full output of one intraday ORB backtest.

    Mirrors the SHAPE of BacktestResult but is a SEPARATE type (the daily result
    stays untouched).  returns/equity_curve are one-per-session, in date order.
    """

    strategy_name: str
    primary_symbol: str
    start_date: datetime | None
    end_date: datetime | None
    n_sessions: int
    n_trades: int
    equity_curve: np.ndarray
    returns: np.ndarray
    outcomes: list[SessionOutcome]
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_cost_pct: float
    exit_reason_counts: dict[str, int]

    @property
    def trades(self) -> list[Trade]:
        """The completed Trades (one per traded session), in date order."""
        return [o.trade for o in self.outcomes if o.trade is not None]


# ---------------------------------------------------------------------------
# Exit resolution — the stop-first / EOD-flat core (unit-testable in isolation).
# ---------------------------------------------------------------------------


def resolve_exit(
    trading_bars: list[OHLCVBar],
    entry_index: int,
    entry_price: float,
    stop_pct: float,
    target_r: float,
) -> tuple[int, float, str]:
    """Walk bars from entry_index and return (exit_index, exit_price, exit_reason).

    STOP-FIRST: on a bar where the range spans BOTH the stop and the target, the
    stop is returned.  See module docstring for the full exit model.  Assumes
    entry_index is a valid index into a non-empty trading_bars.
    """
    stop_price = entry_price * (1.0 - stop_pct)
    # R (risk per share) = entry - stop = entry * stop_pct; target sits target_r Rs above.
    r = entry_price - stop_price
    target_price = entry_price + target_r * r

    for j in range(entry_index, len(trading_bars)):
        bar = trading_bars[j]
        stop_hit = bar.low <= stop_price
        target_hit = bar.high >= target_price

        # STOP-FIRST: check the stop before the target so a same-bar conflict
        # resolves to the (adverse) stop.  This single ordering IS requirement 1.
        if stop_hit:
            # A gap DOWN through the stop (open below the stop) fills at the open —
            # worse than the stop price.  Otherwise fill at the stop level.
            exit_price = min(stop_price, bar.open)
            return j, exit_price, "stop"

        if target_hit:
            # Never credit a favorable gap: fill at exactly the target level even
            # if the bar opened above it.
            return j, target_price, "target"

    # EOD-flat: no stop/target touched all session → exit at the last bar's close.
    last_index = len(trading_bars) - 1
    return last_index, trading_bars[last_index].close, "eod"


def resolve_exit_trailing(
    trading_bars: list[OHLCVBar],
    entry_index: int,
    entry_price: float,
    trail_pct: float,
) -> tuple[int, float, str]:
    """Trailing-stop exit: walk bars from entry_index, return (idx, price, reason).

    ADDITIVE alternative to resolve_exit (which is left untouched).  Used only when
    config.trail_pct is set; the fixed-stop/target path is unaffected.

    Model:
      * The peak is the highest HIGH reached since entry.  It is INITIALISED to
        entry_price, so the trail starts at entry_price * (1 - trail_pct) — the
        position has a stop from the moment it opens, before any new high prints.
      * At each bar, the breach is tested against the trailing stop computed from
        the PRIOR peak — the peak WITHOUT this bar's high folded in yet.  Only if
        the bar does NOT breach do we raise the peak using this bar's high (for the
        NEXT bar).  This is the conservative within-bar ordering required: if a
        single bar both makes a new high AND its low breaches the prior-peak trail,
        we assume the adverse path (low before high) and stop off the PRIOR peak —
        the same spirit as resolve_exit's stop-first rule.
      * Exit ⟺ bar.low <= trail_level.  Gap handling matches resolve_exit: a gap
        DOWN through the trail fills at min(trail_level, bar.open) — worse.
      * If never stopped, force EOD-flat at the last bar's close.

    No lookahead: the peak (hence the trail) at bar j depends only on bars
    entry_index..j-1 plus entry_price — never a future bar's high.
    """
    # Peak starts at the entry price; the trail is anchored beneath it immediately.
    peak = entry_price

    for j in range(entry_index, len(trading_bars)):
        bar = trading_bars[j]

        # Trail computed from the PRIOR peak (this bar's high is NOT yet included) —
        # the conservative within-bar ordering.
        trail_level = peak * (1.0 - trail_pct)

        if bar.low <= trail_level:
            # A gap DOWN opening below the trail fills at the open (worse); else
            # fill at the trail level.  Mirrors resolve_exit's stop gap handling.
            exit_price = min(trail_level, bar.open)
            return j, exit_price, "trailing_stop"

        # Not stopped this bar → NOW ratchet the peak up with this bar's high so the
        # trail can only ever rise (never fall) for subsequent bars.
        if bar.high > peak:
            peak = bar.high

    # EOD-flat: never breached → exit at the last bar's close (no overnight hold).
    last_index = len(trading_bars) - 1
    return last_index, trading_bars[last_index].close, "eod"


# ---------------------------------------------------------------------------
# Per-session simulation.
# ---------------------------------------------------------------------------


def simulate_session(
    primary: PrimarySession,
    entry: EntryEvent | None,
    config: OpeningRangeBreakoutConfig,
) -> SessionOutcome:
    """Simulate one session given the strategy's entry decision.

    Returns a SessionOutcome with return_log == 0.0 and trade None when there is
    no entry, else the resolved Trade net of round-trip costs.
    """
    if entry is None:
        return SessionOutcome(
            date_et=primary.date_et, return_log=0.0, trade=None, exit_reason=None
        )

    # Select the exit engine: trailing-stop when trail_pct is set, else the
    # original fixed-stop/target logic (unchanged).  Only ONE branch runs, so the
    # fixed-stop path stays bit-for-bit identical when trail_pct is None.
    if config.trail_pct is None:
        exit_index, exit_price, reason = resolve_exit(
            primary.trading_bars,
            entry.entry_index,
            entry.entry_price,
            config.stop_pct,
            config.target_r,
        )
    else:
        exit_index, exit_price, reason = resolve_exit_trailing(
            primary.trading_bars,
            entry.entry_index,
            entry.entry_price,
            config.trail_pct,
        )
    exit_bar = primary.trading_bars[exit_index]

    # Gross long log return, then subtract the per-side cost twice (enter + exit).
    gross_log = math.log(exit_price / entry.entry_price)
    cost_rate = (config.fee_bps + config.slippage_bps) / 10000.0
    round_trip_cost = 2.0 * cost_rate
    net_log = gross_log - round_trip_cost

    trade = Trade(
        entry_time=entry.entry_time,
        exit_time=exit_bar.timestamp,
        entry_price=entry.entry_price,
        exit_price=exit_price,
        direction=1,  # ORB is long-only
        return_pct=net_log,  # NET of costs — see module docstring
        bars_held=exit_index - entry.entry_index,
    )
    return SessionOutcome(
        date_et=primary.date_et, return_log=net_log, trade=trade, exit_reason=reason
    )


# ---------------------------------------------------------------------------
# Full backtest.
# ---------------------------------------------------------------------------


def run_orb_backtest(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    strategy: IntradayStrategy,
    config: OpeningRangeBreakoutConfig,
    initial_capital: float = 1.0,
    annualization_factor: int = 252,
) -> IntradayBacktestResult:
    """Run the ORB strategy across every session and assemble the result.

    One session per ET date the primary symbol trades (the decision clock).  The
    strategy picks at most one entry per session; the sim resolves its exit.
    Per-session net log returns compound into the equity curve; metrics reuse the
    daily stack's pure functions (annualization_factor=252 → daily-frequency Sharpe).

    Args:
        bars_by_symbol:       symbol -> OHLCVBar list (primary + confirmations).
        strategy:             an IntradayStrategy (e.g. OpeningRangeBreakout).
        config:               the config the strategy was built from (its risk/cost
                              fields drive the sim; passed explicitly so the sim
                              never reaches into strategy internals).
        initial_capital:      equity-curve anchor (1.0 → curve is a growth multiple).
        annualization_factor: bars-per-year for Sharpe (252 = one obs per session).

    Returns:
        IntradayBacktestResult with equity curve, per-session returns, trades, and
        cost-aware metrics.
    """
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0, got {initial_capital}")

    aligned: list[AlignedSession] = build_sessions(
        bars_by_symbol,
        config.primary_symbol,
        list(config.confirmation_symbols),
        config.or_minutes,
    )

    outcomes: list[SessionOutcome] = []
    for session in aligned:
        entry = strategy.find_entry(session)
        outcomes.append(simulate_session(session.primary, entry, config))

    # Per-session net log returns → equity curve via cumulative-sum in log space
    # (time-additive, no repeated-multiply FP drift), matching the daily engine.
    returns = np.array([o.return_log for o in outcomes], dtype=np.float64)
    equity_curve = initial_capital * np.exp(np.cumsum(returns))

    trades = [o.trade for o in outcomes if o.trade is not None]
    cost_rate = (config.fee_bps + config.slippage_bps) / 10000.0
    total_cost = float(len(trades) * 2.0 * cost_rate)

    # Exit-reason histogram for reporting.  Includes "trailing_stop" so the
    # trailing-mode reason increments cleanly; the fixed-stop keys stay present
    # regardless of mode (a mode simply leaves the other keys at 0).
    reason_counts: dict[str, int] = {"stop": 0, "target": 0, "trailing_stop": 0, "eod": 0}
    for o in outcomes:
        if o.exit_reason is not None:
            reason_counts[o.exit_reason] += 1

    # Metrics reuse (read-only): total return over the equity curve, daily-frequency
    # Sharpe over the per-session returns, drawdown over the curve, net win rate.
    start_date = aligned[0].primary.date_et if aligned else None
    end_date = aligned[-1].primary.date_et if aligned else None
    total_return_pct = (
        metrics.total_return(equity_curve, initial_capital) if len(equity_curve) else 0.0
    )
    sharpe = metrics.sharpe_ratio(returns, annualization_factor) if len(returns) else 0.0
    max_dd = metrics.max_drawdown(equity_curve) if len(equity_curve) else 0.0
    win = metrics.win_rate(trades)

    return IntradayBacktestResult(
        strategy_name=strategy.name,
        primary_symbol=config.primary_symbol,
        # start/end are ET session dates; wrap as datetime midnight for the field type.
        start_date=datetime(start_date.year, start_date.month, start_date.day) if start_date else None,
        end_date=datetime(end_date.year, end_date.month, end_date.day) if end_date else None,
        n_sessions=len(aligned),
        n_trades=len(trades),
        equity_curve=equity_curve,
        returns=returns,
        outcomes=outcomes,
        total_return_pct=float(total_return_pct),
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=win,
        total_cost_pct=total_cost,
        exit_reason_counts=reason_counts,
    )


