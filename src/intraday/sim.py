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
from src.intraday.obsp_strategy import OpenBuySellProfitConfig
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


def resolve_exit_profit_target(
    trading_bars: list[OHLCVBar],
    entry_index: int,
    entry_price: float,
    profit_target: float,
) -> tuple[int, float, str]:
    """Profit-target exit (the OBSP "sell as soon as you're at a profit" rule).

    ADDITIVE sibling of resolve_exit / resolve_exit_trailing; those are untouched.

    Model:
      * target_price = entry_price * (1 + profit_target).
      * Walk bars from entry_index (INCLUSIVE, so the entry bar's own high can be a
        valid SAME-BAR exit).  The FIRST bar with high >= target_price exits, filled
        at EXACTLY target_price — a favorable gap that opens above the target is NOT
        credited (conservative, matching resolve_exit's target handling).
      * If the target is never reached, force-flat at the last bar's CLOSE ("eod").

    No lookahead: the walk is strictly chronological and returns the FIRST touching
    bar, so a later bar's high can never trigger an earlier exit.
    """
    target_price = entry_price * (1.0 + profit_target)

    for j in range(entry_index, len(trading_bars)):
        if trading_bars[j].high >= target_price:
            # Fill at the target level, never the (possibly higher) gap-up price.
            return j, target_price, "target"

    # Never reached the target → sell at the last bar's close (flat by close).
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


# ---------------------------------------------------------------------------
# OBSP (Open-Buy, Sell-at-Profit) — its own result type and runner.
#
# OBSP needs HONEST gross-vs-net accounting that the ORB result does not carry:
# "reached the profit target" (a GROSS win) must never be conflated with "net
# positive after costs".  So OBSP gets a dedicated result type exposing BOTH, and
# a dedicated runner — while REUSING build_sessions, resolve_exit_profit_target,
# the Trade record, and the pure metrics.  run_orb_backtest above is untouched.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OBSPOutcome:
    """One session's OBSP result, carrying BOTH gross and net returns separately."""

    date_et: date
    exit_reason: str          # "target" (a win) or "eod" (never reached target)
    gross_log: float          # log(exit/entry) BEFORE costs
    net_log: float            # gross_log minus round-trip cost
    trade: Trade | None


@dataclass(frozen=True)
class OBSPResult:
    """OBSP backtest output with gross and net kept strictly separate.

    The headline honesty guarantee: `win_rate` counts TARGET-REACHED sessions (a
    gross win), independent of whether costs pushed that trade net-negative.  The
    avg_*_win / avg_*_loss numbers report the target bucket vs the EOD bucket, in
    both gross and net terms, as SIMPLE percentages (expm1 of the log return) so
    they read naturally (+0.50%, -3.1%).  Equity/Sharpe/drawdown are NET.
    """

    strategy_name: str
    primary_symbol: str
    start_date: datetime | None
    end_date: datetime | None
    n_sessions: int
    n_trades: int

    # Distinct, non-conflated accounting (requirement 3).
    win_rate: float               # fraction of trades that REACHED the target (gross win)
    avg_gross_win: float          # mean simple-% return of target-reached trades
    avg_gross_loss: float         # mean simple-% return of EOD (never-target) trades
    avg_net_win: float            # same target bucket, net of costs
    avg_net_loss: float           # same EOD bucket, net of costs
    win_loss_size_ratio: float    # |avg_gross_win| / |avg_gross_loss| (0.0 if no losses)

    total_return_pct: float       # NET compounded return over the curve
    gross_total_return_pct: float  # GROSS compounded return (costs excluded)
    sharpe_ratio: float           # NET, daily-frequency
    max_drawdown_pct: float       # NET
    total_cost_pct: float

    equity_curve: np.ndarray
    returns: np.ndarray           # per-session NET log returns
    outcomes: list[OBSPOutcome]
    exit_reason_counts: dict[str, int]

    @property
    def trades(self) -> list[Trade]:
        return [o.trade for o in self.outcomes if o.trade is not None]


def _mean_simple_pct(logs: list[float]) -> float:
    """Mean of a list of log returns, expressed as a simple percent (expm1). 0.0 if empty."""
    if not logs:
        return 0.0
    return float(np.expm1(np.mean(logs)))


def run_obsp_backtest(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    strategy: IntradayStrategy,
    config: OpenBuySellProfitConfig,
    initial_capital: float = 1.0,
    annualization_factor: int = 252,
) -> OBSPResult:
    """Run OBSP across every session; report gross and net separately.

    Sessions are built with or_minutes=0 and no confirmation symbols, so
    primary.trading_bars is the FULL regular session starting at the 09:30 open —
    exactly what "buy at the open" needs.  Reuses resolve_exit_profit_target for the
    exit, the Trade record, and the pure metrics.
    """
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0, got {initial_capital}")

    # or_minutes=0 → no opening-range strip (full session from 09:30); no confirmation
    # symbols → OBSP has none.  Verified against session.build_sessions semantics.
    aligned: list[AlignedSession] = build_sessions(
        bars_by_symbol, config.primary_symbol, confirmation_symbols=[], or_minutes=0
    )

    cost_rate = (config.fee_bps + config.slippage_bps) / 10000.0
    round_trip_cost = 2.0 * cost_rate

    outcomes: list[OBSPOutcome] = []
    for session in aligned:
        primary = session.primary
        entry = strategy.find_entry(session)
        if entry is None:
            outcomes.append(
                OBSPOutcome(primary.date_et, "none", 0.0, 0.0, None)
            )
            continue

        exit_index, exit_price, reason = resolve_exit_profit_target(
            primary.trading_bars, entry.entry_index, entry.entry_price, config.profit_target
        )
        exit_bar = primary.trading_bars[exit_index]

        gross_log = math.log(exit_price / entry.entry_price)
        net_log = gross_log - round_trip_cost

        trade = Trade(
            entry_time=entry.entry_time,
            exit_time=exit_bar.timestamp,
            entry_price=entry.entry_price,
            exit_price=exit_price,
            direction=1,          # long-only
            return_pct=net_log,   # NET, consistent with the ORB intraday convention
            bars_held=exit_index - entry.entry_index,
        )
        outcomes.append(
            OBSPOutcome(primary.date_et, reason, gross_log, net_log, trade)
        )

    # Per-session NET log returns → equity curve (cumsum/exp, matching the ORB sim).
    net_logs = np.array([o.net_log for o in outcomes], dtype=np.float64)
    equity_curve = initial_capital * np.exp(np.cumsum(net_logs))

    trades = [o.trade for o in outcomes if o.trade is not None]
    total_cost = float(len(trades) * round_trip_cost)

    reason_counts: dict[str, int] = {"target": 0, "eod": 0}
    for o in outcomes:
        if o.exit_reason in reason_counts:
            reason_counts[o.exit_reason] += 1

    # Buckets: WIN = target reached; LOSS = EOD (never reached target).  Kept
    # separate in BOTH gross and net terms — the crux of the honest accounting.
    win_gross = [o.gross_log for o in outcomes if o.exit_reason == "target"]
    loss_gross = [o.gross_log for o in outcomes if o.exit_reason == "eod"]
    win_net = [o.net_log for o in outcomes if o.exit_reason == "target"]
    loss_net = [o.net_log for o in outcomes if o.exit_reason == "eod"]

    avg_gross_win = _mean_simple_pct(win_gross)
    avg_gross_loss = _mean_simple_pct(loss_gross)
    avg_net_win = _mean_simple_pct(win_net)
    avg_net_loss = _mean_simple_pct(loss_net)
    win_loss_size_ratio = (
        abs(avg_gross_win) / abs(avg_gross_loss) if avg_gross_loss != 0.0 else 0.0
    )

    win_rate = (len(win_gross) / len(trades)) if trades else 0.0

    # NET metrics via the reused pure functions; GROSS compounded return computed
    # directly from the gross log sum so it is genuinely cost-free.
    total_return_pct = (
        metrics.total_return(equity_curve, initial_capital) if len(equity_curve) else 0.0
    )
    gross_total_return_pct = float(np.expm1(np.sum([o.gross_log for o in outcomes])))
    sharpe = metrics.sharpe_ratio(net_logs, annualization_factor) if len(net_logs) else 0.0
    max_dd = metrics.max_drawdown(equity_curve) if len(equity_curve) else 0.0

    start_date = aligned[0].primary.date_et if aligned else None
    end_date = aligned[-1].primary.date_et if aligned else None

    return OBSPResult(
        strategy_name=strategy.name,
        primary_symbol=config.primary_symbol,
        start_date=datetime(start_date.year, start_date.month, start_date.day) if start_date else None,
        end_date=datetime(end_date.year, end_date.month, end_date.day) if end_date else None,
        n_sessions=len(aligned),
        n_trades=len(trades),
        win_rate=win_rate,
        avg_gross_win=avg_gross_win,
        avg_gross_loss=avg_gross_loss,
        avg_net_win=avg_net_win,
        avg_net_loss=avg_net_loss,
        win_loss_size_ratio=win_loss_size_ratio,
        total_return_pct=float(total_return_pct),
        gross_total_return_pct=gross_total_return_pct,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        total_cost_pct=total_cost,
        equity_curve=equity_curve,
        returns=net_logs,
        outcomes=outcomes,
        exit_reason_counts=reason_counts,
    )


