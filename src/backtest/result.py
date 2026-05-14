# src/backtest/result.py

"""
Immutable result objects produced by a backtest run.

These dataclasses are the transport format between the backtester and any
downstream consumer (reports, plotting, parameter optimization, ML training).
Frozen so they can't be mutated after construction — backtest results are
historical facts; mutating them is always a bug.

The Trade dataclass represents a single round-trip (entry → exit).
The BacktestResult dataclass holds everything a backtest produces: equity
curve aligned to bars, list of trades, and pre-computed performance metrics.
"""

# dataclass turns a plain class into a structured data container by
# auto-generating __init__, __repr__, and __eq__ from the field list.
# field is used to declare per-field metadata (e.g. default_factory) when
# a simple default value isn't expressive enough.
from dataclasses import dataclass, field

# datetime is used for the entry/exit timestamps on Trade and the start/end
# timestamps on BacktestResult.  All timestamps in this project are UTC.
from datetime import datetime

# numpy is the storage type for dense per-bar time series (equity curve and
# per-bar returns).  Metric calculations operate on these arrays directly.
import numpy as np


# ---------------------------------------------------------------------------
# Trade — one completed round-trip (entry → exit).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)  # frozen=True → immutable + hashable; see module docstring
class Trade:
    """
    A single round-trip position: opened at entry_time, closed at exit_time.

    One Trade per completed entry/exit pair.  Open positions at the end of a
    backtest are either force-closed on the final bar or excluded — the
    backtester decides; this dataclass only represents closed trades.
    """

    entry_time: datetime   # when the position was opened (UTC); aligns with a bar timestamp
    exit_time: datetime    # when the position was closed (UTC); aligns with a bar timestamp
    entry_price: float     # close price at entry — fills are assumed at the close of the signal bar
    exit_price: float      # close price at exit — same close-fill assumption as entry
    direction: int         # +1 (long) or -1 (short); matches the SIGNAL constants used by strategies
    return_pct: float      # signed log return of this trade; sign already reflects direction
    bars_held: int         # bars from entry to exit (inclusive of entry, exclusive of exit)


# ---------------------------------------------------------------------------
# BacktestResult — the full output of one backtest run.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)  # frozen=True → immutable + hashable; see module docstring
class BacktestResult:
    """
    Everything one backtest run produces.

    Time-series fields (equity_curve, returns) are aligned 1:1 with the input
    bar count — same NaN-safe alignment contract as indicators and strategies.
    Trades are sparse and stored as a list of records rather than parallel
    arrays because trade-level analytics read more naturally that way.
    """

    strategy_name: str     # from strategy.name; used as the label in reports and plots
    start_date: datetime   # first bar's timestamp (UTC); inclusive
    end_date: datetime     # last bar's timestamp (UTC); inclusive
    n_bars: int            # total bars simulated; equal to len(equity_curve) and len(returns)

    # Time series — dense, one value per input bar.  Stored as float64 numpy
    # arrays because the metric calculations (Sharpe, drawdown) and plotting
    # libraries (matplotlib, plotly) consume ndarrays natively and vastly
    # faster than Python lists.
    equity_curve: np.ndarray   # float64, len == n_bars; starts at 1.0 and compounds per-bar returns
    returns: np.ndarray        # float64, len == n_bars; per-bar log returns of the strategy

    # Trades — sparse list of round-trip records.  A 5-year SMA crossover
    # backtest might produce ~7 trades, so a list of dataclasses is more
    # readable than parallel arrays.  default_factory=list gives each instance
    # its own empty list (a bare default=[] would share one list across all
    # instances, which is the classic mutable-default footgun).
    trades: list[Trade] = field(default_factory=list)  # one entry per completed round-trip

    # Pre-computed metrics — calculated once at construction time so downstream
    # consumers (reports, optimization loops) don't recompute them on every read.
    total_return_pct: float = 0.0   # final_equity / initial_equity - 1, as a fraction (0.25 → +25%)
    sharpe_ratio: float = 0.0       # annualized Sharpe ratio; assumes 252 trading days per year
    max_drawdown_pct: float = 0.0   # worst peak-to-trough decline as a positive fraction (0.20 → 20% drawdown)
    win_rate: float = 0.0           # fraction of trades with positive return_pct; 0.0 when n_trades == 0
    n_trades: int = 0               # len(trades); cached so callers don't recompute len() repeatedly
