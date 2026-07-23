# src/intraday/obsp_strategy.py

"""
Open-Buy, Sell-at-Profit (OBSP) intraday strategy — the ENTRY rule.

The claim under test: "buy at the open, sell as soon as you're at a profit, and
if you never go positive, sell at the close."  This is the small-frequent-wins /
rare-large-losses archetype; the honest question is whether the tiny target-hits
are outweighed by the big losses on days price falls from the open and never
recovers.  The sim (src/intraday/sim.resolve_exit_profit_target + run_obsp_backtest)
answers it with GROSS and NET accounting kept strictly separate.

This strategy implements the SAME multi-purpose IntradayStrategy contract as ORB
(src/intraday/orb_strategy.py) but is deliberately trivial on entry: it buys the
FIRST regular-session bar's open.  No opening-range wait, no confirmation symbols.
The interesting behaviour lives entirely in the exit (the sim's job).

Symbol-agnostic: the traded ticker is a config parameter (SOXL today, anything
later) — never hard-coded.
"""

from dataclasses import dataclass

# Reuse the intraday strategy contract + entry-event value object defined alongside
# ORB.  OBSP is a peer strategy under the same ABC — no new contract needed.
from src.intraday.orb_strategy import EntryEvent, IntradayStrategy
from src.intraday.session import AlignedSession


@dataclass(frozen=True)
class OpenBuySellProfitConfig:
    """Everything specifying one OBSP configuration.

    profit_target is read by the sim's exit resolver; fee_bps/slippage_bps drive the
    sim's per-side cost.  Frozen so a configuration is a single reproducible fact,
    matching OpeningRangeBreakoutConfig / LiveRotationConfig.
    """

    # The traded symbol, e.g. "SOXL".  A parameter — never hard-coded downstream.
    primary_symbol: str
    # Profit target as a fraction above entry (0.005 == +0.5%).  Read by the sim.
    profit_target: float = 0.005
    # Per-side transaction cost in basis points (fees + slippage), read by the sim.
    fee_bps: float = 0.0
    slippage_bps: float = 0.0


class OpenBuySellProfit(IntradayStrategy):
    """Buy the first regular-session bar's open; the sim handles the profit exit."""

    def __init__(self, config: OpenBuySellProfitConfig) -> None:
        # profit_target must be a positive fraction below 1 (a +100%+ intraday
        # target is nonsensical for this strategy).  Fail at construction.
        if not (0.0 < config.profit_target < 1.0):
            raise ValueError(
                f"profit_target must be in (0, 1), got {config.profit_target}"
            )
        if config.fee_bps < 0 or config.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be >= 0")
        self.config = config

    @property
    def name(self) -> str:
        c = self.config
        return f"OBSP({c.primary_symbol}, target={c.profit_target:.2%})"

    def find_entry(self, session: AlignedSession) -> EntryEvent | None:
        """Enter at the FIRST regular-session bar's open (09:30 ET), or None if empty.

        The sim builds sessions with or_minutes=0, so primary.trading_bars is the FULL
        regular session and trading_bars[0] is the 09:30 open bar.  Entry index 0 means
        exit monitoring begins on the entry bar itself (a same-bar target hit is valid).
        """
        bars = session.primary.trading_bars
        if not bars:
            return None
        first = bars[0]
        return EntryEvent(
            entry_index=0,
            entry_time=first.timestamp,
            entry_price=first.open,  # buy AT THE OPEN — the defining feature
        )
