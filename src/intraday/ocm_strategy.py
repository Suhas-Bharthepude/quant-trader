# src/intraday/ocm_strategy.py

"""
Opening-Candle Momentum (OCM) intraday strategy — the ENTRY rule.

The claim under test: "if the first N one-minute candles after the open are green
(close > open), enter long on the next candle; if any are red, take no trade."
This is a momentum-continuation bet — the easiest kind to fool yourself on, because
you remember the mornings it worked.  The honest evaluation therefore lives in the
sim/runner (train/test split + a buy-the-open baseline), not here; this file is just
the pure entry rule.

Implements the SAME multi-purpose IntradayStrategy contract as ORB and OBSP.  The
exit is chosen by the runner via config.exit_mode ("close" or "bracket"), not by
this strategy — a strategy decides ENTRY, the sim owns EXITS.

Symbol-agnostic: the traded ticker is a config parameter (SOXL today, anything
later) — never hard-coded.
"""

from dataclasses import dataclass

from src.intraday.orb_strategy import EntryEvent, IntradayStrategy
from src.intraday.session import AlignedSession


# The exit modes OCM supports.  "close" = hold to the session close; "bracket" =
# fixed profit-target + stop (resolved by the reused resolve_exit in sim.py).
VALID_EXIT_MODES = ("close", "bracket")


@dataclass(frozen=True)
class OpeningCandleMomentumConfig:
    """Everything specifying one OCM configuration.

    n_candles/exit_mode drive entry + exit selection; profit_target/stop_pct are
    only used when exit_mode == "bracket"; fee_bps/slippage_bps drive the sim's
    per-side cost.  Frozen — a configuration is a single reproducible fact.
    """

    # The traded symbol, e.g. "SOXL".  A parameter — never hard-coded downstream.
    primary_symbol: str
    # How many opening candles must ALL be green to fire the signal.
    n_candles: int = 2
    # "close" → hold to session close; "bracket" → fixed profit-target + stop.
    exit_mode: str = "close"
    # Bracket profit target as a fraction above entry (0.005 == +0.5%).
    profit_target: float = 0.005
    # Bracket stop as a fraction below entry (0.01 == 1%).
    stop_pct: float = 0.01
    # Per-side transaction cost in basis points (fees + slippage).
    fee_bps: float = 0.0
    slippage_bps: float = 0.0


class OpeningCandleMomentum(IntradayStrategy):
    """Enter long at candle N+1's open when the first N candles are all green."""

    def __init__(self, config: OpeningCandleMomentumConfig) -> None:
        if config.n_candles < 1:
            raise ValueError(f"n_candles must be >= 1, got {config.n_candles}")
        if config.exit_mode not in VALID_EXIT_MODES:
            raise ValueError(
                f"exit_mode must be one of {VALID_EXIT_MODES}, got {config.exit_mode!r}"
            )
        # profit_target/stop_pct only matter for the bracket exit, but validate them
        # whenever bracket is selected so a bad bracket config fails at construction.
        if config.exit_mode == "bracket":
            if not (0.0 < config.profit_target < 1.0):
                raise ValueError(f"profit_target must be in (0, 1), got {config.profit_target}")
            if not (0.0 < config.stop_pct < 1.0):
                raise ValueError(f"stop_pct must be in (0, 1), got {config.stop_pct}")
        if config.fee_bps < 0 or config.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be >= 0")
        self.config = config

    @property
    def name(self) -> str:
        c = self.config
        exit_desc = (
            "exit@close" if c.exit_mode == "close"
            else f"bracket({c.profit_target:.2%}/{c.stop_pct:.2%})"
        )
        return f"OCM({c.primary_symbol}, N={c.n_candles}, {exit_desc})"

    def find_entry(self, session: AlignedSession) -> EntryEvent | None:
        """Enter at candle N+1's OPEN iff the first N candles are all green, else None.

        Uses ONLY candles 0..N-1 for the decision (no lookahead) and fills at candle
        N's OPEN — knowable the instant candle N opens, NOT its close.  Sessions are
        built with or_minutes=0, so trading_bars is the FULL regular session and
        trading_bars[0] is the 09:30 open candle.
        """
        bars = session.primary.trading_bars
        n = self.config.n_candles

        # Need at least the N signal candles PLUS the entry candle (index N).
        if len(bars) < n + 1:
            return None

        # Signal: every one of the first N candles must be green (close > open).
        # Strict '>' means a doji (close == open) is NOT green → no trade.
        for i in range(n):
            if not (bars[i].close > bars[i].open):
                return None

        # All green → enter long at candle N+1's OPEN (0-indexed candle N).
        entry_bar = bars[n]
        return EntryEvent(
            entry_index=n,
            entry_time=entry_bar.timestamp,
            entry_price=entry_bar.open,  # the OPEN — not the close (no lookahead)
        )
