# src/intraday/orb_strategy.py

"""
Opening-Range Breakout (ORB) intraday strategy — the ENTRY rule.

This module defines a NEW, multi-symbol strategy contract (IntradayStrategy)
that is PARALLEL to — never a modification of — the single-symbol daily
Strategy ABC in src/strategies/base.py.  The daily contract maps ONE symbol's
bars to a per-bar position array; ORB instead needs the traded symbol PLUS
confirmation symbols (SPY/QQQ), which the daily signature cannot express.

Division of responsibility (deliberate):
  * The STRATEGY decides ENTRY only — where, when, and at what price a long is
    opened, gated by opening-range breakout + confirmation.  It is PURE.
  * The SIM (src/intraday/sim.py) owns every EXIT — the 6% stop, the 2R target
    (stop-first on same-bar conflicts), and forced end-of-day flat.  Path-
    dependent exit modelling is the sim's job, not the strategy's.

Everything is symbol-agnostic: the traded and confirmation tickers are config
parameters (SOXL/SPY/QQQ today, TQQQ/TNA/… later), never hard-coded.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time

# Session mechanics: AlignedSession bundles the primary session + confirmations;
# as_of_bar is the backward-only sampler that makes confirmation lookahead-safe.
from src.intraday.session import AlignedSession, as_of_bar


# ---------------------------------------------------------------------------
# Config — one immutable object fully specifying the strategy + its risk levels.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpeningRangeBreakoutConfig:
    """Everything that specifies one ORB configuration.

    Entry-rule fields (primary_symbol, confirmation_symbols, or_minutes,
    entry_cutoff) are read by the strategy.  Risk/cost fields (stop_pct,
    target_r, fee_bps, slippage_bps) are read by the SIM.  Keeping them in one
    frozen object makes a configuration a single reproducible fact — the same
    spirit as LiveRotationConfig in the daily track.
    """

    # The traded symbol, e.g. "SOXL".  A parameter — never hard-coded downstream.
    primary_symbol: str
    # Symbols whose state must confirm a breakout, e.g. ("SPY", "QQQ").  tuple so
    # the frozen dataclass stays hashable/immutable.
    confirmation_symbols: tuple[str, ...] = ("SPY", "QQQ")
    # Opening-range width in minutes (09:30..09:30+or_minutes).  15 → 09:45.
    or_minutes: int = 15
    # Stop distance as a fraction below entry (0.06 == 6%).  Read by the sim.
    stop_pct: float = 0.06
    # Target as a multiple of R (risk = entry-stop).  2.0 → target = entry + 2R.
    target_r: float = 2.0
    # No NEW entries at/after this ET wall time (e.g. time(15, 0)); None = allow
    # entries any time in the trading window.  Guards against opening a position
    # seconds before the forced EOD-flat.
    entry_cutoff: time | None = None
    # Per-side transaction cost in basis points (fees + slippage), read by the sim.
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    # Trailing-stop mode selector, read by the sim.  None (default) → the sim uses
    # the fixed-stop/target exit logic unchanged.  When set (e.g. 0.05 for a 5%
    # trail), the sim ignores stop_pct/target_r and exits on a trailing stop that
    # ratchets up under the running peak.  Additive: leaving it None preserves the
    # original behaviour bit-for-bit.
    trail_pct: float | None = None


@dataclass(frozen=True)
class EntryEvent:
    """A single long entry the sim should open, in one session.

    entry_index points into the primary session's trading_bars list, so the sim
    knows where to BEGIN monitoring exits (the entry bar itself is the first bar
    the stop/target are live, under the sim's next-bar-onward exit walk).
    """

    entry_index: int      # index into AlignedSession.primary.trading_bars
    entry_time: datetime  # the breakout bar's timestamp (UTC)
    entry_price: float    # buy-stop fill: OR-high, or the bar open if it gapped through


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class IntradayStrategy(ABC):
    """Multi-symbol intraday entry contract.  Parallel to the daily Strategy ABC.

    A concrete intraday strategy inspects one AlignedSession and returns at most
    one EntryEvent (or None for "no trade this session").  It must be PURE — no
    I/O, no broker calls, no clock — so it is deterministic and unit-testable.
    Exits are the sim's responsibility, not the strategy's.
    """

    @abstractmethod
    def find_entry(self, session: AlignedSession) -> EntryEvent | None:
        """Return the long entry for this session, or None if no valid entry."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable configuration label for reports and logs."""
        ...


# ---------------------------------------------------------------------------
# Opening-Range Breakout
# ---------------------------------------------------------------------------


class OpeningRangeBreakout(IntradayStrategy):
    """Long on the first CONFIRMED break above the opening-range high.

    Rule, evaluated over the primary session's trading_bars (which already
    EXCLUDE the 09:30–09:45 opening-range window, so the "no trades in the OR"
    constraint is structural):

      1. No opening range (no OR-window bars) → no trade.
      2. Scan trading_bars in time order; a bar "breaks out" when its high
         reaches or exceeds the OR high (high >= or_high) — the level a resting
         buy-stop at or_high would trigger on.
      3. The FIRST breakout bar that is also CONFIRMED (see `confirm`) and before
         the optional entry cutoff triggers the entry.
      4. Entry fill price: or_high if the bar opened below it (the stop triggers
         at the level), else the bar's open (a gap through the level fills WORSE).

    Confirmation failing on an early breakout bar does not disqualify the day —
    scanning continues, so entry occurs on the first bar where breakout AND
    confirmation hold together.
    """

    def __init__(self, config: OpeningRangeBreakoutConfig) -> None:
        # Validate the risk/rule parameters up front so a nonsensical config
        # fails at construction, not deep in the sim.
        if config.or_minutes < 1:
            raise ValueError(f"or_minutes must be >= 1, got {config.or_minutes}")
        if not (0.0 < config.stop_pct < 1.0):
            raise ValueError(f"stop_pct must be in (0, 1), got {config.stop_pct}")
        if config.target_r <= 0.0:
            raise ValueError(f"target_r must be > 0, got {config.target_r}")
        if config.fee_bps < 0 or config.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps must be >= 0")
        # trail_pct is optional; only validate its range when the trailing mode is
        # actually engaged.  None (fixed-stop mode) is always valid.
        if config.trail_pct is not None and not (0.0 < config.trail_pct < 1.0):
            raise ValueError(f"trail_pct must be in (0, 1) when set, got {config.trail_pct}")
        self.config = config

    @property
    def name(self) -> str:
        c = self.config
        return (
            f"ORB({c.primary_symbol}, OR={c.or_minutes}m, stop={c.stop_pct:.0%}, "
            f"{c.target_r:g}R, confirm={list(c.confirmation_symbols)})"
        )

    # ------------------------------------------------------------------
    # Confirmation — overridable pluggable rule.
    # ------------------------------------------------------------------
    def confirm(self, session: AlignedSession, t: datetime) -> bool:
        """True when EVERY confirmation symbol is above its own OR high as of t.

        Fail-closed at every gap (per the alignment contract):
          * a confirmation symbol with no OR window (opening_range is None) fails;
          * a confirmation symbol with no bar at-or-before t fails;
        Any single failure means the whole confirmation fails → no entry.

        Uses as_of_bar (backward-only) so this can NEVER read a future bar: the
        confirmation state at minute t depends only on data through t.
        """
        for sym in self.config.confirmation_symbols:
            conf = session.confirmations.get(sym)
            # Symbol not attached at all, or no OR window → cannot confirm.
            if conf is None or conf.opening_range is None:
                return False
            # Sample the confirmation symbol as of the breakout minute (or the
            # last bar before it — forward-fill; never a future bar).
            sampled = as_of_bar(conf.session_bars, t)
            if sampled is None:
                return False
            # "Confirming" = trading above its own opening-range high.
            if not (sampled.close > conf.opening_range.high):
                return False
        return True

    # ------------------------------------------------------------------
    # Entry search.
    # ------------------------------------------------------------------
    def find_entry(self, session: AlignedSession) -> EntryEvent | None:
        primary = session.primary

        # (1) No opening range → no breakout level → no trade.
        if primary.opening_range is None:
            return None
        or_high = primary.opening_range.high

        cutoff = self.config.entry_cutoff

        # (2)+(3) Scan the trading window for the first confirmed breakout.
        for i, bar in enumerate(primary.trading_bars):
            # Optional entry cutoff (ET wall time): stop opening new positions
            # late in the session.  Compared in ET to match the config's intent.
            if cutoff is not None:
                from src.intraday.session import to_et  # local import: pure helper

                if to_et(bar.timestamp).time() >= cutoff:
                    break

            # Breakout test: the OR high was reached/exceeded this bar.
            if bar.high < or_high:
                continue

            # Confirmation as of THIS bar's minute (lookahead-safe).
            if not self.confirm(session, bar.timestamp):
                continue

            # (4) Buy-stop fill: at the level, or the open if the bar gapped
            # through it (a gap-up fills worse — conservative).
            entry_price = or_high if bar.open < or_high else bar.open
            return EntryEvent(entry_index=i, entry_time=bar.timestamp, entry_price=entry_price)

        # No confirmed breakout before the cutoff → no trade this session.
        return None
