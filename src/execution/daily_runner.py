# src/execution/daily_runner.py

"""
Daily autopilot runner: the entry point that makes the rotation bot autonomous.

It wakes once per invocation, decides whether to act, and if so computes today's
target weights and rebalances the account; otherwise it no-ops cleanly.  It is
safe to run every day: it acts AT MOST ONCE per new completed month-end and
no-ops on every other day (idempotent).

TWO-GATE STRUCTURE (both must be true to act):
  - GATE 2 (market): the market must be open (checked FIRST, so a closed-market
    day is a clean no-op regardless of month-end state).
  - GATE 1 (new month-end): the most recent COMPLETED month-end must differ from
    the last month-end we already acted on (so we rebalance once per new month-end
    and never twice within the same month).

PURE / IO SPLIT (mirrors live_target_weights vs run_rebalance already in this repo):
  - decide_rebalance(...) is PURE: no clock, no broker, no filesystem.  It takes a
    plain bool for the market gate and returns a RebalanceDecision.  Fully
    unit-testable with plain data.
  - run_daily(...) is the THIN IO SHELL: it evaluates the market-open callable,
    reads/writes the state file, and calls the already-built live_target_weights +
    run_rebalance.  ALL I/O lives here; it reimplements no weight or order logic.

STATE IS WRITTEN ONLY AFTER SUCCESS: the "last acted-on month-end" is persisted
strictly AFTER run_rebalance returns without raising.  A failed/partial rebalance
propagates its exception and leaves the state file untouched, so the same month-end
is retried on the next run.  NOTE: a returned run_rebalance means the orders were
ACCEPTED by the broker, not necessarily FILLED (fills are async); confirming fills
before recording the month-end is a documented later refinement, not done here.

cost_rate deliberately does NOT appear in this module: it is an execution/accounting
concern (later performance reporting), never part of the act/no-act decision logic.
"""

# Modern type-hint syntax (datetime | None, dict[str, float]) without quoting.
from __future__ import annotations

# dataclass builds the frozen RebalanceDecision result; frozen=True makes each
# decision an immutable value, matching LiveRotationConfig / OrderResult.
from dataclasses import dataclass

# datetime types the `today` argument and the resolved month-end; fromisoformat
# parses the persisted ISO string back into a datetime in _read_state.
from datetime import datetime

# json (de)serialises the tiny state file; Path gives the mkdir/exists/write API
# used the same way scripts/limit_order.py persists logs/last_limit_order_id.txt.
import json
from pathlib import Path

# Broker is the interface type of the injected broker; run_daily programs only to
# the interface, never to a concrete broker (AlpacaBroker in production).
from src.brokers.base import Broker

# OHLCVBar is the bar schema each symbol's list holds (typed on the signature).
from src.data.schema import OHLCVBar

# LIVE_CONFIG is the default shipped rotation config; live_target_weights forms the
# target weights through the shared single_date_weights seam (no drift from backtest).
from src.execution.live_target import LIVE_CONFIG, LiveRotationConfig, live_target_weights

# run_rebalance is the already-built IO shell that reconciles the account to a target
# weight dict and submits the orders; run_daily orchestrates it, never re-does its work.
from src.execution.runner import run_rebalance

# most_recent_completed_month_end is the LIVE right-edge resolver: it drops an
# in-progress current month so the decision never rests on a partial-month window.
from src.strategies.time_series_momentum import most_recent_completed_month_end


# ---------------------------------------------------------------------------
# RebalanceDecision — the immutable outcome of the pure decision core.
# ---------------------------------------------------------------------------

# frozen=True → an immutable record of what the runner decided and why.
@dataclass(frozen=True)
class RebalanceDecision:
    """Whether to act, the month-end that justifies it, and a short human reason."""

    # True when the runner should rebalance now; False on every no-op path.
    should_act: bool
    # The completed month-end whose appearance justifies acting; None when not acting.
    as_of_month_end: datetime | None
    # Short human string: "acted", "no new month-end", "market closed",
    # "no completed month-end yet".
    reason: str


# ---------------------------------------------------------------------------
# decide_rebalance — the PURE decision core (no IO, no clock, no broker).
# ---------------------------------------------------------------------------

def decide_rebalance(
    today: datetime,
    reference_bars: list[OHLCVBar],
    last_acted_month_end: datetime | None,
    is_market_open: bool,
) -> RebalanceDecision:
    """Decide whether to rebalance today, purely from data + two gate inputs.

    Args:
        today:                the live "as of" instant (tz-aware UTC to match bars).
        reference_bars:       the reference spine's bars; its month-ends are the schedule.
        last_acted_month_end: the month-end we last rebalanced on, or None if never.
        is_market_open:       the already-evaluated market gate (the shell resolves the
                              broker/callable; this stays a plain bool so the core is pure).

    Returns:
        A RebalanceDecision; should_act is True only when the market is open AND a new
        completed month-end has appeared since last_acted_month_end.
    """
    # GATE 2 (market) FIRST: a closed market is a clean no-op regardless of the
    # month-end state, so we never even resolve the grid on a closed day.
    if not is_market_open:
        return RebalanceDecision(False, None, "market closed")

    # Resolve the as-of decision date: the most recent COMPLETED month-end at-or-before
    # today (an in-progress current month is dropped by the resolver).
    as_of = most_recent_completed_month_end(reference_bars, today)

    # No completed month-end yet: too little history, or only a partial current month
    # is visible.  Nothing to act on -> clean no-op.
    if as_of is None:
        return RebalanceDecision(False, None, "no completed month-end yet")

    # GATE 1 (new month-end) / IDEMPOTENCY: if we have already acted on THIS exact
    # completed month-end, do nothing -> running again the same month is a no-op.
    if last_acted_month_end is not None and as_of == last_acted_month_end:
        return RebalanceDecision(False, None, "no new month-end")

    # Both gates pass: the market is open and a new completed month-end has appeared,
    # so we should rebalance and record `as_of` as the month-end we acted on.
    return RebalanceDecision(True, as_of, "acted")


# ---------------------------------------------------------------------------
# State persistence helpers — the ONLY durable state the runner keeps.
# ---------------------------------------------------------------------------

def _read_state(path: Path) -> datetime | None:
    """Read the last acted-on month-end from the JSON state file, or None if absent.

    A missing file means "first-ever run" (no month-end acted on yet) -> None.
    """
    # First-ever run (or a wiped state dir): no file -> nothing acted on yet.
    if not path.exists():
        return None

    # Parse the small JSON document {"last_acted_month_end": "<iso>"}.
    data = json.loads(path.read_text())

    # Pull the ISO timestamp string out of the document.
    iso = data["last_acted_month_end"]

    # Rehydrate it into a tz-aware datetime (fromisoformat round-trips the isoformat()
    # written below, preserving the UTC offset).
    return datetime.fromisoformat(iso)


def _write_state(path: Path, dt: datetime) -> None:
    """Persist the acted-on month-end as JSON, creating the parent dir if needed."""
    # Ensure the containing dir exists (mirrors scripts/limit_order.py's Path pattern:
    # parents=True builds intermediate dirs, exist_ok=True is a no-op if present).
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise the single-field document; isoformat() writes a round-trippable ISO
    # string that _read_state parses back with fromisoformat.
    path.write_text(json.dumps({"last_acted_month_end": dt.isoformat()}))


# ---------------------------------------------------------------------------
# run_daily — the THIN IO SHELL (all I/O lives here, nothing else).
# ---------------------------------------------------------------------------

def run_daily(
    broker: Broker,
    bars_by_symbol: dict[str, list[OHLCVBar]],
    today: datetime,
    reference_symbol: str = "SPY",
    state_path: Path = Path("logs/rebalance_state.json"),
    config: LiveRotationConfig = LIVE_CONFIG,
    is_open_fn=None,
) -> RebalanceDecision:
    """Wake, decide, and (only if a new completed month-end appeared) rebalance.

    Orchestration only: it evaluates the market gate, reads/writes state, and calls
    live_target_weights + run_rebalance.  It reimplements NO weight or order logic.

    Args:
        broker:           the (paper) Broker; run_rebalance's own guard enforces paper-only.
        bars_by_symbol:   symbol -> that symbol's bars, loaded through today by the caller.
        today:            the live "as of" instant (tz-aware UTC).
        reference_symbol: the spine whose month-ends define the schedule (default "SPY").
        state_path:       JSON file holding the last acted-on month-end (default under logs/).
        config:           rotation parameters passed to live_target_weights (default LIVE_CONFIG).
        is_open_fn:        optional injected market-open callable; None -> broker.is_market_open.

    Returns:
        The RebalanceDecision made this run.  On a no-op path no state is written and no
        broker calls beyond the market-open check are made.

    Raises:
        ValueError:  reference_symbol not in bars_by_symbol.
        (propagated) whatever run_rebalance raises on a failed/partial rebalance — in which
                     case the state file is deliberately NOT updated, so the same month-end
                     is retried on the next run.
    """
    # STEP a — MARKET GATE via an INJECTED callable.  Default calls the broker's live
    # clock; tests inject a lambda returning a canned bool.  This is the seam that lets a
    # trading-day calendar check replace the intraday clock later WITHOUT touching the
    # pure decide_rebalance.
    is_open = (is_open_fn or (lambda: broker.is_market_open()))()

    # STEP b — READ the last acted-on month-end from the state file (None on first-ever run).
    last_acted = _read_state(state_path)

    # STEP c — GUARD: the reference symbol must be present, since its month-ends ARE the
    # schedule.  Echo the missing name so an absent/typo'd spine fails loud and legibly.
    if reference_symbol not in bars_by_symbol:
        raise ValueError(
            f"reference_symbol {reference_symbol!r} not in bars_by_symbol "
            f"(keys: {sorted(bars_by_symbol.keys())})"
        )

    # STEP d — DECIDE (pure): feed today, the reference spine's bars, the persisted state,
    # and the resolved market bool into the pure core.
    decision = decide_rebalance(
        today,
        bars_by_symbol[reference_symbol],
        last_acted,
        is_open,
    )

    # STEP e — NO-OP PATH: if we are not acting, return the decision unchanged.  No state
    # write, no broker calls beyond the market-open check above.
    if not decision.should_act:
        return decision

    # STEP f — ACT: form today's target weights through the SHARED seam (live and backtest
    # cannot drift), then reconcile the account to them via the already-built runner.
    target = live_target_weights(bars_by_symbol, today, reference_symbol, config)

    # run_rebalance is the IO shell that prices, sizes, and submits the orders.  If it
    # raises (a rejected/partial rebalance), the exception propagates out of run_daily and
    # the state write below is skipped -> the month-end is retried next run.
    run_rebalance(broker, target)

    # STEP g — STATE WRITE ONLY AFTER SUCCESS: reached ONLY because run_rebalance returned
    # without raising.  We record the acted-on month-end so subsequent runs this month
    # no-op (idempotency).  A failed rebalance never reaches this line.
    _write_state(state_path, decision.as_of_month_end)

    # STEP h — return the decision we acted on.
    return decision
