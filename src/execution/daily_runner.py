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

LOGGING + NOTIFICATIONS: logging is stdlib (log = logging.getLogger(__name__)); this
module configures NO handlers/format (the caller owns that).  The notifier is INJECTED
into run_daily (notify_fn, default None = no notification) and gated by a NotifyPolicy
(default ACTED_AND_FAILED).  A failed rebalance is logged (with traceback) and notified
BEFORE the exception propagates; a raising notifier is caught and logged separately, so a
notification-delivery failure can NEVER mask the trading result or the original error.
The missing-reference config error is logged but does NOT fire the notifier (a config bug
is not a trading alert).
"""

# Modern type-hint syntax (datetime | None, dict[str, float]) without quoting.
from __future__ import annotations

# Standard-library logging: this module uses a module-level logger named `log`
# (defined below), matching cli_common.py's `log = logging.getLogger(__name__)` idiom.
# It configures NO handlers/format; the caller (scheduler/entry point) owns that.
import logging

# Callable types the injected notifier parameter (notify_fn) on run_daily's signature.
from typing import Callable

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

# Notification payload, policy, and pure helpers from the notify seam.  The actual delivery
# backend is an INJECTED callable owned by the caller; this module only builds the payload,
# applies the policy (should_notify), and formats the human line (format_notification).
from src.execution.notify import (
    Notification,
    NotifyPolicy,
    should_notify,
    format_notification,
)

# most_recent_completed_month_end is the LIVE right-edge resolver: it drops an
# in-progress current month so the decision never rests on a partial-month window.
from src.strategies.time_series_momentum import most_recent_completed_month_end


# Module-level logger, named `log` per cli_common.py's idiom; getLogger(__name__) ties
# records to this module so the caller's log config can filter them without extra setup.
log = logging.getLogger(__name__)


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
# _emit — log the outcome and (policy-gated) fire the injected notifier.
# ---------------------------------------------------------------------------

def _emit(
    outcome: str,                                   # "acted" | "no-op" | "failed"
    decision: RebalanceDecision,                    # the decision this run produced
    orders: tuple[tuple[str, str, int], ...],       # (symbol, side, qty) per submitted order
    target: dict[str, float],                       # the target weights when acting, else {}
    error: str | None,                              # str(exception) on failure, else None
    notify_fn,                                       # injected Callable[[Notification], None] | None
    notify_policy: NotifyPolicy,                     # gates WHEN notify_fn fires
) -> None:
    """Log the run outcome and, if the policy allows, deliver it to the injected notifier.

    Logging always happens; notification is gated by should_notify(outcome, policy) and only
    fires when a notifier is injected.  A notifier that RAISES is caught and logged
    separately so it can never propagate or mask the caller's trading result / exception.
    """
    # Build the immutable payload once for both the log line and the notifier.  On a
    # failure the human-facing `reason` is the short failure summary (str(exc)); otherwise
    # it is the decision's own reason (matching Notification.reason's documented contract).
    notification = Notification(
        outcome=outcome,
        reason=(error if outcome == "failed" else decision.reason),
        as_of_month_end=decision.as_of_month_end,
        orders=orders,
        target_weights=target,
        error=error,
    )

    # Render the compact one-line human summary shared by the log record and the notifier.
    line = format_notification(notification)

    # LOG at the level matching the outcome.
    if outcome == "failed":
        # log.exception is invoked from within the except block (see run_daily's failure
        # wiring), so it records at ERROR level AND attaches the active traceback.
        log.exception(line)
    else:
        # "acted" and "no-op" are both routine INFO records.
        log.info(line)

    # NOTIFY only when a notifier is injected AND the policy says to fire for this outcome.
    if notify_fn is not None and should_notify(outcome, notify_policy):
        # Deliver inside its OWN try/except so a raising notifier is CONTAINED: the delivery
        # failure is logged separately and swallowed, never re-raised, so it cannot change
        # the trading outcome or mask the caller's original exception.
        try:
            notify_fn(notification)
        except Exception as exc:  # noqa: BLE001 - delivery failure must never affect trading
            # Separate ERROR record; deliberately swallowed (no re-raise) per the invariant.
            log.error("notifier failed: %s", exc)


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
    # INJECTED notifier (default None = no notification), mirroring the is_open_fn seam so
    # the network/secrets stay at the caller's impure edge and tests can pass a fake.
    notify_fn: Callable[[Notification], None] | None = None,
    # Policy gating WHEN notify_fn fires (default: acted + failed only, no daily no-op spam).
    notify_policy: NotifyPolicy = NotifyPolicy.ACTED_AND_FAILED,
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
        notify_fn:         optional injected notifier called with a Notification; None -> no
                           notification (logging still happens).  Owned by the caller, so all
                           network/secrets stay at the impure edge.
        notify_policy:     when notify_fn fires (default ACTED_AND_FAILED: acted + failed only).

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
        # LOG the config error at ERROR level.  A missing reference is a CONFIG bug, not a
        # trading event, so the notifier is deliberately NOT fired here.
        log.error(
            "reference_symbol %r not in bars_by_symbol (keys: %s)",
            reference_symbol,
            sorted(bars_by_symbol.keys()),
        )
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

    # STEP e — NO-OP PATH: if we are not acting, log + (policy-gated) notify, then return the
    # decision unchanged.  No state write, no broker calls beyond the market-open check above.
    if not decision.should_act:
        # No-op emit: empty orders, empty target, no error; decision.reason carries why.
        _emit(
            outcome="no-op",
            decision=decision,
            orders=(),
            target={},
            error=None,
            notify_fn=notify_fn,
            notify_policy=notify_policy,
        )
        return decision

    # STEP f — ACT: form today's target weights through the SHARED seam (live and backtest
    # cannot drift), then reconcile the account to them via the already-built runner.
    target = live_target_weights(bars_by_symbol, today, reference_symbol, config)

    # run_rebalance is the IO shell that prices, sizes, and submits the orders.  Wrap it so a
    # failure is logged + (policy-gated) notified BEFORE the exception propagates; the bare
    # raise below keeps the Day-57 crux intact (state is NOT written on a failed rebalance).
    try:
        # Capture the per-order results so the SUCCESS notification can list the orders.
        results = run_rebalance(broker, target)
    except Exception as exc:  # noqa: BLE001 - re-raised below after logging + notifying
        # FAILURE emit: reason/error carry str(exc); _emit logs via log.exception (capturing
        # the active traceback) and fires the notifier per policy.  No state write here.
        _emit(
            outcome="failed",
            decision=decision,
            orders=(),
            target=target,
            error=str(exc),
            notify_fn=notify_fn,
            notify_policy=notify_policy,
        )
        # Bare raise: preserve the original traceback and propagate, so the month-end is
        # retried next run and the state file stays untouched (the Day-57 crux holds).
        raise

    # STEP g — STATE WRITE ONLY AFTER SUCCESS: reached ONLY because run_rebalance returned
    # without raising.  We record the acted-on month-end so subsequent runs this month
    # no-op (idempotency).  A failed rebalance never reaches this line.
    _write_state(state_path, decision.as_of_month_end)

    # SUCCESS emit: AFTER state is durably written, log + (policy-gated) notify, listing the
    # submitted orders as (symbol, side, qty) tuples for the human-readable summary.
    _emit(
        outcome="acted",
        decision=decision,
        orders=tuple((r.symbol, r.side.value, r.qty) for r in results),
        target=target,
        error=None,
        notify_fn=notify_fn,
        notify_policy=notify_policy,
    )

    # STEP h — return the decision we acted on.
    return decision
