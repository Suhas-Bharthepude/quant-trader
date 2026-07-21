# src/execution/notify.py

"""
Notification SEAM for the daily runner: payload type, policy, and formatter only.

This module is deliberately PURE and delivery-agnostic.  It defines WHAT a run
reports (the Notification payload), WHEN a report should fire (the NotifyPolicy +
the pure should_notify predicate), and HOW it reads as one human line
(format_notification).  It contains NO real delivery backend: no Slack, no email,
no webhook, no HTTP client, no secrets, no environment reads, no print().

The ACTUAL notifier is an injected callable owned by run_daily's caller (the
scheduler/entry point), so the network/secrets stay at the impure edge and this
module remains unit-testable with plain data.  That mirrors the codebase's
pure-predicate style (scripts/data_health.should_alert) and the pure/IO split used
throughout (decide_rebalance pure vs run_daily IO).
"""

# Modern type-hint syntax (datetime | None, str | None) without quoting.
from __future__ import annotations

# dataclass builds the frozen Notification payload; field supplies the error default.
from dataclasses import dataclass, field

# datetime types the as_of_month_end field (the completed month-end acted on).
from datetime import datetime

# Enum backs NotifyPolicy; subclassing str makes the members compare/serialise as their
# string values (so a policy round-trips cleanly through config/JSON if a caller wants).
from enum import Enum


# ---------------------------------------------------------------------------
# Notification — the immutable payload a human reads (phone / log line).
# ---------------------------------------------------------------------------

# frozen=True -> an immutable record of one run's outcome, matching RebalanceDecision.
@dataclass(frozen=True)
class Notification:
    """Everything a human needs to understand what the runner did on one day."""

    # "acted" | "no-op" | "failed" -- the coarse outcome class.
    outcome: str
    # The RebalanceDecision.reason on a no-op, or a short failure summary on failure.
    reason: str
    # The completed month-end acted on (datetime), or None when not acting.
    as_of_month_end: datetime | None
    # (symbol, side, qty) per submitted order; empty tuple when none were placed.
    orders: tuple[tuple[str, str, int], ...]
    # The target weight dict when acted (symbol -> weight), else {}.
    target_weights: dict[str, float]
    # str(exception) on a failure path, else None (defaulted so success callers omit it).
    error: str | None = field(default=None)


# ---------------------------------------------------------------------------
# NotifyPolicy — controls WHEN the injected notifier actually fires.
# ---------------------------------------------------------------------------

# str + Enum: members ARE their string values, so a policy is config-friendly.
class NotifyPolicy(str, Enum):
    """When the injected notifier fires (logging happens regardless of this policy)."""

    # DEFAULT: fire on acted + failed only -- no daily no-op spam from an idle runner.
    ACTED_AND_FAILED = "acted_and_failed"
    # Fire on EVERY outcome (acted, no-op, failed) -- verbose / debugging mode.
    ALL = "all"
    # Never fire the notifier (a log line is still emitted by the caller).
    NEVER = "never"


def should_notify(outcome: str, policy: NotifyPolicy) -> bool:
    """Pure predicate: should the injected notifier fire for this outcome under this policy?

    Mirrors data_health.should_alert's pure-decision style so the fire/skip policy is
    unit-testable without any delivery side effect.

    Args:
        outcome: "acted" | "no-op" | "failed".
        policy:  the configured NotifyPolicy.

    Returns:
        True when the notifier should fire, else False.
    """
    # ALL: fire for every outcome, no filtering.
    if policy == NotifyPolicy.ALL:
        return True
    # NEVER: suppress the notifier entirely (the caller still logs).
    if policy == NotifyPolicy.NEVER:
        return False
    # ACTED_AND_FAILED (the default): fire only on a real rebalance or a failure, so a
    # routine no-op day is silent.
    return outcome in {"acted", "failed"}


# ---------------------------------------------------------------------------
# format_notification — a compact, plain-ASCII, single-line human summary.
# ---------------------------------------------------------------------------

def format_notification(n: Notification) -> str:
    """Render one Notification as a single plain-ASCII line safe for a log or an SMS.

    Examples:
        acted:  "ACTED 2024-08-28: 1 order(s) [A BUY 50]; target {A: 1.0}"
        no-op:  "NO-OP: no new month-end"
        failed: "FAILED: simulated broker rejection"

    Pure string function: no I/O, no side effects.
    """
    # ACTED: date, order count + per-order list, and the target weight dict.
    if n.outcome == "acted":
        # The completed month-end as YYYY-MM-DD (date only keeps the line short); "?" if
        # somehow absent, so the renderer never raises on a malformed payload.
        date_str = n.as_of_month_end.strftime("%Y-%m-%d") if n.as_of_month_end is not None else "?"
        # One "SYM SIDE QTY" token per order; side upper-cased for a consistent BUY/SELL
        # regardless of the caller's input casing.
        order_tokens = [f"{sym} {side.upper()} {qty}" for (sym, side, qty) in n.orders]
        # The bracketed, comma-joined order list (empty brackets when nothing was placed).
        orders_part = f"{len(n.orders)} order(s) [{', '.join(order_tokens)}]"
        # The target dict rendered as {SYM: weight, ...} in insertion order.
        target_part = "{" + ", ".join(f"{sym}: {w}" for sym, w in n.target_weights.items()) + "}"
        # Assemble the single acted line.
        return f"ACTED {date_str}: {orders_part}; target {target_part}"

    # NO-OP: just the reason (why the runner did not act today).
    if n.outcome == "no-op":
        return f"NO-OP: {n.reason}"

    # FAILED: the short failure summary (reason carries str(exception) upstream).
    if n.outcome == "failed":
        return f"FAILED: {n.reason}"

    # FALLBACK for any unexpected outcome string: uppercase it and append the reason, so an
    # unknown outcome still renders a legible line rather than dropping information.
    return f"{n.outcome.upper()}: {n.reason}"
