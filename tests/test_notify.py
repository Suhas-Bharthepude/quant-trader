# tests/test_notify.py

"""Unit tests for src/execution/notify.py (pure payload, policy, and formatter)."""

# datetime + timezone build the as_of_month_end value used in the acted payload.
from datetime import datetime, timezone

# The pure units under test: the payload type, the policy enum, the fire predicate,
# and the one-line formatter.
from src.execution.notify import (
    Notification,
    NotifyPolicy,
    should_notify,
    format_notification,
)


# ---------------------------------------------------------------------------
# should_notify — the pure fire/skip predicate under each policy.
# ---------------------------------------------------------------------------


def test_should_notify_acted_and_failed_policy():
    """ACTED_AND_FAILED fires on 'acted' and 'failed' but NOT 'no-op'."""
    # A real rebalance should fire.
    assert should_notify("acted", NotifyPolicy.ACTED_AND_FAILED) is True
    # A failure should fire.
    assert should_notify("failed", NotifyPolicy.ACTED_AND_FAILED) is True
    # A routine no-op must NOT fire under the default policy (no daily spam).
    assert should_notify("no-op", NotifyPolicy.ACTED_AND_FAILED) is False


def test_should_notify_all_policy():
    """ALL fires on every outcome."""
    # Every outcome fires under the verbose policy.
    assert should_notify("acted", NotifyPolicy.ALL) is True
    assert should_notify("no-op", NotifyPolicy.ALL) is True
    assert should_notify("failed", NotifyPolicy.ALL) is True


def test_should_notify_never_policy():
    """NEVER fires on none."""
    # No outcome fires when notifications are disabled.
    assert should_notify("acted", NotifyPolicy.NEVER) is False
    assert should_notify("no-op", NotifyPolicy.NEVER) is False
    assert should_notify("failed", NotifyPolicy.NEVER) is False


# ---------------------------------------------------------------------------
# format_notification — the compact one-line human renderer.
# ---------------------------------------------------------------------------


def test_format_acted_contains_month_end_order_and_target():
    """An 'acted' payload renders one line with the month-end, the order, and the target."""
    # A representative acted payload: one BUY of 50 shares of A, target fully in A.
    n = Notification(
        outcome="acted",
        reason="acted",
        as_of_month_end=datetime(2024, 8, 28, tzinfo=timezone.utc),
        orders=(("A", "buy", 50),),
        target_weights={"A": 1.0},
        error=None,
    )
    # Render the line.
    s = format_notification(n)
    # It must be a single line (safe for a log record or an SMS).
    assert "\n" not in s
    # The month-end date appears (YYYY-MM-DD form).
    assert "2024-08-28" in s
    # The order's symbol, side (upper-cased), and quantity all appear.
    assert "A" in s
    assert "BUY" in s
    assert "50" in s
    # The target weight appears.
    assert "1.0" in s


def test_format_noop_renders_reason():
    """A 'no-op' payload renders its reason."""
    # A no-op payload carrying the decision reason.
    n = Notification(
        outcome="no-op",
        reason="no new month-end",
        as_of_month_end=None,
        orders=(),
        target_weights={},
        error=None,
    )
    # Render the line.
    s = format_notification(n)
    # It names the no-op and echoes the reason (substring, robust to spacing).
    assert "NO-OP" in s
    assert "no new month-end" in s


def test_format_failed_renders_error():
    """A 'failed' payload renders its error summary."""
    # A failure payload whose reason/error carry the exception text.
    n = Notification(
        outcome="failed",
        reason="simulated broker rejection",
        as_of_month_end=None,
        orders=(),
        target_weights={},
        error="simulated broker rejection",
    )
    # Render the line.
    s = format_notification(n)
    # It names the failure and includes the error summary.
    assert "FAILED" in s
    assert "simulated broker rejection" in s
