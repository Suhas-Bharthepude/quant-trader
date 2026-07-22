# src/execution/email_notify.py

"""
Real EMAIL notifier for the daily runner: stdlib SMTP (Gmail), behind an injectable transport.

Wires into the Day-58 notify_fn seam so an unattended run_daily reaches an inbox on the
acted + failed outcomes.  It is stdlib-only (smtplib + email.message) - NO new dependency.

DESIGN INVARIANTS (all load-bearing):
  - The SMTP call sits behind an INJECTED transport_factory, so the send logic is
    unit-tested with a fake transport and NEVER touches the network in CI.
  - Missing/partial SMTP env vars -> the notifier DEGRADES to logging-only (warns once,
    does not raise), so the bot still trades with no email configured.
  - Any send failure (auth, network) is CAUGHT, logged as "email notifier failed: ...", and
    SWALLOWED - it never re-raises.  This is the Day-58 invariant: a broken email channel can
    never break or mask a trade.
  - Secrets come from env ONLY (SMTP_USER, SMTP_APP_PASSWORD, ALERT_EMAIL_TO); they are never
    hard-coded and never logged (only the caught exception is interpolated into a log line).

Gmail SMTP: smtp.gmail.com port 587 with STARTTLS; SMTP_APP_PASSWORD must be a Gmail APP
PASSWORD (16 chars), never a real account password.
"""

# Modern type-hint syntax (tuple[...] | None, Callable[...]) without quoting.
from __future__ import annotations

# logging: module-level logger named `log`, matching cli_common's idiom; configures no
# handlers (the entry point owns that).  Used for the degrade warning and the swallowed-error.
import logging

# os supplies the default environment mapping for read_smtp_config (injected in tests).
import os

# smtplib is the stdlib SMTP client; the default transport factory returns one for Gmail.
import smtplib

# EmailMessage is the stdlib message builder (headers + body) build_email returns.
from email.message import EmailMessage

# Callable types the returned notify_fn; Notification is the payload; format_notification is
# the SHARED one-line renderer reused verbatim for the email body (no re-rendering here).
from typing import Callable
from src.execution.notify import Notification, format_notification


# Module-scoped logger (getLogger(__name__) per cli_common.py's idiom; no handler config here).
log = logging.getLogger(__name__)


def build_email(
    notification: Notification,
    sender: str,
    recipient: str,
) -> EmailMessage:
    """Build the EmailMessage for one Notification.  PURE: no I/O, no network.

    Sets From/To/Subject and a plain-text body that reuses the SHARED format_notification
    renderer (so the email body matches the log line exactly).

    Subject encodes the outcome, plus the as-of month-end date when acted:
        acted  -> "[quant-trader] ACTED 2024-08-28"
        failed -> "[quant-trader] FAILED"
        no-op  -> "[quant-trader] NO-OP"
    """
    # Start with the acted subject WITHOUT a date, then append the month-end date if present.
    if notification.outcome == "acted":
        # Guard against a missing as_of (should not happen on an acted run, but never raise):
        # append the ISO date only when we actually have one.
        if notification.as_of_month_end is not None:
            subject = f"[quant-trader] ACTED {notification.as_of_month_end.date().isoformat()}"
        else:
            subject = "[quant-trader] ACTED"
    elif notification.outcome == "failed":
        # A failure needs no date in the subject; the body carries the error detail.
        subject = "[quant-trader] FAILED"
    else:
        # Any other outcome (no-op, or an unexpected string) -> uppercase it into the subject.
        subject = f"[quant-trader] {notification.outcome.upper()}"

    # Construct the message and set the standard headers.
    msg = EmailMessage()
    msg["From"] = sender          # the authenticated Gmail address
    msg["To"] = recipient         # where the alert is delivered
    msg["Subject"] = subject      # the outcome-encoding subject built above

    # Body = the SHARED one-line human summary; set_content writes a plain-text body.
    msg.set_content(format_notification(notification))

    # The fully-built message (no I/O performed).
    return msg


def read_smtp_config(env=None) -> tuple[str, str, str] | None:
    """Read (user, app_password, recipient) from a mapping; None if any is missing/empty.

    Pure given the mapping.  env defaults to os.environ when None, so production reads the
    real environment while tests pass a plain dict they control.  Returns None when ANY of
    the three keys is absent or empty, which is the caller's signal to degrade to logging-only.
    """
    # Default to the process environment when no mapping is injected.
    if env is None:
        env = os.environ

    # Read the three keys (empty string default so a blank value is treated as missing).
    user = env.get("SMTP_USER", "")
    app_password = env.get("SMTP_APP_PASSWORD", "")
    recipient = env.get("ALERT_EMAIL_TO", "")

    # All three must be present and non-empty; otherwise signal "not configured" with None.
    if not user or not app_password or not recipient:
        return None

    # A complete config tuple (order: user, app_password, recipient).
    return (user, app_password, recipient)


def _default_transport_factory() -> smtplib.SMTP:
    """Return a real Gmail SMTP client (smtp.gmail.com:587) for use as a context manager.

    The caller uses the result as a context manager and calls starttls(), login(...),
    send_message(...); smtplib.SMTP.__exit__ closes the connection.  Tests inject a fake
    factory instead of this, so this real client is never constructed in the suite.
    """
    # Connect to Gmail's submission endpoint; STARTTLS is issued by the caller after connect.
    return smtplib.SMTP("smtp.gmail.com", 587)


def make_email_notifier(transport_factory=None) -> Callable[[Notification], None]:
    """Return a notify_fn(Notification) that emails on send, degrading/failing SAFELY.

    Args:
        transport_factory: a zero-arg callable returning an SMTP context manager
            (starttls/login/send_message).  Defaults to the real Gmail factory; tests inject
            a fake so no network is touched.

    Returns:
        A Callable[[Notification], None] suitable as run_daily's notify_fn.  It NEVER raises:
        an unconfigured env degrades to a one-time warning, and any send exception is caught,
        logged, and swallowed (the Day-58 invariant that a notifier cannot break the trade).
    """
    # Resolve the factory once: injected fake in tests, real Gmail client in production.
    factory = transport_factory or _default_transport_factory

    # Closure flag so the "not configured" warning is logged only ONCE per notifier instance
    # (an unattended daily run with no email configured must not spam a warning every day).
    warned = {"done": False}

    def notify(n: Notification) -> None:
        # STEP a - read config; None means at least one SMTP env var is missing.
        cfg = read_smtp_config()
        if cfg is None:
            # Degrade to logging-only: warn ONCE, then return without sending or raising.  The
            # password is not involved here, and nothing secret is logged.
            if not warned["done"]:
                log.warning(
                    "email notifier not configured (missing SMTP_USER/SMTP_APP_PASSWORD/"
                    "ALERT_EMAIL_TO); degrading to logging-only"
                )
                warned["done"] = True
            return

        # Unpack the complete config (user is also the From/authenticated address).
        user, app_password, recipient = cfg

        # STEP b - build and send, with the ENTIRE send wrapped so nothing propagates.
        try:
            # Build the message (pure) from the notification.
            msg = build_email(n, sender=user, recipient=recipient)
            # Open the transport as a context manager (real SMTP or an injected fake).
            with factory() as smtp:
                # Upgrade to TLS before authenticating (Gmail requires STARTTLS on 587).
                smtp.starttls()
                # Authenticate with the app password (NOT logged anywhere).
                smtp.login(user, app_password)
                # Send the built message.
                smtp.send_message(msg)
        except Exception as exc:  # noqa: BLE001 - a broken email channel must never break a trade
            # Catch EVERYTHING, log the exception only (never the password), and swallow it so
            # the notifier can never re-raise and mask the caller's trading result/exception.
            log.error("email notifier failed: %s", exc)

    # Hand back the closure as the injectable notify_fn.
    return notify
