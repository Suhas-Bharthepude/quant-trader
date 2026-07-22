# tests/test_email_notify.py

"""Hermetic tests for src/execution/email_notify.py (fake SMTP transport, NO real network)."""

# datetime + timezone build the acted Notification's as_of_month_end.
from datetime import datetime, timezone

# logging: caplog assertions use INFO level to capture the WARNING (degrade) and ERROR
# (send failure) records.
import logging

# The units under test: the pure builder, the config reader, and the notifier factory.
from src.execution.email_notify import build_email, read_smtp_config, make_email_notifier

# Notification is the payload; format_notification is the shared body renderer we assert on.
from src.execution.notify import Notification, format_notification


# ---------------------------------------------------------------------------
# Fake SMTP transport — the smtplib.SMTP surface the notifier uses, no network.
# ---------------------------------------------------------------------------


class _FakeSMTP:
    """A stand-in for smtplib.SMTP usable as a context manager, recording every call."""

    def __init__(self, recorder: dict):
        # Share a recorder dict so the test can inspect this instance after the call.
        self.recorder = recorder
        # Ordered log of method names invoked (for order assertions).
        self.calls: list[str] = []
        # Messages passed to send_message (the "sent" mailbox).
        self.sent: list = []
        # The (user, password) tuple login was called with.
        self.login_args = None
        # Whether send_message should raise (simulates auth/network failure).
        self._raise_on_send = recorder.get("raise_on_send", False)

    def __enter__(self):
        # Entering the context manager; return self so `with factory() as smtp` binds it.
        self.calls.append("enter")
        return self

    def __exit__(self, *exc):
        # Exiting; return False so any exception raised inside the with-block propagates.
        self.calls.append("exit")
        return False

    def starttls(self):
        # Record the TLS upgrade.
        self.calls.append("starttls")

    def login(self, user, password):
        # Record the credentials the notifier authenticated with.
        self.calls.append("login")
        self.login_args = (user, password)

    def send_message(self, msg):
        # Record the send; optionally simulate a transport failure.
        self.calls.append("send_message")
        if self._raise_on_send:
            raise RuntimeError("smtp boom")
        self.sent.append(msg)

    def quit(self):
        # Record an explicit quit (not used by the context-manager path, but part of surface).
        self.calls.append("quit")


def _make_fake_factory(recorder: dict):
    """Return a zero-arg factory that constructs a _FakeSMTP and records the instance."""
    def factory():
        # Count constructions so the degrade test can assert the factory was NEVER called.
        recorder["constructed"] = recorder.get("constructed", 0) + 1
        # Build the fake and expose it on the recorder for post-call inspection.
        inst = _FakeSMTP(recorder)
        recorder["instance"] = inst
        return inst
    return factory


# A representative acted Notification reused across the send tests.
_ACTED = Notification(
    outcome="acted",
    reason="acted",
    as_of_month_end=datetime(2024, 8, 28, tzinfo=timezone.utc),
    orders=(("A", "buy", 50),),
    target_weights={"A": 1.0},
    error=None,
)


# ---------------------------------------------------------------------------
# 1. build_email — headers + subject + body.
# ---------------------------------------------------------------------------


def test_build_email_sets_headers_subject_and_body():
    """build_email sets From/To, an outcome-encoding subject with the month-end, and the body."""
    # Build the message for the acted notification.
    msg = build_email(_ACTED, sender="me@gmail.com", recipient="you@example.com")
    # From/To are the given sender/recipient.
    assert msg["From"] == "me@gmail.com"
    assert msg["To"] == "you@example.com"
    # Subject encodes the acted outcome and the month-end date.
    assert msg["Subject"].startswith("[quant-trader] ACTED")
    assert "2024-08-28" in msg["Subject"]
    # Body reuses format_notification's text (substring, robust to a trailing newline).
    assert format_notification(_ACTED) in msg.get_content()


# ---------------------------------------------------------------------------
# 2. read_smtp_config — complete vs any-missing.
# ---------------------------------------------------------------------------


def test_read_smtp_config_returns_tuple_when_complete():
    """All three keys present -> the (user, app_password, recipient) tuple."""
    # A complete mapping (passed in, so os.environ is never touched).
    env = {
        "SMTP_USER": "u@gmail.com",
        "SMTP_APP_PASSWORD": "abcd efgh ijkl mnop".replace(" ", ""),
        "ALERT_EMAIL_TO": "to@example.com",
    }
    # The reader returns the tuple in (user, password, recipient) order.
    assert read_smtp_config(env) == ("u@gmail.com", "abcdefghijklmnop", "to@example.com")


def test_read_smtp_config_returns_none_when_any_missing():
    """Any missing/empty key -> None (the degrade signal)."""
    # A complete base mapping to remove one key from at a time.
    base = {
        "SMTP_USER": "u@gmail.com",
        "SMTP_APP_PASSWORD": "pw",
        "ALERT_EMAIL_TO": "to@example.com",
    }
    # Removing any single key yields None.
    for missing in ("SMTP_USER", "SMTP_APP_PASSWORD", "ALERT_EMAIL_TO"):
        # Copy the base and drop the one key under test.
        env = {k: v for k, v in base.items() if k != missing}
        # Missing that key -> None.
        assert read_smtp_config(env) is None
    # An empty-string value also counts as missing.
    assert read_smtp_config({**base, "SMTP_USER": ""}) is None


# ---------------------------------------------------------------------------
# 3. make_email_notifier — configured send via a FAKE transport (no network).
# ---------------------------------------------------------------------------


def test_notifier_sends_via_fake_transport_when_configured(monkeypatch):
    """A configured env + fake factory -> starttls, login, send_message once with the message."""
    # Configure the environment the module-level read_smtp_config() reads.
    monkeypatch.setenv("SMTP_USER", "u@gmail.com")
    monkeypatch.setenv("SMTP_APP_PASSWORD", "app-pw-16chars00")
    monkeypatch.setenv("ALERT_EMAIL_TO", "to@example.com")
    # Recorder + fake factory (records construction and the instance).
    recorder: dict = {}
    notifier = make_email_notifier(transport_factory=_make_fake_factory(recorder))
    # Fire the notifier on the acted notification.
    notifier(_ACTED)
    # The transport was constructed exactly once.
    assert recorder["constructed"] == 1
    # The fake recorded starttls -> login -> send_message in that order.
    inst = recorder["instance"]
    assert inst.calls.index("starttls") < inst.calls.index("login") < inst.calls.index("send_message")
    # login used the configured user + app password.
    assert inst.login_args == ("u@gmail.com", "app-pw-16chars00")
    # Exactly one message was sent, and its subject is the acted subject.
    assert len(inst.sent) == 1
    assert inst.sent[0]["Subject"] == "[quant-trader] ACTED 2024-08-28"


# ---------------------------------------------------------------------------
# 4. DEGRADE — unconfigured env warns once and never constructs the transport.
# ---------------------------------------------------------------------------


def test_notifier_degrades_when_unconfigured(monkeypatch, caplog):
    """Empty SMTP env -> a WARNING, NO transport construction, and NO raise."""
    # Ensure none of the three keys are set (delete if present in the real env).
    for key in ("SMTP_USER", "SMTP_APP_PASSWORD", "ALERT_EMAIL_TO"):
        monkeypatch.delenv(key, raising=False)
    # Recorder + fake factory that must NEVER be constructed on the degrade path.
    recorder: dict = {}
    notifier = make_email_notifier(transport_factory=_make_fake_factory(recorder))
    # Calling it must not raise; capture INFO+ to see the warning.
    with caplog.at_level(logging.INFO):
        notifier(_ACTED)
    # A degrade warning was logged.
    assert any(r.levelname == "WARNING" and "not configured" in r.getMessage() for r in caplog.records)
    # The transport factory was never called (no send attempted).
    assert recorder.get("constructed", 0) == 0


# ---------------------------------------------------------------------------
# 5. RAISING TRANSPORT — a send failure is caught, logged, and swallowed.
# ---------------------------------------------------------------------------


def test_notifier_swallows_send_failure(monkeypatch, caplog):
    """A transport whose send_message raises -> logged 'email notifier failed', no propagation."""
    # Configure the env so the send path is taken.
    monkeypatch.setenv("SMTP_USER", "u@gmail.com")
    monkeypatch.setenv("SMTP_APP_PASSWORD", "app-pw-16chars00")
    monkeypatch.setenv("ALERT_EMAIL_TO", "to@example.com")
    # Recorder flagged so the fake's send_message raises.
    recorder: dict = {"raise_on_send": True}
    notifier = make_email_notifier(transport_factory=_make_fake_factory(recorder))
    # The notifier must NOT raise despite the transport failure.
    with caplog.at_level(logging.INFO):
        notifier(_ACTED)
    # An ERROR record naming the failure was logged (the Day-58 invariant: never propagate).
    assert any(r.levelname == "ERROR" and "email notifier failed" in r.getMessage() for r in caplog.records)
