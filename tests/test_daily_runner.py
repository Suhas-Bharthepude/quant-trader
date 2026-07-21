# tests/test_daily_runner.py

"""Hermetic tests for src/execution/daily_runner.py (no live broker, clock, or DuckDB)."""

# datetime + timezone build the tz-aware UTC "today" instants and month-end dates.
from datetime import datetime, timezone

# json pre-writes / reads the tiny state file directly in the run_daily tests.
import json

# Path types the tmp_path-based state file location.
from pathlib import Path

# pytest.raises drives the ValueError and propagated-RuntimeError assertions; tmp_path
# is the built-in per-test temporary directory fixture used for the state file.
import pytest

# The broker contract and the value types the fake constructs.  A local fake subclasses
# Broker so run_rebalance's genuine paper guard runs against it.
from src.brokers.base import Broker, OrderResult, AccountSnapshot

# OHLCVBar is the bar schema the synthetic fixture builds.
from src.data.schema import OHLCVBar

# The units under test: the pure decision core, the IO shell, and the result type.
from src.execution.daily_runner import decide_rebalance, run_daily, RebalanceDecision


# ---------------------------------------------------------------------------
# Local fake broker — copied/adapted per the repo's hermetic per-module convention
# (no cross-test-module imports).  Extends the existing test_rebalance fake with the
# two gaps the Day 57 inspection found: a canned is_market_open and a raise_on_nth
# submit_order failure switch.
# ---------------------------------------------------------------------------


class _RunnerFakeBroker(Broker):
    """In-memory Broker for testing run_daily/run_rebalance offline.

    Subclasses the real Broker ABC, so verify_paper_account() runs the GENUINE guard
    from src/brokers/base.py (it calls this fake's get_account() and raises on a live
    account).  Every submitted OrderRequest is recorded in self.submitted.
    """

    def __init__(
        self,
        *,
        is_open: bool,                      # canned is_market_open() return
        portfolio_value: float = 100_000.0,  # account budget the weights size against
        positions=None,                    # list[PositionSnapshot]; default no holdings
        prices=None,                       # symbol -> latest price the runner will fetch
        raise_on_nth=None,                 # when set to k, the k-th submit_order raises
    ):
        self.calls: list[str] = []              # ordered log of method names invoked
        self.submitted: list = []               # every OrderRequest passed to submit_order
        self._is_open = is_open                 # drives is_market_open()
        self._portfolio_value = portfolio_value  # drives get_account().portfolio_value
        self._positions = positions or []       # current holdings (empty by default)
        self._prices = prices or {}             # symbol -> price returned by get_latest_price
        self._raise_on_nth = raise_on_nth       # simulate a broker rejection mid-rebalance
        self._submit_count = 0                  # counts submit_order calls for raise_on_nth

    # --- Methods run_rebalance actually uses -------------------------------

    def verify_paper_account(self) -> None:
        # Record the guard call, then run the REAL guard (raises on a live account).
        self.calls.append("verify_paper_account")
        return super().verify_paper_account()

    def get_account(self) -> AccountSnapshot:
        # Return a paper account snapshot with the configured portfolio value.
        self.calls.append("get_account")
        return AccountSnapshot(
            account_number="FAKE-0001",       # arbitrary id
            buying_power=self._portfolio_value,
            cash=self._portfolio_value,
            portfolio_value=self._portfolio_value,
            is_paper=True,                    # paper -> the real guard passes
        )

    def get_positions(self):
        # Return the configured current holdings (empty list by default).
        self.calls.append("get_positions")
        return self._positions

    def submit_order(self, request) -> OrderResult:
        # Count this submission first so raise_on_nth can target the k-th call.
        self._submit_count += 1
        # Simulate a broker rejection on the k-th order: raise BEFORE recording it, so a
        # failed submission is never counted as submitted (mirrors a real mid-rebalance
        # rejection that leaves earlier orders live but this one un-placed).
        if self._raise_on_nth is not None and self._submit_count == self._raise_on_nth:
            raise RuntimeError("simulated broker rejection")
        # Otherwise record the request and return an accepted ("new") result.
        self.submitted.append(request)
        return OrderResult(
            order_id=f"fake-{self._submit_count}",  # unique id per submission
            symbol=request.symbol,                  # echo the request fields back
            qty=request.qty,
            side=request.side,
            order_type=request.order_type,
            status="new",                           # accepted, not yet filled (async)
            submitted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),  # fixed placeholder
        )

    def get_latest_price(self, symbol: str) -> float:
        # Log per-symbol so ordering/pricing can be asserted; return the canned price.
        self.calls.append(f"get_latest_price:{symbol}")
        return self._prices[symbol]

    def is_market_open(self) -> bool:
        # Canned bool set via the constructor (NOT NotImplementedError) so the shell's
        # default gate path is exercisable without a live clock.
        return self._is_open

    # --- Abstract methods the runner never calls (stubbed to satisfy the ABC) --

    def get_order(self, order_id: str) -> OrderResult:  # pragma: no cover - unused
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def list_recent_orders(self, limit: int = 50):  # pragma: no cover - unused
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Synthetic bar helper — copied verbatim from tests/test_portfolio.py so this test
# stays hermetic (no DuckDB): one OHLCVBar per (date, close), close == adj_close.
# ---------------------------------------------------------------------------


def make_month_end_bars(
    dates: list[datetime],
    closes: list[float],
    symbol: str = "T",
) -> list[OHLCVBar]:
    """Build one OHLCVBar per (date, close), in the given (ascending) order."""
    # One bar per (date, close); price fields all equal so basis is irrelevant.
    return [
        OHLCVBar(
            symbol=symbol,        # arbitrary; code reads only timestamp + price
            timestamp=ts,         # the bar's real trading date
            open=c,               # dummy
            high=c,               # dummy
            low=c,                # dummy
            close=c,              # the field the default price_field reads
            adj_close=c,          # equal to close -> basis-agnostic
            volume=1000,          # dummy
            timeframe="1d",       # daily
            source="test",        # provenance marker
        )
        for ts, c in zip(dates, closes)
    ]


# ---------------------------------------------------------------------------
# Shared deterministic fixture: 8 month-ends (day 28 of Jan..Aug 2024, so each bar
# is its own calendar month), SPY as reference plus A/B/C, crossing price paths so
# the top-1 pick (LIVE_CONFIG: lookback=3, top_n=1) has a definite positive winner.
# ---------------------------------------------------------------------------


# One bar per month on day 28 (valid in every month), Jan..Aug 2024, tz-aware UTC.
_DATES = [
    datetime(2024, 1, 28, tzinfo=timezone.utc),  # month-end index 0
    datetime(2024, 2, 28, tzinfo=timezone.utc),  # index 1
    datetime(2024, 3, 28, tzinfo=timezone.utc),  # index 2
    datetime(2024, 4, 28, tzinfo=timezone.utc),  # index 3
    datetime(2024, 5, 28, tzinfo=timezone.utc),  # index 4
    datetime(2024, 6, 28, tzinfo=timezone.utc),  # index 5
    datetime(2024, 7, 28, tzinfo=timezone.utc),  # index 6 (Jul month-end)
    datetime(2024, 8, 28, tzinfo=timezone.utc),  # index 7 (Aug month-end, last bar)
]

# Per-symbol closes; SPY is the reference spine.  At Aug 28 (index 7) vs May 28 (index 4,
# lookback=3) the trailing returns rank A strongest and positive, so top-1 holds "A".
_CLOSES = {
    "SPY": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0],  # steady riser
    "A": [100.0, 130.0, 90.0, 140.0, 95.0, 150.0, 100.0, 160.0],      # 160/95 - 1 > 0 (winner)
    "B": [100.0, 101.0, 150.0, 102.0, 160.0, 103.0, 170.0, 104.0],    # 104/160 - 1 < 0
    "C": [100.0, 90.0, 95.0, 92.0, 88.0, 94.0, 90.0, 96.0],           # 96/88 - 1 small
}

# Prices the fake returns for get_latest_price (only positively-weighted target symbols
# are actually fetched by run_rebalance, but all are supplied to be safe).
_PRICES = {"SPY": 114.0, "A": 160.0, "B": 104.0, "C": 96.0}

# A "today" in September, a month AFTER the last bar (Aug 28), so every month-end is
# completed and the resolver returns Aug 28.
_TODAY = datetime(2024, 9, 15, tzinfo=timezone.utc)

# The completed month-end the resolver lands on for _TODAY.
_AUG_ME = datetime(2024, 8, 28, tzinfo=timezone.utc)


def _bars() -> dict[str, list[OHLCVBar]]:
    """Build a fresh bars_by_symbol fixture per test."""
    # One OHLCVBar list per symbol over the shared monthly date grid.
    return {sym: make_month_end_bars(_DATES, closes, sym) for sym, closes in _CLOSES.items()}


# ===========================================================================
# decide_rebalance (PURE) unit tests.
# ===========================================================================


def test_decide_market_closed_is_noop_even_with_new_month_end():
    """1. MARKET CLOSED: is_market_open=False -> no act, reason 'market closed'."""
    # Reference spine bars; a new month-end (Aug 28) clearly exists as of _TODAY.
    ref = _bars()["SPY"]
    # Market gate is False, so the runner must no-op regardless of month-end state.
    decision = decide_rebalance(_TODAY, ref, last_acted_month_end=None, is_market_open=False)
    # No action, no month-end recorded, and the reason names the closed market.
    assert decision == RebalanceDecision(False, None, "market closed")


def test_decide_no_completed_month_end_yet():
    """2. NO COMPLETED MONTH-END YET: today before the first bar, market open -> no act."""
    # Reference spine bars.
    ref = _bars()["SPY"]
    # "Today" precedes the first bar (2024-01-28), so no month-end is completed yet.
    today = datetime(2024, 1, 15, tzinfo=timezone.utc)
    # Market open, but there is nothing to act on.
    decision = decide_rebalance(today, ref, last_acted_month_end=None, is_market_open=True)
    # No action, and the reason names the absent completed month-end.
    assert decision == RebalanceDecision(False, None, "no completed month-end yet")


def test_decide_new_month_end_never_acted_acts():
    """3. NEW MONTH-END, NEVER ACTED: last_acted=None, open -> act at the resolved month-end."""
    # Reference spine bars.
    ref = _bars()["SPY"]
    # Never acted before; market open; today resolves to Aug 28.
    decision = decide_rebalance(_TODAY, ref, last_acted_month_end=None, is_market_open=True)
    # Should act, recording Aug 28 as the justifying month-end.
    assert decision == RebalanceDecision(True, _AUG_ME, "acted")


def test_decide_already_acted_this_month_end_is_idempotent():
    """4. ALREADY ACTED (idempotency): last_acted == resolved as_of -> no new month-end."""
    # Reference spine bars.
    ref = _bars()["SPY"]
    # We already acted on Aug 28; today still resolves to Aug 28.
    decision = decide_rebalance(_TODAY, ref, last_acted_month_end=_AUG_ME, is_market_open=True)
    # No action: the same month-end must not trigger a second rebalance.
    assert decision == RebalanceDecision(False, None, "no new month-end")


def test_decide_host_down_across_month_end_catches_up():
    """5. HOST DOWN ACROSS A MONTH-END: last_acted is OLD -> act once on the LATER month-end."""
    # Reference spine bars.
    ref = _bars()["SPY"]
    # Last acted on an OLD month-end (Jun 28); the host missed July/August.
    old_me = datetime(2024, 6, 28, tzinfo=timezone.utc)
    # Today (September) resolves to the LATER completed month-end, Aug 28.
    decision = decide_rebalance(_TODAY, ref, last_acted_month_end=old_me, is_market_open=True)
    # Should act once, catching up to Aug 28 (not Jun 28, not July).
    assert decision == RebalanceDecision(True, _AUG_ME, "acted")


# ===========================================================================
# run_daily (IO SHELL) hermetic tests — tmp_path state file, injected is_open_fn.
# ===========================================================================


def test_run_daily_acts_and_writes_state(tmp_path):
    """6. ACTS AND WRITES STATE: fresh state, new month-end, open -> rebalances, state written."""
    # A paper fake with the canned prices and no current holdings (so a BUY is generated).
    broker = _RunnerFakeBroker(is_open=True, prices=_PRICES)
    # Fresh (absent) state file under the per-test tmp dir.
    state_path = tmp_path / "rebalance_state.json"
    # The state file must not exist before the run.
    assert not state_path.exists()
    # Run the shell: market open (injected), today on the new Aug 28 month-end.
    decision = run_daily(
        broker,
        _bars(),
        _TODAY,
        state_path=state_path,
        is_open_fn=lambda: True,
    )
    # It should have acted on Aug 28.
    assert decision.should_act is True
    assert decision.as_of_month_end == _AUG_ME
    # An order was submitted (empty positions + a positive-weight target -> a BUY).
    assert len(broker.submitted) >= 1
    # The state file now exists and records the acted-on month-end iso.
    assert state_path.exists()
    assert json.loads(state_path.read_text())["last_acted_month_end"] == _AUG_ME.isoformat()


def test_run_daily_noop_does_not_write_state(tmp_path):
    """7. NO-OP DOES NOT WRITE STATE: market closed -> no act, no order, no state file."""
    # A fake whose market is closed (both canned and via the injected fn below).
    broker = _RunnerFakeBroker(is_open=False, prices=_PRICES)
    # Fresh (absent) state file.
    state_path = tmp_path / "rebalance_state.json"
    # Run with the market gate injected as closed.
    decision = run_daily(
        broker,
        _bars(),
        _TODAY,
        state_path=state_path,
        is_open_fn=lambda: False,
    )
    # No action, reason names the closed market.
    assert decision.should_act is False
    assert decision.reason == "market closed"
    # No order was submitted on a no-op.
    assert broker.submitted == []
    # The state file was NOT written (a no-op never records a month-end).
    assert not state_path.exists()


def test_run_daily_idempotent_across_two_runs(tmp_path):
    """8. IDEMPOTENT: state already at today's month-end -> no act, no order, state unchanged."""
    # A paper fake with market open.
    broker = _RunnerFakeBroker(is_open=True, prices=_PRICES)
    # Pre-write the state file to today's resolved month-end (Aug 28) -- simulates a prior run.
    state_path = tmp_path / "rebalance_state.json"
    state_path.write_text(json.dumps({"last_acted_month_end": _AUG_ME.isoformat()}))
    # Capture the exact file content before the run for an unchanged-comparison.
    before = state_path.read_text()
    # Run again the same month: today still resolves to Aug 28, already acted on.
    decision = run_daily(
        broker,
        _bars(),
        _TODAY,
        state_path=state_path,
        is_open_fn=lambda: True,
    )
    # No action: idempotent second run within the same completed month-end.
    assert decision.should_act is False
    assert decision.reason == "no new month-end"
    # No order submitted.
    assert broker.submitted == []
    # The state file content is byte-identical to before (no rewrite on a no-op).
    assert state_path.read_text() == before


def test_run_daily_does_not_write_state_on_failed_rebalance(tmp_path):
    """9. STATE NOT WRITTEN ON FAILED REBALANCE (crux): submit raises -> propagate, no state."""
    # A paper fake whose FIRST submit_order raises, simulating a mid-rebalance rejection.
    broker = _RunnerFakeBroker(is_open=True, prices=_PRICES, raise_on_nth=1)
    # Fresh (absent) state file.
    state_path = tmp_path / "rebalance_state.json"
    # The failed submission must propagate out of run_daily as a RuntimeError.
    with pytest.raises(RuntimeError):
        run_daily(
            broker,
            _bars(),
            _TODAY,
            state_path=state_path,
            is_open_fn=lambda: True,
        )
    # CRUX: because the rebalance did not succeed, the month-end was NOT recorded, so the
    # state file must not exist -> the same month is retried on the next run.
    assert not state_path.exists()


def test_run_daily_missing_reference_raises(tmp_path):
    """10. MISSING REFERENCE: reference_symbol not in bars_by_symbol -> ValueError."""
    # A paper fake with market open.
    broker = _RunnerFakeBroker(is_open=True, prices=_PRICES)
    # Fresh state file (unused; the guard fires before any state write).
    state_path = tmp_path / "rebalance_state.json"
    # An absent reference spine must fail loud with ValueError.
    with pytest.raises(ValueError):
        run_daily(
            broker,
            _bars(),
            _TODAY,
            reference_symbol="NOPE",
            state_path=state_path,
            is_open_fn=lambda: True,
        )
