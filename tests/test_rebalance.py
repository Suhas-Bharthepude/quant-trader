# tests/test_rebalance.py

"""
Hermetic tests for the rebalance execution layer.

Two groups, both fully offline (no network, no integration mark):
- GROUP 1 exercises the PURE reconciliation function reconcile_to_target with
  plain dicts and no broker at all.
- GROUP 2 exercises the guarded runner run_rebalance against a minimal in-file
  _FakeBroker that subclasses the real Broker ABC, so the runner's safety guard
  is the REAL verify_paper_account from src/brokers/base.py, not a stand-in.

Run with:
    uv run pytest tests/test_rebalance.py -v
"""

# Fixed, timezone-aware timestamp for the fake's OrderResult, mirroring the
# convention in tests/test_alpaca_broker.py.
from datetime import datetime, timezone

# pytest is the test runner; pytest.raises checks the guard/validation errors.
import pytest

# The broker contract: the ABC our fake implements, the value enums, and the
# dataclasses the fake and the pure function pass around.
from src.brokers.base import (
    AccountSnapshot,
    Broker,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    PositionSnapshot,
    TimeInForce,
)

# The code under test.
from src.execution.rebalance import reconcile_to_target
from src.execution.runner import run_rebalance


# ===========================================================================
# GROUP 1 - the pure reconciliation function (no broker, plain dicts)
# ===========================================================================


def test_reconcile_opens_new_positions():
    """From flat, a two-symbol equal-weight target opens both positions."""
    # 50% SPY at $100 -> 50 shares; 50% TLT at $50 -> 100 shares. Portfolio $10k.
    orders = reconcile_to_target(
        target_weights={"SPY": 0.5, "TLT": 0.5},
        current_positions={},
        prices={"SPY": 100.0, "TLT": 50.0},
        portfolio_value=10000.0,
    )
    # Two orders, sorted by symbol so SPY precedes TLT.
    assert len(orders) == 2
    spy, tlt = orders
    # SPY: buy 50 shares, market + DAY.
    assert (spy.symbol, spy.side, spy.qty) == ("SPY", OrderSide.BUY, 50)
    assert spy.order_type == OrderType.MARKET and spy.time_in_force == TimeInForce.DAY
    # TLT: buy 100 shares, market + DAY.
    assert (tlt.symbol, tlt.side, tlt.qty) == ("TLT", OrderSide.BUY, 100)
    assert tlt.order_type == OrderType.MARKET and tlt.time_in_force == TimeInForce.DAY


def test_reconcile_closes_dropped_symbol():
    """A held symbol dropped from the target is sold to zero; a held target is topped up."""
    # SPY target 100 shares but only 50 held -> buy 50. TLT not in target -> sell all 100.
    orders = reconcile_to_target(
        target_weights={"SPY": 1.0},
        current_positions={"SPY": 50, "TLT": 100},
        prices={"SPY": 100.0, "TLT": 50.0},
        portfolio_value=10000.0,
    )
    assert len(orders) == 2
    spy, tlt = orders
    assert (spy.symbol, spy.side, spy.qty) == ("SPY", OrderSide.BUY, 50)
    assert (tlt.symbol, tlt.side, tlt.qty) == ("TLT", OrderSide.SELL, 100)


def test_reconcile_skips_already_at_target():
    """A symbol already holding exactly its target share count produces no order."""
    # SPY target = 10000/100 = 100 shares, and we already hold 100 -> delta 0.
    orders = reconcile_to_target(
        target_weights={"SPY": 1.0},
        current_positions={"SPY": 100},
        prices={"SPY": 100.0},
        portfolio_value=10000.0,
    )
    assert orders == []


def test_reconcile_floors_fractional_shares():
    """Share sizing floors via int(): 10000/300 = 33.33 becomes 33, never 34."""
    orders = reconcile_to_target(
        target_weights={"SPY": 1.0},
        current_positions={},
        prices={"SPY": 300.0},
        portfolio_value=10000.0,
    )
    assert len(orders) == 1
    assert (orders[0].symbol, orders[0].side, orders[0].qty) == ("SPY", OrderSide.BUY, 33)


def test_reconcile_empty_target_flattens():
    """An empty target closes every held position to zero."""
    orders = reconcile_to_target(
        target_weights={},
        current_positions={"SPY": 10},
        prices={"SPY": 100.0},
        portfolio_value=10000.0,
    )
    assert len(orders) == 1
    assert (orders[0].symbol, orders[0].side, orders[0].qty) == ("SPY", OrderSide.SELL, 10)


def test_reconcile_rejects_negative_weight():
    """A negative target weight is rejected: this version is long/flat only."""
    with pytest.raises(ValueError):
        reconcile_to_target(
            target_weights={"SPY": -0.5},
            current_positions={},
            prices={"SPY": 100.0},
            portfolio_value=10000.0,
        )


def test_reconcile_rejects_overallocation():
    """Target weights summing above 100% are rejected."""
    with pytest.raises(ValueError):
        reconcile_to_target(
            target_weights={"SPY": 0.6, "TLT": 0.6},  # sums to 1.2
            current_positions={},
            prices={"SPY": 100.0, "TLT": 50.0},
            portfolio_value=10000.0,
        )


# ===========================================================================
# GROUP 2 - the guarded runner (against a real-Broker-subclass fake)
# ===========================================================================


class _FakeBroker(Broker):
    """
    Minimal in-memory Broker for testing run_rebalance offline.

    It subclasses the real Broker ABC, so verify_paper_account() is the genuine
    guard from src/brokers/base.py: it calls this fake's get_account() and
    raises RuntimeError when is_paper is False. We override verify_paper_account
    only to record the call for ordering assertions, then delegate to the real
    implementation via super().

    Every broker method appends its own name to `self.calls` so tests can assert
    call ordering; submitted OrderRequests are recorded in `self.submitted`.
    """

    def __init__(
        self,
        *,
        is_paper: bool,
        portfolio_value: float,
        positions: list[PositionSnapshot],
        prices: dict[str, float] | None = None,
    ):
        self.calls: list[str] = []            # ordered log of method names invoked
        self.submitted: list[OrderRequest] = []  # every OrderRequest passed to submit_order
        self._is_paper = is_paper             # drives the real guard's pass/raise
        self._portfolio_value = portfolio_value
        self._positions = positions
        self._prices = prices or {}           # symbol -> price the runner will fetch

    # --- Methods the runner actually uses ---------------------------------

    def verify_paper_account(self) -> None:
        # Record that the runner called the guard, then run the REAL guard logic.
        self.calls.append("verify_paper_account")
        return super().verify_paper_account()

    def get_account(self) -> AccountSnapshot:
        self.calls.append("get_account")
        return AccountSnapshot(
            account_number="FAKE-0001",
            buying_power=self._portfolio_value,
            cash=self._portfolio_value,
            portfolio_value=self._portfolio_value,
            is_paper=self._is_paper,
        )

    def get_positions(self) -> list[PositionSnapshot]:
        self.calls.append("get_positions")
        return self._positions

    def submit_order(self, request: OrderRequest) -> OrderResult:
        self.calls.append("submit_order")
        self.submitted.append(request)
        return OrderResult(
            order_id=f"fake-{len(self.submitted)}",
            symbol=request.symbol,
            qty=request.qty,
            side=request.side,
            order_type=request.order_type,
            status="new",
            submitted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

    def get_latest_price(self, symbol: str) -> float:
        # Log per-symbol so tests can assert EXACTLY which symbols were priced.
        self.calls.append(f"get_latest_price:{symbol}")
        return self._prices[symbol]

    # --- Abstract methods the runner never calls (stubbed to satisfy ABC) --

    def get_order(self, order_id: str) -> OrderResult:  # pragma: no cover - unused by runner
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover - unused by runner
        raise NotImplementedError

    def list_recent_orders(self, limit: int = 50) -> list[OrderResult]:  # pragma: no cover
        raise NotImplementedError

    def is_market_open(self) -> bool:  # pragma: no cover - unused by runner
        raise NotImplementedError


def test_run_rebalance_paper_places_orders():
    """On a paper account, the runner submits exactly the orders the pure fn computes."""
    # Prices now live on the broker: the runner fetches them via get_latest_price.
    broker = _FakeBroker(is_paper=True, portfolio_value=10000.0, positions=[], prices={"SPY": 100.0})
    results = run_rebalance(broker, target_weights={"SPY": 1.0})
    # Exactly one order recorded: SPY BUY 100 (10000/100).
    assert len(broker.submitted) == 1
    req = broker.submitted[0]
    assert (req.symbol, req.side, req.qty) == ("SPY", OrderSide.BUY, 100)
    # And run_rebalance returned one OrderResult for it.
    assert len(results) == 1
    assert isinstance(results[0], OrderResult)
    assert results[0].symbol == "SPY"


def test_run_rebalance_live_raises_before_any_order():
    """SAFETY PROOF: on a live account the guard raises before any order OR price fetch."""
    # is_paper False -> the real verify_paper_account raises RuntimeError.
    broker = _FakeBroker(is_paper=False, portfolio_value=10000.0, positions=[], prices={"SPY": 100.0})
    with pytest.raises(RuntimeError):
        run_rebalance(broker, target_weights={"SPY": 1.0})
    # The guard blocked before any submission: zero orders reached the broker.
    assert broker.submitted == []
    # And before any market-data I/O: get_latest_price was never called.
    assert not any(c.startswith("get_latest_price") for c in broker.calls)


def test_run_rebalance_calls_guard_first():
    """The guard is the runner's first broker interaction (ordering proof)."""
    broker = _FakeBroker(is_paper=True, portfolio_value=10000.0, positions=[], prices={"SPY": 100.0})
    run_rebalance(broker, target_weights={"SPY": 1.0})
    # verify_paper_account is logged first, before get_account/get_positions/submit_order.
    assert broker.calls[0] == "verify_paper_account"


def test_run_rebalance_skips_price_for_sold_symbol():
    """A symbol being sold to zero is NOT priced: only positive-weight targets are fetched."""
    # Hold TLT (not in the target) and target SPY only. TLT must sell without a price fetch.
    held_tlt = PositionSnapshot(
        symbol="TLT", qty=100, side="long", market_value=5000.0, avg_entry_price=50.0, unrealized_pl=0.0
    )
    # Note: NO price supplied for TLT -- if the runner tried to price it, get_latest_price
    # would KeyError on self._prices["TLT"], which would itself fail the test.
    broker = _FakeBroker(
        is_paper=True, portfolio_value=10000.0, positions=[held_tlt], prices={"SPY": 100.0}
    )
    run_rebalance(broker, target_weights={"SPY": 1.0})
    # SPY (positive weight) was priced; TLT (sold to zero) was not.
    assert "get_latest_price:SPY" in broker.calls
    assert "get_latest_price:TLT" not in broker.calls
    # TLT was still sold to zero (100 shares), proving the sale needs no price.
    tlt_orders = [r for r in broker.submitted if r.symbol == "TLT"]
    assert len(tlt_orders) == 1
    assert (tlt_orders[0].side, tlt_orders[0].qty) == (OrderSide.SELL, 100)


def test_run_rebalance_prices_only_positive_weights():
    """An explicit weight-0 target symbol is not priced (filtered by weight > 0)."""
    # Only SPY has a price; TLT is weight 0.0. Pricing TLT would KeyError -> test failure.
    broker = _FakeBroker(is_paper=True, portfolio_value=10000.0, positions=[], prices={"SPY": 100.0})
    run_rebalance(broker, target_weights={"SPY": 1.0, "TLT": 0.0})
    assert "get_latest_price:SPY" in broker.calls
    assert "get_latest_price:TLT" not in broker.calls
