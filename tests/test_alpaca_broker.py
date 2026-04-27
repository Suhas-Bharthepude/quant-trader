# tests/test_alpaca_broker.py

"""
Integration smoke test for AlpacaBroker.

Hits the real Alpaca paper-trading API — requires a valid .env file with:
    APCA_API_KEY_ID=...
    APCA_API_SECRET_KEY=...

Run with:
    uv run pytest tests/test_alpaca_broker.py -v

Skip in CI by filtering out the integration mark:
    pytest -m "not integration"
"""

# pytest is the test runner.  The mark decorator is used to tag this test
# so CI pipelines can skip it without modifying this file.
import pytest

# The class under test.  We import it through the package so the module
# boundary is exercised (src.brokers.base is imported inside alpaca_broker).
from src.brokers.alpaca_broker import AlpacaBroker

# Also import our base types so we can assert on the returned dataclasses
# without needing alpaca-py types in this test file.
from src.brokers.base import AccountSnapshot, OrderResult


@pytest.mark.integration
def test_alpaca_broker_smoke() -> None:
    """
    End-to-end read-only check that AlpacaBroker wires up to the paper API.

    This test places NO orders and cancels NOTHING.  It verifies that:
      - credentials load and the client connects
      - the safety guard (verify_paper_account) passes on a paper account
      - account fields have plausible values
      - is_market_open() returns a bool without raising
      - list_recent_orders() returns a list of the correct type
    """
    # ------------------------------------------------------------------
    # Construction — from_env() reads .env, asserts paper=True, and
    # creates the TradingClient + StockHistoricalDataClient internally.
    # If .env is missing or keys are wrong, this raises EnvironmentError
    # or an alpaca-py auth error before any assertion runs.
    # ------------------------------------------------------------------
    broker = AlpacaBroker.from_env()

    # ------------------------------------------------------------------
    # Safety guard — verify_paper_account() is a concrete method on the
    # Broker base class.  It calls get_account() and raises RuntimeError
    # if is_paper is False.  Calling it here confirms the guard works and
    # that we are definitely on the paper sandbox for this test run.
    # ------------------------------------------------------------------
    broker.verify_paper_account()  # must not raise

    # ------------------------------------------------------------------
    # Account snapshot — check that all fields are populated and sensible.
    # ------------------------------------------------------------------
    account = broker.get_account()

    # The return type must be our dataclass, not an alpaca SDK object.
    assert isinstance(account, AccountSnapshot), (
        f"get_account() should return AccountSnapshot, got {type(account)}"
    )

    # Paper accounts always start with "PA" per Alpaca's convention.
    assert account.account_number.startswith("PA"), (
        f"Expected paper account number (starts with 'PA'), got '{account.account_number}'"
    )

    # is_paper must be True — derived from the "PA" prefix in AlpacaBroker.get_account().
    assert account.is_paper is True, (
        "is_paper should be True for a paper account"
    )

    # buying_power should be a positive number on a freshly-funded paper account.
    # We use > 0 rather than a specific floor to avoid hardcoding a dollar amount.
    assert account.buying_power > 0, (
        f"Expected buying_power > 0, got {account.buying_power}"
    )

    # cash and portfolio_value must be non-negative floats.
    assert account.cash >= 0, f"Expected cash >= 0, got {account.cash}"
    assert account.portfolio_value >= 0, (
        f"Expected portfolio_value >= 0, got {account.portfolio_value}"
    )

    # ------------------------------------------------------------------
    # Market clock — is_market_open() wraps GET /v2/clock and returns bool.
    # We can't assert True or False (depends on time of day / day of week),
    # so we only assert the type.
    # ------------------------------------------------------------------
    market_open = broker.is_market_open()

    assert isinstance(market_open, bool), (
        f"is_market_open() should return bool, got {type(market_open)}"
    )

    # ------------------------------------------------------------------
    # Recent orders list — confirms the list endpoint works and each item
    # is our OrderResult, not a raw alpaca-py object.
    # ------------------------------------------------------------------
    orders = broker.list_recent_orders(limit=10)

    # Must always be a list (empty is fine on a brand-new paper account).
    assert isinstance(orders, list), (
        f"list_recent_orders() should return list, got {type(orders)}"
    )

    # Every item in the list must be an OrderResult dataclass.
    for order in orders:
        assert isinstance(order, OrderResult), (
            f"Each order should be OrderResult, got {type(order)}"
        )
        # order_id must be a non-empty string (UUID from Alpaca).
        assert isinstance(order.order_id, str) and order.order_id, (
            f"order_id should be a non-empty string, got {order.order_id!r}"
        )
