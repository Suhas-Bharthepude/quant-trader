# src/brokers/base.py

"""
Broker abstraction layer. Strategy and risk code depends only on this interface,
never on a specific broker SDK. Concrete implementations live in
src/brokers/alpaca_broker.py, src/brokers/paper_broker.py (later), etc.
"""

# Allow forward references in type hints without quoting them (e.g. list[OrderResult]
# instead of "list[OrderResult]"). Backported to Python 3.7+ via this import.
from __future__ import annotations

# abc = Abstract Base Classes. ABC is the base class; abstractmethod is the decorator
# that marks a method as "must be implemented by any concrete subclass".
from abc import ABC, abstractmethod

# dataclass decorator generates __init__, __repr__, __eq__ automatically from
# field definitions. field() lets us set per-field options (e.g. default=None).
from dataclasses import dataclass, field

# datetime is used to timestamp when an order was submitted.
from datetime import datetime

# Enum lets us define a fixed set of named string constants.
# str + Enum means each member's value is a plain string ("buy", "sell", etc.),
# so it serializes cleanly to JSON and compares equal to the raw string.
from enum import Enum

# Optional[X] means the value can be X or None. Used for fields that may be
# absent (e.g. limit_price on a market order, filled_qty before a fill).
from typing import Optional

# ---------------------------------------------------------------------------
# Enums — constrained vocabularies for order fields
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    """Whether we are buying or selling the instrument."""
    BUY = "buy"    # entering a long position (or covering a short)
    SELL = "sell"  # exiting a long position (or shorting)


class OrderType(str, Enum):
    """Execution style of the order."""
    MARKET = "market"  # fill immediately at the best available price
    LIMIT = "limit"    # fill only at limit_price or better; may not fill at all


class TimeInForce(str, Enum):
    """How long the order stays alive if not immediately filled."""
    DAY = "day"  # cancelled at market close if unfilled
    GTC = "gtc"  # Good Till Cancelled — stays open across sessions until filled or cancelled


# ---------------------------------------------------------------------------
# Data models — plain dataclasses so we have zero extra dependencies
# ---------------------------------------------------------------------------

@dataclass
class OrderRequest:
    """
    Everything needed to describe an order we want to send to the broker.
    Strategy code builds one of these; the broker implementation converts it
    to whatever the SDK expects.
    """

    symbol: str              # ticker, e.g. "SPY" or "AAPL"
    qty: int                 # number of shares (whole shares only for simplicity)
    side: OrderSide          # buy or sell
    order_type: OrderType    # market or limit
    time_in_force: TimeInForce  # day or gtc

    # limit_price is only required when order_type == OrderType.LIMIT.
    # We use field(default=None) so callers can omit it for market orders.
    limit_price: Optional[float] = field(default=None)


@dataclass
class OrderResult:
    """
    The broker's response after we submit (or query) an order.
    All fields come from the broker; we never construct this ourselves.
    """

    order_id: str            # broker-assigned unique identifier for this order
    symbol: str              # ticker the order is for
    qty: int                 # shares requested
    side: OrderSide          # buy or sell
    order_type: OrderType    # market or limit
    status: str              # broker status string, e.g. "new", "filled", "cancelled"
    submitted_at: datetime   # UTC timestamp when the broker accepted the order

    # The fields below are None until the order is (at least partially) filled.
    filled_qty: Optional[int] = field(default=None)              # shares actually executed so far
    filled_avg_price: Optional[float] = field(default=None)      # volume-weighted avg fill price
    limit_price: Optional[float] = field(default=None)           # echoed back for limit orders


@dataclass
class AccountSnapshot:
    """
    A point-in-time view of the brokerage account.
    Used by risk checks and the safety guard below.
    """

    account_number: str      # broker-assigned account ID
    buying_power: float      # USD available to open new positions (may include margin)
    cash: float              # settled cash balance
    portfolio_value: float   # total value: cash + open positions at last price
    is_paper: bool           # True = paper/simulated account; False = real money


# ---------------------------------------------------------------------------
# Abstract base class — the contract every broker must fulfil
# ---------------------------------------------------------------------------

class Broker(ABC):
    """
    Abstract interface for a brokerage connection.

    Design rule: no strategy, risk, or execution module may import anything
    from a concrete broker module (alpaca_broker, etc.). They must program
    only to this interface. This keeps strategies testable with a stub broker
    and makes broker replacement a one-file change.
    """

    # ------------------------------------------------------------------
    # Abstract methods — concrete subclasses MUST override every one of
    # these. If they don't, Python raises TypeError at instantiation time.
    # ------------------------------------------------------------------

    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        """
        Fetch current account balances and metadata.

        Returns
        -------
        AccountSnapshot
            Point-in-time view of buying power, cash, portfolio value, and
            whether this is a paper account.
        """
        ...  # ellipsis is the conventional "no body" placeholder for abstract methods

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> OrderResult:
        """
        Send an order to the broker and return its initial status.

        The returned OrderResult reflects the order as accepted by the broker.
        It will typically have status "new" or "pending_new" — not "filled" —
        because market orders fill asynchronously after this call returns.

        Parameters
        ----------
        request : OrderRequest
            Fully described order (symbol, qty, side, type, TIF, optional limit).

        Returns
        -------
        OrderResult
            Broker confirmation with order_id and initial status.
        """
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> OrderResult:
        """
        Look up the current state of a previously submitted order.

        Use this to poll fill status after submit_order returns.

        Parameters
        ----------
        order_id : str
            The broker-assigned ID from a prior OrderResult.

        Returns
        -------
        OrderResult
            Updated snapshot including filled_qty and filled_avg_price if filled.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """
        Request cancellation of an open (unfilled) order.

        Cancellation is best-effort: if the order fills before the cancel
        reaches the exchange, the broker will raise an error. Callers should
        handle that case.

        Parameters
        ----------
        order_id : str
            The broker-assigned ID of the order to cancel.
        """
        ...

    @abstractmethod
    def list_recent_orders(self, limit: int = 50) -> list[OrderResult]:
        """
        Retrieve the most recent orders, newest first.

        Parameters
        ----------
        limit : int
            Maximum number of orders to return. Defaults to 50.

        Returns
        -------
        list[OrderResult]
            Orders sorted by submitted_at descending.
        """
        ...

    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check whether the primary exchange (NYSE/NASDAQ) is currently open.

        Returns
        -------
        bool
            True if orders can be routed immediately; False during pre/post
            market or weekends/holidays.
        """
        ...

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """
        Fetch the most recent traded price for a symbol.

        Parameters
        ----------
        symbol : str
            Ticker to look up, e.g. "SPY" or "AAPL".

        Returns
        -------
        float
            Most recent transaction price from the last recorded trade.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete (non-abstract) methods — shared logic every subclass gets
    # for free by inheriting from Broker.
    # ------------------------------------------------------------------

    def verify_paper_account(self) -> None:
        """
        Safety guard: raise RuntimeError if this is a live-money account.

        Call this at the top of any script or strategy that should only
        ever touch a paper account. Because it calls self.get_account(),
        it works for every concrete broker implementation automatically.

        Raises
        ------
        RuntimeError
            If the connected account has is_paper == False.
        """
        # Fetch the current account state via the subclass's implementation.
        account = self.get_account()

        # Hard stop: we never want to accidentally send real orders.
        # Any script that calls verify_paper_account() is declaring that
        # it must not run against a live account under any circumstances.
        if not account.is_paper:
            raise RuntimeError(
                f"SAFETY VIOLATION: account {account.account_number} is a LIVE account. "
                "This script is only permitted to run against a paper account. "
                "Switch your API keys to the paper environment and retry."
            )
