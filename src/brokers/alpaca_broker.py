# src/brokers/alpaca_broker.py

"""
Alpaca implementation of the Broker interface.

This is the ONLY file in src/ that is allowed to import from the alpaca SDK.
All other modules (strategies, risk, execution) import from src.brokers.base only.
"""

# Forward-reference annotations so type hints work without quoting.
from __future__ import annotations

# os.getenv reads environment variables (API keys loaded from .env).
import os

# datetime is needed when converting Alpaca's timezone-aware timestamps to
# standard Python datetime objects for OrderResult.submitted_at.
from datetime import datetime

# ---------------------------------------------------------------------------
# Alpaca SDK imports — everything from alpaca stays inside this file.
# ---------------------------------------------------------------------------
# StockHistoricalDataClient drives the market data methods (get_latest_price, etc.).
from alpaca.data.historical import StockHistoricalDataClient

# StockLatestTradeRequest wraps the symbol argument for get_stock_latest_trade().
from alpaca.data.requests import StockLatestTradeRequest

# TradingClient handles all order management and account operations.
from alpaca.trading.client import TradingClient

# Alpaca's own enums.  We import them under aliased names (AlpacaOrderSide, etc.)
# to avoid shadowing the identically-named enums from our own base module.
from alpaca.trading.enums import OrderSide as AlpacaOrderSide  # "buy" / "sell"
from alpaca.trading.enums import QueryOrderStatus  # ALL / OPEN / CLOSED — used in GetOrdersRequest
from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce  # "day" / "gtc"

# GetOrdersRequest   — filter object for list_recent_orders().
# LimitOrderRequest  — execute only at the specified price or better.
# MarketOrderRequest — execute immediately at best available price.
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

# load_dotenv() reads a .env file from the working directory into os.environ,
# so we don't have to export keys manually in every terminal session.
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Our own interface — the only import from base.py allowed in this file.
# ---------------------------------------------------------------------------
# Broker is the abstract class we are implementing.
# All the dataclasses and enums below are from the interface layer.
from src.brokers.base import (
    AccountSnapshot,
    Broker,
    OrderRequest,
    OrderResult,
    OrderSide,  # our enum — values "buy" / "sell"
    OrderType,  # our enum — values "market" / "limit"
    PositionSnapshot,  # our dataclass — one open position (symbol, qty, side, values)
)


class AlpacaBroker(Broker):
    """
    Concrete Broker implementation backed by the Alpaca brokerage.

    Always instantiate with paper=True during development. The constructor
    hard-asserts this so a live-account misconfiguration fails loudly at
    startup instead of silently routing real orders.

    Typical usage via from_env():

        broker = AlpacaBroker.from_env()
        broker.verify_paper_account()   # inherited from Broker
        account = broker.get_account()
    """

    def __init__(self, api_key: str, api_secret: str, paper: bool = True) -> None:
        """
        Build the two Alpaca SDK clients and store them as private attributes.

        Parameters
        ----------
        api_key    : Alpaca API key ID (e.g. APCA_API_KEY_ID from .env).
        api_secret : Alpaca API secret key.
        paper      : Must be True. Live trading is disabled in this codebase.
                     Always pass paper=True — this default exists only as documentation.
        """
        # Hard stop: if someone accidentally passes paper=False (wrong .env keys,
        # copy-paste error, etc.) we fail here rather than routing real money.
        assert paper is True, (
            "Live trading is disabled in this codebase. "
            "Set paper=True and use paper trading API keys."
        )

        # TradingClient connects to either paper-trading.alpaca.markets or
        # api.alpaca.markets depending on the paper flag.  With paper=True it
        # always hits the paper sandbox — zero real money involved.
        self._trading = TradingClient(api_key, api_secret, paper=paper)

        # StockHistoricalDataClient is for market data (quotes, bars, latest trade).
        # It's not used by any abstract method yet, but strategies and risk checks
        # will need it, so we wire it up here so they can reach it via broker._data.
        self._data = StockHistoricalDataClient(api_key, api_secret)

    # ------------------------------------------------------------------
    # Class method constructor — the standard way scripts create a broker
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> AlpacaBroker:
        """
        Read credentials from environment and return a ready AlpacaBroker.

        Calls load_dotenv() first so a .env file in the working directory is
        respected automatically.  Raises EnvironmentError if either key is
        absent so the caller gets a clear message rather than an auth failure
        deep inside the SDK.

        Returns
        -------
        AlpacaBroker
            Connected to Alpaca's paper trading sandbox.
        """
        # Populate os.environ from .env if present.  Safe to call even if the
        # keys are already exported — load_dotenv never overwrites existing vars.
        load_dotenv()

        api_key = os.getenv("APCA_API_KEY_ID")        # Paper API key ID
        api_secret = os.getenv("APCA_API_SECRET_KEY")  # Paper API secret

        # Fail fast with a human-readable message pointing to the fix.
        if not api_key:
            raise EnvironmentError(
                "APCA_API_KEY_ID is not set. "
                "Copy .env.example to .env and add your Alpaca paper trading key."
            )
        if not api_secret:
            raise EnvironmentError(
                "APCA_API_SECRET_KEY is not set. "
                "Copy .env.example to .env and add your Alpaca paper trading secret."
            )

        # Always paper=True — from_env() is intentionally paper-only.
        return cls(api_key=api_key, api_secret=api_secret, paper=True)

    # ------------------------------------------------------------------
    # Private helper — converts one Alpaca order object into our OrderResult
    # ------------------------------------------------------------------

    def _to_order_result(self, order: object) -> OrderResult:
        """
        Map an alpaca-py order object to our broker-agnostic OrderResult dataclass.

        Called by submit_order, get_order, and list_recent_orders so the
        conversion logic lives in exactly one place.

        Parameters
        ----------
        order : alpaca-py Order model returned by any trading client method.

        Returns
        -------
        OrderResult
            Our internal representation — no alpaca types leak out.
        """
        # order.id is a UUID object in alpaca-py; str() converts it to the
        # familiar hyphenated form "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".
        order_id: str = str(order.id)

        # order.side and order.order_type are str-Enum members in alpaca-py.
        # .value gives the underlying string ("buy", "sell", "market", "limit")
        # which exactly matches the values in our own OrderSide / OrderType enums.
        side: OrderSide = OrderSide(order.side.value)
        order_type: OrderType = OrderType(order.order_type.value)

        # order.status is also a str-Enum (OrderStatus).  We store it as a
        # plain string in OrderResult so callers don't need to import alpaca's
        # OrderStatus enum.  .value gives e.g. "new", "filled", "cancelled".
        status: str = order.status.value

        # submitted_at is a timezone-aware datetime from alpaca-py.
        # We keep it as-is — datetime objects preserve the timezone info.
        submitted_at: datetime = order.submitted_at

        # filled_qty comes back as a string (Alpaca uses Decimal-like strings).
        # Convert to int only when present — market orders may not be filled yet.
        filled_qty = int(order.filled_qty) if order.filled_qty is not None else None

        # filled_avg_price is also a string from the API; None until partially filled.
        filled_avg_price = (
            float(order.filled_avg_price) if order.filled_avg_price is not None else None
        )

        # limit_price is None for market orders; a numeric string for limit orders.
        limit_price = float(order.limit_price) if order.limit_price is not None else None

        # order.qty is the requested quantity as a string — always present.
        qty: int = int(order.qty)

        return OrderResult(
            order_id=order_id,
            symbol=order.symbol,          # already a plain str
            qty=qty,
            side=side,
            order_type=order_type,
            status=status,
            submitted_at=submitted_at,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            limit_price=limit_price,
        )

    # ------------------------------------------------------------------
    # Private helper — converts one Alpaca position object into our PositionSnapshot
    # ------------------------------------------------------------------

    def _to_position_snapshot(self, position: object) -> PositionSnapshot:
        """
        Map an alpaca-py Position object to our broker-agnostic PositionSnapshot.

        Called by get_positions so the conversion logic lives in exactly one place,
        mirroring how _to_order_result serves the order-returning methods.

        Parameters
        ----------
        position : alpaca-py Position model returned by get_all_positions().

        Returns
        -------
        PositionSnapshot
            Our internal representation — no alpaca types leak out.
        """
        # position.symbol is already a plain str (e.g. "SPY") — pass through verbatim.
        symbol: str = position.symbol

        # position.qty is a STRING from Alpaca (e.g. "10" or "10.0"). This codebase
        # never places fractional orders, so float() first so "10.0" parses cleanly,
        # then int() to land on the whole-shares assumption our OrderResult also uses.
        qty: int = int(float(position.qty))

        # position.side is a PositionSide str-Enum ("long" / "short"); .value gives
        # the plain string we store, mirroring how _to_order_result reads .value.
        side: str = position.side.value

        # position.market_value is Optional[str] on Alpaca (may be None). Convert the
        # string to float when present; default to 0.0 rather than crash if absent.
        market_value: float = (
            float(position.market_value) if position.market_value is not None else 0.0
        )

        # position.avg_entry_price is a STRING that is always present — float() it directly.
        avg_entry_price: float = float(position.avg_entry_price)

        # position.unrealized_pl is Optional[str] on Alpaca (may be None). Convert when
        # present; default to 0.0 if absent so downstream risk code never sees None.
        unrealized_pl: float = (
            float(position.unrealized_pl) if position.unrealized_pl is not None else 0.0
        )

        return PositionSnapshot(
            symbol=symbol,
            qty=qty,
            side=side,
            market_value=market_value,
            avg_entry_price=avg_entry_price,
            unrealized_pl=unrealized_pl,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        """
        Fetch the current account state from Alpaca.

        Alpaca paper accounts always have an account number beginning with "PA".
        We use that convention to populate is_paper rather than relying on the
        paper= flag we passed to TradingClient (which we can't read back out).
        """
        # API call: GET /v2/account
        account = self._trading.get_account()

        # account_number is a plain string like "PA3XYZABC" for paper accounts.
        account_number: str = account.account_number

        # Alpaca returns monetary values as strings (e.g. "94321.50") to avoid
        # floating-point ambiguity at the API layer.  We convert to float here.
        buying_power: float = float(account.buying_power)
        cash: float = float(account.cash)
        portfolio_value: float = float(account.portfolio_value)

        # Paper account detection: Alpaca paper accounts start with "PA".
        # Live accounts have purely numeric identifiers.
        is_paper: bool = account_number.startswith("PA")

        return AccountSnapshot(
            account_number=account_number,
            buying_power=buying_power,
            cash=cash,
            portfolio_value=portfolio_value,
            is_paper=is_paper,
        )

    def submit_order(self, request: OrderRequest) -> OrderResult:
        """
        Convert our OrderRequest to an alpaca-py request object and submit it.

        The broker returns the order in its initial state ("new" or
        "pending_new") — NOT in the "filled" state even for market orders,
        because fills happen asynchronously on the exchange.
        """
        # Map our OrderSide enum to alpaca's.  Both enums share the same
        # underlying string values ("buy", "sell") so .value round-trips cleanly.
        alpaca_side = AlpacaOrderSide(request.side.value)

        # Same pattern for TimeInForce ("day", "gtc").
        alpaca_tif = AlpacaTimeInForce(request.time_in_force.value)

        if request.order_type == OrderType.MARKET:
            # MarketOrderRequest has no price field — Alpaca fills at best price.
            alpaca_request = MarketOrderRequest(
                symbol=request.symbol,
                qty=request.qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
            )
        else:
            # LimitOrderRequest requires limit_price.  If a caller constructs an
            # OrderRequest with order_type=LIMIT but no limit_price, this will
            # raise an error from alpaca-py — which is the correct behaviour.
            alpaca_request = LimitOrderRequest(
                symbol=request.symbol,
                qty=request.qty,
                side=alpaca_side,
                time_in_force=alpaca_tif,
                limit_price=request.limit_price,  # float or None (SDK will reject None)
            )

        # API call: POST /v2/orders — the line that actually sends the order.
        order = self._trading.submit_order(alpaca_request)

        # Convert the alpaca response to our broker-agnostic OrderResult.
        return self._to_order_result(order)

    def get_order(self, order_id: str) -> OrderResult:
        """
        Retrieve the current state of a previously submitted order.

        Use this to poll fill status: call it in a loop after submit_order
        until status == "filled" or another terminal state.
        """
        # API call: GET /v2/orders/{order_id}
        # get_order_by_id accepts the UUID string we stored in OrderResult.order_id.
        order = self._trading.get_order_by_id(order_id)

        return self._to_order_result(order)

    def cancel_order(self, order_id: str) -> None:
        """
        Request cancellation of an open order.

        Cancellation races with the exchange fill.  If the order fills before
        the cancel arrives, alpaca-py raises an APIError — callers should
        catch it and check the order status to confirm what actually happened.
        """
        # API call: DELETE /v2/orders/{order_id}
        # Returns 204 No Content on success; raises on error.
        self._trading.cancel_order_by_id(order_id)
        # Returning None is explicit here — the abstract signature says -> None
        # and some linters warn if a method with a non-None possible return is
        # used where None is expected.
        return None

    def list_recent_orders(self, limit: int = 50) -> list[OrderResult]:
        """
        Return up to `limit` most-recent orders, newest first.

        Fetches ALL statuses (filled, cancelled, rejected, open) so callers
        get a complete activity log rather than just open positions.
        """
        # GetOrdersRequest holds the query filters; unspecified fields are omitted
        # from the API request, letting Alpaca use its own defaults.
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,  # include every terminal and non-terminal status
            direction="desc",             # newest order first — most useful for recency views
            limit=limit,                  # cap at caller-specified count (default 50)
        )

        # API call: GET /v2/orders?status=all&direction=desc&limit=N
        # Returns a list of Order objects.
        orders = self._trading.get_orders(request)

        # Convert every alpaca Order to our OrderResult using the private helper.
        # List comprehension keeps this one logical line.
        return [self._to_order_result(o) for o in orders]

    def get_positions(self) -> list[PositionSnapshot]:
        """
        Return every currently open position, newest state as of the last query.

        Wraps GET /v2/positions via self._trading.get_all_positions(). A flat or
        brand-new paper account has no positions, so this returns [] cleanly —
        never None and never a crash — which is the common real state.
        """
        # API call: GET /v2/positions — returns a List[Position], or an empty
        # list when the account holds nothing (the flat/new-account case).
        positions = self._trading.get_all_positions()

        # Convert every alpaca Position to our PositionSnapshot using the private
        # helper. Same list-comprehension shape as list_recent_orders; [] stays [].
        return [self._to_position_snapshot(p) for p in positions]

    def is_market_open(self) -> bool:
        """
        Return True if the US equities market is currently open for trading.

        The clock endpoint also returns next_open and next_close, which
        strategies can use for session-aware scheduling — but this method
        only surfaces the boolean flag the interface requires.
        """
        # API call: GET /v2/clock — lightweight endpoint, safe to call frequently.
        clock = self._trading.get_clock()

        # clock.is_open is already a bool from alpaca-py's model.
        return bool(clock.is_open)

    def get_latest_price(self, symbol: str) -> float:
        """
        Fetch the most recent traded price for the given symbol.

        Uses the data client (self._data) rather than the trading client,
        because latest trade data lives on a separate Alpaca endpoint.
        """
        # StockLatestTradeRequest wraps the symbol in the structure the SDK expects.
        # symbol_or_symbols accepts a single string or a list — we always pass one.
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)

        # API call: GET /v2/stocks/{symbol}/trades/latest via the data client.
        # Returns a dict keyed by symbol; we extract the entry for our symbol.
        latest_trade = self._data.get_stock_latest_trade(request)

        # .price is a Decimal-like string in alpaca-py; float() gives a usable number.
        return float(latest_trade[symbol].price)
