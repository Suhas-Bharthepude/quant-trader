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
from src.brokers.base import AccountSnapshot, OrderResult, PositionSnapshot


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


# ---------------------------------------------------------------------------
# Hermetic (offline) unit tests — NO integration mark, NO network, NO creds.
# These run in CI. They monkeypatch the Alpaca SDK client classes AS IMPORTED
# INTO src.brokers.alpaca_broker's namespace, so AlpacaBroker.__init__ builds
# fakes instead of real clients and never touches the network.
# ---------------------------------------------------------------------------


class _FakeAccount:
    """Minimal stand-in for an alpaca-py Account model returned by get_account().

    Alpaca returns monetary values as STRINGS (e.g. "94321.50"); we mirror that
    here so the test genuinely exercises AlpacaBroker's str->float conversions.
    """

    # __init__ stores exactly the four attributes AlpacaBroker.get_account() reads.
    def __init__(
        self,
        account_number: str,   # e.g. "PA3XYZABC" (paper) or "123456789" (live)
        buying_power: str,     # money field as a string, mirroring the real API
        cash: str,             # money field as a string, mirroring the real API
        portfolio_value: str,  # money field as a string, mirroring the real API
    ) -> None:
        # Save the account identifier verbatim (used for the "PA" paper check).
        self.account_number = account_number
        # Save buying_power as a string; get_account() will float() it.
        self.buying_power = buying_power
        # Save cash as a string; get_account() will float() it.
        self.cash = cash
        # Save portfolio_value as a string; get_account() will float() it.
        self.portfolio_value = portfolio_value


class _FakeSide:
    """Minimal stand-in for alpaca-py's PositionSide str-Enum.

    AlpacaBroker._to_position_snapshot reads position.side.value, so the fake
    only needs a .value attribute holding the plain "long"/"short" string.
    """

    # __init__ stores the single .value attribute the conversion reads.
    def __init__(self, value: str) -> None:
        # Save the side string verbatim ("long" or "short").
        self.value = value


class _FakePosition:
    """Minimal stand-in for an alpaca-py Position model from get_all_positions().

    Alpaca returns qty and monetary values as STRINGS; we mirror that here so the
    test genuinely exercises AlpacaBroker's str->int / str->float conversions and
    the .side.value extraction.
    """

    # __init__ stores exactly the attributes _to_position_snapshot reads.
    def __init__(
        self,
        symbol: str,            # ticker, e.g. "SPY" — a plain string
        qty: str,               # shares as a string (e.g. "10"), mirroring the real API
        side: str,              # "long"/"short"; wrapped in _FakeSide to expose .value
        market_value: str,      # money field as a string, mirroring the real API
        avg_entry_price: str,   # money field as a string, mirroring the real API
        unrealized_pl: str,     # money field as a string, mirroring the real API
    ) -> None:
        # Save the symbol verbatim (passed through unchanged by the conversion).
        self.symbol = symbol
        # Save qty as a string; _to_position_snapshot will int(float()) it.
        self.qty = qty
        # Wrap the side string so position.side.value works like the real str-Enum.
        self.side = _FakeSide(side)
        # Save market_value as a string; the conversion will float() it.
        self.market_value = market_value
        # Save avg_entry_price as a string; the conversion will float() it.
        self.avg_entry_price = avg_entry_price
        # Save unrealized_pl as a string; the conversion will float() it.
        self.unrealized_pl = unrealized_pl


class _FakeTradingClient:
    """Fake TradingClient: accepts the real constructor args, makes no network call.

    A class attribute holds the _FakeAccount that get_account() should return, so
    each test can set the account state before constructing the broker.
    """

    # Class-level slot for the account object the next-built client will serve.
    # Set by each test (e.g. _FakeTradingClient.account = _FakeAccount(...)).
    account: "_FakeAccount"

    # Class-level slot for the positions list get_all_positions() should return.
    # Defaults to [] (a flat account) so tests that never set it see the common
    # empty state; each position test sets it explicitly to avoid cross-test leakage.
    positions: "list[_FakePosition]" = []

    # __init__ mirrors the real signature (api_key, api_secret, paper=...) and
    # ignores every argument — no client is created, no endpoint is contacted.
    def __init__(self, api_key: str, api_secret: str, paper: bool = True) -> None:
        # Deliberately do nothing: the fake holds no state beyond the class attr.
        pass

    # get_account() returns the pre-set _FakeAccount, standing in for GET /v2/account.
    def get_account(self) -> "_FakeAccount":
        # Hand back whatever account the test assigned to the class attribute.
        return type(self).account

    # get_all_positions() returns the pre-set list, standing in for GET /v2/positions.
    def get_all_positions(self) -> "list[_FakePosition]":
        # Hand back whatever positions the test assigned to the class attribute.
        return type(self).positions


class _FakeDataClient:
    """Fake StockHistoricalDataClient: accepts (api_key, api_secret), does nothing.

    AlpacaBroker.__init__ constructs this too, but the read-path tests here never
    call a data method, so a no-op stub is sufficient.
    """

    # __init__ mirrors the real (api_key, api_secret) signature and ignores both.
    def __init__(self, api_key: str, api_secret: str) -> None:
        # No network, no state — construction alone must be side-effect free.
        pass


def _build_broker_with_fake_account(monkeypatch, account: "_FakeAccount") -> "AlpacaBroker":
    """Patch the SDK client names in the broker module, then build a faked broker.

    monkeypatch auto-reverts the setattr calls after the test, so patches never
    leak between tests.
    """
    # Point the class attribute at the account this test wants served.
    _FakeTradingClient.account = account
    # Replace TradingClient in the broker's namespace so __init__ builds the fake.
    monkeypatch.setattr("src.brokers.alpaca_broker.TradingClient", _FakeTradingClient)
    # Replace StockHistoricalDataClient likewise so no real data client is created.
    monkeypatch.setattr("src.brokers.alpaca_broker.StockHistoricalDataClient", _FakeDataClient)
    # Construct directly (NOT from_env — we read no .env); paper=True satisfies the assert.
    return AlpacaBroker(api_key="fake", api_secret="fake", paper=True)


def test_get_account_parses_paper_snapshot_hermetic(monkeypatch) -> None:
    """get_account() maps a faked paper account to an AccountSnapshot, offline.

    Pins BOTH the string->float conversion of the money fields AND the "PA"
    prefix paper detection, with no network call and no credentials.
    """
    # Build a faked paper account with string money values, as the real API sends.
    account = _FakeAccount(
        account_number="PA3XYZABC",   # "PA" prefix => should be detected as paper
        buying_power="94321.50",      # string that must convert to float 94321.50
        cash="88000.00",             # string that must convert to float 88000.00
        portfolio_value="102345.67",  # string that must convert to float 102345.67
    )
    # Construct the broker with the SDK clients faked to serve that account.
    broker = _build_broker_with_fake_account(monkeypatch, account)
    # Exercise the read path — this calls the fake TradingClient.get_account().
    snapshot = broker.get_account()

    # The return type must be our dataclass, not a raw alpaca object.
    assert isinstance(snapshot, AccountSnapshot), (
        f"get_account() should return AccountSnapshot, got {type(snapshot)}"
    )
    # account_number is passed through verbatim as a plain string.
    assert snapshot.account_number == "PA3XYZABC"
    # buying_power must equal the numeric value AND be a genuine float (str was converted).
    assert snapshot.buying_power == 94321.50
    assert type(snapshot.buying_power) is float
    # cash must equal the converted numeric value.
    assert snapshot.cash == 88000.00
    assert type(snapshot.cash) is float
    # portfolio_value must equal the converted numeric value.
    assert snapshot.portfolio_value == 102345.67
    assert type(snapshot.portfolio_value) is float
    # is_paper is derived from the "PA" prefix and must be True here.
    assert snapshot.is_paper is True


def test_verify_paper_account_passes_on_paper_hermetic(monkeypatch) -> None:
    """verify_paper_account() returns None (does not raise) on a paper account.

    The inherited guard calls get_account(); with a "PA" account it must pass.
    """
    # A paper account (money values are irrelevant to the guard, but kept realistic).
    account = _FakeAccount(
        account_number="PA3XYZABC",   # "PA" prefix => is_paper True => guard passes
        buying_power="94321.50",      # realistic string value; unused by the guard
        cash="88000.00",             # realistic string value; unused by the guard
        portfolio_value="102345.67",  # realistic string value; unused by the guard
    )
    # Construct the broker with faked clients serving the paper account.
    broker = _build_broker_with_fake_account(monkeypatch, account)
    # The guard must complete without raising and return None explicitly.
    assert broker.verify_paper_account() is None


def test_verify_paper_account_raises_on_live_hermetic(monkeypatch) -> None:
    """verify_paper_account() raises RuntimeError on a live (non-"PA") account.

    This is the OFFLINE PROOF of the paper-only safety property: a numeric
    account_number yields is_paper False, and the guard must refuse it.
    """
    # A live-style account: numeric identifier, no "PA" prefix => is_paper False.
    account = _FakeAccount(
        account_number="123456789",   # no "PA" prefix => is_paper False => guard raises
        buying_power="94321.50",      # realistic string value; unused by the guard
        cash="88000.00",             # realistic string value; unused by the guard
        portfolio_value="102345.67",  # realistic string value; unused by the guard
    )
    # Construct the broker with faked clients serving the live-style account.
    broker = _build_broker_with_fake_account(monkeypatch, account)
    # The guard must raise RuntimeError rather than allow a live account through.
    with pytest.raises(RuntimeError):
        broker.verify_paper_account()


def test_get_positions_parses_holdings_hermetic(monkeypatch) -> None:
    """get_positions() maps faked Alpaca positions to PositionSnapshots, offline.

    Pins the str->int qty conversion, the str->float money conversions, and the
    .side.value extraction, with no network call and no credentials.
    """
    # A minimal paper account so _build_broker_with_fake_account can construct cleanly;
    # get_positions never reads it, but the builder sets the account class attr.
    account = _FakeAccount(
        account_number="PA3XYZABC",   # paper account, irrelevant to get_positions
        buying_power="94321.50",      # unused by get_positions
        cash="88000.00",             # unused by get_positions
        portfolio_value="102345.67",  # unused by get_positions
    )
    # Two fake holdings with STRING fields, exactly as the real API delivers them.
    spy = _FakePosition(
        symbol="SPY",              # first holding
        qty="10",                  # must convert to int 10
        side="long",               # must surface as "long" via .value
        market_value="4500.00",    # must convert to float 4500.00
        avg_entry_price="440.00",  # must convert to float 440.00
        unrealized_pl="100.00",    # must convert to float 100.00
    )
    tlt = _FakePosition(
        symbol="TLT",              # second holding
        qty="5",                   # must convert to int 5
        side="long",               # must surface as "long" via .value
        market_value="450.00",     # must convert to float 450.00
        avg_entry_price="88.00",   # must convert to float 88.00
        unrealized_pl="-10.00",    # must convert to float -10.00 (negative PL)
    )
    # Set the positions class attr EXPLICITLY for this test to avoid cross-test leakage.
    _FakeTradingClient.positions = [spy, tlt]
    # Build the broker with faked SDK clients (also sets the account class attr).
    broker = _build_broker_with_fake_account(monkeypatch, account)
    # Exercise the read path — this calls the fake TradingClient.get_all_positions().
    positions = broker.get_positions()

    # Two holdings in, two snapshots out — a plain list of the correct length.
    assert isinstance(positions, list)
    assert len(positions) == 2

    # Every item must be our dataclass, not a raw alpaca object.
    for position in positions:
        assert isinstance(position, PositionSnapshot), (
            f"Each position should be PositionSnapshot, got {type(position)}"
        )

    # First snapshot: SPY, with the string fields converted to their typed values.
    first = positions[0]
    assert first.symbol == "SPY"
    # qty must be the converted numeric value AND a genuine int (str was converted).
    assert first.qty == 10
    assert type(first.qty) is int
    # side must be the plain string pulled from _FakeSide.value.
    assert first.side == "long"
    # market_value must equal the converted numeric value and be a genuine float.
    assert first.market_value == 4500.00
    assert type(first.market_value) is float
    # avg_entry_price must equal the converted numeric value.
    assert first.avg_entry_price == 440.00
    # unrealized_pl must equal the converted numeric value.
    assert first.unrealized_pl == 100.00

    # Second snapshot: TLT, confirming a second row and a negative unrealized_pl.
    second = positions[1]
    assert second.symbol == "TLT"
    assert second.qty == 5
    assert type(second.qty) is int
    assert second.unrealized_pl == -10.00


def test_get_positions_empty_account_returns_empty_list_hermetic(monkeypatch) -> None:
    """get_positions() returns [] for a flat account — the common real state.

    A brand-new/flat paper account holds nothing, so get_all_positions() returns
    an empty list; get_positions must surface [] cleanly, not None and no crash.
    """
    # A minimal paper account so the builder can construct the broker.
    account = _FakeAccount(
        account_number="PA3XYZABC",   # paper account, irrelevant here
        buying_power="94321.50",      # unused
        cash="88000.00",             # unused
        portfolio_value="102345.67",  # unused
    )
    # Set positions EXPLICITLY to empty for this test (guards against leakage from
    # the holdings test, since class attributes persist across tests otherwise).
    _FakeTradingClient.positions = []
    # Build the broker with faked SDK clients serving no positions.
    broker = _build_broker_with_fake_account(monkeypatch, account)
    # Exercise the flat-account read path.
    positions = broker.get_positions()

    # Must be an actual empty list — not None, no exception.
    assert positions == []
    assert isinstance(positions, list)
