# limit_order.py
#
# Submits a GTC limit BUY order for 1 share of SPY priced 20% below market.
# The order is intentionally non-marketable — it will NOT fill under normal conditions.
# Purpose: exercise the limit order code path and the cancel flow safely on paper.
#
# After submission, the order ID is saved to logs/last_limit_order_id.txt so that
# scripts/cancel_order.py can pick it up without you having to copy-paste the UUID.
#
# Safety guarantees (same pattern as first_order.py):
#   - paper=True is hardcoded and must never be changed in this script
#   - Account number prefix "PA" is verified before any order is submitted
#   - Limit price is asserted to be < 85% of market price before submitting
#
# Usage:
#   uv run python scripts/limit_order.py

import logging   # Standard library logging for internal status updates
import os        # For reading environment variables
import sys       # For sys.exit() with specific exit codes
from pathlib import Path  # For writing the order ID file cleanly

from dotenv import load_dotenv  # Loads .env key=value pairs into os.environ

# Alpaca trading client — places orders and fetches account info
from alpaca.trading.client import TradingClient

# LimitOrderRequest describes a limit order (unlike MarketOrderRequest, price is specified)
from alpaca.trading.requests import LimitOrderRequest

# Enums for order side and time-in-force — avoids string typos
from alpaca.trading.enums import OrderSide, TimeInForce

# Separate client used exclusively for market data (quotes, latest trades)
from alpaca.data.historical import StockHistoricalDataClient

# Request object for fetching the most recent trade for a given symbol
from alpaca.data.requests import StockLatestTradeRequest

# ---------------------------------------------------------------------------
# CONSTANTS — do not change these values in this script.
# This script is a controlled exercise for exactly 1 share of SPY.
# ---------------------------------------------------------------------------
SYMBOL: str = "SPY"  # The ticker symbol we are trading
QTY: int = 1         # Number of shares — keep at 1 for this exercise

# How far below market price to set the limit — 20% ensures it won't accidentally fill
LIMIT_DISCOUNT: float = 0.20

# Path where we write the order ID so the cancel script can read it automatically
ORDER_ID_FILE: Path = Path("logs/last_limit_order_id.txt")

# ---------------------------------------------------------------------------
# Logging setup — INFO level, timestamps, written to stdout.
# We use logging for internal status messages and print() for user-facing prompts.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # Logger scoped to this module


def load_credentials() -> tuple[str, str]:
    """
    Reads API credentials from the .env file.
    Returns (api_key, api_secret) or exits with code 1 if either is missing.
    """
    load_dotenv()  # Injects .env variables into os.environ

    api_key = os.getenv("APCA_API_KEY_ID")        # Alpaca key ID
    api_secret = os.getenv("APCA_API_SECRET_KEY")  # Alpaca secret key

    # Both must be present — exit immediately rather than producing a cryptic API error
    if not api_key or not api_secret:
        print(
            "Error: APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env\n"
            "Copy .env.example to .env and fill in your paper trading credentials."
        )
        sys.exit(1)  # Exit code 1 = configuration error

    log.info("Credentials loaded from .env")
    return api_key, api_secret


def verify_paper_account(client: TradingClient) -> object:
    """
    Fetches the Alpaca account and confirms it is a paper account.

    Paper accounts always begin with "PA". If the prefix is missing, we refuse
    to place any order — this guards against accidentally using live credentials.

    Returns the account object on success, exits with code 1 on failure.
    """
    account = client.get_account()  # API call: GET /v2/account

    account_number: str = account.account_number  # e.g. "PA3XYZABC123"

    # SAFETY CHECK: live accounts use a numeric account number without "PA".
    # If this check fails, the wrong credentials are almost certainly loaded.
    if not account_number.startswith("PA"):
        print(
            f"Error: account number '{account_number}' does not look like a paper account.\n"
            "Expected a number starting with 'PA'. Aborting to protect real funds."
        )
        sys.exit(1)  # Exit code 1 = safety check failed

    log.info("Paper account verified: %s", account_number)
    return account  # Return the account for use in the preview step


def fetch_current_price(data_client: StockHistoricalDataClient) -> float:
    """
    Fetches the most recent trade price for SYMBOL.
    Returns the price as a float.
    """
    # Describe what we want: the latest trade for SYMBOL
    request = StockLatestTradeRequest(symbol_or_symbols=SYMBOL)

    # API call: fetches the last recorded transaction price (not a bid/ask quote)
    latest_trade = data_client.get_stock_latest_trade(request)

    # The result is a dict keyed by symbol; extract the price field
    price: float = float(latest_trade[SYMBOL].price)

    log.info("Current %s price: $%.2f", SYMBOL, price)
    return price


def compute_limit_price(current_price: float) -> float:
    """
    Computes a limit price 20% below the current market price.

    This is intentionally non-marketable: Alpaca will accept the order and
    leave it open, but it won't execute unless the price drops to this level
    (which would require a significant crash — unlikely in a normal session).

    Raises AssertionError if the computed price is not safely below market,
    since a rounding error could accidentally create a marketable order.
    """
    # Subtract 20% of the current price to get our intended limit level
    raw_limit: float = current_price * (1 - LIMIT_DISCOUNT)

    # Round to 2 decimal places — Alpaca requires prices in dollars and cents
    limit_price: float = round(raw_limit, 2)

    # SAFETY CHECK: paranoia assertion — if limit_price is >= 85% of current price,
    # something went wrong in the math (e.g. LIMIT_DISCOUNT was set too small).
    # We'd rather crash here than submit a price that could actually fill.
    assert limit_price < current_price * 0.85, (
        f"Computed limit price ${limit_price:.2f} is too close to market price "
        f"${current_price:.2f}. The order might fill. Aborting."
    )

    log.info("Limit price computed: $%.2f (%.0f%% below market)", limit_price, LIMIT_DISCOUNT * 100)
    return limit_price


def build_dry_run_preview(
    account: object,
    current_price: float,
    limit_price: float,
) -> None:
    """
    Prints a human-readable order preview before asking for confirmation.
    No order is placed at this point — this is purely informational.
    """
    print("\n" + "=" * 55)
    print("DRY-RUN PREVIEW — no order has been placed yet")
    print("=" * 55)
    print(f"  Account number:    {account.account_number}")
    print(f"  Symbol:            {SYMBOL}")
    print(f"  Quantity:          {QTY} share(s)")
    print(f"  Side:              BUY")
    print(f"  Order type:        LIMIT")
    print(f"  Time in force:     GTC  (Good-Till-Cancelled)")
    print(f"  Current price:     ${current_price:,.2f}  (latest trade)")
    print(f"  Limit price:       ${limit_price:,.2f}  ({LIMIT_DISCOUNT:.0%} below market)")
    print()
    print("  NOTE: This order is designed NOT to fill.")
    print("  It will sit open in your paper account until you cancel it.")
    print("=" * 55)


def submit_limit_order(client: TradingClient, limit_price: float) -> object:
    """
    Builds and submits the GTC limit BUY order.
    Returns the order object returned by Alpaca.
    """
    # LimitOrderRequest requires an explicit limit_price, unlike MarketOrderRequest
    order_request = LimitOrderRequest(
        symbol=SYMBOL,                   # Ticker to buy
        qty=QTY,                         # Number of shares
        side=OrderSide.BUY,              # BUY (not SELL)
        time_in_force=TimeInForce.GTC,   # GTC = survives across market sessions until cancelled
        limit_price=limit_price,         # The price ceiling we are willing to pay
    )

    log.info(
        "Submitting %s x%d BUY LIMIT GTC @ $%.2f...",
        SYMBOL, QTY, limit_price,
    )

    # API call: POST /v2/orders — this is the only line that places the actual order
    order = client.submit_order(order_request)

    log.info("Order submitted — ID: %s  status: %s", order.id, order.status)
    return order  # Return full order object so caller can print the receipt


def save_order_id(order_id: str) -> None:
    """
    Writes the order UUID to logs/last_limit_order_id.txt.

    This is a simple handoff mechanism between this script and cancel_order.py.
    It is NOT a real persistence layer — just a plain text file with one UUID.
    If you run this script multiple times, the file is overwritten with the latest ID.
    """
    # Ensure the logs/ directory exists — it should already from the scaffold,
    # but this guard prevents a crash if the directory was accidentally deleted
    ORDER_ID_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Write just the UUID on a single line — no JSON, no headers, keep it simple
    ORDER_ID_FILE.write_text(order_id + "\n")

    log.info("Order ID saved to %s", ORDER_ID_FILE)


def main() -> None:
    """
    Entry point. Orchestrates credentials, safety checks, preview, submission,
    receipt printing, and order ID persistence.
    """
    # Step 1: Load API keys from .env — exits with code 1 if either is missing
    api_key, api_secret = load_credentials()

    # Step 2: Create the trading client.
    # SAFETY: paper=True is hardcoded here and must NEVER be changed to False.
    # Removing or toggling this flag would route real money orders to a live account.
    trading_client = TradingClient(api_key, api_secret, paper=True)

    # Create the market data client for fetching the latest SPY trade price
    data_client = StockHistoricalDataClient(api_key, api_secret)

    # Step 3: Fetch account and verify it is a paper account (PA prefix)
    account = verify_paper_account(trading_client)

    # Step 4: Fetch the current SPY market price
    current_price: float = fetch_current_price(data_client)

    # Step 5: Compute a limit price 20% below market — asserts it is safely below
    limit_price: float = compute_limit_price(current_price)

    # Step 6: Print the dry-run preview so the user can review before confirming
    build_dry_run_preview(account, current_price, limit_price)

    # Step 7: Require the user to type "yes" — any other input cancels the script
    confirmation = input('\nType "yes" to submit this order, or anything else to cancel: ').strip()

    if confirmation.lower() != "yes":
        print("Order cancelled. No order was placed.")
        sys.exit(2)  # Exit code 2 = cancelled by user

    # Step 8: Submit the limit order — this is the only call that places an order
    order = submit_limit_order(trading_client, limit_price)

    # Step 9: Print the order receipt immediately after submission
    submitted_at: str = order.submitted_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    print("\n" + "=" * 55)
    print("ORDER SUBMITTED")
    print("=" * 55)
    print(f"  Order ID:     {order.id}")
    print(f"  Status:       {order.status}")
    print(f"  Submitted at: {submitted_at}")
    print(f"  Limit price:  ${float(order.limit_price):,.2f}")
    print(f"\n  Run scripts/cancel_order.py to cancel this order.")
    print("=" * 55)

    # Step 10: Save the order ID to disk so the cancel script can read it automatically
    save_order_id(str(order.id))

    sys.exit(0)  # Exit code 0 = success


# Only run main() when this script is executed directly (not when imported as a module)
if __name__ == "__main__":
    main()
