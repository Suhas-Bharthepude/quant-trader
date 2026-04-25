# first_order.py
#
# Submits a single paper trading market order for 1 share of SPY through Alpaca.
# This script is intentionally verbose — every line is commented for learning purposes.
#
# Safety guarantees:
#   - paper=True is hardcoded and must never be changed in this script
#   - Account number prefix "PA" is verified before any order is submitted
#   - Dry-run preview and explicit "yes" confirmation are required before submitting
#
# Usage:
#   uv run python scripts/first_order.py

import logging  # Standard library logging — used for status updates, not print
import os       # For reading environment variables
import sys      # For sys.exit() with specific exit codes
import time     # For time.sleep() in the polling loop

from datetime import datetime  # For formatting timestamps in a readable way

from dotenv import load_dotenv  # Loads key=value pairs from .env into os.environ

# Alpaca trading client — used to place orders and fetch account info
from alpaca.trading.client import TradingClient

# Data structure that describes the order we want to place
from alpaca.trading.requests import MarketOrderRequest

# Enums for order fields — using enums avoids typos in string literals
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

# Separate client for market data (quotes, trades) — requires no key on free tier
# but we pass keys anyway so rate limits are higher
from alpaca.data.historical import StockHistoricalDataClient

# Request object for fetching the latest trade for a symbol
from alpaca.data.requests import StockLatestTradeRequest

# ---------------------------------------------------------------------------
# CONSTANTS — do not change these values in this script.
# This script is a controlled smoke test for exactly 1 share of SPY.
# ---------------------------------------------------------------------------
SYMBOL: str = "SPY"   # The ticker symbol we are trading
QTY: int = 1          # Number of shares — keep at 1 for this smoke-test script

# ---------------------------------------------------------------------------
# Logging setup — INFO level prints timestamps and messages to stdout.
# We use logging for internal status updates and print() for user-facing prompts.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # Logger scoped to this module


def load_credentials() -> tuple[str, str]:
    """
    Loads API credentials from the .env file.
    Returns (api_key, api_secret) or exits with code 1 if either is missing.
    """
    load_dotenv()  # Reads .env in the current working directory into os.environ

    api_key = os.getenv("APCA_API_KEY_ID")        # Alpaca API key ID
    api_secret = os.getenv("APCA_API_SECRET_KEY")  # Alpaca API secret key

    # Both values must be present — fail fast before trying to connect
    if not api_key or not api_secret:
        print(
            "Error: APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env\n"
            "Copy .env.example to .env and fill in your paper trading credentials."
        )
        sys.exit(1)  # Exit code 1 = configuration error

    log.info("Credentials loaded from .env")
    return api_key, api_secret  # Return both values as a tuple


def verify_paper_account(client: TradingClient) -> object:
    """
    Fetches the account and verifies it is a paper account.

    Alpaca paper accounts always have an account number starting with "PA".
    If the account number does NOT start with "PA", we refuse to proceed.
    This is a belt-and-suspenders check in case the wrong API keys are loaded.

    Returns the account object on success, exits with code 1 on failure.
    """
    account = client.get_account()  # API call: GET /v2/account

    account_number: str = account.account_number  # e.g. "PA3XYZABC123"

    # SAFETY CHECK 2: Verify this is a paper account before doing anything else.
    # Live accounts have numeric account numbers without a "PA" prefix.
    if not account_number.startswith("PA"):
        print(
            f"Error: account number '{account_number}' does not look like a paper account.\n"
            "Expected a number starting with 'PA'. Aborting to protect real funds."
        )
        sys.exit(1)  # Exit code 1 = safety check failed

    log.info("Paper account verified: %s", account_number)
    return account  # Return the full account object for use in the preview


def check_market_status(client: TradingClient) -> bool:
    """
    Checks whether the US stock market is currently open.

    Returns True if open, False if closed.
    If closed, prints a warning with the next open time and asks for confirmation.
    The order will still be submitted if the user confirms — Alpaca will queue it.
    """
    clock = client.get_clock()  # API call: GET /v2/clock

    if clock.is_open:
        # Market is open right now — orders will execute immediately
        log.info("Market is OPEN")
        return True  # No confirmation needed

    # Market is closed — format the next open time for human readability
    next_open: str = clock.next_open.strftime("%A %Y-%m-%d at %I:%M %p %Z")
    print(
        f"\nWarning: the market is currently CLOSED.\n"
        f"Next open: {next_open}\n"
        f"If you proceed, Alpaca will queue the order and submit it at open.\n"
    )

    # Ask the user whether they still want to submit a queued order
    answer = input("Submit anyway? (yes to continue, anything else to cancel): ").strip().lower()

    if answer != "yes":
        # User chose not to submit — exit cleanly with code 2 (user cancellation)
        print("Cancelled.")
        sys.exit(2)  # Exit code 2 = cancelled by user

    log.info("User confirmed submission despite market being closed")
    return False  # Market is closed but user chose to proceed


def build_dry_run_preview(
    account: object,
    data_client: StockHistoricalDataClient,
) -> float:
    """
    Fetches the latest trade price for SYMBOL and prints a human-readable
    order preview. Returns the estimated cost so it can be shown in the prompt.

    This runs BEFORE any order is placed — it is purely informational.
    """
    # Build the request object for the latest trade data
    trade_request = StockLatestTradeRequest(symbol_or_symbols=SYMBOL)

    # Fetch the latest trade — this is the most recent recorded transaction price
    latest_trade = data_client.get_stock_latest_trade(trade_request)

    # The response is a dict keyed by symbol; extract the price as a float
    latest_price: float = float(latest_trade[SYMBOL].price)

    # Estimated cost = price × quantity (no commissions on Alpaca)
    estimated_cost: float = latest_price * QTY

    # Print the full dry-run summary so the user can review before confirming
    print("\n" + "=" * 50)
    print("DRY-RUN PREVIEW — no order has been placed yet")
    print("=" * 50)
    print(f"  Account number:   {account.account_number}")
    print(f"  Buying power:     ${float(account.buying_power):,.2f}")
    print(f"  Symbol:           {SYMBOL}")
    print(f"  Quantity:         {QTY} share(s)")
    print(f"  Side:             BUY")
    print(f"  Order type:       MARKET")
    print(f"  Time in force:    DAY")
    print(f"  Latest price:     ${latest_price:,.2f}  (last trade — actual fill may differ)")
    print(f"  Estimated cost:   ${estimated_cost:,.2f}")
    print("=" * 50)

    return estimated_cost  # Caller can use this in the confirmation prompt


def submit_order(client: TradingClient) -> object:
    """
    Builds and submits a DAY market order for SYMBOL.
    Returns the order object returned by Alpaca.
    """
    # MarketOrderRequest describes the order — Alpaca fills it at the best available price
    order_request = MarketOrderRequest(
        symbol=SYMBOL,                  # Ticker to buy
        qty=QTY,                        # Number of shares
        side=OrderSide.BUY,             # BUY (not SELL)
        time_in_force=TimeInForce.DAY,  # DAY = cancel if not filled by end of session
    )

    log.info("Submitting %s x%d BUY MARKET DAY order...", SYMBOL, QTY)

    # API call: POST /v2/orders — this is the line that places the actual order
    order = client.submit_order(order_request)

    log.info("Order submitted — ID: %s  status: %s", order.id, order.status)
    return order  # Return the full order object so we can poll its status


def poll_order_status(client: TradingClient, order_id: str) -> int:
    """
    Polls the order every 2 seconds for up to 60 seconds, printing status changes.

    Returns an exit code:
      0 — order filled
      1 — order rejected or cancelled
      3 — timeout (still pending after 60 seconds)
    """
    POLL_INTERVAL_SECONDS: int = 2   # How long to wait between status checks
    TIMEOUT_SECONDS: int = 60        # Give up after this many seconds
    MAX_POLLS: int = TIMEOUT_SECONDS // POLL_INTERVAL_SECONDS  # = 30 iterations

    previous_status: str = ""  # Track the last known status to detect changes

    for attempt in range(MAX_POLLS):
        # API call: GET /v2/orders/{order_id} — fetches the current order state
        order = client.get_order_by_id(order_id)

        current_status: str = str(order.status)  # Convert enum to string for comparison

        # Only log when the status actually changes — avoids spammy duplicate lines
        if current_status != previous_status:
            log.info("Order status: %s -> %s", previous_status or "submitted", current_status)
            previous_status = current_status  # Update for next iteration

        # --- Terminal states: stop polling and return the appropriate exit code ---

        if order.status == OrderStatus.FILLED:
            # Order fully executed — print the fill details and exit successfully
            fill_price: float = float(order.filled_avg_price)
            fill_qty: int = int(order.filled_qty)
            fill_time: str = order.filled_at.strftime("%Y-%m-%d %H:%M:%S %Z")

            print("\n" + "=" * 50)
            print("ORDER FILLED")
            print("=" * 50)
            print(f"  Fill price:  ${fill_price:,.4f}")
            print(f"  Filled qty:  {fill_qty} share(s)")
            print(f"  Filled at:   {fill_time}")
            print("=" * 50)
            return 0  # Exit code 0 = success

        if order.status in (OrderStatus.REJECTED, OrderStatus.CANCELED):
            # Something went wrong — print why and exit with error
            reason: str = getattr(order, "failed_at", "no reason provided")
            print(f"\nOrder {order.status}: {reason}")
            return 1  # Exit code 1 = error / rejection

        # Not yet in a terminal state — wait before polling again
        time.sleep(POLL_INTERVAL_SECONDS)

    # We exhausted all polling attempts without a terminal state
    print(
        f"\nOrder is still pending after {TIMEOUT_SECONDS} seconds.\n"
        f"Order ID: {order_id}\n"
        f"Check status at: https://app.alpaca.markets/paper-trading/orders"
    )
    return 3  # Exit code 3 = timeout, order still pending


def main() -> None:
    """
    Entry point. Orchestrates credential loading, safety checks,
    dry-run preview, order submission, and status polling.
    """
    # Step 1: Load API keys from .env — exits if keys are missing
    api_key, api_secret = load_credentials()

    # Step 2: Create the trading client.
    # SAFETY CHECK 1: paper=True is hardcoded here and must NEVER be changed to False.
    # Changing this to paper=False would route real money orders to a live brokerage account.
    trading_client = TradingClient(api_key, api_secret, paper=True)

    # Create the data client for fetching quotes and latest prices
    # We pass keys so we get authenticated rate limits (higher request ceiling)
    data_client = StockHistoricalDataClient(api_key, api_secret)

    # Step 3: Fetch the account and verify it is a paper account (PA prefix)
    account = verify_paper_account(trading_client)

    # Step 4: Check market hours — prompts user if market is closed
    check_market_status(trading_client)

    # Step 5: Fetch latest price and print the full dry-run preview
    build_dry_run_preview(account, data_client)

    # Step 6: Ask the user to type "yes" to confirm — any other input cancels
    confirmation = input('\nType "yes" to submit this order, or anything else to cancel: ').strip()

    if confirmation.lower() != "yes":
        print("Order cancelled. No order was placed.")
        sys.exit(2)  # Exit code 2 = cancelled by user at confirmation prompt

    # Step 7: Submit the market order — this is the only line that sends money
    order = submit_order(trading_client)

    # Print the order receipt immediately after submission
    submitted_at: str = order.submitted_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\nOrder placed — ID: {order.id}")
    print(f"Status: {order.status}  |  Submitted at: {submitted_at}")

    # Step 8: Poll until filled, rejected, or timeout
    exit_code: int = poll_order_status(trading_client, str(order.id))

    # Exit with the code determined by poll_order_status (0, 1, or 3)
    sys.exit(exit_code)


# Only run main() when this script is executed directly (not when imported)
if __name__ == "__main__":
    main()
