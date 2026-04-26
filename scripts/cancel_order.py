# cancel_order.py
#
# Cancels a single open Alpaca paper trading order by ID.
# Designed to pair with scripts/limit_order.py — after placing a non-marketable
# limit order, run this script to cancel it cleanly.
#
# Order ID resolution (in priority order):
#   1. CLI argument:  uv run python scripts/cancel_order.py <order-uuid>
#   2. File on disk:  logs/last_limit_order_id.txt  (written by limit_order.py)
#   3. Neither found: error and exit
#
# Safety guarantees (same pattern as first_order.py and limit_order.py):
#   - paper=True is hardcoded and must never be changed in this script
#   - Account number prefix "PA" is verified before any API write is attempted
#   - Order state is fetched and printed before asking for cancellation
#   - Already-terminal orders are detected and skipped without prompting
#
# Usage:
#   uv run python scripts/cancel_order.py                      # reads from file
#   uv run python scripts/cancel_order.py <order-uuid>        # explicit ID

import logging    # Standard library logging for status updates
import os         # For reading environment variables
import sys        # For sys.argv and sys.exit()
import time       # For time.sleep() in the polling loop
from pathlib import Path  # For reading the order ID file

from dotenv import load_dotenv  # Loads .env key=value pairs into os.environ

# Alpaca trading client — used to fetch, and cancel orders
from alpaca.trading.client import TradingClient

# OrderStatus enum — used to check terminal states without string comparisons
from alpaca.trading.enums import OrderStatus

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Path where limit_order.py writes the most recently submitted order ID
ORDER_ID_FILE: Path = Path("logs/last_limit_order_id.txt")

# States from which an order cannot be cancelled — no action needed
TERMINAL_STATES: frozenset[OrderStatus] = frozenset({
    OrderStatus.FILLED,    # Order fully executed — nothing to cancel
    OrderStatus.CANCELED,  # Already cancelled — no double-cancel needed
    OrderStatus.EXPIRED,   # Expired at end of session (DAY orders, etc.)
    OrderStatus.REJECTED,  # Rejected by Alpaca — cannot be cancelled
})

# ---------------------------------------------------------------------------
# Logging setup — INFO level, timestamps, written to stdout.
# We use logging for internal status messages and print() for user-facing output.
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
    load_dotenv()  # Injects .env key=value pairs into os.environ

    api_key = os.getenv("APCA_API_KEY_ID")        # Alpaca key ID
    api_secret = os.getenv("APCA_API_SECRET_KEY")  # Alpaca secret key

    # Both must be present — fail fast before trying to make any API calls
    if not api_key or not api_secret:
        print(
            "Error: APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env\n"
            "Copy .env.example to .env and fill in your paper trading credentials."
        )
        sys.exit(1)  # Exit code 1 = configuration error

    log.info("Credentials loaded from .env")
    return api_key, api_secret


def verify_paper_account(client: TradingClient) -> None:
    """
    Fetches the Alpaca account and confirms it is a paper account.

    Paper accounts always start with "PA". If the prefix is absent, the wrong
    credentials are loaded and we refuse to take any further action.

    Exits with code 1 if the check fails.
    """
    account = client.get_account()  # API call: GET /v2/account

    account_number: str = account.account_number  # e.g. "PA3XYZABC123"

    # SAFETY CHECK: live account numbers do not start with "PA".
    # This ensures we never cancel a real live order by mistake.
    if not account_number.startswith("PA"):
        print(
            f"Error: account number '{account_number}' does not look like a paper account.\n"
            "Expected a number starting with 'PA'. Aborting."
        )
        sys.exit(1)  # Exit code 1 = safety check failed

    log.info("Paper account verified: %s", account_number)


def resolve_order_id() -> str:
    """
    Determines which order ID to target, in priority order:
      1. sys.argv[1] — explicit CLI argument
      2. logs/last_limit_order_id.txt — written by limit_order.py
      3. Neither available — print an error and exit with code 1

    Returns the order UUID as a string.
    """
    # Check if the caller passed an order ID directly on the command line
    if len(sys.argv) > 1:
        order_id: str = sys.argv[1].strip()  # Strip accidental whitespace
        log.info("Using order ID from CLI argument: %s", order_id)
        return order_id  # Use the explicit ID — skip file lookup

    # No CLI argument — try to read from the file written by limit_order.py
    if ORDER_ID_FILE.exists():
        # read_text() reads the whole file; strip() removes the trailing newline
        order_id = ORDER_ID_FILE.read_text().strip()

        if order_id:  # Guard against an empty file
            log.info("Using order ID from %s: %s", ORDER_ID_FILE, order_id)
            return order_id

        # File exists but is empty — treat as missing
        print(f"Error: {ORDER_ID_FILE} exists but is empty.")
        sys.exit(1)  # Exit code 1 = unusable file

    # Neither source had an ID — tell the user exactly how to fix it
    print(
        f"Error: no order ID found.\n"
        f"  Option 1: pass the UUID as a CLI argument:\n"
        f"            uv run python scripts/cancel_order.py <order-uuid>\n"
        f"  Option 2: run scripts/limit_order.py first — it writes the ID to\n"
        f"            {ORDER_ID_FILE} automatically."
    )
    sys.exit(1)  # Exit code 1 = cannot determine target order


def fetch_and_print_order(client: TradingClient, order_id: str) -> object:
    """
    Fetches the current state of the order and prints a summary.
    Returns the order object for the caller to inspect.

    Exits with code 1 if Alpaca returns an error (e.g. invalid UUID).
    """
    try:
        # API call: GET /v2/orders/{order_id} — retrieves the full order object
        order = client.get_order_by_id(order_id)
    except Exception as exc:
        # Most likely causes: typo in UUID, or order belongs to a different account
        print(f"Error fetching order {order_id}: {exc}")
        sys.exit(1)  # Exit code 1 = API error

    # limit_price is only set on limit orders — use getattr with a default to avoid AttributeError
    limit_price = getattr(order, "limit_price", None)
    limit_str: str = f"${float(limit_price):,.2f}" if limit_price else "N/A (market order)"

    # Format submitted_at timestamp for human readability
    submitted_at: str = order.submitted_at.strftime("%Y-%m-%d %H:%M:%S %Z")

    # Print the current order state so the user can verify before confirming cancellation
    print("\n" + "=" * 55)
    print("ORDER DETAILS")
    print("=" * 55)
    print(f"  Order ID:     {order.id}")
    print(f"  Symbol:       {order.symbol}")
    print(f"  Side:         {order.side}")
    print(f"  Quantity:     {order.qty}")
    print(f"  Order type:   {order.order_type}")
    print(f"  Status:       {order.status}")
    print(f"  Limit price:  {limit_str}")
    print(f"  Submitted at: {submitted_at}")
    print("=" * 55)

    return order  # Return the full object so confirm_cancellation can check status


def confirm_cancellation(order: object) -> None:
    """
    Checks whether the order is in a terminal state. If so, exits cleanly.
    Otherwise, prompts the user to confirm cancellation.

    Exits with code 0 if the order is already terminal (no action needed).
    Exits with code 2 if the user declines to cancel.
    """
    # If the order is already in a terminal state, cancellation is not possible or needed
    if order.status in TERMINAL_STATES:
        print(
            f"\nOrder is already in a terminal state ({order.status}).\n"
            "No cancellation needed."
        )
        sys.exit(0)  # Exit code 0 = nothing to do, not an error

    # Order is still open — ask the user to confirm before issuing the cancel
    answer = input('\nType "yes" to cancel this order, or anything else to abort: ').strip()

    if answer.lower() != "yes":
        print("Aborted. No changes were made.")
        sys.exit(2)  # Exit code 2 = user chose not to proceed

    log.info("User confirmed cancellation")


def cancel_and_wait(client: TradingClient, order_id: str) -> int:
    """
    Sends the cancel request and polls for up to 10 seconds until the order
    reaches a terminal state. Logs each status transition.

    Returns an exit code:
      0 — successfully cancelled
      3 — still not terminal after 10 seconds (rare; Alpaca usually cancels instantly)
      1 — unexpected error during cancellation
    """
    try:
        # API call: DELETE /v2/orders/{order_id} — sends the cancel request
        # This call returns None on success; it does not return an order object
        client.cancel_order_by_id(order_id)
        log.info("Cancel request sent for order %s", order_id)
    except Exception as exc:
        # Possible causes: order already gone, network error, permissions issue
        print(f"Error sending cancel request: {exc}")
        return 1  # Exit code 1 = API error

    POLL_INTERVAL: int = 1   # Seconds between each status check
    TIMEOUT: int = 10        # Maximum seconds to wait for cancellation to confirm
    MAX_POLLS: int = TIMEOUT // POLL_INTERVAL  # = 10 iterations

    previous_status: str = ""  # Track the last seen status to detect transitions

    for _ in range(MAX_POLLS):
        try:
            # API call: GET /v2/orders/{order_id} — fetch updated status
            order = client.get_order_by_id(order_id)
        except Exception as exc:
            # Some brokers delete the order record after cancellation — log and continue
            log.warning("Could not fetch order status: %s", exc)
            time.sleep(POLL_INTERVAL)
            continue

        current_status: str = str(order.status)

        # Log only when the status changes — avoids spammy repeated lines
        if current_status != previous_status:
            log.info("Status: %s -> %s", previous_status or "cancelling", current_status)
            previous_status = current_status

        # Check whether we've landed in a terminal state
        if order.status in TERMINAL_STATES:
            if order.status == OrderStatus.CANCELED:
                # Clean cancellation — the expected outcome
                print(f"\nOrder {order_id} successfully cancelled.")
                return 0  # Exit code 0 = success

            # Terminal but not CANCELED (e.g. filled between cancel request and confirmation)
            print(f"\nOrder reached terminal state: {order.status}. No further action needed.")
            return 0  # Exit code 0 — terminal is good enough, nothing more to do

        # Not yet terminal — wait before polling again
        time.sleep(POLL_INTERVAL)

    # Polled 10 times without a terminal state — this is unusual for Alpaca
    print(
        f"\nOrder {order_id} is still not cancelled after {TIMEOUT} seconds.\n"
        "This is unexpected. Check the Alpaca dashboard manually:\n"
        "https://app.alpaca.markets/paper-trading/orders"
    )
    return 3  # Exit code 3 = timeout, state unknown


def main() -> None:
    """
    Entry point. Orchestrates credentials, safety check, order lookup,
    user confirmation, cancellation, and polling.
    """
    # Step 1: Load API keys from .env — exits with code 1 if missing
    api_key, api_secret = load_credentials()

    # Step 2: Create the trading client.
    # SAFETY: paper=True is hardcoded here and must NEVER be changed to False.
    # Changing this would allow cancellation of real live orders.
    trading_client = TradingClient(api_key, api_secret, paper=True)

    # Step 3: Verify the account is a paper account (PA prefix)
    verify_paper_account(trading_client)

    # Step 4: Determine the order ID from CLI arg or from the handoff file
    order_id: str = resolve_order_id()

    # Step 5: Fetch and display the order's current state
    order = fetch_and_print_order(trading_client, order_id)

    # Step 6: Check if it's already terminal; if not, prompt for confirmation
    confirm_cancellation(order)

    # Step 7: Send the cancel request and poll until terminal or timeout
    exit_code: int = cancel_and_wait(trading_client, order_id)

    sys.exit(exit_code)


# Only run main() when this script is executed directly (not when imported)
if __name__ == "__main__":
    main()
