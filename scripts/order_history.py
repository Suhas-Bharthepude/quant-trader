# order_history.py
#
# Fetches and prints the 50 most recent orders from the Alpaca paper account.
# Shows a formatted table (newest first) and a summary count grouped by status.
#
# Useful for auditing what has been placed, filled, and cancelled during a session
# without having to open the Alpaca dashboard.
#
# Usage:
#   uv run python scripts/order_history.py

import logging    # Standard library logging for internal status messages
import os         # For reading environment variables
import sys        # For sys.exit() with specific exit codes
from collections import Counter  # For counting orders by status in the summary

from dotenv import load_dotenv  # Loads .env key=value pairs into os.environ

# Alpaca trading client — used to fetch account info and order history
from alpaca.trading.client import TradingClient

# GetOrdersRequest is the filter object passed to client.get_orders()
from alpaca.trading.requests import GetOrdersRequest

# QueryOrderStatus controls which status bucket to query (all, open, closed)
from alpaca.trading.enums import QueryOrderStatus

# ---------------------------------------------------------------------------
# Column widths for the printed table — adjust if values get truncated
# ---------------------------------------------------------------------------
COL_SUBMITTED  = 20  # "2026-04-25 14:30:00"
COL_SYMBOL     =  6  # "SPY"
COL_SIDE       =  5  # "buy" / "sell"
COL_QTY        =  5  # "1"
COL_TYPE       =  7  # "market" / "limit"
COL_STATUS     = 10  # "filled" / "canceled"
COL_FILLED_QTY =  10 # "1"
COL_FILL_PRICE = 11  # "$713.9600"
COL_LIMIT      = 10  # "$571.17"

# ---------------------------------------------------------------------------
# Logging setup — INFO level, timestamps, written to stdout.
# We use logging for internal status messages and print() for the table output.
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

    # Both must be present — exit immediately rather than producing a cryptic API error
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

    Paper accounts always begin with "PA". If the prefix is absent, the wrong
    credentials are loaded and we abort immediately.

    Exits with code 1 if the check fails.
    """
    account = client.get_account()  # API call: GET /v2/account

    account_number: str = account.account_number  # e.g. "PA3XYZABC123"

    # SAFETY CHECK: live accounts use a numeric account number without "PA".
    # This read-only script can't place orders, but we keep the check for consistency
    # so the safety pattern is identical across all scripts in this project.
    if not account_number.startswith("PA"):
        print(
            f"Error: account number '{account_number}' does not look like a paper account.\n"
            "Expected a number starting with 'PA'. Aborting."
        )
        sys.exit(1)  # Exit code 1 = wrong account

    log.info("Paper account verified: %s", account_number)


def fetch_orders(client: TradingClient) -> list:
    """
    Fetches the 50 most recent orders across all statuses, sorted newest first.
    Returns a list of order objects.
    """
    # GetOrdersRequest is the filter that the API uses to narrow down results.
    # Without a filter, get_orders() only returns open orders — we want everything.
    request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,  # Include filled, canceled, rejected — not just open
        limit=50,                     # Maximum number of orders to return (50 is the API max per page)
        direction="desc",             # Newest orders first — most useful for a recent-activity view
    )

    log.info("Fetching up to 50 recent orders (all statuses, newest first)...")

    # API call: GET /v2/orders — returns a list of Order objects matching the filter
    orders = client.get_orders(filter=request)

    log.info("Received %d order(s)", len(orders))
    return orders  # Return the raw list for the caller to format and print


def fmt(value: object, width: int, prefix: str = "") -> str:
    """
    Formats a value into a left-justified fixed-width string for table alignment.

    If the value is None or empty, returns "-" padded to the given width.
    The optional prefix (e.g. "$") is prepended only when value is present.

    Examples:
        fmt(713.96, 11, "$")  ->  "$713.96    "
        fmt(None, 11, "$")    ->  "-          "
    """
    if value is None or str(value).strip() == "":
        # No data available for this field — use a dash as a placeholder
        return "-".ljust(width)

    # Build the display string with optional prefix, then pad to fixed width
    text: str = f"{prefix}{value}"
    return text.ljust(width)  # ljust = left-justify, padding with spaces on the right


def print_table(orders: list) -> None:
    """
    Prints the order list as a fixed-width text table.
    Each row is one order; columns are aligned using the COL_* width constants.
    """
    # --- Header row ---
    # Each column name is left-justified to match the data column width below it
    header: str = (
        "SUBMITTED AT".ljust(COL_SUBMITTED)
        + "SYMBOL".ljust(COL_SYMBOL)
        + "SIDE".ljust(COL_SIDE)
        + "QTY".ljust(COL_QTY)
        + "TYPE".ljust(COL_TYPE)
        + "STATUS".ljust(COL_STATUS)
        + "FILL QTY".ljust(COL_FILLED_QTY)
        + "FILL PRICE".ljust(COL_FILL_PRICE)
        + "LIMIT".ljust(COL_LIMIT)
    )

    # Separator line — same total width as the header
    separator: str = "-" * len(header)

    print()  # Blank line before the table for visual breathing room
    print(header)
    print(separator)

    for order in orders:
        # submitted_at is a datetime object — format it to "YYYY-MM-DD HH:MM:SS"
        # We truncate at seconds because milliseconds add noise without value here
        submitted: str = order.submitted_at.strftime("%Y-%m-%d %H:%M:%S")

        # side and order_type come back as enums — convert to string for display
        side: str = str(order.side).replace("OrderSide.", "")
        order_type: str = str(order.order_type).replace("OrderType.", "")
        status: str = str(order.status).replace("OrderStatus.", "")

        # qty is the requested quantity — it's a string in the SDK response
        qty: str = str(order.qty)

        # filled_qty may be "0" even on open orders — show it as-is
        filled_qty: str = str(order.filled_qty) if order.filled_qty is not None else None

        # filled_avg_price is only set after a fill — None on open or cancelled orders
        fill_price = (
            f"{float(order.filled_avg_price):.4f}"
            if order.filled_avg_price is not None
            else None
        )

        # limit_price is only set on limit orders — None on market orders
        limit_price = (
            f"{float(order.limit_price):.2f}"
            if order.limit_price is not None
            else None
        )

        # Build the row by formatting each field into its fixed-width column
        row: str = (
            fmt(submitted,   COL_SUBMITTED)   # e.g. "2026-04-25 14:30:00"
            + fmt(order.symbol,  COL_SYMBOL)  # e.g. "SPY   "
            + fmt(side,          COL_SIDE)    # e.g. "buy  "
            + fmt(qty,           COL_QTY)     # e.g. "1    "
            + fmt(order_type,    COL_TYPE)    # e.g. "market "
            + fmt(status,        COL_STATUS)  # e.g. "filled    "
            + fmt(filled_qty,    COL_FILLED_QTY)          # e.g. "1         "
            + fmt(fill_price,    COL_FILL_PRICE, "$")     # e.g. "$713.9600  "
            + fmt(limit_price,   COL_LIMIT,     "$")      # e.g. "$571.17   "
        )

        print(row)

    print(separator)
    print()  # Blank line after the table


def print_summary(orders: list) -> None:
    """
    Counts orders grouped by status and prints a one-line summary.

    Example output:
        Summary: FILLED: 1  |  CANCELED: 1  |  ACCEPTED: 0
    """
    # Counter tallies occurrences of each unique value — perfect for status grouping
    # We convert status to string and strip the enum prefix for clean labels
    status_counts: Counter = Counter(
        str(order.status).replace("OrderStatus.", "").upper()
        for order in orders
    )

    if not status_counts:
        # No orders at all — say so rather than printing an empty summary
        print("No orders found.")
        return

    # Format each status and its count as "STATUS: N", then join with " | " separator
    summary_parts: list[str] = [
        f"{status}: {count}"
        for status, count in sorted(status_counts.items())  # Sort alphabetically for consistency
    ]

    print("Summary: " + "  |  ".join(summary_parts))


def main() -> None:
    """
    Entry point. Orchestrates credentials, safety check, data fetch, and output.
    """
    # Step 1: Load API keys from .env — exits with code 1 if either is missing
    api_key, api_secret = load_credentials()

    # Step 2: Create the trading client.
    # SAFETY: paper=True is hardcoded here and must NEVER be changed to False.
    # This script is read-only, but we keep the flag consistent with all other scripts.
    trading_client = TradingClient(api_key, api_secret, paper=True)

    # Step 3: Verify the account is a paper account (PA prefix check)
    verify_paper_account(trading_client)

    try:
        # Step 4: Fetch the 50 most recent orders across all statuses
        orders = fetch_orders(trading_client)

        if not orders:
            # No orders exist yet — print a friendly message rather than an empty table
            print("\nNo orders found in this account.")
            sys.exit(0)  # Exit code 0 = success, just nothing to show

        # Step 5: Print the formatted table
        print_table(orders)

        # Step 6: Print the status summary line
        print_summary(orders)

    except Exception as exc:
        # Catch-all for unexpected API errors (network, auth, rate limit, etc.)
        print(f"Error fetching order history: {exc}")
        sys.exit(1)  # Exit code 1 = unexpected error

    sys.exit(0)  # Exit code 0 = success


# Only run main() when this script is executed directly (not when imported)
if __name__ == "__main__":
    main()
