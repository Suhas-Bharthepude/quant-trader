# order_history.py
#
# Fetches and prints the 50 most recent orders from the Alpaca paper account.
# Shows a fixed-width table (newest first) and a summary count grouped by status.
#
# Useful for auditing what has been placed, filled, and cancelled without opening
# the Alpaca dashboard.
#
# Usage:
#   uv run python scripts/order_history.py

import logging  # timestamps and severity to stdout
import sys  # sys.exit() with specific exit codes
from collections import Counter  # tallies occurrences for the status summary

from src.brokers.alpaca_broker import AlpacaBroker  # Alpaca-backed concrete broker

# ---------------------------------------------------------------------------
# Column widths for the printed table — adjust if values get truncated.
# ---------------------------------------------------------------------------
COL_SUBMITTED  = 20  # "2026-04-25 14:30:00"
COL_SYMBOL     =  6  # "SPY"
COL_SIDE       =  5  # "buy" / "sell"
COL_QTY        =  5  # "1"
COL_TYPE       =  7  # "market" / "limit"
COL_STATUS     = 10  # "filled" / "canceled"
COL_FILLED_QTY = 10  # "1"
COL_FILL_PRICE = 11  # "$713.9600"
COL_LIMIT      = 10  # "$571.17"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # logger scoped to this module


def fmt(value: object, width: int, prefix: str = "") -> str:
    """Left-justify value into a fixed-width column; show "-" when value is None or empty.

    The optional prefix (e.g. "$") is prepended only when value is present.
    Examples:  fmt(713.96, 11, "$") -> "$713.96    "
               fmt(None,   11, "$") -> "-          "
    """
    if value is None or str(value).strip() == "":
        return "-".ljust(width)             # placeholder for missing data
    return f"{prefix}{value}".ljust(width)  # prefix + value, padded to fixed width


def print_table(orders: list) -> None:
    """Print all orders as a fixed-width text table, one row per order."""
    # Build header by left-justifying each column name to its defined width
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
    separator: str = "-" * len(header)  # horizontal rule matching header width

    print()
    print(header)
    print(separator)

    for order in orders:
        # submitted_at is a datetime in OrderResult — format to seconds precision
        submitted: str = order.submitted_at.strftime("%Y-%m-%d %H:%M:%S")

        # side and order_type are str-Enums in OrderResult; .value gives "buy"/"sell" etc.
        side: str = order.side.value
        order_type: str = order.order_type.value

        # status is already a plain string in OrderResult — no enum stripping needed
        qty: str = str(order.qty)  # qty is int in OrderResult; convert to str for fmt()

        # filled_qty is Optional[int] — use is-not-None so "0" still renders (not falsy)
        filled_qty = str(order.filled_qty) if order.filled_qty is not None else None

        # filled_avg_price is Optional[float] — already a float, no cast needed
        fill_price = f"{order.filled_avg_price:.4f}" if order.filled_avg_price is not None else None

        # limit_price is Optional[float] — present for limit orders, None for market orders
        limit_price = f"{order.limit_price:.2f}" if order.limit_price is not None else None

        row: str = (
            fmt(submitted,   COL_SUBMITTED)
            + fmt(order.symbol,  COL_SYMBOL)
            + fmt(side,          COL_SIDE)
            + fmt(qty,           COL_QTY)
            + fmt(order_type,    COL_TYPE)
            + fmt(order.status,  COL_STATUS)   # plain string — no cleanup required
            + fmt(filled_qty,    COL_FILLED_QTY)
            + fmt(fill_price,    COL_FILL_PRICE, "$")
            + fmt(limit_price,   COL_LIMIT,     "$")
        )
        print(row)

    print(separator)
    print()


def print_summary(orders: list) -> None:
    """Print a one-line count of orders grouped by status, sorted alphabetically."""
    # status is a plain string in OrderResult; .upper() gives consistent label casing
    status_counts: Counter = Counter(order.status.upper() for order in orders)

    if not status_counts:
        print("No orders found.")
        return

    # Format as "STATUS: N" pairs joined by " | " — sorted for deterministic output
    parts: list[str] = [f"{s}: {c}" for s, c in sorted(status_counts.items())]
    print("Summary: " + "  |  ".join(parts))


def main() -> None:
    """Orchestrate broker init, safety check, order fetch, table, and summary."""
    # Step 1: Build broker from .env — from_env() reads keys; __init__ asserts paper=True.
    broker = AlpacaBroker.from_env()
    log.info("AlpacaBroker initialized (paper sandbox)")

    # Step 2: SAFETY CHECK — raises RuntimeError if the connected account is live.
    try:
        broker.verify_paper_account()       # calls get_account() and checks is_paper
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)                         # exit 1 = live account detected

    try:
        # Step 3: Fetch up to 50 recent orders across all statuses, newest first.
        # list_recent_orders() returns list[OrderResult] — no alpaca SDK types leak out.
        orders = broker.list_recent_orders(limit=50)
        log.info("Received %d order(s)", len(orders))

        if not orders:
            print("\nNo orders found in this account.")
            sys.exit(0)                     # exit 0 = success, nothing to show

        # Step 4: Render the fixed-width table.
        print_table(orders)

        # Step 5: Print the status-grouped summary line.
        print_summary(orders)

    except Exception as exc:
        # Catch-all for unexpected API or network errors
        print(f"Error fetching order history: {exc}")
        sys.exit(1)                         # exit 1 = unexpected error

    sys.exit(0)                             # exit 0 = success


# Only run main() when executed directly (not when imported as a module).
if __name__ == "__main__":
    main()
