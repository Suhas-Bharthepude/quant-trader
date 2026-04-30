# cancel_order.py — cancels a single open Alpaca paper order by ID.
# Designed to pair with limit_order.py — run after placing a non-marketable limit order.
#
# Order ID resolution (in priority order):
#   1. CLI argument:  uv run python scripts/cancel_order.py <order-uuid>
#   2. File on disk:  logs/last_limit_order_id.txt  (written by limit_order.py)
#   3. Neither found: error and exit
#
# Safety: AlpacaBroker asserts paper=True; verify_paper_account() raises for live
# accounts; order details are shown before asking; terminal orders skip the prompt.
#
# Usage:
#   uv run python scripts/cancel_order.py                   # reads from file
#   uv run python scripts/cancel_order.py <order-uuid>     # explicit ID

import logging  # timestamps and severity to stdout
import sys  # sys.argv for CLI args; sys.exit() for exit codes
import time  # time.sleep() between poll attempts
from pathlib import Path  # reads the order-ID handoff file

from src.brokers.alpaca_broker import AlpacaBroker  # Alpaca-backed concrete broker

ORDER_ID_FILE: Path = Path("logs/last_limit_order_id.txt")  # written by limit_order.py
TERMINAL_STATES: frozenset[str] = frozenset({"filled", "canceled", "expired", "rejected"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # logger scoped to this module


def resolve_order_id() -> str:
    """CLI arg > file > error. Returns the UUID string to target."""
    if len(sys.argv) > 1:                               # explicit UUID on the command line
        order_id = sys.argv[1].strip()                  # strip accidental whitespace
        log.info("Using order ID from CLI argument: %s", order_id)
        return order_id
    if ORDER_ID_FILE.exists():                          # UUID file written by limit_order.py
        order_id = ORDER_ID_FILE.read_text().strip()    # strip trailing newline
        if order_id:                                    # guard against an empty file
            log.info("Using order ID from %s: %s", ORDER_ID_FILE, order_id)
            return order_id
        print(f"Error: {ORDER_ID_FILE} exists but is empty.")
        sys.exit(1)                                     # exit 1 = unusable file
    # Neither source found — tell the user exactly how to fix it
    print(
        "Error: no order ID found.\n"
        "  Option 1: uv run python scripts/cancel_order.py <order-uuid>\n"
        f"  Option 2: run limit_order.py first — it writes the ID to {ORDER_ID_FILE}."
    )
    sys.exit(1)                                         # exit 1 = cannot determine target


def fetch_and_print_order(broker: AlpacaBroker, order_id: str):
    """Fetch the order state and print details. Returns OrderResult; exits 1 on API error."""
    try:
        order = broker.get_order(order_id)              # GET /v2/orders/{id} → OrderResult
    except Exception as exc:
        # Most likely causes: typo in UUID, or order belongs to a different account
        print(f"Error fetching order {order_id}: {exc}")
        sys.exit(1)                                     # exit 1 = API error
    # limit_price is None for market orders; already a float in OrderResult, no cast needed
    limit_str = f"${order.limit_price:,.2f}" if order.limit_price else "N/A (market order)"
    submitted_at = order.submitted_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    print("\n" + "=" * 55)
    print("ORDER DETAILS")
    print("=" * 55)
    print(f"  Order ID:     {order.order_id}")
    print(f"  Symbol:       {order.symbol}")
    print(f"  Side:         {order.side}")              # OrderSide str-enum → "buy" / "sell"
    print(f"  Quantity:     {order.qty}")
    print(f"  Order type:   {order.order_type}")        # OrderType str-enum → "market" / "limit"
    print(f"  Status:       {order.status}")            # plain string, e.g. "new" / "accepted"
    print(f"  Limit price:  {limit_str}")
    print(f"  Submitted at: {submitted_at}")
    print("=" * 55)
    return order                                        # caller inspects status before confirming


def confirm_cancellation(order) -> None:
    """Exit 0 if already terminal; exit 2 if user aborts; otherwise return to proceed."""
    if order.status in TERMINAL_STATES:
        # Order is already done — cancellation is neither possible nor needed
        print(f"\nOrder is already in state '{order.status}' — no cancellation needed.")
        sys.exit(0)                                     # exit 0 = nothing to do, not an error
    answer = input('\nType "yes" to cancel this order, or anything else to abort: ').strip()
    if answer.lower() != "yes":
        print("Aborted. No changes were made.")
        sys.exit(2)                                     # exit 2 = user chose not to proceed
    log.info("User confirmed cancellation")


def cancel_and_wait(broker: AlpacaBroker, order_id: str) -> int:
    """Send cancel request and poll up to 10 s for a terminal state. Returns exit code."""
    try:
        broker.cancel_order(order_id)                   # DELETE /v2/orders/{id}
        log.info("Cancel request sent for order %s", order_id)
    except Exception as exc:
        # Possible causes: order already gone, network error, permissions issue
        print(f"Error sending cancel request: {exc}")
        return 1                                        # exit 1 = API error
    POLL_INTERVAL: int = 1                              # seconds between status checks
    MAX_POLLS: int = 10 // POLL_INTERVAL                # 10 iterations = 10-second ceiling
    previous: str = ""                                  # tracks last status for transition logs
    for _ in range(MAX_POLLS):
        try:
            order = broker.get_order(order_id)          # GET /v2/orders/{id} → OrderResult
        except Exception as exc:
            # Some brokers drop the record after cancellation — log and retry
            log.warning("Could not fetch order status: %s", exc)
            time.sleep(POLL_INTERVAL)
            continue
        if order.status != previous:                    # log only on status change, not every poll
            log.info("Status: %s -> %s", previous or "cancelling", order.status)
            previous = order.status
        if order.status in TERMINAL_STATES:
            if order.status == "canceled":              # the expected outcome
                print(f"\nOrder {order_id} successfully cancelled.")
            else:                                       # e.g. filled between cancel and confirm
                print(f"\nOrder reached terminal state: {order.status}. No further action needed.")
            return 0                                    # exit 0 = terminal state reached
        time.sleep(POLL_INTERVAL)
    print(
        f"\nOrder {order_id} is still not cancelled after 10 seconds.\n"
        "This is unexpected. Check the Alpaca dashboard manually:\n"
        "https://app.alpaca.markets/paper-trading/orders"
    )
    return 3                                            # exit 3 = timeout, state unknown


def main() -> None:
    """Orchestrate broker init, safety check, order lookup, confirmation, and cancellation."""
    broker = AlpacaBroker.from_env()                    # reads .env; __init__ asserts paper=True
    log.info("AlpacaBroker initialized (paper sandbox)")
    try:
        broker.verify_paper_account()                   # calls get_account() and checks is_paper
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)                                     # exit 1 = live account detected
    order_id: str = resolve_order_id()                  # CLI arg > handoff file > error
    order = fetch_and_print_order(broker, order_id)     # fetch + display current state
    confirm_cancellation(order)                         # exit 0/2 if terminal/aborted
    sys.exit(cancel_and_wait(broker, order_id))         # cancel + poll for terminal state


# Only run main() when executed directly (not when imported as a module).
if __name__ == "__main__":
    main()
