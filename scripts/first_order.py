# first_order.py — paper-trading smoke test: 1 share of SPY via AlpacaBroker.
# Every line is commented for learning. Do not change SYMBOL, QTY, or paper=True.
#
# Safety: AlpacaBroker asserts paper=True; verify_paper_account() raises for live
# accounts; PA-prefix check below is defense-in-depth; two "yes" prompts required.
#
# Usage: uv run python scripts/first_order.py

import logging  # timestamps and severity levels to stdout
import sys  # sys.exit() with specific exit codes
import time  # time.sleep() between poll attempts

from src.brokers.alpaca_broker import AlpacaBroker  # Alpaca-backed concrete broker
from src.brokers.base import OrderRequest, OrderSide, OrderType, TimeInForce  # our types

# ---------------------------------------------------------------------------
# CONSTANTS — do not change; this script is a controlled smoke test.
# ---------------------------------------------------------------------------
SYMBOL: str = "SPY"   # fixed ticker for this smoke test
QTY: int = 1           # fixed quantity — keep at 1

# Frozenset for O(1) membership checks and immutability.
TERMINAL_STATES: frozenset[str] = frozenset({"filled", "canceled", "expired", "rejected"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # logger scoped to this module


def check_market_status(broker: AlpacaBroker) -> None:
    """Warn and confirm if market is closed; exit code 2 if user declines."""
    if broker.is_market_open():             # GET /v2/clock via the broker
        log.info("Market is OPEN")
        return
    # Market is closed — Alpaca will queue the order and submit it at next open.
    print(
        "\nWarning: the market is currently CLOSED.\n"
        "If you proceed, Alpaca will queue the order and submit it at open.\n"
    )
    answer = input("Submit anyway? (yes to continue, anything else to cancel): ")
    if answer.strip().lower() != "yes":
        print("Cancelled.")
        sys.exit(2)                         # exit code 2 = user cancelled
    log.info("User confirmed submission despite market being closed")


def build_dry_run_preview(broker: AlpacaBroker) -> None:
    """Fetch account + latest price, run PA-prefix safety check, print order preview."""
    account = broker.get_account()          # GET /v2/account → AccountSnapshot

    # SAFETY CHECK (defense-in-depth): verify_paper_account() already checked is_paper,
    # but Alpaca paper accounts also carry a "PA" prefix on account_number — double guard.
    if not account.account_number.startswith("PA"):
        print(
            f"Error: '{account.account_number}' does not look like a paper account.\n"
            "Expected a 'PA' prefix. Aborting to protect real funds."
        )
        sys.exit(1)                         # exit code 1 = safety check failed

    price: float = broker.get_latest_price(SYMBOL)  # GET /v2/stocks/{symbol}/trades/latest

    print("\n" + "=" * 50)
    print("DRY-RUN PREVIEW — no order has been placed yet")
    print("=" * 50)
    print(f"  Account number:   {account.account_number}")
    print(f"  Buying power:     ${account.buying_power:,.2f}")
    print(f"  Symbol:           {SYMBOL}")
    print(f"  Quantity:         {QTY} share(s)")
    print("  Side:             BUY")
    print("  Order type:       MARKET")
    print("  Time in force:    DAY")
    print(f"  Latest price:     ${price:,.2f}  (last trade — actual fill may differ)")
    print(f"  Estimated cost:   ${price * QTY:,.2f}")
    print("=" * 50)


def poll_order_status(broker: AlpacaBroker, order_id: str) -> int:
    """Poll every 2 s for up to 60 s. Returns 0 filled, 1 rejected/cancelled, 3 timeout."""
    POLL_INTERVAL: int = 2                  # seconds between GET /v2/orders/{id} calls
    MAX_POLLS: int = 60 // POLL_INTERVAL    # 30 iterations = 60-second ceiling
    previous: str = ""                      # tracks last known status to avoid duplicate logs

    for _ in range(MAX_POLLS):
        order = broker.get_order(order_id)  # GET /v2/orders/{id} → OrderResult
        if order.status != previous:
            log.info("Order status: %s -> %s", previous or "submitted", order.status)
            previous = order.status

        if order.status == "filled":
            # OrderResult has no filled_at field yet; submitted_at is our best timestamp.
            print("\n" + "=" * 50)
            print("ORDER FILLED")
            print("=" * 50)
            print(f"  Fill price:  ${order.filled_avg_price:,.4f}")
            print(f"  Filled qty:  {order.filled_qty} share(s)")
            print(f"  Filled at:   {order.submitted_at.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print("=" * 50)
            return 0                        # exit code 0 = success

        if order.status in TERMINAL_STATES:  # canceled / expired / rejected
            print(f"\nOrder {order.status}: check Alpaca dashboard for details")
            return 1                        # exit code 1 = error

        time.sleep(POLL_INTERVAL)

    print(
        f"\nOrder is still pending after 60 seconds.\n"
        f"Order ID: {order_id}\n"
        "Check status at: https://app.alpaca.markets/paper-trading/orders"
    )
    return 3                                # exit code 3 = timeout


def main() -> None:
    """Orchestrate broker init, safety checks, dry-run preview, submission, and polling."""
    # Step 1: Build broker from .env — from_env() reads keys; __init__ asserts paper=True.
    broker = AlpacaBroker.from_env()
    log.info("AlpacaBroker initialized (paper sandbox)")

    # Step 2: SAFETY CHECK 1 — raises RuntimeError if the connected account is live.
    try:
        broker.verify_paper_account()       # calls get_account() and checks is_paper
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)                         # exit code 1 = live account detected

    # Step 3: Market-hours check — warns + asks for confirmation if market is closed.
    check_market_status(broker)

    # Step 4: Account + PA-prefix check + latest price + human-readable order preview.
    build_dry_run_preview(broker)

    # Step 5: Second "yes" confirmation — any other input cancels without placing an order.
    confirmation = input('\nType "yes" to submit this order, or anything else to cancel: ')
    if confirmation.strip().lower() != "yes":
        print("Order cancelled. No order was placed.")
        sys.exit(2)                         # exit code 2 = user cancelled at prompt

    # Step 6: Build broker-agnostic OrderRequest — no alpaca SDK types past this line.
    request = OrderRequest(
        symbol=SYMBOL,
        qty=QTY,
        side=OrderSide.BUY,                 # entering a long position
        order_type=OrderType.MARKET,        # fill at best available price
        time_in_force=TimeInForce.DAY,      # cancel at EOD if unfilled
    )
    log.info("Submitting %s x%d BUY MARKET DAY order...", SYMBOL, QTY)

    # POST /v2/orders — the one line that actually sends the paper order.
    order = broker.submit_order(request)
    log.info("Order submitted — ID: %s  status: %s", order.order_id, order.status)

    submitted_at: str = order.submitted_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\nOrder placed — ID: {order.order_id}")
    print(f"Status: {order.status}  |  Submitted at: {submitted_at}")

    # Step 7: Poll until filled, rejected/cancelled, or 60-second timeout.
    sys.exit(poll_order_status(broker, order.order_id))


# Only run main() when this script is executed directly (not when imported).
if __name__ == "__main__":
    main()
