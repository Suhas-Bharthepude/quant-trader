# limit_order.py — GTC limit BUY for 1 share of SPY at 20% below market.
# The order is intentionally non-marketable — it will NOT fill under normal conditions.
# Purpose: exercise the limit order code path and the cancel flow safely on paper.
#
# After submission the order ID is saved to logs/last_limit_order_id.txt so that
# scripts/cancel_order.py can read it without manual copy-pasting.
#
# Safety: AlpacaBroker asserts paper=True; verify_paper_account() raises for live
# accounts; PA-prefix check is defense-in-depth; paranoia assert on limit price.
#
# Usage: uv run python scripts/limit_order.py

import logging  # timestamps and severity levels to stdout
import sys  # sys.exit() with specific exit codes
from pathlib import Path  # clean cross-platform file writes

from src.brokers.alpaca_broker import AlpacaBroker  # Alpaca-backed concrete broker
from src.brokers.base import OrderRequest, OrderSide, OrderType, TimeInForce  # our types

# ---------------------------------------------------------------------------
# CONSTANTS — do not change; this script is a controlled exercise.
# ---------------------------------------------------------------------------
SYMBOL: str = "SPY"          # fixed ticker for this exercise
QTY: int = 1                  # fixed quantity — keep at 1
LIMIT_DISCOUNT: float = 0.20  # set limit 20% below market so it cannot accidentally fill

# Plain-text handoff file — cancel_order.py reads the UUID written here.
ORDER_ID_FILE: Path = Path("logs/last_limit_order_id.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # logger scoped to this module


def compute_limit_price(current_price: float) -> float:
    """Compute a limit price LIMIT_DISCOUNT below market and assert it is safely non-marketable."""
    raw_limit: float = current_price * (1 - LIMIT_DISCOUNT)   # e.g. 0.80 × market
    limit_price: float = round(raw_limit, 2)                   # Alpaca requires cents precision

    # PARANOIA ASSERT: if limit_price is ≥ 85% of market, something went wrong in the
    # math (e.g. LIMIT_DISCOUNT is too small). Crash rather than risk an accidental fill.
    assert limit_price < current_price * 0.85, (
        f"Computed limit ${limit_price:.2f} is too close to market ${current_price:.2f}. "
        "The order might fill. Aborting."
    )

    log.info("Limit price: $%.2f  (%.0f%% below market)", limit_price, LIMIT_DISCOUNT * 100)
    return limit_price


def build_dry_run_preview(account_number: str, current_price: float, limit_price: float) -> None:
    """Print the order summary before asking for confirmation. No order is placed here."""
    print("\n" + "=" * 55)
    print("DRY-RUN PREVIEW — no order has been placed yet")
    print("=" * 55)
    print(f"  Account number:    {account_number}")
    print(f"  Symbol:            {SYMBOL}")
    print(f"  Quantity:          {QTY} share(s)")
    print("  Side:              BUY")
    print("  Order type:        LIMIT")
    print("  Time in force:     GTC  (Good-Till-Cancelled)")
    print(f"  Current price:     ${current_price:,.2f}  (latest trade)")
    print(f"  Limit price:       ${limit_price:,.2f}  ({LIMIT_DISCOUNT:.0%} below market)")
    print()
    print("  NOTE: This order is designed NOT to fill.")
    print("  It will sit open in your paper account until you cancel it.")
    print("=" * 55)


def save_order_id(order_id: str) -> None:
    """Write the order UUID to ORDER_ID_FILE so cancel_order.py can read it automatically."""
    ORDER_ID_FILE.parent.mkdir(parents=True, exist_ok=True)  # guard if logs/ was deleted
    ORDER_ID_FILE.write_text(order_id + "\n")                # one UUID per line, overwrites
    log.info("Order ID saved to %s", ORDER_ID_FILE)


def main() -> None:
    """Orchestrate broker init, safety checks, price fetch, preview, submission, and receipt."""
    # Step 1: Build broker from .env — from_env() reads keys; __init__ asserts paper=True.
    broker = AlpacaBroker.from_env()
    log.info("AlpacaBroker initialized (paper sandbox)")

    # Step 2: SAFETY CHECK 1 — raises RuntimeError if the connected account is live.
    try:
        broker.verify_paper_account()       # calls get_account() and checks is_paper
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)                         # exit code 1 = live account detected

    # Step 3: Fetch account snapshot for the PA-prefix check and preview.
    account = broker.get_account()          # GET /v2/account → AccountSnapshot

    # SAFETY CHECK (defense-in-depth): paper accounts carry a "PA" prefix on account_number.
    if not account.account_number.startswith("PA"):
        print(
            f"Error: '{account.account_number}' does not look like a paper account.\n"
            "Expected a 'PA' prefix. Aborting to protect real funds."
        )
        sys.exit(1)                         # exit code 1 = safety check failed

    # Step 4: Fetch the latest SPY trade price via the broker interface.
    current_price: float = broker.get_latest_price(SYMBOL)  # GET /v2/stocks/{symbol}/trades/latest
    log.info("Current %s price: $%.2f", SYMBOL, current_price)

    # Step 5: Compute limit price and assert it is safely non-marketable.
    limit_price: float = compute_limit_price(current_price)

    # Step 6: Print the dry-run preview so the user can verify before confirming.
    build_dry_run_preview(account.account_number, current_price, limit_price)

    # Step 7: Require explicit "yes" — any other input cancels without placing an order.
    confirmation = input('\nType "yes" to submit this order, or anything else to cancel: ')
    if confirmation.strip().lower() != "yes":
        print("Order cancelled. No order was placed.")
        sys.exit(2)                         # exit code 2 = user cancelled

    # Step 8: Build broker-agnostic OrderRequest — no alpaca SDK types past this line.
    request = OrderRequest(
        symbol=SYMBOL,
        qty=QTY,
        side=OrderSide.BUY,                 # entering a long position
        order_type=OrderType.LIMIT,         # fill only at limit_price or better
        time_in_force=TimeInForce.GTC,      # stays open across sessions until cancelled
        limit_price=limit_price,            # price ceiling we are willing to pay
    )
    log.info("Submitting %s x%d BUY LIMIT GTC @ $%.2f...", SYMBOL, QTY, limit_price)

    # POST /v2/orders — the one line that places the actual paper order.
    order = broker.submit_order(request)    # returns OrderResult with order_id and status
    log.info("Order submitted — ID: %s  status: %s", order.order_id, order.status)

    # Step 9: Print the order receipt immediately after submission.
    submitted_at: str = order.submitted_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    print("\n" + "=" * 55)
    print("ORDER SUBMITTED")
    print("=" * 55)
    print(f"  Order ID:     {order.order_id}")
    print(f"  Status:       {order.status}")
    print(f"  Submitted at: {submitted_at}")
    print(f"  Limit price:  ${order.limit_price:,.2f}")    # already float in OrderResult
    print("\n  Run scripts/cancel_order.py to cancel this order.")
    print("=" * 55)

    # Step 10: Persist the order ID to disk for the cancel script handoff.
    save_order_id(order.order_id)

    sys.exit(0)                             # exit code 0 = success


# Only run main() when executed directly (not when imported as a module).
if __name__ == "__main__":
    main()
