# scripts/rebalance_smoke.py

"""
Manual live-paper smoke check for the rebalance loop.

This is NOT a unit test (those live in tests/ and stay hermetic). It is a
minimal end-to-end run that confirms the wired-together system actually works
against the REAL Alpaca PAPER API: build the broker from .env, confirm the
account is paper, read state, compute the orders, and (optionally) place them.

It adds NO production logic. It only loads credentials, prints loudly, and calls
the already-tested run_rebalance / reconcile_to_target. It can never trade live:
AlpacaBroker.from_env is paper-only behind three guards (assert paper is True in
__init__, paper=True hardcoded in from_env, and verify_paper_account raising on
any non-"PA" account).

DEFAULT is a DRY RUN: it computes and prints the orders but submits nothing.
Pass --submit to actually place them.

Run via:
    uv run python scripts/rebalance_smoke.py            # dry run, submits nothing
    uv run python scripts/rebalance_smoke.py --submit   # actually places orders
"""

# argparse is the standard-library CLI parser - same convention as the other
# scripts in scripts/ (rotation_verdict.py, overfitting_tax.py, ...).
import argparse

# sys is used only for a clean non-zero exit on error, mirroring hello_alpaca.py.
import sys

# AlpacaBroker is the concrete broker wrapping alpaca-py; from_env() reads .env
# and enforces paper=True.
from src.brokers.alpaca_broker import AlpacaBroker

# The pure reconciliation core - used directly in the dry-run branch so we can
# print the orders WITHOUT submitting them.
from src.execution.rebalance import reconcile_to_target

# The complete guarded runner - used in the submit branch to place orders.
from src.execution.runner import run_rebalance


# Hardcoded, deliberately tiny target for a first live-paper run: 1% of the
# portfolio in a single, liquid symbol. Small on purpose so a mistake is cheap.
TARGET_WEIGHTS = {"SPY": 0.01}  # 1% of portfolio in SPY, one symbol, tiny by design


def main() -> None:
    # ONE flag: --submit. Absent => dry run (compute + print, submit nothing).
    parser = argparse.ArgumentParser(
        description="Live-paper smoke check for the rebalance loop (dry run by default)."
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        default=False,
        help=(
            "Actually place orders. Without this flag the script is a dry run: "
            "it computes and prints the orders but submits nothing."
        ),
    )
    args = parser.parse_args()

    try:
        # Loud header stating the mode so there is no ambiguity about whether
        # this run will place orders.
        print("=" * 50)
        if args.submit:
            print("SUBMIT MODE - orders WILL be placed on the paper account")
        else:
            print("DRY RUN - no orders will be placed")
        print("=" * 50)

        # Build the broker from .env. from_env() calls load_dotenv() internally
        # and raises EnvironmentError with a clear message if a key is missing.
        broker = AlpacaBroker.from_env()
        print("Broker built from .env")

        # SAFETY GATE, called explicitly and up front: raises RuntimeError if the
        # connected account is not a paper account. If it raised, the script stops
        # here and no state is read or changed.
        broker.verify_paper_account()
        print("PAPER ACCOUNT CONFIRMED")

        # Read and print the account snapshot, one labeled line per field.
        account = broker.get_account()
        print("-" * 50)
        print(f"Account number:  {account.account_number}")
        print(f"Is paper:        {account.is_paper}")
        print(f"Portfolio value: ${account.portfolio_value:,.2f}")
        print(f"Cash:            ${account.cash:,.2f}")
        print(f"Buying power:    ${account.buying_power:,.2f}")

        # Read and print current holdings (or note that the account is flat).
        positions = broker.get_positions()
        print("-" * 50)
        if positions:
            print("Current positions:")
            for p in positions:
                print(f"  {p.symbol}: {p.qty} shares ({p.side})")
        else:
            print("Current positions: no open positions")

        # Print the target allocation we are reconciling toward.
        print("-" * 50)
        print(f"Target weights: {TARGET_WEIGHTS}")
        print("-" * 50)

        if not args.submit:
            # DRY RUN: reproduce the runner's READ path, but submit nothing.
            # Collapse positions to the symbol -> whole-shares dict the pure fn wants.
            current_positions = {p.symbol: p.qty for p in positions}

            # Fetch a price only for positively-weighted targets (same rule the
            # runner uses); print each fetched price on its own labeled line.
            prices = {}
            for sym, weight in TARGET_WEIGHTS.items():
                if weight > 0:
                    price = broker.get_latest_price(sym)
                    prices[sym] = price
                    print(f"Latest price {sym}: ${price:,.2f}")

            # Compute the orders with the SAME pure function the runner calls.
            order_requests = reconcile_to_target(
                TARGET_WEIGHTS,
                current_positions,
                prices,
                account.portfolio_value,
            )

            # Print the computed orders (or note nothing is needed).
            print("-" * 50)
            if order_requests:
                print("Computed orders (NOT submitted):")
                for req in order_requests:
                    print(
                        f"  {req.symbol}: {req.side.value} {req.qty} "
                        f"{req.order_type.value} {req.time_in_force.value}"
                    )
            else:
                print("no orders needed - already on target")

            print("-" * 50)
            print("DRY RUN complete - NOTHING was submitted. "
                  "Re-run with --submit to place these orders.")
        else:
            # SUBMIT: call the complete guarded runner. It re-runs the paper guard
            # first internally, then reads, prices, computes, and submits.
            results = run_rebalance(broker, TARGET_WEIGHTS)

            # Print the broker's confirmation for each submitted order.
            print("-" * 50)
            if results:
                print("Submitted orders:")
                for r in results:
                    print(
                        f"  {r.order_id}: {r.symbol} {r.side.value} {r.qty} -> {r.status}"
                    )
                print("-" * 50)
                print(f"SUBMITTED {len(results)} orders - check the Alpaca dashboard.")
            else:
                print("no orders needed - already on target")

    except Exception as e:
        # Print any error legibly (auth, network, live-account guard) rather than
        # dumping a raw traceback, mirroring hello_alpaca.py.
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
