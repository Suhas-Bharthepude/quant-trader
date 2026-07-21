# src/execution/runner.py

"""
Thin rebalance runner: the I/O shell around the pure reconciliation core.

Division of responsibility:
- src/execution/rebalance.py does the MATH (pure, no I/O): given current
  holdings, targets, prices, and account value, it returns OrderRequests.
- this module does the I/O: it reads the account and positions from the broker,
  hands the plain data to the pure function, and submits the resulting orders.

Safety: verify_paper_account() is the FIRST executable statement in
run_rebalance. Because that guard raises RuntimeError on a live account, an
order can never be reached on a live account -- the structure of the function,
not a runtime check buried later, is what makes this safe.

Prices are passed in rather than fetched. Wiring broker.get_latest_price into
the runner is a deliberately deferred increment, which keeps this first version
fully hermetic (no data-client dependency) and its test offline.
"""

# Modern type-hint syntax on the signature without quoting.
from __future__ import annotations

# The broker interface (type of the `broker` argument) and the result dataclass
# (what submit_order returns) come from the contract; the runner programs only
# to the interface, never to a concrete broker.
from src.brokers.base import Broker, OrderResult

# The pure sizing core this runner wraps.
from src.execution.rebalance import reconcile_to_target


def run_rebalance(
    broker: Broker,                    # any Broker implementation (a paper AlpacaBroker in practice)
    target_weights: dict[str, float],  # symbol -> desired weight fraction (same shape the pure fn takes)
    prices: dict[str, float],          # symbol -> price, passed in (live fetch is a later increment)
) -> list[OrderResult]:
    """
    Reconcile the account to the target allocation and submit the orders.

    Steps, in order:
    1. Verify the account is paper (raises on a live account) -- FIRST, always.
    2. Read the account for its portfolio_value.
    3. Read current positions and reduce them to a symbol -> shares dict.
    4. Compute the orders with the pure reconciliation function.
    5. Submit each order and collect the broker's results.

    Parameters
    ----------
    broker : Broker
        The brokerage connection. Must be a paper account or step 1 raises.
    target_weights : dict[str, float]
        Desired weight per symbol (see reconcile_to_target).
    prices : dict[str, float]
        Latest price per symbol, supplied by the caller.

    Returns
    -------
    list[OrderResult]
        One result per submitted order; empty if the account was already on
        target.

    Raises
    ------
    RuntimeError
        If the connected account is a live (non-paper) account.
    """
    # 1. LOAD-BEARING SAFETY GATE. This MUST stay the first statement in the
    #    function: it raises RuntimeError on a live account, so no account read
    #    and no order submission below can ever be reached against live money.
    broker.verify_paper_account()

    # 2. Fetch the account to get portfolio_value, the budget for the weights.
    account = broker.get_account()

    # 3. Read open positions and collapse them to the symbol -> whole-shares dict
    #    the pure function expects. (qty is a positive share count; v1 assumes
    #    long/flat, so PositionSnapshot.side is not consulted here.)
    positions = broker.get_positions()
    current_positions = {p.symbol: p.qty for p in positions}

    # 4. PURE step: compute the orders needed to reach the target. No I/O here.
    order_requests = reconcile_to_target(
        target_weights,
        current_positions,
        prices,
        account.portfolio_value,
    )

    # 5. Submit each order in turn, collecting the broker's confirmation for each.
    results = [broker.submit_order(request) for request in order_requests]

    # Return the per-order results (empty list if nothing needed changing).
    return results
