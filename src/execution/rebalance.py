# src/execution/rebalance.py

"""
Pure rebalance reconciliation: turn a target allocation into concrete orders.

This is the PURE core of the execution arc. It performs zero I/O: no broker
calls, no network, no global state. It takes plain data (dicts and a float) and
returns a list of OrderRequest objects. That makes the sizing arithmetic fully
testable with plain dicts, mirroring the pure-vs-IO discipline used by
combine_period_returns in src/research/portfolio.py (math in a pure function,
side effects pushed to a thin caller).

Design decisions locked for this first version (v1):
- Target representation is a dict[str, float] of symbol -> weight fraction, the
  same "weight per symbol" concept the rotation backtester already produces.
  Weights need NOT sum to 1.0; any remainder is implicitly cash and places no
  order (we never emit a "cash" order).
- Long/flat only. A negative target weight is rejected (no shorts yet).
- Sizing floors to whole shares via int(), so a target never over-buys past its
  dollar budget.
- Orders are market + DAY: the simplest execution style, no limit price.
- A symbol with weight 0 (or absent from targets) but currently held is closed
  to zero. A symbol already at its target share count produces no order.
- Orders are emitted sells-first: all SELLs precede all BUYs in the returned
  list, with deterministic alphabetical order preserved within each group. This
  is an ordering choice at the emission level so exit orders come before entry
  orders on a fully-invested rebalance; it does NOT guarantee the broker settles
  the sell cash before the buy executes (fill/settlement sequencing is a
  separate, future runner-level concern).
"""

# Allow modern type-hint syntax (list[OrderRequest], dict[str, float]) on the
# function signature without quoting. Consistent with src/brokers/base.py.
from __future__ import annotations

# The order vocabulary and the request dataclass come from the broker contract.
# We import ONLY these value types -- never a concrete broker or the alpaca SDK --
# which is what keeps this module pure and import-light.
from src.brokers.base import OrderRequest, OrderSide, OrderType, TimeInForce


def reconcile_to_target(
    target_weights: dict[str, float],   # symbol -> desired weight fraction; may sum to <= 1.0 (remainder is cash)
    current_positions: dict[str, int],  # symbol -> current whole shares held (a symbol absent here means 0 shares)
    prices: dict[str, float],           # symbol -> latest price per share, used to convert dollars to shares
    portfolio_value: float,             # total account value in USD, the budget the weights are applied to
) -> list[OrderRequest]:
    """
    Compute the orders needed to move the account from its current holdings to
    the target allocation.

    Parameters
    ----------
    target_weights : dict[str, float]
        Desired weight per symbol. Weights are fractions of portfolio_value.
        A symbol not present is treated as target weight 0.0 (fully exit).
    current_positions : dict[str, int]
        Whole shares currently held per symbol. A symbol not present is 0.
    prices : dict[str, float]
        Latest price per symbol. Required (and must be > 0) for any symbol with
        a positive target weight; not needed to sell a position down to zero.
    portfolio_value : float
        Total account value used to size target dollar amounts.

    Returns
    -------
    list[OrderRequest]
        One market+DAY order per symbol whose share delta is non-zero, ordered
        deterministically by symbol. Empty if everything is already on target.

    Raises
    ------
    ValueError
        If portfolio_value is negative, any target weight is negative (no shorts
        in v1), the target weights over-allocate (sum > 1.0), or a positively
        weighted symbol is missing a positive price.
    """
    # --- Guard 1: a negative portfolio value is nonsensical for sizing. --------
    # Echo the offending value so a failing caller sees what it passed.
    if portfolio_value < 0:
        raise ValueError(f"portfolio_value must be non-negative, got {portfolio_value}")

    # --- Guard 2: long/flat only. Reject any negative target weight. ----------
    # Name the offending symbol so the error is actionable.
    for sym, weight in target_weights.items():
        if weight < 0:
            raise ValueError(
                f"negative target weight for {sym!r} ({weight}); this version is long/flat only, no shorts"
            )

    # --- Guard 3: reject over-allocation (weights summing above 100%). --------
    # A tiny epsilon tolerance lets a legitimate fully-invested target of exactly
    # 1.0 pass despite floating-point noise, while still catching real overshoot.
    total_weight = sum(target_weights.values())
    if total_weight > 1.0 + 1e-9:
        raise ValueError(f"target weights over-allocate the portfolio: sum is {total_weight} (must be <= 1.0)")

    # Accumulate the orders we decide to place.
    orders: list[OrderRequest] = []

    # Iterate the UNION of targeted and held symbols, sorted for a deterministic
    # output order (important so tests can assert positions in a stable sequence).
    for sym in sorted(set(target_weights) | set(current_positions)):
        # Desired weight for this symbol; absent -> 0.0 means "hold nothing".
        target_weight = target_weights.get(sym, 0.0)
        # Shares we hold today; absent -> 0.
        current_shares = current_positions.get(sym, 0)

        if target_weight > 0.0:
            # We intend to hold this symbol, so we must be able to price it.
            price = prices.get(sym)
            if price is None or price <= 0:
                raise ValueError(
                    f"missing or non-positive price for {sym!r} (needed to size a positive target weight)"
                )
            # Dollar budget for this symbol, then floor to whole shares via int().
            # int() truncates toward zero; with positive inputs that is a floor,
            # so we never over-buy past the dollar budget.
            target_dollars = target_weight * portfolio_value
            target_shares = int(target_dollars / price)
        else:
            # Weight 0.0 (or absent): exit entirely. Selling to zero needs no price.
            target_shares = 0

        # Difference between where we want to be and where we are.
        delta = target_shares - current_shares

        # Already on target for this symbol: emit no order (delta filtering).
        if delta == 0:
            continue

        # Positive delta means buy the shortfall; negative means sell the excess.
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        # Order quantity is always a positive share count; direction is in `side`.
        qty = abs(delta)

        # Build a market + DAY order: simplest execution, no limit price in v1.
        orders.append(
            OrderRequest(
                symbol=sym,
                qty=qty,
                side=side,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
            )
        )

    # Partition the alphabetical order list so all SELLs precede all BUYs.
    # WHY: on a fully-invested rebalance, a BUY emitted before the SELL that
    # funds it can be rejected by a live broker for insufficient buying power.
    # Emitting exits first frees the cash the entries need. The list
    # comprehensions are STABLE (they preserve the source order), and the source
    # list is already alphabetical by symbol, so sells stay A->Z and buys stay
    # A->Z within each group. The SET of orders is identical to before - same
    # symbols, sides, quantities; only their SEQUENCE changes.
    sells = [order for order in orders if order.side == OrderSide.SELL]
    buys = [order for order in orders if order.side == OrderSide.BUY]

    # sells + buys: possibly empty, if every symbol was already at its target.
    return sells + buys
