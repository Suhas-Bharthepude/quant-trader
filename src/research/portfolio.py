# src/research/portfolio.py

"""
Pure return-combination core of the portfolio rotation backtester.

Returns + weights IN, one combined per-bar return stream OUT.  This is the pure
ARITHMETIC CORE of the portfolio backtester: it takes ONE rebalance period's
already-computed per-symbol per-bar return arrays plus the per-symbol weights and
folds them into a SINGLE portfolio per-bar return array for that period.

It is deliberately PURE and DUMB, mirroring the discipline of rank_by_trailing_return
and trailing_returns_at in cross_sectional.py: it does NOT compute returns from bars,
does NOT compute weights, does NOT rank, does NOT loop over rebalance periods, and does
NOT touch OHLCVBar, DuckDB, or the engine.  That separation is deliberate: keeping the
combination arithmetic isolated makes it unit-testable with plain dicts of numpy arrays,
and lets the FUTURE rebalance loop call it once per period.  The rebalance loop, weight
computation, and engine wiring are SEPARATE later commits.

The CASH REMAINDER is the one design choice that unifies three cases into one formula.
Weights need NOT sum to 1.0: the remainder (1.0 - sum(weights)) is the cash fraction,
which earns the caller-supplied per-bar cash rate on every bar.  That single rule handles
FULL-invested (weights sum to 1.0 -> no cash), ALL-cash (empty weights -> everything is
cash), and PARTIAL-cash (weights sum to < 1.0 -> the rest earns cash) with no branching.

It works in LOG-return space to stay consistent with the engine and metrics.py: the
engine treats per-bar log returns as additive everywhere (equity = exp(cumsum(returns))),
so a portfolio stream produced here can be fed straight into metrics.py / _stitch_oos the
same way a single-symbol stream is.

It lives in src/research/ alongside cross_sectional.py because, like the ranker and the
returns-computer, it is a pure rotation-research primitive built BEFORE the backtester
that will consume it.
"""

# numpy is the array backend shared with the engine and metrics.py.  Every per-bar
# series (per-symbol returns, the combined portfolio stream) is a float64 ndarray, so
# the combination math is vectorized in C and its dtype/units match the engine exactly.
import numpy as np


def combine_period_returns(
    per_symbol_returns: dict[str, np.ndarray],
    weights: dict[str, float],
    n_bars: int,
    cash_per_bar_return: float = 0.0,
) -> np.ndarray:
    """Combine one period's per-symbol return streams into one portfolio stream.

    Args:
        per_symbol_returns: held symbol -> that symbol's per-bar return array for THIS
                            period.  Every array MUST have length == n_bars.  May be
                            EMPTY, which is the all-cash period (nothing held).
        weights:            held symbol -> its portfolio weight (a fraction) for this
                            period.  Equal-weight callers pass 1/N for each of N held
                            symbols.  Weights need NOT sum to 1.0: the remainder
                            (1.0 - sum(weights)) is the CASH fraction (see module doc).
        n_bars:             the period length (number of bars).  REQUIRED because the
                            all-cash case (empty per_symbol_returns) cannot infer the
                            output length from an empty dict.  Must be >= 1.
        cash_per_bar_return: the per-bar return earned on the cash fraction.  Defaults
                            to 0.0, meaning "no cash yield".  The CALLER supplies the
                            engine's per_bar_cash_yield (log1p(annual_cash_yield) /
                            annualization_factor) here so cash is scored the SAME way
                            the single-symbol engine scores flat bars; this function
                            does NOT compute that conversion, it just applies whatever
                            per-bar cash rate it is handed.

    Returns:
        The combined portfolio per-bar return stream: a float64 ndarray of length
        n_bars, in LOG-return space (see the log-space note below).

    Raises:
        ValueError: n_bars < 1.
        ValueError: any per_symbol_returns array has length != n_bars.
        ValueError: the key sets of weights and per_symbol_returns are not EQUAL.
    """
    # STEP 1 — validate n_bars.  A zero- or negative-length period is meaningless and
    # would make the cash array below empty or ill-formed; echo the bad value in the
    # message, matching the validation style in walk_forward_splits / rank_by_trailing_return.
    if n_bars < 1:
        raise ValueError(f"n_bars must be >= 1, got {n_bars}")

    # STEP 2 — key-set-equality guard.  Every symbol in weights must have a matching
    # returns array and vice versa.  WHY this is a hard error, not a silent skip:
    # a weighted symbol with NO returns array would silently drop that symbol's
    # contribution and leave the portfolio MISWEIGHTED (its weight would vanish from
    # the sum while still counting against the cash remainder); a returns array with
    # NO weight would silently ignore a held symbol.  Both are caller bugs, so we
    # catch them loudly here rather than produce a quietly-wrong portfolio stream.
    if per_symbol_returns.keys() != weights.keys():
        # set(...) difference names exactly which symbols are on each side without a
        # match, so the caller sees the precise mismatch rather than a vague message.
        weighted_without_returns = set(weights.keys()) - set(per_symbol_returns.keys())
        returns_without_weight = set(per_symbol_returns.keys()) - set(weights.keys())
        raise ValueError(
            "per_symbol_returns and weights must have EQUAL key sets; "
            f"symbols with a weight but no returns array: {sorted(weighted_without_returns)}; "
            f"symbols with a returns array but no weight: {sorted(returns_without_weight)}"
        )

    # STEP 3 — validate every array length.  A per-symbol array whose length != n_bars
    # is a caller bug that would otherwise broadcast-fail or misalign in the vectorized
    # add below; catch it loudly and name the FIRST offending symbol and its length so
    # the caller can locate the bad array immediately.
    for symbol, arr in per_symbol_returns.items():
        # len(arr) is the array's bar count; it must equal n_bars exactly.
        if len(arr) != n_bars:
            raise ValueError(
                f"per_symbol_returns[{symbol!r}] has length {len(arr)}, "
                f"expected n_bars == {n_bars}"
            )

    # STEP 4 — cash weight is whatever fraction of capital is NOT allocated to symbols.
    # sum(weights.values()) is the invested fraction; 1.0 minus it is the cash fraction.
    # An empty weights dict sums to 0.0 -> cash_weight 1.0 (all-cash); weights summing
    # to 1.0 -> cash_weight 0.0 (fully invested); anything between -> partial cash.
    cash_weight = 1.0 - sum(weights.values())

    # STEP 5 — seed the portfolio stream with the cash contribution on EVERY bar.
    # np.full builds a length-n_bars float64 array whose every element is the cash
    # fraction times the per-bar cash rate.  When cash_weight is 0.0 (fully invested)
    # this is all zeros and contributes nothing; when cash_weight is 1.0 (all cash)
    # every bar is exactly cash_per_bar_return, which is the all-cash period.
    portfolio_returns = np.full(n_bars, cash_weight * cash_per_bar_return, dtype=np.float64)

    # STEP 6 — add each held symbol's weighted return stream.  This is a LINEAR
    # combination in LOG-return space.  NOTE: summing weighted per-bar LOG returns is
    # an APPROXIMATION of true portfolio arithmetic-return weighting (which would sum
    # weighted SIMPLE returns and then re-log), but it is the SAME convention the engine
    # uses everywhere — the engine works in log space and treats per-bar log returns as
    # additive — so staying in log space keeps this portfolio stream consistent with how
    # metrics.py and _stitch_oos consume single-symbol streams.  We deliberately do NOT
    # convert to simple returns: consistency with the engine's log-space convention is
    # the whole point.
    for symbol, arr in per_symbol_returns.items():
        # weights[symbol] is this symbol's fraction; multiplying its per-bar array and
        # adding it in accumulates the weighted contribution onto the cash-seeded base.
        # np.asarray(..., float64) makes the multiply produce float64 even if the caller
        # handed in an integer-typed array, preserving the output dtype contract.
        portfolio_returns += weights[symbol] * np.asarray(arr, dtype=np.float64)

    # STEP 7 — return the combined stream: float64, length n_bars, in log-return space.
    return portfolio_returns
