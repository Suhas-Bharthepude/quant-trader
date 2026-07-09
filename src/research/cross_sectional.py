# src/research/cross_sectional.py

"""
Cross-sectional ranking primitive for portfolio rotation.

This is the cross-sectional counterpart to TimeSeriesMomentumStrategy.  Where
TSMOM judges each symbol ONLY against its OWN past (absolute momentum, one symbol
at a time), this function ranks symbols AGAINST EACH OTHER at a single rebalance
point and selects the strongest handful — the defining move of a rotation
strategy (hold the top-N performers, rotate as the ranking shifts).

It is intentionally PURE and DUMB: trailing returns go IN (already computed by
the caller), a selected symbol list comes OUT.  It does NOT compute trailing
returns, does NOT touch OHLCVBar, does NOT touch dates or calendars, and does NOT
touch DuckDB or the engine.  That separation is deliberate: the return-computation
piece (which must match TSMOM's trailing-return arithmetic) is a SEPARATE later
module, and keeping ranking isolated makes it unit-testable with plain dicts and
lets a future portfolio-rotation backtester consume the returned symbol list
without this function knowing anything about bars, dates, or scoring.

It lives in src/research/ next to walk_forward_splits because, like that helper,
it is a pure research primitive built BEFORE the engine that will consume it.  It
deliberately does NOT subclass Strategy: the Strategy contract is single-symbol
(list[OHLCVBar] -> np.ndarray), whereas ranking is inherently multi-symbol
(dict[str, float] -> list[str]), so it cannot honestly fulfil that contract.
"""

# math.isnan is used to detect NaN trailing-return values so they can be treated
# as ineligible; a plain `x != x` would also work but isnan reads clearly.
import math


def rank_by_trailing_return(
    trailing_returns: dict[str, float],
    top_n: int,
    hold_when_all_negative: bool = False,
) -> list[str]:
    """Rank symbols by trailing return (desc) and return the selected top-N tickers.

    Pure ranking/selection: the caller supplies each ELIGIBLE symbol's already-
    computed trailing return; a symbol that is ineligible on this rebalance date is
    simply absent from the dict, so the ranker ranks exactly what it is given.

    Args:
        trailing_returns:       symbol -> its trailing return (already computed).
        top_n:                  how many top-ranked symbols to select (>= 1).
        hold_when_all_negative: absolute-filter switch (see below).

    Returns:
        The selected symbols, highest trailing return first, alphabetical among
        ties.  Empty list when nothing qualifies (cash / hold nothing).

    Filter semantics:
        hold_when_all_negative=False (DEFAULT): apply an absolute filter FIRST —
            drop every symbol whose trailing return is <= 0.0 before ranking, so an
            all-<=0 basket returns an EMPTY list (go to cash).  This mirrors
            TimeSeriesMomentumStrategy's own go-flat-on-<=0 discipline.
        hold_when_all_negative=True: no absolute filter — rank ALL provided symbols
            regardless of sign and return the top_n "least bad" (pure relative
            strength, always invested).

    Raises:
        ValueError: if top_n < 1.
    """
    # top_n < 1 is meaningless — you cannot select fewer than one symbol and still
    # call it a selection.  Reject it with an echoing message, matching the
    # validation style in walk_forward_splits and TimeSeriesMomentumStrategy.
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    # STEP 1 — drop NaN-valued entries in BOTH modes.  A NaN trailing return means
    # "no valid return", which is really an eligibility failure the caller should
    # have excluded; we defend against it here so a NaN can never sort as if it
    # were a real value under hold_when_all_negative=True (NaN comparisons are
    # False, which would otherwise leave its ordering undefined).  math.isnan(v)
    # is True exactly for the NaN values, so `not isnan` keeps only real returns.
    eligible = {
        symbol: ret
        for symbol, ret in trailing_returns.items()
        if not math.isnan(ret)
    }

    # STEP 2 — apply the absolute filter unless the caller opted out.  Under the
    # default (hold_when_all_negative=False) we keep only strictly-positive returns:
    # the SAME strict `> 0.0` threshold TSMOM uses at its monthly_signal step in
    # time_series_momentum.py (a return of exactly 0.0 is NOT positive, so it is
    # filtered out).  Sharing the threshold means both strategies apply the
    # identical absolute-momentum risk filter.  When hold_when_all_negative=True we
    # skip this and rank everything regardless of sign (relative strength).
    if not hold_when_all_negative:
        eligible = {symbol: ret for symbol, ret in eligible.items() if ret > 0.0}

    # STEP 3 — deterministic sort.  key=(-ret, symbol) sorts by trailing return
    # DESCENDING (negating flips Python's ascending sort into descending) and, among
    # EQUAL returns, by SYMBOL NAME ASCENDING (the tuple's second element breaks the
    # tie).  A single sorted() over a (-return, symbol) key is what guarantees the
    # deterministic tie-break — no reliance on dict insertion order.
    ranked = sorted(eligible.items(), key=lambda item: (-item[1], item[0]))

    # STEP 4 — take the first min(top_n, len(ranked)) symbols.  A slice past the end
    # simply yields the whole list, so top_n larger than the available count returns
    # ALL available symbols (no error, no padding).  An empty `ranked` (empty input,
    # or everything filtered out as <=0 / NaN) yields an empty list — cash.
    return [symbol for symbol, _ret in ranked[:top_n]]
