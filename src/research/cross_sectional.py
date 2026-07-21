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

# datetime types the rebalance_ts parameter of trailing_returns_at — the "as of"
# point at which each symbol's trailing return is measured.
from datetime import datetime

# OHLCVBar is the bar schema trailing_returns_at consumes (one list per symbol).
# The ranker above touches no bars; only the returns-computer below needs it.
from src.data.schema import OHLCVBar

# trailing_return_series is the SHARED trailing-return primitive (extracted from
# TimeSeriesMomentumStrategy).  Calling it here is the whole point: cross-sectional
# and time-series momentum compute trailing return from ONE definition and cannot
# drift.  This import is the same research->strategies direction optuna_fit.py
# already uses, so there is no circular-import risk.
from src.strategies.time_series_momentum import trailing_return_series


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


def trailing_returns_at(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    rebalance_ts: datetime,
    lookback: int,
    price_field: str = "close",
) -> dict[str, float]:
    """Compute each symbol's trailing return AS OF a rebalance timestamp.

    The returns-computing COUNTERPART to rank_by_trailing_return: this produces the
    dict[str, float] that that function consumes.  For each symbol it computes the
    trailing return via the SHARED trailing_return_series helper (so cross-sectional
    and time-series momentum measure "trailing return" identically and cannot
    drift), selects the value as of the rebalance date, and OMITS any symbol that is
    ineligible or has no valid (non-NaN) return — so the returned dict contains ONLY
    eligible symbols with finite returns, exactly matching rank_by_trailing_return's
    "ineligible symbols are simply absent" input contract.

    The rebalance point is a TIMESTAMP, not an integer index.  The basket's symbols
    have different-length histories, so an integer index would land on a DIFFERENT
    calendar date per symbol — a silent lookahead-style bug (one symbol's "index k"
    could be a later date than another's).  A timestamp is safe because every 1d bar
    is floored to midnight UTC on ingest, so symbols align by date equality and "as
    of D" means the same calendar instant for every symbol.

    "As of D" is resolved on each symbol's OWN month-end grid: the most recent
    month-end AT OR BEFORE D.  s.loc[:D] is right-INCLUSIVE of D on a sorted
    DatetimeIndex, so a rebalance date that IS a month-end selects itself and never a
    month-end after it (no lookahead); a date that is a month-end for one symbol and
    mid-month for another still gives each its own correct most-recent-at-or-before
    value.  We use the explicit .loc[:D].iloc[-1] slice rather than Series.asof(D):
    asof silently returns the last NON-NaN value at-or-before D, which would skip a
    warmup NaN and hand back a stale earlier month-end instead of correctly omitting
    the symbol.  The explicit slice makes the rule literal and auditable.

    Args:
        bars_by_symbol: Maps symbol -> that symbol's time-ordered OHLCVBar list.
        rebalance_ts:   The "as of" timestamp at which returns are measured.
        lookback:       Trailing formation window in month-end observations.
        price_field:    Which price field to read — "close" or "adj_close".

    Returns:
        A dict mapping each ELIGIBLE symbol to its finite as-of trailing return.
        Symbols that are too short (no month-end at-or-before D) or still in warmup
        (as-of value is NaN) are absent.  An all-ineligible basket returns {}.
    """
    # Accumulate only the eligible symbols; symbols failing either guard below are
    # never inserted, so the returned dict already satisfies the ranker's contract.
    result: dict[str, float] = {}

    # Each symbol is computed INDEPENDENTLY on its own bars and its own month-end
    # grid — there is no cross-symbol alignment beyond the shared rebalance date.
    for symbol, bars in bars_by_symbol.items():
        # The full month-end-indexed trailing-return Series, from the SHARED helper.
        # Do NOT re-derive the arithmetic here — consistency with TSMOM is the point.
        s = trailing_return_series(bars, lookback, price_field)

        # AS-OF SELECTION: all month-ends at-or-before the rebalance date.  Label
        # slicing on a sorted DatetimeIndex is right-inclusive of rebalance_ts, so a
        # rebalance date that IS a month-end is included (selects itself), never a
        # later one — structurally no lookahead.
        sliced = s.loc[:rebalance_ts]

        # GUARD (a) — EMPTY: rebalance_ts precedes this symbol's first month-end
        # (too little history / not listed yet).  This MUST come before the NaN
        # guard: .iloc[-1] on an empty slice raises IndexError, so there is nothing
        # to NaN-check yet.  Omit the symbol entirely — the ineligible path.
        if len(sliced) == 0:
            continue

        # The most recent month-end value at-or-before rebalance_ts.  Safe now that
        # the empty case is guarded above.
        value = sliced.iloc[-1]

        # GUARD (b) — NaN: rebalance_ts falls within this symbol's warmup (the first
        # `lookback` month-ends are NaN in trailing_return_series), so there is no
        # valid trailing return yet.  Omit rather than emit a NaN.
        if math.isnan(value):
            continue

        # Eligible with a finite return — include it, coercing to a plain float so
        # the dict is a clean dict[str, float] (not numpy scalars) for the ranker.
        result[symbol] = float(value)

    # Only eligible symbols with finite returns remain; ineligible/warmup symbols
    # are absent, so the ranker's NaN backstop stays defense-in-depth, not primary.
    return result


def single_date_weights(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    as_of_date: datetime,
    lookback: int,
    top_n: int,
    price_field: str = "close",
    hold_when_all_negative: bool = False,
) -> dict[str, float]:
    """The SINGLE birthplace of rotation target weights for ONE decision date.

    Composes the two existing cross-sectional primitives — trailing_returns_at (the
    as-of returns-computer) then rank_by_trailing_return (the ranker/selector) — and
    turns the selected top-N symbols into an equal-weight target: each held symbol
    gets a FIXED 1/top_n (rule (b)), so h held symbols invest h/top_n and the
    remaining (top_n - h)/top_n is implicitly cash.  An empty selection (all-cash)
    returns {}.

    This is the ONE place rotation weights are computed.  Both the backtest path
    (rotation_backtest, which walks month-ends and calls this per decision date) and
    the future live-target path consume THIS function, so the two CANNOT diverge on
    how a decision date's weights are formed — the same reasoning behind sharing
    month_end_indices and trailing_return_series across the stack.

    It is PURE: no I/O, no printing, no DuckDB, no global state — a deterministic
    function of its arguments only, exactly like the two primitives it composes.

    Args:
        bars_by_symbol:         symbol -> that symbol's time-ordered OHLCVBar list.
        as_of_date:             the decision timestamp; weights are formed as of here
                                (trailing_returns_at resolves strictly at-or-before it,
                                so there is no lookahead).
        lookback:               trailing formation window in month-end observations.
        top_n:                  how many symbols to hold AND the fixed weight denominator
                                (each held symbol gets 1/top_n).
        price_field:            which price field to rank on — "close" or "adj_close".
        hold_when_all_negative: absolute-filter switch passed through to the ranker.

    Returns:
        A dict mapping each held symbol to its equal-weight 1/top_n fraction; {} when
        nothing qualifies (the all-cash target).  Weights need NOT sum to 1.0 — the
        remainder is cash (see combine_period_returns' cash-remainder rule).
    """
    # STEP 1 — as-of trailing returns for every eligible symbol at the decision date.
    # trailing_returns_at resolves at-or-before as_of_date, so no future bar is read.
    tr = trailing_returns_at(bars_by_symbol, as_of_date, lookback, price_field)

    # STEP 2 — rank those returns and select the top-N (empty when nothing qualifies).
    held = rank_by_trailing_return(tr, top_n, hold_when_all_negative)

    # STEP 3 — equal-weight 1/top_n over the held names (rule (b)); {} stays all-cash.
    weights = {symbol: 1.0 / top_n for symbol in held}

    # The single per-date target weight vector, shared by backtest and live paths.
    return weights
