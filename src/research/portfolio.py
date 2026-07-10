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

# OHLCVBar is the bar schema the rebalance loop and its per-bar-return helper consume.
# combine_period_returns itself touches no bars; only the loop (which slices per-symbol
# bars into holding-period return arrays) needs the bar dataclass.
from src.data.schema import OHLCVBar

# month_end_indices is the SHARED month-end detector (extracted from TSMOM).  The loop
# derives its rebalance schedule from the reference symbol's month-ends via this one
# definition, so the rotation's calendar cannot drift from TSMOM's "month-end" notion.
from src.strategies.time_series_momentum import month_end_indices

# trailing_returns_at (returns-computer) and rank_by_trailing_return (ranker) are the two
# cross-sectional primitives the loop chains at each rebalance: compute as-of trailing
# returns, then rank+select.  This portfolio->cross_sectional->time_series_momentum import
# is the SAME research->strategies direction cross_sectional.py and optuna_fit.py already
# use, so there is no circular-import risk (nothing in those modules imports portfolio.py).
from src.research.cross_sectional import trailing_returns_at, rank_by_trailing_return


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


def _per_bar_log_returns(bars: list[OHLCVBar], price_field: str = "close") -> np.ndarray:
    """Per-bar log returns for ONE symbol's bar slice, in the engine's exact convention.

    Reproduces Backtester.run's per-bar return arithmetic (engine.py:288-293) WITHOUT
    calling run(): an array of length len(bars) whose element 0 is a structural 0.0 (no
    prior bar to compare against) and whose element i>=1 is log(close[i] / close[i-1]) on
    the chosen price_field.  This is the Day-39 "Decision-A" path — reuse the engine's
    arithmetic conventions, not run() — so the rotation loop computes its per-symbol
    holding-period returns identically to how the single-symbol engine would.

    It is a SEPARATE, independently unit-testable helper on purpose: it is the exact unit
    a FUTURE N=1 engine-equivalence test will target (a single held symbol at full weight
    must reproduce the engine's own stream), so it must stay byte-for-byte the engine's
    `asset_returns = np.zeros(...); asset_returns[1:] = np.log(closes[1:] / closes[:-1])`
    — do NOT change this arithmetic without changing the engine, or that test cannot pass.

    Args:
        bars:        Time-ordered OHLCVBar list (ascending chronological) for ONE symbol.
        price_field: Which price field to read — "close" or "adj_close".

    Returns:
        A float64 ndarray of length len(bars): element 0 is 0.0, element i>=1 is the
        log return from bar i-1 to bar i.

    Raises:
        ValueError: if bars is empty (no return series can be formed).
    """
    # Empty bars cannot form any return series; fail loud and echo, matching the
    # validation style in combine_period_returns / rank_by_trailing_return.
    if not bars:
        raise ValueError(f"bars must be non-empty, got {len(bars)} bars")

    # Read the chosen price field off each bar into a float64 array.  getattr(bar,
    # price_field) picks "close" or "adj_close" exactly as trailing_return_series does,
    # so the loop's return basis matches the ranking basis when the caller passes the
    # same price_field to both.
    closes = np.array([getattr(bar, price_field) for bar in bars], dtype=np.float64)

    # Allocate-then-fill, IDENTICAL to engine.py:288-293.  Element 0 stays 0.0 (the
    # structural "no prior bar" zero); this is the leading zero the loop drops when it
    # uses an anchor bar as the denominator for the first in-period return.
    returns = np.zeros(len(bars), dtype=np.float64)

    # closes[1:] / closes[:-1] is the per-bar price ratio; np.log turns it into the
    # additive log return.  This line MUST match the engine's
    # np.log(closes[1:] / closes[:-1]) with returns[0] == 0.0 (see the docstring's
    # equivalence-test note) — it is the engine's convention, reproduced verbatim.
    returns[1:] = np.log(closes[1:] / closes[:-1])

    # float64, length len(bars), engine-convention log returns.
    return returns


def rotation_backtest(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    reference_symbol: str,
    lookback: int,
    top_n: int,
    price_field: str = "close",
    cash_per_bar_return: float = 0.0,
    hold_when_all_negative: bool = False,
) -> np.ndarray:
    """Walk month-end rebalances, rank+hold the top-N, stitch one portfolio log-return stream.

    The pure ORCHESTRATION core of the cross-sectional rotation backtester: it walks the
    reference symbol's month-ends, and for each adjacent pair (D_k, D_{k+1}) it ranks the
    basket AS OF D_k (no lookahead), selects the top-N, computes each held symbol's per-bar
    log returns over the holding period (D_k, D_{k+1}], weights them per rule (b), and folds
    them into one period stream via combine_period_returns.  All period streams are
    concatenated chronologically into ONE per-bar log-return array and returned.

    It touches NO DuckDB and NO engine: bars_by_symbol is passed in, and every per-bar
    return is computed via _per_bar_log_returns (the engine's arithmetic, not run()).  It
    chains the four locked contracts — month_end_indices, trailing_returns_at,
    rank_by_trailing_return, combine_period_returns — and adds only the scheduling, the
    per-symbol holding-period slicing, the equal-length alignment guard, and the stitch.

    Args:
        bars_by_symbol:         symbol -> that symbol's time-ordered OHLCVBar list.
        reference_symbol:       the long-history spine (e.g. "SPY") whose month-ends define
                                the shared rebalance schedule AND the holding-period date grid.
        lookback:               trailing formation window (month-end observations), passed to
                                trailing_returns_at.
        top_n:                  how many symbols to hold; also the FIXED weight denominator
                                under rule (b) (each held symbol gets 1/top_n).
        price_field:            "close" or "adj_close" — used for BOTH ranking and returns.
        cash_per_bar_return:    per-bar return on the cash fraction (see combine_period_returns).
        hold_when_all_negative: passed to rank_by_trailing_return (absolute-filter switch).

    Returns:
        One float64 ndarray: the per-bar portfolio log-return stream stitched across every
        complete holding period, in the SAME log-return space combine_period_returns /
        metrics.py / _stitch_oos consume — so it can later feed the walk-forward bridge and
        metrics unchanged.

    Raises:
        ValueError: reference_symbol not in bars_by_symbol.
        ValueError: fewer than 2 reference month-ends (cannot form one holding period).
        ValueError: a held symbol's holding-period bars do not align to the reference spine
                    (missing/mismatched interior date) — fail loud, never pad/truncate.
        ValueError: (propagated) top_n < 1, from rank_by_trailing_return.
    """
    # GUARD — the reference symbol must be present, since its month-ends ARE the schedule.
    # Echo the missing name so a typo'd or absent spine fails immediately and legibly.
    if reference_symbol not in bars_by_symbol:
        raise ValueError(
            f"reference_symbol {reference_symbol!r} not in bars_by_symbol "
            f"(keys: {sorted(bars_by_symbol.keys())})"
        )

    # The reference symbol's bars — the long-history spine that defines both the rebalance
    # timestamps and the per-period holding date grid.  Computed ONCE, up front.
    ref_bars = bars_by_symbol[reference_symbol]

    # Positional indices of the reference symbol's month-ends, via the SHARED detector, so
    # the rotation's calendar is the same "month-end" notion TSMOM uses.
    ref_mei = month_end_indices(ref_bars)

    # GUARD — need at least TWO month-ends to form one adjacent pair (D_0, D_1); a single
    # month-end has no next rebalance and thus no holding period.  Echo the count.
    if len(ref_mei) < 2:
        raise ValueError(
            f"need at least 2 reference month-ends to form a holding period, "
            f"got {len(ref_mei)}"
        )

    # The rebalance timestamps: the reference bar timestamp at each month-end index, in
    # ascending chronological order.  These are the D_0 .. D_{K-1} the loop walks.
    rebalance_ts = [ref_bars[i].timestamp for i in ref_mei]

    # Precompute the reference timestamps as a plain list once so the per-period spine can
    # be sliced by simple timestamp comparison against them.
    ref_timestamps = [bar.timestamp for bar in ref_bars]

    # TURNOVER SEAM — prev_holdings carries the previous period's selection across
    # iterations.  Initialized EMPTY so the first rebalance's "added" set is the whole
    # initial selection.  At the END of each iteration we set prev_holdings = set(held);
    # then set(held) - prev_holdings (added) and prev_holdings - set(held) (dropped) are
    # computable HERE for the FUTURE transaction-cost model.  NO cost is applied today —
    # this only reserves the seam.
    prev_holdings: set[str] = set()

    # Collect each complete period's combined per-bar stream; concatenated at the end into
    # the single stitched output.
    period_streams: list[np.ndarray] = []

    # Walk ADJACENT month-end pairs: k from 0 to K-2, so (D_k, D_{k+1}) is always a
    # complete pair.  The final month-end D_{K-1} has no next rebalance, so we STOP there
    # and never emit a partial trailing period — the locked "stop at last complete period"
    # rule.
    for k in range(len(rebalance_ts) - 1):
        # D_k is the decision point (rank as of here); D_{k+1} is the next rebalance and
        # the right edge of this holding period.
        d_k = rebalance_ts[k]
        d_next = rebalance_ts[k + 1]

        # STEP 1 — RANK AS OF D_k (no lookahead).  trailing_returns_at resolves strictly
        # at-or-before D_k, so the ranking is fully determined BEFORE any holding-period
        # bar (all of which are strictly after D_k) is consumed.  held may be EMPTY (cash)
        # or have FEWER than top_n symbols (partial-cash under rule (b)).
        tr = trailing_returns_at(bars_by_symbol, d_k, lookback, price_field)
        held = rank_by_trailing_return(tr, top_n, hold_when_all_negative)

        # STEP 2 — HOLDING-PERIOD DATE SPINE from the reference symbol, HALF-OPEN LEFT /
        # CLOSED RIGHT: reference bars strictly AFTER D_k and AT-OR-BEFORE D_{k+1}, i.e.
        # the interval (D_k, D_{k+1}].  D_k's OWN bar is EXCLUDED because it is the decision
        # point — its return already accrued before the decision was made.  This exclusion
        # is the rotation analogue of the engine's one-bar lag (signals[:-1] * returns[1:]):
        # a signal decided at D_k earns returns starting from the NEXT bar, never D_k's own.
        spine_dates = [ts for ts in ref_timestamps if d_k < ts <= d_next]

        # n_bars is the number of bars in this holding period.  D_{k+1} itself is a
        # reference bar in (D_k, D_{k+1}], so n_bars is always >= 1.
        n_bars = len(spine_dates)

        # Accumulate each held symbol's per-bar return array, keyed by symbol, for the
        # combine step.  Empty when held is empty (the all-cash period).
        per_symbol_returns: dict[str, np.ndarray] = {}

        # STEP 3 — build each held symbol's per-bar return array OVER THE SPINE.
        for symbol in held:
            # This symbol's own bars (ascending chronological).
            sym_bars = bars_by_symbol[symbol]

            # In-period bars: the symbol's bars in the SAME interval (D_k, D_{k+1}] as the
            # spine.  These are the bars whose returns count toward this holding period.
            in_period = [bar for bar in sym_bars if d_k < bar.timestamp <= d_next]

            # ANCHOR bar: the single bar immediately BEFORE the interval — the symbol's
            # last bar AT-OR-BEFORE D_k.  It is the denominator for the FIRST in-period
            # log return (close[first-in-period] / close[anchor]).  A held symbol was
            # ranked as of D_k, so it necessarily has a bar at-or-before D_k, so this list
            # is non-empty and [-1] is safe.
            at_or_before = [bar for bar in sym_bars if bar.timestamp <= d_k]
            anchor = at_or_before[-1]

            # Per-bar log returns over [anchor] + in_period.  The anchor supplies the
            # first return's denominator; _per_bar_log_returns puts a structural 0.0 at
            # index 0 (the anchor has no prior bar), so we DROP that leading 0.0 with [1:]
            # to leave exactly one real log return per in-period bar.
            arr = _per_bar_log_returns([anchor] + in_period, price_field)[1:]

            # EQUAL-LENGTH / ALIGNMENT GUARD — fail loud, never pad or truncate.  The
            # symbol's in-period array must have exactly n_bars elements AND its in-period
            # dates must match the reference spine dates one-for-one.  A held symbol missing
            # an interior day the spine has (a data gap) would otherwise silently misalign
            # the return series and corrupt the backtest, so we raise, naming the symbol and
            # the first mismatching date.  (Hermetic tests share dates, so this passes;
            # the guard protects real ragged data.)
            in_period_dates = [bar.timestamp for bar in in_period]
            if in_period_dates != spine_dates:
                raise ValueError(
                    f"held symbol {symbol!r} does not align to the reference spine over "
                    f"({d_k}, {d_next}]: got {len(in_period_dates)} bars vs spine's "
                    f"{n_bars}; symbol dates {in_period_dates} != spine dates {spine_dates}"
                )

            # Aligned and length-checked — record it for the combine.
            per_symbol_returns[symbol] = arr

        # STEP 4 — WEIGHTS PER RULE (b): FIXED 1/top_n per held symbol (NOT 1/len(held)).
        # So h held symbols invest h/top_n and the remaining (top_n - h)/top_n sits in cash
        # (combine_period_returns' cash remainder handles it).  This is the locked rule (b):
        # de-risk into cash when fewer than top_n qualify — the continuous extension of the
        # all-negative -> cash discipline.  Empty held -> empty weights -> fully cash.
        # NOTE: switching to rule (a) later is a ONE-LINE change to {s: 1.0/len(held) ...}
        # (guarding len(held) > 0), NOT a rewrite — combine_period_returns already handles
        # any weights summing to <= 1.
        weights = {symbol: 1.0 / top_n for symbol in held}

        # STEP 5 — COMBINE this period into one per-bar stream.  For the empty-held case
        # per_symbol_returns and weights are both {} and the cash remainder (1.0) makes the
        # whole period earn cash_per_bar_return on each bar.
        period_returns = combine_period_returns(
            per_symbol_returns=per_symbol_returns,
            weights=weights,
            n_bars=n_bars,
            cash_per_bar_return=cash_per_bar_return,
        )

        # Record this period's stream for the chronological stitch.
        period_streams.append(period_returns)

        # STEP 6 — advance the TURNOVER SEAM: this period's holdings become next period's
        # prev_holdings.  (No cost applied — reserved for the future cost model.)
        prev_holdings = set(held)

    # STITCH — concatenate every complete period's stream in chronological order into ONE
    # per-bar log-return array.  At least one pair exists (len(ref_mei) >= 2 guaranteed
    # above), so period_streams is non-empty and np.concatenate is well-defined.  The
    # result is in the SAME log-return space combine_period_returns / metrics.py /
    # _stitch_oos use, so it can later feed the walk-forward bridge and metrics unchanged.
    return np.concatenate(period_streams)
