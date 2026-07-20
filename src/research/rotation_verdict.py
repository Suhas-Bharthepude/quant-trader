# src/research/rotation_verdict.py

"""
Fixed-parameter walk-forward VERDICT for the cross-sectional rotation backtester.

This module is the walk-forward BRIDGE for rotation_backtest.  The generic harness
in walk_forward.py cannot be reused wholesale for rotation for two structural
reasons:

  1. walk_forward_validate is hard-wired to SINGLE-SYMBOL Strategy objects
     (it calls strategy.generate_signals(bars) and Backtester.run on ONE symbol's
     bars).  Rotation is inherently MULTI-symbol: it consumes bars_by_symbol and
     emits one combined per-bar log-return stream, so it cannot honestly masquerade
     as a Strategy.

  2. walk_forward_splits cuts folds on BAR INDEX.  Rotation's causality lives on the
     reference symbol's MONTH-END grid (it ranks as of a month-end and holds until
     the next one), so a bar-index split could land a train/test boundary in the
     middle of a holding period and score a test fold on a return whose ranking used
     training data — lookahead.  Rotation needs a MONTH-END-aligned splitter.

So we REUSE only the pure SCORING layer — _stitch_oos (from walk_forward.py) and,
through it, the metrics functions — which already operate on plain returns arrays /
BacktestResult lists.  Everything rotation-specific (the month-end fold splitter,
the per-fold rotation returns producer, and the equal-weight-basket buy-and-hold
benchmark) is ADDED here.  Nothing in walk_forward.py, portfolio.py, metrics.py, or
the engine is modified: the bridge is purely additive.

This is the FIXED-PARAMETER bridge (step A).  Per-fold hyperparameter search and the
overfitting tax are a LATER step (B): under fixed parameters a "tax" would be hollow
(there is no per-fold parameter selection to overfit), so no tax is computed or
reported here — RotationVerdict deliberately has NO tax field.
"""

# numpy is the array backend shared with the engine, metrics.py, and portfolio.py.
# Every per-bar series here (fold returns, the prepended-zero returns array, the
# equity curve) is a float64 ndarray so the scoring math matches the rest of the stack.
import numpy as np

# dataclass auto-generates __init__/__repr__/__eq__ for RotationVerdict; frozen=True
# makes it immutable, consistent with WalkForwardResult and BacktestResult.
from dataclasses import dataclass

# datetime + timezone build the placeholder start/end timestamps on the minimal
# per-fold BacktestResult.  _stitch_oos never reads those fields (see _fold_result),
# so a fixed placeholder instant is honest — it is provably unused, not fake data.
from datetime import datetime, timezone

# OHLCVBar is the bar schema every function here slices and forwards to rotation_backtest.
from src.data.schema import OHLCVBar

# month_end_indices is the SHARED month-end detector — the SAME one rotation_backtest
# and TSMOM use internally — so the fold splitter's "month-end" notion cannot drift
# from the rotation loop's own schedule.
from src.strategies.time_series_momentum import month_end_indices

# rotation_backtest is the unit under evaluation: it produces the cost-adjusted per-bar
# log-return stream we walk-forward-validate out-of-sample.
from src.research.portfolio import rotation_backtest

# _stitch_oos is the REUSED scoring seam from the generic harness.  It concatenates
# each fold's returns (stripping a leading structural zero via fr.returns[1:]), builds
# the equity curve, and computes total_return / sharpe / max_drawdown / sortino.  It
# reads ONLY fr.returns off each BacktestResult, which is why our minimal per-fold
# result can leave every other field a safe placeholder (see _fold_result).
from src.research.walk_forward import _stitch_oos

# BacktestResult is the container _stitch_oos iterates over.  We build a minimal one
# per fold whose .returns carries a PREPENDED structural zero (see _fold_result).
from src.backtest.result import BacktestResult

# metrics.sharpe_ratio scores each candidate's in-sample train-span stream in the
# per-fold grid search (FUNCTION 6 below).  Module import (not a from-import of the
# function) matches walk_forward.py's style, so the search scores in-sample on the
# EXACT same function _stitch_oos uses for the OOS side — identical footing, honest tax.
from src.backtest import metrics

# summarize_tax / TaxRow are REUSED from the TSMOM overfitting-tax CLI so the rotation
# tax uses the IDENTICAL formula (mean per-fold in-sample Sharpe − stitched fitted OOS
# Sharpe) and the SAME empty-list guard — the rotation number is therefore directly
# comparable to the TSMOM number.  Imported by the same module path the momentum tax
# test (tests/test_momentum_overfitting_tax.py) uses, which is known to resolve here.
from scripts.overfitting_tax import summarize_tax, TaxRow


# A single fixed placeholder instant for the minimal per-fold BacktestResult's
# start_date/end_date.  _stitch_oos never reads these fields, so the value is provably
# irrelevant to any computed number; a fixed constant is more honest than fabricating
# per-fold dates that would imply a precision we are not actually using.
_PLACEHOLDER_DT = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# RotationVerdict — the out-of-sample verdict for a rotation configuration.
# ---------------------------------------------------------------------------

# frozen=True → immutable historical fact, matching WalkForwardResult / BacktestResult.
@dataclass(frozen=True)
class RotationVerdict:
    """Stitched out-of-sample rotation metrics vs an equal-weight-basket benchmark.

    All oos_* fields are the rotation strategy's stitched OOS aggregates over the
    disjoint test spans; all bh_* fields are the equal-weight-basket buy-and-hold
    benchmark measured over the IDENTICAL test spans by IDENTICAL code (same folds,
    same warm-up, same test-span extraction, same stitch), differing ONLY in top_n
    and the absolute-filter flag — so oos_* vs bh_* is a true edge-vs-beta read.

    There is deliberately NO overfitting-tax field: this is the fixed-parameter
    bridge, and under fixed parameters a tax would be hollow (step B adds per-fold
    parameter search and the tax).
    """

    # Number of (train, test) folds actually run.
    n_folds: int

    # Rotation strategy, stitched OOS aggregates over the disjoint test spans.
    oos_sharpe: float          # annualised Sharpe over the stitched OOS returns
    oos_sortino: float         # annualised Sortino over the same stitched returns
    oos_max_drawdown: float    # worst peak-to-trough of the stitched OOS equity curve
    oos_total_return: float    # total return over the stitched OOS timeline

    # Equal-weight-basket buy-and-hold, stitched over the IDENTICAL test spans.
    bh_sharpe: float           # annualised B&H Sharpe over the stitched benchmark returns
    bh_sortino: float          # annualised B&H Sortino over the same stitched returns
    bh_max_drawdown: float     # worst peak-to-trough of the stitched B&H equity curve
    bh_total_return: float     # total return over the stitched B&H timeline


# ---------------------------------------------------------------------------
# FUNCTION 1 — the month-end fold splitter.
# ---------------------------------------------------------------------------

def rotation_month_end_folds(
    ref_bars: list[OHLCVBar],
    train_months: int,
    test_months: int,
    step_months: int | None = None,
) -> list[tuple[int, int, int]]:
    """Slice the reference symbol's month-end grid into (train, test) fold triples.

    Returns a list of (train_start_me, test_start_me, test_end_me) triples, all
    INDICES INTO THE MONTH-END ARRAY (positions in month_end_indices(ref_bars)), NOT
    bar indices.  A fold is:
        train  = month-ends [train_start_me : test_start_me)
        test   = month-ends [test_start_me  : test_end_me)
    with test_start_me = train_start_me + train_months and
         test_end_me   = test_start_me   + test_months.

    NO-LOOKAHEAD GUARANTEE (the rotation analogue of walk_forward_splits' strict
    train/test adjacency): the split falls ON month-end position test_start_me.  The
    test fold's FIRST rank is taken AS OF D_{test_start_me}, which uses only data
    at-or-before that month-end (all within the train span), and the test HOLDING
    RETURNS are strictly AFTER it (the period (D_{test_start_me}, D_{test_start_me+1}]
    and onward).  So no test return can inform its own ranking — exactly the property
    walk_forward_splits enforces by starting the test window on the bar immediately
    after the last training bar.

    ROTATION-SPECIFIC OFF-BY-ONE vs walk_forward_splits (documented deliberately):
    walk_forward_splits' test window is the last `test_size` BARS, each a standalone
    observation, so a complete fold needs train_size + test_size bars.  A rotation
    test "observation" is a HOLDING PERIOD *between* two month-ends, and the test span
    of `test_months` decision month-ends [test_start_me, test_end_me) needs one MORE
    month-end — the closing boundary D_{test_end_me} that terminates the last test
    holding period (D_{test_end_me-1}, D_{test_end_me}].  Consecutive non-overlapping
    folds SHARE that boundary (it is fold i's closing edge and fold i+1's opening
    decision), which is exactly what makes the test spans TILE with no gap and no
    overlap (fold i's last period ends at D_{test_end_me}; fold i+1's first period
    starts at D_{test_end_me}).  Hence a complete fold requires test_end_me <= M - 1
    (D_{test_end_me} must be a real month-end), and the minimum month-end count is
    train_months + test_months + 1, not train_months + test_months.

    Args:
        ref_bars:     the reference symbol's time-ordered OHLCVBar list (its month-ends
                      define the shared rebalance schedule).
        train_months: number of TRAIN month-ends per fold.  Must be >= 1.
        test_months:  number of TEST decision month-ends per fold.  Must be >= 1.
        step_months:  how many month-ends to advance train_start_me between folds.
                      Defaults to test_months (non-overlapping test spans that tile the
                      post-train timeline exactly once), mirroring walk_forward_splits'
                      step=test_size default.  Must be >= 1 when provided.

    Returns:
        A non-empty list of (train_start_me, test_start_me, test_end_me) triples in
        chronological order.  The guards below guarantee at least one complete fold.

    Raises:
        ValueError: train_months < 1, test_months < 1, step_months < 1 (when given),
                    or fewer than train_months + test_months + 1 month-ends available.
    """
    # GUARD — a train span of zero or fewer month-ends cannot warm any ranking; echo
    # the bad value, matching walk_forward_splits' validation style.
    if train_months < 1:
        raise ValueError(f"train_months must be >= 1, got {train_months}")

    # GUARD — a test span of zero month-ends has nothing to evaluate; echo the value.
    if test_months < 1:
        raise ValueError(f"test_months must be >= 1, got {test_months}")

    # GUARD — step_months is optional (None -> test_months), but an explicit value < 1
    # would either loop forever (0) or walk backwards (negative); reject it, echoing.
    if step_months is not None and step_months < 1:
        raise ValueError(f"step_months must be >= 1 when provided, got {step_months}")

    # Total month-ends available on the reference grid — the ceiling every fold triple
    # must fit under.  Computed once via the SHARED detector so "month-end" here is the
    # exact notion rotation_backtest schedules on.
    m = len(month_end_indices(ref_bars))

    # GUARD — not enough month-ends for even one complete fold.  The +1 is the closing
    # boundary D_{test_end_me} the last test holding period needs (see the docstring's
    # off-by-one note); without it the first fold's test_end_me would land past the
    # last month-end.  State required vs available so the caller can size the window.
    if m < train_months + test_months + 1:
        raise ValueError(
            f"need at least train_months + test_months + 1 = "
            f"{train_months + test_months + 1} reference month-ends "
            f"(the +1 is the closing month-end that terminates the last test holding "
            f"period); got {m}"
        )

    # Resolve the effective step: None -> test_months, which tiles the test spans with
    # no overlap and no gap (see the docstring's tiling note).
    effective_step = step_months if step_months is not None else test_months

    # Accumulate fold triples in chronological order.
    folds: list[tuple[int, int, int]] = []

    # train_start_me is the month-end position of this fold's first TRAIN month-end; it
    # advances by effective_step each iteration so consecutive train windows slide
    # forward uniformly (mirrors walk_forward_splits' `start`).
    train_start_me = 0

    # Emit folds while a COMPLETE one fits.  test_end_me <= m - 1 is the exact tail rule:
    # the closing boundary D_{test_end_me} must be a real month-end (positions run
    # 0..m-1), so the last usable test_end_me is m - 1.  This is the rotation counterpart
    # of walk_forward_splits' `start + train + test <= len(bars)` <= tail rule; the sole
    # difference is the -1, which reserves the closing boundary month-end.
    while True:
        # test_start_me is the boundary the split falls ON — train ends here, test begins
        # its ranking here (as-of, no lookahead).
        test_start_me = train_start_me + train_months

        # test_end_me is the EXCLUSIVE end of the test decision month-ends AND the index
        # of the closing boundary D_{test_end_me} that terminates the last test period.
        test_end_me = test_start_me + test_months

        # Stop the instant the next fold's closing boundary would reach past the last
        # month-end — drop the ragged tail rather than emit a short/undefined fold.
        if test_end_me > m - 1:
            break

        # A complete fold — record it in (train_start, test_start, test_end) order.
        folds.append((train_start_me, test_start_me, test_end_me))

        # Advance the window by the effective step for the next fold.
        train_start_me += effective_step

    # The month-end-count guard above guarantees the first iteration produced a fold
    # (train_start_me=0 -> test_end_me = train_months + test_months <= m - 1), so folds
    # is always non-empty here — no empty-list guard needed, matching walk_forward_splits.
    return folds


# ---------------------------------------------------------------------------
# FUNCTION 2 — per-fold rotation returns producer (test span only).
# ---------------------------------------------------------------------------

def _rotation_fold_returns(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    reference_symbol: str,
    train_start_me: int,
    test_start_me: int,
    test_end_me: int,
    lookback: int,
    top_n: int,
    cost_rate: float,
    price_field: str = "close",
    cash_per_bar_return: float = 0.0,
    hold_when_all_negative: bool = False,
) -> np.ndarray:
    """Produce the rotation per-bar log-return stream for ONE fold's TEST SPAN.

    The rotation is WARMED inside the fold's train span and scored only on its test
    span: we slice every symbol's bars to the fold's [warm-up start, closing boundary]
    window, run rotation_backtest over that slice (which emits warm-up PLUS test
    holding periods), then strip the leading warm-up per-bar returns so exactly the
    test span remains.

    WARM-UP SUFFICIENCY (why the driver enforces train_months > lookback): the first
    test decision at D_{test_start_me} needs a full `lookback` of prior month-ends, i.e.
    D_{test_start_me - lookback}, which lies in the slice iff train_months >= lookback;
    and the warm-up decision IMMEDIATELY before the test span (at D_{test_start_me-1},
    which sets the test span's entry turnover cost) needs D_{test_start_me-1-lookback},
    which lies in the slice iff train_months >= lookback + 1, i.e. train_months >
    lookback.  Under that strict guard, the entire test-span stream — including its
    first-bar entry-turnover cost — is IDENTICAL to what a full-history rotation would
    produce over those same periods, so warming inside the fold introduces no artefact.

    Args:
        bars_by_symbol:         symbol -> that symbol's time-ordered OHLCVBar list.
        reference_symbol:       the spine whose month-ends define the schedule/grid.
        train_start_me:         month-end position of the warm-up start (fold train start).
        test_start_me:          month-end position the split falls on (first test decision).
        test_end_me:            month-end position of the closing boundary (last test period end).
        lookback:               trailing formation window in month-ends.
        top_n:                  how many symbols to hold (also the fixed weight denominator).
        cost_rate:              per-unit-turnover transaction cost fraction (see rotation_backtest).
        price_field:            "close" or "adj_close" — used for BOTH ranking and returns.
        cash_per_bar_return:    per-bar return on the cash fraction.
        hold_when_all_negative: absolute-filter switch passed through to the ranker.

    Returns:
        A float64 ndarray: the test-span-only per-bar log-return stream for this fold.
    """
    # The reference symbol's own bars and month-end grid — the spine that defines both
    # the slice window edges and the warm-up/test per-bar counts below.
    ref_bars = bars_by_symbol[reference_symbol]

    # Positional (bar-index) locations of the reference month-ends, via the SHARED
    # detector, so these positions match rotation_backtest's own scheduling exactly.
    ref_mei = month_end_indices(ref_bars)

    # Timestamp of the fold's WARM-UP START month-end (D_{train_start_me}) — the LEFT
    # edge of the slice (inclusive).  int(...) coerces the numpy index to a Python int.
    d_warm_start = ref_bars[int(ref_mei[train_start_me])].timestamp

    # Timestamp of the fold's CLOSING BOUNDARY month-end (D_{test_end_me}) — the RIGHT
    # edge of the slice (inclusive).  This month-end terminates the last test holding
    # period (D_{test_end_me-1}, D_{test_end_me}], so it MUST be inside the slice.
    d_close = ref_bars[int(ref_mei[test_end_me])].timestamp

    # Slice EVERY symbol's bars to [d_warm_start, d_close] inclusive both ends, so
    # rotation_backtest sees the warm-up month-ends AND the test month-ends (and, for
    # each held symbol, the anchor/holding bars it needs over that window).  Bars strictly
    # before d_warm_start or strictly after d_close are dropped.
    sliced = {
        symbol: [bar for bar in bars if d_warm_start <= bar.timestamp <= d_close]
        for symbol, bars in bars_by_symbol.items()
    }

    # Run the rotation over the slice.  It returns ONE stitched per-bar log-return stream
    # covering ALL rebalance pairs in the slice: the warm-up pairs (from train_start_me up
    # to test_start_me) followed by the test pairs (from test_start_me to test_end_me-1).
    # Pass every parameter by keyword to avoid any positional-order mistake.
    stream = rotation_backtest(
        bars_by_symbol=sliced,
        reference_symbol=reference_symbol,
        lookback=lookback,
        top_n=top_n,
        price_field=price_field,
        cash_per_bar_return=cash_per_bar_return,
        hold_when_all_negative=hold_when_all_negative,
        cost_rate=cost_rate,
    )

    # ALIGNMENT CRUX — count the leading WARM-UP per-bar returns to strip, from the
    # REFERENCE SPINE, NOT by assuming one bar per rebalance pair (months can hold many
    # daily bars).  Period P_k = (D_k, D_{k+1}] contains the ref bars at positions
    # ref_mei[k]+1 .. ref_mei[k+1], i.e. exactly ref_mei[k+1] - ref_mei[k] bars.  Summing
    # that count over the warm-up pairs k in [train_start_me, test_start_me) telescopes to
    # ref_mei[test_start_me] - ref_mei[train_start_me] — the number of leading per-bar
    # returns that belong to warm-up rather than the test span.
    warmup_bars = int(ref_mei[test_start_me]) - int(ref_mei[train_start_me])

    # The same telescoping gives the TEST-span per-bar count: the bars over pairs
    # k in [test_start_me, test_end_me-1] sum to ref_mei[test_end_me] - ref_mei[test_start_me].
    test_bars = int(ref_mei[test_end_me]) - int(ref_mei[test_start_me])

    # DEFENSIVE INVARIANT — the slice's stream length must equal warm-up + test bars.
    # It is guaranteed for clean shared-date data (the slice's month-ends are exactly the
    # global positions train_start_me..test_end_me), so a mismatch means the reference
    # spine slicing is off; fail loud rather than silently mis-slice the test span.
    if len(stream) != warmup_bars + test_bars:
        raise ValueError(
            f"fold stream length {len(stream)} != warmup_bars {warmup_bars} + "
            f"test_bars {test_bars}; reference-spine slicing is misaligned for fold "
            f"(train_start_me={train_start_me}, test_start_me={test_start_me}, "
            f"test_end_me={test_end_me})"
        )

    # Strip the warm-up prefix; what remains is exactly the test span's per-bar returns.
    return stream[warmup_bars:]


# ---------------------------------------------------------------------------
# FUNCTION 3 — equal-weight-basket buy-and-hold benchmark (per fold, test span).
# ---------------------------------------------------------------------------

def _benchmark_fold_returns(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    reference_symbol: str,
    train_start_me: int,
    test_start_me: int,
    test_end_me: int,
    lookback: int,
    cost_rate: float,
    price_field: str = "close",
    cash_per_bar_return: float = 0.0,
) -> np.ndarray:
    """The equal-weight-basket buy-and-hold benchmark for ONE fold's TEST SPAN.

    The TRUE always-hold-everything benchmark: equal-weight ALL basket symbols, no
    absolute filter, held across the test span, charged the SAME cost_rate.  We
    implement it by REUSING _rotation_fold_returns with top_n = len(bars_by_symbol)
    (select every symbol) and hold_when_all_negative=True (no absolute filter, so all
    symbols are held regardless of sign — a genuine buy-and-hold of the basket, NOT a
    momentum-filtered subset).  The SAME lookback, SAME fold positions, and SAME
    test-span extraction are used, so the benchmark and the strategy are scored on
    BYTE-IDENTICAL test spans by IDENTICAL code, differing ONLY in top_n and the filter
    flag — which is exactly what makes oos_* vs bh_* a fair edge-vs-beta comparison.

    The cost is COMPUTED, not assumed: with every symbol always held at 1/N, the
    weight vector is unchanged period-to-period, so test-span turnover (and thus cost)
    is ~zero — but rotation_backtest measures it explicitly rather than us asserting it.

    Args:
        bars_by_symbol:      symbol -> that symbol's time-ordered OHLCVBar list.
        reference_symbol:    the spine whose month-ends define the schedule/grid.
        train_start_me:      warm-up start month-end position (fold train start).
        test_start_me:       split month-end position (first test decision).
        test_end_me:         closing boundary month-end position.
        lookback:            SAME lookback as the strategy call, so the warm-up and
                             test-span extraction are byte-identical.
        cost_rate:           SAME per-unit-turnover cost fraction as the strategy.
        price_field:         "close" or "adj_close".
        cash_per_bar_return: per-bar return on any cash fraction (none here, since all
                             symbols are held, but forwarded for identical code).

    Returns:
        A float64 ndarray: the benchmark's test-span-only per-bar log-return stream.
    """
    # Reuse the strategy's per-fold producer with the two benchmark-defining overrides:
    #   top_n = len(bars_by_symbol)      -> select every basket symbol (equal-weight all)
    #   hold_when_all_negative = True    -> no absolute filter (true buy-and-hold, any sign)
    # Everything else (fold positions, lookback, cost_rate, price_field, cash) is passed
    # through unchanged so the benchmark test span aligns bar-for-bar with the strategy's.
    return _rotation_fold_returns(
        bars_by_symbol=bars_by_symbol,
        reference_symbol=reference_symbol,
        train_start_me=train_start_me,
        test_start_me=test_start_me,
        test_end_me=test_end_me,
        lookback=lookback,
        top_n=len(bars_by_symbol),
        cost_rate=cost_rate,
        price_field=price_field,
        cash_per_bar_return=cash_per_bar_return,
        hold_when_all_negative=True,
    )


# ---------------------------------------------------------------------------
# Minimal per-fold BacktestResult wrapper — the PREPEND-ZERO stitch-reuse crux.
# ---------------------------------------------------------------------------

def _fold_result(test_returns: np.ndarray) -> BacktestResult:
    """Wrap one fold's TEST-span returns in a minimal BacktestResult for _stitch_oos.

    THE PREPEND-ZERO CRUX: _stitch_oos does `fr.returns[1:]` on every fold, because an
    engine-produced returns array always has a STRUCTURAL zero at index 0 (bar 0 has no
    preceding signal to act on).  A rotation test-span stream has NO such leading zero —
    every element is already a real return.  So we PREPEND a single 0.0 to the test-span
    returns before building the result: _stitch_oos's [1:] then strips exactly that
    injected zero and keeps EVERY real rotation return.  WITHOUT this prepend, [1:] would
    wrongly strip the fold's FIRST real test return, silently dropping one observation
    per fold from the stitched OOS metrics.

    _stitch_oos reads ONLY fr.returns off the result (it concatenates fr.returns[1:] and
    derives everything else from that), so every OTHER field below is a safe placeholder:
    strategy_name, start_date, end_date, and all scalar metrics are never consulted.
    equity_curve is filled consistently (exp(cumsum(returns))) purely to keep the object
    internally honest; _stitch_oos does not read it either.

    Args:
        test_returns: this fold's test-span per-bar log returns (NO leading zero).

    Returns:
        A BacktestResult whose .returns is [0.0] + test_returns.
    """
    # Prepend the single structural zero so _stitch_oos's [1:] strips it and preserves
    # every real test-span return.  np.concatenate keeps the float64 dtype.
    returns = np.concatenate([[0.0], np.asarray(test_returns, dtype=np.float64)])

    # Equity curve consistent with those returns (log returns are additive; exp(cumsum)
    # recovers the multiplicative curve anchored at exp(0)=1.0).  Not read by _stitch_oos;
    # constructed only so the result object is internally coherent.
    equity_curve = np.exp(np.cumsum(returns))

    # Build the minimal result.  Only .returns matters to _stitch_oos; the rest are
    # honest placeholders (see docstring): fixed placeholder dates, n_bars from the
    # array length, and the default 0.0 scalar metrics.
    return BacktestResult(
        strategy_name="rotation-fold",   # placeholder label; never read by _stitch_oos
        start_date=_PLACEHOLDER_DT,       # placeholder; never read by _stitch_oos
        end_date=_PLACEHOLDER_DT,         # placeholder; never read by _stitch_oos
        n_bars=len(returns),              # honest length of the returns array
        equity_curve=equity_curve,        # internally-consistent; never read by _stitch_oos
        returns=returns,                  # THE field _stitch_oos consumes (with its [1:])
    )


# ---------------------------------------------------------------------------
# FUNCTION 4 — the verdict.
# ---------------------------------------------------------------------------

def rotation_walk_forward(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    reference_symbol: str,
    lookback: int,
    top_n: int,
    train_months: int,
    test_months: int,
    step_months: int | None = None,
    cost_rate: float = 0.0,
    price_field: str = "close",
    cash_per_bar_return: float = 0.0,
    hold_when_all_negative: bool = False,
    annualization_factor: int = 252,
) -> RotationVerdict:
    """Walk-forward-validate a FIXED-parameter rotation OOS vs an equal-weight benchmark.

    Builds month-end-aligned folds, produces each fold's rotation and benchmark test-span
    returns, wraps each (with a prepended structural zero) in a minimal BacktestResult,
    stitches both through the REUSED _stitch_oos, and returns a RotationVerdict.

    Args:
        bars_by_symbol:         symbol -> that symbol's time-ordered OHLCVBar list.
        reference_symbol:       the spine whose month-ends define the schedule/grid.
        lookback:               trailing formation window in month-ends.
        top_n:                  how many symbols the rotation holds (fixed weight denom).
        train_months:           TRAIN month-ends per fold.  Must be > lookback (see guard).
        test_months:            TEST decision month-ends per fold.
        step_months:            month-end advance between folds (None -> test_months).
        cost_rate:              per-unit-turnover transaction cost fraction.
        price_field:            "close" or "adj_close" — used for BOTH ranking and returns.
        cash_per_bar_return:    per-bar return on the cash fraction.
        hold_when_all_negative: absolute-filter switch for the STRATEGY (the benchmark
                                always uses True internally).
        annualization_factor:   bars per year for Sharpe/Sortino annualisation (252 default).

    Returns:
        A RotationVerdict with stitched OOS strategy metrics and stitched equal-weight
        buy-and-hold benchmark metrics over the identical test spans.

    Raises:
        ValueError: reference_symbol not in bars_by_symbol.
        ValueError: train_months <= lookback (warm-up cannot satisfy the lookback).
        ValueError: (propagated) too few month-ends, from rotation_month_end_folds.
    """
    # GUARD — the spine must be present; its month-ends ARE the schedule.  Echo the name,
    # mirroring rotation_backtest's own missing-reference guard.
    if reference_symbol not in bars_by_symbol:
        raise ValueError(
            f"reference_symbol {reference_symbol!r} not in bars_by_symbol "
            f"(keys: {sorted(bars_by_symbol.keys())})"
        )

    # GUARD — train_months must be STRICTLY greater than lookback, or the warm-up inside a
    # fold cannot satisfy the lookback by the test span's first (and entry-turnover)
    # decisions, and the test-span stream would diverge from a full-history rotation.  The
    # strictness (> not >=) is required so the warm-up decision immediately before the test
    # span also has a full lookback (see _rotation_fold_returns' warm-up-sufficiency note).
    if train_months <= lookback:
        raise ValueError(
            f"train_months must be > lookback so the fold warm-up satisfies the "
            f"lookback; got train_months={train_months}, lookback={lookback}"
        )

    # Build the month-end-aligned folds from the reference spine.  This propagates the
    # month-end-count / train_months / test_months / step_months guards.
    folds = rotation_month_end_folds(
        bars_by_symbol[reference_symbol],
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )

    # Accumulate one minimal BacktestResult per fold for the strategy and the benchmark,
    # in chronological (fold) order — the order _stitch_oos concatenates them in.
    strat_fold_results: list[BacktestResult] = []
    bh_fold_results: list[BacktestResult] = []

    # Per fold: produce the strategy and benchmark test-span returns and wrap each.
    for train_start_me, test_start_me, test_end_me in folds:
        # Rotation strategy test-span returns for this fold (warmed inside the train span).
        strat_returns = _rotation_fold_returns(
            bars_by_symbol=bars_by_symbol,
            reference_symbol=reference_symbol,
            train_start_me=train_start_me,
            test_start_me=test_start_me,
            test_end_me=test_end_me,
            lookback=lookback,
            top_n=top_n,
            cost_rate=cost_rate,
            price_field=price_field,
            cash_per_bar_return=cash_per_bar_return,
            hold_when_all_negative=hold_when_all_negative,
        )

        # Equal-weight-basket buy-and-hold test-span returns over the IDENTICAL fold span.
        bh_returns = _benchmark_fold_returns(
            bars_by_symbol=bars_by_symbol,
            reference_symbol=reference_symbol,
            train_start_me=train_start_me,
            test_start_me=test_start_me,
            test_end_me=test_end_me,
            lookback=lookback,
            cost_rate=cost_rate,
            price_field=price_field,
            cash_per_bar_return=cash_per_bar_return,
        )

        # Wrap each (prepending the structural zero so _stitch_oos's [1:] keeps every real
        # return) and append in fold order.
        strat_fold_results.append(_fold_result(strat_returns))
        bh_fold_results.append(_fold_result(bh_returns))

    # Stitch the STRATEGY folds through the reused scoring seam.  _stitch_oos returns
    # (oos_returns, oos_equity_curve, oos_total_return, oos_sharpe, oos_max_drawdown,
    # oos_sortino) — we keep the four scalars (drop the two arrays with `_`).
    _, _, oos_total_return, oos_sharpe, oos_max_drawdown, oos_sortino = _stitch_oos(
        strat_fold_results, annualization_factor
    )

    # Stitch the BENCHMARK folds through the SAME seam, SAME annualization — only the
    # per-fold returns differ (all-held equal-weight vs the strategy's selection).
    _, _, bh_total_return, bh_sharpe, bh_max_drawdown, bh_sortino = _stitch_oos(
        bh_fold_results, annualization_factor
    )

    # Assemble and return the immutable verdict.  No overfitting-tax field (deferred to B).
    return RotationVerdict(
        n_folds=len(folds),
        oos_sharpe=oos_sharpe,
        oos_sortino=oos_sortino,
        oos_max_drawdown=oos_max_drawdown,
        oos_total_return=oos_total_return,
        bh_sharpe=bh_sharpe,
        bh_sortino=bh_sortino,
        bh_max_drawdown=bh_max_drawdown,
        bh_total_return=bh_total_return,
    )


# ===========================================================================
# STEP B — per-fold hyperparameter search + overfitting tax (ADDITIVE).
#
# Everything BELOW is new and purely additive: it REUSES the Day-43 functions
# above (rotation_month_end_folds, _rotation_fold_returns, _benchmark_fold_returns,
# _fold_result, rotation_walk_forward) by CALLING them, and reuses summarize_tax /
# TaxRow from the TSMOM CLI.  None of the Day-43 code is modified.
# ===========================================================================


# ---------------------------------------------------------------------------
# RotationSearchVerdict — the searched (per-fold-tuned) OOS verdict + the tax.
# ---------------------------------------------------------------------------

# frozen=True → immutable computed fact, matching RotationVerdict / WalkForwardResult.
@dataclass(frozen=True)
class RotationSearchVerdict:
    """Per-fold-searched rotation OOS metrics, a fixed baseline, a benchmark, and the tax.

    Parallel to the TSMOM TaxRow world: for each outer fold the grid search picks the
    (top_n, lookback) with the best IN-SAMPLE (train-span, after-cost) Sharpe, applies the
    winner to the untouched TEST span, and the tax = mean per-fold in-sample Sharpe −
    stitched fitted OOS Sharpe (computed by the REUSED summarize_tax, so the number is
    directly comparable to the momentum tax).

    The fixed_oos_* baseline is the Day-43 rotation_walk_forward run with a FIXED
    (top_n, lookback) = the grid's first sorted candidate, so this verdict carries the
    same fixed-vs-fitted-vs-B&H-vs-in-sample-vs-tax shape the TSMOM TaxRow does.
    """

    # Number of (train, test) folds actually run (== len of the per-fold lists below).
    n_folds: int

    # Per-fold winners, in chronological (fold) order — chosen_top_n[i]/chosen_lookback[i]
    # are the candidate the search picked on fold i by best in-sample Sharpe.
    chosen_top_n: list[int]
    chosen_lookback: list[int]

    # Per-fold best IN-SAMPLE (train-span, after-cost) Sharpe of the chosen candidate, in
    # fold order.  in_sample_sharpes[i] is the score the search MAXIMISED on fold i's train.
    in_sample_sharpes: list[float]

    # Searched (per-fold-tuned) strategy, stitched OOS aggregates over the disjoint tests.
    fitted_oos_sharpe: float        # annualised Sharpe over the stitched fitted OOS returns
    fitted_oos_sortino: float       # annualised Sortino over the same stitched returns
    fitted_oos_max_drawdown: float  # worst peak-to-trough of the stitched fitted OOS curve
    fitted_oos_total_return: float  # total return over the stitched fitted OOS timeline

    # Fixed-parameter baseline OOS Sharpe (Day-43 rotation_walk_forward, first sorted
    # candidate) — the "does per-fold tuning even beat the static config?" reference.
    fixed_oos_sharpe: float

    # Equal-weight-basket buy-and-hold, stitched over the IDENTICAL test spans (unchanged
    # by the search — the benchmark holds everything and is lookback-invariant).
    bh_sharpe: float                # annualised B&H Sharpe over the stitched benchmark returns
    bh_sortino: float               # annualised B&H Sortino over the same stitched returns
    bh_max_drawdown: float          # worst peak-to-trough of the stitched B&H equity curve
    bh_total_return: float          # total return over the stitched B&H timeline

    # The tax scalars, taken VERBATIM off the reused summarize_tax result (see below).
    mean_in_sample_sharpe: float    # mean(in_sample_sharpes) — what the optimiser "saw"
    tax: float                      # mean_in_sample_sharpe − fitted_oos_sharpe (the gap)

    # The full TaxRow itself, stored for convenience so callers can print the exact same
    # row the TSMOM CLI prints without re-deriving it.
    tax_row: TaxRow


# ---------------------------------------------------------------------------
# FUNCTION 5 — the TRAIN-span in-sample stream (the no-leak crux).
# ---------------------------------------------------------------------------

def _rotation_train_span_returns(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    reference_symbol: str,
    train_start_me: int,
    test_start_me: int,
    lookback: int,
    top_n: int,
    cost_rate: float,
    price_field: str = "close",
    cash_per_bar_return: float = 0.0,
    hold_when_all_negative: bool = False,
) -> np.ndarray:
    """Produce a candidate's IN-SAMPLE rotation stream, scored on the TRAIN span ONLY.

    THE NO-LEAK CRUX.  We DELEGATE to the Day-43 _rotation_fold_returns with an INNER
    fold triple that lives entirely inside the outer fold's TRAIN span:

        inner_train_start_me = train_start_me            # warm-up starts at the fold train start
        inner_test_start_me  = train_start_me + lookback + 1   # in-sample scoring starts here
        inner_test_end_me    = test_start_me             # closing boundary = the OUTER split boundary

    Why NO test-span return can enter the in-sample score: the inner scoring window's LAST
    holding period is (D_{inner_test_end_me - 1}, D_{inner_test_end_me}] = (..., D_{test_start_me}],
    and its ranking is taken AS OF D_{test_start_me - 1}.  Both endpoints are <= D_{test_start_me}
    (the OUTER split boundary), so the entire in-sample stream uses ONLY train-span data
    [train_start_me, test_start_me) plus the shared closing boundary D_{test_start_me}.  The
    outer TEST returns are strictly AFTER D_{test_start_me}, so none of them is touched.  Using
    inner_test_end_me = outer test_start_me is the SAME boundary-sharing Day 43 uses between
    consecutive folds (fold i's closing edge is fold i+1's opening decision).

    Why the warm-up is satisfied: inner_test_start_me = train_start_me + lookback + 1 makes the
    inner TRAIN span exactly lookback + 1 month-ends long, i.e. > lookback, which satisfies
    _rotation_fold_returns' warm-up-sufficiency contract (the first in-sample decision AND the
    decision immediately before it both have a full `lookback` of prior month-ends in the slice).

    Args:
        bars_by_symbol:         symbol -> that symbol's time-ordered OHLCVBar list.
        reference_symbol:       the spine whose month-ends define the schedule/grid.
        train_start_me:         outer fold's train-start month-end position (inner warm-up start).
        test_start_me:          outer fold's split month-end position (inner CLOSING boundary).
        lookback:               candidate trailing formation window in month-ends.
        top_n:                  candidate number of symbols to hold.
        cost_rate:              per-unit-turnover transaction cost fraction (in-sample IS costed,
                                matching the after-cost objective the design locks).
        price_field:            "close" or "adj_close" — used for BOTH ranking and returns.
        cash_per_bar_return:    per-bar return on the cash fraction.
        hold_when_all_negative: absolute-filter switch passed through to the ranker.

    Returns:
        A float64 ndarray: the candidate's in-sample (train-span-only) per-bar log-return stream.
    """
    # The inner warm-up starts at the outer fold's train start — the left edge of the slice.
    inner_train_start_me = train_start_me

    # The inner in-sample scoring starts one past a full lookback of warm-up month-ends, so the
    # inner train span (lookback + 1 month-ends) strictly exceeds the lookback (warm-up satisfied).
    inner_test_start_me = train_start_me + lookback + 1

    # The inner CLOSING boundary is the OUTER split boundary: the last in-sample holding period
    # ends at D_{test_start_me}, whose ranking used D_{test_start_me - 1} — both train-span-side.
    inner_test_end_me = test_start_me

    # Delegate to the UNCHANGED Day-43 producer; its returned stream is exactly the in-sample
    # test span of this inner fold, i.e. the candidate's train-span per-bar returns.
    return _rotation_fold_returns(
        bars_by_symbol=bars_by_symbol,
        reference_symbol=reference_symbol,
        train_start_me=inner_train_start_me,
        test_start_me=inner_test_start_me,
        test_end_me=inner_test_end_me,
        lookback=lookback,
        top_n=top_n,
        cost_rate=cost_rate,
        price_field=price_field,
        cash_per_bar_return=cash_per_bar_return,
        hold_when_all_negative=hold_when_all_negative,
    )


# ---------------------------------------------------------------------------
# FUNCTION 6 — the per-fold grid search over (top_n, lookback).
# ---------------------------------------------------------------------------

def _search_fold(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    reference_symbol: str,
    train_start_me: int,
    test_start_me: int,
    test_end_me: int,
    candidates: list[tuple[int, int]],
    cost_rate: float,
    annualization_factor: int,
    price_field: str = "close",
    cash_per_bar_return: float = 0.0,
    hold_when_all_negative: bool = False,
) -> tuple[int, int, float]:
    """Grid-search (top_n, lookback) on ONE outer fold's TRAIN span; return the best.

    For each candidate we build its in-sample train-span stream via
    _rotation_train_span_returns and score it with metrics.sharpe_ratio on the SAME
    annualization the OOS side uses (identical footing → honest tax).  The argmax
    in-sample-Sharpe candidate wins.

    DETERMINISTIC TIEBREAK (locked): candidates are iterated in sorted order by
    (lookback, top_n) ascending, and the running best is updated only on a STRICTLY
    GREATER in-sample Sharpe, so the FIRST candidate achieving the maximum wins.  Ties
    therefore resolve to the smallest lookback, then the smallest top_n — the winner is
    reproducible run-to-run.

    Args:
        bars_by_symbol:         symbol -> that symbol's time-ordered OHLCVBar list.
        reference_symbol:       the spine whose month-ends define the schedule/grid.
        train_start_me:         this fold's train-start month-end position.
        test_start_me:          this fold's split month-end position.
        test_end_me:            this fold's closing-boundary month-end position (unused here
                                but accepted so the signature mirrors the fold triple; the
                                in-sample search never reads test-span data).
        candidates:             list of (top_n, lookback) tuples to search over.
        cost_rate:              per-unit-turnover cost fraction (in-sample IS costed).
        annualization_factor:   bars per year for the Sharpe used to rank candidates.
        price_field:            "close" or "adj_close".
        cash_per_bar_return:    per-bar return on the cash fraction.
        hold_when_all_negative: absolute-filter switch passed through to the ranker.

    Returns:
        (best_top_n, best_lookback, best_in_sample_sharpe) — the fold's chosen candidate
        and the in-sample Sharpe the search maximised.
    """
    # Sort candidates by (lookback, top_n) ascending so the deterministic tiebreak below
    # (first-max-wins) resolves ties to the smallest lookback, then the smallest top_n.
    ordered = sorted(candidates, key=lambda c: (c[1], c[0]))

    # Running winner: start with a sentinel Sharpe below any real value so the first
    # candidate always installs itself, then only strictly-greater scores replace it.
    best_top_n: int = ordered[0][0]
    best_lookback: int = ordered[0][1]
    best_sharpe: float = float("-inf")

    # Evaluate every candidate on THIS fold's train span, in the sorted order.
    for cand_top_n, cand_lookback in ordered:
        # In-sample train-span stream for this candidate (no test-span data touched).
        stream = _rotation_train_span_returns(
            bars_by_symbol=bars_by_symbol,
            reference_symbol=reference_symbol,
            train_start_me=train_start_me,
            test_start_me=test_start_me,
            lookback=cand_lookback,
            top_n=cand_top_n,
            cost_rate=cost_rate,
            price_field=price_field,
            cash_per_bar_return=cash_per_bar_return,
            hold_when_all_negative=hold_when_all_negative,
        )

        # Score it on the SAME function/units the OOS side uses — identical footing.
        cand_sharpe = metrics.sharpe_ratio(stream, annualization_factor)

        # STRICTLY-GREATER update: the first candidate reaching the max stays the winner,
        # so ties resolve to the smallest (lookback, top_n) already fixed by the sort.
        if cand_sharpe > best_sharpe:
            best_sharpe = cand_sharpe
            best_top_n = cand_top_n
            best_lookback = cand_lookback

    # Return the fold's chosen candidate and the in-sample Sharpe it maximised.
    return best_top_n, best_lookback, best_sharpe


# ---------------------------------------------------------------------------
# FUNCTION 7 — the searched walk-forward verdict + the overfitting tax.
# ---------------------------------------------------------------------------

def rotation_walk_forward_search(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    reference_symbol: str,
    candidates: list[tuple[int, int]],
    train_months: int,
    test_months: int,
    step_months: int | None = None,
    cost_rate: float = 0.0,
    price_field: str = "close",
    cash_per_bar_return: float = 0.0,
    hold_when_all_negative: bool = False,
    annualization_factor: int = 252,
    symbol_label: str = "rotation",
) -> RotationSearchVerdict:
    """Walk-forward-validate a per-fold-SEARCHED rotation and report the overfitting tax.

    Per outer fold: grid-search (top_n, lookback) for best IN-SAMPLE (train-span, after-cost)
    Sharpe, apply the winner to the untouched TEST span (the Day-43 _rotation_fold_returns
    path), and stitch the winners' test spans into the fitted OOS aggregates.  The benchmark
    is the Day-43 equal-weight buy-and-hold over the IDENTICAL folds (lookback-invariant, so
    the search does not change it).  The tax is computed by the REUSED summarize_tax, so it is
    the SAME quantity (mean in-sample − stitched fitted OOS) the TSMOM tax reports.

    Args:
        bars_by_symbol:         symbol -> that symbol's time-ordered OHLCVBar list.
        reference_symbol:       the spine whose month-ends define the schedule/grid.
        candidates:             non-empty list of (top_n, lookback) tuples to search per fold.
        train_months:           TRAIN month-ends per fold.  Must be > L_max + 1 (see guard).
        test_months:            TEST decision month-ends per fold.
        step_months:            month-end advance between folds (None -> test_months).
        cost_rate:              per-unit-turnover transaction cost fraction.
        price_field:            "close" or "adj_close" — used for BOTH ranking and returns.
        cash_per_bar_return:    per-bar return on the cash fraction.
        hold_when_all_negative: absolute-filter switch for the STRATEGY (benchmark uses True).
        annualization_factor:   bars per year for Sharpe/Sortino annualisation (252 default).
        symbol_label:           label threaded into the TaxRow (a label, not a data lookup).

    Returns:
        A RotationSearchVerdict with per-fold winners, fitted OOS aggregates, the fixed-parameter
        baseline OOS Sharpe, the benchmark aggregates, and the overfitting tax.

    Raises:
        ValueError: candidates empty.
        ValueError: train_months <= L_max + 1 (largest candidate lookback leaves no in-sample
                    scoring month-ends).
        ValueError: reference_symbol not in bars_by_symbol.
        ValueError: (propagated) too few month-ends, from rotation_month_end_folds.
    """
    # GUARD 1 — an empty grid has nothing to search; echo the (empty) value.
    if not candidates:
        raise ValueError(f"candidates must be non-empty, got {candidates!r}")

    # Largest candidate lookback — the warm-up ceiling every fold's inner search must clear.
    l_max = max(lookback for _top_n, lookback in candidates)

    # GUARD 2 — the L_max warm-up guard.  For the largest candidate lookback the inner in-sample
    # span is train_months − l_max − 1 month-ends; if train_months <= l_max + 1 that is <= 0, so
    # there are NO in-sample scoring month-ends.  Require train_months >= l_max + 2 (recommend
    # >= l_max + 3 for a meaningful in-sample Sharpe); state required vs given.
    if train_months <= l_max + 1:
        raise ValueError(
            f"train_months={train_months} leaves no in-sample scoring month-ends for the "
            f"grid's largest candidate lookback L_max={l_max}: need train_months >= L_max + 2 "
            f"= {l_max + 2} (recommend >= L_max + 3 = {l_max + 3} for a meaningful in-sample "
            f"Sharpe); got {train_months}"
        )

    # GUARD 3 — the spine must be present; its month-ends ARE the schedule.  Echo the name,
    # mirroring rotation_walk_forward / rotation_backtest.
    if reference_symbol not in bars_by_symbol:
        raise ValueError(
            f"reference_symbol {reference_symbol!r} not in bars_by_symbol "
            f"(keys: {sorted(bars_by_symbol.keys())})"
        )

    # Build the month-end-aligned folds from the reference spine (propagates the month-end-count
    # / train_months / test_months / step_months guards from rotation_month_end_folds).
    folds = rotation_month_end_folds(
        bars_by_symbol[reference_symbol],
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )

    # The grid's FIRST sorted candidate (smallest lookback, then smallest top_n) — reused for
    # BOTH the fixed-parameter baseline AND the lookback-invariant benchmark below.
    first_candidate = sorted(candidates, key=lambda c: (c[1], c[0]))[0]
    baseline_top_n, baseline_lookback = first_candidate

    # Per-fold accumulators, in chronological (fold) order — the order _stitch_oos concatenates.
    chosen_top_n: list[int] = []
    chosen_lookback: list[int] = []
    in_sample_sharpes: list[float] = []
    strat_fold_results: list[BacktestResult] = []
    bh_fold_results: list[BacktestResult] = []

    # Per fold: search the train span, apply the winner to the test span, produce the benchmark.
    for train_start_me, test_start_me, test_end_me in folds:
        # Grid-search this fold's TRAIN span for the best in-sample candidate.
        best_top_n, best_lookback, best_in_sample = _search_fold(
            bars_by_symbol=bars_by_symbol,
            reference_symbol=reference_symbol,
            train_start_me=train_start_me,
            test_start_me=test_start_me,
            test_end_me=test_end_me,
            candidates=candidates,
            cost_rate=cost_rate,
            annualization_factor=annualization_factor,
            price_field=price_field,
            cash_per_bar_return=cash_per_bar_return,
            hold_when_all_negative=hold_when_all_negative,
        )

        # Record the fold's winner and the in-sample Sharpe it maximised.
        chosen_top_n.append(best_top_n)
        chosen_lookback.append(best_lookback)
        in_sample_sharpes.append(best_in_sample)

        # Apply the winner to the UNTOUCHED test span via the Day-43 producer (unchanged path).
        strat_returns = _rotation_fold_returns(
            bars_by_symbol=bars_by_symbol,
            reference_symbol=reference_symbol,
            train_start_me=train_start_me,
            test_start_me=test_start_me,
            test_end_me=test_end_me,
            lookback=best_lookback,
            top_n=best_top_n,
            cost_rate=cost_rate,
            price_field=price_field,
            cash_per_bar_return=cash_per_bar_return,
            hold_when_all_negative=hold_when_all_negative,
        )

        # Benchmark for this fold via the Day-43 producer with a FIXED lookback (the grid's
        # smallest).  The benchmark holds EVERY symbol equal-weight regardless of rank, so its
        # stream is lookback-invariant — any valid lookback yields the same benchmark returns.
        bh_returns = _benchmark_fold_returns(
            bars_by_symbol=bars_by_symbol,
            reference_symbol=reference_symbol,
            train_start_me=train_start_me,
            test_start_me=test_start_me,
            test_end_me=test_end_me,
            lookback=baseline_lookback,
            cost_rate=cost_rate,
            price_field=price_field,
            cash_per_bar_return=cash_per_bar_return,
        )

        # Wrap each (prepending the structural zero so _stitch_oos's [1:] keeps every real return).
        strat_fold_results.append(_fold_result(strat_returns))
        bh_fold_results.append(_fold_result(bh_returns))

    # Stitch the SEARCHED strategy folds through the reused scoring seam (fitted OOS aggregates).
    _, _, fitted_oos_total_return, fitted_oos_sharpe, fitted_oos_max_drawdown, fitted_oos_sortino = _stitch_oos(
        strat_fold_results, annualization_factor
    )

    # Stitch the BENCHMARK folds through the SAME seam, SAME annualization.
    _, _, bh_total_return, bh_sharpe, bh_max_drawdown, bh_sortino = _stitch_oos(
        bh_fold_results, annualization_factor
    )

    # Fixed-parameter baseline: the Day-43 rotation_walk_forward with the grid's first sorted
    # candidate, over the SAME folds/cost/basis.  Its oos_sharpe is the fixed_oos reference —
    # the "does per-fold tuning beat the static config?" bar, parallel to the TSMOM fixed track.
    fixed_verdict = rotation_walk_forward(
        bars_by_symbol=bars_by_symbol,
        reference_symbol=reference_symbol,
        lookback=baseline_lookback,
        top_n=baseline_top_n,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        cost_rate=cost_rate,
        price_field=price_field,
        cash_per_bar_return=cash_per_bar_return,
        hold_when_all_negative=hold_when_all_negative,
        annualization_factor=annualization_factor,
    )

    # Compute the tax via the REUSED summarize_tax — the IDENTICAL formula/guard the TSMOM CLI
    # uses (mean per-fold in-sample Sharpe − stitched fitted OOS Sharpe), so the number is
    # directly comparable.  bh is the stitched benchmark Sharpe.
    tax_row = summarize_tax(
        symbol_label,
        len(folds),
        fixed_verdict.oos_sharpe,
        fitted_oos_sharpe,
        bh_sharpe,
        in_sample_sharpes,
    )

    # Assemble and return the immutable searched verdict, storing the TaxRow's mean/tax verbatim.
    return RotationSearchVerdict(
        n_folds=len(folds),
        chosen_top_n=chosen_top_n,
        chosen_lookback=chosen_lookback,
        in_sample_sharpes=in_sample_sharpes,
        fitted_oos_sharpe=fitted_oos_sharpe,
        fitted_oos_sortino=fitted_oos_sortino,
        fitted_oos_max_drawdown=fitted_oos_max_drawdown,
        fitted_oos_total_return=fitted_oos_total_return,
        fixed_oos_sharpe=fixed_verdict.oos_sharpe,
        bh_sharpe=bh_sharpe,
        bh_sortino=bh_sortino,
        bh_max_drawdown=bh_max_drawdown,
        bh_total_return=bh_total_return,
        mean_in_sample_sharpe=tax_row.mean_in_sample_sharpe,
        tax=tax_row.tax,
        tax_row=tax_row,
    )
