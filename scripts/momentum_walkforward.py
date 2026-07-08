# scripts/momentum_walkforward.py

"""
CLI: the FIXED time-series-momentum walk-forward verdict on the etf_basket.

The one question this answers: does monthly 12-month-lookback TSMOM beat simply
buying and holding, OUT OF SAMPLE, on an honest TOTAL-RETURN (adj_close) basis,
after optional transaction costs and cash-on-flat interest?  For each symbol the
strategy is run with a FIXED lookback over rolling (train, test) folds and scored
on the stitched out-of-sample timeline, then compared to the built-in always-long
buy-and-hold benchmark measured over the identical test windows.

This is the FIXED-strategy verdict only: there is NO per-fold parameter search
here.  Per-fold Optuna tuning and the resulting "overfitting tax" (in-sample vs
OOS gap) are a SEPARATE, later runner; mixing them in would conflate "is there a
fixed edge?" with "does tuning manufacture one?"  Keeping this runner fixed makes
the headline number an honest track record, not an optimizer's self-portrait.

Cash-on-flat caveat: the buy-and-hold benchmark is ALWAYS-LONG, so it is never
flat and therefore earns ZERO flat-yield by construction.  TSMOM, by contrast,
sits in cash during its FLAT periods, so a non-zero --cash-yield lifts the
strategy's OOS return relative to B&H.  That is a LEGITIMATE economic effect
(idle capital really does earn interest), not a thumb on the scale — but because
it only moves one side of the comparison, --cash-yield defaults to 0.0 so the
headline verdict makes no cash-rate assumption.  Re-run at ~0.04 to bracket it.

Run via:
    uv run python scripts/momentum_walkforward.py
    uv run python scripts/momentum_walkforward.py --symbols SPY,TLT
    uv run python scripts/momentum_walkforward.py --cash-yield 0.04 --fee-bps 2
"""

# argparse is the standard-library CLI parser — same pattern as
# overfitting_tax.py and compare_walkforward.py across the research CLIs.
import argparse

# logging mirrors the sibling CLIs: per-symbol skip/error warnings go to
# log.warning (stderr-ish) so they do not corrupt the structured stdout table.
import logging

# sys.exit() propagates main()'s integer return value to the shell as $?.
import sys

# mean computes the cross-symbol aggregate means for the MEAN row; statistics.mean
# raises on an empty list, so the loop guards against an empty rows list first.
from statistics import mean

# dataclass auto-generates __init__/__repr__/__eq__ for VerdictRow; frozen=True
# makes each row immutable — a computed summary is a fact, mutating it is a bug.
from dataclasses import dataclass

# TimeSeriesMomentumStrategy is the FIXED strategy under test; its constructor
# takes (lookback, price_field) and we pass the SAME price_field as the engine.
from src.strategies.time_series_momentum import TimeSeriesMomentumStrategy

# Backtester is the scoring engine; we construct exactly ONE per symbol carrying
# the price basis, cash yield, and costs that both the strategy run and the
# built-in B&H inherit through bt.run.
from src.backtest.engine import Backtester

# build_symbol_list / load_bars_for_symbols are the shared CLI helpers the other
# research CLIs use — reused verbatim so symbol resolution and DuckDB reads stay
# identical across runners.
from src.research.cli_common import build_symbol_list, load_bars_for_symbols

# walk_forward_splits slices bars into (train, test) folds; walk_forward_validate
# scores them (fixed strategy here — no fit_fn) and returns the WalkForwardResult
# with stitched OOS metrics and the always-long B&H benchmark.
from src.research.walk_forward import walk_forward_splits, walk_forward_validate


# Module-level logger, matching the sibling CLIs' convention.
log = logging.getLogger(__name__)


# Universe is fixed to etf_basket — the stable long-history basket this verdict
# targets.  Like overfitting_tax.py (and unlike compare_walkforward.py) there is
# no --universe flag: this analysis is specifically about this basket; --symbols
# still overrides for a narrower run.
_UNIVERSE = "etf_basket"

# Bar load window.  The etf_basket deep backfill runs 2008→present (~4638 bars
# per symbol), so this range captures all of it and yields the full fold count at
# train=756/test=252.  Hard-coded (not date.today()) so runs are reproducible
# day-to-day, matching overfitting_tax.py's rationale for fixed default dates.
_START = "2008-01-01"
_END = "2026-12-31"


# ---------------------------------------------------------------------------
# Pure helper — testable without DuckDB or a WalkForwardResult.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerdictRow:
    """One symbol's fixed-TSMOM-vs-B&H summary plus the headline deltas."""

    # Ticker this row describes.
    symbol: str
    # Number of (train, test) folds scored for this symbol.
    n_folds: int
    # Stitched OOS Sharpe of the fixed TSMOM strategy.
    oos_sharpe: float
    # Stitched OOS Sortino of the fixed TSMOM strategy (downside-only denom).
    oos_sortino: float
    # Stitched buy-and-hold Sharpe over the identical test windows.
    bh_sharpe: float
    # Stitched buy-and-hold Sortino over the identical test windows.
    bh_sortino: float
    # Stitched OOS max drawdown of TSMOM (POSITIVE fraction = magnitude).
    oos_max_dd: float
    # Stitched buy-and-hold max drawdown (POSITIVE fraction = magnitude).
    bh_max_dd: float
    # Stitched OOS total return of TSMOM (fraction; 0.25 == +25%).
    oos_return: float
    # Stitched buy-and-hold total return over the same windows (fraction).
    bh_return: float
    # oos_sharpe - bh_sharpe: positive means TSMOM had the better risk-adj return.
    sharpe_delta: float
    # oos_sortino - bh_sortino: positive means TSMOM had the better downside-adjusted return.
    sortino_delta: float
    # bh_max_dd - oos_max_dd: POSITIVE means TSMOM had the SMALLER drawdown
    # (cut risk vs holding) — see the sign-convention comment in summarize_verdict.
    dd_reduction: float


def summarize_verdict(
    symbol: str,
    n_folds: int,
    oos_sharpe: float,
    oos_sortino: float,
    bh_sharpe: float,
    bh_sortino: float,
    oos_max_dd: float,
    bh_max_dd: float,
    oos_return: float,
    bh_return: float,
) -> VerdictRow:
    """Assemble a VerdictRow from plain scalars (no WalkForwardResult dependency).

    Takes scalars rather than a WalkForwardResult so it is trivially testable with
    hand-built numbers — the orchestration in main() does the unpacking.  Same
    design as overfitting_tax.summarize_tax.

    Args:
        symbol:      Ticker this row describes.
        n_folds:     Number of (train, test) folds scored.
        oos_sharpe:  Stitched OOS Sharpe of the fixed TSMOM strategy.
        oos_sortino: Stitched OOS Sortino of the fixed TSMOM strategy.
        bh_sharpe:   Stitched buy-and-hold Sharpe over the same windows.
        bh_sortino:  Stitched buy-and-hold Sortino over the same windows.
        oos_max_dd:  Stitched OOS max drawdown (positive fraction).
        bh_max_dd:   Stitched buy-and-hold max drawdown (positive fraction).
        oos_return:  Stitched OOS total return (fraction).
        bh_return:   Stitched buy-and-hold total return (fraction).

    Returns:
        A frozen VerdictRow.
    """
    # sharpe_delta is the risk-adjusted edge: how much TSMOM's stitched OOS Sharpe
    # exceeds (or trails, if negative) just holding the asset over the same windows.
    sharpe_delta = oos_sharpe - bh_sharpe

    # sortino_delta is the downside-adjusted edge: how much TSMOM's stitched OOS
    # Sortino exceeds (or trails, if negative) just holding the asset over the same
    # windows — the Sharpe delta's twin using a downside-only risk denominator.
    sortino_delta = oos_sortino - bh_sortino

    # SIGN CONVENTION (important): max_dd is stored throughout this codebase as a
    # POSITIVE fraction — the peak-to-trough magnitude — so a LARGER number is a
    # WORSE drawdown.  dd_reduction = bh_max_dd - oos_max_dd is therefore POSITIVE
    # when TSMOM's drawdown was SMALLER than B&H's, i.e. momentum CUT risk (the
    # whole thesis of trend-following: sidestep the worst declines by going flat).
    # A NEGATIVE dd_reduction means TSMOM drew down MORE than simply holding —
    # momentum added drawdown rather than cutting it.
    dd_reduction = bh_max_dd - oos_max_dd

    # Build the immutable row; all fields passed by keyword so field order is
    # irrelevant and each value is self-documenting at the call site.
    return VerdictRow(
        symbol=symbol,
        n_folds=n_folds,
        oos_sharpe=oos_sharpe,
        oos_sortino=oos_sortino,
        bh_sharpe=bh_sharpe,
        bh_sortino=bh_sortino,
        oos_max_dd=oos_max_dd,
        bh_max_dd=bh_max_dd,
        oos_return=oos_return,
        bh_return=bh_return,
        sharpe_delta=sharpe_delta,
        sortino_delta=sortino_delta,
        dd_reduction=dd_reduction,
    )


# ---------------------------------------------------------------------------
# Thin per-symbol runner — hermetically testable on synthetic bars.
# ---------------------------------------------------------------------------


def run_one_symbol(
    bars,
    lookback: int,
    train: int,
    test: int,
    step: int | None,
    price_field: str,
    cash_yield: float,
    fee_bps: float,
    slippage_bps: float,
):
    """Run a FIXED-TSMOM walk-forward for one symbol and return its WalkForwardResult.

    Wires splits → engine → strategy → validator on a BASIS-MATCHED run: the same
    price_field is fed to both the Backtester and the strategy, so the engine's
    scoring basis and the strategy's signal basis agree by construction.

    Args:
        bars:         Time-ordered list of OHLCVBar for this one symbol.
        lookback:     TSMOM formation window in months.
        train:        Training window size in bars per fold.
        test:         Test window size in bars per fold.
        step:         Window advance between folds (None → splits uses test_size).
        price_field:  "close" or "adj_close" — fed to BOTH engine and strategy.
        cash_yield:   Annual interest earned on idle (flat) capital.
        fee_bps:      Per-trade fee in basis points.
        slippage_bps: Per-trade slippage in basis points.

    Returns:
        The WalkForwardResult from walk_forward_validate (fixed strategy, no fit_fn).
    """
    # Slice this symbol's bars into rolling (train, test) folds.  step=None lets
    # walk_forward_splits default to test_size (non-overlapping test windows).
    splits = walk_forward_splits(bars, train_size=train, test_size=test, step=step)

    # Construct EXACTLY ONE Backtester.  This is THE single source of the price
    # basis AND the cash yield (and costs) for this symbol: both the strategy run
    # and the built-in always-long B&H benchmark inherit them through bt.run, so
    # the strategy and its benchmark are scored on byte-identical engine config.
    bt = Backtester(
        price_field=price_field,
        annual_cash_yield=cash_yield,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )

    # Construct the strategy passing the SAME price_field variable used for the
    # engine above.  Because both sides read from ONE variable, the engine's
    # price_field and the strategy's price_field are equal BY CONSTRUCTION — the
    # Day-32 basis-consistency guard in walk_forward_validate is only a backstop
    # here, never the mechanism; it can fire only if a future edit desyncs them.
    strat = TimeSeriesMomentumStrategy(lookback=lookback, price_field=price_field)

    # Score the FIXED strategy over every fold (no fit_fn → static strategy on
    # each fold).  walk_forward_validate warms the lookback inside the train
    # window: with train>=756 (~3y) the FLAT warmup (~the first 252 bars) is fully
    # consumed before the test window starts, so no warmup bleeds into the OOS
    # measurement and TSMOM's own ">lookback month-end observations" guard cannot
    # trip on a properly-sized train window.
    return walk_forward_validate(splits, strat, backtester=bt)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point — parse args, run fixed TSMOM walk-forward per symbol, print verdict."""

    # ------------------------------------------------------------------
    # 1. Parse CLI arguments.
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Fixed time-series-momentum walk-forward verdict vs buy-and-hold on the ETF basket."
    )

    # --lookback: TSMOM formation window in months.  Default 12 is the canonical
    # Moskowitz/Ooi/Pedersen horizon and TimeSeriesMomentumStrategy's own default.
    parser.add_argument(
        "--lookback",
        type=int,
        default=12,
        help="TSMOM formation window in months (default: 12)",
    )

    # --train: training window size in bars per fold.  Default 756 (~3 years)
    # comfortably warms the 12-month lookback with margin to spare.
    parser.add_argument(
        "--train",
        type=int,
        default=756,
        help="Training window size in bars per fold (default: 756 ≈ 3 years)",
    )

    # --test: test window size in bars per fold.  Default 252 (~1 year) gives
    # roughly 12 monthly rebalances per out-of-sample fold.
    parser.add_argument(
        "--test",
        type=int,
        default=252,
        help="Test window size in bars per fold (default: 252 ≈ 1 year)",
    )

    # --step: window advance between folds.  Default None → walk_forward_splits
    # uses test_size, producing non-overlapping (cleanly tiled) test windows.
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Window advance between folds in bars (default: test-size, non-overlapping)",
    )

    # --price-field: the return basis.  Default adj_close gives the honest
    # TOTAL-return basis (dividends/splits folded in, verified in DuckDB); close
    # gives price-only return.  choices reject anything else at parse time.
    parser.add_argument(
        "--price-field",
        type=str,
        default="adj_close",
        choices=["close", "adj_close"],
        help="Return basis: adj_close = total return (default), close = price return",
    )

    # --cash-yield: annual interest on idle cash while FLAT.  Default 0.0 so the
    # headline verdict makes no cash-rate assumption; re-run at ~0.04 to bracket
    # the effect (B&H is always-long so this only lifts the strategy side).
    parser.add_argument(
        "--cash-yield",
        type=float,
        default=0.0,
        help="Annual interest on idle (flat) cash as a fraction (default: 0.0; try 0.04)",
    )

    # --fee-bps: per-trade fee in basis points (1 bp = 0.0001 of turnover).
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=0.0,
        help="Per-trade fee in basis points (default: 0.0)",
    )

    # --slippage-bps: per-trade slippage in basis points, charged on turnover.
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Per-trade slippage in basis points (default: 0.0)",
    )

    # --symbols: comma-separated override of the etf_basket default.  None →
    # the full basket; "SPY,TLT" → just those two.
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help='Comma-separated symbols (e.g. "SPY,TLT"); default: all etf_basket',
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Resolve the symbol list (etf_basket unless --symbols overrides).
    # ------------------------------------------------------------------

    # build_symbol_list takes a CSV string or None; the large limit returns the
    # full basket on the universe path (etf_basket has 17 < 1000) and is ignored
    # on the --symbols path.
    symbols = build_symbol_list(args.symbols, _UNIVERSE, limit=1000)

    # An empty symbol list means there is nothing to run — fail with a clear
    # message rather than letting a downstream component produce a cryptic error.
    if not symbols:
        print("ERROR: Symbol list is empty. Check --symbols.")
        return 1

    # ------------------------------------------------------------------
    # 3. Load bars for every symbol (one DuckDB connection).
    # ------------------------------------------------------------------

    # One-line progress message before the I/O so the operator sees what is
    # happening while DuckDB opens and reads.
    print(f"Loading bars for {len(symbols)} symbols...")

    # load_bars_for_symbols opens DuckDB once, reads all symbols, and logs a
    # warning for any symbol with no data.  Returns a dict in input symbol order.
    bars_dict = load_bars_for_symbols(symbols, _START, _END)

    # Guard: every symbol missing from DuckDB → nothing to run.
    if not bars_dict:
        print(
            f"ERROR: No bars found for any symbol in {_START}–{_END}. "
            "Run the ingest script first."
        )
        return 1

    # ------------------------------------------------------------------
    # 4. Print the run-parameter banner.
    # ------------------------------------------------------------------

    # Echo every parameter that shapes the numbers so the operator can verify the
    # inputs before the per-symbol output begins — reproducibility starts with
    # being able to read back exactly what was run.
    print(
        f"Fixed TSMOM(lookback={args.lookback}) walk-forward vs buy-and-hold "
        f"on {len(bars_dict)} symbols "
        f"(train={args.train}, test={args.test}, step={args.step}, "
        f"basis={args.price_field}, cash_yield={args.cash_yield}, "
        f"fee_bps={args.fee_bps}, slippage_bps={args.slippage_bps})...\n"
    )

    # ------------------------------------------------------------------
    # 5. Per-symbol loop — run fixed TSMOM, summarize, print progress.
    # ------------------------------------------------------------------

    rows: list[VerdictRow] = []

    for symbol, bars in bars_dict.items():
        # WIDE try/except wrapping the WHOLE per-symbol body (splits AND
        # run_one_symbol).  Unlike compare_walkforward.py — which only guards the
        # splits call — TSMOM's month-end guard raises INSIDE generate_signals
        # (i.e. inside walk_forward_validate), not just in walk_forward_splits.
        # So a short symbol, or a lowered --train that fails to clear the
        # lookback's month-end requirement, must skip that symbol via log.warning
        # and continue rather than abort the whole basket.  Catch the generic
        # Exception as a backstop, matching overfitting_tax.py's pattern.
        try:
            # Run the fixed-strategy walk-forward for this symbol.  ONE Backtester
            # and a basis-matched strategy are built inside run_one_symbol.
            result = run_one_symbol(
                bars,
                lookback=args.lookback,
                train=args.train,
                test=args.test,
                step=args.step,
                price_field=args.price_field,
                cash_yield=args.cash_yield,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
            )

            # Build the summary row from the WalkForwardResult's exposed fields.
            # ONLY documented fields are read — no invented metrics.
            row = summarize_verdict(
                symbol,
                result.n_folds,
                result.oos_sharpe,
                result.oos_sortino,
                result.bh_sharpe,
                result.bh_sortino,
                result.oos_max_drawdown,
                result.bh_max_drawdown,
                result.oos_total_return,
                result.bh_return,
            )
            rows.append(row)

            # Progress line as each symbol finishes — a multi-minute basket run
            # needs to show life.  Goes to stdout above the final table; the table
            # is re-printed clean at the end so this scroll is just live feedback.
            print(
                f"  {row.symbol:<5}  folds={row.n_folds:<3}  "
                f"oos_sharpe={row.oos_sharpe:+.2f}  "
                f"bh_sharpe={row.bh_sharpe:+.2f}  "
                f"Δsharpe={row.sharpe_delta:+.2f}  "
                f"Δsortino={row.sortino_delta:+.2f}  "
                f"oos_dd={row.oos_max_dd * 100:.1f}%  "
                f"bh_dd={row.bh_max_dd * 100:.1f}%  "
                f"dd_cut={row.dd_reduction * 100:+.1f}%"
            )

        except Exception as exc:  # noqa: BLE001 — one bad symbol must not abort the basket
            log.warning("Skipping %s: %s", symbol, exc)
            continue

    # ------------------------------------------------------------------
    # 6. Guard: every symbol was skipped.
    # ------------------------------------------------------------------

    if not rows:
        print(
            f"\nERROR: Every symbol was skipped — no symbol had enough bars for "
            f"train={args.train} + test={args.test} = {args.train + args.test} "
            "bars minimum (and enough month-ends to clear the lookback). "
            "Reduce window sizes or ingest more data."
        )
        return 1

    # ------------------------------------------------------------------
    # 7. Final aligned table.
    # ------------------------------------------------------------------

    print()
    # Rule widths bumped 86 → 112 to span the two added Sortino columns
    # (2 × (11-char field + 2-char gap) = +26 chars over the prior 7-column table).
    print("=" * 112)
    header = (
        f"{'Symbol':<8}{'Folds':>6}  {'OOS Sharpe':>11}  {'OOS Sortino':>11}  "
        f"{'B&H Sharpe':>11}  {'B&H Sortino':>11}  "
        f"{'ΔSharpe':>8}  {'OOS MaxDD':>10}  {'B&H MaxDD':>10}  {'DD cut':>8}"
    )
    print(header)
    print("-" * 112)
    for r in rows:
        print(
            f"{r.symbol:<8}{r.n_folds:>6}  "
            f"{r.oos_sharpe:>+11.2f}  {r.oos_sortino:>+11.2f}  "
            f"{r.bh_sharpe:>+11.2f}  {r.bh_sortino:>+11.2f}  "
            f"{r.sharpe_delta:>+8.2f}  "
            f"{r.oos_max_dd * 100:>9.1f}%  {r.bh_max_dd * 100:>9.1f}%  "
            f"{r.dd_reduction * 100:>+7.1f}%"
        )
    print("=" * 112)

    # ------------------------------------------------------------------
    # 8. MEAN row + bottom-line tallies.
    # ------------------------------------------------------------------

    # Cross-symbol means of each column — a one-line read of "on average, does
    # fixed momentum help?"  These average the per-symbol stitched metrics, so
    # they are an average of summaries, fine for a directional basket-level read.
    mean_oos_sharpe = mean(r.oos_sharpe for r in rows)
    mean_oos_sortino = mean(r.oos_sortino for r in rows)
    mean_bh_sharpe = mean(r.bh_sharpe for r in rows)
    mean_bh_sortino = mean(r.bh_sortino for r in rows)
    mean_sharpe_delta = mean(r.sharpe_delta for r in rows)
    mean_oos_dd = mean(r.oos_max_dd for r in rows)
    mean_bh_dd = mean(r.bh_max_dd for r in rows)
    mean_dd_reduction = mean(r.dd_reduction for r in rows)

    # Tallies: how often momentum actually won on each axis.  n_beat_sharpe counts
    # better risk-adjusted return; n_cut_dd counts a SMALLER drawdown than holding
    # (dd_reduction > 0).  These are the bottom-line "did momentum earn its keep?"
    # counts — the honest expectation is that they are modest, not universal.
    n = len(rows)
    n_beat_sharpe = sum(1 for r in rows if r.oos_sharpe > r.bh_sharpe)
    # n_beat_sortino mirrors n_beat_sharpe on the downside-only axis — the direct
    # read of whether the Sortino verdict differs from the Sharpe verdict.
    n_beat_sortino = sum(1 for r in rows if r.oos_sortino > r.bh_sortino)
    n_cut_dd = sum(1 for r in rows if r.dd_reduction > 0)

    print(
        f"{'MEAN':<8}{'':>6}  "
        f"{mean_oos_sharpe:>+11.2f}  {mean_oos_sortino:>+11.2f}  "
        f"{mean_bh_sharpe:>+11.2f}  {mean_bh_sortino:>+11.2f}  "
        f"{mean_sharpe_delta:>+8.2f}  "
        f"{mean_oos_dd * 100:>9.1f}%  {mean_bh_dd * 100:>9.1f}%  "
        f"{mean_dd_reduction * 100:>+7.1f}%"
    )
    print("=" * 112)
    print(
        f"Momentum beat B&H on Sharpe on {n_beat_sharpe}/{n}; "
        f"on Sortino on {n_beat_sortino}/{n}; "
        f"cut max drawdown on {n_cut_dd}/{n}."
    )
    print()

    # ------------------------------------------------------------------
    # 9. Return success.
    # ------------------------------------------------------------------

    return 0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
