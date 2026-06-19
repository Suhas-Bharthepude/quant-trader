# scripts/overfitting_tax.py

"""
CLI: per-fold Optuna tuning vs the fixed SMA(50,200) baseline across the ETF basket,
reporting the "overfitting tax" — how much of the optimizer's in-sample Sharpe
fails to survive out-of-sample.

Run via:
    uv run python scripts/overfitting_tax.py
    uv run python scripts/overfitting_tax.py --symbols SPY TLT --n-trials 10
    uv run python scripts/overfitting_tax.py --train 252 --test 63 --seed 7

For each symbol the same (train, test) splits are scored TWICE:
  * fixed  — walk_forward_validate with a static SMA(50,200) (no fit_fn). This is
             the Day-23/24 baseline track record.
  * fitted — walk_forward_validate with make_sma_optuna_fit_fn, which re-tunes the
             (fast, slow) windows per fold by maximizing warm-only in-sample Sharpe.
Both share the SAME splits object, so fixed and fitted are scored on byte-identical
windows — the only difference is whether the strategy is static or per-fold tuned.

The question this answers: does TUNING the SMA per fold beat the fixed baseline
out-of-sample, or does the optimizer just fool itself in-sample?  The overfitting
tax (mean per-fold in-sample Sharpe − stitched fitted OOS Sharpe) quantifies that
self-deception per symbol.

This is a thin I/O layer: bar loading and split construction mirror
src/research/compare_walkforward.py exactly (same cli_common helpers, same
walk_forward_splits call); the only pure logic — summarize_tax — lives below and
is unit-tested in tests/test_overfitting_tax.py.
"""

# argparse is the standard-library CLI parser — same pattern as
# compare_walkforward.py and the rest of the research CLIs.
import argparse

# logging mirrors compare_walkforward.py: per-symbol skip/error warnings go to
# log.warning (stderr-ish) so they do not corrupt the structured stdout table.
import logging

# sys.exit() propagates main()'s integer return value to the shell as $?.
import sys

# mean computes the per-symbol in-sample average and the cross-symbol aggregate
# means; statistics.mean raises on an empty list, so callers guard for empty.
from statistics import mean

# dataclass auto-generates __init__/__repr__/__eq__ for TaxRow; frozen=True makes
# each row immutable — a computed summary is a fact, mutating it is a bug.
from dataclasses import dataclass

# SMACrossoverStrategy is the baseline (and the strategy the fitter tunes). The
# fitted run still passes one positional strategy, which walk_forward_validate
# ignores when fit_fn is given — we pass SMA(50,200) so the call reads identically.
from src.strategies.sma_crossover import SMACrossoverStrategy

# build_symbol_list / load_bars_for_symbols are the shared CLI helpers
# compare_walkforward.py uses — reused here verbatim so symbol resolution and
# DuckDB reads stay identical across the two CLIs.
from src.research.cli_common import build_symbol_list, load_bars_for_symbols

# walk_forward_splits slices bars into (train, test) folds; walk_forward_validate
# scores them and (via fit_fn) optionally re-tunes per fold.  Same two functions
# compare_walkforward.py orchestrates.
from src.research.walk_forward import walk_forward_splits, walk_forward_validate

# make_sma_optuna_fit_fn builds the per-fold tuning fit_fn plugged into the seam.
from src.research.optuna_fit import make_sma_optuna_fit_fn


# Module-level logger, matching compare_walkforward.py's convention.
log = logging.getLogger(__name__)


# Universe is fixed to etf_basket — the stable long-history basket this analysis
# targets.  Unlike compare_walkforward there is no --universe flag: the tax
# comparison is specifically about this basket, and --symbols still overrides.
_UNIVERSE = "etf_basket"

# Bar load window.  The etf_basket deep backfill runs 2008→present; this wide
# range captures all of it so train=504/test=126 yields the full fold count.
# Hard-coded (not date.today()) so runs are reproducible day-to-day, matching
# compare_walkforward.py's rationale for fixed default dates.
_START = "2000-01-01"
_END = "2026-12-31"


# ---------------------------------------------------------------------------
# Pure helper — testable without a WalkForwardResult.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaxRow:
    """One symbol's fixed-vs-fitted summary plus the overfitting tax."""

    symbol: str
    n_folds: int
    fixed_oos_sharpe: float
    fitted_oos_sharpe: float
    bh_sharpe: float
    mean_in_sample_sharpe: float
    tax: float


def summarize_tax(
    symbol: str,
    n_folds: int,
    fixed_oos: float,
    fitted_oos: float,
    bh: float,
    in_sample_sharpes: list[float],
) -> TaxRow:
    """Assemble a TaxRow from plain scalars (no WalkForwardResult dependency).

    Takes scalars rather than a WalkForwardResult so it is trivially testable
    with hand-built numbers — the orchestration in main() does the unpacking.

    Args:
        symbol:            Ticker this row describes.
        n_folds:           Number of (train, test) folds scored.
        fixed_oos:         Stitched OOS Sharpe of the fixed SMA(50,200) baseline.
        fitted_oos:        Stitched OOS Sharpe of the per-fold tuned strategy.
        bh:                Stitched buy-and-hold Sharpe over the same windows.
        in_sample_sharpes: Per-fold warm-only in-sample Sharpes (one per fold)
                           recorded by the fitter; may be empty if no fold ran.

    Returns:
        A frozen TaxRow.
    """
    # Guard the empty case explicitly: statistics.mean([]) raises, and a symbol
    # that produced no folds (or whose fitter recorded nothing) should report a
    # 0.0 in-sample rather than crash the whole basket sweep.
    mean_in_sample = mean(in_sample_sharpes) if in_sample_sharpes else 0.0

    # WHY tax = in-sample − fitted OOS:
    #   The optimizer SELECTS each fold's params to maximize warm-only in-sample
    #   Sharpe, so mean_in_sample is the performance it "saw" while choosing.
    #   fitted_oos is what those same params actually delivered on the untouched
    #   test windows.  The difference is the gap the optimizer fooled itself by:
    #   a large positive tax means the in-sample view was flattering and most of
    #   the apparent edge evaporated out-of-sample (classic overfitting); a tax
    #   near zero means the in-sample estimate generalized; a negative tax means
    #   OOS beat in-sample (luck / regime shift, not skill).
    tax = mean_in_sample - fitted_oos

    return TaxRow(
        symbol=symbol,
        n_folds=n_folds,
        fixed_oos_sharpe=fixed_oos,
        fitted_oos_sharpe=fitted_oos,
        bh_sharpe=bh,
        mean_in_sample_sharpe=mean_in_sample,
        tax=tax,
    )


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point — parse args, run fixed + fitted walk-forward per symbol, print tax."""

    # ------------------------------------------------------------------
    # 1. Parse CLI arguments.
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Per-fold Optuna tuning vs fixed SMA(50,200): the overfitting tax across the ETF basket."
    )

    # --n-trials: Optuna trials per fold.  Default 30 matches the fitter's own
    # default — a reasonable fit-quality/speed trade-off for the 2-param search.
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials per fold (default: 30)",
    )

    # --seed: TPESampler seed.  Fixing it makes the whole sweep reproducible.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optuna TPESampler seed (default: 42)",
    )

    # --train / --test: fold window sizes.  Defaults 504/126 (≈2y train, ≈6mo
    # test) match the existing baseline walk-forward run on the basket.
    parser.add_argument(
        "--train",
        type=int,
        default=504,
        help="Training window size in bars per fold (default: 504 ≈ 2 years)",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=126,
        help="Test window size in bars per fold (default: 126 ≈ 6 months)",
    )

    # --symbols: explicit ticker list overriding the etf_basket default.
    # nargs="*" so "SPY TLT" parses as two tokens; empty/omitted → full basket.
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Explicit symbols (e.g. --symbols SPY TLT); default: all etf_basket",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Resolve the symbol list (etf_basket unless --symbols overrides).
    # ------------------------------------------------------------------

    # build_symbol_list expects a CSV string or None; --symbols is a token list,
    # so join it back to CSV.  The large limit returns the full basket on the
    # universe path (etf_basket has 17 < 1000); it is ignored on the --symbols path.
    symbols_csv = ",".join(args.symbols) if args.symbols else None
    symbols = build_symbol_list(symbols_csv, _UNIVERSE, limit=1000)

    if not symbols:
        print("ERROR: Symbol list is empty. Check --symbols.")
        return 1

    # ------------------------------------------------------------------
    # 3. Load bars for every symbol (one DuckDB connection).
    # ------------------------------------------------------------------

    print(f"Loading bars for {len(symbols)} symbols...")
    bars_dict = load_bars_for_symbols(symbols, _START, _END)

    if not bars_dict:
        print(
            f"ERROR: No bars found for any symbol in {_START}–{_END}. "
            "Run the ingest script first."
        )
        return 1

    # ------------------------------------------------------------------
    # 4. Print the run note + header.
    # ------------------------------------------------------------------

    # One-line caveat up front so the operator reads the Tax column correctly:
    # the three OOS/B&H columns are stitched track-record Sharpes (one number per
    # symbol), while In-sample is the MEAN of per-fold warm-only train Sharpes —
    # each fold fits its own params, so there is no single stitched in-sample
    # number.  Tax is therefore a directional gap, not a strict like-for-like.
    print(
        "Note: Fixed/Fitted/B&H are stitched OOS track-record Sharpes; "
        "In-sample is the mean of per-fold warm-only train Sharpes "
        "(each fold fits its own params, so there is no single stitched "
        "in-sample). Tax = In-sample − Fitted OOS is a directional gap, "
        "not a strict like-for-like."
    )
    print(
        f"Running fixed SMA(50,200) vs per-fold Optuna fit "
        f"(n_trials={args.n_trials}, seed={args.seed}, "
        f"train={args.train}, test={args.test}) on {len(bars_dict)} symbols...\n"
    )

    # ------------------------------------------------------------------
    # 5. Per-symbol: build splits ONCE, score fixed + fitted, summarize.
    # ------------------------------------------------------------------

    rows: list[TaxRow] = []

    for symbol, bars in bars_dict.items():
        # Wrap each symbol so one failure (too-short history, fitter error, …)
        # logs a warning and continues rather than aborting the whole basket.
        try:
            # Build splits ONCE and reuse for both runs so fixed and fitted are
            # scored on byte-identical windows.
            splits = walk_forward_splits(
                bars,
                train_size=args.train,
                test_size=args.test,
                # step=None → walk_forward_splits defaults to test_size (non-overlapping).
            )

            # Fixed baseline: no fit_fn → the static SMA(50,200) on every fold.
            fixed = walk_forward_validate(splits, SMACrossoverStrategy(50, 200))

            # Fitted: per-fold Optuna tuning.  The positional strategy is ignored
            # when fit_fn is given (we pass SMA(50,200) only so the call mirrors
            # the fixed one).  record captures {fast, slow, in_sample_sharpe} per
            # fold, in fold order.  annualization stays at the 252 default on BOTH
            # the validator and the fit_fn so in-sample and OOS Sharpes share units.
            record: list[dict] = []
            fitted = walk_forward_validate(
                splits,
                SMACrossoverStrategy(50, 200),
                fit_fn=make_sma_optuna_fit_fn(
                    n_trials=args.n_trials,
                    seed=args.seed,
                    record=record,
                ),
            )

            row = summarize_tax(
                symbol,
                fitted.n_folds,
                fixed.oos_sharpe,
                fitted.oos_sharpe,
                fitted.bh_sharpe,
                [r["in_sample_sharpe"] for r in record],
            )
            rows.append(row)

            # Progress line as each symbol finishes — a multi-minute run needs to
            # show life.  Goes to stdout above the final table; the table is
            # re-printed clean at the end so this scroll is just live feedback.
            print(
                f"  {row.symbol:<5}  folds={row.n_folds:<3}  "
                f"fixed={row.fixed_oos_sharpe:+.2f}  "
                f"fitted={row.fitted_oos_sharpe:+.2f}  "
                f"B&H={row.bh_sharpe:+.2f}  "
                f"in-sample={row.mean_in_sample_sharpe:+.2f}  "
                f"tax={row.tax:+.2f}"
            )

        except Exception as exc:  # noqa: BLE001 — one bad symbol must not abort the basket
            log.warning("Skipping %s: %s", symbol, exc)
            continue

    # Guard: every symbol was skipped.
    if not rows:
        print(
            f"\nERROR: Every symbol was skipped — no symbol had enough bars for "
            f"train={args.train} + test={args.test} = {args.train + args.test} "
            "bars minimum. Reduce window sizes or ingest more data."
        )
        return 1

    # ------------------------------------------------------------------
    # 6. Final aligned table.
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    header = (
        f"{'Symbol':<8}{'Folds':>6}  {'Fixed OOS':>10}  {'Fitted OOS':>11}  "
        f"{'B&H':>8}  {'In-sample':>10}  {'Tax':>8}"
    )
    print(header)
    print("-" * 78)
    for r in rows:
        print(
            f"{r.symbol:<8}{r.n_folds:>6}  "
            f"{r.fixed_oos_sharpe:>+10.2f}  {r.fitted_oos_sharpe:>+11.2f}  "
            f"{r.bh_sharpe:>+8.2f}  {r.mean_in_sample_sharpe:>+10.2f}  "
            f"{r.tax:>+8.2f}"
        )
    print("=" * 78)

    # ------------------------------------------------------------------
    # 7. Aggregate summary across symbols.
    # ------------------------------------------------------------------

    # Cross-symbol means of each column — a one-line read of "on average, does
    # fitting help?"  These average the per-symbol stitched Sharpes (and the
    # per-symbol in-sample means), so they are an average of summaries, not a
    # re-stitch across symbols; fine for a directional basket-level read.
    mean_fixed = mean(r.fixed_oos_sharpe for r in rows)
    mean_fitted = mean(r.fitted_oos_sharpe for r in rows)
    mean_bh = mean(r.bh_sharpe for r in rows)
    mean_in_sample = mean(r.mean_in_sample_sharpe for r in rows)
    mean_tax = mean(r.tax for r in rows)

    # Counts: how often fitting actually won, vs the fixed baseline and vs just
    # holding the asset.  These are the bottom-line "did tuning manufacture edge?"
    # tallies — the honest expectation is both stay low.
    n_fitted_beats_fixed = sum(1 for r in rows if r.fitted_oos_sharpe > r.fixed_oos_sharpe)
    n_fitted_beats_bh = sum(1 for r in rows if r.fitted_oos_sharpe > r.bh_sharpe)
    n = len(rows)

    print(
        f"{'MEAN':<8}{'':>6}  "
        f"{mean_fixed:>+10.2f}  {mean_fitted:>+11.2f}  "
        f"{mean_bh:>+8.2f}  {mean_in_sample:>+10.2f}  "
        f"{mean_tax:>+8.2f}"
    )
    print("=" * 78)
    print(
        f"Fitted OOS beat fixed OOS on {n_fitted_beats_fixed}/{n} symbols; "
        f"fitted OOS beat buy-and-hold on {n_fitted_beats_bh}/{n} symbols."
    )
    print()

    return 0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
