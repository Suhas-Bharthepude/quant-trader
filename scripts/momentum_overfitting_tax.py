# scripts/momentum_overfitting_tax.py

"""
CLI: the momentum analogue of scripts/overfitting_tax.py — per-fold Optuna tuning
of the TimeSeriesMomentum lookback vs the fixed TSMOM(12) baseline across the ETF
basket, reporting the "overfitting tax" (how much of the optimizer's in-sample
Sharpe fails to survive out-of-sample).

Run via:
    uv run python scripts/momentum_overfitting_tax.py
    uv run python scripts/momentum_overfitting_tax.py --symbols SPY TLT --n-trials 10
    uv run python scripts/momentum_overfitting_tax.py --train 756 --test 252 --seed 7

For each symbol the same (train, test) splits are scored TWICE, through ONE shared
adj_close Backtester:
  * fixed  — walk_forward_validate with a static TSMOM(12) (no fit_fn). This is the
             fixed momentum verdict baseline (scripts/momentum_walkforward.py).
  * fitted — walk_forward_validate with make_tsmom_optuna_fit_fn, which re-tunes the
             lookback per fold by maximizing warm-only in-sample Sharpe.
Both share the SAME splits object AND the SAME frictionless adj_close Backtester,
so fixed and fitted are scored on byte-identical windows and the identical
total-return basis — the ONLY difference is static-vs-tuned lookback.

The basis is adj_close TOTAL return (frictionless: no fee/slippage/cash-yield), so
these numbers are directly comparable to the fixed momentum verdict in
scripts/momentum_walkforward.py.  The question this answers: does TUNING the
lookback per fold beat the fixed TSMOM(12) out-of-sample, or does the optimizer
just fool itself in-sample?  The overfitting tax (mean per-fold in-sample Sharpe −
stitched fitted OOS Sharpe) quantifies that self-deception per symbol.
"""

# argparse is the standard-library CLI parser — same pattern as overfitting_tax.py.
import argparse

# logging mirrors overfitting_tax.py: per-symbol skip/error warnings go to
# log.warning (stderr-ish) so they do not corrupt the structured stdout table.
import logging

# sys.exit() propagates main()'s integer return value to the shell as $?.
import sys

# mean computes the cross-symbol aggregate means; statistics.mean raises on an
# empty list, so the aggregate block runs only after the all-skipped guard.
from statistics import mean

# Backtester is the scoring engine: this sibling constructs exactly ONE
# (frictionless, adj_close) and passes it to both walk_forward_validate calls.
# overfitting_tax.py does NOT import this (it rides the backtester=None default);
# the momentum sibling MUST, to pin the adj_close total-return basis.
from src.backtest.engine import Backtester

# TimeSeriesMomentumStrategy is the fixed baseline (lookback=12) and the strategy
# the fitter tunes.  Constructed on the adj_close basis to match the engine.
from src.strategies.time_series_momentum import TimeSeriesMomentumStrategy

# build_symbol_list / load_bars_for_symbols are the shared CLI helpers — reused
# verbatim so symbol resolution and DuckDB reads stay identical across CLIs.
from src.research.cli_common import build_symbol_list, load_bars_for_symbols

# walk_forward_splits slices bars into (train, test) folds; walk_forward_validate
# scores them and (via fit_fn) optionally re-tunes per fold.  Same two functions
# overfitting_tax.py orchestrates.
from src.research.walk_forward import walk_forward_splits, walk_forward_validate

# make_tsmom_optuna_fit_fn builds the per-fold lookback-tuning fit_fn plugged into
# the seam — the momentum analogue of make_sma_optuna_fit_fn.
from src.research.optuna_fit import make_tsmom_optuna_fit_fn

# REUSE the pure, already-tested tax helper + row by import rather than redefining
# them: this keeps ONE authoritative definition of the tax math (TaxRow and
# summarize_tax are strategy-agnostic — plain scalars in, frozen row out).  scripts
# is importable here, as tests/test_overfitting_tax.py already relies on.
from scripts.overfitting_tax import TaxRow, summarize_tax


# Module-level logger, matching overfitting_tax.py's convention.
log = logging.getLogger(__name__)


# Universe is fixed to etf_basket — the stable long-history basket this analysis
# targets.  --symbols still overrides; there is no --universe flag, matching
# overfitting_tax.py / momentum_walkforward.py.
_UNIVERSE = "etf_basket"

# Bar load window.  The etf_basket deep backfill runs 2008→present; this wide
# range captures all of it.  Hard-coded (not date.today()) so runs are
# reproducible day-to-day, matching the sibling CLIs' rationale.
_START = "2000-01-01"
_END = "2026-12-31"

# The verdict basis: adj_close TOTAL return.  Used to construct the one Backtester
# AND threaded into both the fixed strategy and the fitter, so engine basis and
# strategy basis agree by construction (Day-32 guard stays inert).
_PRICE_FIELD = "adj_close"

# The fixed baseline lookback in months — TSMOM(12), matching the fixed momentum
# verdict in scripts/momentum_walkforward.py so the two are directly comparable.
_LOOKBACK = 12


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point — parse args, run fixed + fitted momentum walk-forward, print tax."""

    # ------------------------------------------------------------------
    # 1. Parse CLI arguments.
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Per-fold Optuna lookback tuning vs fixed TSMOM(12): the momentum overfitting tax across the ETF basket."
    )

    # --n-trials: Optuna trials per fold.  Default 30 matches the fitter's own
    # default — ample for the single-parameter lookback search.
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

    # --train / --test: fold window sizes.  Defaults 756/252 (≈3y train, ≈1y test)
    # — NOT the SMA tax's 504/126 — because the monthly 12-month-lookback strategy
    # needs ~3y train to warm the lookback with margin and ~1y test (matching the
    # verdict's windows so the tax is comparable to the fixed momentum verdict).
    parser.add_argument(
        "--train",
        type=int,
        default=756,
        help="Training window size in bars per fold (default: 756 ≈ 3 years)",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=252,
        help="Test window size in bars per fold (default: 252 ≈ 1 year)",
    )

    # --lookback-low / --lookback-high: inclusive MONTH bounds for the tuned
    # lookback, exposed so the search range is visible and overridable.  Defaults
    # (3, 18) match make_tsmom_optuna_fit_fn's own default lookback_range.
    parser.add_argument(
        "--lookback-low",
        type=int,
        default=3,
        help="Inclusive low month bound for the tuned lookback (default: 3)",
    )
    parser.add_argument(
        "--lookback-high",
        type=int,
        default=18,
        help="Inclusive high month bound for the tuned lookback (default: 18)",
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
    # 4. Construct the ONE shared Backtester (frictionless adj_close).
    # ------------------------------------------------------------------

    # This single frictionless adj_close engine is passed as backtester= to BOTH
    # the fixed and fitted walk_forward_validate calls below, so:
    #   (a) both score on the SAME total-return (adj_close) basis the verdict used,
    #       not the close price-return basis the backtester=None default would give;
    #   (b) the engine basis matches the strategy basis by construction (both
    #       adj_close), keeping the Day-32 basis-consistency guard inert; and
    #   (c) it stays FRICTIONLESS (no fee/slippage/cash-yield) DELIBERATELY: the
    #       tax must isolate parameter-selection overfitting, not cost drag —
    #       exactly as the SMA tax does on its close-basis default engine.
    engine = Backtester(price_field=_PRICE_FIELD)

    # ------------------------------------------------------------------
    # 5. Print the run note + header.
    # ------------------------------------------------------------------

    # One-line caveat up front so the operator reads the Tax column correctly:
    # the three OOS/B&H columns are stitched track-record Sharpes (one number per
    # symbol), while In-sample is the MEAN of per-fold warm-only train Sharpes —
    # each fold fits its own lookback, so there is no single stitched in-sample
    # number.  Tax is therefore a directional gap, not a strict like-for-like.
    print(
        "Note: Fixed/Fitted/B&H are stitched OOS track-record Sharpes; "
        "In-sample is the mean of per-fold warm-only train Sharpes "
        "(each fold fits its own lookback, so there is no single stitched "
        "in-sample). Tax = In-sample − Fitted OOS is a directional gap, "
        "not a strict like-for-like."
    )
    print(
        f"Running fixed TSMOM({_LOOKBACK}) vs per-fold Optuna lookback fit "
        f"(adj_close, frictionless, n_trials={args.n_trials}, seed={args.seed}, "
        f"train={args.train}, test={args.test}, "
        f"lookback=[{args.lookback_low},{args.lookback_high}]) "
        f"on {len(bars_dict)} symbols...\n"
    )

    # ------------------------------------------------------------------
    # 6. Per-symbol: build splits ONCE, score fixed + fitted, summarize.
    # ------------------------------------------------------------------

    rows: list[TaxRow] = []

    for symbol, bars in bars_dict.items():
        # Wrap each symbol so one failure (too-short history, fitter error, …)
        # logs a warning and continues rather than aborting the whole basket.
        try:
            # Build splits ONCE and reuse for both runs so fixed and fitted are
            # scored on byte-identical windows.  step=None → test_size (non-overlapping).
            splits = walk_forward_splits(
                bars,
                train_size=args.train,
                test_size=args.test,
            )

            # Fixed baseline: no fit_fn → static TSMOM(12) on the adj_close basis,
            # scored through the shared frictionless engine.
            fixed = walk_forward_validate(
                splits,
                TimeSeriesMomentumStrategy(_LOOKBACK, price_field=_PRICE_FIELD),
                backtester=engine,
            )

            # Fitted: per-fold Optuna lookback tuning.  The positional strategy is
            # ignored when fit_fn is given (passed only so the call reads like the
            # fixed one).  The fitter threads price_field=_PRICE_FIELD so its
            # internal engine AND each trial's strategy match the shared engine's
            # basis; record captures {lookback, in_sample_sharpe} per fold, in fold
            # order.  annualization stays at the 252 default on BOTH the validator
            # and the fitter so in-sample and OOS Sharpes share units.
            record: list[dict] = []
            fitted = walk_forward_validate(
                splits,
                TimeSeriesMomentumStrategy(_LOOKBACK, price_field=_PRICE_FIELD),
                backtester=engine,
                fit_fn=make_tsmom_optuna_fit_fn(
                    n_trials=args.n_trials,
                    seed=args.seed,
                    lookback_range=(args.lookback_low, args.lookback_high),
                    price_field=_PRICE_FIELD,
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
    # 7. Final aligned table.
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
    # 8. Aggregate summary across symbols.
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
