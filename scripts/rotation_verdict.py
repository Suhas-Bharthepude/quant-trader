# scripts/rotation_verdict.py

"""
CLI: the real-basket cross-sectional ROTATION verdict.

Runs the per-fold-searched rotation walk-forward (rotation_walk_forward_search) ONCE on the
real 17-ETF basket loaded from DuckDB and prints the honest, after-costs, out-of-sample answer
to the only question that matters: does cross-sectional rotation (hold the top-N momentum ETFs,
rebalanced monthly) actually beat just holding the equal-weight basket?

The printed verdict is the fixed-vs-fitted-vs-B&H OOS Sharpes plus the OVERFITTING TAX — the gap
between the mean per-fold in-sample Sharpe the grid search "saw" while choosing parameters and
the stitched fitted OOS Sharpe those choices actually delivered on the untouched test spans. A
positive tax, a tie against buy-and-hold, or a loss are all LEGITIMATE findings: a negative result
delivered honestly is the product, and this script is built not to flatter the strategy.

Run via:
    uv run python scripts/rotation_verdict.py
    uv run python scripts/rotation_verdict.py --test-months 6      # documented robustness cross-check
    uv run python scripts/rotation_verdict.py --symbols SPY QQQ TLT GLD

LOCKED CONFIG: the grid (_GRID), the windows (--train-months / --test-months defaults), and the
cost (_COST_RATE) were chosen BEFORE this run and are NOT to be retuned after seeing the results —
retuning toward a nicer tax is exactly the overfitting this whole system exists to expose. The
flags exist only for the documented robustness cross-check (e.g. --test-months 6), not for tuning.

This is a thin I/O layer: it only READS bars (via the cli_common helpers, identical to
overfitting_tax.py) and CALLS the already-tested rotation_walk_forward_search. It adds NO
production logic — every fold/scoring/tax computation lives in src/research/rotation_verdict.py
and is unit-tested there.
"""

# argparse is the standard-library CLI parser — same pattern as overfitting_tax.py and the rest
# of the research CLIs, so flag handling stays consistent across the scripts/ layer.
import argparse

# sys.exit() propagates main()'s integer return value to the shell as $? for scripting/CI use.
import sys

# build_symbol_list / load_bars_for_symbols are the shared CLI helpers overfitting_tax.py uses —
# reused here verbatim so symbol resolution and DuckDB reads are byte-identical across the CLIs.
from src.research.cli_common import build_symbol_list, load_bars_for_symbols

# rotation_walk_forward_search is the already-tested per-fold-searched rotation verdict function;
# this script's only real work is loading bars and calling it once with the locked config.
from src.research.rotation_verdict import rotation_walk_forward_search


# Universe is fixed to etf_basket — the stable long-history basket this analysis targets, the
# SAME universe overfitting_tax.py locks. --symbols still overrides for the ad-hoc cross-check.
_UNIVERSE = "etf_basket"

# Bar load window. The etf_basket deep backfill runs 2008→present; this wide range captures all
# of it. Hard-coded (not date.today()) so runs are reproducible day-to-day, matching
# overfitting_tax.py's rationale for fixed default dates.
_START = "2000-01-01"

# Inclusive end of the load window — wide enough to include every stored bar; the actual data
# extent (last bar) is whatever DuckDB holds, this is only an upper clamp.
_END = "2026-12-31"

# The reference symbol whose month-ends define the shared rebalance spine. SPY is in the basket
# and shares the max history, so its month-end grid is the natural schedule for the rotation.
_REFERENCE = "SPY"

# The per-fold search grid: 16 (top_n, lookback) candidates — hold 1/2/3/5 ETFs, ranked over a
# 3/6/9/12-month formation window. Locked BEFORE the run; the search picks one per fold by best
# in-sample after-cost Sharpe, and the tax measures how much of that in-sample edge survives OOS.
_GRID = [(t, l) for t in (1, 2, 3, 5) for l in (3, 6, 9, 12)]

# Transaction cost: 10 bps per unit turnover (fee + slippage), a realistic round-trip friction
# for liquid ETFs. The verdict is AFTER this cost — an edge that only exists at zero cost is not
# an edge. Locked before the run; overridable via --cost-rate only for the robustness cross-check.
_COST_RATE = 0.0010


def main() -> int:
    """Entry point — parse args, load the basket, run the searched rotation verdict, print it."""

    # ------------------------------------------------------------------
    # 1. Parse CLI arguments.
    # ------------------------------------------------------------------

    # All flags are optional; their defaults ARE the locked headline configuration. The flags
    # exist for the documented robustness cross-check (e.g. --test-months 6), NOT for tuning
    # toward a nicer tax after seeing the results.
    parser = argparse.ArgumentParser(
        description="Per-fold-searched cross-sectional rotation verdict vs the equal-weight ETF basket."
    )

    # --train-months: TRAIN month-ends per fold. Default 24 clears the L_max=12 guard comfortably
    # (24 > 12 + 1) and leaves ~11 in-sample scoring month-ends for the grid search.
    parser.add_argument(
        "--train-months",
        type=int,
        default=24,
        help="Train month-ends per fold (default: 24; locked headline config)",
    )

    # --test-months: TEST decision month-ends per fold. Default 12 → disjoint 1-year OOS test
    # spans; --test-months 6 is the documented higher-fold-count robustness cross-check.
    parser.add_argument(
        "--test-months",
        type=int,
        default=12,
        help="Test decision month-ends per fold (default: 12; locked headline config)",
    )

    # --cost-rate: per-unit-turnover transaction cost fraction. Default is _COST_RATE (10 bps);
    # overridable only for the robustness cross-check, not to tune the verdict.
    parser.add_argument(
        "--cost-rate",
        type=float,
        default=_COST_RATE,
        help=f"Per-unit-turnover cost fraction (default: {_COST_RATE}; locked headline config)",
    )

    # --symbols: explicit ticker list overriding the etf_basket default. nargs="*" so
    # "SPY QQQ TLT" parses as tokens; empty/omitted → full basket.
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Explicit symbols overriding the basket (e.g. --symbols SPY QQQ TLT GLD)",
    )

    # --annualization: bars per year for Sharpe/Sortino annualisation. 252 (trading days) matches
    # the rest of the stack; exposed only for symmetry with the underlying function's parameter.
    parser.add_argument(
        "--annualization",
        type=int,
        default=252,
        help="Bars per year for Sharpe/Sortino annualisation (default: 252)",
    )

    # Parse the argv into the args namespace.
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Resolve the symbol list (etf_basket unless --symbols overrides).
    # ------------------------------------------------------------------

    # build_symbol_list expects a CSV string or None; --symbols is a token list, so join it back
    # to CSV. The large limit returns the full basket on the universe path (etf_basket has 17 <
    # 1000); it is ignored on the --symbols path.
    symbols_csv = ",".join(args.symbols) if args.symbols else None

    # Resolve the ordered symbol list — either the parsed CSV or the full etf_basket.
    symbols = build_symbol_list(symbols_csv, _UNIVERSE, limit=1000)

    # Empty-list guard: build_symbol_list may return [] for a bad --symbols input; the CLI owns
    # its own user-facing error copy (cli_common intentionally does not print).
    if not symbols:
        print("ERROR: Symbol list is empty. Check --symbols.")
        return 1

    # ------------------------------------------------------------------
    # 3. Load bars for every symbol (one DuckDB connection).
    # ------------------------------------------------------------------

    # Progress line so a multi-symbol read shows life before the run begins.
    print(f"Loading bars for {len(symbols)} symbols...")

    # Read every symbol's bars over the wide window in a single DuckDB connection. Symbols with
    # no data are logged+skipped by the helper, not raised.
    bars_dict = load_bars_for_symbols(symbols, _START, _END)

    # Empty-dict guard: every symbol was missing (e.g. ingest never ran). Own the error copy.
    if not bars_dict:
        print(
            f"ERROR: No bars found for any symbol in {_START}–{_END}. "
            "Run the ingest script first."
        )
        return 1

    # ------------------------------------------------------------------
    # 4. Guard: the reference spine (SPY) must be present.
    # ------------------------------------------------------------------

    # rotation_walk_forward_search would raise on a missing reference, but a named, early CLI
    # error is friendlier than a traceback — the whole schedule hangs off SPY's month-ends.
    if _REFERENCE not in bars_dict:
        print(
            f"ERROR: reference symbol {_REFERENCE} not in loaded bars "
            f"(loaded: {sorted(bars_dict.keys())}). Ingest {_REFERENCE} first."
        )
        return 1

    # ------------------------------------------------------------------
    # 5. Print the run note (self-documenting, reproducible header).
    # ------------------------------------------------------------------

    # Echo the exact locked config into stdout so the printed verdict carries its own provenance:
    # anyone reading the output later can see the grid, windows, cost, reference, and symbol count
    # that produced it without re-deriving them from the script.
    print(
        f"Rotation walk-forward search on {_UNIVERSE}: "
        f"{len(bars_dict)} symbols, reference={_REFERENCE}, "
        f"grid={_GRID} (|grid|={len(_GRID)}), "
        f"train_months={args.train_months}, test_months={args.test_months}, "
        f"cost_rate={args.cost_rate}, annualization={args.annualization}"
    )

    # Blank line separating the run note from the verdict table.
    print()

    # ------------------------------------------------------------------
    # 6. Run the searched rotation walk-forward ONCE on the whole basket.
    # ------------------------------------------------------------------

    # Rotation is inherently MULTI-symbol — one call on the entire basket, NOT a per-symbol loop
    # like the SMA overfitting tax. hold_when_all_negative is intentionally NOT passed: it defaults
    # to False for the strategy (matching the tested default), while the benchmark forces True
    # internally to be a true always-hold-everything buy-and-hold.
    verdict = rotation_walk_forward_search(
        bars_by_symbol=bars_dict,
        reference_symbol=_REFERENCE,
        candidates=_GRID,
        train_months=args.train_months,
        test_months=args.test_months,
        cost_rate=args.cost_rate,
        annualization_factor=args.annualization,
        symbol_label="rotation-etf-basket",
    )

    # ------------------------------------------------------------------
    # 7. Print the aligned verdict table (overfitting_tax.py style).
    # ------------------------------------------------------------------

    # Top rule.
    print("=" * 78)

    # Header row for the Sharpe/tax block — the headline fixed-vs-fitted-vs-B&H comparison.
    header = (
        f"{'Folds':>6}  {'Fixed OOS':>10}  {'Fitted OOS':>11}  "
        f"{'B&H':>8}  {'In-sample':>10}  {'Tax':>8}"
    )
    print(header)

    # Divider under the header.
    print("-" * 78)

    # The single verdict row: n_folds, then the four Sharpe scalars and the tax, all signed and
    # 2-dp so columns align regardless of sign.
    print(
        f"{verdict.n_folds:>6}  "
        f"{verdict.fixed_oos_sharpe:>+10.2f}  {verdict.fitted_oos_sharpe:>+11.2f}  "
        f"{verdict.bh_sharpe:>+8.2f}  {verdict.mean_in_sample_sharpe:>+10.2f}  "
        f"{verdict.tax:>+8.2f}"
    )

    # Rule closing the Sharpe block.
    print("=" * 78)

    # Secondary metrics block: Sortino / max-drawdown / total-return for the fitted strategy and
    # the benchmark, so the reader sees risk-adjusted and absolute figures beyond Sharpe.
    print(
        f"{'':>14}{'Sortino':>10}  {'MaxDD':>10}  {'TotalRet':>10}"
    )
    print(
        f"{'Fitted OOS':>14}"
        f"{verdict.fitted_oos_sortino:>+10.2f}  "
        f"{verdict.fitted_oos_max_drawdown:>+10.2%}  "
        f"{verdict.fitted_oos_total_return:>+10.2%}"
    )
    print(
        f"{'B&H':>14}"
        f"{verdict.bh_sortino:>+10.2f}  "
        f"{verdict.bh_max_drawdown:>+10.2%}  "
        f"{verdict.bh_total_return:>+10.2%}"
    )
    print("=" * 78)

    # Per-fold detail: what the search actually PICKED each fold and the in-sample Sharpe it
    # maximised — so the reader can see whether the search was stable or thrashed fold-to-fold.
    print("Per-fold chosen (top_n, lookback) @ in-sample Sharpe:")

    # One line per fold, enumerated from 1, showing the chosen candidate and its in-sample score.
    for i, (tn, lb, is_sharpe) in enumerate(
        zip(verdict.chosen_top_n, verdict.chosen_lookback, verdict.in_sample_sharpes),
        start=1,
    ):
        print(f"  fold {i:>2}: top_n={tn}, lookback={lb:>2}  in-sample Sharpe={is_sharpe:+.2f}")

    # Rule closing the per-fold block.
    print("=" * 78)

    # ------------------------------------------------------------------
    # 8. One-line honest read — stated factually, no spin.
    # ------------------------------------------------------------------

    # Did the per-fold-tuned strategy beat buy-and-hold OOS? A tie/loss is a legitimate finding.
    beat_bh = verdict.fitted_oos_sharpe > verdict.bh_sharpe

    # Did tuning even beat the fixed-parameter baseline OOS? If not, the search added nothing.
    beat_fixed = verdict.fitted_oos_sharpe > verdict.fixed_oos_sharpe

    # State the three facts plainly: fitted-vs-B&H, fitted-vs-fixed, and the sign/size of the tax.
    print(
        f"Fitted OOS Sharpe {verdict.fitted_oos_sharpe:+.2f} "
        f"{'BEAT' if beat_bh else 'did NOT beat'} B&H {verdict.bh_sharpe:+.2f}; "
        f"{'BEAT' if beat_fixed else 'did NOT beat'} fixed {verdict.fixed_oos_sharpe:+.2f}; "
        f"overfitting tax = {verdict.tax:+.2f} "
        f"(in-sample {verdict.mean_in_sample_sharpe:+.2f} − fitted OOS {verdict.fitted_oos_sharpe:+.2f})."
    )

    # Trailing blank line so the shell prompt does not butt against the last line of output.
    print()

    # Success exit code.
    return 0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

# Propagate main()'s return value to the shell as $? when run directly.
if __name__ == "__main__":
    sys.exit(main())
