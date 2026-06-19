# src/research/compare_walkforward.py

"""
CLI: walk-forward validation of one SMA crossover configuration, one symbol at a time.

Run via:
    python -m src.research.compare_walkforward
    python -m src.research.compare_walkforward --symbols SPY --fast 10 --slow 50
    python -m src.research.compare_walkforward --symbols SPY,QQQ --train-size 504 --test-size 126

This is the "out-of-sample robustness" axis.  compare_universe answers "does this
strategy generalise across symbols (in-sample)?"  This CLI answers "is this
strategy's per-symbol performance stable across non-overlapping time periods?"
Both ask about generalisation but along orthogonal axes (cross-section vs time).

Walk-forward structure:
    - Each symbol's full bar list is sliced into (train, test) folds.
    - The strategy is run over train+test bars so the indicator is warm before
      the test window begins; only the test-window backtest is scored.
    - The per-fold BacktestResults are printed in a table, followed by an OOS
      aggregate (stitched returns, stitched equity curve, aggregate metrics).

Why a separate CLI rather than adding flags to compare_universe:
    The I/O structure here (per-fold rows + OOS footer, one block per symbol) is
    fundamentally different from compare_universe's single sorted table.  Merging
    them with a mode flag would produce a confusing argparse interface and a main()
    that branches on mode rather than doing one clear thing.

Default behaviour: first 25 S&P 500 symbols, SMA(50, 200), train=252 bars
(≈1 year), test=63 bars (≈1 quarter), 2021-01-01 to 2026-05-14.
"""

# argparse is the standard-library CLI parser; same pattern as compare_universe.py
# and compare_strategies.py — consistent across every CLI in this package.
import argparse

# logging is used for per-symbol skip warnings (log.warning) and module-level
# log setup.  Using log rather than print for skips keeps diagnostics on stderr
# (or the configured handler) and out of the structured per-symbol tables on
# stdout — same rationale as cli_common.load_bars_for_symbols.
import logging

# sys.exit() propagates the integer return value of main() to the shell as
# the process exit code, so CI / cron scripts can branch on $?.
import sys

# pandas builds the per-fold display table with the same fixed-width to_string
# layout compare_universe.py uses for its comparison table.  Using pandas here
# also lets us mutate a display COPY while leaving the raw numeric columns
# untouched — the same "display = df.copy()" pattern from compare_universe.py.
import pandas as pd

# SMACrossoverStrategy is the sole strategy evaluated in this CLI.  Same import
# as compare_universe.py so the two CLIs share identical strategy construction.
from src.strategies.sma_crossover import SMACrossoverStrategy

# build_symbol_list and load_bars_for_symbols are the two shared CLI helpers
# from cli_common.py.  Using them here avoids re-implementing DuckDB reads and
# universe-config resolution, and keeps symbol/bar loading consistent across
# all research CLIs.
from src.research.cli_common import build_symbol_list, load_bars_for_symbols

# walk_forward_splits slices a bar list into (train, test) fold pairs.
# walk_forward_validate orchestrates the per-fold backtests and returns the
# stitched OOS aggregate as a WalkForwardResult.  This CLI is a pure I/O layer
# on top of those two functions — no logic lives here.
from src.research.walk_forward import walk_forward_splits, walk_forward_validate


# Module-level logger following cli_common.py's convention.  getLogger(__name__)
# ties records to "src.research.compare_walkforward" so they are filterable by
# the calling process's log configuration without touching other modules.
log = logging.getLogger(__name__)


def main() -> int:
    """Entry point — parse args, run walk-forward, print per-fold tables.

    Returns an integer exit code: 0 on success, 1 on a recoverable error
    (empty symbol list, no bars, every symbol too short for the windows).
    """

    # ------------------------------------------------------------------
    # 1. Parse CLI arguments.
    # ------------------------------------------------------------------

    # ArgumentParser description appears in --help output; kept to one sentence
    # so it fits on a terminal line without wrapping.
    parser = argparse.ArgumentParser(
        description="Walk-forward validation of one SMA crossover strategy, per symbol."
    )

    # --universe selects the named universe from config/universe.yaml.
    # Default "sp500" matches compare_universe.py — the primary research universe.
    parser.add_argument(
        "--universe",
        default="sp500",
        help="Named universe from config/universe.yaml (default: sp500)",
    )

    # --symbols overrides --universe with an explicit comma-separated list.
    # None means "fall through to --universe" in build_symbol_list.
    # Verbatim from compare_universe.py.
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols, e.g. SPY,QQQ (overrides --universe)",
    )

    # --limit caps how many symbols are taken from --universe.
    # Default 25 matches compare_universe.py; ignored when --symbols is set.
    # Walk-forward output is verbose (one table per symbol) so a cap is
    # important for keeping terminal output readable in a typical session.
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max symbols from --universe (default: 25; ignored when --symbols is set)",
    )

    # --fast and --slow define the SMA crossover windows.
    # Defaults (50, 200) are the canonical "golden cross" — same as compare_universe.py
    # so the two CLIs evaluate the same default strategy for easy comparison.
    parser.add_argument(
        "--fast",
        type=int,
        default=50,
        help="Fast SMA window in bars (default: 50)",
    )
    parser.add_argument(
        "--slow",
        type=int,
        default=200,
        help="Slow SMA window in bars (default: 200)",
    )

    # --start and --end bound the bar date range loaded from DuckDB.
    # Hard-coded defaults (not date.today()) make runs reproducible day-to-day.
    # Verbatim from compare_universe.py.
    parser.add_argument(
        "--start",
        default="2021-01-01",
        help="Start date YYYY-MM-DD (default: 2021-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2026-05-14",
        help="End date YYYY-MM-DD (default: 2026-05-14)",
    )

    # --train-size: number of bars in each fold's training window.
    # Default 252 ≈ one US trading year — enough to warm a 200-bar SMA and
    # give the indicator a representative regime to establish its position.
    # Must be >= slow_window + 1 for SMACrossoverStrategy to generate signals
    # over the full_bars window; walk_forward_splits will raise if not enough
    # total bars, caught per-symbol below.
    parser.add_argument(
        "--train-size",
        type=int,
        default=252,
        help="Training window size in bars per fold (default: 252 ≈ 1 year)",
    )

    # --test-size: number of bars in each fold's out-of-sample test window.
    # Default 63 ≈ one US trading quarter — a common OOS evaluation window in
    # academic walk-forward literature.  Smaller values produce more folds but
    # each fold's metrics are estimated with less precision.
    parser.add_argument(
        "--test-size",
        type=int,
        default=63,
        help="Test window size in bars per fold (default: 63 ≈ 1 quarter)",
    )

    # --step: how many bars to advance the window between consecutive folds.
    # Default None means walk_forward_splits uses step=test_size, which tiles
    # test windows without gap or overlap.  Passing an explicit integer overrides
    # this (e.g. step < test_size for overlapping test windows — only valid when
    # the caller explicitly wants overlapping OOS periods).
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "Window advance between folds in bars "
            "(default: test-size, producing non-overlapping test windows)"
        ),
    )

    # parse_args() reads sys.argv; argparse prints usage and exits on bad input
    # before returning, so no error-handling is needed around this call.
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Build the symbol list.
    # ------------------------------------------------------------------

    # CSV flag overrides universe; limit is applied only on the universe path.
    # Same call as compare_universe.py line 153.
    symbols = build_symbol_list(args.symbols, args.universe, args.limit)

    # An empty symbol list means there is nothing to run.  Print a clear message
    # rather than letting a downstream component produce a cryptic error.
    # Same pattern and wording as compare_universe.py lines 157-159.
    if not symbols:
        print("ERROR: Symbol list is empty. Check --symbols or --universe.")
        return 1

    # ------------------------------------------------------------------
    # 3. Build the strategy.
    # ------------------------------------------------------------------

    # A single SMACrossoverStrategy instance is shared across all folds and
    # symbols.  The strategy is stateless — generate_signals() reads only the
    # bars it receives and carries no mutable state — so the same object can
    # safely evaluate every symbol and fold without reset.  Same construction
    # as compare_universe.py line 168.
    strategy = SMACrossoverStrategy(fast_window=args.fast, slow_window=args.slow)

    # ------------------------------------------------------------------
    # 4. Read bars for every symbol from DuckDB.
    # ------------------------------------------------------------------

    # One-line progress message before the I/O so the operator sees what is
    # happening while DuckDB opens and reads.  Same style as compare_universe.py.
    print(f"Loading bars for {len(symbols)} symbols...")

    # load_bars_for_symbols opens DuckDB once, reads all symbols in a single
    # connection, and logs a warning for any symbol with no data.  Returns a
    # dict in input symbol order.
    bars_dict = load_bars_for_symbols(symbols, args.start, args.end)

    # Report coverage and announce the run parameters so the operator can
    # verify inputs before the long per-symbol output begins.
    total_bars = sum(len(b) for b in bars_dict.values())
    print(
        f"Loaded bars for {len(bars_dict)} symbols ({total_bars} total bars). "
        f"Walk-forward with {strategy.name}, "
        f"train={args.train_size} bars, test={args.test_size} bars..."
    )

    # Guard: every symbol missing from DuckDB → nothing to run.
    # Same wording as compare_universe.py lines 190-194.
    if not bars_dict:
        print(
            f"ERROR: No bars found for any symbol in {args.start}–{args.end}. "
            "Run the ingest script first."
        )
        return 1

    # ------------------------------------------------------------------
    # 5. Walk-forward loop — one block of output per symbol.
    # ------------------------------------------------------------------

    # Track how many symbols produced at least one complete fold so we can
    # detect the "every symbol was skipped" case after the loop.
    n_processed = 0

    for symbol, bars in bars_dict.items():

        # Attempt to slice the bar list into (train, test) fold pairs.
        # walk_forward_splits raises ValueError when the symbol has fewer bars
        # than train_size + test_size — i.e. not enough history for even one
        # complete fold.  We catch per-symbol so a short symbol does not abort
        # the sweep for all other symbols; same spirit as run_universe's
        # min_bars skip in compare_universe.py.
        try:
            splits = walk_forward_splits(
                bars,
                train_size=args.train_size,
                test_size=args.test_size,
                step=args.step,  # None → walk_forward_splits defaults to test_size
            )
        except ValueError as exc:
            # log.warning goes to stderr (or the configured handler) rather than
            # stdout so the skip message does not corrupt the structured per-fold
            # tables that follow.  Same rationale as cli_common's log.warning for
            # missing bar data.
            log.warning("Skipping %s: %s", symbol, exc)
            continue

        # Run the validator — pure function, no I/O.  Returns a WalkForwardResult
        # with per_fold BacktestResults and stitched OOS scalar aggregates.
        result = walk_forward_validate(splits, strategy)

        # Count this symbol as successfully processed so the all-skipped guard
        # below can distinguish "0 processed" from "some processed".
        n_processed += 1

        # ------------------------------------------------------------------
        # 5a. Print the symbol header.
        # ------------------------------------------------------------------

        # Blank line separates consecutive symbol blocks for readability.
        print()

        # Separator line makes each symbol's block visually distinct when the
        # output is scrolled or piped through a pager.
        print("=" * 70)

        # Header names the symbol, strategy, actual loaded date range, and window
        # sizes so the block is self-contained: the reader can interpret it without
        # scrolling back to the command-line flags.  bars[0]/bars[-1] give the
        # actual loaded range, not the requested args.start/args.end — these can
        # differ when the DB starts later or ends earlier than the requested window.
        effective_step = args.step if args.step is not None else args.test_size
        print(
            f"  {symbol}  —  {strategy.name}"
            f"  |  {bars[0].timestamp.date()} to {bars[-1].timestamp.date()}"
            f"  |  train={args.train_size}  test={args.test_size}"
            f"  step={effective_step}  folds={result.n_folds}"
        )
        print("=" * 70)

        # ------------------------------------------------------------------
        # 5b. Build and print the per-fold table.
        # ------------------------------------------------------------------

        # Collect raw numeric values into a list of dicts — one dict per fold.
        # Using the same column-name style as BacktestRunner.compare() output
        # (total_return, sharpe, max_drawdown, win_rate) for consistency across
        # research CLI output.
        fold_rows = []
        for i, fr in enumerate(result.per_fold):
            fold_rows.append(
                {
                    "fold":         i,
                    # Raw fraction; formatted below on the display copy.
                    "total_return": fr.total_return_pct,
                    # Dimensionless ratio; formatted below.
                    "sharpe":       fr.sharpe_ratio,
                    # Raw fraction (positive); formatted below.
                    "max_drawdown": fr.max_drawdown_pct,
                    # Raw fraction in [0, 1]; formatted below.
                    "win_rate":     fr.win_rate,
                    # Integer count; pandas renders cleanly without formatting.
                    "n_trades":     fr.n_trades,
                }
            )

        # Build the raw DataFrame; this is the "source of truth" numeric table.
        fold_df = pd.DataFrame(fold_rows)

        # Operate on a display COPY so the raw numeric columns in fold_df remain
        # usable (e.g. for future aggregate computations) after formatting.
        # This mirrors compare_universe.py lines 240-260 exactly.
        fold_display = fold_df.copy()

        # total_return: signed percent, 2 decimals.  "+" format forces a leading
        # sign so gains and losses are visually distinct at a glance.
        # Verbatim from compare_universe.py line 245.
        fold_display["total_return"] = fold_display["total_return"].apply(
            lambda x: f"{x * 100:+.2f}%"
        )

        # sharpe: dimensionless, 2 decimals, no percent sign.
        # Verbatim from compare_universe.py line 259.
        fold_display["sharpe"] = fold_display["sharpe"].apply(
            lambda x: f"{x:.2f}"
        )

        # max_drawdown: unsigned percent, 2 decimals.  The stored value is already
        # a positive fraction (BacktestResult contract: -drawdown.min()).
        # Verbatim from compare_universe.py line 249.
        fold_display["max_drawdown"] = fold_display["max_drawdown"].apply(
            lambda x: f"{x * 100:.2f}%"
        )

        # win_rate: unsigned percent, 1 decimal.  One fewer decimal than the
        # return columns because win rate is a coarse count-based ratio.
        # Verbatim from compare_universe.py line 254.
        fold_display["win_rate"] = fold_display["win_rate"].apply(
            lambda x: f"{x * 100:.1f}%"
        )

        # fold and n_trades are integer columns — pandas renders them cleanly.

        # Blank line between header and table body, matching compare_universe.py
        # line 268.
        print()

        # to_string(index=False) produces a fixed-width table without the
        # left-side row-index column — same as compare_universe.py line 284.
        print(fold_display.to_string(index=False))

        # ------------------------------------------------------------------
        # 5c. Print the OOS aggregate footer for this symbol.
        # ------------------------------------------------------------------

        # Blank line separates the fold table from the OOS footer, creating a
        # clear visual hierarchy: fold rows → OOS summary → next symbol.
        print()

        # Format the OOS scalar metrics on local string variables — NOT by
        # mutating the WalkForwardResult or its fields.  The validator's raw
        # numbers are read-only; all formatting is the CLI's responsibility.

        # oos_total_return is a raw fraction (same contract as total_return_pct);
        # "+" format shows direction at a glance.
        oos_ret_str = f"{result.oos_total_return * 100:+.2f}%"

        # oos_sharpe is already a Python float; format to 2 decimals to match
        # the fold-table sharpe column.
        oos_sharpe_str = f"{result.oos_sharpe:.2f}"

        # oos_max_drawdown is a positive fraction (same contract as max_drawdown_pct).
        oos_dd_str = f"{result.oos_max_drawdown * 100:.2f}%"

        # oos_win_rate is a fraction in [0, 1]; 1 decimal matches the fold table.
        oos_wr_str = f"{result.oos_win_rate * 100:.1f}%"

        # n_folds_positive_sharpe expressed as a fraction over n_folds gives the
        # proportion of folds with positive edge — the key robustness signal.
        pos_folds_str = f"{result.n_folds_positive_sharpe}/{result.n_folds}"

        # Print the OOS footer as a single labelled line.  Labelling each value
        # (return=, sharpe=, etc.) makes it parseable by eye when scanning many
        # symbols' output without needing to count column positions.
        print(
            f"  OOS  return={oos_ret_str}  sharpe={oos_sharpe_str}"
            f"  max_dd={oos_dd_str}  win_rate={oos_wr_str}"
            f"  pos_folds={pos_folds_str}  trades={result.total_trades}"
        )

        # Buy-and-hold benchmark line, measured over the identical stitched test
        # windows (same validator path, always-long position).  Printing it
        # directly under OOS lets the reader compare the strategy to simply
        # holding the asset at a glance — the "is this edge or just beta?" read.
        bh_ret_str = f"{result.bh_return * 100:+.2f}%"
        bh_sharpe_str = f"{result.bh_sharpe:.2f}"
        bh_dd_str = f"{result.bh_max_drawdown * 100:.2f}%"
        print(
            f"  B&H  return={bh_ret_str}  sharpe={bh_sharpe_str}"
            f"  max_dd={bh_dd_str}"
        )

        # Delta line: strategy − buy-and-hold.  Computed HERE (a display concern)
        # rather than stored on the result, mirroring how oos_* formatting lives
        # in the CLI.  Positive Δ means the strategy beat holding the asset over
        # the same OOS windows; "+" format on both makes the sign explicit.  The
        # label is padded so "return=" aligns under the OOS/B&H lines (Δ is one
        # display column; four trailing spaces match the three-char OOS/B&H tags).
        d_ret_str = f"{(result.oos_total_return - result.bh_return) * 100:+.2f}%"
        d_sharpe_str = f"{result.oos_sharpe - result.bh_sharpe:+.2f}"
        print(
            f"  Δ    return={d_ret_str}  sharpe={d_sharpe_str}"
            f"   (strategy − buy-and-hold)"
        )

    # ------------------------------------------------------------------
    # 6. Guard: every symbol was skipped.
    # ------------------------------------------------------------------

    # If n_processed is still 0, every symbol either had no bar data (handled
    # by cli_common's log.warning + skip) or had fewer bars than train+test
    # (caught in the walk_forward_splits try/except above).  Printing nothing
    # and returning 0 would mislead the caller into thinking the run succeeded;
    # returning 1 with a clear message lets CI / cron detect the failure.
    if n_processed == 0:
        print(
            f"ERROR: Every symbol was skipped — no symbol had enough bars for "
            f"train_size={args.train_size} + test_size={args.test_size} "
            f"= {args.train_size + args.test_size} bars minimum. "
            "Reduce window sizes or ingest more data."
        )
        return 1

    # ------------------------------------------------------------------
    # 7. Return success.
    # ------------------------------------------------------------------

    # Trailing blank line so the shell prompt does not butt up against the
    # last OOS footer line — matching compare_universe.py line 288.
    print()

    # Exit code 0 signals success to the calling shell.  Non-zero returns
    # earlier in main() signal recoverable failures so $? can be tested.
    return 0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

# python -m src.research.compare_walkforward enters here.  sys.exit() converts
# the integer main() return value into the process exit code visible to the
# shell as $?.  Same pattern as every other CLI in this package.
if __name__ == "__main__":
    sys.exit(main())
