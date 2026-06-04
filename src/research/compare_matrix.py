# src/research/compare_matrix.py

"""
CLI: run a hardcoded grid of SMA crossover parameter combinations against many
symbols and display the results as a pivot table (symbols as rows, strategies
as columns, a chosen metric in each cell) plus a strategy ranking summary.

Run via:
    python -m src.research.compare_matrix
    python -m src.research.compare_matrix --universe sp500 --limit 25
    python -m src.research.compare_matrix --symbols SPY,QQQ,AAPL,MSFT,NVDA

This is the "many symbols × many strategies" axis — the cross-product of
compare_universe.py (many symbols × one strategy) and compare_strategies.py
(one symbol × many strategies).  The two degenerate cases fall out naturally:
with one strategy it behaves like compare_universe; with one symbol it behaves
like compare_strategies.

Why the param grid is hardcoded here and not exposed as --fast/--slow flags:
  Matrix tools have a "scan a family of configurations" identity — the whole
  point is to see how a coherent family of strategies distributes across a
  universe.  Universe tools (compare_universe.py) have the opposite identity:
  "test ONE configuration broadly."  Adding --fast/--slow flags would collapse
  this tool into a single-strategy universe scan, which is already
  compare_universe.py.  Hardcoding also means this CLI's output is directly
  comparable with compare_strategies.py, which uses the same six pairs; the
  two tools answer complementary questions about the same parameter family.

Why the pivot is symbol-rows × strategy-columns and not the other way around:
  Humans read variation-across-strategies more naturally as horizontal scanning
  within a row (six numbers side by side), and variation-across-symbols as
  vertical scrolling (one symbol per line).  Symbols also vastly outnumber
  strategies (25 vs. 6), so the "tall" dimension belongs on the row axis.
  Inverting the layout would produce a 6-row × 25-column table that scrolls
  horizontally in the terminal — much harder to read than 25 short rows.

Why min_bars is derived from max(slow) rather than min(slow) of the grid:
  The slowest strategy sets the data floor for the entire matrix.  If we used
  min(slow), faster strategies could run on short symbols while slower ones
  would raise ValueError inside generate_signals() — that would produce a
  ragged pivot where some cells are filled and others are errors or NaN,
  which is confusing and misleading.  Using max(slow)+1 guarantees that every
  symbol either contributes a complete row (all six strategy cells filled) or
  is skipped entirely — a rectangular pivot that readers can interpret cleanly.

Default behavior: first 25 stocks of the S&P 500 universe, six SMA parameter
combinations spanning (10,30) to (50,200), 2021-01-01 to 2026-05-14.
Six strategies × 25 symbols = 150 backtests.  Same grid as
compare_strategies.py for direct cross-tool comparability.
"""

# argparse is the standard-library CLI argument parser; same pattern as all
# other research CLIs in this module.
import argparse

# sys.exit() with an integer propagates the exit code to the shell so cron /
# CI can branch on success vs. failure via $?.
import sys

# pandas is the pivoting and formatting backbone.  pivot_table(), .map(), and
# to_string() are all used in the display pipeline.
import pandas as pd

# SMACrossoverStrategy is the only strategy family evaluated in this CLI.
# The param grid sweeps six (fast, slow) pairs within this family — the same
# six pairs compare_strategies.py uses — keeping the tools directly comparable.
from src.strategies.sma_crossover import SMACrossoverStrategy

# BacktestRunner is the orchestration layer.  run_matrix() is the axis we
# exercise here — many strategies × many symbols simultaneously.
from src.research.runner import BacktestRunner

# build_symbol_list resolves --symbols CSV or --universe/--limit to a plain
# list[str].  load_bars_for_symbols opens DuckDB once and reads all symbols
# in a single connection, skipping missing ones with a log.warning.
from src.research.cli_common import build_symbol_list, load_bars_for_symbols


# ---------------------------------------------------------------------------
# Hardcoded parameter grid.
# ---------------------------------------------------------------------------
# Six (fast, slow) pairs, all valid (slow > fast).  The list spans:
#   * very short:  (10, 30)     — reacts fast, lots of whipsaw
#   * short:       (10, 50), (20, 50)
#   * medium:      (20, 100), (50, 100)
#   * canonical:   (50, 200)    — the classic "golden cross"
#
# Why hardcoded at module level and not inside main():
#   The grid is an intrinsic property of this tool — it is what makes
#   compare_matrix.py a matrix tool rather than a universe tool.  Defining it
#   at module level makes it inspectable (e.g. for tests or the summary
#   docstring) without running main().  See module-level docstring for the
#   full rationale.
#
# Why the same six pairs as compare_strategies.py:
#   Keeping the grid identical means a researcher can compare a single-symbol
#   result (compare_strategies.py) with a universe-wide result
#   (compare_matrix.py) without mentally accounting for different parameter
#   sets — direct comparability is worth the coupling.
PARAM_GRID = [(10, 30), (10, 50), (20, 50), (20, 100), (50, 100), (50, 200)]


# ---------------------------------------------------------------------------
# Metric metadata.
# ---------------------------------------------------------------------------
# Maps the CLI --metric argument (short, interactive-friendly names) to the
# BacktestResult attribute names.  The two namespaces diverge only for
# "sharpe" (CLI) vs. "sharpe_ratio" (attribute) for brevity at the prompt.
# Defined at module level so both the pivot-building and formatting steps
# reference the same mapping without threading it through function arguments.
METRIC_ATTR_MAP = {
    "sharpe": "sharpe_ratio",
    "total_return": "total_return_pct",
    "max_drawdown": "max_drawdown_pct",
    "win_rate": "win_rate",
    "n_trades": "n_trades",
}


def main() -> int:
    """Entry point — parse args, run matrix, print pivot and ranking table.

    Returns an integer exit code: 0 on success, 1 on a recoverable error
    (e.g. empty symbol list, all symbols too short for the slowest strategy).
    """

    # ------------------------------------------------------------------
    # 1. Parse CLI arguments.
    # ------------------------------------------------------------------
    # argparse converts command-line tokens into a structured Namespace.
    # Every argument has a default so the bare command works without any flags.
    parser = argparse.ArgumentParser(
        description=(
            "Run a hardcoded SMA crossover parameter grid against many symbols "
            "and display a pivot table of results."
        )
    )

    # --universe selects the named universe to load from config/universe.yaml.
    # "sp500" is the default because it is the primary research universe.
    parser.add_argument(
        "--universe",
        default="sp500",
        help="Named universe to load from config/universe.yaml (default: sp500)",
    )

    # --symbols overrides --universe with an explicit comma-separated list.
    # Useful for quick ad-hoc sweeps (e.g. a FAANG basket) without editing
    # the universe config.  If set, --universe and --limit are both ignored.
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol list, e.g. SPY,QQQ,AAPL (overrides --universe)",
    )

    # --limit caps the number of symbols taken from --universe.  Default 25
    # keeps the pivot table readable on screen and the run time under ~10 s.
    # Ignored when --symbols is provided because the caller already controls
    # the list size explicitly.
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max symbols from --universe (default: 25; ignored when --symbols is set)",
    )

    # --start and --end bound the backtest window.  Hard-coded defaults make
    # runs reproducible — using date.today() would silently shift results daily.
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

    # --metric selects which BacktestResult field fills the pivot cells AND
    # which the strategy-ranking summary sorts by.  choices= whitelists the
    # same keys as METRIC_ATTR_MAP so an invalid value is caught by argparse
    # with a friendly message before it reaches any downstream code.
    parser.add_argument(
        "--metric",
        default="sharpe",
        choices=list(METRIC_ATTR_MAP.keys()),
        help=(
            "Metric to display in pivot cells and rank strategies by "
            f"(default: sharpe; choices: {', '.join(METRIC_ATTR_MAP)})"
        ),
    )

    # parse_args() reads sys.argv; on bad input it prints usage and exits
    # before returning, so we don't need to handle that case.
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Build the symbol list.
    # ------------------------------------------------------------------
    # CSV overrides universe; limit is applied only on the universe path.
    # See cli_common.build_symbol_list for the full rationale.
    symbols = build_symbol_list(args.symbols, args.universe, args.limit)

    # Report the matrix dimensions so the operator knows exactly how much
    # work is about to happen before any I/O starts.
    print(
        f"Loaded {len(symbols)} symbols, {len(PARAM_GRID)} strategies = "
        f"{len(symbols) * len(PARAM_GRID)} backtests"
    )

    # An empty symbol list means there is nothing to backtest.  Printing a
    # clear message here is more helpful than letting the runner raise its
    # own (more cryptic) "bars_by_symbol must be non-empty" error.
    if not symbols:
        print("ERROR: Symbol list is empty. Check --symbols or --universe.")
        return 1

    # ------------------------------------------------------------------
    # 3. Build the strategies list.
    # ------------------------------------------------------------------
    # One SMACrossoverStrategy instance per (fast, slow) pair from the
    # module-level PARAM_GRID.  Strategy is stateless (pure):
    # generate_signals() reads only the bars it receives and carries no
    # mutable internal state, so the same instance safely evaluates every
    # symbol across the entire matrix without reset.
    strategies = [
        SMACrossoverStrategy(fast_window=f, slow_window=s) for f, s in PARAM_GRID
    ]

    # ------------------------------------------------------------------
    # 4. Read bars for every symbol from DuckDB.
    # ------------------------------------------------------------------
    bars_by_symbol = load_bars_for_symbols(symbols, args.start, args.end)

    # Report actual coverage so the operator can see how many symbols had
    # data vs. how many were requested — the fraction signals data quality.
    print(f"Loaded bars for {len(bars_by_symbol)}/{len(symbols)} symbols")

    # Guard: if every symbol was missing from DuckDB, there is nothing to run.
    if not bars_by_symbol:
        print(
            f"ERROR: No bars found for any symbol in {args.start}–{args.end}. "
            "Run the ingest script first."
        )
        return 1

    # ------------------------------------------------------------------
    # 5. Determine min_bars from the slowest strategy in the grid.
    # ------------------------------------------------------------------
    # The slowest strategy sets the data floor for the entire matrix.
    # Using max(slow) rather than min(slow) ensures every symbol either
    # contributes a complete row of all six strategy cells or is skipped
    # entirely — see module-level docstring for the full rationale.
    max_slow = max(slow for _, slow in PARAM_GRID)

    # +1 because SMACrossoverStrategy.generate_signals() raises ValueError
    # when len(bars) <= slow_window.  Passing slow+1 lets run_matrix() skip
    # short symbols cleanly before generate_signals() is called at all,
    # rather than aborting the sweep mid-run.
    min_bars = max_slow + 1

    # ------------------------------------------------------------------
    # 6. Run the full (symbol × strategy) matrix.
    # ------------------------------------------------------------------
    # Default BacktestRunner() → default Backtester(initial_capital=1.0,
    # annualization=252); no custom config needed for this comparison.
    runner = BacktestRunner()

    # run_matrix raises ValueError with a descriptive message when every
    # symbol is below min_bars.  We catch only ValueError — any other
    # exception is a genuine bug and should propagate with a full stack trace.
    try:
        results = runner.run_matrix(bars_by_symbol, strategies, min_bars=min_bars)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    # ------------------------------------------------------------------
    # 7. Build the pivot DataFrame.
    # ------------------------------------------------------------------
    # We first build a long-format DataFrame (one row per BacktestResult),
    # then pivot to wide format.  The long intermediate is easier to debug —
    # you can inspect it to verify that the " on " splitting worked correctly
    # before the pivot operation folds it down.

    # The BacktestResult attribute name for the chosen metric.
    attr = METRIC_ATTR_MAP[args.metric]

    # Build the long-format list.  result.strategy_name has the format
    # "SMA(10, 30) on AAPL", which both run_universe and run_matrix produce
    # via f"{strategy.name} on {symbol}".  Splitting on " on " is robust:
    #   * " on " (space-on-space) never appears in a bare SMA strategy name
    #     (only digits, commas, spaces inside parens, and the "SMA" prefix).
    #   * Stock symbols are uppercase letters only — no " on " substring.
    #   * Both runner methods produce the identical format, so the split
    #     logic is stable across compare_universe and compare_matrix output.
    # maxsplit=1 is a safety measure: the first " on " is always the correct
    # split point, and limiting to one split avoids issues if a future
    # strategy name ever contained the substring "on".
    rows = []
    for r in results:
        strategy_label, symbol = r.strategy_name.split(" on ", maxsplit=1)
        rows.append(
            {
                "strategy_name": strategy_label,
                "symbol": symbol,
                args.metric: getattr(r, attr),
            }
        )

    long_df = pd.DataFrame(rows)

    # Pivot to wide format: one row per symbol, one column per strategy.
    # aggfunc="first" because the matrix guarantees at most one result per
    # (symbol, strategy) cell — there is nothing to aggregate, but
    # pivot_table() requires an explicit aggfunc argument.
    pivot = long_df.pivot_table(
        index="symbol",
        columns="strategy_name",
        values=args.metric,
        aggfunc="first",
    )

    # Remove the column axis name ("strategy_name") that pandas sets
    # automatically on the columns attribute after pivot_table.  If left in
    # place it would print as a spurious header line above the column names
    # in to_string() output.
    pivot.columns.name = None

    # Reindex columns to match PARAM_GRID insertion order.  pivot_table()
    # sorts column labels alphabetically by default; reindex restores the
    # logical order (fastest-window → slowest-window) so the table reads
    # left-to-right from aggressive to conservative.
    pivot = pivot.reindex(columns=[s.name for s in strategies])

    # ------------------------------------------------------------------
    # 8. Sort pivot rows by each symbol's row-mean, descending.
    # ------------------------------------------------------------------
    # Sorting by row-mean gives "best overall symbol" at the top, consistent
    # with the "top of table = best" invariant the other CLIs follow.
    #
    # Special case for max_drawdown: smaller is better, so ascending=True
    # puts the symbol with the lowest average drawdown (best) at the top.
    # For every other metric: larger is better → ascending=False (best first).
    row_means = pivot.mean(axis=1)
    ascending_sort = args.metric == "max_drawdown"
    pivot = pivot.loc[row_means.sort_values(ascending=ascending_sort).index]

    # ------------------------------------------------------------------
    # 9. Format the pivot copy for display.
    # ------------------------------------------------------------------
    # Operate on a *copy* so the underlying numeric pivot stays available
    # for the summary calculations in step 11.  Mutating the original would
    # silently convert float columns to strings, breaking column means.

    pivot_display = pivot.copy()

    # Build a cell formatter for the chosen metric.  All formatters handle
    # NaN explicitly — a symbol might be missing data for a specific sub-
    # range even after passing min_bars — returning "—" (em-dash) so the
    # table remains readable without blank cells.
    if args.metric == "sharpe":
        # Dimensionless ratio; 2 decimals, no percent sign — matches finance
        # convention where negative Sharpes print naturally with "−" prefix.
        cell_fmt = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    elif args.metric == "total_return":
        # Signed percent, 2 decimals.  "+" forces an explicit sign on gainers
        # so winners and losers are visually distinct at a glance.
        cell_fmt = lambda x: "—" if pd.isna(x) else f"{x*100:+.2f}%"
    elif args.metric == "max_drawdown":
        # Unsigned percent, 2 decimals.  BacktestResult stores drawdown as a
        # positive fraction already, so no sign treatment is needed.
        cell_fmt = lambda x: "—" if pd.isna(x) else f"{x*100:.2f}%"
    elif args.metric == "win_rate":
        # Unsigned percent, 1 decimal.  One fewer decimal than the return
        # columns because win rate is a coarse aggregate over discrete
        # trades — extra precision implies false accuracy.
        cell_fmt = lambda x: "—" if pd.isna(x) else f"{x*100:.1f}%"
    else:  # n_trades
        # Integer count; cast to int to drop the ".0" that pandas stores
        # all-numeric columns as float64 when NaNs are present.
        cell_fmt = lambda x: "—" if pd.isna(x) else f"{int(x)}"

    # DataFrame.map() applies the formatter element-wise to every cell
    # (value) in the frame.  The symbol index (row labels) is not touched —
    # .map() operates on the data cells only, which is what "value columns"
    # means in a pivot where the index carries the symbol identity.
    # (pandas ≥ 2.1 renamed DataFrame.applymap to DataFrame.map; behaviour
    # is identical.)
    pivot_display = pivot_display.map(cell_fmt)

    # ------------------------------------------------------------------
    # 10. Print the pivot table.
    # ------------------------------------------------------------------
    # Blank line separates the pivot block from the progress lines above.
    print()
    print(
        f"{args.metric} matrix: {len(strategies)} strategies × {len(bars_by_symbol)} symbols"
    )
    print(f"Range: {args.start} to {args.end}")
    print()

    # to_string() with the default index=True keeps symbol names as row
    # labels — the index IS the meaningful content here (it identifies each
    # row as a specific stock), so hiding it with index=False would make the
    # table unreadable.
    print(pivot_display.to_string())
    print()

    # ------------------------------------------------------------------
    # 11. Build and print the strategy ranking summary.
    # ------------------------------------------------------------------
    # The summary answers "which strategy configuration generalises best
    # across this universe?"  Each row represents one strategy (one column
    # of the pivot); the mean metric across all surviving symbols is the
    # primary ranking signal.
    #
    # "positive_<metric>" gives a hit-rate signal: for how many symbols
    # did the strategy earn positive edge?  For max_drawdown the question
    # is re-framed as "below-median drawdown for this strategy" because
    # max_drawdown is always ≥ 0, so "> 0" would count almost every symbol
    # and convey no information.  For n_trades we skip the positive count
    # entirely — a trade count is a description of activity, not an edge
    # direction signal.

    summary_rows = []
    for col in pivot.columns:
        # dropna() excludes symbols that were NaN in this column from the
        # mean and count — we want the mean over symbols that actually ran,
        # not over the full universe including skipped ones.
        col_series = pivot[col].dropna()
        mean_val = col_series.mean()

        row: dict = {"strategy": col, f"mean_{args.metric}": mean_val}

        if args.metric == "n_trades":
            # n_trades has no natural "positive direction" threshold — any
            # non-zero count means the strategy traded.  Omit the positive_
            # column rather than print a meaningless "> 0 trades" count.
            pass
        elif args.metric == "max_drawdown":
            # Re-frame "positive" as "better than median drawdown" for this
            # strategy column: count the symbols where drawdown was lower than
            # the column median.  This avoids the "drawdown > 0 is always true"
            # problem and gives a sensible peer-relative edge signal.
            median_val = col_series.median()
            row[f"positive_{args.metric}"] = int((col_series < median_val).sum())
        else:
            # sharpe, total_return, win_rate: count symbols where the metric
            # is strictly positive — any edge above a random walk (Sharpe > 0,
            # return > 0) or above zero-win-rate.
            row[f"positive_{args.metric}"] = int((col_series > 0).sum())

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Sort by mean metric: descending for most metrics (larger = better),
    # ascending for max_drawdown (smaller average drawdown = better ranking).
    summary_df = summary_df.sort_values(
        by=f"mean_{args.metric}", ascending=ascending_sort
    ).reset_index(drop=True)

    # Format the mean column for readability — same style as the pivot cells
    # so the two outputs are visually consistent.
    mean_col = f"mean_{args.metric}"
    if args.metric == "sharpe":
        summary_df[mean_col] = summary_df[mean_col].apply(lambda x: f"{x:.2f}")
    elif args.metric == "total_return":
        summary_df[mean_col] = summary_df[mean_col].apply(lambda x: f"{x*100:+.2f}%")
    elif args.metric == "max_drawdown":
        summary_df[mean_col] = summary_df[mean_col].apply(lambda x: f"{x*100:.2f}%")
    elif args.metric == "win_rate":
        summary_df[mean_col] = summary_df[mean_col].apply(lambda x: f"{x*100:.1f}%")
    else:  # n_trades
        summary_df[mean_col] = summary_df[mean_col].apply(lambda x: f"{int(round(x))}")

    # Print the summary.  index=False because the 0..N-1 integer index carries
    # no information after sorting — the strategy column is the identity key.
    print(f"Strategy ranking by mean {args.metric} (best first):")
    print()
    print(summary_df.to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # 12. Return success.
    # ------------------------------------------------------------------
    # Exit code 0 signals to the calling shell that the run completed without
    # recoverable errors.  Non-zero returns earlier in the function signal
    # failure so cron / CI can branch on $?.
    return 0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
# python -m src.research.compare_matrix enters here.  sys.exit() propagates
# the integer return value of main() to the shell as the process exit code.
if __name__ == "__main__":
    sys.exit(main())
