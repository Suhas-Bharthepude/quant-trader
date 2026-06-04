# src/research/compare_universe.py

"""
CLI: compare one SMA crossover configuration across many symbols.

Run via:
    python -m src.research.compare_universe
    python -m src.research.compare_universe --fast 10 --slow 50 --limit 50
    python -m src.research.compare_universe --symbols AAPL,MSFT,GOOG,SPY

This is the "many symbols × one strategy" axis.  Day 16 was "one symbol ×
many strategies."  Same runner abstraction — BacktestRunner.run_universe()
instead of run_many() — different iteration axis.

Default behavior: first 25 stocks of the S&P 500 universe, SMA(50, 200),
2021-01-01 to 2026-05-14.  25 symbols is enough to detect a generalisation
signal while keeping the table readable and the run time under ~10 seconds.
Phase 2 will sweep the full 500, but today the goal is getting the workflow
right, not exhaustive coverage.

Why a separate CLI rather than a --symbol multi-value flag in
compare_strategies.py:
- compare_strategies.py answers "which parameter combo works best on this
  symbol?"  compare_universe.py answers "does this parameter combo generalise
  across symbols?"  Two distinct research questions deserve two distinct entry
  points.  Conflating them with mode flags would produce confusing UX and
  tangled argparse logic.

Why min_bars = slow + 1:
- SMACrossoverStrategy.generate_signals() raises ValueError when
  len(bars) <= slow_window.  Setting min_bars = slow + 1 catches that
  condition in the runner BEFORE generate_signals() is called, so the runner
  can skip the symbol cleanly rather than aborting the whole sweep.
"""

# argparse is the standard-library CLI argument parser; same pattern as
# compare_strategies.py and the scripts/ directory tools.
import argparse

# sys.exit() with an integer propagates the exit code to the shell so
# cron / CI can branch on success vs. failure via $?.
import sys

# pandas is the format runner.compare() returns; we mutate a *copy* for
# display so the underlying numeric values remain usable.
import pandas as pd

# SMACrossoverStrategy is the sole strategy evaluated in this CLI.  Phase 2
# may generalise to multiple strategy classes; for now we fix the family and
# sweep symbols.
from src.strategies.sma_crossover import SMACrossoverStrategy

# BacktestRunner is the orchestration layer.  run_universe() is the axis
# we exercise here — one strategy, many symbols.
from src.research.runner import BacktestRunner

# build_symbol_list resolves --symbols CSV or --universe/--limit to a plain
# list[str].  load_bars_for_symbols opens DuckDB once and reads all symbols
# in a single connection, skipping missing ones with a log.warning.
from src.research.cli_common import build_symbol_list, load_bars_for_symbols


def main() -> int:
    """Entry point — parse args, run backtests, print the comparison table.

    Returns an integer exit code: 0 on success, 1 on a recoverable error
    (e.g. empty symbol list, all symbols too short for the strategy).
    """

    # ------------------------------------------------------------------
    # 1. Parse CLI arguments.
    # ------------------------------------------------------------------
    # argparse converts command-line tokens into a structured Namespace.
    # Every argument has a default so the bare command works without any flags.
    parser = argparse.ArgumentParser(
        description="Compare one SMA crossover strategy across many symbols."
    )

    # --universe selects the named universe to load from config/universe.yaml.
    # "sp500" is the default because it is the primary research universe.
    parser.add_argument(
        "--universe",
        default="sp500",
        help="Named universe to load from config/universe.yaml (default: sp500)",
    )

    # --symbols overrides --universe with an explicit comma-separated list.
    # Useful for quick ad-hoc sweeps without touching the universe config.
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol list, e.g. AAPL,MSFT,GOOG,SPY (overrides --universe)",
    )

    # --limit caps the number of symbols taken from --universe.  Default 25
    # keeps the table readable on screen and the run time under ~10 s.
    # Ignored when --symbols is provided because the caller already controls
    # the list size explicitly.
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max symbols from --universe (default: 25; ignored when --symbols is set)",
    )

    # --fast and --slow define the SMA crossover windows.  Defaults (50, 200)
    # are the canonical "golden cross" — the most widely followed configuration.
    parser.add_argument(
        "--fast",
        type=int,
        default=50,
        help="Fast SMA window (default: 50)",
    )
    parser.add_argument(
        "--slow",
        type=int,
        default=200,
        help="Slow SMA window (default: 200)",
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

    # --sort selects which metric anchors the table ordering.  choices=
    # whitelists the same five keys BacktestRunner.compare() accepts, so an
    # invalid value is rejected here by argparse before it reaches the runner.
    parser.add_argument(
        "--sort",
        default="sharpe_ratio",
        choices=["sharpe_ratio", "total_return", "max_drawdown", "win_rate", "n_trades"],
        help="Metric to sort by (default: sharpe_ratio)",
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

    # An empty symbol list means there is nothing to backtest.  Print a clear
    # message rather than letting the runner raise its own (more cryptic) error.
    if not symbols:
        print("ERROR: Symbol list is empty. Check --symbols or --universe.")
        return 1

    # ------------------------------------------------------------------
    # 3. Build the strategy.
    # ------------------------------------------------------------------
    # A single SMACrossoverStrategy instance shared across all symbols.
    # Strategy is stateless (pure): generate_signals() reads only the bars
    # it receives and carries no mutable internal state, so the same object
    # can safely evaluate every symbol without reset.
    strategy = SMACrossoverStrategy(fast_window=args.fast, slow_window=args.slow)

    # ------------------------------------------------------------------
    # 4. Read bars for every symbol from DuckDB.
    # ------------------------------------------------------------------
    # One-line status before the I/O so the operator sees what is happening
    # while DuckDB opens the file.
    print(f"Loading bars for {len(symbols)} symbols...")

    bars_dict = load_bars_for_symbols(symbols, args.start, args.end)

    # Report how many symbols had data and the total bar count.  The total
    # gives a quick sanity-check on data coverage without printing per-symbol
    # row counts (which would flood the terminal for large universes).
    total_bars = sum(len(b) for b in bars_dict.values())
    print(
        f"Loaded bars for {len(bars_dict)} symbols ({total_bars} total bars). "
        f"Running backtests with {strategy.name}..."
    )

    # Guard: if every symbol was missing from DuckDB, there is nothing to run.
    if not bars_dict:
        print(
            f"ERROR: No bars found for any symbol in {args.start}–{args.end}. "
            "Run the ingest script first."
        )
        return 1

    # ------------------------------------------------------------------
    # 5. Run the universe backtest.
    # ------------------------------------------------------------------
    # Default BacktestRunner() → default Backtester(initial_capital=1.0,
    # annualization=252); no custom config needed for this comparison.
    runner = BacktestRunner()

    # min_bars = slow + 1 because SMACrossoverStrategy.generate_signals()
    # raises ValueError when len(bars) <= slow_window.  Passing slow+1 here
    # lets run_universe() skip those symbols cleanly before calling the strategy
    # at all, rather than crashing mid-sweep.
    try:
        results = runner.run_universe(
            bars_by_symbol=bars_dict,
            strategy=strategy,
            min_bars=args.slow + 1,
        )
    except ValueError as exc:
        # run_universe raises ValueError with a clear message when every
        # symbol was skipped (all too short for min_bars).  We catch only
        # ValueError here — any other exception is a genuine bug and should
        # propagate so the stack trace is visible.
        print(f"ERROR: {exc}")
        return 1

    # ------------------------------------------------------------------
    # 6. Build the comparison table.
    # ------------------------------------------------------------------
    # For max_drawdown, smaller is better → ascending=True puts the best
    # (lowest drawdown) symbol first.  For every other metric, larger is
    # better → ascending=False (best first).  Encoded as a single equality
    # check to keep the logic compact.
    ascending = args.sort == "max_drawdown"

    # compare() returns a fresh DataFrame; the results list is not mutated.
    df = runner.compare(results, sort_by=args.sort, ascending=ascending)

    # ------------------------------------------------------------------
    # 7. Format for display.
    # ------------------------------------------------------------------
    # Operate on a *copy* so the underlying numeric columns in df stay
    # available for the footer aggregate computations in step 9.  Mutating
    # df directly would silently convert numeric columns to strings,
    # breaking comparisons like df["sharpe"] > 0.
    display = df.copy()

    # total_return: signed percent, 2 decimals.  The "+" format code forces
    # a leading sign on positive values so gainers and losers are visually
    # distinct at a glance.
    display["total_return"] = display["total_return"].apply(lambda x: f"{x*100:+.2f}%")

    # max_drawdown: unsigned percent, 2 decimals.  The stored value is already
    # a positive fraction (BacktestResult contract), so no sign treatment needed.
    display["max_drawdown"] = display["max_drawdown"].apply(lambda x: f"{x*100:.2f}%")

    # win_rate: unsigned percent, 1 decimal.  One fewer decimal than the return
    # columns because win rate is a coarse aggregate over discrete trades —
    # extra precision would imply false accuracy.
    display["win_rate"] = display["win_rate"].apply(lambda x: f"{x*100:.1f}%")

    # sharpe: dimensionless ratio, 2 decimals.  No percent sign and no forced
    # sign formatter — negative Sharpes get the natural "-" prefix, positive
    # ones print unadorned, matching finance convention.
    display["sharpe"] = display["sharpe"].apply(lambda x: f"{x:.2f}")

    # n_trades is already an integer column — pandas renders it cleanly
    # without further formatting, so we leave it alone.

    # ------------------------------------------------------------------
    # 8. Print the header and table.
    # ------------------------------------------------------------------
    # Blank line separates the table block from the progress lines above.
    print()

    # Header names the strategy, date range, symbol count, sort metric,
    # and sort direction — everything the reader needs to interpret the table
    # without re-running the command.
    print(
        f"{strategy.name} from {args.start} to {args.end}, "
        f"{len(results)} symbols, sorted by {args.sort} "
        f"({'ascending' if ascending else 'descending'}):"
    )

    # Second blank line creates breathing room between header and table body.
    print()

    # to_string(index=False) gives a clean fixed-width table without the
    # left-side row-index column — that index is meaningless after sorting.
    print(display.to_string(index=False))

    # Trailing blank line so the shell prompt does not butt up against the
    # last row of the table.
    print()

    # ------------------------------------------------------------------
    # 9. Print the footer summary.
    # ------------------------------------------------------------------
    # The footer answers "is this strategy showing real edge across the
    # universe?"  If 24/25 symbols have positive Sharpe, the strategy
    # probably generalises.  If 13/25, it might be noise or data-snooping.
    # We read from the raw numeric df, not the formatted display copy, so
    # the comparisons are on floats, not strings.

    n_total = len(df)  # total symbols in the table (may be < len(symbols) after skipping)

    # Count symbols with Sharpe > 0: any edge above a random walk.
    n_positive_sharpe = int((df["sharpe"] > 0).sum())

    # Count symbols with Sharpe > 0.5: a commonly used threshold for "meaningful"
    # risk-adjusted performance in daily equity strategies.
    n_sharpe_half = int((df["sharpe"] > 0.5).sum())

    # Count symbols with total_return > 0: a simpler (but noisier) signal than
    # Sharpe because it ignores volatility and duration.
    n_positive_return = int((df["total_return"] > 0).sum())

    # Print the footer as a single line.  Fraction format (X/N) is more
    # informative than a bare count because it conveys the base rate at a glance.
    print(
        f"Summary: {n_positive_sharpe}/{n_total} positive Sharpe, "
        f"{n_sharpe_half}/{n_total} Sharpe > 0.5, "
        f"{n_positive_return}/{n_total} positive return."
    )
    print()

    # ------------------------------------------------------------------
    # 10. Return success.
    # ------------------------------------------------------------------
    # Exit code 0 signals to the calling shell that the run completed without
    # recoverable errors.  Non-zero returns earlier in the function signal
    # failure so cron / CI can branch on $?.
    return 0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
# python -m src.research.compare_universe enters here.  sys.exit() propagates
# the integer return value of main() to the shell as the process exit code.
if __name__ == "__main__":
    sys.exit(main())
