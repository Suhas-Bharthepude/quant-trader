# src/research/compare_strategies.py

"""
CLI: compare SMA crossover parameter combinations on a single symbol.

Run via:
    python -m src.research.compare_strategies
    python -m src.research.compare_strategies --symbol AAPL --start 2018-01-01

Default behavior: SPY from 2021-01-01 to today, sweep across 6 reasonable
(fast, slow) parameter pairs spanning shorter-horizon (10, 30) to canonical
(50, 200) — the comparison illustrates the speed/stability tradeoff inherent
to the SMA crossover family.

CLI args:
    --symbol  default SPY
    --start   default 2021-01-01
    --end     default 2026-05-14
    --sort    default sharpe_ratio (one of: sharpe_ratio, total_return,
              max_drawdown, win_rate, n_trades)

This is a thin glue layer: it reads bars from DuckDB, builds a list of
strategies, hands them to BacktestRunner, then formats the resulting
DataFrame for terminal display.  No backtest logic lives here — all
computation is delegated to the runner and the underlying engine.
"""

# argparse is the standard-library CLI argument parser; consistent with the
# scripts/ directory's existing CLI tools (data_health.py, backfill_universe.py).
import argparse

# sys.exit() with an integer code makes this script cron-friendly: 0 means
# success, non-zero means the caller should investigate.
import sys

# pandas is the format the runner's compare() returns; we mutate a *copy*
# of the frame for display so the original numeric values stay usable.
import pandas as pd

# DuckDBStore is the read side of the persistence layer.  We use it as a
# context manager so the file lock is released even if an exception fires.
from src.data.duckdb_store import DuckDBStore

# SMACrossoverStrategy is the only strategy this CLI compares.  Phase 2 may
# generalize this to multiple strategy classes; for now we sweep parameters
# within one family because that's already a meaningful comparison.
from src.strategies.sma_crossover import SMACrossoverStrategy

# BacktestRunner is the orchestration layer defined in src/research/runner.py.
# It runs N strategies against the same bars and aggregates the results.
from src.research.runner import BacktestRunner


def main() -> int:
    """Entry point — parse args, run backtests, print the comparison table.

    Returns an integer exit code: 0 on success, 1 on a recoverable error
    (e.g. not enough bars in DB for the requested slowest window).
    """

    # ------------------------------------------------------------------
    # 1. Parse CLI arguments.
    # ------------------------------------------------------------------
    # argparse converts the command-line tokens into a structured Namespace.
    # Each argument has a default so the bare command works for the common
    # "show me SPY from 2021 to today" case with no flags.
    parser = argparse.ArgumentParser(
        description="Compare SMA crossover strategies on one symbol."
    )

    # --symbol is the ticker to backtest.  Default SPY because it has long
    # history, deep liquidity, and is the canonical benchmark to compare against.
    parser.add_argument(
        "--symbol",
        default="SPY",
        help="Symbol to backtest (default: SPY)",
    )

    # --start is the inclusive start date.  2021-01-01 gives ~5 years of
    # history through the default end date — enough bars for even the
    # (50, 200) crossover to warm up and produce many signal transitions.
    parser.add_argument(
        "--start",
        default="2021-01-01",
        help="Start date YYYY-MM-DD",
    )

    # --end is the inclusive end date.  Hard-coded default 2026-05-14
    # matches the project's current date and makes runs reproducible —
    # using date.today() would silently change results day to day.
    parser.add_argument(
        "--end",
        default="2026-05-14",
        help="End date YYYY-MM-DD",
    )

    # --sort selects which metric anchors the table's ordering.  choices=
    # whitelists the same five keys that BacktestRunner.compare() accepts,
    # so an invalid value is caught by argparse with a friendly message
    # rather than surfacing as a ValueError from inside compare().
    parser.add_argument(
        "--sort",
        default="sharpe_ratio",
        choices=["sharpe_ratio", "total_return", "max_drawdown", "win_rate", "n_trades"],
        help="Metric to sort by (default: sharpe_ratio)",
    )

    # parse_args() reads sys.argv; on bad input it prints usage and exits
    # the process before returning, so we don't need to handle that case.
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Define the parameter grid.
    # ------------------------------------------------------------------
    # Six (fast, slow) pairs, all valid (slow > fast).  The list spans:
    #   * very short:  (10, 30)     — reacts fast, lots of whipsaw
    #   * short:       (10, 50), (20, 50)
    #   * medium:      (20, 100), (50, 100)
    #   * canonical:   (50, 200)    — the classic "golden cross"
    # The spread illustrates the speed/stability tradeoff in a single table.
    param_grid = [
        (10, 30),
        (10, 50),
        (20, 50),
        (20, 100),
        (50, 100),
        (50, 200),
    ]

    # Build one SMACrossoverStrategy per (fast, slow) pair.  List comprehension
    # because each construction is a single expression — no intermediate
    # state to track between iterations.
    strategies = [
        SMACrossoverStrategy(fast_window=fast, slow_window=slow)
        for fast, slow in param_grid
    ]

    # ------------------------------------------------------------------
    # 3. Read bars from DuckDB.
    # ------------------------------------------------------------------
    # Print a status line before the I/O so the operator sees what's happening
    # while DuckDB opens the file — the open is fast but the human-perceived
    # latency benefits from a "loading…" cue regardless.
    print(f"Loading {args.symbol} bars from {args.start} to {args.end}...")

    # Context manager guarantees the connection is closed (and the write
    # lock released) even if read_bars raises.  We only need read access,
    # but DuckDBStore's default open mode is read/write — that's fine here
    # because the script is the only process touching the file.
    with DuckDBStore() as store:
        # read_bars returns bars sorted by timestamp ascending — exactly
        # the order the strategies and backtester expect.
        bars = store.read_bars(args.symbol, args.start, args.end)

    # ------------------------------------------------------------------
    # 4. Guard: enough bars for the slowest strategy's warmup window?
    # ------------------------------------------------------------------
    # Compute the largest slow_window in the grid — that's the strictest
    # data requirement.  SMACrossoverStrategy raises ValueError if it has
    # fewer than slow_window bars; surfacing that here with a friendly
    # message is more useful than letting it bubble up from the runner loop.
    max_slow = max(slow for _, slow in param_grid)
    if len(bars) <= max_slow:
        # Print to stdout (not stderr) because this is the operator-facing
        # report channel.  Return 1 so a calling shell script can detect
        # the failure via $?.
        print(
            f"ERROR: Only {len(bars)} bars available for {args.symbol}; "
            f"need more than {max_slow} for the slowest strategy."
        )
        return 1

    # Status line — shows the row count actually loaded so the operator
    # can sanity-check it against the requested date range.
    print(f"Loaded {len(bars)} bars. Running {len(strategies)} backtests...")

    # ------------------------------------------------------------------
    # 5. Run all backtests through the runner.
    # ------------------------------------------------------------------
    # Default BacktestRunner() → default Backtester(initial_capital=1.0,
    # annualization=252).  No custom config needed for this comparison.
    runner = BacktestRunner()

    # run_many returns a list of BacktestResult in the same order as the
    # input strategies.  We don't need that ordering downstream (compare()
    # re-sorts), but it's preserved for callers that do.
    results = runner.run_many(bars, strategies)

    # ------------------------------------------------------------------
    # 6. Build the comparison table.
    # ------------------------------------------------------------------
    # For max_drawdown, smaller is better → ascending=True so the row with
    # the smallest drawdown appears first.  For every other metric, larger
    # is better → ascending=False (best-first).  Encoded as a single
    # equality check rather than a separate if/else to keep the logic compact.
    ascending = args.sort == "max_drawdown"

    # compare() returns a fresh DataFrame; it does not mutate the results list.
    df = runner.compare(results, sort_by=args.sort, ascending=ascending)

    # ------------------------------------------------------------------
    # 7. Format for display.
    # ------------------------------------------------------------------
    # We intentionally operate on a *copy* of the frame so the underlying
    # numeric values stay available to any downstream caller (e.g. a future
    # CSV export or plotting step).  Mutating the original would silently
    # convert the float columns to strings.
    display = df.copy()

    # total_return: signed percent, 2 decimals.  The "+" in :+ forces a
    # leading sign on positive values too, so the column visually distinguishes
    # gainers from losers at a glance.
    display["total_return"] = display["total_return"].apply(lambda x: f"{x*100:+.2f}%")

    # max_drawdown: unsigned percent, 2 decimals.  The stored value is already
    # a positive fraction (see BacktestResult), so no sign treatment needed.
    display["max_drawdown"] = display["max_drawdown"].apply(lambda x: f"{x*100:.2f}%")

    # win_rate: unsigned percent, 1 decimal.  One fewer decimal than the
    # return columns because win rate is already a coarse aggregate over
    # discrete trades — more precision would imply false accuracy.
    display["win_rate"] = display["win_rate"].apply(lambda x: f"{x*100:.1f}%")

    # sharpe: dimensionless ratio, 2 decimals.  No percent sign and no
    # forced-sign formatter — negative Sharpes get the natural "-" prefix
    # and positive ones print unadorned, which matches finance convention.
    display["sharpe"] = display["sharpe"].apply(lambda x: f"{x:.2f}")

    # n_trades is already an integer column — pandas renders it cleanly
    # without further formatting, so we leave it alone.

    # ------------------------------------------------------------------
    # 8. Print the table.
    # ------------------------------------------------------------------
    # Blank line separates the table block from the "Loaded N bars…" status
    # lines above so the report reads as a distinct section.
    print()
    # Header tells the operator exactly which metric anchors the ordering
    # and which direction was applied — important because some metrics
    # (max_drawdown) sort ascending, the rest descending, and a reader
    # glancing at the table shouldn't have to remember that convention.
    print(f"Comparison sorted by {args.sort} ({'ascending' if ascending else 'descending'}):")
    # Second blank line creates breathing room between header and table.
    print()
    # to_string(index=False) gives a clean fixed-width table without the
    # left-side row-index column — that index is meaningless after sorting.
    print(display.to_string(index=False))
    # Trailing blank line so a subsequent shell prompt doesn't butt up
    # against the table's last row.
    print()

    # Success — exit code 0.
    return 0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
# python -m src.research.compare_strategies enters here.  sys.exit() with
# the return value of main() propagates the exit code to the shell so cron
# / CI can branch on success vs. failure.
if __name__ == "__main__":
    sys.exit(main())
