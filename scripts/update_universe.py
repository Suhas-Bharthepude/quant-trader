# update_universe.py — daily incremental OHLCV update for a named universe.
#
# This is the fast, cron-friendly companion to backfill_universe.py.  Instead
# of re-fetching years of history, it reads each symbol's latest stored
# timestamp from DuckDB and only requests bars that postdate that timestamp
# (minus a configurable lookback window for late-published corrections).
#
# Run backfill_universe.py first to seed the DB with historical data; then
# schedule this script to run daily to keep the data current.  A full S&P 500
# update typically completes in ~60 seconds because each fetch covers only the
# past few days rather than years.
#
# Usage:
#   uv run python scripts/update_universe.py --universe sp500
#   uv run python scripts/update_universe.py --universe test
#   uv run python scripts/update_universe.py --universe sp500 --lookback-days 14

import argparse   # declarative CLI argument parsing
import logging    # per-symbol INFO / WARNING messages
import sys        # sys.exit() with specific exit codes
from datetime import date, timedelta  # compute per-symbol date windows

from tqdm import tqdm  # progress bar with ETA; install with: uv add tqdm

from src.data.duckdb_store import DuckDBStore          # persistence layer
from src.data.universe import load_universe            # YAML universe loader
from src.data.yfinance_fetcher import YFinanceFetcher  # data source

# ---------------------------------------------------------------------------
# LOGGING — same format string as backfill_universe.py for consistency.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # logger scoped to this module

# The earliest date we will ever request from yfinance.  Prevents absurdly
# large fetches when a symbol has a very old or missing last_timestamp.
FLOOR_DATE: date = date(2010, 1, 1)


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Incrementally update daily OHLCV bars for a named universe.",
    )

    # --universe is required — must match a key in config/universe.yaml.
    parser.add_argument(
        "--universe",
        required=True,
        type=str,
        help='Universe name from config/universe.yaml, e.g. "test" or "sp500".',
    )

    # --lookback-days is optional — controls how many days before last_timestamp
    # we re-fetch to capture late-published bars or price corrections.  Duplicate
    # rows are silently discarded by the INSERT OR IGNORE constraint in write_bars.
    parser.add_argument(
        "--lookback-days",
        default=7,
        type=int,
        dest="lookback_days",
        help="Days before each symbol's last stored date to start fetching (default: 7).",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point — parse args, load universe, update each symbol, summarise."""
    args = parse_args()

    # -----------------------------------------------------------------------
    # Universe
    # -----------------------------------------------------------------------
    # load_universe raises KeyError for unknown names — let it propagate as a
    # plain traceback so the operator sees the exact bad name.
    tickers: list[str] = load_universe(args.universe)

    # Operator-facing header confirms which universe and how many symbols loaded.
    print(f"Universe: {args.universe} — {len(tickers)} tickers")

    # -----------------------------------------------------------------------
    # Infrastructure
    # -----------------------------------------------------------------------
    fetcher = YFinanceFetcher()  # stateless; no persistent connection to open

    # -----------------------------------------------------------------------
    # Date ceiling — shared across all symbols; each symbol computes its own
    # floor from last_timestamp, but every fetch ends at today.
    # -----------------------------------------------------------------------
    end: str = date.today().isoformat()  # YYYY-MM-DD ceiling for all fetches

    # -----------------------------------------------------------------------
    # Counters — accumulated across the loop to build the final summary.
    # -----------------------------------------------------------------------
    success_count: int = 0    # symbols whose fetch+write completed without error
    total_inserted: int = 0   # total new rows written across all successful symbols
    up_to_date_count: int = 0 # symbols whose last stored date is already >= today
    skipped_count: int = 0    # symbols with no stored data (need backfill first)
    failed_count: int = 0     # symbols that raised an exception during fetch/write

    # Keep DuckDB open for the entire loop — a single held connection is far
    # faster than acquiring and releasing the file lock once per symbol.
    with DuckDBStore() as store:
        # tqdm wraps the ticker list and draws an ETA progress bar in the terminal.
        for symbol in tqdm(tickers, desc="Updating"):
            # ------------------------------------------------------------------
            # Step 1: look up the latest stored timestamp for this symbol.
            # ------------------------------------------------------------------
            # last_timestamp() returns a UTC-aware datetime, or None if the symbol
            # has no rows in ohlcv_bars (i.e. backfill was never run for it).
            last_ts = store.last_timestamp(symbol)

            # ------------------------------------------------------------------
            # Step 2: handle the "not in DB" case — requires backfill, not update.
            # ------------------------------------------------------------------
            if last_ts is None:
                # This symbol has no history in DuckDB.  The incremental updater
                # cannot synthesise a start date from nothing; skip and tell the
                # operator to run backfill_universe.py for this symbol first.
                log.info("%s: no stored bars — run backfill_universe.py first", symbol)
                skipped_count += 1  # tally so the summary reflects this clearly
                continue            # move on to the next symbol immediately

            # ------------------------------------------------------------------
            # Step 3: compute the fetch window for this symbol.
            # ------------------------------------------------------------------
            # Subtract lookback_days from the calendar date of the last stored bar
            # (not from today) so we re-fetch any recently corrected bars near the
            # tail of the stored history.
            raw_start: date = last_ts.date() - timedelta(days=args.lookback_days)

            # Clamp to FLOOR_DATE so we never request data before yfinance's
            # reliable history begins, even if lookback_days is very large.
            start_date: date = max(raw_start, FLOOR_DATE)

            # Convert to ISO string — the format expected by fetch_daily and
            # the same format used by the end variable above.
            start: str = start_date.isoformat()

            # ------------------------------------------------------------------
            # Step 4: skip symbols that are already fully current.
            # ------------------------------------------------------------------
            # ISO date string comparison is lexicographically equivalent to
            # chronological comparison for YYYY-MM-DD formatted strings.
            if start >= end:
                # The computed start is today or in the future — nothing new to
                # fetch; the stored history is already up-to-date.
                log.info("%s: already current (last bar %s)", symbol, last_ts.date())
                up_to_date_count += 1  # tally for the summary
                continue               # nothing to write; move to next symbol

            # ------------------------------------------------------------------
            # Step 5: fetch and store new bars.
            # ------------------------------------------------------------------
            try:
                # Request only the bars in [start, end] — a window of a few days
                # for most symbols, not years of history like backfill does.
                bars = fetcher.fetch_daily(symbol, start, end)

                # write_bars uses INSERT OR IGNORE, so any bars overlapping the
                # lookback window that are already stored are silently discarded.
                inserted: int = store.write_bars(bars)

                # Accumulate running totals for the post-loop summary.
                total_inserted += inserted  # count of genuinely new rows
                success_count += 1         # this symbol completed without error

                log.info(
                    "%s: fetched %d bars (start %s), wrote %d new",
                    symbol, len(bars), start, inserted,
                )

            except Exception as exc:  # noqa: BLE001 — isolate per-symbol failures
                # One bad ticker (delisted, API hiccup, bad data) should not abort
                # the entire run.  Log a warning and keep the loop going.
                log.warning("Failed %s: %s", symbol, exc)
                failed_count += 1  # tally so the operator can see what broke

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    # Print an operator-facing summary that mirrors the style of backfill_universe.
    print(f"\nUniverse: {args.universe}")
    print(
        f"Date range: dynamic per symbol "
        f"(latest stored − {args.lookback_days} days → today)"
    )
    print(f"Updated:         {success_count} symbols ({total_inserted} new bars)")
    print(f"Already current: {up_to_date_count}")
    print(f"Skipped (not in DB): {skipped_count}")
    print(f"Failed:          {failed_count}")

    # Exit 1 only if every single symbol ended up in the failed bucket — a run
    # where some symbols are current or skipped is still a successful execution.
    if failed_count == len(tickers):
        log.error("All %d symbols failed — check connectivity and ticker list.", len(tickers))
        sys.exit(1)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
