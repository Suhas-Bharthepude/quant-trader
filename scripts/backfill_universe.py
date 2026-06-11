# backfill_universe.py — bulk-ingest N years of daily OHLCV history for a
# named universe into DuckDB.
#
# This is the canonical entry point for the first-time or from-scratch ingest.
# Daily incremental updates (Day 12) will be a separate, lighter script that
# only fetches bars since the last stored date per symbol.  Run this once per
# universe to seed the database; re-run any time you want to extend history.
#
# Usage:
#   uv run python scripts/backfill_universe.py --universe test --years 5
#   uv run python scripts/backfill_universe.py --universe sp500 --years 10
#   uv run python scripts/backfill_universe.py --universe etf_basket --start 2008-01-01
#
# Rate-limit note: yfinance allows ~2 000 requests/hour.  500 tickers is
# well within that budget.  If you ever scale past 5 000 symbols, uncomment
# the time.sleep(0.1) line inside the loop to throttle automatically.

import argparse   # declarative CLI argument parsing
import logging    # per-symbol INFO / WARNING messages
import sys        # sys.exit() with specific exit codes
from datetime import date, timedelta  # compute start/end from --years

from tqdm import tqdm  # progress bar with ETA; install with: uv add tqdm

from src.data.duckdb_store import DuckDBStore        # persistence layer
from src.data.universe import load_universe          # YAML universe loader
from src.data.yfinance_fetcher import YFinanceFetcher  # data source

# ---------------------------------------------------------------------------
# LOGGING — same pattern as other scripts in this repo.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # logger scoped to this module


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill daily OHLCV history for a named universe into DuckDB.",
    )

    # --universe is required — must match a key in config/universe.yaml.
    parser.add_argument(
        "--universe",
        required=True,
        type=str,
        help='Universe name from config/universe.yaml, e.g. "test" or "sp500".',
    )

    # The lookback window can be specified one of two mutually-exclusive ways.
    # Exactly one is required (preserving the original "must specify a range"
    # guarantee that --years=required previously enforced on its own).
    window = parser.add_mutually_exclusive_group(required=True)

    # --years: original behavior, unchanged — fetch the last N years from today.
    window.add_argument(
        "--years",
        type=int,
        help="Number of years of history to fetch (e.g. 5 → today minus 1825 days).",
    )

    # --start: explicit ISO start date for deep backfills where an exact
    # calendar anchor matters more than "N years back from today" (e.g. seeding
    # the ETF basket from 2008-01-01). End defaults to today unless --end given.
    # type=date.fromisoformat validates the format at parse time — a malformed
    # date fails with a clear argparse error here instead of dying deep in the
    # fetcher, and hands main() a real date object rather than a raw string.
    window.add_argument(
        "--start",
        type=date.fromisoformat,
        help="Explicit ISO start date YYYY-MM-DD (alternative to --years).",
    )

    # --end: optional explicit ISO end date; only meaningful alongside --start.
    # Defaults to today, matching the --years path. Same parse-time validation
    # as --start (see above).
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="Optional ISO end date YYYY-MM-DD; defaults to today.",
    )

    # --end has no meaning on its own — it only narrows an explicit --start
    # window. The mutually-exclusive group can't express "end depends on start",
    # so enforce it here while the parser is still in scope to emit the standard
    # argparse usage/error and exit(2).
    args = parser.parse_args()
    if args.end is not None and args.start is None:
        parser.error("--end is only valid together with --start")
    return args


def compute_date_range(years: int) -> tuple[str, str]:
    """Return (start, end) as ISO strings covering the last N years up to today."""
    # Today's date — no timezone needed; yfinance accepts plain date strings.
    today: date = date.today()

    # Approximate years as 365 * years days — leap years shift this by at most
    # one day per four years, which is acceptable for a backfill window.
    start_date: date = today - timedelta(days=365 * years)

    # Convert to YYYY-MM-DD strings — both YFinanceFetcher and DuckDBStore
    # accept ISO date strings and handle the parsing internally.
    return start_date.isoformat(), today.isoformat()


def resolve_window(
    years: int | None, start: date | None, end: date | None
) -> tuple[str, str]:
    """Resolve (start_iso, end_iso). Explicit --start wins; else fall back to the --years window."""
    # Pure (modulo date.today()) and side-effect-free so it can be unit-tested
    # without argparse: feed it the three parsed values, assert the ISO pair.
    if start is not None:
        # Explicit window: end defaults to today when --end was omitted.
        return start.isoformat(), (end or date.today()).isoformat()
    # start is None here, so years must be set. Guard makes this explicit:
    # it narrows int|None -> int for the type checker AND turns a misuse
    # like resolve_window(None, None, None) into a clear error at the
    # boundary instead of a cryptic TypeError inside compute_date_range.
    if years is None:
        raise ValueError("resolve_window requires either start or years")
    return compute_date_range(years)


def main() -> None:
    """Entry point — parse args, load universe, fetch, store, summarise."""
    args = parse_args()

    # -----------------------------------------------------------------------
    # Date range
    # -----------------------------------------------------------------------
    # All window logic lives in resolve_window (testable in isolation); the
    # mutually-exclusive group guarantees exactly one of years/start is set.
    start, end = resolve_window(args.years, args.start, args.end)

    # -----------------------------------------------------------------------
    # Universe
    # -----------------------------------------------------------------------
    # load_universe raises KeyError for unknown names and ValueError for empty
    # lists — both are user errors; let them propagate as plain tracebacks.
    tickers: list[str] = load_universe(args.universe)

    # User-facing header so the operator knows what is about to run.
    print(f"Universe: {args.universe} — {len(tickers)} tickers")
    print(f"Date range: {start} to {end}")

    # -----------------------------------------------------------------------
    # Infrastructure
    # -----------------------------------------------------------------------
    fetcher = YFinanceFetcher()  # stateless; no connection to open

    # -----------------------------------------------------------------------
    # Ingest loop
    # -----------------------------------------------------------------------
    # Counters accumulate across the loop to build the final summary line.
    total_bars: int = 0      # bars returned by yfinance (fetched)
    total_inserted: int = 0  # bars actually written (non-duplicates)
    success_count: int = 0   # symbols that fetched without error

    # Keep DuckDB open for the entire loop — opening and closing per ticker
    # acquires and releases the file lock 500 times, which is ~500x slower.
    with DuckDBStore() as store:
        # tqdm wraps the iterable and draws a progress bar with ETA in the
        # terminal.  desc= sets the label shown to the left of the bar.
        for symbol in tqdm(tickers, desc="Backfilling"):
            try:
                # Fetch all bars for this symbol in the computed date range.
                bars = fetcher.fetch_daily(symbol, start, end)

                # Write to DuckDB; write_bars returns the count of new rows
                # (duplicates are silently skipped via INSERT OR IGNORE).
                inserted: int = store.write_bars(bars)

                # Accumulate totals for the summary printed after the loop.
                total_bars += len(bars)       # total fetched (including dupes)
                total_inserted += inserted    # total new rows written
                success_count += 1            # this symbol succeeded

                log.info(
                    "%s: fetched %d bars, wrote %d new",
                    symbol, len(bars), inserted,
                )

                # Uncomment if fetching >5 000 symbols to stay under rate limit:
                # time.sleep(0.1)

            except Exception as exc:  # noqa: BLE001 — isolate per-symbol failures
                # Log and continue — one bad ticker should not abort the batch.
                log.warning("Skipping %s: %s", symbol, exc)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total: int = len(tickers)  # total symbols attempted
    duplicates: int = total_bars - total_inserted  # bars already in the DB

    # Direct prints for the operator-facing summary (tqdm flushes stdout first).
    print(
        f"\nDone. Fetched {total_bars} bars across "
        f"{success_count}/{total} symbols."
    )
    print(
        f"Wrote {total_inserted} new bars to DuckDB "
        f"({duplicates} duplicates skipped)."
    )

    # Exit 1 only if every single symbol failed — a partial success is still
    # useful and should not fail a shell pipeline.
    if success_count == 0:
        log.error("All %d symbols failed — check connectivity and ticker list.", total)
        sys.exit(1)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
