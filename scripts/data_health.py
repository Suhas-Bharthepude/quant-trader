# data_health.py — read-only diagnostic report for the OHLCV DuckDB store.
#
# This is a diagnostic tool, not a fix-it tool.  It surfaces issues — stale
# symbols, short histories, gaps between the configured universe and the data
# actually present — and leaves the decision to the operator.  When the report
# flags a problem, the right next step is usually to run scripts/backfill_universe.py
# (for missing symbols or short history) or scripts/update_universe.py (for stale data).
#
# The script opens DuckDB in read-only mode, so it is safe to run while another
# process is updating the database — there is no risk of contending for the
# write lock or accidentally mutating data.
#
# Usage:
#   uv run python scripts/data_health.py
#   uv run python scripts/data_health.py --universe sp500
#   uv run python scripts/data_health.py --universe sp500 --stale-days 5
#   uv run python scripts/data_health.py --universe sp500 --verbose

import argparse   # declarative CLI argument parsing
import logging    # consistent INFO-level logging across scripts
import sys        # sys.exit() with cron-friendly exit codes
from datetime import date, datetime  # today's date + per-symbol last_ts handling
from pathlib import Path  # cross-platform file size lookup via Path.stat()

import duckdb  # used directly — DuckDBStore has no read-only mode

from src.data.schema import DUCKDB_TABLE_NAME  # canonical "ohlcv_bars" string
from src.data.universe import load_universe    # YAML universe loader

# ---------------------------------------------------------------------------
# LOGGING — same format as backfill_universe.py / update_universe.py.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)  # module-scoped logger

# Default DB path — matches the DuckDBStore default so this script inspects the
# same file that backfill / update write to without further configuration.
DEFAULT_DB_PATH: str = "data/quant_trader.duckdb"

# Threshold under which a symbol's stored history is considered "short".
# 252 ≈ trading days in a year, the standard cutoff for "less than 1 year".
SHORT_HISTORY_THRESHOLD: int = 252


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Read-only health report for the OHLCV DuckDB store.",
    )

    # --universe is optional — without it, the script inspects every symbol
    # present in the DB; with it, the inspection is restricted to the named
    # universe and the missing-from-DB section is enabled.
    parser.add_argument(
        "--universe",
        default=None,
        type=str,
        help='Optional universe name from config/universe.yaml (e.g. "sp500").',
    )

    # --stale-days controls the threshold for the STALE section.  Default 3 is
    # generous enough to tolerate weekends without flagging Monday-morning runs.
    parser.add_argument(
        "--stale-days",
        default=3,
        type=int,
        dest="stale_days",
        help="A symbol is stale if its last bar is older than this many days (default: 3).",
    )

    # --verbose lifts the truncation cap on the STALE section so the operator
    # can see every stale symbol when investigating a widespread issue.
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all stale symbols instead of just the first 20.",
    )

    return parser.parse_args()


def format_mb(num_bytes: int) -> str:
    """Return a human-readable MB string with one decimal place."""
    # 1024*1024 is the binary megabyte; consistent with how `du -h` reports.
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def days_since(ts: datetime, today: date) -> int:
    """Return the integer number of calendar days between ts.date() and today."""
    # ts may be a UTC-aware datetime; .date() drops the time component cleanly
    # so the subtraction yields a timedelta in whole days.
    return (today - ts.date()).days


def main() -> None:
    """Entry point — open DB read-only, gather stats, print the report."""
    args = parse_args()

    # -----------------------------------------------------------------------
    # Open DuckDB in read-only mode.
    # -----------------------------------------------------------------------
    # read_only=True is the critical safety flag here: it lets this script
    # run while another process holds the write lock, and it guarantees we
    # cannot accidentally mutate the data even if a query is malformed.
    db_path = Path(DEFAULT_DB_PATH)

    # Fail fast with a friendly error if the DB file does not exist yet —
    # otherwise duckdb.connect would create an empty file in read-only mode
    # which would then look like "DB exists but has zero rows", a confusing state.
    if not db_path.exists():
        log.error("DB file not found: %s — run backfill_universe.py first.", db_path)
        sys.exit(1)

    # Open the connection.  Connection is closed implicitly when the script exits.
    conn = duckdb.connect(str(db_path), read_only=True)

    # -----------------------------------------------------------------------
    # Determine which symbols to inspect.
    # -----------------------------------------------------------------------
    # If --universe was given, restrict the per-symbol diagnostics to that
    # universe; otherwise inspect every symbol that has at least one bar in DB.
    if args.universe is not None:
        # load_universe raises KeyError for unknown names — let it surface.
        universe_tickers: list[str] = load_universe(args.universe)
        # The set of symbols we will iterate over for per-symbol diagnostics.
        inspect_symbols: list[str] = universe_tickers
    else:
        # No universe filter — pull every distinct symbol present in the table.
        rows = conn.execute(
            f"SELECT DISTINCT symbol FROM {DUCKDB_TABLE_NAME} ORDER BY symbol"
        ).fetchall()
        # Each row is a single-element tuple; extract index 0 to get the string.
        universe_tickers = []  # not applicable when --universe is omitted
        inspect_symbols = [row[0] for row in rows]

    # -----------------------------------------------------------------------
    # Header section — overall DB-wide stats (unfiltered by --universe).
    # -----------------------------------------------------------------------
    # Single aggregate query: total rows, distinct symbols, MIN/MAX timestamp.
    overall_sql = (
        f"SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(timestamp), MAX(timestamp) "
        f"FROM {DUCKDB_TABLE_NAME}"
    )
    total_bars, distinct_symbols, db_first_ts, db_last_ts = conn.execute(
        overall_sql
    ).fetchone()

    # File size on disk — useful trend metric over time as data accumulates.
    file_size_str = format_mb(db_path.stat().st_size)

    # Print the header — operator-facing report, not a log line.
    print("=" * 70)
    print("DuckDB OHLCV Health Report")
    print("=" * 70)
    print(f"DB file:           {db_path}  ({file_size_str})")
    print(f"Total bars:        {total_bars:,}")
    print(f"Distinct symbols:  {distinct_symbols:,}")
    # Guard against an empty DB — MIN/MAX on zero rows return None.
    if db_first_ts is not None and db_last_ts is not None:
        print(f"Date range:        {db_first_ts.date()} to {db_last_ts.date()}")
    else:
        print("Date range:        (no data)")
    if args.universe:
        # Show the universe name and size when filtering, so the report's scope
        # is unambiguous.
        print(f"Universe filter:   {args.universe} ({len(universe_tickers)} tickers)")
    print()

    # -----------------------------------------------------------------------
    # Per-symbol stats query.
    # -----------------------------------------------------------------------
    # One aggregate query computes everything we need across all symbols; we
    # then filter the result in Python to the inspect_symbols set.  This is
    # cheaper than running 500 separate queries.
    per_symbol_sql = (
        f"SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp) "
        f"FROM {DUCKDB_TABLE_NAME} GROUP BY symbol"
    )
    per_symbol_rows = conn.execute(per_symbol_sql).fetchall()

    # Build a {symbol: (count, first_ts, last_ts)} dict for O(1) lookup below.
    stats: dict[str, tuple[int, datetime, datetime]] = {
        row[0]: (row[1], row[2], row[3]) for row in per_symbol_rows
    }

    # Today's calendar date — used as the reference point for staleness.
    today: date = date.today()

    # -----------------------------------------------------------------------
    # Build the diagnostic lists.
    # -----------------------------------------------------------------------
    # STALE: symbols whose latest bar is older than today minus stale_days.
    # Only consider symbols actually present in the DB (stats lookup).
    stale: list[tuple[str, date, int]] = []
    # SHORT-HISTORY: symbols with bar count below the 252-day cutoff.
    short_history: list[tuple[str, int, date]] = []

    for symbol in inspect_symbols:
        # If the symbol has no rows at all, it is handled by the missing
        # section below — not by stale or short-history.
        if symbol not in stats:
            continue

        bar_count, first_ts, last_ts = stats[symbol]

        # Days since the most recent stored bar — the staleness metric.
        d_stale: int = days_since(last_ts, today)

        # Compare against the configured threshold.
        if d_stale > args.stale_days:
            stale.append((symbol, last_ts.date(), d_stale))

        # Independent check: history shorter than ~1 trading year.
        if bar_count < SHORT_HISTORY_THRESHOLD:
            short_history.append((symbol, bar_count, first_ts.date()))

    # MISSING: only computed when --universe was given, since it requires a
    # known set of expected symbols to compare against.
    missing: list[str] = []
    if args.universe is not None:
        # Symbols listed in the YAML but absent from the per-symbol stats dict.
        # Sorting keeps the output stable across runs.
        missing = sorted(set(universe_tickers) - set(stats.keys()))

    # -----------------------------------------------------------------------
    # Section a — STALE SYMBOLS.
    # -----------------------------------------------------------------------
    # Sort by most stale first so the worst offenders surface at the top.
    stale.sort(key=lambda row: row[2], reverse=True)

    print(f"STALE SYMBOLS (last bar > {args.stale_days} days old): {len(stale)}")
    print("-" * 70)
    if stale:
        # Header row for the columnar layout.
        print(
            f"{'SYMBOL'.ljust(10)}{'LAST DATE'.ljust(14)}{'DAYS STALE'.rjust(10)}"
        )
        # Cap output at 20 rows unless --verbose is set, to keep cron-mail readable.
        cap = len(stale) if args.verbose else 20
        for symbol, last_date, d in stale[:cap]:
            # ljust pads symbol/date to fixed widths; rjust right-aligns the count.
            print(f"{symbol.ljust(10)}{str(last_date).ljust(14)}{str(d).rjust(10)}")
        # If the cap was hit, tell the operator how to see the rest.
        if not args.verbose and len(stale) > cap:
            print(f"... and {len(stale) - cap} more (re-run with --verbose to see all)")
    else:
        print("(none)")
    print()

    # -----------------------------------------------------------------------
    # Section b — SHORT-HISTORY SYMBOLS.
    # -----------------------------------------------------------------------
    # Sort by fewest bars first — the shortest histories are the most likely
    # to be recent IPOs or accidentally truncated symbols.
    short_history.sort(key=lambda row: row[1])

    print(
        f"SHORT-HISTORY SYMBOLS (< {SHORT_HISTORY_THRESHOLD} bars): "
        f"{len(short_history)}"
    )
    print("-" * 70)
    if short_history:
        # Column header for the short-history table.
        print(
            f"{'SYMBOL'.ljust(10)}{'BARS'.rjust(8)}  {'FIRST DATE'.ljust(14)}"
        )
        # Cap at 10 — these tend to be a long tail of small or recent symbols
        # and do not need full enumeration to be actionable.
        for symbol, count, first_date in short_history[:10]:
            print(
                f"{symbol.ljust(10)}{str(count).rjust(8)}  "
                f"{str(first_date).ljust(14)}"
            )
        # Tail indicator if truncated.
        if len(short_history) > 10:
            print(f"... and {len(short_history) - 10} more")
    else:
        print("(none)")
    print()

    # -----------------------------------------------------------------------
    # Section c — IN UNIVERSE BUT NOT IN DB (only when --universe given).
    # -----------------------------------------------------------------------
    if args.universe is not None:
        print(f"MISSING FROM DB (in universe '{args.universe}', no bars stored): "
              f"{len(missing)}")
        print("-" * 70)
        if missing:
            # Print 6 symbols per line in fixed-width columns to keep the report
            # compact even when dozens are missing.
            line_buffer: list[str] = []
            for sym in missing:
                # Pad each symbol to 10 chars so the columns line up cleanly.
                line_buffer.append(sym.ljust(10))
                if len(line_buffer) == 6:
                    print("".join(line_buffer))
                    line_buffer = []
            # Flush any trailing partial row of symbols.
            if line_buffer:
                print("".join(line_buffer))
            # Operator hint pointing at the canonical fix.
            print()
            print("→ Run: uv run python scripts/backfill_universe.py "
                  f"--universe {args.universe} --years <N>")
        else:
            print("(none)")
        print()

    # -----------------------------------------------------------------------
    # One-line summary suitable for cron-mail subject lines / Slack snippets.
    # -----------------------------------------------------------------------
    print("=" * 70)
    print(
        f"{distinct_symbols} symbols | "
        f"{len(stale)} stale | "
        f"{len(short_history)} short-history | "
        f"{len(missing)} missing from DB"
    )
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Exit code — cron-friendly: zero means "nothing to do".
    # -----------------------------------------------------------------------
    # Exit 1 when there is something the operator should act on (stale or
    # missing); exit 0 otherwise.  Short-history is informational, not an
    # alert, so it does not affect the exit code.
    if stale or missing:
        sys.exit(1)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
