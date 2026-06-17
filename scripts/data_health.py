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
from itertools import pairwise  # adjacent (prev, cur) pairs without copying the list
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

# Bounds on the consecutive close-to-close ratio (current / prior) outside which
# a daily bar is flagged as an extreme jump. A single-session move beyond +100%
# (ratio > 2.0) or −50% (ratio < 0.5) does not happen on a non-leveraged, liquid
# ETF for a genuine market reason — it is almost always an unadjusted split or
# other corporate action leaking into the raw `close` the backtester consumes
# (engine.py reads .close, not .adj_close). If `close` is properly split-adjusted
# this check comes back clean, which is the useful confirmation; any flag is an
# invitation to investigate that symbol, not an automatic verdict.
EXTREME_JUMP_RATIO_LOW: float = 0.5
EXTREME_JUMP_RATIO_HIGH: float = 2.0


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


# A bar series row as consumed by the content checks: (date, close, volume),
# already sorted ascending by date. Kept as a plain tuple (not OHLCVBar) so the
# helpers are trivially unit-testable with hand-built fixtures and never touch
# the DB. main() builds these from the read-only query below.
def find_zero_volume_bars(bars: list[tuple[date, float, int]]) -> list[date]:
    """Return the dates of bars whose volume is exactly zero.

    Pure function — no I/O. On a liquid ETF a zero-volume trading day is almost
    always a data gap or a bad bar (a real halt is rare and worth seeing anyway),
    so we surface the dates rather than silently tolerating them.
    """
    # volume is BIGINT in the schema, so an exact == 0 comparison is correct
    # (no float tolerance needed). Order is preserved from the input series.
    return [bar_date for bar_date, _close, volume in bars if volume == 0]


def find_extreme_jumps(
    bars: list[tuple[date, float, int]],
) -> list[tuple[date, float, float, float]]:
    """Return (date, prior_close, current_close, ratio) for each extreme jump.

    A jump is "extreme" when the consecutive close-to-close ratio falls outside
    [EXTREME_JUMP_RATIO_LOW, EXTREME_JUMP_RATIO_HIGH]. Pure function — no I/O.
    Operates on the close series the caller supplies (main() passes raw close).
    """
    flags: list[tuple[date, float, float, float]] = []

    # Walk adjacent pairs so each comparison is "this bar vs the one before it".
    # pairwise(bars) yields (prev, cur) without materializing a copy of the list.
    for (_prev_date, prev_close, _pv), (cur_date, cur_close, _cv) in pairwise(bars):
        # Guard against a non-positive prior close: it would make the ratio
        # meaningless (division by zero or a sign flip). A close <= 0 is itself
        # corrupt, but it is not this check's job to flag it, so we skip the pair.
        if prev_close <= 0:
            continue

        ratio: float = cur_close / prev_close

        # Outside the band → flag it. Inclusive bounds: a clean ETF sits well
        # inside [0.5, 2.0], so the boundary choice never matters in practice.
        if ratio < EXTREME_JUMP_RATIO_LOW or ratio > EXTREME_JUMP_RATIO_HIGH:
            flags.append((cur_date, prev_close, cur_close, ratio))

    return flags


def should_alert(
    stale: list[tuple[str, date, int]],
    missing: list[str],
    zero_volume: list[tuple[str, int]],
) -> bool:
    """Return True when the run should exit 1 (something actionable / bad data).

    Pure decision function so the exit policy is unit-testable without running
    main(). Deliberately excludes extreme_jumps: that heuristic also fires on
    genuine extreme moves (e.g. GL's real -53% day), so it stays informational
    like short-history. Only unambiguous bad data or behind-data hard-fails.
    """
    return bool(stale or missing or zero_volume)


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
    # Content checks — fetch the ordered (date, close, volume) series per symbol.
    # -----------------------------------------------------------------------
    # Structural checks above only need counts/timestamps; the content checks
    # need the bar values themselves. One ordered query feeds both helpers.
    # WHERE timeframe = '1d': a close-to-close ratio is only meaningful within a
    # single timeframe, and the basket/backtester operate on daily bars — so we
    # never want a 1h bar interleaved into a daily series here. ORDER BY symbol,
    # timestamp guarantees each symbol's list is chronological for the jump walk.
    content_sql = (
        f"SELECT symbol, timestamp, close, volume "
        f"FROM {DUCKDB_TABLE_NAME} WHERE timeframe = '1d'"
    )
    # When a universe filter is active, inspect_symbols holds exactly the symbols
    # we will iterate over below, so push that restriction into SQL rather than
    # loading every symbol's bars and discarding most in Python. With 17 tickers
    # out of 520 that is ~30x less data crossing the DB boundary. Parameterized
    # with ? placeholders (never string-interpolated) to stay injection-safe.
    # When no --universe is given, inspect_symbols is the full DB set, so the
    # unfiltered query is the intended scope and we add no IN clause.
    content_params: list[str] = []
    skip_content = False
    if args.universe is not None:
        if inspect_symbols:
            # Non-empty universe: restrict to exactly those symbols. Parameterized
            # with ? placeholders (never string-interpolated) to stay injection-safe.
            placeholders = ", ".join(["?"] * len(inspect_symbols))
            content_sql += f" AND symbol IN ({placeholders})"
            content_params = inspect_symbols
        else:
            # Universe given but resolves to no symbols → nothing to fetch; skip
            # rather than fall through to a full-DB scan that checks nothing.
            skip_content = True
    content_sql += " ORDER BY symbol, timestamp"

    bars_by_symbol: dict[str, list[tuple[date, float, int]]] = {}
    if not skip_content:
        for sym, ts, close_val, volume_val in conn.execute(content_sql, content_params).fetchall():
            # setdefault appends in query order, which is already chronological.
            bars_by_symbol.setdefault(sym, []).append((ts.date(), close_val, volume_val))

    # ZERO-VOLUME: (symbol, count) for symbols with one or more zero-volume bars.
    zero_volume: list[tuple[str, int]] = []
    # EXTREME-JUMP: one row per flagged jump, (symbol, date, prior, current, ratio).
    extreme_jumps: list[tuple[str, date, float, float, float]] = []

    for symbol in inspect_symbols:
        bars = bars_by_symbol.get(symbol)
        # No bars → already covered by the stale/short/missing sections above.
        if not bars:
            continue

        zero_dates = find_zero_volume_bars(bars)
        if zero_dates:
            zero_volume.append((symbol, len(zero_dates)))

        # Prefix each flagged jump with its symbol for the flat report table.
        for jump_date, prior_close, current_close, ratio in find_extreme_jumps(bars):
            extreme_jumps.append((symbol, jump_date, prior_close, current_close, ratio))

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
    # Section b2 — ZERO-VOLUME BARS (content check).
    # -----------------------------------------------------------------------
    # Sort worst-first so a symbol riddled with zero-volume bars surfaces on top.
    zero_volume.sort(key=lambda row: row[1], reverse=True)

    print(f"ZERO-VOLUME BARS (volume == 0): {len(zero_volume)} symbols affected")
    print("-" * 70)
    if zero_volume:
        print(f"{'SYMBOL'.ljust(10)}{'ZERO-VOL BARS'.rjust(14)}")
        for symbol, count in zero_volume:
            print(f"{symbol.ljust(10)}{str(count).rjust(14)}")
    else:
        print("(none)")
    print()

    # -----------------------------------------------------------------------
    # Section b3 — EXTREME CLOSE-TO-CLOSE JUMPS (content check).
    # -----------------------------------------------------------------------
    print(
        f"EXTREME CLOSE-TO-CLOSE JUMPS "
        f"(ratio outside [{EXTREME_JUMP_RATIO_LOW}, {EXTREME_JUMP_RATIO_HIGH}]): "
        f"{len(extreme_jumps)}"
    )
    print("-" * 70)
    if extreme_jumps:
        # Columns: symbol, the date of the jump, the two closes, and the ratio.
        print(
            f"{'SYMBOL'.ljust(10)}{'DATE'.ljust(14)}"
            f"{'PRIOR'.rjust(12)}{'CURRENT'.rjust(12)}{'RATIO'.rjust(10)}"
        )
        for symbol, jump_date, prior_close, current_close, ratio in extreme_jumps:
            # 2dp on prices, 3dp on the ratio — enough to read a split (e.g. 0.250).
            print(
                f"{symbol.ljust(10)}{str(jump_date).ljust(14)}"
                f"{prior_close:>12.2f}{current_close:>12.2f}{ratio:>10.3f}"
            )
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
        f"{len(missing)} missing from DB | "
        f"{len(zero_volume)} zero-vol | "
        f"{len(extreme_jumps)} extreme-jump"
    )
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Exit code — cron-friendly: zero means "nothing to do".
    # -----------------------------------------------------------------------
    # Exit 1 when there is something the operator should act on. Two categories:
    #   - freshness/coverage: stale or missing symbols (data is behind).
    #   - data integrity: a zero-volume bar (a liquid name should never have one).
    # extreme-jump is a heuristic that also fires on genuine extreme moves (e.g.
    # GL's real -53% day), so it stays informational like short-history. Only
    # unambiguous bad data hard-fails.
    if should_alert(stale, missing, zero_volume):
        sys.exit(1)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
