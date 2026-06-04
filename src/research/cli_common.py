# src/research/cli_common.py

"""
Shared helpers for the research CLI layer.

Both compare_universe.py and compare_matrix.py perform two identical
operations before their CLI-specific logic begins:

  1. Resolve a symbol list — either from a raw CSV string or from a named
     universe config, optionally capped to a limit.

  2. Load bars for every symbol from DuckDB in a single connection, skipping
     symbols with no data while logging a warning for each one skipped.

Extracting these two operations here gives three benefits:
  * Unit-testable without constructing an argparse Namespace.
  * A single place to fix bugs or change DuckDB open semantics.
  * The CLIs themselves become thin glue that calls helpers and owns only
    the parts that genuinely differ (error messages, runner calls, display).

Nothing in this module prints to stdout — all user-facing output is left to
the caller, because the wording diverges across CLIs and should stay owned
by the CLI that knows what it wants to say.
"""

# logging is used in load_bars_for_symbols to warn about symbols with no data.
# Using log.warning() rather than print() keeps diagnostics on stderr (or
# whatever handler the caller configures) and out of the structured stdout
# stream — diff-stable, silenceable by adjusting log level, and consistent
# with src/data/yfinance_fetcher.py's convention.
import logging

# DuckDBStore is the read side of the persistence layer.  Used as a context
# manager so the exclusive file lock is released even if read_bars raises.
from src.data.duckdb_store import DuckDBStore

# load_universe returns the full ordered list of tickers for a named universe
# (e.g. "sp500") from config/universe.yaml — the single source of truth for
# which symbols are in scope for a given research sweep.
from src.data.universe import load_universe

# OHLCVBar is the typed return element of DuckDBStore.read_bars().  Including
# it in the return annotation makes the function's contract explicit without
# forcing callers to import the type separately just to annotate their own
# local variables.
from src.data.schema import OHLCVBar


# Module-level logger.  getLogger(__name__) ties log records to this module's
# fully-qualified name ("src.research.cli_common") so they are filterable by
# the caller's log configuration without affecting other modules' log output.
log = logging.getLogger(__name__)


def build_symbol_list(
    symbols_csv: str | None,
    universe: str,
    limit: int,
) -> list[str]:
    """Return the ordered list of symbols to backtest.

    Two resolution paths — explicit CSV overrides named universe:

      * symbols_csv is not None  →  parse the CSV and return immediately.
      * symbols_csv is None      →  load the named universe, then cap to limit.

    Args:
        symbols_csv: Raw comma-separated string from --symbols, or None if the
            flag was not provided.  May contain surrounding whitespace around
            each token ("AAPL, MSFT , GOOG") — this function normalises it.
        universe:    Name of the universe to load when symbols_csv is None
            (e.g. "sp500").  Passed verbatim to load_universe().
        limit:       Maximum number of symbols to return from the universe path.
            Ignored when symbols_csv is provided.

    Returns:
        A list[str] of uppercase ticker strings, possibly empty.  The empty
        case is intentionally allowed — the caller owns the error message and
        the return-1 because the wording differs per CLI.
    """

    if symbols_csv is not None:
        # Explicit --symbols CSV takes priority over --universe.
        #
        # Why CSV overrides universe (explicit beats implicit):
        #   The caller wrote out every symbol they want to test; silently
        #   substituting a universe config would be a violation of the
        #   principle of least surprise.  Explicit input should always win
        #   over configured defaults.
        #
        # Why strip + filter rather than a bare split(","):
        #   A user may type "AAPL, MSFT , GOOG" with stray spaces, or end the
        #   string with a trailing comma ("AAPL,MSFT,").  strip() normalises
        #   each token's whitespace; the `if s.strip()` filter drops the empty
        #   string that a trailing comma produces — both are silent user errors
        #   that we correct rather than reject.
        #
        # Why --limit is intentionally ignored here:
        #   The caller already controls the list size by writing an explicit CSV.
        #   Silently truncating it would be confusing: "I asked for AAPL, MSFT,
        #   GOOG but only saw AAPL" is a worse outcome than respecting the full
        #   explicit list.
        return [s.strip() for s in symbols_csv.split(",") if s.strip()]

    # No explicit CSV — resolve from the named universe config.
    #
    # Why load_universe() is called before slicing:
    #   load_universe returns the canonical ordered list for this universe.
    #   Slicing [:limit] afterward is a pure O(1) view operation; it does not
    #   change what load_universe reads or the order of the surviving symbols.
    #   Calling load_universe(universe, limit=limit) would require that function
    #   to know about limits, which is a CLI concern, not a data concern.
    #   Keeping the slice here preserves clean separation of concerns.
    #
    # Why slicing is a no-op when len(symbols) <= limit:
    #   Python's slice [:n] when n >= len(seq) returns the full sequence, so
    #   the caller does not need to guard against over-slicing.
    symbols = load_universe(universe)
    return symbols[:limit]


def load_bars_for_symbols(
    symbols: list[str],
    start: str,
    end: str,
) -> dict[str, list[OHLCVBar]]:
    """Read bars from DuckDB for every symbol in the list, skipping missing ones.

    Opens DuckDB once for the entire batch and returns a dict mapping each
    symbol that had data to its bar list.  Symbols with no data are logged
    as warnings and excluded from the result — they do not cause an error.

    Args:
        symbols: Ordered list of ticker strings to look up.
        start:   Inclusive start date string, "YYYY-MM-DD".
        end:     Inclusive end date string, "YYYY-MM-DD".

    Returns:
        A dict[str, list[OHLCVBar]] in insertion order (= input symbol order,
        for the subset of symbols that had data).  May be empty if every
        symbol was missing — the caller owns that guard and its error message.
    """

    # bars_by_symbol accumulates the results.  Dict preserves insertion order
    # in Python 3.7+ (CPython 3.6+), which matches the input symbol order for
    # the surviving symbols.  Downstream callers — pivot builders, compare()
    # calls — rely on this ordering to produce stable, reproducible output.
    bars_by_symbol: dict[str, list[OHLCVBar]] = {}

    # Why a single context manager rather than one open/close per symbol:
    #   DuckDBStore uses an exclusive file lock for the duration of the
    #   connection.  Opening and closing once per symbol would incur N lock
    #   acquires, N file opens, and N kernel round-trips — measurable overhead
    #   for a universe of 25+ symbols.  Opening once amortises that cost across
    #   the entire batch so the loop body is pure row-reads with no I/O
    #   ceremony on each iteration.
    with DuckDBStore() as store:
        for symbol in symbols:
            # read_bars returns bars sorted by timestamp ascending — exactly
            # the order SMACrossoverStrategy.generate_signals() and the
            # Backtester expect.  No re-sorting is needed downstream.
            symbol_bars = store.read_bars(symbol, start, end)

            if not symbol_bars:
                # Why log.warning + continue rather than raising:
                #   A symbol missing from DuckDB is routine in a large universe
                #   sweep: the ingest script may not have run yet for symbols
                #   recently added to the universe config, or a ticker may have
                #   been delisted and never ingested.  Aborting the entire sweep
                #   for one missing symbol would destroy the value of the multi-
                #   symbol comparison for all the other symbols.  Skipping it
                #   and logging a warning is the right tradeoff: the operator
                #   sees the diagnostic, and the rest of the run continues.
                #
                # Why the skip goes to log (stderr-ish) rather than print (stdout):
                #   The research CLIs pipe their stdout into diffs, markdown
                #   exports, and test fixtures.  A "No bars for X" line injected
                #   into that stream would corrupt any tool that parses the
                #   structured output.  log.warning() goes to whatever handler
                #   the caller configures — by default the console — without
                #   touching stdout.  The caller can also silence it entirely by
                #   setting the log level above WARNING if a missing symbol is
                #   expected (e.g. in a test).
                #
                # Why the en-dash "–" between start and end:
                #   Matches compare_matrix.py's existing wording exactly so
                #   grep / log analysis scripts that scan for this pattern do
                #   not need two variants.
                log.warning(
                    "No bars for %s in %s–%s; skipping", symbol, start, end
                )
                continue

            # Insertion order is preserved: symbols arrive in the caller's
            # order and are added in that same order.  Downstream tools that
            # display or sort the dict (pivot builders, compare()) will see
            # symbols in a stable, input-determined sequence rather than an
            # arbitrary hash order.
            bars_by_symbol[symbol] = symbol_bars

    # Return the dict as-is — including the empty-dict case.
    # Why no empty-dict guard here:
    #   Both compare_universe.py and compare_matrix.py print different error
    #   messages when bars_by_symbol is empty ("ERROR: No bars found for any
    #   symbol in {start}–{end}. Run the ingest script first.").  The wording
    #   is identical today but could diverge; keeping the guard in the caller
    #   ensures each CLI owns its own user-facing copy.
    return bars_by_symbol
