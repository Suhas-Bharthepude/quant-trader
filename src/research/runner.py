# src/research/runner.py

"""
Backtest orchestration layer.

BacktestRunner takes a list of strategies and one symbol's bars, runs each
strategy through the Backtester, and returns a list of BacktestResult.

Two iteration axes are supported:
  * run_many()     — one symbol, many strategies  (strategy sweep)
  * run_universe() — one strategy, many symbols   (universe scan)

compare() aggregates a list of BacktestResult into a pandas DataFrame with
one row per backtest and columns for the key metrics, sorted by Sharpe
descending by default.

This is the foundation for Phase 2 parameter optimization — instead of one
strategy, you'll later pass hundreds of parameter combinations through the
same runner.  Pure logic, no I/O: the runner doesn't read from DuckDB or
print to terminal; it just orchestrates.
"""

# dataclass turns a plain class into a structured data container by
# auto-generating __init__, __repr__, and __eq__.  field is used to declare
# per-field metadata (e.g. default_factory) when a simple default isn't
# expressive enough.  Imported here for forward-compatibility — if Phase 2
# adds a RunnerConfig dataclass, these are already in scope.
from dataclasses import dataclass, field

# numpy is the array backbone shared with the rest of the system; not used
# directly here yet, but Phase 2 parameter sweeps will lean on it for
# vectorized grid construction, so we import it now for consistency with
# the rest of src/.
import numpy as np

# pandas is the natural format for the comparison table: sorting, filtering,
# and formatting are one-liners, and DataFrame.to_string() gives a readable
# terminal output for free.  Already a project-wide dependency (yfinance
# uses it under the hood).
import pandas as pd

# OHLCVBar is the canonical bar dataclass.  The runner doesn't read fields
# off it directly — it just forwards bars to the Backtester — but importing
# the type keeps the run_many() signature honest and self-documenting.
from src.data.schema import OHLCVBar

# Strategy is the abstract base every concrete strategy inherits from.  We
# isinstance-check against it in run_many() so a caller passing a stray
# non-Strategy object (e.g. a bare function) fails loudly at the boundary.
from src.strategies.base import Strategy

# Backtester is injected at construction time; the runner is just an
# orchestration shell around it.  Dependency injection here means the
# runner is trivially testable with a fake backtester.
from src.backtest.engine import Backtester

# BacktestResult is the immutable transport object the backtester emits and
# this runner aggregates.  compare() reads the metric attributes off it.
from src.backtest.result import BacktestResult


# ---------------------------------------------------------------------------
# BacktestRunner — runs many strategies against one symbol's bars and
# aggregates the results into a comparison table.
# ---------------------------------------------------------------------------

class BacktestRunner:
    """Orchestrate multiple backtests and aggregate their results."""

    def __init__(self, backtester: Backtester | None = None):
        # If no backtester is passed, instantiate a default one with the
        # Backtester's own defaults (initial_capital=1.0, annualization=252).
        # This makes the runner trivially constructable for the common case
        # — `BacktestRunner()` — while still allowing dependency injection
        # for custom configurations (different capital, different annualization).
        self.backtester = backtester if backtester is not None else Backtester()

    def run_many(
        self,
        bars: list[OHLCVBar],
        strategies: list[Strategy],
    ) -> list[BacktestResult]:
        """Run each strategy against the same bars; return results in input order.

        Pre-conditions enforced at the top:
          * bars is non-empty
          * strategies is non-empty
          * every element of strategies is a Strategy instance

        For each strategy: generate signals from the bars, run the backtester,
        and append the result.  Order of the output list matches the input.
        """

        # ------------------------------------------------------------------
        # 1. Validate inputs.  All checks raise ValueError or TypeError with
        #    a message that identifies the exact failure — easier to debug
        #    than a later AttributeError deep in the backtester.
        # ------------------------------------------------------------------

        # Empty bars would make the Backtester reject the call anyway, but
        # catching it here gives a clearer message ("runner got no bars")
        # than the engine's lower-level "bars must be non-empty".
        if len(bars) == 0:
            raise ValueError("bars must be non-empty")

        # Empty strategies → nothing to run; the resulting list would be
        # empty and the comparison table would be empty too.  Almost
        # certainly a caller bug, so reject it loudly rather than returning
        # a silently-empty result.
        if len(strategies) == 0:
            raise ValueError("strategies must be non-empty")

        # isinstance check against the abstract base — Python's ABC machinery
        # makes Strategy() un-instantiable, so anything that passes this
        # check is a concrete subclass that implements generate_signals()
        # and the name property.  Catches stray functions or duck-typed
        # objects at the boundary.
        for i, strategy in enumerate(strategies):
            if not isinstance(strategy, Strategy):
                raise TypeError(
                    f"strategies[{i}] must be a Strategy instance, got {type(strategy).__name__}"
                )

        # ------------------------------------------------------------------
        # 2. Run each strategy through the backtester in input order.
        # ------------------------------------------------------------------

        # Pre-allocate the results list to keep ordering explicit; we append
        # in the loop rather than using a comprehension because the loop
        # body has two named intermediate values (signals, result) which
        # read better as discrete steps than as a nested expression.
        results: list[BacktestResult] = []

        # Iterate strategies in order; the output list mirrors the input
        # order so callers can zip(strategies, results) if they need to.
        for strategy in strategies:
            # generate_signals() is the strategy's only computational
            # responsibility — turn bars into an int signal array of the
            # same length.  The Strategy contract guarantees alignment.
            signals = strategy.generate_signals(bars)

            # Pass the strategy's own name through to the result so the
            # comparison table is self-labelling — no need for the caller
            # to track which result came from which strategy.
            result = self.backtester.run(bars, signals, strategy_name=strategy.name)

            # Append in chronological (input) order.
            results.append(result)

        # Return the full list.  Callers typically feed this straight into
        # compare(), but the raw list is also useful for per-strategy
        # plotting or trade-level inspection.
        return results

    def run_universe(
        self,
        bars_by_symbol: dict[str, list[OHLCVBar]],
        strategy: Strategy,
        min_bars: int | None = None,
    ) -> list[BacktestResult]:
        """
        Run one strategy against many symbols.

        Args:
            bars_by_symbol: maps symbol → that symbol's bars (ascending time).
            strategy: a Strategy instance to evaluate on each symbol.
            min_bars: if set, skip any symbol whose len(bars) < min_bars.
                      Used to skip symbols too short for the strategy's slow window.
                      If None, no symbol is skipped at this layer.

        Returns:
            A list of BacktestResult. Each result's strategy_name is set to
            f"{strategy.name} on {symbol}" so the symbol is visible in compare().
            Results are returned in the iteration order of bars_by_symbol
            (dict insertion order, guaranteed in Python 3.7+).

        Raises:
            ValueError: if bars_by_symbol is empty, if any bars list is empty,
                        or if every symbol is skipped by min_bars.
            TypeError:  if strategy is not a Strategy instance.
        """

        # ------------------------------------------------------------------
        # 1. Validate inputs.  Same defensive boundary pattern as run_many().
        # ------------------------------------------------------------------

        # An empty dict means no symbols to run — almost certainly a caller
        # bug (e.g. a failed data fetch returned {}).  Reject loudly rather
        # than returning a silently-empty list that masks the problem upstream.
        if len(bars_by_symbol) == 0:
            raise ValueError("bars_by_symbol must be non-empty")

        # isinstance check against the abstract base — same guard as run_many().
        # Catches stray functions or duck-typed objects at the boundary before
        # any symbol-level work begins, keeping the error message simple.
        if not isinstance(strategy, Strategy):
            raise TypeError(
                f"strategy must be a Strategy instance, got {type(strategy).__name__}"
            )

        # Validate every symbol's bar list up front.  Checking here (rather
        # than lazily inside the loop) means we surface all empty-bar problems
        # before any backtest runs — partial progress followed by a crash is
        # harder to debug than an upfront validation failure.
        for symbol, bars in bars_by_symbol.items():
            if len(bars) == 0:
                raise ValueError(
                    f"bars for symbol {symbol!r} must be non-empty"
                )

        # ------------------------------------------------------------------
        # 2. Run the strategy against each symbol in insertion order.
        # ------------------------------------------------------------------

        # Pre-allocate results list; we append inside the loop because the
        # loop body has named intermediates (signals, result) that read more
        # clearly as discrete steps than as a nested comprehension.
        results: list[BacktestResult] = []

        # dict.items() gives (symbol, bars) pairs in insertion order so both
        # the symbol label and its bars are in scope on every iteration.
        for symbol, bars in bars_by_symbol.items():

            # Skip symbols that don't have enough history for the strategy's
            # indicator warm-up period (e.g. a 200-bar SMA needs ≥ 200 bars).
            # We SKIP rather than RAISE here because having a short symbol in
            # a universe is normal — not every ticker has the same history —
            # and the caller uses min_bars as a filter knob, not a hard error.
            # A per-symbol TypeError/ValueError would prevent all other symbols
            # from running, which defeats the purpose of a universe scan.
            if min_bars is not None and len(bars) < min_bars:
                continue  # not enough history for this symbol; move on

            # Strategy is stateless (pure): generate_signals() reads only the
            # bars it is passed and produces a new signal array each time.
            # There is no internal state to reset between symbols, so we can
            # safely reuse the same instance across the entire universe without
            # cloning or re-instantiating it — which keeps memory and setup
            # cost constant regardless of universe size.
            signals = strategy.generate_signals(bars)

            # Compose the per-symbol label inside the loop because the label
            # is inherently per-iteration: the strategy name is shared across
            # all symbols, but the symbol string changes on every pass.
            # We must NOT mutate strategy.name — the strategy is reused across
            # calls (and potentially across concurrent runners), so mutating
            # it would corrupt labels for other symbols.  The strategy_name
            # kwarg on run() is the correct injection point: it overrides the
            # label in the result without touching the strategy object.
            result = self.backtester.run(
                bars,
                signals,
                strategy_name=f"{strategy.name} on {symbol}",
            )

            # Append preserves insertion order of bars_by_symbol so callers
            # can zip(bars_by_symbol.keys(), results) if needed.
            results.append(result)

        # ------------------------------------------------------------------
        # 3. Guard against the degenerate case where every symbol was skipped.
        # ------------------------------------------------------------------

        # Silently returning [] here would be a silent misconfiguration: the
        # caller set min_bars too high (or provided a universe that's entirely
        # too short for the strategy) and would receive an empty list with no
        # indication that something went wrong.  Raising forces the caller to
        # either lower min_bars, use a faster strategy, or supply longer data —
        # all of which are intentional corrective actions, not silent failures.
        if len(results) == 0:
            raise ValueError(
                "all symbols were skipped because their bar counts are below "
                f"min_bars={min_bars}; supply symbols with more history or "
                "lower min_bars"
            )

        # Return results in the same order as bars_by_symbol was iterated.
        return results

    # ------------------------------------------------------------------
    # compare() is @staticmethod because it doesn't read self.backtester
    # or any other instance state — it's a pure transformation from a
    # list of results into a DataFrame.  Marking it static signals to
    # readers "this is a pure function" and lets callers use it without
    # constructing a runner first (e.g. when loading results from disk).
    # ------------------------------------------------------------------
    @staticmethod
    def compare(
        results: list[BacktestResult],
        sort_by: str = "sharpe_ratio",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Aggregate results into a sorted DataFrame, one row per backtest.

        Columns:
          * strategy      — result.strategy_name
          * total_return  — result.total_return_pct
          * sharpe        — result.sharpe_ratio
          * max_drawdown  — result.max_drawdown_pct
          * win_rate      — result.win_rate
          * n_trades      — result.n_trades

        sort_by accepts the BacktestResult attribute names (whitelist:
        'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate',
        'n_trades') and is mapped internally to the DataFrame column name
        before sorting.  Default is sharpe descending (best first).

        Note: for max_drawdown the natural "best" direction is ascending
        (smaller drawdown is better).  We do NOT auto-invert — the caller
        passes ascending=True when sorting by drawdown.  Keeping direction
        explicit avoids surprising behavior when sort_by is parameterized.
        """

        # ------------------------------------------------------------------
        # 1. Validate inputs.
        # ------------------------------------------------------------------

        # An empty results list would produce an empty DataFrame and the
        # downstream sort_values call would no-op silently.  Reject up
        # front so the failure is obvious.
        if len(results) == 0:
            raise ValueError("results must be non-empty")

        # Whitelist the legal sort keys.  Using a set (not a list) gives
        # O(1) membership testing and is the canonical Python idiom for
        # "is this one of N allowed values".  The whitelist deliberately
        # uses the BacktestResult attribute names — that's what callers
        # already know from the result dataclass.
        valid_sort_keys = {
            "sharpe_ratio",
            "total_return",
            "max_drawdown",
            "win_rate",
            "n_trades",
        }
        if sort_by not in valid_sort_keys:
            raise ValueError(
                f"sort_by must be one of {sorted(valid_sort_keys)}, got {sort_by!r}"
            )

        # ------------------------------------------------------------------
        # 2. Build the DataFrame.  One dict per row, keyed by column name.
        #    Using a list-of-dicts (vs. a dict-of-lists) makes each row's
        #    construction self-contained and trivially readable.
        # ------------------------------------------------------------------

        # List comprehension over results — each result becomes one row.
        # The column names here are the public-facing names in the table;
        # the BacktestResult attribute names (total_return_pct, etc.) are
        # an implementation detail we rename for terseness in the report.
        rows = [
            {
                "strategy": r.strategy_name,           # human-readable label from the strategy
                "total_return": r.total_return_pct,    # fraction; 0.25 → +25%
                "sharpe": r.sharpe_ratio,              # annualized Sharpe
                "max_drawdown": r.max_drawdown_pct,    # positive fraction; 0.20 → 20% drawdown
                "win_rate": r.win_rate,                # fraction of trades with positive return
                "n_trades": r.n_trades,                # cached len(trades)
            }
            for r in results
        ]

        # Construct the frame.  pandas infers dtypes per column; strategy
        # becomes object, n_trades becomes int64, everything else float64.
        df = pd.DataFrame(rows)

        # ------------------------------------------------------------------
        # 3. Map the sort_by parameter (BacktestResult attribute name) to
        #    the DataFrame column name.  Only 'sharpe_ratio' differs from
        #    its column ('sharpe' for terseness in the printed table); the
        #    other four whitelist entries match column names exactly.
        # ------------------------------------------------------------------

        # Single-entry map: the only attribute-to-column rename.  Using a
        # dict (not an if/else) keeps the door open for future renames
        # without restructuring the code.
        sort_column_map = {"sharpe_ratio": "sharpe"}
        # .get() with a default of sort_by itself means "rename if mapped,
        # otherwise pass through unchanged" — the four columns that match
        # their attribute names need no special-case handling.
        sort_column = sort_column_map.get(sort_by, sort_by)

        # ------------------------------------------------------------------
        # 4. Sort and return.  reset_index drops the original (now-shuffled)
        #    row indices so the returned frame is indexed 0..N-1 in sorted
        #    order — cleaner for downstream printing and CSV export.
        # ------------------------------------------------------------------

        # sort_values returns a new frame (does not mutate); ascending
        # follows the caller's request.  Default False → descending →
        # best-first when sort_by is sharpe or total_return.
        df = df.sort_values(by=sort_column, ascending=ascending).reset_index(drop=True)

        return df
