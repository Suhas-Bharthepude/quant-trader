# tests/test_rotation_verdict_script.py

"""
Hermetic smoke test for scripts/rotation_verdict.py.

The rotation MATH is already covered in tests/test_rotation_search.py and
tests/test_rotation_verdict.py. This test does NOT re-check any numbers and does NOT touch
DuckDB. It only proves the script's main() wires together — argparse → cli_common load →
rotation_walk_forward_search → table print — and returns 0 on a tiny synthetic basket.

We achieve DuckDB independence by monkeypatching the two cli_common names AS IMPORTED INTO the
script's namespace (build_symbol_list / load_bars_for_symbols), plus shrinking the module-level
_GRID to a 2-candidate grid and passing small --train-months/--test-months via sys.argv so the
run is fast.
"""

# sys is monkeypatched (sys.argv) to drive main() through argparse without real CLI args.
import sys

# datetime + timezone build the synthetic month-end timestamps.
from datetime import datetime, timezone

# OHLCVBar is the bar schema the synthetic basket is built from.
from src.data.schema import OHLCVBar

# The module under test — imported as a module so we can monkeypatch names in its namespace.
import scripts.rotation_verdict as rv


# ---------------------------------------------------------------------------
# Synthetic-data helpers (copied from tests/test_rotation_search.py so this test stays hermetic).
# ---------------------------------------------------------------------------


def make_month_end_bars(
    dates: list[datetime],
    closes: list[float],
    symbol: str = "T",
) -> list[OHLCVBar]:
    """Build one OHLCVBar per (date, close), in the given (ascending) order.

    Copied from tests/test_rotation_search.py: close == adj_close so the default price_field
    ="close" reads the intended value; OHLC are dummies.
    """
    # One bar per (date, close) pair, in the given order (caller supplies ascending dates).
    return [
        OHLCVBar(
            symbol=symbol,        # arbitrary; the code reads only timestamp + price
            timestamp=ts,         # the bar's real trading date
            open=c,               # dummy
            high=c,               # dummy
            low=c,                # dummy
            close=c,              # the field the default price_field reads
            adj_close=c,          # equal to close -> basis-agnostic
            volume=1000,          # dummy
            timeframe="1d",       # daily
            source="test",        # provenance marker
        )
        for ts, c in zip(dates, closes)
    ]


def monthly_dates(n: int, start_year: int = 2015, start_month: int = 1) -> list[datetime]:
    """Return n ascending month-end-ish dates, one per consecutive calendar month.

    Copied from tests/test_rotation_search.py. Uses day 28 (valid in every month) so each bar
    lands in a distinct calendar month; with one bar per month EVERY bar is a month-end and M == n.
    """
    # Walk consecutive (year, month) pairs, emitting one timestamp each.
    dates: list[datetime] = []
    year, month = start_year, start_month
    for _ in range(n):
        # Day 28 is safe for all 12 months; the calendar month is what matters.
        dates.append(datetime(year, month, 28, tzinfo=timezone.utc))
        # Advance one calendar month, rolling the year over after December.
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


def synthetic_basket(n_months: int = 15) -> dict[str, list[OHLCVBar]]:
    """Build a 4-symbol synthetic basket (SPY reference) on a shared monthly grid.

    n_months=15 month-ends is enough for a couple of folds under --train-months 6 --test-months 3
    (min month-ends = train + test + 1 = 10). Every symbol shares the SAME date grid so the
    rotation's alignment never trips; closes are distinct per symbol so momentum ranking is
    non-degenerate (a real top_n selection, not a tie).
    """
    # One shared ascending monthly date grid for every symbol.
    dates = monthly_dates(n_months)

    # SPY is the reference spine; each symbol compounds at a different monthly rate so their
    # trailing momentum differs and the ranker has a genuine ordering to pick from.
    rates = {"SPY": 0.010, "QQQ": 0.020, "TLT": 0.005, "GLD": 0.015}

    # Build each symbol's close series by compounding from 100.0 at its own rate.
    basket: dict[str, list[OHLCVBar]] = {}
    for symbol, rate in rates.items():
        # closes[k] = 100 * (1 + rate)**k — a smooth, strictly-increasing series per symbol.
        closes = [100.0 * (1.0 + rate) ** k for k in range(n_months)]
        # Wrap into OHLCVBars keyed by symbol.
        basket[symbol] = make_month_end_bars(dates, closes, symbol=symbol)

    # Return the basket keyed by symbol, with "SPY" the reference spine.
    return basket


# ---------------------------------------------------------------------------
# The smoke test.
# ---------------------------------------------------------------------------


def test_main_runs_and_returns_zero_on_synthetic_basket(monkeypatch, capsys):
    """main() wires together and returns 0 on a tiny synthetic basket, no DuckDB touched."""
    # Build the synthetic basket once.
    basket = synthetic_basket(n_months=15)

    # Monkeypatch the two cli_common names AS IMPORTED INTO the script's namespace so no DuckDB
    # connection is ever opened: build_symbol_list returns the basket's keys, load returns the
    # basket dict directly (ignoring its args).
    monkeypatch.setattr(
        rv, "build_symbol_list", lambda symbols_csv, universe, limit: list(basket.keys())
    )
    monkeypatch.setattr(
        rv, "load_bars_for_symbols", lambda symbols, start, end: basket
    )

    # Shrink the module-level grid to 2 candidates (both lookback=3, so L_max=3) so the run is
    # fast and the L_max warm-up guard is satisfied by --train-months 6 (6 > 3 + 1).
    monkeypatch.setattr(rv, "_GRID", [(1, 3), (2, 3)])

    # Drive main() through argparse with small windows: train=6/test=3 → a couple of folds over
    # the 15 synthetic month-ends. Only the script name + these flags are on argv.
    monkeypatch.setattr(
        sys,
        "argv",
        ["rotation_verdict.py", "--train-months", "6", "--test-months", "3"],
    )

    # Run the entry point.
    rc = rv.main()

    # It must succeed.
    assert rc == 0

    # And it must have printed the verdict table (header text present) — a cheap check that the
    # print path ran, not just that main() returned.
    out = capsys.readouterr().out
    assert "Fitted OOS" in out
    assert "overfitting tax" in out
