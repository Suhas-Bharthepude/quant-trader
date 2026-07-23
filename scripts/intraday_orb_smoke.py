# scripts/intraday_orb_smoke.py

"""
End-to-end smoke run for the intraday opening-range breakout (ORB) track.

This is NOT a unit test (those live in tests/ and stay hermetic).  It is a
minimal, manual end-to-end run that wires the real pieces together:

  1. pull 1-minute bars for the primary + confirmation symbols from Alpaca
     (AlpacaBroker.get_minute_bars — paper-only, read-only market data),
  2. persist them to the existing DuckDB store as timeframe="1m",
  3. read them back, run the event-driven ORB backtest, and print the metrics.

It adds NO production logic — it only fetches, stores, and calls the already-
tested run_orb_backtest.  It never submits an order (market-data only).

Everything is symbol-agnostic: --symbol is only a DEFAULT of SOXL.  Swap it for
TQQQ/TNA/anything, and change --confirm, without touching code.

Run via:
    uv run python scripts/intraday_orb_smoke.py \
        --symbol SOXL --confirm SPY,QQQ --start 2024-01-02 --end 2024-03-28

    uv run python scripts/intraday_orb_smoke.py --symbol TQQQ   # different ticker
"""

import argparse
import sys

from src.brokers.alpaca_broker import AlpacaBroker
from src.data.duckdb_store import DuckDBStore
from src.intraday.orb_strategy import OpeningRangeBreakout, OpeningRangeBreakoutConfig
from src.intraday.sim import run_orb_backtest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intraday ORB end-to-end smoke run (Alpaca 1m -> DuckDB -> sim)."
    )
    parser.add_argument("--symbol", default="SOXL", help="Traded symbol (default: SOXL).")
    parser.add_argument(
        "--confirm",
        default="SPY,QQQ",
        help="Comma-separated confirmation symbols (default: SPY,QQQ).",
    )
    parser.add_argument("--start", required=True, help="Start date, ISO (e.g. 2024-01-02).")
    parser.add_argument("--end", required=True, help="End date, ISO (e.g. 2024-03-28).")
    parser.add_argument("--stop-pct", type=float, default=0.06, help="Stop distance (default 0.06).")
    parser.add_argument("--target-r", type=float, default=2.0, help="Target in R (default 2.0).")
    parser.add_argument(
        "--trail-pct", type=float, default=None,
        help=(
            "Trailing-stop percent (e.g. 0.05 for 5%%). Omit for the default "
            "fixed-stop/target exit; when set, stop-pct/target-r are ignored and "
            "the trailing-stop exit is used instead."
        ),
    )
    parser.add_argument("--or-minutes", type=int, default=15, help="Opening-range width (default 15).")
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Fee per side in bps (default 1.0).")
    parser.add_argument(
        "--slippage-bps", type=float, default=5.0,
        help="Slippage per side in bps (default 5.0 — SOXL spreads are wide).",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Skip the Alpaca fetch and use bars already in the DuckDB store.",
    )
    args = parser.parse_args()

    confirmation_symbols = tuple(s.strip() for s in args.confirm.split(",") if s.strip())
    all_symbols = [args.symbol, *confirmation_symbols]

    # ------------------------------------------------------------------
    # 1. Fetch + persist (unless --no-fetch).
    # ------------------------------------------------------------------
    if not args.no_fetch:
        try:
            broker = AlpacaBroker.from_env()
            broker.verify_paper_account()  # data-only run, but assert paper anyway
        except Exception as exc:  # noqa: BLE001 — surface any wiring/credential error
            print(f"ERROR building Alpaca broker: {exc}", file=sys.stderr)
            sys.exit(1)

        with DuckDBStore() as store:
            for sym in all_symbols:
                print(f"Fetching 1m bars for {sym} {args.start}..{args.end} ...")
                bars = broker.get_minute_bars(sym, args.start, args.end)
                written = store.write_bars(bars)
                print(f"  fetched {len(bars)} bars, wrote {written} new to store")

    # ------------------------------------------------------------------
    # 2. Read back + run the sim.
    # ------------------------------------------------------------------
    with DuckDBStore() as store:
        bars_by_symbol = {
            sym: store.read_bars(sym, args.start, args.end, timeframe="1m")
            for sym in all_symbols
        }

    for sym in all_symbols:
        n = len(bars_by_symbol.get(sym, []))
        print(f"  {sym}: {n} 1m bars loaded")
    if not bars_by_symbol.get(args.symbol):
        print(f"ERROR: no bars for primary symbol {args.symbol}; nothing to backtest.",
              file=sys.stderr)
        sys.exit(1)

    config = OpeningRangeBreakoutConfig(
        primary_symbol=args.symbol,
        confirmation_symbols=confirmation_symbols,
        or_minutes=args.or_minutes,
        stop_pct=args.stop_pct,
        target_r=args.target_r,
        trail_pct=args.trail_pct,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    strategy = OpeningRangeBreakout(config)
    result = run_orb_backtest(bars_by_symbol, strategy, config)

    # ------------------------------------------------------------------
    # 3. Report.
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"Strategy      : {result.strategy_name}")
    exit_mode = (
        f"trailing-stop {args.trail_pct:.0%}"
        if args.trail_pct is not None
        else f"fixed stop {args.stop_pct:.0%} / {args.target_r:g}R target"
    )
    print(f"Exit mode     : {exit_mode}")
    print(f"Window        : {result.start_date} .. {result.end_date}")
    print(f"Sessions      : {result.n_sessions}")
    print(f"Trades        : {result.n_trades}  "
          f"(stop={result.exit_reason_counts['stop']}, "
          f"target={result.exit_reason_counts['target']}, "
          f"trailing={result.exit_reason_counts['trailing_stop']}, "
          f"eod={result.exit_reason_counts['eod']})")
    print(f"Total return  : {result.total_return_pct:+.2%}  (net of costs)")
    print(f"Sharpe (252)  : {result.sharpe_ratio:+.2f}")
    print(f"Max drawdown  : {result.max_drawdown_pct:.2%}")
    print(f"Win rate      : {result.win_rate:.2%}  (net)")
    print(f"Total cost    : {result.total_cost_pct:.4%} of capital")
    print("=" * 64)
    print(
        "\nReminder: a single-run backtest is not validation. Honest OOS testing "
        "(intraday walk-forward) is the deferred next step, not this smoke run."
    )


if __name__ == "__main__":
    main()
