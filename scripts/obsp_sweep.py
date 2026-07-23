# scripts/obsp_sweep.py

"""
Sweep the Open-Buy, Sell-at-Profit (OBSP) profit_target and print an honest table.

OBSP: buy at the 09:30 open, sell at the first bar that reaches +profit_target
(fill at the target), else force-flat at the close.  This runner sweeps a few
profit_target values over a date range and prints, PER TARGET, gross and net
numbers side by side so "reached the target" is never confused with "made money
after costs":

    target | trades | win% | avgGrossWin | avgGrossLoss | W/L | avgNetWin | avgNetLoss | netRet | grossRet | Sharpe | maxDD

  * win%          = fraction of sessions that REACHED the target (a GROSS win)
  * avgGrossWin   = mean return of target-reached sessions, before costs
  * avgGrossLoss  = mean return of never-reached (EOD-close) sessions, before costs
  * W/L           = |avgGrossWin| / |avgGrossLoss| (size ratio — small wins vs big losses)
  * avgNet*       = the same buckets AFTER per-side costs
  * netRet        = NET compounded return; grossRet = cost-free compounded return

Not validation — a single-window backtest.  It answers one honest question: are
the small frequent wins outweighed by the big losses when price falls from the
open and never recovers?

Run (on stored 1m data, no network):
    uv run python scripts/obsp_sweep.py --symbol SOXL --start 2025-07-01 \
        --end 2026-07-22 --targets 0.003,0.005,0.01,0.02 \
        --fee-bps 1 --slippage-bps 5 --no-fetch

Add --symbol TQQQ (etc.) to run any other ticker; drop --no-fetch to pull+store
fresh 1m bars from Alpaca paper first.
"""

import argparse
import sys

from src.brokers.alpaca_broker import AlpacaBroker
from src.data.duckdb_store import DuckDBStore
from src.intraday.obsp_strategy import OpenBuySellProfit, OpenBuySellProfitConfig
from src.intraday.sim import run_obsp_backtest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep OBSP profit_target and print gross-vs-net results."
    )
    parser.add_argument("--symbol", default="SOXL", help="Traded symbol (default: SOXL).")
    parser.add_argument("--start", required=True, help="Start date, ISO (e.g. 2025-07-01).")
    parser.add_argument("--end", required=True, help="End date, ISO (e.g. 2026-07-22).")
    parser.add_argument(
        "--targets", default="0.003,0.005,0.01,0.02",
        help="Comma-separated profit_target values (default: 0.003,0.005,0.01,0.02).",
    )
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Fee per side in bps (default 1.0).")
    parser.add_argument(
        "--slippage-bps", type=float, default=5.0,
        help="Slippage per side in bps (default 5.0 — SOXL spreads are wide).",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Skip the Alpaca fetch and use 1m bars already in the DuckDB store.",
    )
    args = parser.parse_args()

    targets = [float(t.strip()) for t in args.targets.split(",") if t.strip()]

    # ------------------------------------------------------------------
    # Optional fetch + persist (single symbol; OBSP needs no confirmation data).
    # ------------------------------------------------------------------
    if not args.no_fetch:
        try:
            broker = AlpacaBroker.from_env()
            broker.verify_paper_account()
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR building Alpaca broker: {exc}", file=sys.stderr)
            sys.exit(1)
        with DuckDBStore() as store:
            print(f"Fetching 1m bars for {args.symbol} {args.start}..{args.end} ...")
            bars = broker.get_minute_bars(args.symbol, args.start, args.end)
            print(f"  fetched {len(bars)}, wrote {store.write_bars(bars)} new")

    # ------------------------------------------------------------------
    # Read once; reuse the same bars for every target in the sweep.
    # ------------------------------------------------------------------
    with DuckDBStore() as store:
        bars = store.read_bars(args.symbol, args.start, args.end, timeframe="1m")
    print(f"Loaded {len(bars)} 1m bars for {args.symbol} ({args.start}..{args.end})")
    if not bars:
        print(f"ERROR: no 1m bars for {args.symbol}; nothing to sweep.", file=sys.stderr)
        sys.exit(1)
    bars_by_symbol = {args.symbol: bars}

    # ------------------------------------------------------------------
    # Sweep + table.
    # ------------------------------------------------------------------
    print(
        f"\nOBSP sweep — {args.symbol}  |  costs {args.fee_bps:g}+{args.slippage_bps:g} bps/side  "
        f"(win% = target-reached; gross vs net kept separate)\n"
    )
    header = (
        f"{'target':>7} {'trades':>6} {'win%':>6} {'avgGrsWin':>10} {'avgGrsLoss':>11} "
        f"{'W/L':>5} {'avgNetWin':>10} {'avgNetLoss':>11} {'netRet':>9} {'grossRet':>9} "
        f"{'Sharpe':>7} {'maxDD':>7}"
    )
    print(header)
    print("-" * len(header))

    for pt in targets:
        config = OpenBuySellProfitConfig(
            primary_symbol=args.symbol,
            profit_target=pt,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )
        r = run_obsp_backtest(bars_by_symbol, OpenBuySellProfit(config), config)
        print(
            f"{pt:>7.3%} {r.n_trades:>6d} {r.win_rate:>6.1%} "
            f"{r.avg_gross_win:>+10.3%} {r.avg_gross_loss:>+11.3%} "
            f"{r.win_loss_size_ratio:>5.2f} {r.avg_net_win:>+10.3%} {r.avg_net_loss:>+11.3%} "
            f"{r.total_return_pct:>+9.2%} {r.gross_total_return_pct:>+9.2%} "
            f"{r.sharpe_ratio:>+7.2f} {r.max_drawdown_pct:>7.2%}"
        )

    print(
        "\nRead: win% is how often you hit the target; W/L is the size of the average "
        "win vs the average losing (never-recovered) day. A high win% with W/L << 1 is "
        "the classic 'pick up pennies' shape — netRet is the honest bottom line."
    )


if __name__ == "__main__":
    main()
