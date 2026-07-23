# scripts/ocm_sweep.py

"""
Sweep Opening-Candle Momentum (OCM) over N and exit modes, IN-SAMPLE vs OUT-OF-SAMPLE.

OCM: if the first N 1-minute candles are all green, buy at candle N+1's open; else no
trade.  Momentum claims are the easiest to fool yourself on, so this runner prints, for
every (N, exit_mode), a THREE-row block — full / in-sample (first half) / out-of-sample
(second half) — alongside a BUY-OPEN baseline on the same fired days.  A real edge must
(a) beat the baseline (edgeNet > 0) and (b) do so in BOTH halves, not just in-sample.

Columns:
  window sessions signals sig% win% avgGrsWin avgGrsLoss netRet grossRet baseNet edgeNet Sharpe
  * win%     = fraction of trades with GROSS return > 0
  * netRet   = NET compounded over fired days; grossRet = cost-free
  * baseNet  = buy 09:30 open -> sell close on the SAME fired days, net
  * edgeNet  = netRet - baseNet  (the excess over just riding intraday drift)

Run (stored 1m data, no network):
    uv run python scripts/ocm_sweep.py --symbol SOXL --start 2025-07-01 --end 2026-07-22 \
        --n 2,3 --exit-modes close,bracket --profit-target 0.01 --stop-pct 0.01 \
        --fee-bps 1 --slippage-bps 5 --no-fetch
"""

import argparse
import sys

from src.brokers.alpaca_broker import AlpacaBroker
from src.data.duckdb_store import DuckDBStore
from src.intraday.ocm_strategy import OpeningCandleMomentum, OpeningCandleMomentumConfig
from src.intraday.sim import run_ocm_backtest


def _row(stats) -> str:
    return (
        f"{stats.label:>13} {stats.n_sessions:>8d} {stats.n_signals:>7d} "
        f"{stats.signal_rate:>5.0%} {stats.win_rate:>5.0%} "
        f"{stats.avg_gross_win:>+9.3%} {stats.avg_gross_loss:>+10.3%} "
        f"{stats.total_return_pct:>+9.2%} {stats.gross_total_return_pct:>+9.2%} "
        f"{stats.baseline_net_return_pct:>+9.2%} {stats.edge_net_pct:>+9.2%} "
        f"{stats.sharpe_ratio:>+7.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep OCM over N and exit modes; print in-sample vs out-of-sample + baseline."
    )
    parser.add_argument("--symbol", default="SOXL", help="Traded symbol (default: SOXL).")
    parser.add_argument("--start", required=True, help="Start date, ISO.")
    parser.add_argument("--end", required=True, help="End date, ISO.")
    parser.add_argument("--n", default="2,3", help="Comma-separated N values (default: 2,3).")
    parser.add_argument(
        "--exit-modes", default="close,bracket",
        help="Comma-separated exit modes: close,bracket (default: both).",
    )
    parser.add_argument("--profit-target", type=float, default=0.01, help="Bracket target (default 0.01).")
    parser.add_argument("--stop-pct", type=float, default=0.01, help="Bracket stop (default 0.01).")
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Fee per side in bps (default 1.0).")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage per side in bps (default 5.0).")
    parser.add_argument("--no-fetch", action="store_true", help="Use stored 1m bars; skip Alpaca fetch.")
    args = parser.parse_args()

    ns = [int(x.strip()) for x in args.n.split(",") if x.strip()]
    modes = [m.strip() for m in args.exit_modes.split(",") if m.strip()]

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

    with DuckDBStore() as store:
        bars = store.read_bars(args.symbol, args.start, args.end, timeframe="1m")
    print(f"Loaded {len(bars)} 1m bars for {args.symbol} ({args.start}..{args.end})")
    if not bars:
        print(f"ERROR: no 1m bars for {args.symbol}; nothing to sweep.", file=sys.stderr)
        sys.exit(1)
    bars_by_symbol = {args.symbol: bars}

    print(
        f"\nOCM sweep — {args.symbol}  |  costs {args.fee_bps:g}+{args.slippage_bps:g} bps/side  "
        f"|  bracket target {args.profit_target:.2%} / stop {args.stop_pct:.2%}"
    )
    header = (
        f"{'window':>13} {'sessions':>8} {'signals':>7} {'sig%':>5} {'win%':>5} "
        f"{'avgGrsWin':>9} {'avgGrsLoss':>10} {'netRet':>9} {'grossRet':>9} "
        f"{'baseNet':>9} {'edgeNet':>9} {'Sharpe':>7}"
    )

    for n in ns:
        for mode in modes:
            config = OpeningCandleMomentumConfig(
                primary_symbol=args.symbol, n_candles=n, exit_mode=mode,
                profit_target=args.profit_target, stop_pct=args.stop_pct,
                fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
            )
            result = run_ocm_backtest(bars_by_symbol, OpeningCandleMomentum(config), config)
            print(f"\n### N={n}  exit={mode}   ({result.strategy_name})")
            print(header)
            print("-" * len(header))
            print(_row(result.full))
            print(_row(result.in_sample))
            print(_row(result.out_of_sample))
            # Honest one-line verdict.
            io = result.in_sample
            oos = result.out_of_sample
            survives = (io.edge_net_pct > 0) and (oos.edge_net_pct > 0)
            print(
                f"  verdict: edge over buy-open is "
                f"{'POSITIVE in BOTH halves' if survives else 'NOT present in both halves'} "
                f"(in-sample edgeNet {io.edge_net_pct:+.2%}, OOS edgeNet {oos.edge_net_pct:+.2%})"
            )

    print(
        "\nRead: a credible edge needs edgeNet > 0 in BOTH in-sample AND out-of-sample. "
        "Beating buy-open only in-sample is the signature of fitting to remembered days."
    )


if __name__ == "__main__":
    main()
