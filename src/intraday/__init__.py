# src/intraday/

"""
Intraday trading track — parallel to the daily/swing stack, never replacing it.

The daily system (src/strategies/base.py Strategy ABC + src/backtest/engine.py
vectorized Backtester) models close-to-close returns over full daily bars. That
engine structurally CANNOT represent an intrabar stop, an intrabar profit target,
or a forced end-of-day flat — which are exactly the mechanics of an intraday
opening-range breakout. So this package adds a SEPARATE, event-driven track:

  session.py      — group 1-minute bars into ET trading sessions; extract the
                    opening-range window; align confirmation symbols onto the
                    primary symbol's decision timeline (backward-only as-of join).
  orb_strategy.py — IntradayStrategy ABC (multi-symbol, unlike the single-symbol
                    daily Strategy) + the OpeningRangeBreakout entry rule.
  sim.py          — event-driven bar-by-bar simulator that fills stops/targets
                    intrabar (stop-first on same-bar conflicts) and forces EOD-flat.

Reuse discipline: this package REUSES src/data/schema.OHLCVBar, DuckDBStore,
src/backtest/result.Trade, and src/backtest/metrics read-only. It does NOT modify
the daily Backtester, its no-lookahead test, or the daily Strategy ABC.
"""
