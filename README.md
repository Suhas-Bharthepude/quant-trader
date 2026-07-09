# quant-trader

Systematic swing-trading bot for US equities — built rigor-first, as a research-and-validation pipeline rather than a get-rich strategy.

![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)
![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)

---

## Scope & Thesis

This project builds a rigorous research-and-validation pipeline (walk-forward validation, overfitting controls, clean survivorship-bias-free data). The strategies it currently runs — an SMA-crossover baseline and a time-series momentum model on a fixed ETF basket — are vehicles to build and prove that pipeline, not alpha claims. The longer-term goal is to use the pipeline to search for a real, validated edge before risking capital. **Profitability is not assumed and will only be claimed if it survives out-of-sample validation.**

---

## Status

Phases 0–2 complete; Phase 3 (execution realism + risk) in progress. The custom backtester, both strategies, and the full walk-forward + Optuna + overfitting-tax harness are built and tested, with a hermetic test suite running green in CI on every push. The three pre-verdict accounting items — transaction costs, total-return (dividend) accounting, and interest on idle cash — are done; the honest out-of-sample verdict on the momentum strategy is the next milestone.

A headline result so far: the SMA-crossover baseline has **no out-of-sample edge** — fixed *or* per-fold-tuned — carrying roughly a 1.0-Sharpe overfitting tax and losing to buy-and-hold on 15 of 17 instruments. That negative result, measured honestly rather than hidden, is the proof the validation machinery tells the truth.

---

## Overview

`quant-trader` is a research-to-production algorithmic trading system targeting US equity swing trades (multi-day to multi-month holds). It is built around a custom vectorized backtesting engine, a modular rules-based signal layer, an out-of-sample validation harness (walk-forward + per-fold optimization), and — in later phases — a live execution layer connected to the Alpaca brokerage API.

The architecture minimizes the gap between backtest and live behavior: the same strategy and (eventually) risk logic that runs in simulation is promoted to paper and live trading without modification.

Key design goals:

- **Honesty over optimism** — every result is measured out-of-sample; the overfitting tax is quantified, not hidden. A strategy is presumed edgeless until it survives validation.
- **Reproducibility** — data fetches are stored locally, deduplicated, and timezone-canonical; backtests are deterministic and result objects are immutable.
- **Modularity** — strategies, backtester, validation, and (later) risk/execution are decoupled and independently testable.
- **Incrementalism** — built in phases, with each layer tested before the next is added and before any live capital is risked.

---

## Universe

The system trades a fixed basket of 17 liquid, long-history ETFs, chosen to avoid the survivorship bias of an individual-stock universe (ETFs are not delisted out of the sample the way single companies are). It is backfilled to January 2008, so every strategy is tested across the 2008 financial crisis, the 2020 COVID crash, and the 2022 rate shock.

| Group | Tickers |
|---|---|
| Broad market | SPY, QQQ, IWM |
| Sectors (S&P 500 SPDRs) | XLK (technology), XLF (financials), XLE (energy), XLV (health care), XLY (consumer discretionary), XLP (consumer staples), XLI (industrials), XLU (utilities), XLB (materials) |
| Bonds | TLT, IEF |
| Commodities | GLD |
| International | EFA, EEM |

---

## Strategies

- **SMA crossover** (baseline / control) — the classic golden-cross / death-cross rule. Deliberately simple and known to be edgeless on this universe; it exists to exercise and validate the pipeline. Its measured out-of-sample failure is a feature, not a disappointment — it is how the validation machinery is proven trustworthy before a real strategy is judged.
- **Time-series momentum** (the first strategy with a genuine economic thesis) — long/flat, monthly rebalance, 12-month lookback: hold an instrument while its own trailing-12-month return is positive, step aside to cash otherwise. Judged primarily on drawdown and downside risk, not return alone, because crisis-avoidance is what it is meant to deliver.

Both implement a single position-signal contract (`list[OHLCVBar] -> int8 signal array`), so any new strategy — or a future ML model — drops into the same backtester and validation harness unchanged.

---

## Architecture

```
[ built & tested ]

+--------+    +-----------+    +--------------+    +-----------------------+
|  Data  | -> | Strategy  | -> |  Backtester  | -> | Walk-forward + Optuna  |
+--------+    +-----------+    +--------------+    | validation + tax       |
 yfinance      SMA /            custom,            +-----------------------+
 DuckDB        momentum         vectorized,
               (rules)          no-lookahead,
                                costs / total-return / cash-yield

[ not yet built — later phases ]

+------+    +-----------+    +-----------------+
| Risk | -> | Execution | -> | Broker (Alpaca) |
+------+    +-----------+    +-----------------+
 sizing,     order mgr,       paper -> small
 stops       fills            live -> scaled
```

Data is fetched once and cached locally in DuckDB (deduplicated, timezone-canonical, content-checked). Strategies consume OHLCV bars and emit position signals. The backtester turns signals into a P&L curve under a next-bar-execution model that makes lookahead bias structurally impossible, with optional transaction costs, total-return (adjusted-close) accounting, and interest on idle cash. The validation layer runs each strategy walk-forward — warming indicators on the train window and scoring only on the unseen test window — optionally re-tuning parameters per fold with Optuna, and reports the in-sample-vs-out-of-sample gap against a buy-and-hold benchmark.

---

## Tech Stack

| Library / Service | Role | Why |
|---|---|---|
| [alpaca-py](https://github.com/alpacahq/alpaca-py) | Brokerage API (paper now, live later) | commission-free US equities, clean REST + WebSocket API, paper trading support |
| [DuckDB](https://duckdb.org) | Local data storage | columnar, zero-infrastructure, fast analytical queries |
| [pandas / numpy](https://pandas.pydata.org) | Data manipulation | industry standard for time-series feature engineering |
| [yfinance](https://github.com/ranaroussi/yfinance) | Historical OHLCV data | free daily bar data for research and backfill |
| [Optuna](https://optuna.org) | Hyperparameter search | per-fold parameter tuning inside walk-forward validation |
| [pytest](https://docs.pytest.org) + [GitHub Actions](https://github.com/features/actions) | Testing / CI | hermetic test suite run on every push; the green check is machine truth, not a prose claim |
| [uv](https://docs.astral.sh/uv/) | Environment & package management | fast, reproducible dependency installs |

**Backtesting is a custom, in-house vectorized engine — intentionally not VectorBT or Backtrader.** Owning the engine keeps the execution model (next-bar fills, no lookahead), the transaction-cost model, and the total-return / cash-yield accounting fully transparent and auditable. Machine-learning signals (e.g. LightGBM) are a later, exploratory phase, pursued only if rules-based research warrants it — they are not part of the current system.

---

## Project Structure

```
quant-trader/
├── src/
│   ├── brokers/        # Alpaca client wrapper + broker abstraction (hard paper-only guard)
│   ├── data/           # yfinance fetcher, DuckDB store, schema, universe loader
│   ├── features/       # vectorized indicators (SMA, RSI, log returns)
│   ├── strategies/     # position-signal strategies (SMA crossover, time-series momentum)
│   ├── backtest/       # custom vectorized engine, result types, shared metrics
│   ├── research/       # walk-forward, Optuna fitter, runners, overfitting-tax, compare CLIs
│   ├── risk/           # position sizing, drawdown limits, exposure rules (planned)
│   └── execution/      # order manager, fill tracking, rebalance scheduler (planned)
├── config/             # universe.yaml (etf_basket + sp500)
├── scripts/            # backfill, daily update, data-health, one-off migrations
├── tests/              # pytest unit + integration tests (integration gated out of CI)
├── notebooks/          # research and signal exploration (not promoted to src)
├── data/               # local DuckDB database (gitignored)
├── logs/               # runtime logs (gitignored)
├── DEV_LOG.md          # running development log
├── CLAUDE.md           # project conventions for AI-assisted sessions
├── pyproject.toml
└── .env                # API keys and environment config (gitignored)
```

---

## Roadmap

- [x] **Phase 0 — Foundations**: project scaffold, uv environment, broker abstraction over Alpaca with a hard paper-only safety guard
- [x] **Phase 1 — Data & research infrastructure**: yfinance + DuckDB ingestion, fixed ETF basket backfilled to 2008, daily incremental update, structural + content health checks, vectorized indicators
- [x] **Phase 2 — Strategy, backtester & validation**: custom vectorized backtester (next-bar execution, structurally no lookahead, frozen results), SMA-crossover baseline + time-series momentum, walk-forward validation, per-fold Optuna tuning, overfitting-tax measurement against a buy-and-hold benchmark
- [ ] **Phase 3 — Execution realism & risk** (in progress): transaction costs ✓, total-return / adjusted-close accounting ✓, interest on idle cash ✓ → momentum out-of-sample verdict, then position sizing, drawdown circuit-breakers, exposure limits
- [ ] **Phase 4 — Machine learning** (exploratory): feature pipeline and model-based signals (e.g. LightGBM) run through the same validation harness — only if rules-based research warrants it
- [ ] **Phase 5 — Live execution**: Alpaca paper → small live → scaled, with position reconciliation, monitoring, and failsafe shutdown

---

## Getting Started

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the virtual environment and install dependencies
uv sync

# Copy and configure environment variables (Alpaca paper keys)
cp .env.example .env

# Run the test suite
uv run pytest -q
```

---

## Disclaimer

This project is built for educational and research purposes. Nothing in this repository constitutes financial advice. Past backtest performance does not guarantee future results. Use at your own risk.