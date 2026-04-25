# quant-trader

Systematic swing trading bot for US equities.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)

---

## Overview

`quant-trader` is a research-to-production algorithmic trading system targeting US equity swing trades (multi-day to multi-week holds). The system is built around an event-driven backtesting engine, a modular signal pipeline, and a live execution layer connected to the Alpaca brokerage API.

The architecture is designed to minimize the gap between backtest and live behavior: the same strategy and risk logic that runs in simulation is promoted to paper and live trading without modification.

Key design goals:

- **Reproducibility** — all data fetches are versioned and stored locally; backtests are deterministic
- **Modularity** — strategies, risk rules, and execution are decoupled and independently testable
- **Incrementalism** — the system is built in phases, with paper trading validating each layer before live capital is risked

---

## Architecture

```
+--------+     +-----------+     +------+     +-----------+     +----------------+
|  Data  | --> | Strategy  | --> | Risk | --> | Execution | --> | Broker (Alpaca)|
+--------+     +-----------+     +------+     +-----------+     +----------------+
    |                |                              |
  DuckDB         Signal                       Order Manager
  yfinance       Ensemble                     Position Tracker
  Alpaca         (LightGBM)
```

Data is fetched once and cached locally in DuckDB. Strategies consume OHLCV bars and emit signals. The risk engine sizes and filters those signals. The execution layer translates approved orders into Alpaca API calls.

---

## Tech Stack

| Library / Service | Role | Why |
|---|---|---|
| [alpaca-py](https://github.com/alpacahq/alpaca-py) | Brokerage API | Commission-free US equities, clean REST + WebSocket API, paper trading support |
| [DuckDB](https://duckdb.org) | Local data storage | Columnar, zero-infrastructure, fast analytical queries on Parquet/CSV without a server |
| [pandas / numpy](https://pandas.pydata.org) | Data manipulation | Industry standard for time-series feature engineering |
| [yfinance](https://github.com/ranaroussi/yfinance) | Historical OHLCV data | Free daily bar data for research and backfill |
| [VectorBT / Backtrader](https://vectorbt.dev) | Backtesting engine | VectorBT for vectorized parameter sweeps; Backtrader for event-driven simulation |
| [LightGBM](https://lightgbm.readthedocs.io) | ML signal ensemble | Fast gradient boosting, handles tabular financial features well, interpretable |
| [Oracle Cloud (OCI)](https://www.oracle.com/cloud/) | Deployment | Always-free tier sufficient for the scheduler and execution daemon |
| [ruff](https://docs.astral.sh/ruff/) | Linting / formatting | Single tool replacing flake8 + isort + black, orders of magnitude faster |

---

## Project Structure

```
quant-trader/
├── src/
│   ├── brokers/        # Alpaca client wrapper and broker abstraction
│   ├── data/           # Data fetchers, DuckDB schema, bar storage
│   ├── strategies/     # Signal generation logic (technical + ML)
│   ├── backtest/       # Event-driven and vectorized backtesting engine
│   ├── risk/           # Position sizing, drawdown limits, exposure rules
│   ├── execution/      # Order manager, fill tracking, rebalance scheduler
│   └── utils/          # Config loader, logger, common helpers
├── notebooks/          # Research and signal exploration (not promoted to src)
├── tests/              # pytest unit and integration tests
├── scripts/            # One-off utilities (data backfill, diagnostics)
├── data/               # Local DuckDB databases and Parquet files (gitignored)
├── logs/               # Runtime logs (gitignored)
├── pyproject.toml
└── .env                # API keys and environment config (gitignored)
```

---

## Roadmap

- [x] **Phase 0 — Foundations**: project scaffold, pyproject.toml, gitignore, folder structure
- [ ] **Phase 1 — Data & Research**: Alpaca + yfinance ingestion, DuckDB schema, bar storage, basic EDA notebooks
- [ ] **Phase 2 — First Strategy**: momentum / mean-reversion signal, vectorized backtest, baseline performance metrics
- [ ] **Phase 3 — Production Backtest**: event-driven engine, realistic fills, slippage and commission modeling, walk-forward validation
- [ ] **Phase 4 — Risk Engine**: Kelly / fixed-fractional sizing, max drawdown circuit breakers, sector exposure limits
- [ ] **Phase 5 — Paper Deployment**: Alpaca paper account, live data feed, end-of-day order submission, P&L tracking
- [ ] **Phase 6 — ML Layer**: feature engineering pipeline, LightGBM ensemble, signal confidence scoring, out-of-sample validation
- [ ] **Phase 7 — Live Trading**: promote to live account, position reconciliation, failsafe shutdown logic
- [ ] **Phase 8 — Observability**: structured logging, trade journal, equity curve dashboard, alerting
- [ ] **Phase 9 — Continuous Research**: rolling retrain schedule, regime detection, strategy rotation

---

## Getting Started

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Copy and configure environment variables
cp .env.example .env
```

---

## Disclaimer

This project is built for educational and research purposes. Nothing in this repository constitutes financial advice. Past backtest performance does not guarantee future results. Use at your own risk.
