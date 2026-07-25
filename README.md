# quant-trader

A systematic swing-trading research and execution system for US equity ETFs, built rigor-first.

Most trading-strategy projects show an impressive backtest. This one shows why several reasonable strategies *do not* beat simply holding a basket of ETFs once you test them honestly - and it quantifies exactly how much a naively-tuned version would have fooled you. The honest negative result, measured with tooling built specifically to catch self-deception, is the point of the project.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-425%20passing-brightgreen)

---

## What this is

`quant-trader` is a from-scratch research-to-execution pipeline for swing trading (multi-day to multi-month holds) a fixed basket of 17 liquid ETFs. It has three parts, each independently tested:

1. A **custom vectorized backtester** that turns strategy signals into a P&L curve under a next-bar-execution model that makes lookahead bias structurally impossible.
2. A **validation harness** - walk-forward testing with per-fold parameter tuning - that measures how a strategy performs on data it was never allowed to see, and quantifies how much it overfit.
3. An **execution layer** that connects to the Alpaca brokerage API and places real orders on a paper account, behind a hard paper-only safety guard.

It is built rigor-first: the goal was never a get-rich strategy, but a system honest enough to tell the difference between a real edge and statistical noise - and to say so plainly when there is no edge.

---

## The honest verdict

Three strategies were tested through the same harness. Every one of them, tested out-of-sample and after realistic costs, **fails to beat a simple buy-and-hold of the same basket.** These are not cherry-picked; they are the results the harness produced, reproduced on the current codebase.

| Strategy | Out-of-sample Sharpe | Buy-and-hold Sharpe | Overfitting tax | Beat buy-and-hold on |
|---|---|---|---|---|
| SMA crossover (control) | +0.04 | +0.41 | +1.03 | 2 of 17 |
| Time-series momentum (fixed) | +0.43 | +0.52 | +0.51 | 4 of 17 |
| Cross-sectional rotation | +0.44 | +0.59 | +0.72 | 0 of 1 (loses) |

*Sharpe ratio = return per unit of risk. The **overfitting tax** is the gap between how good a strategy looked while its parameters were being tuned (in-sample) and how it actually performed on unseen data (out-of-sample) - a direct, dollar-free measure of self-deception.*

Two things stand out, and both are the kind of finding this system exists to surface:

- **The overfitting tax is consistently large and positive.** The SMA control looked like a +1.01 Sharpe strategy while being tuned and delivered roughly zero (in fact slightly negative, -0.03) out-of-sample - a +1.03 tax. Momentum and rotation show the same pattern (+0.51 and +0.72). This is the machinery catching optimism in the act: a backtest that looks good because it was fit to its own test data.
- **Per-fold tuning did not rescue any of them.** Re-optimizing parameters on each fold made rotation *worse* than a fixed rule and barely moved momentum. More tuning bought more overfitting, not more edge.

The one genuinely defensible positive: **fixed time-series momentum reduced maximum drawdown on 13 of 17 instruments** (mean 32.9% vs buy-and-hold's 38.4%). It does not beat the market on return, but it delivers the downside protection that absolute momentum is theorized to provide. Knowing the difference between "made money" and "reduced risk" is the whole point.

**Bottom line:** on this universe, these classic strategies are beta, not alpha. That is a real, useful finding - and reporting it honestly, rather than tuning until a backtest looks profitable, is what makes the rest of the system trustworthy.

---

## Why the overfitting tax matters

The single most common failure in quantitative trading is a strategy that looks brilliant in backtest and loses money live. It happens because the strategy was, knowingly or not, fit to the historical data it was tested on. A backtest Sharpe of 2.5 usually means "I tried many variations and kept the luckiest one," not "I found an edge."

This project measures that directly. For each strategy it runs a **walk-forward** test: tune parameters on an early window of history, then score - untouched - on the next window the strategy has never seen, and roll forward. The **overfitting tax** is the average gap between the in-sample Sharpe the tuner saw and the out-of-sample Sharpe it actually delivered. A large positive tax means the strategy's apparent skill was mostly hindsight. Building this measurement first, and trusting its verdict even when the verdict is disappointing, is the discipline the whole repo is organized around.

---

## Universe

A fixed basket of 17 liquid, long-history ETFs, backfilled to January 2008 so every strategy is tested across the 2008 crisis, the 2020 COVID crash, and the 2022 rate shock. ETFs (rather than individual stocks) avoid survivorship bias - they are not delisted out of the sample the way bankrupt companies are.

| Group | Tickers |
|---|---|
| Broad market | SPY, QQQ, IWM |
| Sectors | XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB |
| Bonds | TLT, IEF |
| Commodities | GLD |
| International | EFA, EEM |

---

## Architecture

```
[ research + validation ]                                    [ execution ]

 Data  ->  Strategy  ->  Backtester  ->  Walk-forward +        Rebalance  ->  Broker (Alpaca)
                                          Optuna + tax          engine
 yfinance  SMA /         custom,         in-sample vs          observe ->     paper-only,
 DuckDB    momentum /    vectorized,     out-of-sample         compute ->     guarded;
           rotation      no lookahead,   vs buy-and-hold       submit         live paper
                         costs + returns                                      smoke-tested
```

- **Data** is fetched once via yfinance and cached in DuckDB (deduplicated, timezone-canonical).
- **Strategies** implement a single position-signal contract, so any new strategy - or a future ML model - drops into the backtester and harness unchanged.
- **The backtester** uses next-bar execution (a signal at bar i earns bar i's return, never bar i+1's), which makes lookahead bias structurally impossible, with optional transaction costs, total-return accounting, and interest on idle cash.
- **The validation harness** runs each strategy walk-forward, optionally re-tuning per fold with Optuna, and reports the in-sample-vs-out-of-sample gap against buy-and-hold.
- **The execution layer** reconciles a target allocation to concrete orders (a pure, tested function) and submits them through a broker abstraction whose `verify_paper_account()` guard is the first statement in the run path - so an order can never reach a live account. It has been run end-to-end against the real Alpaca paper API.

---

## Engineering rigor

- **425 tests passing** (429 including live-API integration tests, which are gated out of continuous integration). Every test in CI is hermetic - no network, no credentials, no live data.
- **CI on every push** (GitHub Actions): `uv run pytest -q -m "not integration"` on a clean Ubuntu environment. The green check is machine truth, not a prose claim.
- **Custom backtester, deliberately not a library.** Owning the engine keeps the execution model, cost model, and return accounting fully transparent and auditable.
- **Paper-only safety, structurally enforced.** Three independent guards (a constructor assertion, a hardcoded paper flag, and an account-type check that runs before any order) make live trading impossible in the current codebase.
- **Pure/impure separation** throughout: arithmetic (sizing, return combination, ranking) lives in pure functions testable with plain data; I/O is pushed to thin shells. This is why the whole thing is testable offline.

---

## Tech stack

| Tool | Role |
|---|---|
| Python 3.11+ (CI on 3.13), uv | language + fast, reproducible env management |
| DuckDB | zero-infrastructure local columnar data store |
| pandas / numpy | time-series computation |
| yfinance | historical daily OHLCV data |
| Optuna | per-fold hyperparameter search inside walk-forward validation |
| alpaca-py | brokerage API (paper trading) |
| pytest + GitHub Actions | hermetic test suite, run on every push |

---

## Project structure

```
quant-trader/
├── src/
│   ├── data/         # yfinance fetcher, DuckDB store, schema, universe loader
│   ├── features/     # vectorized indicators
│   ├── strategies/   # position-signal strategies (SMA, time-series momentum)
│   ├── backtest/     # custom vectorized engine, result types, metrics
│   ├── research/     # walk-forward, Optuna fitter, rotation, overfitting-tax, compare CLIs
│   ├── brokers/      # Alpaca client + broker abstraction (hard paper-only guard)
│   └── execution/    # rebalance reconciliation + guarded runner
├── config/           # universe.yaml (the 17-ETF basket)
├── scripts/          # verdict CLIs, data health, backfill, paper smoke check
├── tests/            # pytest unit + integration tests (integration gated out of CI)
├── DEV_LOG.md        # running development log
└── pyproject.toml
```

Reproduce the verdicts yourself:

```bash
uv sync
uv run pytest -q -m "not integration"           # the hermetic suite
uv run python scripts/overfitting_tax.py         # SMA control verdict
uv run python scripts/momentum_walkforward.py    # time-series momentum verdict
uv run python scripts/rotation_verdict.py         # rotation verdict
```

---

## Since the initial verdict

The execution arc has since been completed. The rebalance engine now emits orders sells-before-buys - all SELLs precede all BUYs so a rotation's funding sells clear before the buys they fund. An autonomous daily runner ingests, decides, and rebalances unattended on a paper account with logging and email notification, deployed via a macOS launchd LaunchAgent as best-effort local scheduling (it runs on time when the machine is awake and catches up a run missed during sleep on the next wake). This is paper-only and best-effort by design; the live acted-and-notify path awaits its first real weekday rebalance decision.

## What's next

The research arc has delivered its honest verdict. The one genuinely-open thread, scoped honestly:

- **Continued edge search** - new hypotheses (cross-asset signals, others) tested one at a time through the same harness, accepting each verdict. Most will fail; that is the nature of the search, and the tax is the referee.

---

## Disclaimer

Built for educational and research purposes. Nothing here is financial advice. Past backtest performance does not guarantee future results.
