# Day-by-Day Trading Bot Build Plan

**Your profile:** Intermediate Python, 1–2 hrs/day, nights & weekends
**Total duration:** ~52 weeks (Days 1–365)
**Cadence:** 5 days/week of work (weekends flexible — catch-up or rest)

---

## How to Use This Plan

- **Each day has one concrete task.** Finish it or log where you stopped.
- **Weekends are float time.** Use them to catch up when a weekday task overruns, or to rest.
- **Keep a daily log.** A single `dev_log.md` file in your repo: date, what you did, what broke, what's next. This becomes priceless at month 6 when you can't remember why you made a choice.
- **When you fall behind, don't skip — just shift.** Falling a week behind by month 3 is normal. Falling 4 weeks behind means something's wrong with scope or time budget.
- **Red flag:** if a single task takes more than 3 days, stop and reassess. Either the task is too big (split it) or you're stuck (ask for help, search GitHub, read docs).

---

# MONTH 1 — Foundations & First Strategy Baseline

## Week 1 — Environment & First API Call

**Day 1 (Mon):** Install Python 3.11+ via `pyenv`. Install `uv` as package manager. Install VS Code or Cursor. Create GitHub account if needed.

**Day 2 (Tue):** Create private GitHub repo `trading-bot`. Clone locally. Initialize `pyproject.toml`. Write `.gitignore` (include `.env`, `__pycache__`, `*.db`, `data/`).

**Day 3 (Wed):** Create folder structure from Phase 0. Write a basic `README.md` explaining what the project is (for your future self). First commit.

**Day 4 (Thu):** Sign up for Alpaca paper account at alpaca.markets. Generate API keys. Create `.env` file locally with `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`. Install `alpaca-py` SDK.

**Day 5 (Fri):** Write `scripts/hello_alpaca.py` — connect to Alpaca, print account info (buying power, equity, status). Verify it runs.

**Weekend:** Catch-up or rest. If ahead, read Alpaca's API docs cover-to-cover.

## Week 2 — First Orders & Data Fetch

**Day 6 (Mon):** Write `scripts/first_order.py` — submit a paper market order for 1 share of SPY. Verify it fills in Alpaca's dashboard.

**Day 7 (Tue):** Extend the script: submit a limit order, cancel it, submit again and let it fill. Get comfortable with order states (new, filled, canceled, rejected).

**Day 8 (Wed):** Create `src/brokers/alpaca_broker.py`. Wrap Alpaca SDK in a clean class with methods: `get_account()`, `get_positions()`, `submit_order()`, `cancel_order()`, `get_bars()`.

**Day 9 (Thu):** Install `yfinance`. Create `src/data/yfinance_fetcher.py` — function that fetches N years of daily bars for a list of tickers.

**Day 10 (Fri):** Install `duckdb`. Create `src/data/storage.py` — functions to save DataFrames to DuckDB and read them back. Write a test: save AAPL 5-year bars, read them back, verify integrity.

**Weekend:** Catch-up. Commit everything.

## Week 3 — Bulk Historical Data

**Day 11 (Mon):** Get the current S&P 500 ticker list (scrape Wikipedia or use `pandas_datareader`). Save to `data/sp500_current.csv`.

**Day 12 (Tue):** Write a script that fetches 10 years of daily bars for all 500 tickers, stores in DuckDB. Run it overnight if needed. Expect some tickers to fail — log and skip.

**Day 13 (Wed):** Write data validation: missing dates, zero-volume days, extreme price jumps (check for unadjusted splits). Print a quality report.

**Day 14 (Thu):** Research survivorship bias. Read 2–3 articles. Download a historical S&P 500 membership dataset from Kaggle ($0, search "sp500 historical constituents"). Load it into DuckDB.

**Day 15 (Fri):** Write a function `get_sp500_members_on(date)` that returns the actual index members on any historical date. Verify against known changes (e.g., Tesla added Dec 2020).

**Weekend:** Read the first two chapters of *Advances in Financial Machine Learning* (López de Prado). Uncomfortable? That's the point.

## Week 4 — Features & Exploration

**Day 16 (Mon):** Set up Jupyter Lab. Create `notebooks/01_exploration.ipynb`. Plot SPY price, returns, rolling vol over 10 years. Get comfortable.

**Day 17 (Tue):** Write `src/data/features.py` with functions: `simple_returns`, `log_returns`, `rolling_volatility`, `sma`, `ema`, `rsi`. Unit-test each on a known series.

**Day 18 (Wed):** For each S&P 500 member, compute 12-month returns and 1-month returns (trailing). Store as features table in DuckDB.

**Day 19 (Thu):** In a notebook, compute 12-1 momentum for every month-end from 2015–2024. Look at the distribution. Verify the top-decile stocks look reasonable for known dates.

**Day 20 (Fri):** Buffer day — catch up on any week's slippage, or refactor messy code from week 3. Commit everything. Tag `v0.1-foundations`.

**Weekend:** You're at end of Month 1. Reread the roadmap. Check your progress honestly.

---

# MONTH 2 — First Real Backtest

## Week 5 — Strategy Logic

**Day 21 (Mon):** Create `src/strategies/base.py` with an abstract `Strategy` class: `generate_signals(date) -> target_weights_dict`.

**Day 22 (Tue):** Implement `src/strategies/momentum_12_1.py`. On each rebalance date, rank SP500 members by 12-1 momentum, target equal weights in top decile, zero elsewhere.

**Day 23 (Wed):** Test the strategy on 3 specific dates (e.g., Jan 2018, Mar 2020, Dec 2023). Verify by hand that the top 10 stocks look right.

**Day 24 (Thu):** Install `vectorbt`. Read the tutorial. Bookmark the docs — you'll refer to them often.

**Day 25 (Fri):** Write `src/backtest/vbt_runner.py` — a thin wrapper that takes a Strategy object, runs it through vectorbt, returns results.

**Weekend:** Rest. Let last week sink in.

## Week 6 — First Backtest Results

**Day 26 (Mon):** Run your first full backtest: 12-1 momentum on S&P 500, 2014–2024, weekly rebalance. No costs yet. Print metrics.

**Day 27 (Tue):** Compare your results to academic literature (the Jegadeesh-Titman baseline). If your Sharpe is wildly higher than ~0.8, you have a bug — likely look-ahead or survivorship bias.

**Day 28 (Wed):** Audit for look-ahead bias: at signal time on date T, are you using any data from date T or later? Fix any violations.

**Day 29 (Thu):** Audit for survivorship bias: are you using point-in-time S&P 500 membership? If not, fix it. Expect Sharpe to drop.

**Day 30 (Fri):** Add realistic costs: 5bps slippage per trade, re-run. Sharpe will drop again. This is reality.

**Weekend:** Take a breath. You have a real backtest now.

## Week 7 — Validation & Walk-Forward

**Day 31 (Mon):** Split data: in-sample 2014–2020, out-of-sample 2021–2024. Re-run strategy, report both Sharpes. Document the gap.

**Day 32 (Tue):** Stress-test: what does performance look like in 2018 (vol spike), March 2020 (COVID), 2022 (momentum crash)?

**Day 33 (Wed):** Write a backtest report notebook: equity curve, drawdown chart, monthly returns heatmap, summary stats table. This becomes your template.

**Day 34 (Thu):** Implement 2 variations: quarterly rebalance, and monthly rebalance. Compare results. Learn how much rebalance frequency matters.

**Day 35 (Fri):** Buffer day. Commit, tag `v0.2-first-backtest`.

**Weekend:** Read Chapter 3 of López de Prado — on sample weights and data labeling.

## Week 8 — Strategy Polish

**Day 36 (Mon):** Add turnover constraint: don't rebalance a position if the new weight differs from current by less than 1%. Re-backtest.

**Day 37 (Tue):** Add minimum holding period (e.g., 2 weeks) to reduce whipsaw. Compare results.

**Day 38 (Wed):** Add basic sector cap: no more than 25% in any GICS sector. Pull sector data from yfinance.

**Day 39 (Thu):** Compile Month 2's results into a single polished notebook. This is your "baseline strategy" reference going forward.

**Day 40 (Fri):** Buffer / rest. End of Month 2. You now have a real, validated, simple strategy.

**Weekend:** Honest self-review. Would you put real money on this? Why or why not?

---

# MONTH 3 — Event-Driven Backtest & Paper Trading Setup

## Week 9 — Nautilus/Backtrader Setup

**Day 41 (Mon):** Decide: Nautilus Trader (modern, harder) or Backtrader (simpler, older). For 1–2 hrs/day, pick **Backtrader** — you can upgrade later.

**Day 42 (Tue):** Install Backtrader. Run the quickstart tutorial. Get a simple SMA crossover strategy working.

**Day 43 (Wed):** Design your broker abstraction: `src/brokers/base.py` with abstract methods. Two implementations: `BacktestBroker` (Backtrader-backed) and `AlpacaBroker` (already done).

**Day 44 (Thu):** Port your momentum strategy to run through Backtrader. Start simple: single stock first (SPY momentum).

**Day 45 (Fri):** Debug until the single-stock Backtrader version matches your vectorbt result within a few bps.

**Weekend:** Rest — this week was hard.

## Week 10 — Multi-Asset Backtrader

**Day 46 (Mon):** Extend to multi-asset: load all S&P 500 members as data feeds in Backtrader.

**Day 47 (Tue):** Implement the weekly rebalance logic in Backtrader. This is the tricky part — event-driven means you think in bars, not matrices.

**Day 48 (Wed):** Add proper fill logic: fill at next bar's open, not current bar's close.

**Day 49 (Thu):** Run the full event-driven backtest. Compare to vectorbt version. Any divergence is a bug — investigate.

**Day 50 (Fri):** Document the differences. Usually the event-driven version is slightly worse (more realistic).

**Weekend:** Commit. Tag `v0.3-eventdriven`.

## Week 11 — Fill & Cost Modeling

**Day 51 (Mon):** Model commission (zero for Alpaca, but code it anyway for other brokers).

**Day 52 (Tue):** Model slippage as a function of bar volatility: `slippage_bps = k * daily_vol`. Calibrate `k` to match observed Alpaca fills later.

**Day 53 (Wed):** Handle corporate actions: dividends (Backtrader does some of this natively), stock splits (usually handled by yfinance adjustment), mergers (trickier — punt for now but document).

**Day 54 (Thu):** Add cash accounting: T+1 settlement, initial margin for any shorts (we're long-only for now, but build the infrastructure).

**Day 55 (Fri):** Re-run backtest with all realistic costs. Honestly evaluate: is the Sharpe still meaningful?

**Weekend:** Rest.

## Week 12 — Paper Trading Pipeline

**Day 56 (Mon):** Create `src/execution/order_manager.py`. Takes target weights, current positions, and emits order list.

**Day 57 (Tue):** Make it work for both brokers (paper backtest broker and live Alpaca paper). Same code, different broker.

**Day 58 (Wed):** Write `scripts/run_live_paper.py` — script that runs daily, generates signals, places paper orders via Alpaca.

**Day 59 (Thu):** Add market-hours awareness: don't trade on weekends, market holidays. Use `pandas_market_calendars` library.

**Day 60 (Fri):** Run the script manually end-of-day. Verify it placed correct paper orders in Alpaca dashboard. End of Month 3.

**Weekend:** Commit. Tag `v0.4-paper-pipeline`. Big milestone.

---

# MONTH 4 — Risk Engine

## Week 13 — Position Sizing

**Day 61 (Mon):** Read about Kelly Criterion. Watch Ed Thorp's lectures on YouTube if helpful.

**Day 62 (Tue):** Implement `src/risk/kelly.py` — fractional Kelly sizing (0.25x) given expected return and variance estimates.

**Day 63 (Wed):** Replace equal-weight with Kelly-weighted targets in your strategy. Backtest. Compare.

**Day 64 (Thu):** Add single-position cap: max 5% per stock. Re-backtest.

**Day 65 (Fri):** Add sector exposure cap: max 25% per GICS sector. Re-backtest.

**Weekend:** Read *Active Portfolio Management* Chapter 1–2.

## Week 14 — Drawdown Circuit Breakers

**Day 66 (Mon):** Design circuit breaker states: NORMAL, REDUCED (halve sizes), HALTED (flatten).

**Day 67 (Tue):** Implement `src/risk/circuit_breaker.py`. Monitor weekly and total drawdown. Transition states automatically.

**Day 68 (Wed):** Plug the circuit breaker into the order manager. Test by simulating a drawdown scenario.

**Day 69 (Thu):** Add a manual override: a flag file or environment variable that forces HALTED state. You need a kill switch.

**Day 70 (Fri):** Write a test that literally triggers the circuit breaker. Verify it actually stops trading.

**Weekend:** Rest.

## Week 15 — Portfolio Construction

**Day 71 (Mon):** Read about CVaR optimization. The `cvxpy` library docs are a good reference.

**Day 72 (Tue):** Install `cvxpy`. Write a toy example: minimize portfolio variance subject to a target return.

**Day 73 (Wed):** Extend to CVaR: minimize 95% CVaR subject to return target and position caps.

**Day 74 (Thu):** Make it a drop-in replacement for Kelly sizing. Backtest with CVaR weighting.

**Day 75 (Fri):** Compare results: Kelly vs CVaR vs equal-weight. Pick the winner — but only if the improvement is robust across periods.

**Weekend:** Commit. Tag `v0.5-risk-engine`.

## Week 16 — Volatility Targeting & Logging

**Day 76 (Mon):** Add portfolio vol targeting: scale all positions so estimated portfolio vol = 10% annualized.

**Day 77 (Tue):** Backtest vol-targeted version. Note how drawdowns compress.

**Day 78 (Wed):** Build `src/utils/logger.py` — structured JSON logging. Every risk decision, every order, every fill logged with timestamp.

**Day 79 (Thu):** Pipe logs into a DuckDB table. Write queries to audit: "show me every order rejected by risk yesterday."

**Day 80 (Fri):** Buffer. End of Month 4. You now have a legitimate risk-managed system.

**Weekend:** You're 1/3 done. Review honestly.

---

# MONTH 5 — Live Paper Deployment

## Week 17 — Deployment Infrastructure

**Day 81 (Mon):** Sign up for Oracle Cloud Free Tier. Provision a free ARM VM (4 cores, 24GB RAM).

**Day 82 (Tue):** SSH into it. Install Python 3.11, git, systemd basics. Clone your repo.

**Day 83 (Wed):** Get your bot running manually on the VM. Use paper keys. Verify connectivity to Alpaca.

**Day 84 (Thu):** Write a `systemd` service file that runs your bot daily at a fixed time (e.g., 3:45pm ET). Test the scheduling.

**Day 85 (Fri):** Add auto-restart on failure. If the bot crashes, systemd restarts it. Log to a file.

**Weekend:** Rest.

## Week 18 — Monitoring Basics

**Day 86 (Mon):** Create a Discord server for alerts (or use email). Set up a webhook.

**Day 87 (Tue):** Write `src/utils/alerts.py` — send Discord messages on: successful daily run, any error, any circuit breaker trigger.

**Day 88 (Wed):** Create a daily summary: PnL, positions, exposures. Send to Discord at end of each trading day.

**Day 89 (Thu):** Create a weekly summary: live vs backtest performance, signal counts, any anomalies.

**Day 90 (Fri):** Test all alerting paths. Cause a fake error. Verify alert fires.

**Weekend:** Tag `v0.6-deployed`.

## Week 19 — Go-Live (Paper)

**Day 91 (Mon):** Flip the switch. Bot runs live on the VM, paper money, starting today. Monitor closely.

**Day 92 (Tue):** First full day of live paper trading. Review every decision the bot made. Does it match what your backtest would have done?

**Day 93 (Wed):** Any divergence from expected behavior — pause and investigate. The early days surface the worst bugs.

**Day 94 (Thu):** Create a "live vs backtest" comparison notebook. Reconcile daily.

**Day 95 (Fri):** End of week 1 of paper live. Document every bug found. Commit fixes.

**Weekend:** Let it run.

## Week 20 — Stabilization

**Day 96 (Mon):** Review weekend — did anything break? Data feed issues over the weekend?

**Day 97 (Tue):** Fix any remaining bugs. Bugs found in this phase are the ones that would have cost you real money.

**Day 98 (Wed):** Start the 60-day clock. From this point, target 60 consecutive trading days of clean operation before considering real money.

**Day 99 (Thu):** Implement a pre-trade validation hook: before any order, check that positions reconcile with broker state.

**Day 100 (Fri):** End of Month 5. Bot is live-paper-trading. Big milestone. Commit, tag `v1.0-paper-live`.

**Weekend:** Rest. You earned it.

---

# MONTH 6 — Observability & Signal Expansion (Part 1)

## Week 21 — Proper Monitoring

**Day 101 (Mon):** Install Prometheus client library. Instrument your bot: counters for orders placed, fills, errors.

**Day 102 (Tue):** Install Prometheus server on your VM. Configure it to scrape your bot's metrics.

**Day 103 (Wed):** Install Grafana. Connect to Prometheus. Build your first dashboard: orders, PnL, positions.

**Day 104 (Thu):** Add alerts to Grafana: no data for 2 hours, error rate spike, drawdown threshold.

**Day 105 (Fri):** Polish the dashboard. This is your daily morning view.

**Weekend:** Let everything run, observe.

## Week 22 — Data Quality

**Day 106 (Mon):** Implement data freshness checks: did yfinance return data as expected today? Alert if not.

**Day 107 (Tue):** Implement price sanity checks: any stock with a move >20% intraday gets flagged for manual review.

**Day 108 (Wed):** Implement position reconciliation: compare your internal position state to broker's state. Alert on mismatch.

**Day 109 (Thu):** Implement cash reconciliation. Alert on mismatch.

**Day 110 (Fri):** Run all data quality checks on historical data — find the bugs now, not later.

**Weekend:** Rest.

## Week 23 — Mean Reversion Signal

**Day 111 (Mon):** Research short-term mean reversion (Lehmann 1990, Jegadeesh 1990). Read the basics.

**Day 112 (Tue):** Implement a simple mean-reversion signal: rank stocks by past 5-day return, buy bottom decile.

**Day 113 (Wed):** Backtest standalone. Expect lower Sharpe than momentum, but uncorrelated returns are valuable.

**Day 114 (Thu):** Verify it's actually uncorrelated to momentum signal. Compute correlation of daily signal values.

**Day 115 (Fri):** Add proper costs — mean reversion has high turnover and eats costs alive.

**Weekend:** Read Chapters 1–3 of *Expected Returns* by Ilmanen.

## Week 24 — Quality Signal

**Day 116 (Mon):** Define quality factor: high ROE + low leverage + earnings stability. Pull data from yfinance or SEC filings.

**Day 117 (Tue):** Build the quality score for S&P 500 members.

**Day 118 (Wed):** Backtest quality standalone. Expect it to be slow-moving, low-turnover, mediocre Sharpe.

**Day 119 (Thu):** Check correlation to momentum and mean-reversion signals. Uncorrelated? Good.

**Day 120 (Fri):** End of Month 6. Commit, tag `v1.1-signals-expanded`.

**Weekend:** Self-review. Are you shipping code that's clean enough to maintain?

---

# MONTH 7 — Signal Expansion (Part 2) & Meta-Learner

## Week 25 — Low Vol & Earnings Drift

**Day 121 (Mon):** Implement low-volatility signal: rank by trailing 1-year volatility, buy lowest-vol quintile.

**Day 122 (Tue):** Backtest low-vol. Compare Sharpe to market.

**Day 123 (Wed):** Research post-earnings announcement drift (PEAD).

**Day 124 (Thu):** Get historical earnings dates and surprise data. Free source: scrape from Yahoo Finance or use `pandas_datareader`. Check point-in-time availability.

**Day 125 (Fri):** Implement a simple PEAD signal: buy stocks that beat earnings EPS by >5%, hold 60 days.

**Weekend:** Rest.

## Week 26 — Signal Consolidation

**Day 126 (Mon):** Create a unified signals table in DuckDB: one row per (date, ticker), columns for each signal.

**Day 127 (Tue):** Compute all signals daily for historical range. This is now your alpha research database.

**Day 128 (Wed):** Plot signal correlations, signal decay curves (how long does a signal predict returns?).

**Day 129 (Thu):** Compute each signal's standalone Sharpe on fresh walk-forward. Rank them.

**Day 130 (Fri):** Document signal characteristics in a research notebook.

**Weekend:** Read López de Prado Chapter 7 (cross-validation for finance).

## Week 27 — Meta-Learner Setup

**Day 131 (Mon):** Install LightGBM, Optuna. Verify both work.

**Day 132 (Tue):** Prepare training data: features = signals, target = forward 5-day return (winsorized).

**Day 133 (Wed):** Implement purged time-series cross-validation. This is critical — do not skip.

**Day 134 (Thu):** Train first LightGBM model. Check feature importances. Verify nothing is leaking future info.

**Day 135 (Fri):** Compare meta-learner predictions to single signals. Is there meaningful improvement?

**Weekend:** Commit progress.

## Week 28 — Hyperparameter Search

**Day 136 (Mon):** Set up Optuna study for LightGBM. Define search space (num_leaves, learning_rate, etc.).

**Day 137 (Tue):** Run 100-trial search with purged CV. This may take a full evening.

**Day 138 (Wed):** Review results. Any params at search boundaries? Expand range if so.

**Day 139 (Thu):** Final model with best params. Backtest the full ensemble strategy.

**Day 140 (Fri):** End of Month 7. Commit, tag `v1.2-meta-learner`.

**Weekend:** Rest.

---

# MONTH 8 — Overfitting Check & Regime Awareness

## Week 29 — Deflated Sharpe & PBO

**Day 141 (Mon):** Read López de Prado's paper on the deflated Sharpe ratio.

**Day 142 (Tue):** Implement deflated Sharpe calculation. Apply to your ensemble.

**Day 143 (Wed):** Implement Probability of Backtest Overfitting (PBO). Run it.

**Day 144 (Thu):** Honest evaluation: if PBO > 0.5, you're probably overfitting. Scale back model complexity.

**Day 145 (Fri):** Document all overfitting diagnostics in a report.

**Weekend:** Sobering weekend. Did your "great" strategy hold up?

## Week 30 — Regime Classifier

**Day 146 (Mon):** Define regime features: VIX level, VIX term structure (VIX/VXV), credit spreads (HYG/LQD), 2s10s.

**Day 147 (Tue):** Pull historical data for these. Store in DuckDB.

**Day 148 (Wed):** Build a simple Hidden Markov Model or k-means clustering on regime features.

**Day 149 (Thu):** Classify every historical day into a regime. Visualize — does it match known events (2008, 2020, 2022)?

**Day 150 (Fri):** Add regime as a feature to the meta-learner. Re-train. Does it help?

**Weekend:** Rest.

## Week 31 — Ensemble Validation

**Day 151 (Mon):** Full end-to-end backtest of regime-aware ensemble vs baseline momentum.

**Day 152 (Tue):** Walk-forward validation — train 2014–2020, test 2021–2024 with no retraining.

**Day 153 (Wed):** Stress test: performance in COVID, 2022 momentum crash, 2023 SVB period.

**Day 154 (Thu):** Decision point: does the ensemble clearly beat the baseline? If only marginal, ship the baseline. Simpler is safer.

**Day 155 (Fri):** Document decision and rationale.

**Weekend:** Commit. Tag `v1.3-ensemble-validated`.

## Week 32 — Shadow Deployment

**Day 156 (Mon):** Deploy the ensemble as a **shadow** strategy — runs alongside baseline, doesn't place orders, just logs what it *would* do.

**Day 157 (Tue):** First day of shadow running. Verify it matches backtest predictions for that day.

**Day 158 (Wed):** Any mismatch — investigate. Usually a data issue.

**Day 159 (Thu):** Set up a daily comparison: shadow vs live baseline.

**Day 160 (Fri):** End of Month 8. Shadow ensemble running, live baseline running, both observable.

**Weekend:** Rest.

---

# MONTH 9 — Road to Live Money

## Week 33 — Pre-flight Checklist

**Day 161 (Mon):** Review the entire go-live checklist from the main roadmap. Honest assessment of each item.

**Day 162 (Tue):** Write a **runbook** for common incidents: bot down, broker API error, data feed outage, positions don't reconcile.

**Day 163 (Wed):** Document a manual kill-switch procedure: exact commands to flatten all positions.

**Day 164 (Thu):** Test the kill switch on paper. It should work in under 60 seconds.

**Day 165 (Fri):** Write a disaster recovery plan: what if the VM dies? Can you restore within 1 hour?

**Weekend:** Do the disaster recovery drill. Delete the VM, rebuild from scratch. Time yourself.

## Week 34 — Live Account Setup

**Day 166 (Mon):** Open a live Alpaca account. Provide KYC info. Takes a few days.

**Day 167 (Tue):** While waiting — audit your paper results. Are they meeting your pre-flight criteria (60+ days, Sharpe matches backtest)?

**Day 168 (Wed):** Continue auditing. Look for any unexplained trades, any risk rule that never triggered (maybe it should have).

**Day 169 (Thu):** Live account approved. Generate live API keys. Do NOT use them yet.

**Day 170 (Fri):** Decide starting capital: 25% of your intended amount. If target is $4,000, start with $1,000.

**Weekend:** Rest. This is a big transition.

## Week 35 — First Live Trade

**Day 171 (Mon):** ACH transfer the starting capital to Alpaca. Takes 1–3 days.

**Day 172 (Tue):** While funds clear — one last code audit. Look for `paper=True` anywhere that should be configurable.

**Day 173 (Wed):** Funds cleared. Swap live keys into `.env.live` (keep separate from `.env.paper`).

**Day 174 (Thu):** **Go live.** Flip the config. Bot trades real money for the first time. Do not touch it today.

**Day 175 (Fri):** Review Day 1 live carefully. Every fill, every decision. Compare to what paper would have done (run shadow paper alongside).

**Weekend:** Do not stare at the PnL. Go outside.

## Week 36 — Live Stabilization

**Day 176 (Mon):** First full week review. Live vs shadow paper vs backtest — three-way comparison.

**Day 177 (Tue):** Watch for slippage surprises. Real fills will be slightly worse than paper fills. Measure the gap.

**Day 178 (Wed):** If gap is large (>20bps per trade), investigate — liquidity issues, wrong venue, etc.

**Day 179 (Thu):** First circuit-breaker test in production (as a drill, not triggered by losses): verify the kill switch works live.

**Day 180 (Fri):** End of Month 9. Bot live with real money. Scale decision in 30 days based on performance.

**Weekend:** Big milestone. Acknowledge it.

---

# MONTH 10 — Live Operation & Scaling

## Week 37 — First 30-Day Review Prep

**Day 181 (Mon):** Build a live performance attribution notebook. Which signals drove PnL? Which lost money?

**Day 182 (Tue):** Compare realized costs (slippage, spread) to modeled costs. Calibrate models.

**Day 183 (Wed):** Compare realized Sharpe to paper Sharpe. Expect live to be worse; quantify by how much.

**Day 184 (Thu):** Identify any one-off events: bad fills, data outages, unexpected corporate actions.

**Day 185 (Fri):** Document everything learned in first month live.

**Weekend:** Rest.

## Week 38 — First Scaling Decision

**Day 186 (Mon):** Review 30-day live results. Is live Sharpe within 30% of backtest? If yes, scale.

**Day 187 (Tue):** If scaling: ACH another 25% of intended capital. Don't double-down recklessly.

**Day 188 (Wed):** If NOT scaling: document why, create a remediation plan. Pause and fix before adding money.

**Day 189 (Thu):** Whatever you decide, update your runbook with lessons learned.

**Day 190 (Fri):** Buffer. Live operation continues.

**Weekend:** Rest.

## Week 39 — Research Pipeline

**Day 191 (Mon):** Now that live is stable, return to research. Set up a weekly research cadence.

**Day 192 (Tue):** Pick one new signal idea from your backlog. Hypothesis first, data second.

**Day 193 (Wed):** Quick sanity backtest on the idea. If it doesn't beat random, kill it now.

**Day 194 (Thu):** If promising: plan a rigorous test with proper CV.

**Day 195 (Fri):** Establish your "research Friday" habit — every Friday is for new research.

**Weekend:** Rest.

## Week 40 — Signal Decay Monitoring

**Day 196 (Mon):** Build signal-decay monitoring: is each signal's IC (information coefficient) still positive in recent months?

**Day 197 (Tue):** Alert if any signal's rolling 60-day IC goes negative.

**Day 198 (Wed):** Define a decommission rule: if a signal's IC is <0 for 90 consecutive days, pull it from the ensemble.

**Day 199 (Thu):** Implement the decommission logic. Dry-run on historical data to see if the rule fires reasonably.

**Day 200 (Fri):** End of Month 10. Day 200. You have a live, monitored, evolving system.

**Weekend:** Celebrate. Seriously.

---

# MONTH 11 — Robustness & Second Scaling

## Week 41 — Edge Cases

**Day 201 (Mon):** Handle scheduled events: Fed days, CPI days, FOMC minutes. Do you want to trade through these or flatten before?

**Day 202 (Tue):** Implement earnings calendar awareness: flag positions with earnings in next 5 days.

**Day 203 (Wed):** Decide policy: reduce size into earnings? Exit before? Document and implement.

**Day 204 (Thu):** Handle dividends and ex-div dates properly in the live bot.

**Day 205 (Fri):** Handle stock splits manually when they occur. Set up an alert.

**Weekend:** Rest.

## Week 42 — Broker Redundancy

**Day 206 (Mon):** Research a backup broker (IBKR paper, primarily).

**Day 207 (Tue):** Implement `IBKRBroker` adapter using `ib_insync`.

**Day 208 (Wed):** Verify it implements the full Broker interface.

**Day 209 (Thu):** Make broker selection configurable — primary Alpaca, fallback IBKR.

**Day 210 (Fri):** Test failover manually.

**Weekend:** Rest.

## Week 43 — Second Scaling Decision

**Day 211 (Mon):** Review 60-day live results. Sharpe, drawdown, cost calibration.

**Day 212 (Tue):** If performance is solid: scale to 75% of intended capital.

**Day 213 (Wed):** Update risk parameters for new capital level (position caps in dollar terms, not just percentages).

**Day 214 (Thu):** Re-verify all risk rules at new scale. Bigger positions = bigger market impact.

**Day 215 (Fri):** Buffer.

**Weekend:** Rest.

## Week 44 — Stress Test Simulation

**Day 216 (Mon):** Design a simulated market shock: -5% overnight gap, 2x normal volatility.

**Day 217 (Tue):** Run the bot against this shock scenario in backtest mode.

**Day 218 (Wed):** Verify the risk engine behaves correctly. Circuit breakers fire as expected.

**Day 219 (Thu):** Run a liquidity crisis scenario: widen all bid-asks by 3x.

**Day 220 (Fri):** End of Month 11. Document stress test results.

**Weekend:** Rest.

---

# MONTH 12 — Maturity & What's Next

## Week 45 — Performance Attribution

**Day 221 (Mon):** Build a complete factor-attribution report: how much PnL came from market beta, sector bets, idiosyncratic alpha?

**Day 222 (Tue):** If you're getting paid mostly for market beta, that's a problem — you could just hold SPY.

**Day 223 (Wed):** Run Fama-French regression on your daily returns. Check your factor loadings.

**Day 224 (Thu):** Adjust strategy if you're accidentally concentrated in one factor (usually momentum or size).

**Day 225 (Fri):** Document true alpha (after factor neutralization).

**Weekend:** Reflect honestly. How much is real edge vs luck?

## Week 46 — Full Capital Deployment

**Day 226 (Mon):** If 90-day live performance justifies: scale to 100% of intended capital.

**Day 227 (Tue):** Final risk parameter review at full scale.

**Day 228 (Wed):** Observation period — do not touch the bot this week.

**Day 229 (Thu):** Daily reviews continue, but no changes.

**Day 230 (Fri):** Buffer.

**Weekend:** Rest.

## Week 47 — Documentation & Knowledge Base

**Day 231 (Mon):** Full system architecture doc. Diagram every component.

**Day 232 (Tue):** Runbook expansion: every scenario you've handled, documented.

**Day 233 (Wed):** Strategy doc: every signal, every parameter, every rationale.

**Day 234 (Thu):** Research log consolidation: every idea tested, every outcome.

**Day 235 (Fri):** This documentation is your moat against your own forgetfulness.

**Weekend:** Rest.

## Week 48 — Year-End Review

**Day 236 (Mon):** Compile full 12-month review: what worked, what didn't, what surprised you.

**Day 237 (Tue):** Compare actual outcomes to the original plan. Where did reality diverge?

**Day 238 (Wed):** Financial review: net P&L after all costs. Honest Sharpe.

**Day 239 (Thu):** Code quality review: any technical debt that's now dangerous?

**Day 240 (Fri):** Plan for Year 2: new signals? Options? More capital? Business layer?

**Weekend:** Long break. You've earned it.

---

# MONTHS 13+ (Optional): Business Layer

These weeks are optional and only relevant if live results justify it. Do NOT start the business layer without 12+ months of audited live track record.

## Week 49 — Legal Foundations (if pursuing RIA)

**Day 241 (Mon):** Consult an RIA attorney (Cole-Frieman, or a regional boutique). Budget $10–30k for formation.

**Day 242 (Tue):** Form the LLC.

**Day 243 (Wed):** Begin ADV Part 1 and 2 drafting.

**Day 244 (Thu):** Set up compliance infrastructure (ComplySci or similar).

**Day 245 (Fri):** Timeline: 3–6 months to registered.

## Weeks 50–52 — Track Record Audit & First External Capital

**Day 246–260:** Engage a performance audit firm. GIPS-compliant verification. Expect ~$20k and 2–3 months.

**Day 261+:** With audited track record and RIA registered, begin taking friends-and-family capital. Small ($50–200k total). Test operational workflows at new scale.

**End of Year 1 / Start of Year 2:** If everything holds together, you have a real, legal, defensible trading business.

---

# Reality Check Boxes

After each quarter, honestly answer these. If the answer to any of them is "no", stop and reassess before proceeding.

**End of Month 3:**
- Does my baseline strategy have out-of-sample Sharpe within 30% of in-sample? ______
- Do I understand every line of code I've written? ______
- Am I tracking a daily dev log? ______

**End of Month 6:**
- Is my paper bot running without unexplained crashes for 30+ days? ______
- Do paper results match backtest within tolerance? ______
- Have I read at least 1 López de Prado chapter this month? ______

**End of Month 9:**
- Am I ready to put real money on this honestly? ______
- Do I have a tested kill switch? ______
- Have I tested disaster recovery? ______

**End of Month 12:**
- Is my live Sharpe positive after all costs? ______
- Am I generating alpha or just beta? ______
- Would I do this again, knowing what I know now? ______

---

*Day 1 starts Monday. When you've finished Day 5 and the Alpaca hello-world script runs, message me and I'll help you scaffold the production broker wrapper as your first real piece of production code.*