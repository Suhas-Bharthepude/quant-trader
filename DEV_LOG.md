# DEV_LOG — quant-trader

Running log of development work. Most recent entry first.

---

## Day 11 — 2026-05-04

**Worked on:** Scaled the data layer to the S&P 500. Created config/universe.yaml with named universes ("test" with 5 tickers, "sp500" auto-populated). Built src/data/universe.py with load_universe() and list_universes() helpers. Built scripts/refresh_sp500_universe.py — scrapes Wikipedia constituents via pandas.read_html (with browser User-Agent header to bypass 403 and io.StringIO wrapping to avoid pandas printing HTML to stdout), normalizes ticker dots to dashes for Yahoo (BRK.B → BRK-B), updates the YAML. Added YFinanceFetcher.fetch_daily_batch() with on_error=skip|raise — partial failures don't kill the run. Built scripts/backfill_universe.py — argparse CLI taking --universe and --years, with tqdm progress bar and per-symbol error isolation. Tested end-to-end: backfilled 5 years of S&P 500 daily bars (~626k rows, 503/503 symbols, ~20 minutes). Verified idempotency at scale — re-running the test backfill produces 0 duplicate inserts. Added 3 unit tests for the universe loader (pure, no network).

**Why it matters:** The bot now has 5 years of daily price history for every S&P 500 stock stored locally — all 500+ companies, downloaded once and ready to query instantly. Before this, any strategy test would have to reach out to the internet every single time, which is slow and breaks if the data provider is unavailable. Now the bot can test a trading idea across hundreds of stocks in seconds, using data that's already on disk.

**Blocked on / Bugs:** Two pandas-related issues fixed: (1) Wikipedia returned 403 with default urllib User-Agent — fixed by passing a browser UA via urllib.request.Request. (2) pd.read_html(html_string) was printing the raw HTML to stdout as a side effect — fixed by wrapping in io.StringIO. Also needed to add lxml dependency (pandas.read_html requires it).

**Next up:** Day 12 — daily incremental update script (only fetch the latest N days, not full history), gap detection (find missing dates per symbol), and a "data health" diagnostic script.

**Time spent:** —

---


## 2026-05-01 — Day 10
**Worked on:** Started Phase 1 (data infrastructure). Created `src/data/schema.py` with frozen `OHLCVBar` dataclass and `CREATE_TABLE_SQL`. Built `src/data/yfinance_fetcher.py` — thin adapter around `yfinance.Ticker.history` that returns `list[OHLCVBar]` sorted ascending. Built `src/data/duckdb_store.py` with `INSERT OR IGNORE`-based `write_bars` (idempotent), `read_bars` range query, and context-manager support. Added `tests/test_data_layer.py` with three integration tests covering fetcher output, roundtrip persistence, and idempotency. Verified end-to-end: fetched 252 SPY daily bars for 2024, wrote to DuckDB, re-ran the pipeline and confirmed 0 duplicate rows on second run.
**Why it matters:** Before this, the bot had no memory of what prices did in the past. This day gave it a database — a local file that stores the open, high, low, close, and volume for any stock, for any day. Think of it as building the library the bot will read before making any decision. The design is also "safe to re-run": if the daily download job runs twice by accident, it won't create duplicate entries or corrupt the numbers.
**Blocked on / Bugs:** None.
**Next up:** Day 11 — extend fetcher to support batch fetching across the S&P 500, add a "universe" config file listing tickers, build a backfill script that ingests N years of history for the full universe.
**Time spent:** —

---

## 2026-04-29 — Day 9
**Worked on:** Refactored all four operational scripts (`first_order.py`, `limit_order.py`, `cancel_order.py`, `order_history.py`) to use the `AlpacaBroker` abstraction. Removed every `from alpaca.*` import from `scripts/` — verified with `grep -rn "from alpaca" scripts/` returning empty. Added `Broker.get_latest_price()` abstract method and Alpaca implementation to eliminate the temporary `_data` attribute leak — `grep -rn "broker._data" scripts/` also returns empty. Status comparisons now use plain strings (`"filled"`, `"canceled"`) instead of alpaca's `OrderStatus` enum. Combined script line count dropped roughly 50% across the four files. Confirmed all four scripts produce identical user-facing output. Smoke test still passes.
**Why it matters:** This confirmed that none of the trading scripts need to know they're using Alpaca anymore — they just say "place this order" and the broker layer handles the rest. That means if we ever want to switch to a different broker, only one file changes and every strategy works without modification. It also means we can plug in a fake "paper broker" for testing strategies without any internet connection at all.
**Blocked on / Bugs:** None.
**Next up:** Day 10 — start Phase 1 of the roadmap. Build `src/data/` layer: yfinance + Alpaca historical data fetcher, DuckDB schema, basic OHLCV storage and retrieval.
**Time spent:** — 1 hour

---

## 2026-04-27 — Day 8
**Worked on:** Built the broker abstraction layer. Created `src/brokers/base.py` with the `Broker` ABC plus typed dataclasses (`OrderRequest`, `OrderResult`, `AccountSnapshot`) and string Enums (`OrderSide`, `OrderType`, `TimeInForce`). Built `src/brokers/alpaca_broker.py` implementing the full interface against alpaca-py, with a hard assertion preventing `paper=False` and a `from_env()` classmethod. Added `tests/test_alpaca_broker.py` as a read-only integration smoke test — 1 passed. Added `pythonpath` and `integration` marker to `pyproject.toml`, created `conftest.py` at repo root for reliable imports. Refactored `scripts/hello_alpaca.py` to use `AlpacaBroker` — script shrank from 33 lines to 29 lines, all SDK code now behind the abstraction.
**Why it matters:** This is the foundation that everything else builds on. The bot now speaks a common language for placing orders — it says "buy 10 shares of SPY" and doesn't care how that gets executed. Alpaca is just one possible answer. This makes the bot portable: swap in a different broker, a simulator, or a backtester, and the strategies don't change at all. Without this layer, every strategy would be glued to Alpaca's specific code and impossible to test offline.
**Blocked on / Bugs:** None.
**Next up:** Day 9 — refactor `first_order.py`, `limit_order.py`, `cancel_order.py`, and `order_history.py` to use `AlpacaBroker`.
**Time spent:** — 30 mins

---

## 2026-04-26 — Day 7
**Worked on:** Built `scripts/limit_order.py` (submits an intentionally non-marketable limit BUY 20% below market, GTC, with a paranoia assertion preventing accidental marketable orders; saves order ID to `logs/last_limit_order_id.txt` for handoff). Built `scripts/cancel_order.py` (cancels by ID from CLI arg or from the handoff file, polls for terminal state up to 10 s). Built `scripts/order_history.py` (fetches last 50 orders across all statuses, prints fixed-width table with fill price and limit price columns, plus a grouped summary line). Verified full lifecycle end-to-end: limit order placed at $571.17, cancelled successfully, order history table confirmed `ACCEPTED: 1 | CANCELED: 1`.
**Why it matters:** Market orders just buy at whatever price is available — limit orders let the bot say "only buy if the price drops to X." That's essential for any real strategy. This day also added the ability to cancel an order that hasn't filled yet, and to look up a full history of what the bot has done. Without that history, there's no way to know if a trade actually went through or why it didn't.
**Blocked on / Bugs:** None.
**Next up:** Day 8 — design the broker abstraction layer in `src/brokers/`. Refactor shared credential loading and paper-account safety check out of scripts into a reusable module.
**Time spent:** — 30 mins

---

## 2026-04-25 — Day 6
**Worked on:** Built `scripts/first_order.py` with three safety checks: `paper=True` hardcoded (must never be changed), PA account prefix verification before any order is submitted, and hardcoded `SYMBOL`/`QTY` constants. Added dry-run preview with live SPY price fetch via `StockHistoricalDataClient`, market-hours detection with next-open timestamp, and double confirmation (`"yes"` typed explicitly) before submitting. Submitted first paper order — 1 share of SPY at $713.96, order ID `c5559e4b-bf39-42b4-9658-1e9da0639004`, status ACCEPTED, queued for Monday open. Verified order appeared correctly in Alpaca dashboard. Rotated API keys after accidental exposure in screenshot.
**Why it matters:** This was the first proof that the bot can actually do the one thing it exists to do — place a trade. Everything before this was setup; this was the moment it became real. The multiple safety checks (paper-mode lock, account verification, typing "yes" twice) are deliberate: one mistaken click in a live trading system can cost real money instantly, so the guards have to be there from the very first order.
**Blocked on / Bugs:** API key accidentally visible in a dashboard screenshot — immediately regenerated keys and updated `.env`.
**Next up:** Day 7 — limit orders, order cancellation, querying order history.
**Time spent:** — 1 hour 

---

## 2026-04-25 — Days 1–5
**Worked on:** Project initialization — pyproject.toml, uv-managed virtual environment, folder scaffold (src/, tests/, notebooks/, scripts/, data/, logs/), .gitignore, .env.example, README.md. Alpaca paper account created and connected; $200k buying power confirmed. GitHub repo initialized and remote connected.
**Why it matters:** Getting the foundation right means not having to redo it later. Keeping API keys out of the code means they can never accidentally end up on GitHub. Separating strategy code from scripts and data means the project stays navigable as it grows. These decisions feel invisible when they're done right — and extremely painful to fix after the fact when they're skipped.
**Blocked on / Bugs:** None.
**Next up:** First commit. Begin Phase 1 — data ingestion layer (Alpaca bar fetch, DuckDB schema, yfinance backfill).
**Time spent:** — 30 minutes

