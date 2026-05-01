# DEV_LOG — quant-trader

Running log of development work. Most recent entry first.

---

## 2026-05-01 — Day 10
**Worked on:** Started Phase 1 (data infrastructure). Created `src/data/schema.py` with frozen `OHLCVBar` dataclass and `CREATE_TABLE_SQL`. Built `src/data/yfinance_fetcher.py` — thin adapter around `yfinance.Ticker.history` that returns `list[OHLCVBar]` sorted ascending. Built `src/data/duckdb_store.py` with `INSERT OR IGNORE`-based `write_bars` (idempotent), `read_bars` range query, and context-manager support. Added `tests/test_data_layer.py` with three integration tests covering fetcher output, roundtrip persistence, and idempotency. Verified end-to-end: fetched 252 SPY daily bars for 2024, wrote to DuckDB, re-ran the pipeline and confirmed 0 duplicate rows on second run.
**Why it matters:** A trading bot is only as good as its data. Without a local historical price store, every strategy run would need to re-fetch from the internet — slow, rate-limited, and fragile. The DuckDB layer gives the bot a fast local cache of OHLCV history that strategies and backtests can query in milliseconds. The idempotent write design means a daily ingest job can run on a schedule without ever producing duplicate data, which would corrupt signal calculations.
**Blocked on / Bugs:** None.
**Next up:** Day 11 — extend fetcher to support batch fetching across the S&P 500, add a "universe" config file listing tickers, build a backfill script that ingests N years of history for the full universe.
**Time spent:** —

---

## 2026-04-29 — Day 9
**Worked on:** Refactored all four operational scripts (`first_order.py`, `limit_order.py`, `cancel_order.py`, `order_history.py`) to use the `AlpacaBroker` abstraction. Removed every `from alpaca.*` import from `scripts/` — verified with `grep -rn "from alpaca" scripts/` returning empty. Added `Broker.get_latest_price()` abstract method and Alpaca implementation to eliminate the temporary `_data` attribute leak — `grep -rn "broker._data" scripts/` also returns empty. Status comparisons now use plain strings (`"filled"`, `"canceled"`) instead of alpaca's `OrderStatus` enum. Combined script line count dropped roughly 50% across the four files. Confirmed all four scripts produce identical user-facing output. Smoke test still passes.
**Why it matters:** This validated that the `Broker` abstraction is complete and usable — if the scripts could be fully migrated without leaking any Alpaca-specific code, the interface is the right shape. Any future strategy or execution engine can now be written entirely against `src/brokers/base.py`, which means switching from Alpaca to a different broker (Interactive Brokers, Tradier, a sim broker for backtesting) is a single-file change with zero impact on strategy code.
**Blocked on / Bugs:** None.
**Next up:** Day 10 — start Phase 1 of the roadmap. Build `src/data/` layer: yfinance + Alpaca historical data fetcher, DuckDB schema, basic OHLCV storage and retrieval.
**Time spent:** — 1 hour

---

## 2026-04-27 — Day 8
**Worked on:** Built the broker abstraction layer. Created `src/brokers/base.py` with the `Broker` ABC plus typed dataclasses (`OrderRequest`, `OrderResult`, `AccountSnapshot`) and string Enums (`OrderSide`, `OrderType`, `TimeInForce`). Built `src/brokers/alpaca_broker.py` implementing the full interface against alpaca-py, with a hard assertion preventing `paper=False` and a `from_env()` classmethod. Added `tests/test_alpaca_broker.py` as a read-only integration smoke test — 1 passed. Added `pythonpath` and `integration` marker to `pyproject.toml`, created `conftest.py` at repo root for reliable imports. Refactored `scripts/hello_alpaca.py` to use `AlpacaBroker` — script shrank from 33 lines to 29 lines, all SDK code now behind the abstraction.
**Why it matters:** This is the architectural keystone of the whole bot. Without a broker abstraction, every strategy would be hardwired to Alpaca's SDK — untestable without a live connection and impossible to migrate. The `Broker` interface means strategies express intent ("buy 10 shares of SPY") and the broker layer handles the translation. It also makes it possible to build a `PaperBroker` later that simulates fills locally, enabling realistic backtesting without touching a real exchange.
**Blocked on / Bugs:** None.
**Next up:** Day 9 — refactor `first_order.py`, `limit_order.py`, `cancel_order.py`, and `order_history.py` to use `AlpacaBroker`.
**Time spent:** — 30 mins

---

## 2026-04-26 — Day 7
**Worked on:** Built `scripts/limit_order.py` (submits an intentionally non-marketable limit BUY 20% below market, GTC, with a paranoia assertion preventing accidental marketable orders; saves order ID to `logs/last_limit_order_id.txt` for handoff). Built `scripts/cancel_order.py` (cancels by ID from CLI arg or from the handoff file, polls for terminal state up to 10 s). Built `scripts/order_history.py` (fetches last 50 orders across all statuses, prints fixed-width table with fill price and limit price columns, plus a grouped summary line). Verified full lifecycle end-to-end: limit order placed at $571.17, cancelled successfully, order history table confirmed `ACCEPTED: 1 | CANCELED: 1`.
**Why it matters:** Real trading strategies almost never use market orders — they use limit orders to control entry price, and they need to cancel orders when market conditions change. This day proved the bot can manage the full order lifecycle: place, monitor, and cancel. The order history script is also the first observability tool — without it there's no way to audit what the bot actually did or diagnose why a fill didn't happen.
**Blocked on / Bugs:** None.
**Next up:** Day 8 — design the broker abstraction layer in `src/brokers/`. Refactor shared credential loading and paper-account safety check out of scripts into a reusable module.
**Time spent:** — 30 mins

---

## 2026-04-25 — Day 6
**Worked on:** Built `scripts/first_order.py` with three safety checks: `paper=True` hardcoded (must never be changed), PA account prefix verification before any order is submitted, and hardcoded `SYMBOL`/`QTY` constants. Added dry-run preview with live SPY price fetch via `StockHistoricalDataClient`, market-hours detection with next-open timestamp, and double confirmation (`"yes"` typed explicitly) before submitting. Submitted first paper order — 1 share of SPY at $713.96, order ID `c5559e4b-bf39-42b4-9658-1e9da0639004`, status ACCEPTED, queued for Monday open. Verified order appeared correctly in Alpaca dashboard. Rotated API keys after accidental exposure in screenshot.
**Why it matters:** This was the proof of concept that the entire stack works end-to-end — credentials, connection, safety checks, price fetch, order submission, and exchange acknowledgement all in one flow. The safety-first design (hardcoded paper mode, account prefix check, explicit confirmation) establishes the pattern that every future execution path must follow. A single accidental live order could lose real money; the layered guards here make that nearly impossible.
**Blocked on / Bugs:** API key accidentally visible in a dashboard screenshot — immediately regenerated keys and updated `.env`.
**Next up:** Day 7 — limit orders, order cancellation, querying order history.
**Time spent:** — 1 hour 

---

## 2026-04-25 — Days 1–5
**Worked on:** Project initialization — pyproject.toml, uv-managed virtual environment, folder scaffold (src/, tests/, notebooks/, scripts/, data/, logs/), .gitignore, .env.example, README.md. Alpaca paper account created and connected; $200k buying power confirmed. GitHub repo initialized and remote connected.
**Why it matters:** A trading bot running against real (or paper) money needs disciplined hygiene from day one — dependency isolation via a virtual environment prevents version conflicts later, a clean folder structure keeps strategy code separate from scripts and data, and `.env`-based credentials mean API keys never touch the codebase. Setting this up correctly at the start avoids painful retrofits when the project grows.
**Blocked on / Bugs:** None.
**Next up:** First commit. Begin Phase 1 — data ingestion layer (Alpaca bar fetch, DuckDB schema, yfinance backfill).
**Time spent:** — 30 minutes

