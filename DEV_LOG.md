# DEV_LOG — quant-trader

Running log of development work. Most recent entry first.

---

## 2026-04-25 — Day 7
**Worked on:** Built `scripts/limit_order.py` (submits an intentionally non-marketable limit BUY 20% below market, GTC, with a paranoia assertion preventing accidental marketable orders; saves order ID to `logs/last_limit_order_id.txt` for handoff). Built `scripts/cancel_order.py` (cancels by ID from CLI arg or from the handoff file, polls for terminal state up to 10 s). Built `scripts/order_history.py` (fetches last 50 orders across all statuses, prints fixed-width table with fill price and limit price columns, plus a grouped summary line). Verified full lifecycle end-to-end: limit order placed at $571.17, cancelled successfully, order history table confirmed `ACCEPTED: 1 | CANCELED: 1`.
**Blocked on / Bugs:** None.
**Next up:** Day 8 — design the broker abstraction layer in `src/brokers/`. Refactor shared credential loading and paper-account safety check out of scripts into a reusable module.
**Time spent:** — 30 mins

---

## 2026-04-25 — Day 6
**Worked on:** Built `scripts/first_order.py` with three safety checks: `paper=True` hardcoded (must never be changed), PA account prefix verification before any order is submitted, and hardcoded `SYMBOL`/`QTY` constants. Added dry-run preview with live SPY price fetch via `StockHistoricalDataClient`, market-hours detection with next-open timestamp, and double confirmation (`"yes"` typed explicitly) before submitting. Submitted first paper order — 1 share of SPY at $713.96, order ID `c5559e4b-bf39-42b4-9658-1e9da0639004`, status ACCEPTED, queued for Monday open. Verified order appeared correctly in Alpaca dashboard. Rotated API keys after accidental exposure in screenshot.
**Blocked on / Bugs:** API key accidentally visible in a dashboard screenshot — immediately regenerated keys and updated `.env`.
**Next up:** Day 7 — limit orders, order cancellation, querying order history.
**Time spent:** — 1 hour 

---

## 2026-04-25 — Day 1
**Worked on:** Project initialization — pyproject.toml, uv-managed virtual environment, folder scaffold (src/, tests/, notebooks/, scripts/, data/, logs/), .gitignore, .env.example, README.md. Alpaca paper account created and connected; $200k buying power confirmed. GitHub repo initialized and remote connected.
**Blocked on / Bugs:** None.
**Next up:** First commit. Begin Phase 1 — data ingestion layer (Alpaca bar fetch, DuckDB schema, yfinance backfill).
**Time spent:** — 30 minutes

