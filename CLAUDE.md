# CLAUDE.md — quant-trader

Portfolio-grade algorithmic swing-trading research system. Rules-based strategies validated with walk-forward + Optuna. The deliverable is engineering rigor and honest out-of-sample results, not a profitable backtest.

## Environment
- Python 3.13.7, managed by `uv`. ALWAYS invoke via `uv run` (`uv run python`, `uv run pytest`). Never bare `python` or `pytest` — the .venv requires it.

## Strategy contract (every strategy matches this)
- Signature: takes `list[OHLCVBar]`, returns `np.ndarray` of `int8`, same length as the input.
- Values: only the signal constants — `SIGNAL_LONG` (1), `SIGNAL_FLAT` (0), `SIGNAL_SHORT` (-1). A long/flat strategy emits only 1 and 0, never -1.
- Warmup: emit `SIGNAL_FLAT` (0) for bars without enough history. NOT NaN — an int8 array cannot hold NaN. (NaN belongs to indicator arrays, not position arrays.)
- Purity: no I/O, no broker calls, no global state. A strategy maps bars → positions, nothing else.
- NO internal lag or shift. The backtester applies the only lag (`signals[:-1] * returns[1:]`); a strategy emits, at each bar, the position implied by data through that bar. A strategy-side shift double-lags.

## Backtester invariant
- Next-bar execution: a signal at bar i earns the return from i to i+1. Lookahead bias is structurally impossible and must stay that way. `test_no_lookahead_bias` is the load-bearing test — never weaken it to make something pass.

## Verification (non-negotiable)
- Never trust editor buffers or prose summaries (including your own) over disk. Verify with `git log`, `git diff`, `grep`, and `uv run pytest`. State only what you have verified on disk; if you didn't check it, say so.

## Git discipline
- Two commits per feature: the code commit first, then a SEPARATE DEV_LOG commit. Write the DEV_LOG entry before committing it — `git add DEV_LOG.md && commit` with no written entry is a no-op.
- DEV_LOG.md is reverse-chronological (newest on top), fixed sections: Worked on / Why it matters / Architectural note / Blocked on / Next up / Time spent.
- Stage files explicitly by name (`git add <file>`). NEVER `git add .`, `git add -A`, or the VS Code commit-all button.
- `trading_explained.md` and `daily_prompt.md` are intentionally UNTRACKED. Never stage them.

## Scope
- Edit only the files the task names. Do not modify the backtester, the walk-forward validator, or the Optuna fitter unless the task explicitly says to. If a change seems to require touching them, stop and flag it.

## Data
- Daily bars are floored to midnight UTC — one row per trading day. Sanity-check bar counts against trading-day expectations.
- Universe: fixed 17-ETF basket, backfilled to 2008.

## Research ethos
- A negative result, delivered honestly, is the product. Do not optimize toward a profit target — that is what drives overfitting. Model transaction costs before declaring any verdict.