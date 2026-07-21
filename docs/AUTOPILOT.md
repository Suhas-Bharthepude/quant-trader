# AUTOPILOT - operator runbook

Concise guide to running the rotation bot unattended.

## 1. What runs

`scripts/run_daily.py` is the entry point. It is idempotent (safe to run every
market day; it acts at most once per new completed month-end) and exits 0 on a
clean run - including a no-op - and 1 on any failure.

## 2. Daily order of operations (critical precondition)

The scheduler MUST run the ingest BEFORE the runner each day, or the runner
refuses to trade on stale bars (exit 1). In order:

    uv run python scripts/update_universe.py
    uv run python scripts/run_daily.py

Why: `run_daily` loads bars from DuckDB and gates on freshness (`stale_symbols`);
`update_universe.py` is what makes today's bars current in DuckDB first.

## 3. Cron example

    # weekdays 21:30 UTC (after US close): ingest then run the bot
    30 21 * * 1-5 cd /path/to/quant-trader && uv run python scripts/update_universe.py && uv run python scripts/run_daily.py >> logs/cron.log 2>&1

The time is the HOST's timezone - pick a time comfortably after 16:00 ET (21:30
UTC is well after close). The `&&` chaining means a failed ingest aborts before
trading, so the runner never sees stale data from a broken ingest.

## 4. Systemd alternative (brief)

For a host that is not reliably awake at the cron minute, a systemd timer is the
equivalent: a `.timer` + `.service` pair calling the same two commands in order.
Do not hand-roll a unit file here; just note the `.service` must set
`WorkingDirectory` to the repo directory and load the `.env` (EnvironmentFile=)
so `from_env()` finds the keys.

## 5. Secrets

`.env` holds `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` (paper keys). It is
git-ignored and must be present in the repo directory (or the vars exported into
the host environment) for `from_env()` to work. Never commit `.env`.

## 6. Exit codes + monitoring

    0 = ran cleanly (acted or no-op) - nothing to do.
    1 = failure to investigate: bad/missing keys, stale bars, or a failed rebalance.

The notifier is currently logging-only, writing to `logs/daily_runner.log`, and
fires on the `acted` and `failed` outcomes (routine no-ops are logged but not
alerted). Monitor exit code 1 and the ERROR lines in that log. A real
Slack/email notifier can later be wired into `run_daily`'s `notify_fn` seam via
stdlib `urllib` with no new dependency.

## 7. Honest note

This runs the validated rotation config (lookback=3, top_n=1), whose
out-of-sample Sharpe (+0.44) LOSES to buy-and-hold (+0.59) after costs. The bot
is a demonstration of a complete, rigorously-validated research-to-execution
pipeline, not a profitable strategy.
