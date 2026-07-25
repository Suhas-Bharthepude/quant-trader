# AUTOPILOT - operator runbook

Concise guide to running the rotation bot unattended.

## 1. What runs

`scripts/run_daily.py` is the entry point. It is idempotent (safe to run every
market day; it acts at most once per new completed month-end) and exits 0 on a
clean run - including a no-op - and 1 on any failure.

## 2. Daily order of operations (critical precondition)

The scheduler MUST run the ingest BEFORE the runner each day, or the runner
refuses to trade on stale bars (exit 1). In order:

    uv run python scripts/update_universe.py --universe etf_basket
    uv run python scripts/run_daily.py

Why: `run_daily` loads bars from DuckDB and gates on freshness (`stale_symbols`);
`update_universe.py` is what makes today's bars current in DuckDB first.

## 3. Cron example

This cron line is the PORTABLE alternative - for a Linux box or any always-on
host. The CHOSEN local deployment on this machine is launchd (see "Local
scheduling on macOS (launchd)" below); cron is documented here for other hosts.
The pairing/ordering rule it encodes - ingest with `--universe etf_basket`
BEFORE `run_daily` - applies identically to both.

    # weekdays 21:30 UTC (after US close): ingest then run the bot
    30 21 * * 1-5 cd /path/to/quant-trader && uv run python scripts/update_universe.py --universe etf_basket && uv run python scripts/run_daily.py >> logs/cron.log 2>&1

The time is the HOST's timezone - pick a time comfortably after 16:00 ET (21:30
UTC is well after close). The `&&` chaining means a failed ingest aborts before
trading, so the runner never sees stale data from a broken ingest.

## Local scheduling on macOS (launchd)

This is the settled local deployment for this machine: a macOS LaunchAgent fires
the daily run on weekday afternoons. Best-effort local scheduling is a deliberate
design choice here, not a limitation being worked around (see SCOPE below).

What runs: the committed wrapper `scripts/run_daily_autopilot.sh`, invoked by a
LaunchAgent with label `local.quant-trader.daily`. The wrapper first cd's to the
repo root so every CWD-relative dependency resolves - `.env` (read by
`from_env()`), the `data/` DuckDB, and `logs/` including the idempotency state
file `logs/rebalance_state.json` - then runs the ingest with
`--universe etf_basket` and ONLY then runs `run_daily`. It propagates the real
exit codes and appends stdout and stderr to a dated log,
`logs/autopilot_YYYY-MM-DD.log`.

Operating assumption: the machine is open and awake at the scheduled fire time.
In that case the job runs on time regardless of whether it is plugged in. The
`pmset` wake and launchd's missed-run catch-up (both below) are secondary safety
nets, not the primary reliance.

Why launchd and not cron: launchd re-fires a job that was missed while the
machine slept, on the next wake; cron silently drops missed runs. On a laptop
that sleeps, that catch-up behaviour is the whole point.

The bare-environment gotcha: launchd does NOT load your shell PATH, so `uv`
cannot be found by name. It must be reached by absolute path. Both `PATH` and
`UV_BIN` are set in the plist's `EnvironmentVariables`, and the wrapper reads
`UV_BIN`. This is the single most common silent launchd failure - if the job
"runs" but does nothing, check this first.

Setup, from the committed template
`deploy/local.quant-trader.daily.plist.template`:

    1. Copy the template to ~/Library/LaunchAgents/local.quant-trader.daily.plist
    2. Replace __REPO_ROOT__, __UV_BIN__, and __UV_BIN_DIR__ with real values
       (from `pwd`, `which uv`, and `dirname "$(which uv)"`).
    3. Ensure logs/ exists FIRST: `mkdir -p <repo>/logs`. launchd opens its
       StandardOutPath / StandardErrorPath at load time, before the wrapper's own
       `mkdir -p logs` can run.
    4. Validate: `plutil -lint ~/Library/LaunchAgents/local.quant-trader.daily.plist`
    5. Bootstrap: `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.quant-trader.daily.plist`

The real `~/Library/LaunchAgents` plist is machine-specific and is NOT committed -
only the template lives in the repo. Never commit the filled-in version.

Fire time: 18:00 Eastern on weekdays, comfortably after the 16:00 ET close so
yfinance's daily bar has settled. `RunAtLoad` is false, so logging in does not
trigger a trade. On-demand testing uses
`launchctl kickstart -k gui/$(id -u)/local.quant-trader.daily`, which has been
verified to reach a decision and exit 0 - a market-closed run is a clean no-op
with no email.

pmset: `sudo pmset repeat wakeorpoweron MTWRF 17:58:00` wakes the Air a couple of
minutes before the job when it is plugged in and left closed.

### SCOPE

This is best-effort local scheduling by deliberate choice. The job runs on time
whenever the machine is awake, and launchd catches up a run missed during sleep
on the next wake. A machine that is fully off, or on battery with the lid closed
(for example in a bag while traveling), may miss a day entirely. That is an
ACCEPTED outcome for a swing system holding positions for days at a time, where a
late or occasionally skipped rebalance decision costs essentially nothing.
Market-closed days are a clean no-op (exit 0); an ingest failure aborts
`run_daily` and surfaces as a visible nonzero exit in the log. If on-time daily
execution ever proves to matter, the path is an always-on host - an always-on Mac
mini, a small Linux VM, or GitHub Actions - and that is explicitly not needed for
the current use case.

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
