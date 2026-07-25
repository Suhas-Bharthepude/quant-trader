#!/bin/bash
# scripts/run_daily_autopilot.sh
#
# launchd-invoked wrapper for the daily autopilot: ingest today's bars, then
# run the daily rebalance. Written to survive a bare launchd environment (no
# shell PATH, arbitrary CWD) by resolving the repo root from BASH_SOURCE and
# cd-ing there before touching any CWD-relative path.

# set -u: treat an unset variable as an error (catches typos in var names).
set -u
# set -o pipefail: a pipeline's exit status is the last non-zero command's, so a
# failing command feeding a pipe is not masked by a succeeding tail.
set -o pipefail
# NOTE: deliberately NOT set -e. We want to LOG a failing ingest and then exit
# nonzero on OUR terms (controlled below), not have the script die silently the
# instant a command returns nonzero.

# Resolve this script's own directory, following through however launchd calls
# us. dirname of BASH_SOURCE gives the scripts/ dir; cd + pwd makes it absolute.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The repo root is the parent of scripts/. Absolute, CWD-independent.
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# cd to the repo root. This is MANDATORY: .env (load_dotenv, alpaca_broker.py:143),
# data/quant_trader.duckdb (duckdb_store.py:63 default), logs/ and
# logs/rebalance_state.json (daily_runner.py:269 idempotency state) are ALL
# CWD-relative with no __file__ anchor; and `uv run` itself needs the repo root
# to find .venv/pyproject.toml. If we cannot cd, bail before doing any work.
cd "$REPO_ROOT" || exit 1

# Path to the uv binary. Read from the UV_BIN env var so the machine-specific
# absolute path lives in the plist (EnvironmentVariables UV_BIN), NOT hardcoded
# in this committed script. Fall back to a bare "uv" on PATH when unset.
UV_BIN="${UV_BIN:-uv}"

# Ensure the logs/ directory exists (CWD-relative, now that we are at repo root).
mkdir -p logs

# One dated log file per calendar day; runs on the same day append to it.
LOG_FILE="logs/autopilot_$(date +%Y-%m-%d).log"

# Run-start header: enough environment to debug launchd env problems from the
# log alone (timestamp, host, the inherited PATH, which uv, and the CWD).
echo "==== autopilot run START $(date '+%Y-%m-%d %H:%M:%S %z') ====" >> "$LOG_FILE"
echo "[wrapper] hostname=$(hostname)" >> "$LOG_FILE"
echo "[wrapper] PATH=$PATH" >> "$LOG_FILE"
echo "[wrapper] UV_BIN=$UV_BIN" >> "$LOG_FILE"
echo "[wrapper] PWD=$(pwd)" >> "$LOG_FILE"

# Step 1: the ingest. EXACT documented invocation (docs/AUTOPILOT.md:16), and
# --universe etf_basket belongs ONLY here. Append stdout AND stderr to the log.
"$UV_BIN" run python scripts/update_universe.py --universe etf_basket >> "$LOG_FILE" 2>&1
# Capture the ingest exit code immediately, before any other command overwrites $?.
INGEST_RC=$?

# Chain guard: only run run_daily if the ingest succeeded. Ordering is
# load-bearing -- ingest must complete before run_daily or the freshness gate
# reads stale bars and exits 1. An explicit guard logs the reason more clearly
# than a bare && would.
if [ "$INGEST_RC" -ne 0 ]; then
    echo "[wrapper] ingest failed rc=$INGEST_RC, skipping run_daily" >> "$LOG_FILE"
    # Nonzero exit, visible to launchd (last-exit-status) and recorded in the log.
    exit "$INGEST_RC"
fi

# Step 2: run_daily. EXACT documented invocation (docs/AUTOPILOT.md:17).
# run_daily.py takes NO --universe flag (basket hardcoded at run_daily.py:102);
# passing one would error at argparse. Same log redirection as the ingest.
"$UV_BIN" run python scripts/run_daily.py >> "$LOG_FILE" 2>&1
# Capture run_daily's exit code immediately.
RUN_RC=$?

# Run-end footer: both exit codes and a timestamp, so the log is self-contained.
echo "[wrapper] ingest_rc=$INGEST_RC run_daily_rc=$RUN_RC" >> "$LOG_FILE"
echo "==== autopilot run END $(date '+%Y-%m-%d %H:%M:%S %z') ====" >> "$LOG_FILE"

# Propagate run_daily's real outcome so launchd's last-exit-status reflects it.
exit "$RUN_RC"
