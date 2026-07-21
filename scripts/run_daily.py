# scripts/run_daily.py

"""
Thin entry-point assembler for the daily autopilot run.

This script OWNS no strategy, order, weight, decision, or freshness logic - every
piece of real logic is imported from already-tested src/ modules and merely wired
and sequenced here.  Its whole job is: configure logging, build the (paper)
broker, load bars through today, REFUSE to trade on stale data, then hand the
tested run_daily the injected clock/notifier and translate the outcome into a
shell exit code.

Skeleton modelled on scripts/rotation_verdict.py (def main() -> int; sys.exit(
main()); argparse) and logging block on scripts/first_order.py.

Exit codes (cron/scheduler-friendly):
    0 = clean run, INCLUDING a no-op day (market closed / no new month-end).
    1 = any failure: missing API keys, empty symbols, missing reference symbol,
        STALE bars (the freshness gate), or a failed rebalance.

Notifications today are LOGGING-ONLY (see _log_notifier): the auditable record
goes through the same handlers, adding no new dependency.  A real Slack/email
backend via stdlib urllib is a documented future option, not built today.
"""

# argparse gives a --help and a place to grow flags, matching rotation_verdict.py.
import argparse

# logging is configured HERE (the entry point owns handler setup); src/ modules only
# getLogger and leave configuration to this caller.
import logging

# sys.exit() propagates main()'s int return to the shell as $? for the scheduler.
import sys

# date/datetime/timezone: date for the load window + freshness reference, datetime for
# the tz-aware "today" passed into run_daily.  These clock reads live HERE at the impure
# edge; every pure function downstream takes the date/time injected.
from datetime import date, datetime, timezone

# Path creates the logs/ dir for the file handler (mirrors scripts/limit_order.py).
from pathlib import Path

# AlpacaBroker.from_env() reads .env (via python-dotenv) and hardcodes paper=True.
from src.brokers.alpaca_broker import AlpacaBroker

# build_symbol_list / load_bars_for_symbols are the shared CLI helpers (same ones
# rotation_verdict.py uses) for resolving the basket and reading bars from DuckDB.
from src.research.cli_common import build_symbol_list, load_bars_for_symbols

# stale_symbols is the PURE freshness gate: it decides whether the loaded bars are
# current through today.  All the staleness arithmetic lives in the tested src/ module.
from src.data.freshness import stale_symbols

# run_daily is the tested IO shell (decide -> live_target_weights -> run_rebalance ->
# state write), and the notify seam types the injected _log_notifier below.
from src.execution.daily_runner import run_daily
from src.execution.notify import Notification, format_notification


# ---------------------------------------------------------------------------
# Logging configuration — this entry point owns the handlers.
# ---------------------------------------------------------------------------

# Ensure logs/ exists before attaching a file handler (parents/exist_ok make it a no-op
# when already present, matching scripts/limit_order.py's Path pattern).
Path("logs").mkdir(parents=True, exist_ok=True)

# Console handler + shared format/datefmt, matching scripts/first_order.py's block.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# Add a FileHandler so an unattended run also leaves a durable on-disk record under
# logs/daily_runner.log (git-ignored via the logs/ rule).  Same format as the console.
_file_handler = logging.FileHandler("logs/daily_runner.log")
_file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"))
# Attach to the ROOT logger so every module's records (run_daily, freshness, etc.) land
# in the file as well as the console.
logging.getLogger().addHandler(_file_handler)

# Module-scoped logger, named per cli_common.py's idiom.
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module constants — mirror scripts/rotation_verdict.py's locked config.
# ---------------------------------------------------------------------------

# The fixed 17-ETF basket universe (resolved from config/universe.yaml).
_UNIVERSE = "etf_basket"

# The reference spine whose month-ends define the rebalance schedule.
_REFERENCE = "SPY"

# Load-window start; the deep backfill runs 2008->present, this captures all of it.
_START = "2000-01-01"

# Staleness threshold in CALENDAR days.  Must be generous enough to span a normal
# non-trading gap (a Fri->Mon weekend is 3 days; a long holiday weekend up to 4), or a
# healthy run would falsely read as stale.  4 tolerates weekends and most holidays.
_MAX_STALENESS_DAYS = 4


# ---------------------------------------------------------------------------
# _log_notifier — the injected notify_fn (logging-only, no network dependency).
# ---------------------------------------------------------------------------

def _log_notifier(n: Notification) -> None:
    """Injected notify_fn: log the notification line at ERROR on failure else INFO.

    This is a delivery backend that reuses the already-configured logging handlers, so
    it needs NO new dependency and still leaves an auditable record.  It wires into the
    Day-58 notify seam (run_daily's notify_fn); a real Slack/email backend via stdlib
    urllib is a documented future option, not built today.
    """
    # Render the compact one-line human summary via the tested formatter.
    line = format_notification(n)
    # A failure is an ERROR-level record; acted/no-op are routine INFO.
    if n.outcome == "failed":
        log.error(line)
    else:
        log.info(line)


# ---------------------------------------------------------------------------
# main — assemble and sequence the tested pieces; translate outcome to exit code.
# ---------------------------------------------------------------------------

def main() -> int:
    """Wire the tested pieces into one daily run and return a shell exit code."""
    # Minimal argparse: no flags today (a dry-run already exists via
    # scripts/rebalance_smoke.py), but keep the parser so the script has --help and a
    # place to grow, matching rotation_verdict.py's skeleton.
    parser = argparse.ArgumentParser(
        description="Daily autopilot run: freshness-gated, paper-only rotation rebalance."
    )
    # Parse (accepts no custom flags; -h works).
    parser.parse_args()

    # STEP 1 — BUILD THE BROKER from .env.  Missing keys raise EnvironmentError inside
    # from_env; catch it and fail loud with a non-zero exit rather than a raw traceback.
    try:
        broker = AlpacaBroker.from_env()
    except EnvironmentError as exc:
        # Missing/mis-set API keys: log the actionable message and fail (exit 1).
        log.error("broker construction failed (check .env keys): %s", exc)
        return 1

    # STEP 2 — PAPER GUARD (defense-in-depth; run_rebalance re-runs it as its first
    # statement).  A live account raises RuntimeError -> log and fail non-zero.
    try:
        broker.verify_paper_account()
    except RuntimeError as exc:
        log.error("paper-account guard failed (refusing to run on a live account): %s", exc)
        return 1

    # STEP 3 — RESOLVE SYMBOLS from the fixed universe (None -> use _UNIVERSE).
    symbols = build_symbol_list(None, _UNIVERSE, limit=1000)
    # Guard: an empty basket means a config/universe problem -> fail non-zero.
    if not symbols:
        log.error("symbol list is empty for universe %r", _UNIVERSE)
        return 1
    # The reference spine MUST be in the resolved universe (its month-ends are the schedule).
    if _REFERENCE not in symbols:
        log.error("reference symbol %r not in resolved universe %r", _REFERENCE, _UNIVERSE)
        return 1

    # STEP 4 — LOAD BARS through today (end = today's date so the window includes today's
    # bar once ingest has stored it).  Reads DuckDB; the runner never ingests.
    bars_by_symbol = load_bars_for_symbols(symbols, _START, date.today().isoformat())
    # Guard: no bars at all means ingest never ran -> fail non-zero.
    if not bars_by_symbol:
        log.error("no bars loaded for any symbol (run the ingest first)")
        return 1
    # PRESENCE check (which the freshness gate does NOT do): the reference spine must have
    # data, or there is no schedule to run.
    if _REFERENCE not in bars_by_symbol:
        log.error("reference symbol %r has no bars loaded", _REFERENCE)
        return 1

    # STEP 5 — FRESHNESS GATE (the safety crux).  Map each symbol to its most recent bar
    # date, then ask the PURE stale_symbols whether anything is behind as of today.
    last_bar_dates = {sym: bars[-1].timestamp.date() for sym, bars in bars_by_symbol.items()}
    stale = stale_symbols(last_bar_dates, date.today(), _MAX_STALENESS_DAYS)
    # If ANY symbol is stale, REFUSE to trade: log the offenders and fail non-zero WITHOUT
    # calling run_daily.  The bot must never rank/rebalance on bars that are not current.
    if stale:
        log.error(
            "STALE bars, refusing to trade (threshold %d days): %s",
            _MAX_STALENESS_DAYS,
            stale,
        )
        return 1

    # STEP 6 — the tz-aware UTC decision timestamp.  This clock read lives HERE at the
    # impure edge; run_daily and all the pure functions take `today` injected.
    today = datetime.now(timezone.utc)

    # STEP 7 — RUN the tested daily runner.  is_open_fn defaults to broker.is_market_open()
    # and notify_policy defaults to ACTED_AND_FAILED; we inject the logging notifier.  A
    # failed rebalance is already logged (with traceback) and notified inside run_daily's
    # failure path, so here we just add a short line and fail non-zero.
    try:
        decision = run_daily(
            broker,
            bars_by_symbol,
            today,
            reference_symbol=_REFERENCE,
            notify_fn=_log_notifier,
        )
    except Exception:  # noqa: BLE001 - already logged+notified inside run_daily; exit non-zero
        log.error("run_daily failed, see traceback above")
        return 1

    # STEP 8 — SUCCESS (including a no-op): log the decision reason and exit 0.  A clean
    # no-op day (market closed / no new month-end) is a successful run, not an error.
    log.info("daily run complete: %s", decision.reason)
    return 0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

# Propagate main()'s return value to the shell as $? when run directly.
if __name__ == "__main__":
    sys.exit(main())
