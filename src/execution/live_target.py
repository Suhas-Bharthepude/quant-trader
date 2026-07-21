# src/execution/live_target.py

"""
Pure live-target resolver: today's bars -> today's rotation target weights.

This is the LIVE right-edge of the rotation strategy.  It answers one question and
delegates everything else: "given data as of today, what target weights should the
account hold?"  It does NOT re-derive ranking, trailing returns, or the 1/top_n
weighting — it resolves the correct as-of decision date (the most recent COMPLETED
month-end, never a partial current month) and hands that date to the SAME
single_date_weights seam the validated backtest uses.  That shared seam is the whole
point: the shipped live strategy CANNOT drift from the validated one, because both
form weights through one function.

It is PURE: no DuckDB, no broker, no printing, no clock.  Bars and `today` are passed
in by the (impure) runner, so this module stays deterministic and unit-testable with
plain dicts.  The mirror of the pure-vs-IO discipline used by reconcile_to_target and
combine_period_returns.
"""

# dataclass builds the frozen LiveRotationConfig; frozen=True makes the shipped config
# an immutable historical fact, matching RotationVerdict / BacktestResult / OrderRequest.
from dataclasses import dataclass

# datetime types the `today` right-edge argument threaded to the month-end resolver.
from datetime import datetime

# OHLCVBar is the bar schema every symbol's list holds; typed here so the signature
# matches the rest of the stack (single_date_weights / rotation_backtest consume the same).
from src.data.schema import OHLCVBar

# single_date_weights is the SINGLE birthplace of rotation target weights (Day 55 seam).
# Importing and delegating to it is what guarantees live and backtest cannot diverge on
# weight formation — this file adds NO weight arithmetic of its own.
from src.research.cross_sectional import single_date_weights

# most_recent_completed_month_end is the LIVE right-edge resolver (Day 55): it drops an
# incomplete current month so a live run never ranks on a partial-month formation window.
from src.strategies.time_series_momentum import most_recent_completed_month_end


# ---------------------------------------------------------------------------
# LiveRotationConfig — the immutable shipped rotation parameters.
# ---------------------------------------------------------------------------

# frozen=True → the config is an immutable value; reassigning a field raises
# FrozenInstanceError, so the shipped parameters cannot be mutated at runtime.
@dataclass(frozen=True)
class LiveRotationConfig:
    """The four rotation parameters that fully specify a live target-weight decision.

    Deliberately excludes cost_rate: cost is an execution/accounting concern applied by
    the runner/verdict, NOT an input to weight FORMATION (single_date_weights takes no
    cost_rate).  Keeping it out of this config makes that separation structural.
    """

    # Trailing formation window in month-end observations (ranking lookback).
    lookback: int
    # How many symbols to hold AND the fixed weight denominator (each held gets 1/top_n).
    top_n: int
    # Which price field to rank on — "close" or "adj_close".
    price_field: str
    # Absolute-filter switch: False applies the >0 filter (cash when nothing qualifies).
    hold_when_all_negative: bool


# The DESIGNATED live config recorded in DEV_LOG Day 55.  This is the ONLY rotation
# config validated as a STATIC WHOLE out-of-sample (the sorted(candidates)[0] baseline:
# min lookback 3, min top_n 1), with a recorded fixed-parameter OOS Sharpe of +0.44 —
# which LOSES to equal-weight buy-and-hold (+0.59).  It is shipped to run the validated
# strategy end-to-end, NOT because it is a proven edge (the honest negative result is the
# project thesis).  cost_rate is deliberately ABSENT: it is an execution/accounting
# concern, not weight formation, so it never belongs in a weight-forming config.
LIVE_CONFIG = LiveRotationConfig(
    lookback=3,
    top_n=1,
    price_field="close",
    hold_when_all_negative=False,
)


# ---------------------------------------------------------------------------
# live_target_weights — resolve the as-of date, delegate to the shared seam.
# ---------------------------------------------------------------------------

def live_target_weights(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    today: datetime,
    reference_symbol: str,
    config: LiveRotationConfig = LIVE_CONFIG,
) -> dict[str, float]:
    """Resolve today's rotation target weights from bars, delegating to single_date_weights.

    The output — a dict[str, float] of symbol -> weight fraction, where an ABSENT symbol
    means exit-to-zero, weights sum to <= 1.0, and the remainder is implicitly cash — is
    EXACTLY reconcile_to_target's target_weights contract (confirmed Day 54), so it feeds
    the runner with NO adapter.

    The whole point of this function is anti-drift: live and backtest form weights through
    the SAME single_date_weights, so the shipped strategy cannot silently diverge from the
    validated one.  This function ONLY resolves the correct as-of decision date and
    delegates; it contains no ranking, no trailing-return math, and no 1/top_n arithmetic.

    Pure: no DuckDB, no broker, no printing, no clock.  Bars and `today` are passed in.

    Args:
        bars_by_symbol:   symbol -> that symbol's time-ordered OHLCVBar list.
        today:            the live "as of" instant.  Expected tz-aware UTC (matching the
                          bar timestamps); otherwise the timestamp comparison inside the
                          month-end resolver raises a naive-vs-aware TypeError.
        reference_symbol: the spine whose month-ends define the rebalance schedule.
        config:           the rotation parameters to apply (defaults to LIVE_CONFIG).

    Returns:
        A dict mapping each held symbol to its 1/top_n weight; {} when there is no
        completed month-end yet (all-cash) or when the ranking holds nothing.

    Raises:
        ValueError: reference_symbol not in bars_by_symbol.
    """
    # STEP a — GUARD: the reference symbol must be present, since its month-ends ARE the
    # schedule.  Echo the missing name so a typo'd/absent spine fails immediately and
    # legibly, mirroring rotation_backtest's own missing-reference guard.
    if reference_symbol not in bars_by_symbol:
        raise ValueError(
            f"reference_symbol {reference_symbol!r} not in bars_by_symbol "
            f"(keys: {sorted(bars_by_symbol.keys())})"
        )

    # STEP b — RESOLVE the as-of decision date: the most recent COMPLETED month-end on the
    # reference spine at-or-before today.  This drops an in-progress current month, so the
    # live ranking is never taken on a partial-month formation window (the Day 54 risk).
    as_of = most_recent_completed_month_end(bars_by_symbol[reference_symbol], today)

    # STEP c — NO COMPLETED MONTH-END YET: today precedes the first completed month-end
    # (too little history, or only a partial current month is visible).  There is nothing
    # to hold, so the all-cash target {} is returned (reconcile_to_target reads absent
    # symbols as exit-to-zero, so {} correctly means "hold nothing").
    if as_of is None:
        return {}

    # STEP d — DELEGATE to the shared weight seam AS OF the resolved date.  Every parameter
    # comes from the config; this is the ONLY weight source, so live weights are formed by
    # the exact same code path the validated backtest uses.
    return single_date_weights(
        bars_by_symbol,
        as_of,
        config.lookback,
        config.top_n,
        config.price_field,
        config.hold_when_all_negative,
    )
