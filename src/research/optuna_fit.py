# src/research/optuna_fit.py

"""
Optuna-based per-fold parameter fitting for walk-forward validation.

This module supplies a `fit_fn` factory that plugs into the seam in
walk_forward_validate (src/research/walk_forward.py): the validator calls
fit_fn(train_bars) once per fold and uses the returned Strategy on that fold.

The factory here optimises SMACrossoverStrategy's (fast, slow) windows by
maximising WARM-ONLY in-sample Sharpe over the fold's training bars (the
leading slow-SMA warm-up region, where the position is forced FLAT, is dropped
before scoring), then returns the best-found strategy.  The recorded
in_sample_sharpe is therefore the active-regime Sharpe — directly comparable to
the validator's OOS Sharpe, which is itself measured over an already-warm test
window.  Critically, the objective touches ONLY train_bars — the
no-lookahead guarantee of walk-forward validation lives in the seam (which
slices train vs test), but it is RESPECTED here by never letting the objective
see a single test bar.  Fitting on training data and scoring on the untouched
test window is the whole point of walk-forward; an objective that peeked at
test bars would silently destroy that guarantee.

Separation of concerns: this module knows how to FIT a strategy to a window; it
knows nothing about folds, stitching, or OOS aggregation — that is the
validator's job.  The two meet only at the fit_fn(train_bars) -> Strategy
signature.
"""

# Callable types the returned fit_fn.  Imported from collections.abc (not
# typing) per the project's modern-typing convention, matching walk_forward.py.
from collections.abc import Callable

# Optuna runs the hyperparameter search.  TPESampler with a fixed seed makes the
# trial sequence deterministic, which the reproducibility test relies on.
import optuna

# OHLCVBar is the bar type fit_fn receives; Strategy is the return type so the
# factory's signature matches the seam's Callable[[list[OHLCVBar]], Strategy].
from src.data.schema import OHLCVBar
from src.strategies.base import Strategy

# SMACrossoverStrategy is the concrete strategy whose (fast, slow) windows we
# tune.  Its generate_signals raises when len(bars) <= slow_window, which is
# exactly why the param clamps below pin slow < len(train_bars).
from src.strategies.sma_crossover import SMACrossoverStrategy

# TimeSeriesMomentumStrategy is the strategy the momentum fitter below tunes
# (its single `lookback` knob); month_end_indices is the shared month-end helper
# extracted from that strategy, reused here so the fitter's warm-up boundary is
# computed from the SAME month-end definition the strategy uses — they cannot
# drift apart, by construction.
from src.strategies.time_series_momentum import (
    TimeSeriesMomentumStrategy,
    month_end_indices,
)

# Backtester runs each trial to produce the per-bar return series; one instance
# is hoisted per fit_fn call (see fit_fn) since .run is pure and reusable.
from src.backtest.engine import Backtester

# metrics.sharpe_ratio scores the WARM-ONLY return slice (warm-up dropped),
# called directly so the objective controls exactly which returns enter the
# mean/std rather than relying on BacktestResult.sharpe_ratio's full-window value.
from src.backtest import metrics


# Silence Optuna's per-trial INFO logs at import time.  A 30-trial study per fold
# across many folds and many symbols would otherwise flood the run output with
# one line per trial; WARNING keeps genuine problems visible without the spam.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_sma_optuna_fit_fn(
    n_trials: int = 30,
    seed: int = 42,
    fast_range: tuple[int, int] = (5, 50),
    slow_range: tuple[int, int] = (50, 250),
    annualization_factor: int = 252,
    record: list[dict] | None = None,
) -> Callable[[list[OHLCVBar]], Strategy]:
    """Build a fit_fn that tunes SMACrossoverStrategy windows via Optuna.

    The returned callable matches the walk_forward_validate seam contract:
    fit_fn(train_bars) -> Strategy.  Each call runs an independent Optuna study
    that maximises in-sample Sharpe over `train_bars`, then returns an
    SMACrossoverStrategy built from the best (fast, slow) found.

    Precondition: len(train_bars) >= fast_range[0] + 2.  The clamps below shrink
    the search bounds to fit the window (fast <= len - 2, slow <= len - 1); if
    the window is so small that fast_range[0] exceeds the clamped fast upper
    bound, suggest_int receives low > high and raises.  Additionally fast_range
    must sit below slow_range (fast_range[0] < slow_range[1]); otherwise on a
    mid-size window the slow suggest_int can receive low (fast + 1) > high
    (slow_hi) and raise.  And len(train_bars) must exceed slow_range[0], so the
    floor fits under slow_hi = min(slow_range[1], len - 1); otherwise
    max(slow_range[0], fast + 1) can exceed slow_hi and suggest_int raises low >
    high.  Sane configs (realistic train_size, fast below slow) never hit any of
    these; they only matter for pathological inputs (e.g. a tiny window paired
    with a large slow floor).

    Args:
        n_trials:             Optuna trials per fold.  More trials = better
                              optimum but slower; the default 30 is a reasonable
                              fit-quality/speed trade-off for a two-param search.
        seed:                 TPESampler seed.  Fixing it makes the trial
                              sequence — and therefore the chosen (fast, slow) —
                              deterministic for a given train window, which is
                              what the reproducibility test pins.
        fast_range:           Inclusive (low, high) bounds for the fast window
                              before window-clamping.
        slow_range:           Inclusive (low, high) bounds for the slow window
                              before window-clamping.  low is a RESPECTED floor:
                              the effective low is max(slow_range[0], fast + 1),
                              so the search stays at or above the configured floor
                              while still guaranteeing slow > fast.
        annualization_factor: Bars per year for Sharpe annualisation.  Must match
                              the value the validator uses (default 252) so the
                              recorded in_sample_sharpe and the validator's OOS
                              Sharpe annualise to the same units — step 3 compares
                              the two directly.  It does NOT change which (fast,
                              slow) wins (a monotonic scale leaves the argmax
                              unchanged); it only sets the units of the recorded
                              value.
        record:               Optional side-channel list.  When provided, one
                              dict is appended per fit_fn call (i.e. per fold, in
                              fold order) capturing the chosen params and the
                              warm-only in-sample Sharpe — see the append below.

    Returns:
        Callable[[list[OHLCVBar]], Strategy] — the fit_fn for the seam.
    """

    def fit_fn(train_bars: list[OHLCVBar]) -> Strategy:
        """Optimise SMA windows on train_bars and return the best strategy."""

        # Hoist the Backtester out of the objective: .run is pure (no per-call
        # state mutation), so a single instance serves every trial and we avoid
        # reallocating one per trial.  The annualization_factor is fixed here so
        # every trial's Sharpe — and the recorded best — use the validator's units.
        bt = Backtester(annualization_factor=annualization_factor)

        def objective(trial: optuna.Trial) -> float:
            # --- fast window ---------------------------------------------------
            # Clamp the fast upper bound to len(train_bars) - 2 so that, even at
            # the largest fast, there is still room for slow = fast + 1 to stay
            # strictly below len(train_bars).  Concretely: if fast could reach
            # len - 1, then slow = fast + 1 = len, which violates the strategy's
            # len(bars) > slow_window contract and would raise.  The -2 reserves
            # one slot for slow's mandatory +1 gap plus the strict-less-than.
            fast_hi = min(fast_range[1], len(train_bars) - 2)
            fast = trial.suggest_int("fast", fast_range[0], fast_hi)

            # --- slow window ---------------------------------------------------
            # Lower bound is max(slow_range[0], fast + 1): it keeps slow at or
            # above the configured floor AND is always > fast (it is >= fast + 1),
            # so the fast < slow invariant holds BY CONSTRUCTION — Optuna can
            # never suggest slow <= fast, so SMACrossoverStrategy's "slow must be
            # > fast" check can never fire.  When the floor sits above fast + 1
            # (the normal golden-cross case — e.g. slow_range[0]=50, fast<=50) the
            # floor binds and the search stays in the intended regime instead of
            # roaming down to fast-ish/fast-ish crossovers; when fast is large
            # enough that fast + 1 exceeds the floor, fast + 1 binds and fast <
            # slow still holds.  The upper clamp len(train_bars) - 1 guarantees
            # slow < len(train_bars), so generate_signals' "need more than
            # slow_window bars" check can never fire either.  Together these
            # bounds make every suggested trial a valid, constructible strategy.
            slow_hi = min(slow_range[1], len(train_bars) - 1)
            slow = trial.suggest_int("slow", max(slow_range[0], fast + 1), slow_hi)

            # Warm-only in-sample Sharpe is the objective.  This is the ONLY
            # place the objective reads bars, and it reads train_bars exclusively
            # — never test bars.  Fitting must never see test data; respecting
            # that here is what keeps the walk-forward OOS measurement honest.
            # Scoring the active regime (warm-up dropped) means the search rewards
            # genuine signal quality, NOT shorter warm-ups — a shorter slow window
            # would otherwise score better merely by diluting the full-window
            # Sharpe with fewer leading FLAT (zero-return) bars, a measurement
            # artifact rather than real edge.
            strat = SMACrossoverStrategy(fast, slow)
            signals = strat.generate_signals(train_bars)
            result = bt.run(train_bars, signals)
            # Warm-only Sharpe: drop the first `slow` returns, which cover the slow-SMA
            # warm-up window where the position is forced FLAT (zero-return) before the
            # indicator is defined. WHY: the OOS window the validator scores is already
            # warm (warmed by the preceding train bars), so it has no warm-up dilution.
            # Measuring in-sample over the FULL train window would include those leading
            # zeros and understate the in-sample Sharpe, making the step-3 overfitting
            # tax (in-sample vs OOS) apples-to-oranges. Scoring the active regime on both
            # sides makes the tax honest. Dropping `slow` (one past the last NaN-SMA bar)
            # is deliberately one-conservative so no warm-up bar can leak in.
            warm_returns = result.returns[slow:]
            return metrics.sharpe_ratio(warm_returns, annualization_factor)

        # A fresh study per fold: maximise Sharpe, seeded TPE for determinism.
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials)

        best_fast = study.best_params["fast"]
        best_slow = study.best_params["slow"]

        # Side-channel for the caller.  record is appended ONCE per fit_fn call,
        # and the validator calls fit_fn once per fold in chronological order, so
        # record[i] aligns with WalkForwardResult.per_fold[i].  Step 3 pairs the
        # two to compute the overfitting tax — this fold's WARM-ONLY in-sample
        # Sharpe against that same fold's realised OOS Sharpe.  Because the
        # objective is itself warm-only, study.best_value IS the warm-only Sharpe
        # of the winning params — no separate recomputation is needed.
        if record is not None:
            record.append(
                {
                    "fast": best_fast,
                    "slow": best_slow,
                    "in_sample_sharpe": study.best_value,
                }
            )

        # Return the best strategy; the seam warms it over train+test and scores
        # it on the test window only.
        return SMACrossoverStrategy(best_fast, best_slow)

    return fit_fn


def make_tsmom_optuna_fit_fn(
    n_trials: int = 30,
    seed: int = 42,
    lookback_range: tuple[int, int] = (3, 18),
    annualization_factor: int = 252,
    price_field: str = "close",
    record: list[dict] | None = None,
) -> Callable[[list[OHLCVBar]], Strategy]:
    """Build a fit_fn that tunes TimeSeriesMomentumStrategy's lookback via Optuna.

    The returned callable matches the walk_forward_validate seam contract:
    fit_fn(train_bars) -> Strategy.  Each call runs an independent Optuna study
    that maximises WARM-ONLY in-sample Sharpe over `train_bars`, then returns a
    TimeSeriesMomentumStrategy built from the best lookback found.  Like the SMA
    fitter, the recorded in_sample_sharpe is the active-regime (warm-up-dropped)
    Sharpe, directly comparable to the validator's already-warm OOS Sharpe, and
    the objective reads train_bars EXCLUSIVELY — never test bars — so the
    walk-forward no-lookahead guarantee is respected here too.

    TSMOM has ONE tuned knob (lookback), unlike SMA's two (fast, slow), so the
    search is over a single parameter and the record carries a single key.

    Args:
        n_trials:             Optuna trials per fold.  More trials = better
                              optimum but slower; the default 30 matches the SMA
                              fitter — a reasonable trade-off for a one-param
                              search (more than enough to cover a small integer
                              lookback range).
        seed:                 TPESampler seed.  Fixing it makes the trial
                              sequence — and therefore the chosen lookback —
                              deterministic for a given train window, which is
                              what the reproducibility test pins.
        lookback_range:       Inclusive (low, high) MONTH bounds for the single
                              tuned parameter.  high is clamped down per-window so
                              a trial can never exceed the available month-ends
                              (see the objective's lookback_hi); low is respected
                              as-is.  TSMOM has one knob, so this single range
                              replaces the SMA fitter's fast_range/slow_range.
        annualization_factor: Bars per year for Sharpe annualisation.  Must match
                              the value the validator uses (default 252) so the
                              recorded in_sample_sharpe and the validator's OOS
                              Sharpe annualise to the same units.  It does NOT
                              change which lookback wins (a monotonic scale leaves
                              the argmax unchanged); it only sets the recorded
                              value's units.
        price_field:          Price basis ("close" or "adj_close").  Threaded into
                              BOTH the internal Backtester AND every trial's
                              TimeSeriesMomentumStrategy from this ONE variable, so
                              in-sample scoring uses the SAME basis the OOS verdict
                              uses; the engine basis and the strategy basis are
                              therefore equal BY CONSTRUCTION (the Day-32
                              basis-consistency guard in walk_forward_validate is a
                              backstop, never the mechanism).  This is the one real
                              divergence from the SMA fitter, which has no basis.
        record:               Optional side-channel list.  When provided, one dict
                              is appended per fit_fn call (i.e. per fold, in fold
                              order) carrying the chosen lookback and the warm-only
                              in-sample Sharpe: {"lookback": ..., "in_sample_sharpe":
                              ...} — the SMA fitter's fast/slow keys collapse to a
                              single lookback key.

    Returns:
        Callable[[list[OHLCVBar]], Strategy] — the fit_fn for the seam.
    """

    def fit_fn(train_bars: list[OHLCVBar]) -> Strategy:
        """Optimise the TSMOM lookback on train_bars and return the best strategy."""

        # Month-end positions of THIS train window, computed ONCE via the shared
        # helper before the objective.  Using month_end_indices (the very function
        # TimeSeriesMomentumStrategy uses internally) means the lookback clamp and
        # the warm-up slice below cannot drift from the strategy's actual warm-up:
        # "month-end" has exactly one definition across both.
        mei_train = month_end_indices(train_bars)

        # Number of month-end observations available in this train window — the
        # ceiling the lookback must stay strictly below (TSMOM needs M > lookback).
        n_month_ends = len(mei_train)

        # Hoist the Backtester out of the objective: .run is pure, so one instance
        # serves every trial.  THE divergence from the SMA fitter: price_field is
        # threaded in so in-sample returns use the same basis (e.g. adj_close total
        # return) the verdict measures OOS.  Deliberately NOT threading
        # cash_yield/fee/slippage — in-sample stays frictionless exactly like the
        # SMA fitter, so the overfitting tax isolates parameter-selection
        # overfitting, not cost modeling.  annualization_factor is fixed here so
        # every trial's Sharpe — and the recorded best — use the validator's units.
        bt = Backtester(annualization_factor=annualization_factor, price_field=price_field)

        def objective(trial: optuna.Trial) -> float:
            # --- lookback -----------------------------------------------------
            # Clamp the lookback upper bound to n_month_ends - 1 so a trial can
            # NEVER trip TSMOM's "M > lookback" guard on this train window:
            # TimeSeriesMomentumStrategy.generate_signals raises when the number
            # of month-ends M is <= lookback, and n_month_ends - 1 guarantees the
            # suggested lookback is strictly fewer than the available month-ends,
            # so every trial is constructible and scoreable.  This mirrors the SMA
            # fitter's min(..., len-2) / min(..., len-1) window clamps.
            lookback_hi = min(lookback_range[1], n_month_ends - 1)
            lookback = trial.suggest_int("lookback", lookback_range[0], lookback_hi)

            # Build the strategy on the SAME price_field variable as the engine
            # above, so signal basis and scoring basis agree by construction.
            # generate_signals reads train_bars only — never test bars — keeping
            # the OOS measurement honest.
            strat = TimeSeriesMomentumStrategy(lookback=lookback, price_field=price_field)
            signals = strat.generate_signals(train_bars)
            result = bt.run(train_bars, signals)

            # WARM-ONLY in-sample slice via the month-end boundary — the momentum
            # analogue of the SMA fitter's result.returns[slow:].  The trailing
            # return is first DEFINED at the lookback-th month-end (0-indexed
            # position `lookback`), whose bar index is mei_train[lookback];
            # everything before it is the forced-FLAT warm-up (no prior month-end
            # `lookback` positions earlier to divide against).
            #
            # CRITICAL: we slice at the warm-up BOUNDARY, NOT at the first
            # non-FLAT signal.  Momentum legitimately sits FLAT after warm-up
            # whenever the trailing return is negative (a downtrend) — those flat
            # bars are REAL positions the OOS window also scores, so dropping them
            # would inflate the in-sample Sharpe and corrupt the tax.  Slicing at
            # the warm-up boundary keeps both the LONG and the legitimate-flat
            # post-warm-up bars, matching how the validator scores the OOS window.
            #
            # lookback <= lookback_hi <= n_month_ends - 1, so mei_train[lookback]
            # is always a valid index into mei_train.
            warm_start = int(mei_train[lookback])
            warm_returns = result.returns[warm_start:]
            return metrics.sharpe_ratio(warm_returns, annualization_factor)

        # A fresh study per fold: maximise Sharpe, seeded TPE for determinism —
        # IDENTICAL construction to the SMA fitter.
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials)

        best_lookback = study.best_params["lookback"]

        # Side-channel for the caller.  record is appended ONCE per fit_fn call,
        # and the validator calls fit_fn once per fold in chronological order, so
        # record[i] aligns with WalkForwardResult.per_fold[i].  Because the
        # objective is itself warm-only, study.best_value IS the warm-only Sharpe
        # of the winning lookback — no separate recomputation is needed, exactly
        # as in the SMA fitter.
        if record is not None:
            record.append(
                {
                    "lookback": best_lookback,
                    "in_sample_sharpe": study.best_value,
                }
            )

        # Return the best strategy on the SAME price_field, so the strategy the
        # validator warms over train+test and scores OOS uses the identical basis.
        return TimeSeriesMomentumStrategy(lookback=best_lookback, price_field=price_field)

    return fit_fn
