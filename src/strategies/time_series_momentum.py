# src/strategies/time_series_momentum.py

"""
Time-Series Momentum (TSMOM) strategy — long/flat monthly absolute momentum.

Rule:
  - At each month-end, look at the symbol's OWN simple return over the
    trailing `lookback` months.
  - Go LONG  when that trailing return is strictly positive (> 0): the
    symbol has been going up, so we bet it keeps going up.
  - Stay FLAT otherwise (trailing return <= 0, or not enough history yet):
    no position, we sit the period out.

This is "time-series" (a.k.a. absolute) momentum: each symbol is judged
ONLY against its own past, one symbol at a time, never ranked against other
symbols. That is the defining contrast with cross-sectional momentum, and
it is why the rule can be expressed as a pure function of this symbol's bars.

Default parameter is lookback=12 — the canonical 12-month formation window
from the Moskowitz/Ooi/Pedersen "Time Series Momentum" literature, which is
the most widely cited horizon for monthly TSMOM. The window is a constructor
argument so callers can sweep it (3, 6, 12 months are all common) without
touching the strategy code.

This module is deliberately LONG/FLAT only: it emits SIGNAL_LONG or
SIGNAL_FLAT and never SIGNAL_SHORT. That is why SIGNAL_SHORT is not even
imported — referencing it would be a NameError, which is a cheap structural
guarantee that no short can ever leak out of this strategy.

Like sma_crossover.py, this produces a *position* signal, not a *trade*
signal: each element says "I should be long/flat on this bar", and the
backtester derives trades by diffing consecutive positions. Crucially, this
strategy applies NO internal lag — each month-end's signal takes effect ON
that month-end bar and is forward-filled across the following daily bars.
The backtester applies the one and only one-bar lag (signals[:-1] *
returns[1:]); adding a shift here would double-lag and diverge from SMA.
"""

# numpy backs the final positional int8 signal array we return, matching the
# return-type contract every strategy in this project obeys.
import numpy as np

# pandas is used PURELY internally here: it gives us timestamp-indexed Series,
# month-end selection, trailing-return shifting, and forward-fill onto the
# daily index far more cleanly than hand-rolled numpy loops would.
import pandas as pd

# OHLCVBar is the canonical bar dataclass. We type the input as
# `list[OHLCVBar]` to match the contract declared in src.strategies.base.
from src.data.schema import OHLCVBar

# Strategy is the ABC we inherit from; SIGNAL_LONG / SIGNAL_FLAT are the only
# two legal output values for this long/flat strategy. SIGNAL_SHORT is
# intentionally NOT imported — this strategy must never emit a short, and
# leaving the name undefined makes an accidental short a hard NameError.
from src.strategies.base import (
    Strategy,
    SIGNAL_LONG,
    SIGNAL_FLAT,
)


# ---------------------------------------------------------------------------
# Shared month-end detection
# ---------------------------------------------------------------------------


def month_end_indices(bars: list[OHLCVBar]) -> np.ndarray:
    """Return the ascending positional indices of each calendar month's LAST bar.

    A bar is a month-end when the NEXT bar belongs to a different calendar month;
    by convention the FINAL bar of the series is ALWAYS treated as a month-end
    (it has no "next" bar to compare against) — matching the is_month_end[-1] =
    True rule the strategy has always used.

    The year*12+month integer arithmetic is used DELIBERATELY instead of
    pandas to_period("M"): to_period drops timezone info on our tz-aware UTC
    index and would emit a UserWarning on every call, whereas the arithmetic is
    tz-safe.  This is the same rationale carried over from the inline block.

    This helper assumes NON-EMPTY input: generate_signals validates bars
    non-empty upstream (and is the only caller today), so an empty guard here
    would be dead code that changes nothing.

    Extracted so the strategy and any consumer that needs the warm-up boundary
    (e.g. a momentum Optuna fitter computing the lookback-th month-end index)
    share ONE definition of "month-end" — structural agreement rather than two
    copies that could silently drift, the same reasoning behind _stitch_oos in
    walk_forward.py.
    """
    # Build a tz-aware DatetimeIndex from the bars' real timestamps, IN ARRIVAL
    # ORDER (bars are assumed ascending chronological, as everywhere else).
    index = pd.DatetimeIndex([b.timestamp for b in bars])

    # Convert each timestamp to a single integer "month code" (year*12+month)
    # so consecutive bars in the same calendar month share a code and a month
    # boundary shows up as a code change.  tz-safe (see docstring).
    month_codes = index.year * 12 + index.month

    # Pre-allocate the boolean month-end mask, one slot per bar.  We fill it
    # explicitly below so the month-end is a REAL trading date, never a synthetic
    # calendar month-end that could land on a non-trading day.
    mask = np.empty(len(bars), dtype=bool)

    # A bar is a month-end when the NEXT bar's month code differs from its own:
    # month_codes[:-1] != month_codes[1:] is True exactly at those boundary bars.
    mask[:-1] = month_codes[:-1] != month_codes[1:]

    # The very last bar has no "next" bar to compare against and, by convention,
    # always qualifies as a month-end observation.
    mask[-1] = True

    # np.where(mask)[0] returns the ascending positional indices where mask is
    # True — exactly the month-end bar positions, in order.
    return np.where(mask)[0]


# ---------------------------------------------------------------------------
# Time-Series Momentum concrete strategy
# ---------------------------------------------------------------------------

class TimeSeriesMomentumStrategy(Strategy):
    """Long when the trailing `lookback`-month return is positive, else flat."""

    def __init__(self, lookback: int = 12, price_field: str = "close") -> None:
        # price_field selects which OHLCVBar price field the trailing-return
        # signal is computed from. "close" (the default) computes momentum from
        # PRICE return and reproduces the pre-existing behaviour bit-for-bit;
        # "adj_close" computes it from TOTAL return (dividends/splits folded in).
        # It is the LAST parameter and defaults to "close" so every existing
        # construction site (positional/keyword lookback, bare
        # TimeSeriesMomentumStrategy()) is unchanged. WARNING: when this strategy
        # is run against a buy-and-hold benchmark, this price_field and the
        # Backtester's price_field must match, or the comparison mixes price
        # return with total return — the caller is responsible for passing the
        # same basis to both.

        # lookback < 1 has no meaning — you cannot measure a return over zero
        # or fewer months. Catching it here gives a clear, actionable error
        # rather than a confusing empty/degenerate result deeper down.
        if lookback < 1:
            # f-string echoes the actual value so the traceback is actionable,
            # matching the validation style in sma_crossover.py.
            raise ValueError(f"lookback must be >= 1, got {lookback}")

        # price_field must name a real, loggable price column. Only "close" and
        # "adj_close" are valid bases; anything else (a typo or non-price field)
        # is rejected OUTRIGHT here — no silent fallback — mirroring the engine's
        # own guard. {price_field!r} quotes the bad value so the message reads
        # cleanly even for empty strings.
        if price_field not in {"close", "adj_close"}:
            raise ValueError(
                f"price_field must be 'close' or 'adj_close', got {price_field!r}"
            )

        # Stash the validated parameter on the instance so generate_signals and
        # the `name` property can read it. Plain attribute (no property/
        # dataclass) keeps this minimal and effectively immutable by
        # convention — construct a new strategy rather than mutate this one.
        self.lookback = lookback

        # Stash the validated price basis right after lookback, same no-mutation
        # convention: generate_signals reads it via getattr to pick the field.
        self.price_field = price_field

    @property
    def name(self) -> str:
        # f-string includes the actual lookback so logs and backtest reports
        # identify the exact configuration that produced a result — critical
        # when comparing parameter sweeps side by side.
        #
        # The default-basis branch MUST stay byte-for-byte "TSMOM({lookback})":
        # existing tests and any committed result labels reference this exact
        # string, so emitting the basis here only for the non-default case keeps
        # the default fully backward-compatible while still making an adj_close
        # run distinguishable in reports and parameter sweeps.
        if self.price_field == "close":
            return f"TSMOM({self.lookback})"

        # Non-default basis: surface it so a total-return run is never silently
        # confused with a price-return run of the same lookback.
        return f"TSMOM({self.lookback}, {self.price_field})"

    def generate_signals(self, bars: list[OHLCVBar]) -> np.ndarray:
        # Empty input is almost always an upstream bug (e.g. a symbol with no
        # data slipped through). Fail loudly with the exact same wording
        # sma_crossover.py uses, for consistency across strategies.
        if not bars:
            raise ValueError("bars must be non-empty")

        # Build a pandas Series of close prices indexed by each bar's actual
        # timestamp, IN THE ORDER THE BARS ARRIVE. We assume bars are already
        # in ascending chronological order — the same assumption sma_crossover
        # and the backtester make — so we do NOT re-sort here.
        close_series = pd.Series(
            # The values are the per-bar prices in arrival order. The basis is
            # configurable via self.price_field; getattr reads the chosen field
            # ("close" or "adj_close") off each bar. The local is still named
            # close_series because it is the reference price series the
            # month-end/trailing-return logic builds on — renaming it would
            # ripple for zero gain; the price_field meaning is carried by the
            # constructor param/comments.
            data=[getattr(bar, self.price_field) for bar in bars],
            # The index is the bars' real timestamps, so every downstream
            # selection and forward-fill happens at true trading dates.
            index=pd.DatetimeIndex([bar.timestamp for bar in bars]),
            # float64 keeps the division math precise and dtype-stable.
            dtype=np.float64,
        )

        # Positional indices of each calendar month's last bar, via the shared
        # helper above.  Extracting this means the strategy and any consumer that
        # needs the warm-up boundary (e.g. the momentum Optuna fitter) compute
        # "month-end" from ONE definition, so they cannot drift apart.
        mei = month_end_indices(bars)

        # Select just the month-end closes, still indexed at their real
        # trading-date timestamps. This is the monthly series we reason over.
        # .iloc[mei] selects the same positions, in the same order, that the old
        # close_series[boolean_mask] selected — np.where(mask)[0] is exactly the
        # True positions of that mask — so month_end_close, its DatetimeIndex, M,
        # the guard, trailing_return, monthly_signal, and the ffill are unchanged.
        month_end_close = close_series.iloc[mei]

        # M is the number of month-end observations we actually have.
        M = len(month_end_close)

        # We need strictly MORE month-ends than `lookback` so that at least one
        # month-end has a prior observation `lookback` positions earlier to
        # compute a trailing return against. M == lookback would leave every
        # return NaN, so `<=` is the correct too-short guard — mirroring the
        # "need more than slow_window bars" guard in sma_crossover.py.
        if M <= self.lookback:
            # Echo both numbers so the user sees the gap at a glance.
            raise ValueError(
                f"need more than lookback ({self.lookback}) month-end observations "
                f"to compute time-series momentum, got {M}"
            )

        # Trailing-`lookback`-month simple return at each month-end: today's
        # month-end close divided by the month-end close `lookback` positions
        # earlier, minus 1. shift(lookback) pulls the earlier month-end value
        # onto the current row; the first `lookback` rows have no prior value,
        # so they become NaN — exactly the warmup period we want.
        trailing_return = month_end_close / month_end_close.shift(self.lookback) - 1.0

        # Map each month-end's trailing return to a monthly position signal:
        # SIGNAL_LONG where the return is STRICTLY greater than 0, otherwise
        # SIGNAL_FLAT. The strict `> 0.0` makes the tie-break explicit — a
        # trailing return of exactly 0 maps to FLAT. NaN > 0.0 is False, so the
        # warmup month-ends fall through to FLAT automatically, which is the
        # FLAT warmup the strategy contract requires. We use the named
        # constants (not raw 1/0) so the intent reads clearly.
        monthly_signal = pd.Series(
            # np.where evaluates the strict-greater-than mask elementwise; True
            # -> SIGNAL_LONG, False (including NaN comparisons) -> SIGNAL_FLAT.
            data=np.where(trailing_return.to_numpy() > 0.0, SIGNAL_LONG, SIGNAL_FLAT),
            # Keep these signals labeled at the same real month-end timestamps
            # so the forward-fill below aligns them to the right daily bars.
            index=month_end_close.index,
            # int8 matches the signal dtype used across the project (values are
            # only ever in {0, 1} here, so int8 is ample and memory-cheap).
            dtype=np.int8,
        )

        # Forward-fill the monthly signal onto the FULL daily bar index: each
        # daily bar takes the most recent month-end signal whose date is on or
        # before that bar. method="ffill" propagates the last month-end value
        # forward across the intervening daily bars until the next month-end.
        # We do NOT shift the signal forward — the month-end's signal takes
        # effect ON its own bar; the backtester supplies the single one-bar lag.
        daily_signal = monthly_signal.reindex(close_series.index, method="ffill")

        # Bars that fall BEFORE the very first month-end have no prior signal to
        # forward-fill from and arrive as NaN; the contract says they must be
        # FLAT, so fill those leading NaNs with SIGNAL_FLAT.
        daily_signal = daily_signal.fillna(SIGNAL_FLAT)

        # Return a positional numpy array of int8, length exactly len(bars),
        # aligned one-to-one with the input bars and carrying NO index — the
        # same shape contract SMACrossoverStrategy returns. to_numpy drops the
        # pandas index; dtype=np.int8 casts the forward-filled values (which are
        # only ever 0 or 1) back to the canonical signal dtype. This array
        # never contains -1: this strategy is long/flat only by construction.
        return daily_signal.to_numpy(dtype=np.int8)
