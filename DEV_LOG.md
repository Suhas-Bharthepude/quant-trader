# DEV_LOG — quant-trader

Running log of development work. Most recent entry first.

---

## Day 28 — 2026-06-23

## Day 28

Worked on:
- Added a transaction-cost model (fees + slippage, in basis points, charged
  per unit of turnover) to the Backtester in src/backtest/engine.py. Both
  fee_bps and slippage_bps default to 0.0, so a default Backtester is
  cost-free and bit-for-bit identical to the pre-cost behavior; all 156 prior
  tests stayed green unchanged.
- Cost is a turnover-proportional log-return drag: held position is signals
  shifted by one (the engine's existing one-bar lag), turnover is the absolute
  per-bar change in held position, and cost = turnover * (fee+slippage)/10000.
  Subtracted from gross returns in place, so equity, Sharpe, total return, and
  max drawdown are all net.
- Added total_cost_pct to BacktestResult (appended as the last field, default
  0.0, so the second construction site in test_research_runner.py is untouched).
- 7 new cost tests in tests/test_backtest.py (zero-cost identity, single round
  trip = 2 units, holding = 1 unit, long-to-short flip = 3 units, costs lower
  Sharpe and return, negative bps raise, fee+slippage add). 163 tests green.

Why it matters:
- This is the friction that turns a backtest from a fantasy into something
  closer to honest. It is one of the three things (with adj_close total-return
  accounting and cash-on-flat) that have to be in before any TSMOM-vs-B&H
  verdict means anything.

Architectural note:
- Cost is charged as a linear log-return drag (turnover*cost_rate subtracted),
  not the multiplicative (1 - turnover*cost_rate). At basis-point magnitudes
  the gap is negligible (second order in the rate); the approximation's
  validity is bounded by a small cost_rate, which realistic costs respect.
- win_rate and Trade records stay GROSS; per-trade cost attribution is a
  deliberately deferred scope boundary. Only the aggregate metrics are net.

Verification:
- SPY TSMOM(12) gross-vs-net sanity (one-off, not committed) at 3 bps total:
  total return 3.1236 -> 3.1026, Sharpe 0.5300 -> 0.5281, max drawdown
  unchanged, total_cost_pct 0.0051 over 9 round trips. Net strictly worse on
  return and Sharpe, drawdown untouched, cost tiny - momentum's low turnover
  means costs barely bite, which is the point.

Blocked on:
- Nothing.

Next up:
- DATA DEFECT found during the SPY sanity: SPY's stored history has a NaN
  close on its final bar (2026-06-10; open and volume present, close and
  adj_close NaN). It poisons total return and max drawdown to NaN on the full
  series. Likely a stale/partial last-day fetch. Must re-fetch or clean SPY
  AND audit the other 16 ETFs for the same NaN-tail before any verdict run.
- ENGINE GAP: run() validates signal length/dtype/values but not finite
  closes, so a NaN close silently produces NaN metrics. Add a finite-close
  guard in run() with its own tests.
- Then the remaining pre-verdict accounting: adj_close total-return (switch
  signal and engine together), cash yield on flat periods. Then the full
  walk-forward + Optuna + overfitting-tax + B&H run, folds sized against the
  12-month lookback.

Time spent: 1 hour

## Day 27 — 2026-06-22

**Worked on:** Added CI via GitHub Actions (.github/workflows/ci.yml) — runs `uv run pytest -q -m "not integration"` on every push and PR to main in a clean Ubuntu environment, excluding the 4 live-API/network integration tests (152 hermetic tests run in CI). Added a project CLAUDE.md encoding the engineering conventions (Strategy contract, verify-on-disk discipline, two-commit git flow, untracked-docs rule, scope limits). 

**Why it matters:** Moves test verification off conversational attestation onto machine-produced logs — the green check is machine truth, not a prose claim, and catches a lookahead or contract regression the moment it lands. CLAUDE.md makes every Claude Code session start aligned with the conventions instead of re-deriving them per prompt. Neither touches trading edge; both serve result integrity and portfolio credibility. Made the four load_bars_for_symbols tests in test_cli_common.py hermetic via a tmp_path DuckDB fixture seeded with synthetic SPY/QQQ/AAPL bars (monkeypatching cli_common.DuckDBStore), after the first CI run surfaced that three silently depended on the local ingested DB and a fourth passed only because CI's DB was empty. CI now runs 152 hermetic tests green.

**Architectural note:** No application or test code changed. CI is tests-only (no coverage gate, lint, or deploy). The `-m "not integration"` filter is the hermeticity boundary — live-API tests need secrets and would be a separate secret-gated job later. The first red CI run did its job: it caught three tests that were integration tests in disguise (depending on un-versioned local DB state) and fixed them by isolating the DB, not by tagging them out, so the logic stays covered in CI.

**Blocked on:** None.

**Next up:** Wire TimeSeriesMomentumStrategy into the walk-forward / Optuna / overfitting-tax / buy-and-hold harness; add a transaction-cost model before any verdict.

**Time spent:** 1.5 hours


## Day 26 — 2026-06-21

**Worked on:** Added TimeSeriesMomentumStrategy (long/flat, monthly rebalance, 12-month default lookback) in src/strategies/time_series_momentum.py, matching the existing Strategy contract: list[OHLCVBar] → int8 np.ndarray, emitting only SIGNAL_LONG and SIGNAL_FLAT, with a FLAT warmup and no internal lag. Added 12 unit tests in tests/test_strategies.py (uptrend/downtrend/flat, warmup, output contract, long-flat-only, monthly cadence, no-lookahead, too-short raise, empty raise) — 156 tests green. SPY sanity check: flat through 2008–2009 and through 2022, long through the recoveries — the crisis-avoidance behavior TSMOM is supposed to show.


**Worked on:** Added CI via GitHub Actions (.github/workflows/ci.yml) — runs `uv run pytest` on every push and PR to main in a clean Ubuntu environment. Added a project CLAUDE.md encoding the engineering conventions (Strategy contract, verify-on-disk discipline, two-commit git flow, untracked-docs rule, scope limits).




**Why it matters:** First strategy with a real economic thesis (momentum), unlike the edgeless SMA dummy. It drops into the existing backtester/walk-forward harness with no contract change — the primitive the Optuna sweep, cost model, and B&H verdict all hang off.


**Why it matters:** Moves test verification off conversational attestation onto machine-produced logs — the green check is machine truth now, not a prose claim. CLAUDE.md makes every Claude Code session start aligned with the conventions instead of re-deriving them per prompt.

**Architectural note:** No internal shift: the backtester's signals[:-1] * returns[1:] is the only lag — each month-end's signal takes effect on its own month-end bar and is forward-filled across the following days. The final bar is forced to be a month-end by convention; this is inert for the backtest (the engine uses signals[:-1], so the last signal earns no return) but needs an exchange-calendar check before live execution, since "is the last bar a month-end" is undecidable from price data alone. It uses raw close, matching the engine, so the backtest is price-return, not total-return.

**Architectural note:** No application or test code changed. CI is tests-only (no coverage gate, lint, or deploy). CLAUDE.md is tracked, unlike the intentionally-untracked trading_explained.md and daily_prompt.md.


**Blocked on:** Nothing.

**Next up:** Before any TSMOM-vs-B&H verdict, make the comparison honest: (1) total-return accounting — switch the signal AND the engine to adj_close together, never just one; (2) credit cash yield on flat periods (currently 0); (3) the transaction-cost model. Raw-close plus zero-cash currently flatters TSMOM vs an always-invested B&H. Then run TSMOM through walk-forward + Optuna + overfitting-tax + B&H, sizing folds against the 12-month lookback (a test fold needs multiple years to clear the 12-month warmup with usable post-warmup signal). Cross-sectional rotation stays a separate later phase (it breaks the per-symbol contract).

**Time spent:** 1 hour

---

## Day 25 — 2026-06-19

**Worked on:** Built scripts/overfitting_tax.py — the CLI that runs the per-fold Optuna fitter (make_sma_optuna_fit_fn) against the fixed SMA(50,200) baseline across the full etf_basket and reports the overfitting tax. It mirrors compare_walkforward's loading (build_symbol_list / load_bars_for_symbols), builds each symbol's (train=504, test=126) splits ONCE and scores them twice on byte-identical windows — fixed (static SMA, no fit_fn) and fitted (fit_fn re-tuning fast/slow per fold by warm-only in-sample Sharpe) — then prints per-symbol Fixed OOS / Fitted OOS / B&H / mean in-sample / tax, plus a basket aggregate and two beat-counts. The only non-glue logic, summarize_tax (tax = mean per-fold in-sample Sharpe − stitched fitted OOS), is a pure function unit-tested in tests/test_overfitting_tax.py (the formula + the empty-fold guard). Full suite 144. Verified the harness against the committed baseline: every Fixed OOS value reproduces the Day-24 baseline run exactly across all 17 symbols, so the fitted column is trustworthy.

**Why it matters:** This is the clean negative result the whole validation arc was built to produce, and it is unambiguous. Across the basket, per-fold tuning does NOT create out-of-sample edge: mean fitted OOS Sharpe −0.02 vs the fixed baseline's +0.04 — tuning is, if anything, slightly worse — and fitted beat the static baseline on only 6 of 17 symbols (a coin flip). The overfitting tax is enormous and universal: mean in-sample Sharpe +1.01 collapses to mean fitted OOS −0.02, a ~1.0-Sharpe-unit gap on EVERY symbol (range +0.67 to +1.41). The optimizer reliably finds ~1.0 in-sample Sharpe that evaporates entirely out-of-sample — textbook overfitting, measured. And buy-and-hold dominates both: mean B&H Sharpe +0.41 beats fixed (+0.04) and fitted (−0.02) by a wide margin, winning on 15 of 17 symbols. Fitted beat B&H on exactly 2 — TLT (+0.26 vs +0.01) and IEF (+0.16 vs +0.09), the two bond ETFs whose own buy-and-hold is ~flat, so the trend filter wins only by sidestepping the 2020–22 bond drawdown (crisis alpha against a near-zero benchmark, not deployable edge). A methodological note worth recording: a 2-symbol preview (SPY, TLT) had shown fitted beating fixed on both, which would have suggested tuning helps — the full basket flipped that, the same small-sample cherry-pick trap QQQ illustrated earlier.

**Architectural note:** The harness's value is that it makes the comparison structural and verifiable — the fixed control reproduces the prior committed baseline number-for-number across all 17 symbols, which is the proof the fitted measurement is honest rather than a coincidence of a new code path. summarize_tax takes plain scalars (not a WalkForwardResult) so the tax formula is unit-tested without a live run, following the same extract-the-pure-decision pattern as should_alert and resolve_window. The script is fixed to etf_basket (no --universe flag) because the tax question is specifically about this stable, long-history basket; --symbols still overrides for spot checks.

**Blocked on:** None.

**Next up:** The SMA crossover is now conclusively edgeless on this basket — both as a fixed rule and tuned per fold — and the pipeline has proven it can identify a no-edge strategy as no-edge while quantifying the overfitting tax. The methodology vehicle has done its job. The next phase is to test a strategy with an actual economic alpha hypothesis (not a moving-average rule with no reason to work), running it through the same seam + walk-forward + tax harness, which now exists and is validated.

**Time spent:** ~3 hours

---

## Day 24 — 2026-06-19

**Worked on:** Built the full fit_fn pipeline that turns the walk-forward validator from a fixed-strategy scorer into a per-fold optimizer, in three layered pieces — a benchmark to measure against, the seam to plug fitting in, and the Optuna fitter itself. (1) Buy-and-hold benchmark in src/research/walk_forward.py: extracted the fold-stitching + return-metrics logic into a shared `_stitch_oos(per_fold, annualization_factor)` used by BOTH the strategy path and the benchmark, so the apples-to-apples comparison is structural, not two copies that could drift. The benchmark runs an always-long position (np.full(len, SIGNAL_LONG)) through the identical backtester.run on the identical test_bars per fold — same windows, same warm-up exclusion, same seam-zero stripping — so only the position series differs. Added bh_return/bh_sharpe/bh_max_drawdown to the frozen WalkForwardResult; compare_walkforward.py prints OOS / B&H / Δ per symbol. An invariant test pins it: an always-long *strategy* yields a stitched OOS identical to the benchmark. (2) fit_fn seam: added `fit_fn: Callable[[list[OHLCVBar]], Strategy] | None = None` to walk_forward_validate; the single seam line is now `fold_strategy = strategy if fit_fn is None else fit_fn(train_bars)`, with everything downstream byte-identical, so a fitted strategy is warmed over train+test yet scored only on the test window. No lookahead — fit_fn sees TRAIN bars only and signals are causal; the linchpin test asserts the i-th call receives exactly splits[i][0]. (3) Optuna fitter in new src/research/optuna_fit.py: `make_sma_optuna_fit_fn(...)` returns a fit_fn that runs a seeded-TPE study per train window, maximizing WARM-ONLY in-sample Sharpe and returning the best SMACrossoverStrategy. Search is window-clamped so every trial is constructible — fast ≤ len−2, slow ∈ [max(slow_range[0], fast+1), min(slow_range[1], len−1)] — so fast < slow and slow < len both hold by construction, and slow_range[0] is a respected floor that keeps the search in the golden-cross regime. A record side-channel captures {fast, slow, in_sample_sharpe} per fold, aligned by index with per_fold[i], for the step-3 tax. Seven tests including reproducibility (seed → identical params), a direction test on a trend-reversal series (a minimize/constant objective fails it), and a warm-only test (recorded Sharpe == warm-only slice, ≠ full-window). uv add optuna (4.9.0). Full suite 142.

**Why it matters:** The benchmark is the honesty check the project hinges on: across etf_basket, buy-and-hold BEATS the fixed SMA(50,200) on 15 of 17 symbols. QQQ — whose 0.53 OOS Sharpe looked tempting last session — LOSES to simply holding QQQ (B&H 0.80, Δ −0.27), which quantifies and kills the cherry-pick. The only positive alpha is TLT (Δ +0.22, the filter sidestepped the 2020–22 bond drawdown) and IEF (Δ +0.04) — real but modest crisis-alpha, not deployable. The seam + fitter exist to ask the next question rigorously: does *tuning* the SMA per fold beat the fixed baseline out-of-sample? The warm-only objective is what makes that measurement honest — the recorded in-sample Sharpe is scored over the same active regime as the OOS window (warm-up dropped on both sides), so the in-sample-vs-OOS tax is apples-to-apples rather than flattered by leading zero-return bars.

**Architectural note:** _stitch_oos makes the OOS-vs-B&H comparison rest on byte-identical code rather than discipline. The seam is a true one-line substitution — the no-lookahead guarantee lives in the validator's slice (train prefix warms, test suffix scores), and the fitter respects it by never letting its objective touch a test bar. The window clamps and slow floor guarantee every Optuna trial is a valid SMACrossoverStrategy without reject-and-retry, and keep the search semantically meaningful. Warm-only is a better *objective*, not just a cleaner report: a full-window Sharpe would quietly reward shorter slow windows for having fewer warm-up zeros — a measurement artifact, not signal quality. The fitter is reusable infra: the make_*_fit_fn pattern generalizes to any future strategy through the same seam.

**Blocked on:** None.

**Next up:** Step 3 — wire make_sma_optuna_fit_fn into compare_walkforward (or a sibling script), run across all 17 etf_basket symbols, and pair record[i] (warm-only in-sample Sharpe) with per_fold[i] (realized OOS Sharpe) to print the overfitting tax alongside the fixed-baseline OOS and the buy-and-hold line. Honest expectation: fitted OOS lands back near the fixed-SMA baseline and stays under buy-and-hold — the clean non-result that proves tuning a no-edge signal does not manufacture OOS edge.

**Time spent:** ~3 hours

---

## Day 23 — 2026-06-17

**Worked on:** Added content-level validation to scripts/data_health.py — until now the health report only caught *structural* problems (stale, missing, short history); it never looked at the bar values themselves. Two pure, fully-tested helpers do the work: `find_zero_volume_bars` (flags `volume == 0`) and `find_extreme_jumps` (flags any consecutive close-to-close ratio outside [0.5, 2.0], i.e. a >+100% or >−50% single-session move). Both are wired into the per-symbol report via one parameterized query filtered to exactly the inspected universe (`AND symbol IN (?, …)`), which loads ~30x less data than scanning all 520 symbols when `--universe etf_basket` is given, and skips the query entirely if the universe resolves to no symbols. Also extracted `should_alert(stale, missing, zero_volume)` so the process exit policy is unit-testable without running `main()`: zero-volume escalates to exit 1 (a liquid name should never have a zero-volume day), but extreme-jump deliberately stays informational — it also fires on genuine extreme moves, so it cannot be an automatic verdict. Added tests/test_data_health.py with 9 tests: both helpers (including the strict upper/lower [0.5, 2.0] boundaries, a drop-to-zero, and the non-positive-prior-close skip that avoids a divide-by-zero) plus `should_alert` (all-clear → no alert, each flag individually → alert). Full suite now 130.

**Why it matters:** This validates bar *content*, not just freshness, closing the gap where the backtester could silently read a corrupt bar that passed every structural check. Running it on `etf_basket` came back content-clean (0 zero-volume, 0 extreme-jump), which is the useful confirmation that yfinance's `close` is split-adjusted for the basket. The full-DB scan earned its keep immediately: it surfaced real bad data outside the basket — SW carries 376 zero-volume bars — alongside a textbook false positive, GL's *real* −53% day on 2024-04-11 (a short-seller report, not a data error, ratio 0.469). That single false positive is the whole reason extreme-jump is informational rather than exit-1: the heuristic cannot distinguish an unadjusted split from a genuine crash, so it flags for a human to investigate instead of failing CI.

**Architectural note:** The two checks are pure functions over a `list[(date, close, volume)]` — no DB, no argparse — so they test with hand-built fixtures and the boundary behavior (`< 0.5` / `> 2.0`, strict) is pinned directly rather than inferred from a live run. `should_alert` follows the same extract-the-decision pattern as `resolve_window` from Day 22: the policy lives in a tiny pure function and `main()` just calls it. The asymmetry in the exit policy is the considered call here — unambiguous bad data (zero volume) hard-fails red, ambiguous heuristics (extreme jump) report but don't fail — so the report stays trustworthy as a CI gate without crying wolf on legitimate market events.

**Blocked on:** None.

**Next up:** Day 24 — the fit_fn seam integration that Day 22 earmarked for Day 23, pushed one slot by this content-validation work. Replace `fold_strategy = strategy` with `fold_strategy = fit_fn(train_bars)` running an Optuna sweep per train window, evaluated across the now content-validated etf_basket OOS sample to measure the overfitting tax against the fixed-strategy baseline.

**Time spent:** ~2 hours

---

## Day 22 — 2026-06-11

**Worked on:** Stabilized the data foundation before resuming Phase 2. Added a fixed `etf_basket` universe to config/universe.yaml — 17 liquid, long-history, non-return-selected ETFs (SPY, QQQ, IWM; sectors XLK/XLF/XLE/XLV/XLY/XLP/XLI/XLU/XLB; bonds TLT/IEF; GLD; international EFA/EEM). Deep-backfilled 2008-01-02 → present through the Day 21 fixed write path: ~4,639 bars/symbol, and verified on disk that every symbol has zero duplicate dates and all timestamps at midnight UTC. Hardened scripts/backfill_universe.py: added `--start`/`--end` with parse-time validation via `date.fromisoformat` (a malformed date now fails at argparse instead of deep in the fetcher) and an "--end requires --start" guard the mutually-exclusive group can't express; extracted the window-selection logic into a pure, importable `resolve_window(years, start, end)` helper that raises `ValueError` on the no-input misuse. The original `--years` path is byte-for-byte unchanged. Hardened src/data/universe.py: `load_universe` now raises on a ticker that parsed as a YAML boolean (e.g. an unquoted `ON`) rather than silently emitting `"True"`, since `bool` subclasses `int` and would slip past the existing `str()` cast. Added tests/test_backfill_window.py — 4 tests for `resolve_window` (start-only defaults end to today, start+end verbatim, years delegates to `compute_date_range`, no-input raises) — bringing the suite to 121. Confirmed by reading scripts/refresh_sp500_universe.py that the sp500 refresh does load-modify-dump (loads the full YAML, overwrites only `universes['sp500']['tickers']`, re-dumps the whole dict), so `etf_basket` and `test` survive a refresh untouched.

**Why it matters:** The strategy's trade base was one ticker. Walk-forward and the coming Optuna sweeps need enough independent trades to be statistically meaningful, on data that spans multiple regimes — so the basket goes back to 2008-01-02, through the 2008 crash, the 2020 COVID drawdown, and 2022's rate shock. ETFs were chosen over individual names specifically to sidestep survivorship bias by construction: the basket membership is fixed and the instruments don't get delisted out of the sample the way single stocks do. This is deliberately a data-foundation detour ahead of the originally-forecast Optuna seam work (now Day 23) — fit-on-train evaluation is only worth running once the OOS sample under it is broad and clean.

**Architectural note:** `resolve_window` was extracted as a pure function precisely so the window arithmetic is testable without argparse or the network — the CLI now just feeds it the three parsed values. The YAML-boolean guard in `load_universe` is a loud safety net, not a fix for a present bug: `'ON'` is correctly quoted in config today, but an unquoted edit would otherwise corrupt the universe silently. Note for backtest time: SPY and QQQ are shared across sp500/test/etf_basket and now carry 2008-onward history, while most sp500 names do not — so the sp500 universe has a per-symbol length asymmetry that downstream code must handle (e.g. via `min_bars` or per-symbol date alignment) rather than assuming uniform series lengths.

**Blocked on:** None.

**Next up:** The originally-forecast fit_fn seam integration (now Day 23) — replace `fold_strategy = strategy` with `fold_strategy = fit_fn(train_bars)` running an Optuna sweep per train window, now evaluated across the deeper, broader etf_basket OOS sample to measure the overfitting tax against the fixed-strategy baseline.

**Time spent:** ~2 hours

---

## Day 21 — 2026-06-08

**Worked on:** Discovered and fixed a timezone-duplication bug that was corrupting the entire bar dataset, then completed the planned walk-forward validator work. The bug surfaced during a sanity check on the new walk-forward CLI: SPY showed 1,848 bars for 2021-01-04 → 2026-05-08, impossible for a daily series (~1,350 trading days). Root cause: PRIMARY KEY (symbol, timestamp, timeframe) keys on the raw timestamp value, not the calendar date. Two ingest runs using different yfinance versions produced distinct UTC timestamps for the same trading date — an older version returned tz-naive pd.Timestamps, stored by the fetcher as 00:00 UTC; a newer version returned America/New_York-aware timestamps, stored as 04:00 UTC (summer/EDT) or 05:00 UTC (winter/EST). Both cleared the PK as numerically distinct keys; INSERT OR IGNORE never fired; and read_bars, which matches on CAST(timestamp AS DATE), returned both rows per affected date. SPY had 505 duplicated dates out of 1,343 unique ones — 27% of the series were spurious zero-return bars, deflating measured volatility and making annualization_factor=252 wrong (the data was effectively ~346 bars/year). Consequence stated plainly: every Sharpe computed before today, including Day 16's in-sample 0.22, was on corrupted data and is not trustworthy. The fix has two parts. Prevention: duckdb_store._bar_to_tuple now floors 1d bars to midnight UTC (ts.replace(hour=0, minute=0, second=0, microsecond=0) after the existing astimezone(utc).replace(tzinfo=None) line), so a 00:00 bar and an Eastern-midnight bar for the same date collapse to the same PK value and any future double-ingest deduplicates automatically. Regression test test_daily_bar_tz_duplicate_is_rejected: writes the same bar at 00:00 and 05:00 UTC, asserts exactly one stored row, confirmed it fails without the floor (INSERT OR IGNORE silently admitted the second row, returning 2 rows on read). Cleanup: one-time migration scripts/migrate_dedup_daily_bars.py backed up the DB, deduped on (symbol, date, timeframe) via ROW_NUMBER() OVER (PARTITION BY symbol, CAST(timestamp AS DATE), timeframe ORDER BY timestamp DESC) rn=1 — keeping the newer-ingest row — then re-stamped survivors to midnight UTC via CAST(CAST(timestamp AS DATE) AS TIMESTAMP), swapped the table atomically (CREATE staging + DROP original + RENAME). Verified: SPY 1,848 → 1,343 rows, zero dup dates store-wide, all timestamps at 00:00, SPY (date, close) sequence byte-identical to the pre-migration DISTINCT sequence, second symbol spot-check clean. On the planned work: built walk_forward_validate in src/research/walk_forward.py — generates signals over train+test bars so SMA indicators are warm at the test start, backtests only the test window, then stitches the disjoint test windows into one OOS track record (concatenated log-returns with structural index-0 zeros stripped at each seam, equity curve anchored at 1.0, OOS scalars: sharpe, total_return, max_drawdown, win_rate, n_folds_positive_sharpe, total_trades). Result is a frozen WalkForwardResult dataclass. Guards: explicit empty-splits ValueError, overlap guard rejecting shared test bars between folds. Per-fold strategy assignment is isolated to the single seam line fold_strategy = strategy for Day 22. Also pulled inline engine math into src/backtest/metrics.py (total_return, sharpe_ratio, max_drawdown, win_rate) shared by engine and validator; hardened the Sharpe zero-variance guard from == 0.0 to < 1e-12, because a constant series has residual std ~1e-18 from float rounding that slips past exact equality and produces a garbage Sharpe in the billions, ranking a loser as the best parameter. Golden-verified byte-identical on real SPY data. Built compare_walkforward.py — per-symbol per-fold table plus stitched OOS footer; header prints the actual loaded date range (bars[0]/bars[-1]) rather than the requested args.start/args.end, which would have surfaced the dup bug at a glance had it existed from day one.

**Why it matters:** First honest result on clean data: SPY, fixed SMA(50, 200), train=504 bars / test=126 bars, six folds — OOS Sharpe 0.36, total return +17.82%, 4/6 folds positive, including one −19% whipsaw fold the validator correctly surfaced without crashing. Nothing has been fitted to these windows, so this is the fixed-strategy baseline, not an overfitting-tax measurement — that comparison arrives Day 22 when Optuna fitting runs inside each train window. The data bug matters beyond the immediate numbers: every finding from Days 16–18 was on corrupted data, and the bug was invisible in the output because the CLI printed the requested date range rather than the actual loaded bar count. Closing both the root cause (write-path floor) and the observability gap (header now shows actual loaded range) means the same class of corruption cannot hide again.

**Architectural note:** The write-path floor is the single most important design property of today's fix: because _bar_to_tuple is the only entry point for every bar this project will ever store, any future fetcher inherits the dedup guarantee without modification — one choke point, one fix. The seam line pattern (fold_strategy = strategy as the single isolated assignment in the fold loop) keeps the validator flat and makes Day 22's optimizer integration a genuine one-line substitution rather than a structural refactor. Each fold re-enters flat at its first test bar, so one boundary bar's return per fold is not captured; this is a small conservative understatement that does not distort cross-strategy ranking and is preferable to the complexity of mid-fold position hand-off. Very small test windows yield degenerate OOS metrics rather than crashes — acceptable for a research tool where the caller controls window sizes.

**Blocked on:** None.

**Next up:** Day 22 — fit_fn seam integration: replace fold_strategy = strategy with fold_strategy = fit_fn(train_bars) where fit_fn runs an Optuna sweep over the train window and returns an optimized SMACrossoverStrategy. First genuine out-of-sample test: fit on train, evaluate on test, aggregate OOS Sharpe across folds, compare to today's fixed-strategy baseline to measure the overfitting tax.

**Time spent:** ~3 hours

---

## Day 20 — 2026-06-04

**Worked on:** Phase 2 — built the walk-forward splitter, the foundation of out-of-sample validation. Created src/research/walk_forward.py with one pure function, walk_forward_splits(bars, train_size, test_size, step=None) -> list[tuple[list[OHLCVBar], list[OHLCVBar]]], that slices a time-ordered bar list into successive (train, test) window pairs using a rolling/sliding window. Each fold's test window begins on the exact bar immediately after its train window (strict adjacency — no gap, no overlap), step defaults to test_size so test windows tile the post-training timeline exactly once, and leftover tail bars too few for a complete fold are dropped rather than emitted as a short fold (keeps every fold size-identical so fold-aggregated metrics stay comparable). Four validation guards raise before any slicing (train_size < 1, test_size < 1, step < 1 when provided, and len(bars) < train_size + test_size — the "not enough data for one fold" case, which raises rather than silently returning [], mirroring run_universe's all-skipped guard). The function is pure: no I/O, depends only on src.data.schema.OHLCVBar, assumes ascending-time input (guaranteed upstream by read_bars) and deliberately does not re-sort. Added tests/test_walk_forward.py with 11 tests built on a make_bars(n) helper that encodes each bar's index into its close value, so tests verify which bars land in which window by reading close rather than comparing timestamps. Tests cover: fold count on clean division, exact window sizes, strict train/test adjacency (the lookahead guard), no test-window overlap under default step, step-advance semantics, custom step > test_size, the four validation errors, and the exact-minimum-bars boundary (15 bars = train 10 + test 5 yields exactly one fold, proving the <= loop condition is inclusive). Test count grew from 80 to 91.

**Why it matters:** This is the overfitting defense the Day 16-18 research arc made concrete. Day 16 found a candidate winner on one symbol; Day 17 showed it was symbol-specific noise; Day 18 confirmed no SMA crossover has positive average edge on large-caps. All of those measured in-sample. Walk-forward is the machinery that will measure genuinely out-of-sample performance — train on one window, evaluate on the next, never let the strategy see its test data. The splitter is deliberately scoped to window generation alone: the validator that runs strategies through the folds and aggregates out-of-sample metrics is a later day. Splitting the boundary arithmetic (where an off-by-one silently reintroduces lookahead bias) into its own independently-tested unit means that when the validator is built, the windows underneath it are already proven correct.

**Architectural note:** The splitter has no dependency on the runner, the backtester, or DuckDB — it transforms one list into a list of sub-list pairs and nothing more. That isolation is why its 11 tests run in 0.02 seconds with no fixtures or real data. The adjacency property — test window starts at exactly start + train_size — is the single line where correctness lives, because a one-bar overlap there would leak a training bar into evaluation, the exact failure walk-forward exists to prevent; the test suite asserts it on bar position (via the close-encodes-index trick), not just timestamp ordering, so a gap or overlap cannot pass silently. The function works in bar-count terms rather than calendar terms (train_size=504 bars, not "2 years"), which is simpler and deterministic; calendar-aware splitting is a later refinement only if needed.

**Blocked on:** None.

**Next up:** The walk-forward validator — run a strategy (or the full parameter grid) through each fold via BacktestRunner, training/selecting on the train window and recording performance only on the test window, then aggregate out-of-sample metrics across folds. It reuses the splitter built today plus the existing runner and cli_common.load_bars_for_symbols, so it is orchestration over proven parts rather than new plumbing.

**Time spent:** 45 minutes

---

## Day 19 — 2026-06-04

**Worked on:** Phase 2 opener — a pure refactor with zero behavior change. Extracted the duplicated symbol-list-building and DuckDB read-loop logic that compare_universe.py and compare_matrix.py both carried into a new shared module src/research/cli_common.py with two pure functions: build_symbol_list(symbols_csv, universe, limit) -> list[str] (the --symbols CSV vs --universe/--limit branch, with whitespace and empty-token cleanup) and load_bars_for_symbols(symbols, start, end) -> dict[str, list[OHLCVBar]] (opens DuckDB once for the whole batch, skips missing symbols with log.warning + continue, preserves insertion order). Both take plain values rather than an argparse Namespace so they are unit-testable without a parser, and neither prints nor owns its error guards — the CLIs keep their own wording. Rewired both CLIs to call the helpers: compare_universe.py 375 to 336 lines, compare_matrix.py 557 to 501 lines, with DuckDBStore and load_universe imports removed (plus logging in matrix) once unused. compare_strategies.py was deliberately left untouched — it is single-symbol and shares no surface. Added tests/test_cli_common.py with 10 tests (6 pure for build_symbol_list, 4 DuckDB-backed for load_bars_for_symbols: known symbols, missing-symbol skip, input-order preservation, all-missing empty dict). Test count grew from 70 to 80.

**Why it matters:** The extraction was deferred on purpose until three call sites existed — at two it would have been premature abstraction. With three research CLIs now sharing the same load skeleton, the helper pays for itself and gives the upcoming walk-forward validator a tested bar-loading primitive to reuse instead of a fourth copy. The refactor was verified by capturing each CLI's stdout to golden files BEFORE any change, then diffing after every edit: all three CLIs produced byte-for-byte identical output, and all 80 tests stayed green. That discipline — golden-output capture, scope-before-cut, one CLI at a time, diff after each — is as much the point of the day as the code.

**Architectural note:** cli_common depends only on the data layer (duckdb_store, universe, schema), never on the sibling CLIs, so there is no circular import. The two functions deliberately stop short of owning user-facing output: each CLI still prints its own "Loading..."/"Loaded..." lines and its own empty-symbols / empty-bars error guards, because the wording diverges per CLI and belongs to the tool that knows its own context. This keeps the helper a pure data primitive (list in, dict out) and preserves the project-wide "raw data internally, formatting only at the edge" rule. A pre-existing minor quirk in compare_matrix (it prints the dimensions line before the empty-symbols guard) was left untouched — fixing it would be a behavior change and belongs in a separate commit.

**Blocked on:** None.

**Next up:** The walk-forward validator. The runner's stateless design plus the now-extracted load_bars_for_symbols helper mean the validator can slice each symbol's bars into train/test windows and call the existing runner methods inside a sliding-window loop, with no re-plumbing of data loading. This is the overfitting defense the Day 16-18 findings made concrete.

**Time spent:** 45 minutes

---

## Day 18 — 2026-06-02

**Worked on:** Extended src/research/runner.py with BacktestRunner.run_matrix(), the cross-product of run_many and run_universe: it takes bars_by_symbol (dict) plus a list of strategies and runs every (symbol, strategy) combination, returning a list[BacktestResult] in symbol-major, strategy-minor order. Same validation pattern as the other two methods (non-empty dict, non-empty strategies, isinstance check naming the bad index, non-empty bars per symbol, min_bars >= 1). min_bars filters at the SYMBOL level, not the per-cell level: a symbol below the threshold is skipped for all strategies, keeping the downstream pivot table rectangular rather than ragged. The all-skipped guard raises ValueError reporting the largest available bar count. The degenerate cases (one strategy → run_universe behavior, one symbol → run_many behavior) fall out of the nested loop without special-casing. Built src/research/compare_matrix.py — the third research CLI (python -m src.research.compare_matrix) — with a hardcoded six-pair SMA grid (same as compare_strategies.py for cross-tool comparability), --universe/--symbols/--limit/--start/--end/--metric flags, and a pivot-table view (symbols as rows, strategies as columns, chosen metric in cells) plus a strategy-ranking summary by mean metric. min_bars is derived from max(slow)+1 so every symbol contributes a full row or none. Pivot columns are reindexed to grid order (overriding pandas' alphabetical default); rows sort by row-mean with max_drawdown auto-inverted. Added 8 unit tests for run_matrix (empty-dict, empty-strategies, strategy-type, empty-bars-value, cross-product size, symbol-major ordering across all six positions, symbol-level skip, all-skipped raise). Test count grew from 62 to 70; suite runs in ~1.6 seconds.

**Why it matters:** This is the first tool that surfaces strategy-symbol fit, which neither per-axis CLI could. The 25-symbol Sharpe matrix produced three findings. First, SMA(50, 200) ranked best for the third independent time: highest mean Sharpe (-0.01, essentially flat) and the most symbols with positive Sharpe (14/25), while Day 16's SMA(10, 50) "winner" ranked second-to-last on mean Sharpe (-0.25) — the overfitting verdict from Day 17 holds under the fuller view. Second, EVERY strategy had negative mean Sharpe across the 25 large-caps: no simple SMA crossover configuration produced positive average risk-adjusted edge on individual stocks over 2021-2026. The "best" strategy is merely the least-bad. This is the honest ceiling of naive trend-following, and the framework reported it without flattery. Third, the best strategy varies by symbol — QQQ peaks at SMA(50, 200)=1.52, NVDA at SMA(50, 100)=1.00, while AIG and ABNB are negative across all six. Some symbols are structurally trend-friendly and others trend-hostile, which motivates per-symbol parameter selection as a Phase 2 research direction.

**Architectural note:** run_matrix completes the 2x2 of research axes: one/many symbols crossed with one/many strategies. The three CLIs (compare_strategies, compare_universe, compare_matrix) remain separate entry points rather than one tool with mode flags — each answers one well-defined question. With three call sites now sharing the same DuckDB-read / runner-call / compare-or-pivot / format-and-print skeleton, the shared core is finally worth extracting; that refactor (a common load-and-run helper) is now justified where at two call sites it would have been premature. The matrix CLI's pivot/summary display logic is the one genuinely new piece and stays local to compare_matrix.py.

**Blocked on:** None. VS Code's editor buffer again drifted from disk on the test file (the recurring 9+/M badge), but git add stages from disk so the commit captured the correct clean file; verified via git diff --cached --stat showing 141 insertions matching the 8 new tests.

**Next up:** Phase 2 begins — parameter optimization and walk-forward validation. The matrix is the data structure Optuna-driven sweeps will populate (hundreds of parameter combinations × symbols), and walk-forward validation will defend against the overfitting this week's findings made concrete. First Phase 2 task: extract the shared load-and-run helper now that three CLIs justify it, then build the walk-forward splitter on top of the runner.

**Time spent:** - 30mins

---

## Day 17 — 2026-05-26

**Worked on:** Extended src/research/runner.py with a new BacktestRunner.run_universe() method that runs one strategy against many symbols and returns a list[BacktestResult] in dict insertion order. Same dependency-injected Backtester and isinstance validation as run_many(), but the iteration axis is symbols instead of strategies. Each result's strategy_name is composed as f"{strategy.name} on {symbol}" — injected via the strategy_name kwarg on Backtester.run() rather than mutating the strategy object — so the per-symbol label survives into compare() output without callers having to track it separately. Added an optional min_bars guard so symbols too short for the strategy's slow-window warmup are silently skipped (continue inside the loop), with a ValueError raised after the loop if every symbol ends up skipped (silent empty result would mask a config mistake). Built src/research/compare_universe.py — a second CLI entry point (python -m src.research.compare_universe) with --universe, --symbols, --limit, --fast, --slow, --start, --end, --sort flags. The CLI reads bars from DuckDB (single context-managed read, silently skipping symbols with no DB data), passes them to runner.run_universe with min_bars=slow+1, and prints the same sorted comparison table from Day 16 plus an aggregate footer counting positive Sharpe, Sharpe > 0.5, and positive return. Added 8 unit tests for run_universe covering empty-dict validation, strategy-type isinstance check, empty-bars-value validation, result-per-symbol invariant, dict insertion order preservation, min_bars skip behavior, all-symbols-skipped raise, and result type. Test count grew from 54 to 62; full suite runs in ~2 seconds.

**Why it matters:** Day 16 produced a "SMA(10, 50) beats SMA(50, 200) on SPY" result. That was suspicious — a single 5-year window on a single symbol can find any winner by chance. Today's multi-symbol axis was the first real check, and it inverted the conclusion: across the first 25 S&P 500 symbols, SMA(50, 200) had 14/25 positive Sharpe and 5/25 Sharpe > 0.5; SMA(10, 50) had only 9/25 positive Sharpe and 2/25 Sharpe > 0.5. The faster crossover that "won" on SPY underperformed on the broader universe — classic single-symbol overfitting that the framework now catches automatically. SMA(10, 50) also trades 5-10x more (e.g. 36 trades vs 13 on ABNB), which would compound into a transaction-cost drag the no-cost backtester currently hides. This is the smallest version of out-of-sample testing the system supports; formal walk-forward comes in Phase 2.

**Architectural note:** run_universe and run_many are deliberately separate methods, not one method with a "mode" flag. Each maps to a distinct research question ("which strategy is best on this symbol?" vs "which symbols does this strategy work on?"), and a unified API would force callers to construct degenerate single-element lists for the dimension they're not sweeping. Same reasoning applies to compare_strategies.py and compare_universe.py as separate CLIs rather than one with a mode toggle. Day 18 will add a third method (run_matrix) and a third CLI (compare_matrix.py) for the strategy × symbol cross-product — at three call sites the shared core will be ready to extract; at two it would be premature.

**Blocked on:** None. Tooling friction during the day (VS Code save behavior corrupted the test file mid-edit, requiring a git checkout and a clean re-paste of the Claude Code generated tests) cost time but did not affect the final artifact.

**Next up:** Day 18 — strategy × symbol cross-product matrix (run_matrix on BacktestRunner, compare_matrix.py CLI). Same primitives, one more layer of orchestration. This is the building block Phase 2's Optuna parameter optimization will feed on.

**Time spent:** - 1 hour

---

## Day 16 — 2026-05-15

**Worked on:** First research tooling. Built src/research/ package with two new files: src/research/runner.py defining BacktestRunner with run_many() (executes N strategies against one symbol's bars, returns ordered list[BacktestResult]) and a static compare() method (aggregates results into a pandas DataFrame, sorted by configurable metric). Built src/research/compare_strategies.py — a CLI entry point (python -m src.research.compare_strategies) that sweeps 6 SMA crossover parameter combinations on SPY by default, with --symbol/--start/--end/--sort flags. Output is a formatted terminal table with the best strategy on top, including a header line "Comparison sorted by X (descending)" and auto-inverted direction for max_drawdown. Added 10 unit tests covering runner validation, result ordering, DataFrame columns, sort direction, and synthetic-BacktestResult construction (make_result factory bypasses the engine so compare()'s sort logic is tested in isolation). Test count grew from 44 to 54.

**Why it matters:** this is the smallest unit of real research — comparing alternatives. Before today the only way to compare two strategies was to run two `python -c` scripts and eyeball the outputs. Now there's one command that runs the comparison, formats the table, and tells you which configuration wins on Sharpe (or any other metric). The runner is the foundation for Phase 2 parameter optimization: instead of 6 hand-picked SMA combos, the same code will eventually consume hundreds of (fast, slow) combinations generated by Optuna, or hundreds of stocks from the S&P 500 universe. Same shape, more rows.

**First real research finding:** on SPY 2021-2026, SMA(10, 50) beat canonical SMA(50, 200) on both total return (+38.34% vs +25.22%) AND max drawdown (20.83% vs 21.13%). Shorter-horizon variant won on a single window — but a single 5-year window is statistically thin, so this is a hypothesis to stress-test, not a conclusion. Day 17 (multi-symbol) and Phase 2 (walk-forward validation) will determine whether the finding holds out-of-sample.

**Architectural detail worth noting:** BacktestRunner.compare() is a @staticmethod that returns a DataFrame with raw numeric values; the CLI does its own percentage formatting on a copy. This keeps the runner's output usable for downstream consumers (CSV export, plotting, optimization loops) while letting the CLI present numbers human-readably. Same principle as the indicator/strategy/backtester separation — each layer produces a clean output and lets the next layer decide how to consume it.

**Blocked on / Bugs:** None. One thing worth flagging for future-me: the parameter grid is currently hardcoded in compare_strategies.py. That's fine for Day 16's "smallest research script" goal but won't scale to Phase 2 parameter sweeps. Day 17+ will probably extract grid generation into a separate config or generator function.

**Next up:** Day 17 — multi-symbol comparison. Same runner, but the dimension being compared shifts from "one symbol × many strategies" to "many symbols × one strategy." This is the first step toward portfolio-level analysis where you'll see which sectors/stocks the strategy actually works on.

**Time spent:** — 20 minutes

---


## Day 15 — 2026-05-14

**Worked on:** First backtester. Built src/backtest/ package with two new files: src/backtest/result.py defining frozen dataclasses Trade (single round-trip record) and BacktestResult (equity curve, trades list, and precomputed metrics), and src/backtest/engine.py defining the Backtester class. The engine implements vectorized next-bar-execution simulation: signals[i-1] earns the return from bar i-1 to bar i, which makes lookahead bias structurally impossible. Computes equity curve via cumsum of log returns then exp, extracts trades from signal transitions (handling 0→nonzero entries, nonzero→0 exits, sign flips, and end-of-data force-close via a factored _make_trade helper), and pre-computes total return, annualized Sharpe ratio (252 trading days, ddof=1), max drawdown, win rate, and trade count. No transaction costs or position sizing — that's Phase 3 work; this measures raw strategy edge. Smoke-tested on 5 years of SPY SMA(50,200) signals from Day 14: +25.22% total return, 0.22 Sharpe, 21.13% max drawdown, 57.1% win rate, 7 trades. Underperforms SPY buy-and-hold (~+70%) as trend-followers do in choppy bull markets — this is the truth-telling layer working correctly. Added 15 unit tests including the critical test_no_lookahead_bias guard. Test count grew from 29 to 44.

**Why it matters:** this is the layer that judges strategies. Before today, the codebase could only produce strategies that LOOKED sensible. The backtester is what tells you whether they would actually make money — and in this case, told you that the canonical SMA crossover on SPY alone is mediocre. That's a useful finding: it sets the realistic baseline against which future strategies (RSI mean reversion, multi-factor ensembles, ML-based) will be compared. The result dataclasses being frozen means historical backtest results can't be silently mutated. The next-bar execution model means accidentally writing a lookahead-biased strategy is now structurally hard.

**Blocked on / Bugs:** One sharp edge caught and fixed during testing: numpy's std(ddof=1) is undefined for samples of size <= 1 and produces NaN, which the original `if std_return == 0.0` guard didn't catch. This surfaced as 8 RuntimeWarnings when running the 2-bar unit tests. Fixed by short-circuiting on len(active_returns) < 2 before computing std. Lesson: sample-size guards belong before the math, not after.

**Next up:** Day 16 — strategy comparison and reporting. Build a small CLI that runs multiple strategies (or one strategy across multiple parameter combos) and produces a comparison table. Foundation for the parameter optimization work in Phase 2.

**Time spent:** —

---


## Day 14 — 2026-05-10

**Worked on:** First strategy. Created src/strategies/ package with two new files: src/strategies/base.py defining the Strategy abstract base class (mirrors the architectural pattern of src/brokers/base.py) and three module-level signal constants SIGNAL_LONG (+1), SIGNAL_FLAT (0), SIGNAL_SHORT (-1). Created src/strategies/sma_crossover.py with SMACrossoverStrategy(fast_window=50, slow_window=200) implementing the canonical Golden Cross / Death Cross rule. The strategy returns a numpy int8 array aligned 1:1 with input bars — SIGNAL_FLAT during warmup (positions where either SMA is NaN), SIGNAL_LONG when fast > slow, SIGNAL_SHORT when fast < slow. Smoke-tested on 5 years of SPY data (1848 bars, 2021-01-04 to 2026-05-08): 69.8% LONG, 19.5% SHORT, 10.8% FLAT, 7 regime transitions matching real market history (2022 bear market, recent April 2026 selloff). Added 11 unit tests in tests/test_strategies.py covering ABC enforcement, parameter validation, name property, length and dtype contracts, warmup behavior, uptrend/downtrend signals, qualitative crossover detection, and input validation. Test count grew from 18 to 29.

**Why it matters:** this is the conceptual transition from features to decisions. Up to today the code summarized prices into numbers; now it emits opinions about them. The Strategy ABC is what every future strategy will subclass — RSI mean reversion, momentum, pairs trading, ML-based — so getting the contract right today (numpy int8 output, length alignment, warmup-as-FLAT, pure-function semantics) saves the entire Phase 2+ refactor. The signal-as-position (not signal-as-trade) design lets the same strategy work for backtest and live execution without special cases — the backtester will derive trades by diffing consecutive positions.

**Blocked on / Bugs:** None. One discovery during smoke testing: the initial SPY data in DuckDB only had ~2 years of history because the Day 11 backfill skipped pre-existing 2024 data via INSERT OR IGNORE. Re-fetched explicitly with fetch_daily('SPY', '2021-01-01', '2026-05-10') to get the full 1343 bars (1848 with overlap from neighboring stocks). This is a documented limitation of the idempotent backfill approach — fresh universes get full history, but symbols pre-seeded with partial history don't get backfilled to the requested years. Acceptable trade-off; the explicit re-fetch is a clean workaround.

**Next up:** Day 15 — first backtester. Build src/backtest/ that takes (bars, signals) and simulates what the strategy would have earned over the historical period. Compute realistic metrics: total return, Sharpe ratio, max drawdown, win rate. This is where strategies prove (or disprove) edge.

**Time spent:** — 30 minutes

---


## Day 13 — 2026-05-10

**Worked on:** First technical indicators. Created src/features/ package with src/features/indicators.py containing three vectorized functions: sma (simple moving average via pandas.rolling), log_returns (np.log of close ratio), and rsi (Wilder's RSI with explicit two-phase implementation — simple-mean seed at index period, then the recursive smoothing). All three operate on lists of OHLCVBar and return numpy arrays aligned 1:1 with input — NaN values where lookback is insufficient, never silently dropped. Single private helper _bars_to_close_series centralizes the OHLCVBar → pandas conversion. Added 10 unit tests in tests/test_indicators.py covering known values, edge cases (all-gains → 100, all-losses → 0), input validation (empty bars, invalid window/period), and length-alignment guarantees. Smoke-tested against real SPY data from DuckDB — SMA-50, RSI-14, and log returns all compute in milliseconds for 252 bars. Smoke-tested current SPY (2025-01-01 to today): RSI-14 = 75.47 (overbought), SMA-50 = 682.55.

**Why it matters:** Every strategy and every ML feature in Phase 2+ ultimately reduces to numerical arrays computed from OHLCV bars. Today's three functions are the simplest examples of the pattern — input contract (list of OHLCVBar), output contract (np.ndarray of len(bars) with NaN-padded prefix), implementation strategy (pandas internals, numpy boundary). Every future indicator follows the same shape.

**Blocked on / Bugs:** One subtle issue caught and fixed during development. The first RSI implementation used pandas.ewm(alpha=1/period, adjust=False), which runs the Wilder recursion from the first observation rather than seeding with a simple mean of the first `period` deltas. The two methods converge after ~3× period bars but diverge noticeably on the very first emitted RSI values — meaning values would not match TradingView, ta-lib, Bloomberg, or StockCharts on short series. Replaced with explicit two-phase implementation: simple arithmetic mean of first `period` gain/loss values at index `period`, then Wilder recursion for all subsequent indices. Verified against canonical Wilder reference series — agreement within 0.006 of published values [70.46, 66.25, 66.48, 69.35, 66.30, 57.92]. Test suite expanded from 8 to 18 tests.

**Next up:** Day 14 — first signal generation. Build src/strategies/ with a crossover strategy (e.g. SMA(50) vs SMA(200)) that takes bars, runs indicators, and emits buy/sell/hold signals as a numpy array aligned to the input bars.

**Time spent:** — 1 hour

---


## Day 12 — 2026-05-07

**Worked on:** Made the data layer production-ready for daily operations. Three major additions: (1) DuckDBStore.last_timestamp(symbol, timeframe="1d") — returns the most recent bar timestamp via SELECT MAX, with the same UTC re-attach guard as _tuple_to_bar. Powers incremental updates. (2) scripts/update_universe.py — daily incremental update CLI with --universe and --lookback-days flags. Per-symbol logic: if no stored data, skip with a hint to run backfill; otherwise compute start = max(last_ts.date() - lookback_days, 2010-01-01) and fetch only the tail. Tested on test universe (5/5 already current after re-runs proving double idempotency) and sp500 (503/503 in 68 seconds, 1500 new bars — ~17x speedup vs the 20-minute backfill). (3) scripts/data_health.py — read-only diagnostic that opens DuckDB with read_only=True, reports total bars, distinct symbols, date range, file size, stale symbols (sorted by most-stale-first, capped at 20 unless --verbose), short-history symbols (<252 bars, capped at 10), and (with --universe filter) symbols missing from DB. Exit 0 if clean, exit 1 if stale or missing — cron-friendly. Verified all four code paths via smoke tests. (4) Added 1 unit test (test_last_timestamp_returns_max) — pure unit, no network, uses tmp_path.

**Why it matters:** Backfill is a 20-minute one-time bootstrap; daily updates need to be 1-minute or cron jobs become impractical. Health diagnostics turn silent corruption (stale or missing tickers) into loud, actionable warnings before they poison a backtest.

**Blocked on / Bugs:** Discovered and fixed a latent timezone bug in DuckDBStore. The DuckDB Python driver was converting tz-aware datetimes to LOCAL time when binding to TIMESTAMP columns, then stripping tzinfo. The pipeline only worked because yfinance daily bars at 05:00 UTC happen to land on the correct calendar date when shifted to US/Eastern (UTC-5). This would have silently corrupted data on any machine in another timezone, or for any non-yfinance source, or for intraday bars. Fix: _bar_to_tuple now does bar.timestamp.astimezone(timezone.utc).replace(tzinfo=None) before binding, so the stored value is unambiguously UTC. The test_last_timestamp_returns_max test caught this — it's exactly the kind of bug a unit test with non-aligned timestamps surfaces. Existing 627k bars in DuckDB are unaffected as calendar dates (which is all daily bars need); future intraday data will be timezone-correct from day one.

**Documented limitation:** A delisted symbol's last_timestamp will keep growing stale silently — the update script will compute a start date relative to it and yfinance will return nothing or an error. The script handles this gracefully (skip + log warning) but doesn't actively detect "this symbol is dead." Will fix in Phase 2+ with point-in-time universe membership.

**Next up:** Day 13 — first technical indicators (SMA, RSI, returns). Build src/features/ with vectorized pandas/numpy implementations operating on lists of OHLCVBar.

**Time spent:** — 1 hour

---


## Day 11 — 2026-05-04

**Worked on:** Scaled the data layer to the S&P 500. Created config/universe.yaml with named universes ("test" with 5 tickers, "sp500" auto-populated). Built src/data/universe.py with load_universe() and list_universes() helpers. Built scripts/refresh_sp500_universe.py — scrapes Wikipedia constituents via pandas.read_html (with browser User-Agent header to bypass 403 and io.StringIO wrapping to avoid pandas printing HTML to stdout), normalizes ticker dots to dashes for Yahoo (BRK.B → BRK-B), updates the YAML. Added YFinanceFetcher.fetch_daily_batch() with on_error=skip|raise — partial failures don't kill the run. Built scripts/backfill_universe.py — argparse CLI taking --universe and --years, with tqdm progress bar and per-symbol error isolation. Tested end-to-end: backfilled 5 years of S&P 500 daily bars (~626k rows, 503/503 symbols, ~20 minutes). Verified idempotency at scale — re-running the test backfill produces 0 duplicate inserts. Added 3 unit tests for the universe loader (pure, no network).

**Why it matters:** The bot now has 5 years of daily price history for every S&P 500 stock stored locally — all 500+ companies, downloaded once and ready to query instantly. Before this, any strategy test would have to reach out to the internet every single time, which is slow and breaks if the data provider is unavailable. Now the bot can test a trading idea across hundreds of stocks in seconds, using data that's already on disk.

**Blocked on / Bugs:** Two pandas-related issues fixed: (1) Wikipedia returned 403 with default urllib User-Agent — fixed by passing a browser UA via urllib.request.Request. (2) pd.read_html(html_string) was printing the raw HTML to stdout as a side effect — fixed by wrapping in io.StringIO. Also needed to add lxml dependency (pandas.read_html requires it).

**Next up:** Day 12 — daily incremental update script (only fetch the latest N days, not full history), gap detection (find missing dates per symbol), and a "data health" diagnostic script.

**Time spent:** —

---


## 2026-05-01 — Day 10
**Worked on:** Started Phase 1 (data infrastructure). Created `src/data/schema.py` with frozen `OHLCVBar` dataclass and `CREATE_TABLE_SQL`. Built `src/data/yfinance_fetcher.py` — thin adapter around `yfinance.Ticker.history` that returns `list[OHLCVBar]` sorted ascending. Built `src/data/duckdb_store.py` with `INSERT OR IGNORE`-based `write_bars` (idempotent), `read_bars` range query, and context-manager support. Added `tests/test_data_layer.py` with three integration tests covering fetcher output, roundtrip persistence, and idempotency. Verified end-to-end: fetched 252 SPY daily bars for 2024, wrote to DuckDB, re-ran the pipeline and confirmed 0 duplicate rows on second run.
**Why it matters:** Before this, the bot had no memory of what prices did in the past. This day gave it a database — a local file that stores the open, high, low, close, and volume for any stock, for any day. Think of it as building the library the bot will read before making any decision. The design is also "safe to re-run": if the daily download job runs twice by accident, it won't create duplicate entries or corrupt the numbers.
**Blocked on / Bugs:** None.
**Next up:** Day 11 — extend fetcher to support batch fetching across the S&P 500, add a "universe" config file listing tickers, build a backfill script that ingests N years of history for the full universe.
**Time spent:** —

---

## 2026-04-29 — Day 9
**Worked on:** Refactored all four operational scripts (`first_order.py`, `limit_order.py`, `cancel_order.py`, `order_history.py`) to use the `AlpacaBroker` abstraction. Removed every `from alpaca.*` import from `scripts/` — verified with `grep -rn "from alpaca" scripts/` returning empty. Added `Broker.get_latest_price()` abstract method and Alpaca implementation to eliminate the temporary `_data` attribute leak — `grep -rn "broker._data" scripts/` also returns empty. Status comparisons now use plain strings (`"filled"`, `"canceled"`) instead of alpaca's `OrderStatus` enum. Combined script line count dropped roughly 50% across the four files. Confirmed all four scripts produce identical user-facing output. Smoke test still passes.
**Why it matters:** This confirmed that none of the trading scripts need to know they're using Alpaca anymore — they just say "place this order" and the broker layer handles the rest. That means if we ever want to switch to a different broker, only one file changes and every strategy works without modification. It also means we can plug in a fake "paper broker" for testing strategies without any internet connection at all.
**Blocked on / Bugs:** None.
**Next up:** Day 10 — start Phase 1 of the roadmap. Build `src/data/` layer: yfinance + Alpaca historical data fetcher, DuckDB schema, basic OHLCV storage and retrieval.
**Time spent:** — 1 hour

---

## 2026-04-27 — Day 8
**Worked on:** Built the broker abstraction layer. Created `src/brokers/base.py` with the `Broker` ABC plus typed dataclasses (`OrderRequest`, `OrderResult`, `AccountSnapshot`) and string Enums (`OrderSide`, `OrderType`, `TimeInForce`). Built `src/brokers/alpaca_broker.py` implementing the full interface against alpaca-py, with a hard assertion preventing `paper=False` and a `from_env()` classmethod. Added `tests/test_alpaca_broker.py` as a read-only integration smoke test — 1 passed. Added `pythonpath` and `integration` marker to `pyproject.toml`, created `conftest.py` at repo root for reliable imports. Refactored `scripts/hello_alpaca.py` to use `AlpacaBroker` — script shrank from 33 lines to 29 lines, all SDK code now behind the abstraction.
**Why it matters:** This is the foundation that everything else builds on. The bot now speaks a common language for placing orders — it says "buy 10 shares of SPY" and doesn't care how that gets executed. Alpaca is just one possible answer. This makes the bot portable: swap in a different broker, a simulator, or a backtester, and the strategies don't change at all. Without this layer, every strategy would be glued to Alpaca's specific code and impossible to test offline.
**Blocked on / Bugs:** None.
**Next up:** Day 9 — refactor `first_order.py`, `limit_order.py`, `cancel_order.py`, and `order_history.py` to use `AlpacaBroker`.
**Time spent:** — 30 mins

---

## 2026-04-26 — Day 7
**Worked on:** Built `scripts/limit_order.py` (submits an intentionally non-marketable limit BUY 20% below market, GTC, with a paranoia assertion preventing accidental marketable orders; saves order ID to `logs/last_limit_order_id.txt` for handoff). Built `scripts/cancel_order.py` (cancels by ID from CLI arg or from the handoff file, polls for terminal state up to 10 s). Built `scripts/order_history.py` (fetches last 50 orders across all statuses, prints fixed-width table with fill price and limit price columns, plus a grouped summary line). Verified full lifecycle end-to-end: limit order placed at $571.17, cancelled successfully, order history table confirmed `ACCEPTED: 1 | CANCELED: 1`.
**Why it matters:** Market orders just buy at whatever price is available — limit orders let the bot say "only buy if the price drops to X." That's essential for any real strategy. This day also added the ability to cancel an order that hasn't filled yet, and to look up a full history of what the bot has done. Without that history, there's no way to know if a trade actually went through or why it didn't.
**Blocked on / Bugs:** None.
**Next up:** Day 8 — design the broker abstraction layer in `src/brokers/`. Refactor shared credential loading and paper-account safety check out of scripts into a reusable module.
**Time spent:** — 30 mins

---

## 2026-04-25 — Day 6
**Worked on:** Built `scripts/first_order.py` with three safety checks: `paper=True` hardcoded (must never be changed), PA account prefix verification before any order is submitted, and hardcoded `SYMBOL`/`QTY` constants. Added dry-run preview with live SPY price fetch via `StockHistoricalDataClient`, market-hours detection with next-open timestamp, and double confirmation (`"yes"` typed explicitly) before submitting. Submitted first paper order — 1 share of SPY at $713.96, order ID `c5559e4b-bf39-42b4-9658-1e9da0639004`, status ACCEPTED, queued for Monday open. Verified order appeared correctly in Alpaca dashboard. Rotated API keys after accidental exposure in screenshot.
**Why it matters:** This was the first proof that the bot can actually do the one thing it exists to do — place a trade. Everything before this was setup; this was the moment it became real. The multiple safety checks (paper-mode lock, account verification, typing "yes" twice) are deliberate: one mistaken click in a live trading system can cost real money instantly, so the guards have to be there from the very first order.
**Blocked on / Bugs:** API key accidentally visible in a dashboard screenshot — immediately regenerated keys and updated `.env`.
**Next up:** Day 7 — limit orders, order cancellation, querying order history.
**Time spent:** — 1 hour 

---

## 2026-04-25 — Days 1–5
**Worked on:** Project initialization — pyproject.toml, uv-managed virtual environment, folder scaffold (src/, tests/, notebooks/, scripts/, data/, logs/), .gitignore, .env.example, README.md. Alpaca paper account created and connected; $200k buying power confirmed. GitHub repo initialized and remote connected.
**Why it matters:** Getting the foundation right means not having to redo it later. Keeping API keys out of the code means they can never accidentally end up on GitHub. Separating strategy code from scripts and data means the project stays navigable as it grows. These decisions feel invisible when they're done right — and extremely painful to fix after the fact when they're skipped.
**Blocked on / Bugs:** None.
**Next up:** First commit. Begin Phase 1 — data ingestion layer (Alpaca bar fetch, DuckDB schema, yfinance backfill).
**Time spent:** — 30 minutes

