# DEV_LOG — quant-trader

Running log of development work. Most recent entry first.

---

## Day 55

**Worked on:**
- Factored the single-date rotation weight logic out of rotation_backtest into
  single_date_weights(bars_by_symbol, as_of_date, lookback, top_n, price_field,
  hold_when_all_negative) -> dict[str,float] in cross_sectional.py (commit "Factor
  single_date_weights seam out of rotation_backtest (behavior-preserving)").  It composes the
  existing trailing_returns_at + rank_by_trailing_return primitives and applies the
  equal-weight 1/top_n rule (rule b); rotation_backtest now calls it at the decision date d_k
  and holds NO weight arithmetic.  The tr/held locals are retained in the loop because held
  still feeds STEP 3 (the per-symbol return-array loop at line 362); only the weight-birth
  line moved into the shared function. [+1 test]
- Added test_rotation_backtest_goldenmaster_equivalence: a golden-master array captured from
  the pre-refactor code on a fixed 4-symbol / 8-month-end synthetic fixture (lookback=3,
  top_n=2), asserted equal post-refactor via np.testing.assert_allclose(rtol=0, atol=1e-12)
  plus shape/dtype fingerprints.  280 non-integration green (was 279).

**Why it matters:**
- This is the shared seam guaranteeing the future live-target path and the validated backtest
  compute target weights with byte-identical logic - they call ONE function, so the strategy
  cannot silently drift between research and live.  Proven behavior-preserving: the 1/top_n
  literal now exists in exactly ONE place (verified by grep - one hit in cross_sectional.py,
  zero in portfolio.py) and rotation_backtest's output is bit-for-bit unchanged on the golden
  fixture.

**Architectural note:**
- Pure extraction only - rotation_backtest's returned array is identical for all inputs;
  single_date_weights does zero I/O.  Placed in cross_sectional.py alongside the two
  primitives it composes, keeping portfolio.py as the backtest driver.  No behavior, no new
  strategy, no month-end / live code today.  Diff touched exactly three files
  (cross_sectional.py, portfolio.py, test_portfolio.py).
- DESIGNATED LIVE CONFIG (recorded, not yet consumed by code): lookback=3, top_n=1,
  price_field="close", hold_when_all_negative=False, cost_rate=0.0010.  This is the ONLY
  rotation config validated as a static whole out-of-sample in the record - it is the
  sorted(candidates)[0] baseline (min lookback 3, min top_n 1) whose fixed-parameter OOS
  Sharpe +0.44 is recorded in Day 45 and the README.  The per-fold search picks are explicitly
  disqualified (per-fold winners, never validated as one static tuple).  HONEST CAVEAT: this
  config's fixed OOS Sharpe +0.44 LOSES to equal-weight buy-and-hold (+0.59); its Sortino/MaxDD
  were never captured for the fixed baseline.  It is shipped to run the validated strategy
  end-to-end, NOT because it is a proven edge - the honest negative result is the project
  thesis.
- COST-RATE WIRING NOTE: rotation_backtest and rotation_walk_forward both default
  cost_rate=0.0, but the validated run used 0.0010 (10 bps).  single_date_weights takes no
  cost_rate (costs are an execution concern, not weight formation), so this does not affect
  the Day 56 live-target function - but any live performance report/reconciliation must
  explicitly use 10 bps to stay comparable to the validated number.

**Verification:**
- uv run pytest -q -m "not integration": 280 passed, 4 deselected (279 + 1 new goldenmaster).
  grep confirms one 1/top_n birthplace (cross_sectional.py) and zero in portfolio.py.  git
  diff --stat: only the three intended files.  Golden-master test passes with the SAME
  baked-in array post-refactor - bit-for-bit proof.

**Blocked on:**
- Nothing.

**Next up:**
- most_recent_completed_month_end helper (drop the incomplete current month; month_end_indices
  sets mask[-1]=True unconditionally, so "last grid element" on bars-through-today is unsafe
  live) and the pure live_target_weights(bars_by_symbol, today, config) that composes it with
  single_date_weights, reading the designated config above as a pinned constant.
- Then the daily runner (market-open + rebalance-day guards, compute target, rebalance) with
  logging and notifications, then a scheduler.  The daily runner's cost accounting must use
  cost_rate=0.0010 to match the validated verdict.

## Day 53

**Worked on:**
- Changed reconcile_to_target to emit all SELL orders before all BUY orders (commit "Emit
  sells before buys in reconcile_to_target").  The order-building loop is unchanged (same
  sizing, guards, delta logic, alphabetical iteration); only the final return changed - a
  stable partition (sells = [o for o in orders if o.side == SELL]; buys = [...BUY]; return
  sells + buys).  Because the source list is already alphabetical and the comprehensions are
  stable, sells stay A-Z and buys stay A-Z within each group. [+1 test]
- Updated one positional test (test_reconcile_closes_dropped_symbol) whose unpacking flipped
  (the TLT sell now precedes the SPY buy); its asserted CONTENT is unchanged (same SPY BUY 50
  and TLT SELL 100), only the positions swapped.  Added
  test_reconcile_emits_all_sells_before_all_buys, which builds a mix (two drops -> sells, two
  opens -> buys) whose alphabetical interleaving would have put a buy before a sell under the
  old ordering, and asserts last-sell-index < first-buy-index.  test_reconcile_opens_new_positions
  was NOT modified and still passes (two buys, alphabetical order preserved).  test_rebalance.py
  12 -> 13; full non-integration 278 -> 279, all green.

**Why it matters:**
- This closes the first of the two deferred limitations recorded on Day 49: orders were
  emitted alphabetically, so on a fully-invested rebalance a BUY could precede the SELL that
  funds it, and a live broker could reject the buy for insufficient buying power.  Emitting
  all exits before all entries means the cash freed by sells is available in list order
  before the buys that need it.  This is a prerequisite for running the rebalance loop
  unattended, where a buy-before-sell rejection would silently break a rebalance with no one
  watching to rerun it.

**Architectural note:**
- HONEST SCOPE: this is an ORDERING fix at the emission level only.  It guarantees sells
  appear before buys in the returned list; it does NOT guarantee the broker settles the sell
  cash before the buy executes - that depends on broker fill and settlement sequencing (T+1)
  and would require the runner to confirm sell fills before submitting buys.  That
  fill-confirmation sequencing is a separate, future runner-level refinement and remains on
  the Next-up list.  For the current paper bot with market orders it is sufficient in
  practice; the docstring states this boundary explicitly rather than overclaiming.
- The set of orders reconcile_to_target produces is bit-for-bit identical to before (same
  symbols, sides, quantities); only their sequence changed.  The change was kept to the
  single return statement so the sizing/guard/delta logic is provably untouched.

**Verification:**
- uv run pytest -q -m "not integration": 279 passed, 4 deselected (278 + 1 new).  Only
  src/execution/rebalance.py and tests/test_rebalance.py changed; runner.py and all other
  files untouched.
- The 13 tests in test_rebalance.py all green, including the new sells-before-buys property
  test and the flipped closes_dropped_symbol.

**Blocked on:**
- Nothing.

**Next up:**
- Fill-confirmation sequencing at the runner level (confirm sell fills before submitting
  buys) - the remaining, more-robust half of the buy-funding concern, deferred as a future
  refinement.
- The autopilot arc: a live-target function (data as of today -> target weights from the
  momentum ranking), then a daily runner (wake, check market-open and rebalance-day, compute
  target, rebalance) with logging and trade notifications, then a scheduler on an always-on
  host so it runs each market day unattended.
- Continued edge search: new hypotheses tested one at a time through the harness, accepting
  each verdict.

## Day 51

**Worked on:**
- Added scripts/rebalance_smoke.py (commit "Add live-paper rebalance smoke script"): a
  standalone, manual live-paper smoke check (a minimal end-to-end run confirming the
  wired-together system works, versus a hermetic unit test).  It builds
  AlpacaBroker.from_env(), calls verify_paper_account() explicitly, prints the account
  snapshot and current positions, then either DRY-RUNs (default: computes and prints orders
  via reconcile_to_target, submits nothing) or, with --submit, calls the real run_rebalance
  and prints each OrderResult.  Target is a hardcoded tiny {"SPY": 0.01}.  It is a tracked
  script in scripts/ (matching the existing paper-hitting scripts hello_alpaca.py /
  first_order.py); it is NOT a pytest test, so tests/ stays hermetic and CI is untouched.  No
  src/ change, no test change.
- Ran it live against the real Alpaca PAPER account.  Dry run and --submit agreed: paper
  account confirmed (PA3XE1TLPR93, is_paper True, portfolio value ~$100,067); observed an
  existing 2-share SPY position; target 1% of ~$100,067 at SPY ~$741.85 sized to 1 target
  share; delta from held 2 to target 1 produced a SELL 1 SPY market/day order.  --submit
  placed it: order accepted by Alpaca (order id recorded), confirmed in the dashboard.

**Why it matters:**
- This is the first time the assembled loop touched the real API.  Everything through Day 50
  was hermetically tested against fakes; this run proves the SAME run_rebalance /
  reconcile_to_target, unchanged, works end-to-end against real Alpaca paper: it observed
  real account value and real positions, fetched a real price, computed the correct order,
  and placed it.  The floor sizing, delta filter, guard-first ordering, and minimal price
  fetch all behaved in the wild exactly as the unit tests said.  "I think it works" is now "I
  watched it work."
- The success criterion was MECHANICAL, not profit: the order placed matched what
  reconcile_to_target computed (SELL 1 SPY), and it appeared in the dashboard.  By the honest
  research verdicts this strategy has no durable edge; a correct bot faithfully executing an
  edgeless target is the point, not a P&L claim.

**Architectural note:**
- DRY-RUN-BY-DEFAULT: the script submits nothing unless --submit is passed.  The dry-run
  reproduces the runner's read path (build current_positions, fetch prices for positive-weight
  targets, call reconcile_to_target) and prints the computed orders, so the exact orders can
  be eyeballed BEFORE any real order exists.  First contact with the API placed zero orders
  until the computation was verified.
- The run happened to exercise the SELL/trim path (held 2 SPY, target 1 -> sell 1), not the
  open-from-flat path, because the paper account carried a pre-existing 2-share SPY position
  from earlier scripts.  Both paths are unit-tested; the live run confirmed the trim path
  specifically.
- The three paper guards held in the wild: from_env hardcodes paper=True, __init__ asserts
  it, and verify_paper_account confirmed the PA-prefixed account before any read or order.

**Verification:**
- Dry run: printed PAPER ACCOUNT CONFIRMED, account fields, SPY price ~$741.85, and one
  computed order (SELL 1 SPY market day), submitting nothing.
- --submit: same reads, then run_rebalance placed the order; Alpaca returned status
  "accepted" with an order id; confirmed present in the Alpaca paper dashboard.
- No committed code or tests changed by the RUN itself; the only tracked change is the new
  scripts/rebalance_smoke.py.  Full non-integration suite unaffected (282 with integration,
  278 non-integration), since nothing in src/ or tests/ changed.

**Blocked on:**
- Nothing.

**Next up:**
- Sells-before-buys ordering in reconcile_to_target (so a rotation's funding SELLs execute
  before the BUYs they fund) - the last deferred execution limitation.
- The writeup/polish arc: a README presenting the TSMOM and rotation verdicts and the
  overfitting-tax methodology as the project's distinctive contribution.  This is the
  highest-leverage remaining work for the portfolio goal.

## Day 50

**Worked on:**
- Changed run_rebalance to fetch its own prices (commit "Fetch prices inside run_rebalance
  via get_latest_price"): the signature dropped the prices parameter and is now
  run_rebalance(broker, target_weights) -> list[OrderResult].  The runner builds the prices
  dict internally by calling broker.get_latest_price only for positively-weighted target
  symbols.  The pure reconcile_to_target is UNCHANGED (still takes prices as its 3rd arg);
  only WHO builds that dict moved from caller to runner. [+2 tests]
- Updated tests/test_rebalance.py: _FakeBroker gained a prices dict and a real
  get_latest_price (was a NotImplementedError stub) that logs f"get_latest_price:{symbol}"
  per call; the three existing runner tests drop the prices= kwarg.  Two new tests added.
  test_rebalance.py went 10 -> 12 tests; full non-integration 276 -> 278, all green, 4
  integration deselected.  Scope: only runner.py + test_rebalance.py; rebalance.py untouched.

**Why it matters:**
- This makes run_rebalance(broker, target_weights) the COMPLETE production entry point: it
  observes (get_account, get_positions), prices (get_latest_price), computes
  (reconcile_to_target), and acts (submit_order) - all behind the paper guard, with no
  caller-supplied prices.  That is exactly what makes the upcoming manual live-paper smoke
  check meaningful: it will run THIS function against the real paper account, not a
  stripped-down variant that needs prices hand-fed.

**Architectural note:**
- STRENGTHENED SAFETY PROPERTY: the guard now precedes ALL I/O, market data included.
  verify_paper_account() remains the first statement, before any get_latest_price call, so a
  live account triggers no market-data fetch AND no order.
  test_run_rebalance_live_raises_before_any_order now asserts BOTH broker.submitted == [] AND
  that get_latest_price was never called (no c.startswith("get_latest_price") in the call
  log) - proving the guard blocks before any network activity, not just before submission.
- MINIMAL FETCH: only positively-weighted targets are priced - {sym: broker.get_latest_price
  (sym) for sym, weight in target_weights.items() if weight > 0}.  A symbol being sold to
  zero (dropped from the target, or weight 0) needs no price, so it is never fetched.  This
  makes the minimum number of price calls and satisfies reconcile_to_target's "positive
  weight requires a positive price" guard by construction.  Two tests pin it:
  test_run_rebalance_skips_price_for_sold_symbol (a dropped holding is sold without being
  priced) and test_run_rebalance_prices_only_positive_weights (an explicit weight-0 symbol is
  not priced - and would KeyError if the runner tried, a built-in failure signal).
- The _FakeBroker's get_latest_price moved from the "abstract methods the runner never calls"
  block into the used-methods section, since the runner now calls it - keeping the class
  comment honest.

**Verification:**
- uv run pytest -q -m "not integration": 278 passed, 4 deselected (276 baseline + 2 new).  No
  network, no credentials.  Only src/execution/runner.py and tests/test_rebalance.py
  modified; rebalance.py untouched.
- The 12 tests in test_rebalance.py: 7 pure-fn (unchanged), 5 runner - paper places orders,
  live raises before any order (now also asserts no price fetch), guard called first, sold
  symbol not priced, only positive weights priced.

**Blocked on:**
- Nothing.

**Next up:**
- A manual live-paper smoke check: run run_rebalance against the real Alpaca paper account (a
  standalone local script, NOT a committed test) to confirm the assembled loop works
  end-to-end and orders appear in the Alpaca dashboard.  This is the first time the runner
  touches the real API - everything so far is hermetically tested but the assembled machine
  has never run live.
- Sells-before-buys ordering in the pure fn (so a rotation's funding SELLs execute before the
  BUYs they fund).
- Then the writeup/polish arc: a README presenting the TSMOM and rotation verdicts and the
  overfitting-tax methodology as the project's distinctive contribution.

## Day 49

**Worked on:**
- First real code in src/execution/ (commit "Add rebalance reconciliation function and
  guarded runner"): a PURE reconcile_to_target(target_weights, current_positions, prices,
  portfolio_value) -> list[OrderRequest] in src/execution/rebalance.py (zero broker calls,
  zero I/O), and a thin run_rebalance(broker, target_weights, prices) -> list[OrderResult]
  in src/execution/runner.py that calls verify_paper_account() FIRST, reads account and
  positions, delegates the sizing to the pure fn, and submits.  Plus tests/test_rebalance.py
  with 10 hermetic tests (7 for the pure fn, 3 for the runner). [+10 tests]
- 266 non-integration tests + 10 new = 276 passing under the CI selection (uv run pytest -q
  -m "not integration"), 4 integration tests deselected.  Purely additive: three NEW files,
  zero existing files modified.

**Why it matters:**
- This is the payoff of the execution arc: it turns the tested broker READ/WRITE toolkit
  (Days 46-48: get_account, get_positions, submit_order, all hermetically covered) into an
  actual rebalance LOOP - observe current holdings, compute the orders that close the gap to
  a target allocation, act.  The pure-vs-IO split keeps all the sizing arithmetic (weight ->
  target dollars -> floored whole shares -> share delta) in a function with zero I/O,
  testable with plain dicts, mirroring the discipline of combine_period_returns in
  src/research/portfolio.py.  The paper-only guard is now a LOAD-BEARING gate at a real call
  site: verify_paper_account() is the first executable statement in run_rebalance, so the
  function structurally cannot place an order without confirming paper.  This is the v1
  paper base-bot milestone: observe, compute, act-behind-guard, all hermetically proven
  offline.

**Architectural note:**
- TARGET REPRESENTATION REUSED, NOT REINVENTED: the rebalance target is a dict[str, float]
  of symbol -> weight fraction - the SAME "weight per symbol" concept the rotation
  backtester already produces (weights need not sum to 1.0; the remainder is implicitly cash
  and places no order).  No parallel allocation concept was introduced.
- THE SAFETY PROOF IS STRONGER THAN A MOCK: the runner test's _FakeBroker SUBCLASSES the
  real Broker ABC, so verify_paper_account() runs the GENUINE base-class guard
  (super().verify_paper_account() -> real get_account() -> real is_paper check -> real
  raise), not a fake stand-in.  test_run_rebalance_live_raises_before_any_order asserts BOTH
  that RuntimeError is raised AND that zero submit_order calls were recorded (broker.submitted
  == []), and test_run_rebalance_calls_guard_first asserts the call-log's first entry is
  "verify_paper_account".  So the live-account rejection is proven against the actual guard
  logic, offline.
- SIZING: target dollars = weight * portfolio_value; target shares = int(target_dollars /
  price) - int() floors toward zero so a target never over-buys past its dollar budget.  A
  symbol already at its target share count emits no order (delta filtering); a held symbol
  dropped from the target (weight 0) is closed to zero.  Orders are market + DAY for v1.
- TWO DEFERRED LIMITATIONS (long/flat v1 only, both safe today): (1) orders are emitted
  alphabetically by symbol, so a BUY can precede the SELL that funds it - hermetic tests do
  not care, but on live paper a fully-invested rotation could see an early BUY rejected for
  insufficient buying power before the funding SELL executes; fix later by submitting sells
  before buys.  (2) the runner builds current_positions from position qty and ignores
  PositionSnapshot.side, so it assumes no short positions - safe now because the guard
  rejects negative target weights (long/flat only), but note it before any short-enabled
  version.  Live-price fetch is also deferred: prices are passed into run_rebalance as an
  argument rather than fetched via get_latest_price, which keeps this first version fully
  hermetic (no data-client dependency).

**Verification:**
- uv run pytest -q -m "not integration": 276 passed, 4 deselected (266 baseline unchanged +
  10 new).  No network, no credentials.  Three new files only (src/execution/rebalance.py,
  src/execution/runner.py, tests/test_rebalance.py); zero existing files modified.  The pure
  fn imports only OrderRequest/OrderSide/OrderType/TimeInForce from src.brokers.base (no
  broker, no alpaca, no network).
- The 10 tests: pure fn - opens new positions, closes a dropped symbol while topping up a
  held one, skips a symbol already at target (empty list), floors fractional shares (33 not
  34 for 10000/300), flattens on empty target, rejects negative weight, rejects
  over-allocation; runner - paper account places the computed orders, live account raises
  before any order (submitted == []), guard is called first.

**Blocked on:**
- Nothing.

**Next up:**
- Sells-before-buys ordering in the pure fn (so a rotation's funding SELLs execute before
  the BUYs they fund) - the first deferred limitation above.
- Wiring get_latest_price into the runner (so the caller need not supply prices) - would add
  a get_stock_latest_trade stand-in to the broker test fake.
- A manual live-paper smoke check (run run_rebalance locally against the real paper account,
  NOT a committed test).
- Then the writeup/polish arc: a README presenting the TSMOM and rotation verdicts and the
  overfitting-tax methodology as the project's distinctive contribution.
- Housekeeping (non-urgent): backfill the [fill in] time-spent placeholders in the
  Day 29-39 entries.

## Day 48

**Worked on:**
- Added 2 hermetic, mocked tests to tests/test_alpaca_broker.py (commit "Add hermetic
  mocked tests for AlpacaBroker submit_order mapping and parse"): a market-order test and a
  limit-order test.  Each proves BOTH directions of the translation - our OrderRequest maps
  to the correct alpaca request object (MarketOrderRequest / LimitOrderRequest), and the
  returned alpaca Order parses back through _to_order_result into a correct OrderResult -
  with NO live submission, NO network, NO credentials. [+2 tests]
- Test-only, additions-heavy: 251 insertions / 2 deletions in tests/test_alpaca_broker.py
  (the 2 deletions are the in-place import-block rewrite; no existing test body changed).
  No src/ file touched - submit_order was already correct (inspection found no bug).
- 264 non-integration tests + 2 new = 266 passing under the CI selection (uv run pytest -q
  -m "not integration"), 4 integration tests deselected.

**Why it matters:**
- This was the highest-stakes mocking day of the arc.  Every prior hermetic test called
  READ methods, where a mock leak would at worst be a harmless read; submit_order is the
  first method whose real invocation PLACES AN ORDER (POST /v2/orders).  The safety rests
  on the same namespace monkeypatch used all arc: _build_broker_with_fake_account patches
  src.brokers.alpaca_broker.TradingClient BEFORE __init__, so self._trading is the fake and
  submit_order CANNOT reach the network - a real submission was never possible in these
  tests.  With this, the broker's ENTIRE surface is now hermetically covered: the read side
  (get_account, get_positions) and the write TRANSLATION side (submit_order's request
  mapping + response parse), all offline in CI.  What remains deliberately untested-in-CI
  is a live submission itself (that is the integration path's job, plus eventual manual
  paper checks).

**Architectural note:**
- THE CAPTURE TECHNIQUE (the new wrinkle vs the read-path tests): the read tests only faked
  a RETURN value; the write test must also assert the OUTBOUND mapping.  The fake
  _FakeTradingClient.submit_order stores the alpaca request it is handed
  (captured_order_request) and returns a pre-set fake Order (order_to_return).  Because
  AlpacaBroker.submit_order builds a REAL MarketOrderRequest/LimitOrderRequest before the
  intercepted call, the captured object is a genuine alpaca request instance - so the test
  asserts isinstance(captured, MarketOrderRequest/LimitOrderRequest) and reads its fields to
  prove the mapping (symbol, qty, side.value, time_in_force.value, and the limit price on
  the limit branch).  Both submit tests reset captured_order_request and order_to_return
  explicitly to prevent class-attribute leakage.
- TWO SDK ASSERTION TRAPS (documented so they are not re-hit): (1) MarketOrderRequest has NO
  limit_price field at all - accessing it raises AttributeError, not None - so the market
  test asserts `not hasattr(captured, "limit_price")`, never `is None`.  (2) The captured
  request's .qty is a FLOAT (pydantic coerces our int 10 -> 10.0, since alpaca
  OrderRequest.qty is Optional[float]) - so the mapping asserts qty by VALUE equality
  (captured.qty == 10), while our RETURNED OrderResult.qty is a genuine int (via
  int(order.qty)) and asserts type is int.  The two qty checks are deliberately different
  because they are on two different objects (alpaca's request vs our dataclass).
- WHAT THE PARSE PROVES: the fake Order carries string fields and .value-bearing enum
  stand-ins (reusing _FakeSide), with order_type NOT type (since _to_order_result reads
  order.order_type.value).  The parse asserts status == "new" and
  filled_qty/filled_avg_price == None - the real async-fill behavior, since a market order
  is not filled the instant it is accepted.

**Verification:**
- uv run pytest -q -m "not integration": 266 passed, 4 deselected (264 baseline unchanged +
  2 new).  No network, no credentials.  git diff --numstat tests/test_alpaca_broker.py:
  251/2 (the 2 deletions are the in-place import rewrite; grep of deleted lines confirms
  only the import block changed, no test body).  No src/ file in the diff.
- The 2 tests: market OrderRequest -> MarketOrderRequest (right fields, no limit_price attr)
  parsing to an OrderResult (qty int, status "new", fills None); limit OrderRequest ->
  LimitOrderRequest (WITH limit_price) parsing to an OrderResult (limit_price float,
  order_type LIMIT).

**Blocked on:**
- Nothing.

**Next up:**
- The increment that turns the tested toolkit into a bot: wire the strategy's target
  positions into submit_order calls behind the paper-only guard - a rebalance step combining
  get_positions (where am I?), the strategy signal (where do I want to be?), and
  submit_order (how do I get there?), gated by verify_paper_account() so it can only ever
  act on paper.  This is where the paper guard stops being a tested-in-isolation property
  and becomes a load-bearing gate at a real call site, so it wants the same care: guard
  called before any order, wiring tested hermetically before anything runs live.
- Then the writeup/polish arc: a README presenting the TSMOM and rotation verdicts and the
  overfitting-tax methodology as the project's distinctive contribution.
- Housekeeping (non-urgent): backfill the [fill in] time-spent placeholders in the
  Day 29-39 entries.

## Day 47

**Worked on:**
- Added get_positions to the broker (commit "Add get_positions to broker contract and
  AlpacaBroker"): a new PositionSnapshot dataclass + a new abstract method on the Broker
  ABC in src/brokers/base.py, the AlpacaBroker implementation (a _to_position_snapshot
  helper + get_positions) in src/brokers/alpaca_broker.py, and 2 hermetic tests in
  tests/test_alpaca_broker.py.  All read-only - no order submission. [+2 tests]
- 262 non-integration tests + 2 new = 264 passing under the CI selection (uv run pytest -q
  -m "not integration"), 4 integration tests deselected.  Additions-only where it counts:
  base.py 28/0, alpaca_broker.py 73/0 (zero deletions - no existing method/dataclass
  removed); the test file's single deletion is the import line rewritten in place (added
  PositionSnapshot), not a touched test body.

**Why it matters:**
- This is the first order-path increment and the first execution-arc change to touch the
  broker ABC (base.py), which is shared contract code - so backward compatibility was the
  live risk.  Adding an abstract method to an ABC is NOT silently backward-compatible:
  every concrete Broker subclass must implement it or raise TypeError at instantiation.
  Read-only inspection confirmed AlpacaBroker is the ONLY concrete Broker subclass in the
  repo (the test fakes are fakes of the Alpaca SDK clients, not Broker subclasses), so
  adding the abstract method AND implementing it in AlpacaBroker in the same commit keeps
  instantiation working.  The green suite is the proof: the hermetic tests construct
  AlpacaBroker and pass, which they could not do if the enlarged abstract contract were
  unsatisfied.  This completes the READ surface of the broker (account balances via
  get_account, open positions via get_positions) - the full "observe account state" half
  of execution, built and hermetically tested before any "act on it" code.

**Architectural note:**
- THE POSITIONSNAPSHOT SHAPE + qty CONVENTION (a locked decision): PositionSnapshot carries
  symbol, qty (int), side (str), market_value, avg_entry_price, unrealized_pl.  Alpaca's
  Position reports a POSITIVE qty plus a separate side ("long"/"short"), so the convention
  was locked as positive whole-share qty + a side string carrying direction - mirroring
  OrderResult's positive-qty + side convention, so downstream code reads ONE mental model,
  not two.  qty is int (int(float(position.qty)) - float-parse first so "10.0" parses, then
  int) under the whole-shares assumption this codebase already uses (it never places
  fractional orders).  market_value and unrealized_pl are Optional[str] on Alpaca and are
  guarded (float(x) if x is not None else 0.0) so downstream risk code never sees None;
  avg_entry_price is always present.
- THE CONVERSION + EMPTY CASE: _to_position_snapshot mirrors _to_order_result exactly (call
  the SDK, convert Alpaca's string fields, return our dataclass, no alpaca types leaking
  out), and get_positions mirrors list_recent_orders (a list comprehension over the SDK
  list).  The empty-account case is the one explicitly tested correctness point: a
  flat/new paper account has NO positions, so get_all_positions() returns [] and
  get_positions() surfaces [] cleanly (never None, no crash) - and this is the COMMON real
  state, since the paper account is currently flat, so every get_positions call before any
  order will return [].
- HERMETIC MOCKING (same pattern, extended): the tests extend the Day-46 fake - a
  _FakePosition with STRING money/qty fields (mirroring the real API so the str->int/float
  conversions are genuinely exercised) and a _FakeSide exposing .value (mirroring the
  PositionSide str-Enum), plus a get_all_positions() on _FakeTradingClient returning a
  class-attr list.  Both tests set that class attr EXPLICITLY (to [spy, tlt] and to []) to
  prevent class-attribute leakage between tests.

**Verification:**
- uv run pytest -q -m "not integration": 264 passed, 4 deselected (262 baseline unchanged
  + 2 new).  No network, no credentials.  git status --porcelain: only the three permitted
  files.  git diff --numstat: base.py 28/0, alpaca_broker.py 73/0, test file 154/1 (the 1
  deletion is the rewritten import line).  AlpacaBroker still instantiates (the passing
  hermetic tests prove the enlarged ABC contract is satisfied).
- The 2 tests: get_positions parses two faked holdings into a list[PositionSnapshot]
  (str->int qty, str->float money, side.value extraction, a negative unrealized_pl);
  get_positions returns [] on a flat account.

**Blocked on:**
- Nothing.

**Next up:**
- The write path, first increment: a hermetic mocked test of submit_order (mapping our
  OrderRequest -> the alpaca MarketOrderRequest/LimitOrderRequest, parsing the returned
  Order via _to_order_result) with NO live submission in CI.  This is the increment where
  the mocking discipline is most load-bearing - a real submit_order places an actual paper
  order, so the fake must intercept it completely.
- Then: wiring strategy target positions to orders behind the paper-only guard (its own
  increment).
- Then the writeup/polish arc: a README presenting the TSMOM and rotation verdicts and the
  overfitting-tax methodology as the project's distinctive contribution.
- Housekeeping (non-urgent): backfill the [fill in] time-spent placeholders in the
  Day 29-39 entries.

## Day 46

**Worked on:**
- Added 3 hermetic, mocked tests to tests/test_alpaca_broker.py (commit "Add hermetic
  mocked tests for AlpacaBroker read path and paper-only guard").  They exercise
  AlpacaBroker.get_account() (string->float parsing of the money fields, "PA"-prefix paper
  detection) and the inherited verify_paper_account() guard - with NO network and NO
  credentials - by monkeypatching the Alpaca SDK client classes (TradingClient /
  StockHistoricalDataClient) as imported into src.brokers.alpaca_broker's namespace, so
  __init__ builds fakes and never connects. [+3 tests]
- Test-only, purely additive: 160 insertions / 0 deletions in tests/test_alpaca_broker.py;
  no src/ file touched (base.py and alpaca_broker.py byte-for-byte unchanged).  The
  existing integration smoke test is intact and still deselected in CI.
- 262 tests green under the CI selection (259 non-integration baseline + 3 new), with 4
  integration tests deselected.  The count is reported as the non-integration total
  because that is what CI runs; the broker file now carries both the deselected
  integration smoke test and the 3 new hermetic tests.

**Why it matters:**
- The execution arc has a different enemy than the research arc: in research it was
  self-deception; here it is an unintended order.  So the first brick deliberately hardens
  the safety guarantee BEFORE any order path exists.  Read-only inspection found the broker
  was ALREADY fully implemented and safety-conscious - three independent paper-only guards
  (the __init__ assert paper is True, from_env hardcoding paper=True, and
  verify_paper_account raising on a non-paper account), credentials from env vars via
  load_dotenv, and .gitignore already covering .env/.env.local/.env.live/.env.paper.  The
  real gap was CI-safe COVERAGE: the sole existing broker test was @pytest.mark.integration
  (needs live keys + network), so it is deselected in CI and the paper-only guard's
  REJECTION path had never been tested at all - an integration test runs against a real
  paper account, which is is_paper True by construction and can only exercise the PASS path.
  The new live-rejection test (a numeric, non-"PA" account -> verify_paper_account must
  raise RuntimeError) is the offline proof of the paper-only property, and it now runs on
  every push.  The guarantee is enforced, not merely asserted.

**Architectural note:**
- THE MOCK TARGET (the crux): the tests monkeypatch src.brokers.alpaca_broker.TradingClient
  and ...StockHistoricalDataClient - the names AS IMPORTED INTO the broker's namespace,
  where the attribute lookup actually happens - NOT alpaca's own module path.  That is why
  AlpacaBroker.__init__ constructs fakes and makes no network call.  The fake account holds
  the money fields as STRINGS (buying_power="94321.50" etc.), mirroring the real Alpaca API,
  so the test genuinely exercises get_account's str->float conversion rather than passing
  trivially.  monkeypatch.setattr auto-reverts per test so patches never leak.
- SCOPE - deferred deliberately: NO order submission today, and NO get_positions.  The
  Broker ABC exposes get_account (balances) only - there is no positions method anywhere.
  Adding get_positions would mean a new abstract method on base.py + a new PositionSnapshot
  dataclass + the Alpaca impl; that is its own scoped increment and gets its own commit, not
  a bolt-on to a test-hardening day.  Today touched only tests/.

**Verification:**
- uv run pytest -q -m "not integration": 262 passed, 4 deselected (259 baseline unchanged
  + 3 new).  No network, no credentials.  git diff --numstat tests/test_alpaca_broker.py:
  160/0 (additions only).  No src/ file in the diff; the integration smoke test present and
  unchanged at its original lines.
- The 3 tests: get_account parses a faked paper snapshot (types, str->float, is_paper True);
  verify_paper_account passes on a "PA" account (returns None); verify_paper_account RAISES
  RuntimeError on a numeric non-"PA" account (the offline paper-only proof).

**Blocked on:**
- Nothing.

**Next up:**
- The order path, built incrementally and each brick tested: (a) get_positions - a new
  abstract method + PositionSnapshot dataclass + Alpaca impl + hermetic tests (its own
  commit, touches base.py); (b) a hermetic mocked test of submit_order (mapping OrderRequest
  -> alpaca request, parsing the response) with NO live submission in CI; (c) later, wiring
  strategy target positions to orders behind the paper-only guard.  Never let a code path
  reach a live account by default or by accident.
- Then the writeup/polish arc: a README presenting the TSMOM and rotation verdicts and the
  overfitting-tax methodology as the project's distinctive contribution.
- Housekeeping (non-urgent): backfill the [fill in] time-spent placeholders in the
  Day 29-39 entries.

## Day 45

**Worked on:**
- Added scripts/rotation_verdict.py (commit "Add real-basket rotation verdict script")
  plus a hermetic smoke test in a NEW tests/test_rotation_verdict_script.py.  A thin
  research CLI that loads the real 17-ETF basket from DuckDB via the cli_common helpers
  and runs the already-tested rotation_walk_forward_search ONCE on the whole basket
  (rotation is multi-symbol - not a per-symbol loop), printing the fixed-vs-fitted-vs-B&H
  OOS Sharpes, the overfitting tax, the secondary metrics (Sortino/MaxDD/TotalReturn),
  and the per-fold chosen (top_n, lookback).  It adds NO production logic - all
  fold/scoring/tax math lives in the tested src/research/rotation_verdict.py. [+1 test]
- 263 tests green (was 262).  Two new files only; no existing file touched.
- DATA-INTEGRITY GATE PASSED before trusting any number: all 17 symbols, identical 4638
  rows, identical 2008-01-02 to 2026-06-09 span, ZERO duplicate (symbol,date) groups (the
  historical tz-duplication bug is not present - the midnight-UTC floor in
  duckdb_store._bar_to_tuple holds), longest zero-return run 2 bars.  SPY spine = 222
  month-ends.  Verified on the REAL data before running the verdict.

**The verdict** (recorded straight - grid/windows/cost LOCKED before the run, NOT retuned):
- Config: grid = top_n in {1,2,3,5} x lookback in {3,6,9,12} months (16 candidates);
  reference SPY; cost_rate 0.0010 (10 bps per unit turnover); all locked before the run.
- Headline (train 24mo / test 12mo, 16 folds): Fitted OOS Sharpe +0.38 vs B&H +0.59 vs
  Fixed +0.44.  Overfitting tax +0.72 (in-sample +1.10 - fitted OOS +0.38).  Fitted
  Sortino +0.52 vs B&H +0.81; Fitted MaxDD 34.65% vs B&H 30.32%; Fitted total return
  +187.35% vs B&H +270.10%.
- Robustness cross-check (train 24mo / test 6mo, 32 folds): Fitted OOS Sharpe +0.40 vs
  B&H +0.59 vs Fixed +0.44.  Overfitting tax +0.72 (identical).  Fitted MaxDD 29.10% vs
  B&H 30.32%; Fitted total return +220.06% vs B&H +270.10%.
- FINDING: cross-sectional rotation on this basket does NOT beat equal-weight
  buy-and-hold out-of-sample after costs - it loses on Sharpe, Sortino, and total return,
  with drawdown roughly a wash (marginally worse on the 12mo window, marginally better on
  the 6mo).  Per-fold search also did NOT beat its own fixed-parameter baseline, so the
  tuning subtracted value.  The +0.72 tax - stable across BOTH windows - shows roughly
  two-thirds of the in-sample Sharpe was curve-fitting that did not survive OOS.  The
  per-fold picks thrash across the grid with no stable winner (fold 7 even scores negative
  in-sample), which is the visible mechanism behind the tax.

**Why it matters:**
- This is the honest-outcome principle delivering the product.  The value of Arc A was
  never a winning strategy - it was a validated pipeline that can take a plausible
  strategy, test it genuinely out-of-sample with a proven no-leak in-sample fit, charge
  realistic costs, and QUANTIFY how much apparent edge is self-deception.  The +0.72 tax,
  stable across two window layouts, is that quantification.  This corroborates the TSMOM
  verdict (no broad money-making edge on this basket); here even the drawdown benefit
  bonds gave TSMOM does not robustly appear for rotation.  Being able to SHOW - not assert
  - that momentum-style strategies do not produce durable cost-surviving OOS edge on this
  basket is the actual portfolio-grade result and the skill a research interview tests for.

**Verification:**
- Full suite 263 passed (was 262, +1 smoke test).  git status --porcelain: only the two
  new files.  Data-integrity gate passed on the real basket before the run (see above).
  Both verdict windows agree, and the tax is identical (+0.72) across them.

**Blocked on:**
- Nothing.  Arc A (the cross-sectional rotation arc) is COMPLETE: ranker (36), returns
  (38), combiner (39), rebalance loop (40), engine-equivalence (41), costs (42),
  walk-forward bridge (43), per-fold search + tax (44), real-basket verdict (45).

**Next up:**
- The execution/autonomy arc: wire AlpacaBroker so the base bot can run autonomously
  (paper) - the v1 milestone (a working, honestly-validated base bot a serious quant would
  respect), NOT a profit claim.
- Then the writeup/polish arc: a README/writeup presenting the TSMOM and rotation verdicts
  and the overfitting-tax methodology as the project's distinctive contribution.
- Housekeeping (non-urgent): backfill the [fill in] time-spent placeholders in the
  Day 29-39 entries.

## Day 44

**Worked on:**
- Added a rotation-specific per-fold GRID SEARCH to src/research/rotation_verdict.py
  (commit "Add rotation per-fold search and overfitting tax"), plus 8 hermetic tests
  in a NEW tests/test_rotation_search.py.  Per outer fold it searches (top_n,
  lookback) candidates for best IN-SAMPLE Sharpe (after costs) on the TRAIN span,
  applies the winner to the TEST span, and reports the overfitting tax via the SAME
  summarize_tax helper the TSMOM CLI uses.  New: the RotationSearchVerdict dataclass,
  _rotation_train_span_returns, _search_fold, and rotation_walk_forward_search. [+8 tests]
- 262 tests green (was 254).  Purely additive: 470 insertions / 0 deletions in
  rotation_verdict.py (the Day-43 functions are byte-for-byte unchanged) plus the new
  test file, so all 254 existing tests are unchanged.

**Why it matters:**
- This gives the rotation verdict its distinctive measure.  The fixed-parameter bridge
  (Day 43) produced a legitimate OOS number but no overfitting tax - nothing was fit
  per fold, so a tax would be hollow.  The per-fold search fits (top_n, lookback) on
  each train span and scores the winner OOS, so the in-sample-minus-OOS Sharpe gap is
  now a REAL measurement of how much the per-fold tuning was curve-fitting rather than
  genuine edge.  The tax is the project's signature contribution to rigor; computing it
  via the SAME summarize_tax the TSMOM path uses makes the rotation tax directly
  comparable to the TSMOM tax - by construction, not by careful matching.

**Architectural note:**
- THE NO-LEAK IN-SAMPLE STREAM (the crux): _rotation_train_span_returns produces a
  candidate's in-sample stream by DELEGATING to the unchanged Day-43
  _rotation_fold_returns with an INNER triple (train_start_me, train_start_me +
  lookback + 1, test_start_me).  The inner closing boundary is test_start_me - the
  OUTER split boundary - so the inner scoring window's last period is (..., D_{test_start_me}]
  and its ranking uses D_{test_start_me - 1}; both endpoints are <= the boundary, so
  the in-sample stream uses ONLY train-span data.  NO bar strictly after D_{test_start_me}
  (no outer test return) can enter the in-sample score - the walk-forward-search analogue
  of the Day-40/43 boundary discipline, one level deeper.  A test PROVES it
  experimentally: mutating a test-span bar (strictly after the boundary) leaves the
  in-sample stream BYTE-IDENTICAL (assert_array_equal) - if the fit cannot see a changed
  test bar, no test data leaked in.  The +1 in the inner triple warms the candidate's
  lookback inside the train span (inner train = lookback+1 > lookback), so the in-sample
  stream (including its entry-turnover cost) matches a full-history rotation.
- THE TAX = summarize_tax, REUSED not reimplemented: the tax is computed by calling
  scripts.overfitting_tax.summarize_tax (the SAME helper the TSMOM overfitting-tax CLI
  uses), so tax = mean(per-fold in-sample Sharpes) - stitched fitted OOS Sharpe,
  identical definition to TSMOM.  RotationSearchVerdict carries the parallel fields
  (fixed_oos / fitted_oos / bh / mean_in_sample / tax) so a rotation row reads like a
  TSMOM TaxRow.  fixed_oos comes from running the Day-43 fixed-parameter
  rotation_walk_forward once on the baseline candidate, matching the TSMOM CLI's
  fixed-vs-fitted structure.
- COST-AWARE IN-SAMPLE OBJECTIVE - a DELIBERATE divergence from TSMOM (recorded so it
  is a choice, not an inconsistency): the rotation in-sample search scores candidates
  AFTER costs (same cost_rate threaded to both the in-sample train-span stream and the
  OOS test-span stream).  The TSMOM fitter deliberately searches FRICTIONLESS (its
  comment: in-sample stays frictionless so the tax isolates parameter-selection
  overfitting, not cost modeling).  Rotation diverges because its turnover - and thus
  cost - IS a function of the searched params (top_n and lookback change rebalance
  frequency), so a frictionless in-sample search would pick params that ignore their own
  turnover cost and then be penalised OOS, inflating the tax with a COST ARTIFACT rather
  than genuine parameter-selection overfitting.  Scoring both sides after costs keeps the
  rotation tax honest for a strategy whose parameter choice drives its trading frequency.
- THE DETERMINISTIC TIEBREAK: _search_fold sorts candidates by (lookback, top_n)
  ascending and uses a strictly-greater argmax update, so on an in-sample Sharpe tie the
  FIRST (smallest lookback, then smallest top_n) wins - reproducible, no
  np.argmax-on-ties nondeterminism.  A grid (not Optuna) was chosen for two
  low-cardinality params: exhaustive (true argmax, not sampled), deterministic, no
  dependency, no seed.  A test pins the tiebreak with two identical-stream candidates.
- THE L_max WARMUP GUARD: rotation_walk_forward_search raises if train_months <=
  L_max + 1 where L_max is the grid's LARGEST candidate lookback - the inner in-sample
  scoring window for L_max is train_months - L_max - 1 month-ends, so <= L_max + 1 leaves
  zero (a degenerate len<2 -> 0.0 in-sample Sharpe).  Recommends train_months >=
  L_max + 3 for a meaningful in-sample Sharpe.  Guard is on the whole grid's max, not a
  single lookback.
- THE BENCHMARK is UNCHANGED from Day 43 and lookback-invariant (it holds every symbol
  regardless of rank), so it is produced via the existing _benchmark_fold_returns at the
  grid's smallest lookback and is the SAME bar the fixed-parameter bridge measures.  A
  test asserts the searched run's bh_sharpe / bh_total_return equal the Day-43
  rotation_walk_forward's on the same basket - the bar the strategy must clear did not move.
- ADDITIVE: the winner's OOS stream uses the unchanged Day-43 _rotation_fold_returns
  (outer triple); both strategy and benchmark are stitched via the reused _stitch_oos.
  rotation_walk_forward and RotationVerdict stay byte-for-byte as the no-search baseline
  (470 insertions, 0 deletions).

**Verification:**
- Full suite 262 passed (was 254, +8).  git diff --numstat shows 470/0 on
  rotation_verdict.py (purely additive; git diff | grep '^-[^-]' empty -> no existing
  Day-43 line changed) plus the new tests/test_rotation_search.py.  No file outside the
  two touched; no existing test changed status.
- The 8 tests: the no-leak proof (mutate a post-boundary test bar -> in-sample stream
  byte-identical); the search picks the clear in-sample winner; the deterministic
  tiebreak (identical-stream candidates -> smaller (lookback, top_n)); the L_max guard
  raises; the empty-grid guard raises; the searched verdict fields populated with tax ==
  mean(in_sample) - fitted_oos (the summarize_tax formula); the benchmark equals the
  Day-43 benchmark; and a Day-43-path-unchanged smoke test.

**Blocked on:**
- Nothing.

**Next up:**
- The real-basket verdict run: run rotation_walk_forward_search on the actual 17-ETF
  basket and record the honest OOS, after-costs, tax-reported answer to whether
  cross-sectional rotation beats holding the basket.  This is the Arc C payoff - the
  question the entire rotation arc was built to answer.  Keep the honest-outcome
  principle: given the TSMOM verdict (edge concentrated in bonds, drawdown reduction the
  durable value), a result where rotation does NOT clearly beat buy-and-hold is entirely
  plausible and is a legitimate, reportable finding - the machinery makes either answer
  trustworthy.
- Then: the execution/autonomy arc (wire AlpacaBroker) = v1 base bot; writeup/polish arc.
- Housekeeping (non-urgent): backfill the [fill in] time-spent placeholders in the
  Day 29-39 entries.

## Day 43

**Worked on:**
- Added a NEW module src/research/rotation_verdict.py (commit "Add rotation
  walk-forward bridge and verdict"), plus 9 hermetic tests in a NEW
  tests/test_rotation_verdict.py.  It walk-forward-validates the cost-adjusted
  rotation_backtest stream OUT-OF-SAMPLE against an equal-weight-basket
  buy-and-hold benchmark, reusing the existing _stitch_oos scoring seam.  Four
  functions: rotation_month_end_folds (the month-end-aligned splitter),
  _rotation_fold_returns (per-fold test-span returns, warmed inside the train
  span), _benchmark_fold_returns (the always-hold-everything benchmark), and
  rotation_walk_forward (the orchestrator returning a RotationVerdict). [+9 tests]
- 254 tests green (was 245).  Purely additive: two NEW files only, no existing
  file touched (git status --porcelain shows only the two untracked files), so
  the 245 existing tests are bit-for-bit unchanged.  This is the FIXED-PARAMETER
  bridge (step A); per-fold hyperparameter search and the overfitting tax are
  step B.

**Why it matters:**
- This is the foundation of the final Arc A brick: rotation can now be scored
  OUT-OF-SAMPLE and AFTER COSTS against a FAIR benchmark.  The two prerequisites
  for a legitimate verdict are both satisfied - Day 41 proved rotation's returns
  are on the same log-return basis as the engine/benchmark, Day 42 made costs
  correct - so the OOS rotation Sharpe/Sortino/drawdown are directly comparable
  to the equal-weight-basket buy-and-hold's over the identical test spans.
  Whatever the verdict turns out to be (rotation may or may not beat the basket),
  it is now an HONEST, defensible number rather than a cost-free in-sample
  backtest.

**Architectural note:**
- WHY A NEW MODULE, NOT walk_forward_validate: the generic harness is hard-wired
  to single-symbol Strategy objects (it drives generate_signals + Backtester.run
  on one symbol), and its walk_forward_splits cuts folds on BAR INDEX.  Rotation
  is multi-symbol and its causality lives on the reference symbol's MONTH-END
  grid, so a bar-index split could land a train/test boundary mid-holding-period
  and score a test fold on a return whose ranking used training data - lookahead.
  So the bridge REUSES only the pure scoring seam (_stitch_oos, and through it
  metrics.py) and ADDS a rotation-specific month-end-aligned splitter.  Nothing
  in walk_forward.py/portfolio.py/metrics.py/engine.py changed.
- THE MONTH-END FOLD SPLITTER (the no-lookahead crux): rotation_month_end_folds
  returns (train_start_me, test_start_me, test_end_me) triples that are INDICES
  INTO THE MONTH-END ARRAY, not bar indices.  The split falls ON month-end
  position test_start_me, so the test fold's first rank is as-of that month-end
  (train-span data only) and the test holding returns are strictly after it - the
  walk-forward analogue of walk_forward_splits' strict train/test adjacency.  A
  test pins this with a winner that FLIPS at the boundary: the correct
  as-of-boundary pick earns a different, detectable number than a lookahead bug
  would.
- GUARD CORRECTION (found during implementation, documented): a rotation test
  "observation" is a HOLDING PERIOD between two month-ends, so a test span of
  test_months decision month-ends needs one MORE closing month-end D_{test_end_me}
  to terminate the last period (D_{test_end_me-1}, D_{test_end_me}].  The minimum
  month-end count is therefore train_months + test_months + 1 (not +0), and the
  tail rule is test_end_me <= M-1.  This +1 is also what makes consecutive
  non-overlapping folds TILE with no gap and no overlap (fold i's last period ends
  at D_{test_end_me}; fold i+1's first period starts there), preserving
  walk_forward_splits' no-gap/no-overlap guarantee.  (The original plan omitted the
  +1; the closing-boundary requirement makes it necessary.)
- THE TEST-SPAN EXTRACTION (the alignment crux): each fold slices every symbol's
  bars to [D_{train_start_me}, D_{test_end_me}] inclusive, runs rotation_backtest
  over the slice (warm-up pairs + test pairs), and strips the warm-up prefix.  The
  warm-up bar count is computed from the TELESCOPING reference spine
  (ref_mei[test_start_me] - ref_mei[train_start_me]), NOT by assuming one bar per
  pair - so it stays correct on multi-bar months.  A defensive invariant raises if
  the slice stream length != warmup_bars + test_bars (fail loud on any spine
  misalignment rather than silently mis-slice).
- WARM-UP SUFFICIENCY - why train_months > lookback (STRICT): the test span's
  first decision needs `lookback` prior month-ends, but the warm-up decision
  IMMEDIATELY BEFORE the test span (which sets the test span's first-bar
  entry-turnover cost) needs lookback+1, so train_months must strictly exceed
  lookback for the test-span stream - including its entry cost - to match a
  full-history rotation exactly.  Guarded, with a test.
- THE PREPEND-ZERO STITCH REUSE: _stitch_oos does fr.returns[1:] per fold,
  assuming the engine's structural index-0 zero.  Rotation streams have NO leading
  zero, so each fold's returns get a single 0.0 prepended before wrapping in a
  minimal BacktestResult - _stitch_oos's [1:] then strips the injected zero and
  keeps every real return.  Confirmed _stitch_oos reads ONLY fr.returns (it
  concatenates fr.returns[1:] and derives all metrics from that), so the other
  BacktestResult fields on the minimal per-fold result are provably-unused safe
  placeholders.  A test pins the stitched length == real return count (not one
  fewer).
- THE BENCHMARK - true always-hold-everything, charged fairly:
  _benchmark_fold_returns reuses _rotation_fold_returns with top_n=len(basket) and
  hold_when_all_negative=True (all symbols always held, no absolute filter - a real
  buy-and-hold of the basket, NOT a momentum-filtered subset), same
  lookback/folds/test-span extraction/cost_rate as the strategy.  So strategy and
  benchmark are scored on byte-identical spans by identical code, differing ONLY in
  top_n and the filter flag - a fair edge-vs-beta comparison.  A test pins the
  benchmark == equal-weight blend (0.5*rA + 0.5*rB), distinct from the top_n=1
  rotation.
- NO OVERFITTING TAX under fixed parameters: RotationVerdict deliberately has no
  tax field.  Under fixed parameters nothing is fit per fold, so a tax would be
  hollow - it is deferred to step B (per-fold Optuna search), where the
  in-sample-vs-OOS gap will measure real curve-fitting.

**Verification:**
- Full suite 254 passed (was 245, +9).  git status --porcelain shows ONLY the two
  new untracked files - no existing file modified, so the 245 existing tests are
  bit-for-bit unchanged.  _stitch_oos and metrics reused by import.
- The 9 tests: fold basics (exact triples, non-overlapping tiling); step + ragged
  tail; guards (train/test/step < 1, too-few-month-ends); the on-month-end
  no-lookahead pin (winner flips at the boundary, lookahead bug would give a
  detectably different number); prepend-zero keeps all real returns through
  _stitch_oos; the whole-basket equal-weight benchmark distinct from top_n=1
  rotation; cost monotonicity (costed total return <= free, both strategy and
  benchmark); all verdict fields finite with correct n_folds; and the
  train_months > lookback guard.

**Blocked on:**
- Nothing.

**Next up:**
- Step B: per-fold hyperparameter search + the meaningful overfitting tax.  On
  each train span, search top_n/lookback to maximize in-sample Sharpe, score those
  params on the test span, and report the in-sample-vs-OOS Sharpe gap as the
  overfitting tax.  This layers onto the now-proven fixed-parameter bridge and
  completes the rotation verdict - the Arc C payoff.  (The B search cannot reuse
  the TSMOM Optuna fitter, which returns a Strategy for walk_forward_validate's
  seam; it needs a rotation-specific per-fold search on the month-end-fold
  returns.)
- Then: run the verdict on the real 17-ETF basket and record the honest
  rotation-vs-basket result (after costs, OOS).  Then the execution/autonomy arc.
- Housekeeping (non-urgent): backfill the [fill in] time-spent placeholders in the
  Day 29-39 entries.

## Day 42

**Worked on:**
- Added a transaction-cost model to rotation_backtest in src/research/portfolio.py
  (commit "Add transaction-cost model to rotation backtest"), plus 6 hermetic tests
  in tests/test_portfolio.py.  A new cost_rate parameter (a fraction, fee+slippage
  combined, defaulting to 0.0) charges each rebalance's turnover as a single-bar
  log-space linear drag on the first bar of the holding period. [+6 tests]
- 245 tests green (was 239).  Purely additive to behaviour: the cost_rate=0.0
  default is a bit-for-bit no-op, so all 18 prior portfolio tests (and the rest of
  the suite) are unchanged.

**Why it matters:**
- This is the brick that lets a rotation number become a VERDICT rather than a
  cost-free backtest.  No strategy result is meaningful until costs are applied -
  the standing discipline that has governed every verdict since TSMOM.  A monthly
  rotation strategy trades on every rebalance, so friction eats directly into its
  edge; a cost-free rotation Sharpe is not a number you can quote.  With the cost
  model in place a rotation stream can now be charged against realistic
  fees/slippage, and because the cost uses the engine's exact log-space convention,
  a rotation Sharpe after costs is directly comparable to a TSMOM Sharpe after
  costs - the comparability the Day-41 equivalence test established for returns now
  extends to costs.

**Architectural note:**
- THE TURNOVER DEFINITION (the day's real design work, locked before code via a
  read-only inspection).  Turnover at each rebalance is the L1 WEIGHT change over
  the RISKY symbols only, CASH EXCLUDED: turnover_k = sum over (union of symbols in
  prev_weights or current weights) of |w_k(s) - w_{k-1}(s)|.  Cash is deliberately
  EXCLUDED because the engine's turnover is |diff(held)| - notional traded, with NO
  separate cash leg - so including the cash weight's change would DOUBLE-COUNT on
  any entry-from-cash or exit-to-cash.  The inspection proved this with three
  reference cases: a full entry from cash must be turnover 1 (not 2); a full
  name-for-name switch is turnover 2 (matching the engine's long/short flip = 2); a
  retained holding at the same weight is turnover 0.  The cash-inclusive sum agrees
  ONLY on the pure-switch case (where cash is unchanged) - it would have penalized
  every de-risk-to-cash rebalance at double the true cost, making the strategy look
  systematically worse than it is.  Halving the cash-inclusive sum does NOT fix it
  (it would make the switch case 1, wrong).  The fix is exclude cash, not scale.
- CARRYING prev_weights, not the held set: the turnover is computed from the actual
  weight DICTS of consecutive periods (prev_weights, initialized empty = all-cash
  prior, reassigned to the current weights each iteration), NOT from
  (|added|+|dropped|)/top_n set arithmetic.  Under the current rule (b) with fixed
  top_n those are equal, but carrying the weight dict is RULE-AGNOSTIC: if the
  weighting ever switches to rule (a) (1/len(held)), a RETAINED symbol's weight
  changes when len(held) changes, and a set-difference formula would silently
  undercount.  The weight-space L1 sum stays correct either way, and it reads like
  its definition rather than a coincidence-of-fixed-top_n shortcut.
- THE DRAG matches the engine EXACTLY: period_returns[0] = period_returns[0] -
  turnover_k * cost_rate.  A LOG-SPACE LINEAR drag (subtracted from the log return,
  not a multiplicative (1 - cost) factor), a SINGLE-BAR hit at the rebalance
  boundary (the first bar of the holding period, which always exists since
  n_bars >= 1).  This mirrors the engine's cost_returns = turnover * cost_rate;
  strategy_returns -= cost_returns, so rotation costs and single-symbol engine costs
  are on ONE convention - the whole point of keeping the two comparable.
- THE PARAMETER: cost_rate is a raw FRACTION (fee+slippage already combined and
  divided by 10000), matching the engine's self.cost_rate and rotation's own
  raw-fraction style (cash_per_bar_return is likewise a raw fraction).  Added as the
  LAST parameter so no positional call site breaks.  Guards cost_rate < 0 ->
  ValueError (a cost is never a credit; mirrors the engine's fee_bps>=0).  Defaults
  to 0.0 (cost-free, bit-for-bit identical to pre-cost).
- The cost attaches ENTIRELY inside rotation_backtest: combine_period_returns and
  _per_bar_log_returns are untouched, and the function still returns a bare
  np.ndarray (no total-cost field - reporting rotation's cumulative cost is a
  separate additive decision for later).

**Verification:**
- Full suite 245 passed (was 239, +6).  Only two files changed: portfolio.py
  (rotation_backtest only) and test_portfolio.py (6 tests).  combine_period_returns
  and _per_bar_log_returns byte-for-byte unchanged; engine.py and all other files
  untouched.  The 18 prior portfolio tests unchanged in status (the no-op default
  confirmed by a bit-for-bit regression test).
- The 6 tests: a zero-cost no-op regression (assert_array_equal, not allclose -
  proves the default is bit-for-bit); a single-bar drag test (multi-bar period,
  first bar charged, later bar untouched); entry-from-cash is turnover 1 (and
  explicitly NOT 2 - the test that proves the cash-excluded definition, the one the
  cash-inclusive formula would fail); a full name-for-name switch is turnover 2
  (matches the engine flip); a retained holding is turnover 0 (no drag); a negative
  cost_rate raises.

**Blocked on:**
- Nothing.

**Next up:**
- The walk-forward bridge + rotation verdict - the FINAL Arc A brick.  Run the
  now-cost-adjusted rotation stream through the existing harness (walk-forward,
  overfitting tax, Sharpe/Sortino, drawdown, AFTER costs) against an
  equal-weight-basket buy-and-hold benchmark.  This is the Arc C payoff: the honest,
  cost-adjusted answer to whether cross-sectional rotation beats holding the basket.
  The Day-41 equivalence test and today's cost model are what make that verdict
  legitimate - rotation's returns and costs are both on the same basis as the
  benchmark's.
- Housekeeping (non-urgent): backfill the [fill in] time-spent placeholders in the
  Day 29-39 entries.

## Day 41

**Worked on:**
- Added test_rotation_n1_equals_engine_run to tests/test_portfolio.py (commit
  "Add N=1 engine-equivalence test for rotation backtest").  It proves
  rotation_backtest on a single always-held symbol reproduces Backtester.run's
  per-bar return stream on that symbol over the aligned held tail.  TEST-ONLY:
  no production file changed. [+1 test]
- 239 tests green (was 238).  The equivalence assertion passed on the FIRST run
  at rtol=1e-12 - no production edit, no tolerance loosening.

**Why it matters:**
- This CLOSES the Day-39 named risk.  When the portfolio backtester was built
  NEW rather than wrapping the per-symbol engine (Decision A), it REIMPLEMENTED
  the engine's per-bar log-return arithmetic (_per_bar_log_returns) instead of
  calling run().  That reimplementation was a named risk: if it drifted from the
  engine's np.log(closes[1:]/closes[:-1]) convention even slightly, every
  rotation number would be on a different basis than the TSMOM verdict, making
  any rotation-vs-TSMOM comparison invalid.  The equivalence test is the
  independent proof that the two code paths agree - so the NEW-not-wrap decision
  is now VALIDATED, not merely asserted, and rotation numbers are confirmed to be
  on the same log-return basis as everything else the engine produces.
- An equivalence test's whole value is that it is INDEPENDENT: two
  separately-written code paths asserted to produce the same numbers, with
  neither edited to force the match.  It passed first-run with no production
  change, which is the strongest possible form of that result.

**Architectural note:**
- THE ALIGNMENT (the day's real design work, locked before any test code via a
  read-only inspection).  The two streams are NOT trivially comparable: the
  engine's always-long run has a structural index-0 zero and covers every bar
  from index 0; the rotation stream has a warmup-cash front (the symbol is not
  yet eligible) and never covers the pre-first-month-end region or D_0's own bar.
  So the test compares the ALIGNED HELD TAIL, not the full streams.
- Construction: ONE bar per calendar month (so every bar is a month-end and
  ref_mei = [0..N-1], making the index math exact), N=6, STRICTLY INCREASING
  closes with DISTINCT ratios [100, 108, 121, 130, 145, 160].  Strictly-increasing
  => every trailing return > 0 => the symbol is always held after warmup (never
  drops to cash mid-stream).  Distinct ratios => every per-bar log return is
  distinct, so a wrong alignment offset FAILS rather than coincidentally passing
  on a uniform series - the distinctness is what makes the offset genuinely tested.
- lookback L=1 => exactly one warmup period.  top_n=1 with one symbol => weight
  1/top_n = 1.0, cash_weight = 0.0, so each held period is exactly the symbol's
  own per-bar stream (the N=1 identity).
- THE INDEX CORRESPONDENCE (load-bearing): with one bar per month, period k spans
  engine bar k+1, so rot_returns[k] = log(close[k+1]/close[k]) = engine_returns[k+1].
  Two SEPARATE assertions: (a) warmup front rot_returns[:L] all exactly 0.0;
  (b) held tail rot_returns[L:] == engine_returns[L+1:] via assert_allclose at
  rtol=1e-12/atol=1e-15.  The +1 on the ENGINE side is essential: rotation's
  period 0 covers engine bar 1 (the bar after D_0), never engine bar 0, so the
  engine tail starts one index LATER than the rotation tail.  A length-equality
  guard precedes the allclose so a mismatch fails loudly rather than broadcasting;
  the structural length len(rot_returns) == len(bars) - 1 is also pinned.
- WHY the seams are contiguous (confirmed in the inspection, proven by the test):
  at each month-end D_k the bar appears EXACTLY ONCE in the stitched stream - as
  the right endpoint (numerator) of period (D_{k-1}, D_k], and only as the anchor
  DENOMINATOR of period (D_k, D_{k+1}] (excluded from its spine, dropped via [1:]).
  No gap, no duplicate at any seam.  This contiguity is precisely what would have
  broken if the (D_k, D_{k+1}] boundary or the anchor-drop were off by one - and
  the equivalence test would have caught it.
- Tight tolerance is not slack: both sides run the identical np.log on the
  identical closes, so the values are bit-identical in principle; rtol=1e-12
  catches any REAL drift (orders of magnitude larger than float noise) while
  passing genuine equivalence.  A failure would have been a discovered arithmetic
  bug to surface, NOT a tolerance to loosen.

**Verification:**
- Full suite 239 passed (was 238, +1).  Only tests/test_portfolio.py changed (one
  test + two imports: Backtester from src.backtest.engine, SIGNAL_LONG from
  src.strategies.base).  No production file touched - _per_bar_log_returns,
  rotation_backtest, combine_period_returns, and engine.py all unchanged.  The
  existing 17 portfolio tests unchanged in status.
- The equivalence assertion passed on the first run at rtol=1e-12:
  _per_bar_log_returns matches the engine's np.log(closes[1:]/closes[:-1])
  convention bit-for-bit over the aligned held tail.

**Blocked on:**
- Nothing.

**Next up:**
- The transaction-cost model at the turnover seam: apply fees/slippage on
  rebalance turnover (the set(held) - prev_holdings added / prev_holdings -
  set(held) dropped differences already carried in the loop).  This is the brick
  that finally lets a rotation number become a VERDICT rather than a cost-free
  backtest - no strategy result is meaningful until costs are applied.
- Then: the walk-forward bridge + rotation verdict through the existing harness
  (walk-forward, overfitting tax, Sharpe/Sortino, drawdown, AFTER costs) vs an
  equal-weight-basket buy-and-hold = the Arc C payoff.
- Housekeeping (non-urgent): backfill the [fill in] time-spent placeholders in the
  Day 29-39 entries.

## Day 40

**Worked on:**
- Added rotation_backtest(bars_by_symbol, reference_symbol, lookback, top_n,
  price_field="close", cash_per_bar_return=0.0, hold_when_all_negative=False) ->
  np.ndarray and its helper _per_bar_log_returns(bars, price_field="close") ->
  np.ndarray to src/research/portfolio.py (commit "Add monthly cross-sectional
  rotation rebalance loop"), plus 9 hermetic tests in tests/test_portfolio.py.
  This is the orchestration core of the rotation backtester: it walks month-end
  rebalances and at each one chains the full pipeline - trailing_returns_at
  (as-of returns) -> rank_by_trailing_return (select top-N) -> equal weights per
  rule (b) -> combine_period_returns (one period stream) - then stitches all
  periods into ONE portfolio log-return stream. [+9 tests]
- 238 tests green (was 229): +9.  Purely additive: combine_period_returns
  untouched, no existing file modified.

**Why it matters:**
- This chains the whole rotation arc end-to-end for the first time: the ranking
  primitive (Day 36), the returns-computer (Day 38), and the combination core
  (Day 39) are now driven by one loop that produces a real portfolio return
  stream from a basket of bars.  It is a working cross-sectional rotation
  backtest - not yet cost-adjusted or walk-forward-validated, but the pipeline is
  complete and provable on hand-computed numbers (the end-to-end test hand-traces
  which symbol is held at each rebalance and its holding-period return).

**Architectural note:**
- TWO decisions locked before code (both real, both recorded).  SCHEDULE:
  rebalance dates are the REFERENCE symbol's month-ends (SPY as the long-history
  spine), giving ONE shared rebalance calendar across the ragged basket rather
  than per-symbol month-ends; the loop walks adjacent month-end pairs and STOPS
  at the last complete period (no trailing partial period, which would be a
  variable-length segment that complicates later walk-forward stitching).
  trailing_returns_at still resolves each OTHER symbol to its own most-recent
  month-end at-or-before the shared date internally, so the shared calendar and
  the per-symbol as-of logic coexist correctly.
- WEIGHTING - rule (b): each held symbol gets a FIXED 1/top_n weight, and the
  unfilled (top_n - h)/top_n sits in cash.  This de-risks into cash when fewer
  than top_n names clear the momentum filter - the continuous extension of the
  Day-36 all-negative -> empty -> cash discipline (zero qualifiers is all-cash
  under both rules; rule (b) makes the partial case continuous with that).  The
  alternative rule (a) - split full capital among survivors, 1/h each, always
  fully invested - is a defensible always-in stance but creates a discontinuity
  (one qualifier = 100% one name, zero = 100% cash) and concentrates risk exactly
  when momentum is scarce.  Rule (b) is a WEIGHT-COMPUTATION choice only
  (combine_period_returns handles any weights summing to <= 1 via the cash
  remainder), so rule (a) remains a one-line swap later, not a rewrite.
- NO-LOOKAHEAD BOUNDARY: the holding period is HALF-OPEN LEFT, CLOSED RIGHT - the
  interval (D_k, D_{k+1}].  Ranking happens AS OF D_k (trailing_returns_at is
  right-inclusive of D_k), and the holding-period returns are computed strictly
  AFTER D_k: D_k's own bar is used only as the denominator anchor for the first
  in-period return and is then dropped, so D_k's bar-to-bar return never appears
  in the period it was selected at.  This is the rotation analogue of the
  engine's one-bar lag (signals[:-1] * returns[1:]) - decide on D_k's
  information, earn the return that begins accruing after D_k.
- EQUAL-LENGTH GUARD (fail loud, never pad/truncate): the midnight-UTC-floor
  dedup guarantees at most one bar per (symbol, date) but does NOT guarantee every
  symbol has a bar on every trading day, so a held symbol with an interior data
  gap could misalign against the shared spine.  The loop asserts each held
  symbol's in-period dates match the reference spine exactly and RAISES (naming
  the symbol and dates) on a mismatch - a misaligned return series would corrupt
  the backtest silently, so it must fail loud.  A test pins this (a held symbol
  missing an interior spine date raises).
- PER-BAR RETURNS reuse the engine's arithmetic, not run(): _per_bar_log_returns
  reproduces engine.py's asset_returns convention byte-for-byte (element 0 = 0.0;
  element i = log(close[i]/close[i-1])) via getattr(bar, price_field), on the SAME
  price_field used for ranking so both are on one basis.  This is the Day-39
  Decision-A path; the helper is a SEPARATE, independently testable unit precisely
  because it is the anchor the FUTURE N=1 engine-equivalence test will target (a
  single held symbol at full weight must reproduce run()'s stream).  That
  equivalence test is NOT built today.
- TURNOVER SEAM reserved, not built: the loop carries prev_holdings across
  iterations (empty before D_0, set to the current selection at each iteration
  end) so the added/dropped set differences are computable at each rebalance for
  the future transaction-cost model.  No cost is applied today.
- The loop lives in src/research/portfolio.py alongside combine_period_returns
  (which it calls once per period); imports portfolio -> cross_sectional ->
  time_series_momentum, the same research -> strategies direction already used, no
  circular import.

**Verification:**
- Full suite 238 passed (was 229, +9).  Only two files changed: portfolio.py (two
  new functions + imports) and test_portfolio.py (9 tests + imports + a local bar
  helper copied from test_cross_sectional.py to stay hermetic).
  combine_period_returns byte-for-byte unchanged; no other file touched.
- The 9 tests cover: _per_bar_log_returns matching the engine convention
  (element 0 = 0.0, then log ratios) and rejecting empty bars; a two-symbol
  end-to-end run hand-traced across four rebalances (warmup -> cash, then the
  monthly winner alternating A/B/A with the absolute filter dropping a negative
  name); an all-cash period earning the cash rate; a partial-cash period (top_n=2,
  one qualifier) returning (1/2)*symbol + (1/2)*cash - THE test distinguishing
  rule (b) from rule (a); a no-lookahead test proving D_k's own bar return never
  enters the period; an interior-gap test proving the equal-length guard raises;
  and the reference-absent and too-few-month-ends guards.

**Blocked on:**
- Nothing.

**Next up:**
- The N=1 engine-equivalence test: prove rotation_backtest on a single
  always-held symbol (top_n=1, that symbol always positive) reproduces
  Backtester.run's per-bar return stream on that symbol bit-for-bit - the test
  that pins the reimplemented per-bar arithmetic against the engine and closes the
  Day-39 named risk of NEW-not-wrap drift.
- Then: the transaction-cost model at the turnover seam (fees/slippage on
  rebalance turnover - every rotation verdict must be issued AFTER costs), and the
  walk-forward bridge + rotation verdict through the existing harness
  (walk-forward, overfitting tax, Sharpe/Sortino, drawdown, after costs) vs an
  equal-weight-basket buy-and-hold = the Arc C payoff.
- Housekeeping (non-urgent): backfill the [fill in] time-spent placeholders in the
  Day 29-39 entries.

## Day 39

**Worked on:**
- Added combine_period_returns(per_symbol_returns, weights, n_bars,
  cash_per_bar_return=0.0) -> np.ndarray as a NEW module src/research/portfolio.py
  (commit "Add portfolio period-return combination primitive"), plus 8 hermetic
  tests in tests/test_portfolio.py.  This is the pure arithmetic core of the
  portfolio rotation backtester: one period's already-computed per-symbol per-bar
  return streams plus per-symbol weights go IN, one combined per-bar portfolio
  return stream comes OUT.  Purely additive - nothing imports it yet. [+8 tests]
- 229 tests green (was 221 at Day 38 close): +8.  Purely additive: no existing
  file touched.
- Locked the four architecture decisions for the portfolio backtester after a
  read-only inspection of engine.py / result.py / metrics.py / walk_forward.py
  (see Architectural note).

**Why it matters:**
- This is the first brick of the portfolio backtester, the centerpiece of Arc A
  (the rotation strategy).  The combiner is the piece the future rebalance loop
  calls once per period: at each month-end the loop will compute per-symbol returns
  (trailing_returns_at), rank them (rank_by_trailing_return), turn the held top-N
  into weights, and call this function to produce that period's portfolio return
  stream.  Building the pure arithmetic core first - provable with hand-computed
  numbers, touching no bars or engine - keeps the hard part (the rebalance loop and
  its lookahead/turnover correctness) separate from the arithmetic.

**Architectural note:**
- FOUR decisions locked after inspecting the engine.  (A) NEW multi-asset
  construct, NOT a wrap of the per-symbol Backtester.  Decisive reason: the engine's
  cost/turnover model is per-symbol (turnover = |diff(held)| sees only ONE symbol's
  position changes), so a per-symbol run() structurally CANNOT see cross-symbol
  rebalance turnover - dropping symbol A and adding symbol B at a rebalance is
  invisible to any single-symbol run().  Since every rotation verdict must
  eventually be issued AFTER costs, and costs attach to portfolio turnover, an
  architecture that cannot see portfolio turnover is disqualified.  Second strike:
  N disjoint per-symbol mini-runs would each carry the engine's structural index-0
  zero and a force-closed final trade, forcing seam-stripping N times per month (the
  same tax _stitch_oos already pays).  The NEW construct reuses metrics.py and the
  engine's ARITHMETIC CONVENTIONS (log returns np.log(closes[1:]/closes[:-1]), the
  per_bar_cash_yield formula) but not the run() call.  Named risk: NEW reimplements
  the log-return/cash arithmetic, so it must be pinned by an N=1 engine-equivalence
  test (a single-symbol portfolio must equal run() on that symbol) when the
  backtester is built - flagged, not built today; test 4 (single symbol, full
  weight, nonzero cash rate that must not leak) is the anchor for that future test.
- (B) Equal-weight (1/N), applied as an INPUT weight vector, not hard-coded in the
  summation - so an alternative weighting (rank- or vol-weighted) later slots in as
  a different weight vector, not a rewrite.  The combiner takes weights IN; WHO
  computes them is the caller's job (same separation as the ranker taking returns
  in).
- (C) The cash path reuses the engine's EXACT per-bar cash treatment:
  per_bar_cash_yield = log1p(annual_cash_yield)/annualization_factor, applied to the
  cash fraction.  So a rotation that goes to cash (ranker returned empty, the Day-36
  all-negative path) is scored on the identical cash convention the TSMOM verdict
  used - otherwise the rotation-vs-buy-and-hold comparison would be dishonest
  (different return on idle capital).  The combiner APPLIES a caller-supplied
  per-bar cash rate; it does not compute the conversion.
- (D) The turnover seam is RESERVED, not built.  The future rebalance loop will
  carry prev_holdings period-to-period so turnover (the set difference between last
  period's holdings and this period's) is computable, and the transaction-cost model
  attaches there later without a rewrite.  Not built today.
- The combiner's one unifying idea: cash_weight = 1.0 - sum(weights), seeded onto
  every bar before the held-symbol contributions are added.  That single formula
  handles full-invested (weights sum 1.0 -> no cash), all-cash (empty weights ->
  every bar the cash rate), and partial-cash (weights sum < 1.0 -> remainder earns
  cash) with NO branching.  It is a LINEAR combination in LOG-return space -
  deliberately NOT converted to simple returns - to stay consistent with how the
  engine, metrics.py, and _stitch_oos treat per-bar log returns as additive.
- The walk-forward bridge (Arc C, not today): walk_forward_validate is hard-wired to
  the single-symbol Strategy contract (generate_signals(list[OHLCVBar]) -> array).
  A multi-symbol portfolio does not fit it; the future bridge is at the
  per-fold-OOS-returns-array level (the portfolio emits a returns array, the same
  currency _stitch_oos consumes), NOT the Strategy.generate_signals level.  Today's
  slice does not preclude it - the combiner already produces exactly that currency.

**Verification:**
- Full suite 229 passed (was 221, +8).  Only two NEW files: src/research/portfolio.py
  and tests/test_portfolio.py; no existing file modified (verified additive -
  nothing imports the module yet, so no prior test could change status).
- The 8 tests cover: equal-weight fully-invested weighted sum; all-cash period
  earning the cash rate on every bar (empty holdings, no crash, length from n_bars);
  partial-cash remainder earning cash; single-symbol full-weight identity with a
  NONZERO cash rate that must not leak (the N=1 anchor); n_bars < 1 raises;
  array-length mismatch raises; key-set-inequality raises in BOTH directions (weight
  without returns, returns without weight); and the float64/length-n_bars output
  contract.

**Blocked on:**
- Nothing.

**Next up:**
- Build the rebalance loop (the next brick): walk the month-end rebalance dates, and
  at each one call trailing_returns_at -> rank_by_trailing_return -> compute equal
  weights -> combine_period_returns, carrying prev_holdings so the turnover seam
  exists.  Design questions to settle first (next session, before code): the exact
  month-end rebalance schedule the loop iterates; and the partial-cash weighting
  decision - when the ranker returns fewer than top_n symbols, do the held symbols
  split full capital (1/held each) or does each get 1/top_n with the unfilled slots
  sitting in cash.  Both are honest; it is a real weighting choice that changes
  returns, so decide it deliberately.
- Then: the N=1 engine-equivalence test (prove the NEW construct's per-bar
  arithmetic matches run() on a single always-held symbol), the transaction-cost
  model at the turnover seam, and finally the walk-forward bridge + rotation verdict
  through the existing harness (walk-forward, overfitting tax, Sharpe/Sortino,
  drawdown, AFTER costs) vs an equal-weight-basket buy-and-hold = the Arc C payoff.
- Housekeeping (non-urgent): commit README.md separately; backfill the [fill in]
  time-spent placeholders in the Day 29-38 entries.

## Day 38

**Worked on:**
- Added trailing_returns_at(bars_by_symbol, rebalance_ts, lookback,
  price_field="close") -> dict[str, float] as a SECOND function in
  src/research/cross_sectional.py (commit "Add cross-sectional trailing-returns-at
  returns computer"), plus 6 new hermetic tests in tests/test_cross_sectional.py.
  This is the second brick of the rotation arc: the returns-COMPUTER that turns the
  basket's bars into the dict[str, float] that Day 36's rank_by_trailing_return
  consumes.  It computes each symbol's trailing return via the shared
  trailing_return_series helper extracted Day 37, so cross-sectional and
  time-series momentum measure "trailing return" identically and cannot drift.
  [+6 tests]
- 221 tests green (was 215 at Day 37 close): +6.  Purely additive:
  rank_by_trailing_return untouched, the existing 10 cross_sectional tests and the
  ~26 TSMOM tests all unchanged in status.

**Why it matters:**
- This is the bridge between "I have bars" and "I have a ranking": at each
  rebalance date the future portfolio backtester will call trailing_returns_at to
  get the returns dict, then hand it to rank_by_trailing_return to get the held
  symbols.  With both halves now in place and connected end-to-end (a test feeds
  one straight into the other), the compute-returns -> rank pipeline is complete -
  the portfolio backtester that drives them is the next brick.

**Architectural note:**
- The rebalance point is a TIMESTAMP, not an integer index.  The 17 ETFs have
  different-length histories, so an integer index would land on a DIFFERENT
  calendar date per symbol - a silent lookahead-style bug.  A timestamp is safe
  because every 1d bar is floored to midnight UTC on ingest (the dedup migration
  guarantees this), so symbols align by date equality.  "As of D" is resolved on
  each symbol's OWN month-end grid - the most recent month-end at-or-before D - via
  s.loc[:rebalance_ts].iloc[-1], which is right-inclusive of D (a rebalance date
  that IS a month-end selects itself, never one after it, so no lookahead).  Two
  symbols with month-ends on different dates each resolve to their own
  most-recent-at-or-before value at the same rebalance date.
- Two design points pinned by test because they silently corrupt the returns dict:
  (a) the as-of selection uses the explicit .loc[:D].iloc[-1] slice, NOT
  Series.asof - asof returns the last NON-NaN value at-or-before D, which would
  skip a warmup NaN and hand back a stale earlier month-end instead of correctly
  omitting the symbol; (b) two OMIT guards in a fixed order - the empty-slice guard
  (len == 0, symbol has no month-end at-or-before D) MUST come before the NaN guard
  (iloc[-1] is a warmup NaN), because .iloc[-1] on an empty slice raises IndexError
  so there is nothing to NaN-check yet.  A symbol failing either guard is omitted
  entirely, never emitted with NaN - so the dict handed to the ranker contains only
  eligible symbols with finite returns, and the ranker's own NaN backstop stays
  defense-in-depth rather than the primary mechanism.  Tests 3 (non-empty slice,
  warmup NaN) and 4 (empty slice) are deliberately distinct scenarios so both
  guards are exercised separately.

**Verification:**
- Full suite 221 passed (was 215, +6).  Only two files changed: cross_sectional.py
  (3 imports + the new function) and test_cross_sectional.py (6 tests + imports +
  docstring note).  rank_by_trailing_return is byte-for-byte unchanged; no other
  file touched (the research->strategies import is one-directional and
  non-circular).
- The 6 new tests cover: hand-computed as-of returns at a shared month-end; a
  too-short symbol omitted via the empty-slice guard; a warmup-NaN symbol omitted
  via the NaN guard (distinct from the empty path); a rebalance date before any
  month-end yielding an empty dict with no IndexError; each symbol resolving on its
  own month-end grid (A on its March month-end, B - no March bar - on its February
  one, at the same rebalance date); and the output dict fed straight into
  rank_by_trailing_return, proving the returns-computer output is valid ranker
  input.
- One code commit plus this DEV_LOG entry.

**Blocked on:**
- Nothing.

**Next up:**
- Build the portfolio-level backtester (the centerpiece of Arc A): one combined
  equity curve across the held symbols with a monthly rebalance, driving the
  compute-returns -> rank pipeline now complete.  At each month-end it will call
  trailing_returns_at to get the returns dict, rank_by_trailing_return to pick the
  held top-N, and produce a single portfolio return stream.  Open design questions
  to settle first (next session, before code): equal-weight vs other weighting
  across held symbols; how cash is handled when the ranker returns fewer than
  top_n (or empty); and how the monthly-rebalance turnover will later carry the
  transaction-cost model that every verdict needs before it is issued.
- After that: the walk-forward wrapper around the portfolio backtester, then the
  rotation verdict through the existing harness (walk-forward, overfitting tax,
  Sharpe/Sortino, drawdown, after costs) vs an equal-weight-basket buy-and-hold.
- Housekeeping (non-urgent): commit README.md separately; backfill the [fill in]
  time-spent placeholders in the Day 29-37 entries.

## Day 37

**Worked on:**
- Extracted trailing_return_series(bars, lookback, price_field="close") ->
  pd.Series as a module-level helper in src/strategies/time_series_momentum.py
  (commit "Extract trailing_return_series shared helper from TSMOM"), and rewired
  TimeSeriesMomentumStrategy.generate_signals to call it.  Added one direct unit
  test in tests/test_strategies.py (section 13).  This is a BEHAVIOR-PRESERVING
  refactor: the trailing-return arithmetic that was inline in generate_signals now
  lives in ONE shared helper that both TSMOM and (next session) the cross-sectional
  returns-computer will call. [+1 test]
- 215 tests green (was 214 at Day 36 close): +1.  The change is a pure internal
  move plus one new test - nothing imports the new helper yet.

**Why it matters:**
- The cross-sectional rotation strategy must compute each symbol's trailing return
  the SAME way TSMOM does, or the two momentum strategies become two
  subtly-different definitions of "momentum" and every verdict comparing them is
  invalid.  Extracting ONE shared definition guarantees they cannot drift - the
  identical rationale behind extracting month_end_indices (Day 33), which
  optuna_fit.py already imports.  trailing_return_series is the natural next member
  of that shared-primitive family.  Building it as a standalone, proven helper
  FIRST - before the module that consumes it - is the same discipline used for
  month_end_indices and walk_forward_splits: the harder consumer later builds on a
  piece already known correct.

**Architectural note:**
- The extraction boundary is arithmetic-only: the helper owns the close_series
  construction, the month_end_indices call, the month_end_close selection, and the
  month_end_close / month_end_close.shift(lookback) - 1.0 simple-return arithmetic.
  It deliberately does NOT own the empty-bars guard, the M <= lookback guard, the
  > 0.0 signal mapping, or the ffill/fillna - those stay in generate_signals, so
  the helper is a pure returns primitive with no strategy-specific policy baked in.
  The one rewiring subtlety: generate_signals previously built close_series once and
  used it BOTH for the month-end selection (now inside the helper) AND as the daily
  ffill target.  So generate_signals now rebuilds daily_index =
  pd.DatetimeIndex([bar.timestamp for bar in bars]) directly - byte-equal to the old
  close_series.index (same timestamps, same order) - and the monthly_signal index
  uses trailing_return.index, which IS the old month_end_close.index because
  ratio-minus-1 preserves the pandas index.  Both are plumbing, not arithmetic, so
  no second definition of anything meaningful was created.
- Why extract rather than duplicate: duplicating the arithmetic into the new module
  would have been purely additive (touching nothing), but would leave two
  definitions of trailing return that could silently drift if someone later changed
  one.  Consistency-with-TSMOM is the entire reason the cross-sectional module
  exists, so a shared definition is the correct trade even though it touches the
  most-tested file in the repo.

**Verification:**
- Full suite 215 passed (was 214, +1).  Only two files changed:
  time_series_momentum.py (helper + rewire) and test_strategies.py (one new test).
  No other file touched - the refactor has no external consumer yet.
- The ~26 TSMOM tests across test_strategies.py, test_optuna_fit.py,
  test_momentum_overfitting_tax.py, and test_momentum_walkforward.py all stayed
  bit-for-bit green.  None of them reads trailing_return directly - they assert on
  the strategy's OUTPUT signals across uptrend, downtrend, flat, warmup,
  no-lookahead, month-end cadence, and the adj_close-vs-close divergence - so their
  staying green is the proof the arithmetic did not move.  The new direct test pins
  the extracted helper's arithmetic independently: returns a pd.Series, length
  equals the month-end count, first lookback entries NaN (warmup), and a specific
  non-warmup month-end equals the hand-computed ratio-minus-1 (120/100 - 1 = 0.20).
- One code commit plus this DEV_LOG entry.

**Blocked on:**
- Nothing.

**Next up:**
- Build the cross-sectional returns-computer (next brick of Arc A): a new function
  trailing_returns_at(bars_by_symbol, rebalance_ts, lookback, price_field="close")
  -> dict[str, float] as a SECOND function in src/research/cross_sectional.py,
  computing each symbol's trailing return via the now-shared trailing_return_series
  helper and producing the dict[str, float] that feeds rank_by_trailing_return.
  Design locked this session: (a) the rebalance point is a TIMESTAMP, not an integer
  index - the 17 ETFs have different-length histories, so an integer index would be
  a different date per symbol (a silent lookahead-style bug); a timestamp is safe
  because the dedup migration floors every 1d bar to midnight UTC, so symbols align
  by date equality; (b) "as of date D" means each symbol's OWN most-recent month-end
  at-or-before D via an explicit .loc[:D].iloc[-1] slice (not .asof, which silently
  skips NaN); (c) the returns-computer OMITS ineligible symbols (fewer than
  lookback+1 month-ends) AND NaN-valued symbols from the returned dict entirely, so
  the ranker only ever sees eligible symbols with finite returns.
- After that: the portfolio-level backtester (one combined equity curve across held
  symbols, monthly rebalance), then the walk-forward wrapper, then the rotation
  verdict through the existing harness (walk-forward, overfitting tax,
  Sharpe/Sortino, drawdown, after costs) vs an equal-weight-basket buy-and-hold.
- Housekeeping (non-urgent): commit README.md separately; backfill the [fill in]
  time-spent placeholders in the Day 29-36 entries.

## Day 36

**Worked on:**
- Added src/research/cross_sectional.py with
  rank_by_trailing_return(trailing_returns: dict[str, float], top_n: int,
  hold_when_all_negative: bool = False) -> list[str] (commit "Add cross-sectional
  trailing-return ranking function"), plus tests/test_cross_sectional.py.  This is
  the FIRST brick of the rotation arc: a pure, dumb ranking/selection primitive
  that takes already-computed trailing returns IN and returns the selected top-N
  symbols OUT.  It computes no returns, touches no bars/dates/DuckDB/engine, and
  does NOT subclass Strategy (the Strategy contract is single-symbol
  list[OHLCVBar] -> array; ranking is inherently multi-symbol dict[str, float] ->
  list[str]). [+10 tests]
- 214 tests green (was 204 at Day 35 close): +10, all in the new
  test_cross_sectional.py.  Purely additive: two NEW files, no existing file
  touched.

**Why it matters:**
- Cross-sectional rotation ranks symbols AGAINST EACH OTHER and holds the
  strongest handful - the defining move that time-series momentum (which judges
  each symbol only against its own past) cannot express.  It structurally breaks
  the per-symbol Strategy contract, so the rotation arc needs a portfolio-level
  backtester built over several days.  Building the pure ranking primitive FIRST -
  isolated and unit-tested - mirrors how walk_forward_splits was built and proven
  before the validator that consumed it, so the harder portfolio engine later
  consumes a piece already known correct.

**Architectural note:**
- The seam is deliberately pure: returns-in / selection-out.  Separating "compute
  a trailing return" from "rank and select" keeps this function trivially testable
  with plain dicts and keeps the return-computation concern (which must match
  TSMOM's trailing-return arithmetic) in a SEPARATE later module.  Two design
  decisions were made and locked today: (a) DEFER the trailing-return computation -
  today's file computes no returns, so the TSMOM extract-vs-duplicate question
  (whether to share TSMOM's inline month_end_close / shift(lookback) - 1.0
  arithmetic or duplicate it) is pushed to the module that will actually feed this
  function, where the needed shape will be known; (b) the all-negative-basket
  behavior is a PARAMETER hold_when_all_negative defaulting to False (absolute
  filter ON - an all-<=0 basket returns empty = cash), matching TSMOM's own strict
  > 0.0 go-flat discipline, with True available to test pure always-invested
  relative strength.  Parameterizing lets the harness test both rather than betting
  on one.
- Two correctness points were pinned by test because they silently corrupt a
  backtester: the tie-break is deterministic by MECHANISM (sorted key = (-return,
  symbol), so equal returns resolve alphabetically by the KEY, never by dict
  insertion order - the test uses reversed insertion order to prove it), and
  NaN-valued returns are dropped up front in BOTH filter modes by an explicit
  math.isnan guard (not as a byproduct of the > 0.0 filter, since hold mode has no
  such filter), so a NaN can never sort to an arbitrary position or be selected.

**Verification:**
- Full suite 214 passed (was 204, +10).  Only the two new files changed; no
  existing module or test touched (additive - nothing imports the new module yet).
- New-file verbose run: all 10 tests green, covering basic top-N ordering,
  alphabetical tie-break (reversed insertion order), over-large top_n returns all,
  top_n < 1 raises, empty input, absolute filter excluding negatives, all-negative
  -> cash (default), all-negative -> least-bad (hold mode), exactly-zero filtered
  by strict > 0.0, and NaN never selected in either mode.
- One code commit plus this DEV_LOG entry.

**Blocked on:**
- Nothing.

**Next up:**
- The next brick of Arc A is the returns-computation module: given the basket's
  bars and a rebalance point, produce the dict[str, float] of trailing returns that
  feeds rank_by_trailing_return, computing each symbol's trailing return the SAME
  way TSMOM does.  This is where the deferred extract-vs-duplicate decision gets
  made (extracting TSMOM's inline trailing-return arithmetic into a shared helper
  is the DRY choice but touches time_series_momentum.py, so it is its own
  behavior-preserving change - likely its own day).  After that: the
  portfolio-level backtester (one combined equity curve across held symbols,
  monthly rebalance), then the walk-forward wrapper, then the rotation verdict
  through the existing harness (walk-forward, overfitting tax, Sharpe/Sortino,
  drawdown, after costs) vs an equal-weight-basket buy-and-hold.
- Housekeeping (non-urgent): commit README.md separately; backfill the [fill in]
  time-spent placeholders in the Day 29-32 entries.

## Day 35

**Worked on:**
- Surfaced the Sortino ratio in scripts/momentum_walkforward.py's verdict table
  (commit "Surface Sortino columns in fixed-momentum verdict table"), reading
  oos_sortino/bh_sortino which already existed on WalkForwardResult (added
  Day 34).  Added oos_sortino, bh_sortino, and a computed sortino_delta =
  oos_sortino - bh_sortino as VerdictRow fields (each grouped next to its Sharpe
  sibling); threaded the two new params through summarize_verdict's signature and
  its main() call site; added OOS Sortino and B&H Sortino columns to the table
  (rule widened 86 -> 112), the MEAN row, and the per-symbol progress line
  (Δsortino); and added an n_beat_sortino bottom-line tally next to n_beat_sharpe.
  CLI-only change: no metric, engine, or WalkForwardResult code touched - the
  fields were already populated. [test updated in place, count unchanged]
- 204 tests green (unchanged from Day 34 close): the summarize_verdict test gained
  sortino_delta assertions in place; the positional-lockstep was proven by the
  existing sharpe_delta/dd_reduction assertions still passing with their original
  values after the two new params were inserted.

**Why it matters:**
- This closes the Sortino thread Day 34 opened: the metric existed and was correct
  but printed nowhere, so the verdict could not yet be read through the
  downside-only lens.  Now it can.  Sortino is the most charitable metric for a
  defensive trend-follower (it does not penalise upside volatility as risk), so it
  was the fair test of whether momentum's Sharpe deficit was real underperformance
  or a Sharpe artifact.

**Architectural note:**
- summarize_verdict is called POSITIONALLY at three sites (signature, main() call
  site, test).  Inserting oos_sortino/bh_sortino after their Sharpe siblings shifts
  every following positional arg, so all three had to move in lockstep or a value
  silently misassigns.  The proof it was done right: the existing sharpe_delta
  (+0.5) and dd_reduction (+0.15 / -0.10) assertions still pass with their ORIGINAL
  expected values after the insertion - those deltas are computed from the args
  after the sortino params, so unchanged results prove nothing drifted.
- sortino_delta is a computed FIELD shown in the progress line and used by the
  tally, but is NOT a table column - adding a ΔSortino column would push the table
  past ~120 chars; the two raw Sortino columns beside their Sharpe siblings are the
  readable side-by-side comparison, and ΔSharpe alone represents the delta axis in
  the table.
- The table lives in main(), which has NO hermetic test, so column alignment was
  verified by eye on a --symbols SPY,TLT run (header, data rows, and MEAN row all
  aligned under the 112-char rule) rather than by pytest.

**Verification:**
- Full suite 204 passed (unchanged).  Only scripts/momentum_walkforward.py and
  tests/test_momentum_walkforward.py changed; walk_forward.py, metrics.py, the
  engine, and momentum_overfitting_tax.py untouched.
- SPY,TLT alignment eyeball: columns rendered aligned, non-Sortino numbers
  (oos_sharpe/bh_sharpe/dd_cut) matched the Day-32 fixed verdict exactly,
  confirming no existing number moved.
- FULL-BASKET SORTINO VERDICT (all 17 ETFs, adj_close, frictionless,
  train=756/test=252): momentum beat B&H on Sharpe on 4/17, on Sortino on 3/17
  (ONE FEWER, not more), and cut max drawdown on 13/17 (unchanged from Day 32).
  MEAN OOS Sharpe +0.43 vs B&H +0.52 (ΔSharpe -0.09); MEAN OOS Sortino +0.59 vs
  B&H +0.72 (a -0.13 gap - WIDER than the Sharpe gap).  The Sortino lens did NOT
  rescue momentum: on nearly every symbol the ΔSortino is more negative than the
  ΔSharpe, because both strategy and B&H have upside-concentrated volatility so the
  downside-only denominator magnifies both ratios, widening the absolute gap when
  B&H already leads on return.  The three Sortino wins are the same defensive
  corner as always: IEF (+0.25, standout - Sortino 0.77 vs 0.52, drawdown
  24%->10%), GLD (+0.05), EEM (+0.05).

**Blocked on:**
- Nothing.

**Next up:**
- The momentum verdict is now complete on every lens: fixed TSMOM(12) loses to B&H
  on risk-adjusted return (Sharpe 4/17, Sortino 3/17 - both confirm
  underperformance), and its durable value is drawdown reduction (13/17), with real
  return-edge only in bonds/gold.  Tuning the lookback adds no OOS edge (Day 33
  overfitting tax).  This closes momentum.
- The natural next direction is cross-sectional sector / cross-asset rotation on
  the ETF basket (rank symbols against each other, rotate into the strongest) -
  this breaks the current per-symbol Strategy contract and needs a portfolio-level
  backtester, a real architectural step up and likely a multi-day effort.
- Housekeeping (non-urgent): commit README.md separately; backfill the [fill in]
  time-spent placeholders in the Day 29-32 entries.

## Day 34

**Worked on:**
- Added downside_deviation(returns, annualization_factor=252, target=0.0) and
  sortino_ratio(returns, annualization_factor=252, target=0.0) to metrics.py
  (commit "Add downside-deviation and Sortino metrics to walk-forward output"),
  mirroring sharpe_ratio's shape.  The four existing metrics (total_return,
  sharpe_ratio, max_drawdown, win_rate) and _ZERO_STD_TOLERANCE are byte-for-byte
  unchanged; the two new functions are purely additive.  sortino_ratio CALLS
  downside_deviation, so there is one authoritative definition of the
  denominator. [+6 tests]
- Threaded oos_sortino and bh_sortino onto WalkForwardResult: _stitch_oos now
  returns a 6-tuple (sortino appended last, computed from the SAME stitched
  oos_returns and SAME annualization_factor as the Sharpe directly above it), both
  unpack sites (strategy path and B&H path) take the trailing name, the frozen
  dataclass gained two REQUIRED fields (no defaults), and the single construction
  site sets both.  So Sortino is populated everywhere Sharpe is, on both the
  strategy and buy-and-hold paths. [+1 test, an end-to-end field-population check]
- 204 tests green (was 197 at Day 33 close): +7 (six pure-metric tests in
  test_metrics.py, one end-to-end field-population test in test_walk_forward.py).

**Why it matters:**
- Sharpe penalises upside volatility as if it were risk, which misjudges a
  defensive trend-follower like TSMOM that sits flat in downturns and rides trends
  up.  Sortino divides return by DOWNSIDE deviation only (the volatility of
  below-target returns), so it does not punish a strategy for its good months.
  This is the honest lens for the "loses on Sharpe, wins on drawdown" momentum
  verdict: it will show whether momentum's Sharpe deficit is real underperformance
  or a Sharpe artifact from penalised upside.  The metric is now computed and
  stored on every WalkForwardResult; reading it into the momentum verdict is a
  later step.

**Architectural note:**
- Annualisation algebra (the subtle spot): downside_deviation returns the
  ANNUALISED downside deviation (per-bar dd * sqrt(annualization_factor)), so
  sortino_ratio annualises the RATIO by multiplying by the FULL
  annualization_factor, NOT sqrt.  Derivation: sharpe = (mean / per_bar_std) *
  sqrt(af); here dd = per_bar_dd * sqrt(af) already, so mean/dd =
  (mean/per_bar_dd)/sqrt(af), and * af yields (mean/per_bar_dd)*sqrt(af), matching
  sharpe's form.  Using sqrt(af) instead would under-annualise Sortino by ~16x
  (sqrt 252) and make it silently incomparable to the Sharpe column.  The
  derivation is commented in the code and pinned by
  test_sortino_matches_hand_computation.
- Downside deviation divides the sum of squared below-target deviations by N
  (TOTAL observations, population RMS, ddof=0), NOT by the below-target count and
  NOT ddof=1.  This is the published-Sortino convention (comparable to how Sortino
  is normally reported) and is DELIBERATELY different from sharpe's ddof=1 sample
  std - a code comment says so explicitly so nobody "fixes" it.
- Zero-downside guard mirrors sharpe's zero-variance guard EXACTLY: when every
  return is at or above target there are no below-target deviations, so downside
  deviation is 0.0 and a naive divide would be inf/nan.  sortino_ratio returns 0.0
  (NOT nan) via the SAME _ZERO_STD_TOLERANCE constant, guard-before-divide, so no
  downstream table ever prints a nan.  Also mirrors sharpe's len<2 short-circuit
  to 0.0.
- WalkForwardResult fields are REQUIRED, not defaulted: grep confirmed exactly ONE
  construction site (walk_forward.py) and zero hand-constructions in tests, so a
  required field is safe and prevents a future construction site silently omitting
  Sortino (a default 0.0 would let that pass unnoticed).

**Verification:**
- Full suite 204 passed (was 197), only the pre-existing websockets
  DeprecationWarning.  All 197 prior tests green unchanged (the change is purely
  additive - no existing metric recomputed, no existing field moved).
- New tests pin the load-bearing behaviour: a by-hand downside_deviation
  known-value check (divides by N, not the below-target count); the
  all-above-target -> 0.0 case; the CRITICAL zero-downside -> 0.0-not-nan guard
  test; the len<2 -> 0.0 short-circuit; test_sortino_exceeds_sharpe_when_upside_
  volatile (Sortino > Sharpe on an upside-volatile array - proves the metric
  measures something different from Sharpe); test_sortino_matches_hand_computation
  (pins the * annualization_factor algebra); and an end-to-end test that
  oos_sortino/bh_sortino are finite floats on both paths.
- Smoke check: sortino_ratio on an all-positive array printed 0.0 (not nan),
  confirming the zero-downside guard fires end-to-end.
- One code commit plus this DEV_LOG entry.

**Blocked on:**
- Nothing.

**Next up:**
- Surface Sortino in the momentum CLIs (deferred from today):
  momentum_walkforward.py's table is on a "="*86 rule with 7 columns and needs a
  widened rule plus two new VerdictRow fields threaded through summarize_verdict;
  momentum_overfitting_tax.py builds rows from the imported TaxRow (no Sortino
  field), so surfacing there means editing the shared SMA-path helper - out of
  scope for the metric commit.  The fields EXIST on WalkForwardResult now; printing
  them is a separate later commit.
- Then re-read the momentum verdict through the Sortino lens: does momentum's
  Sharpe deficit (lost to B&H on Sharpe 13/17) shrink under Sortino, or is the
  return shortfall genuine?  Read what it says - do not assume Sortino rescues the
  strategy.
- Later / bigger: cross-sectional sector and cross-asset rotation on the ETF
  basket (needs a portfolio-level backtester - breaks the current per-symbol
  Strategy contract, a real architectural step up).
- Housekeeping (non-urgent): commit README.md separately; backfill the [fill in]
  time-spent placeholders in recent DEV_LOG entries.

## Day 33

**Worked on:**
- Extracted month_end_indices(bars) into a shared module-level helper in
  time_series_momentum.py and rewired generate_signals to use it (commit "Extract
  month_end_indices helper from time-series momentum").  Pure no-op refactor: the
  inline month-code / boolean-mask block became one helper call
  (close_series.iloc[mei] selects byte-identically to the old
  close_series[boolean_mask]), so every existing TSMOM signal is unchanged.  WHY
  extract: the next commit's fitter needs the warm-up boundary computed from the
  SAME month-end definition the strategy uses, so they cannot drift — the same
  structural-agreement reasoning behind _stitch_oos. [+1 test, an honest by-hand
  index check; all prior green is the no-op proof]
- Added make_tsmom_optuna_fit_fn to optuna_fit.py (commit "Add
  time-series-momentum Optuna fitter"), the momentum analogue of
  make_sma_optuna_fit_fn.  Additive: the SMA fitter is byte-for-byte unchanged.
  It tunes the SINGLE lookback (not fast/slow), threads ONE price_field variable
  into BOTH the internal in-sample Backtester AND every trial's
  TimeSeriesMomentumStrategy (so in-sample scoring matches the OOS basis by
  construction; the Day-32 guard is a backstop, never the mechanism — the one
  real divergence from the SMA fitter, which has no basis knob), clamps the
  lookback upper bound to min(lookback_range[1], n_month_ends - 1) so a trial can
  never trip TSMOM's M>lookback guard mid-search, and scores warm-only in-sample
  Sharpe sliced at the warm-up BOUNDARY (mei_train[lookback]) — NOT the first
  non-flat signal. [+6 tests]
- Added scripts/momentum_overfitting_tax.py (commit "Add momentum overfitting-tax
  CLI"), the momentum analogue of overfitting_tax.py: fixed TSMOM(12) vs per-fold
  Optuna-tuned lookback across the etf_basket, reporting the overfitting tax.  It
  constructs ONE frictionless adj_close Backtester and passes it as backtester= to
  BOTH the fixed and fitted walk_forward_validate calls (overfitting_tax.py passes
  none and rides the close default), threads price_field=adj_close into the fixed
  strategy and the fitter too, defaults --train 756 / --test 252 (matching the
  fixed momentum verdict's windows so the tax is comparable, NOT
  overfitting_tax.py's 504/126), and reuses the pure summarize_tax / TaxRow by
  import. [+2 tests]
- 197 tests green (was 188 at Day 32 close).

**Why it matters:**
- This completes the machinery for the momentum verdict's second half: the fixed
  TSMOM(12) verdict landed Day 32, and this adds the honest "does tuning the
  lookback add OOS edge or just overfit?" comparison on the same adj_close basis.
  The shared month-end helper makes the fitter's in-sample warm-up boundary agree
  with the strategy's actual warm-up by construction, not by two copies that could
  silently diverge and corrupt the tax.

**Architectural note:**
- The warm-only in-sample slice is the warm-up BOUNDARY (the lookback-th
  month-end's bar index), NOT the first non-flat signal.  WHY: momentum
  legitimately sits FLAT after warm-up whenever the trailing return is negative (a
  downtrend), and those flat bars are REAL positions the OOS window also scores —
  slicing past them to the first LONG would inflate the in-sample Sharpe and
  corrupt the tax in any fold starting in a downtrend.  The boundary slice keeps
  long AND legitimate-flat post-warm-up bars, matching how the validator scores
  OOS.  A dedicated test pins this: on a decline-then-recover path that leads with
  post-warm-up flats, the recorded Sharpe equals the boundary slice and differs
  from the first-non-flat slice (it would fail if the fitter sliced at the first
  long).
- In-sample stays FRICTIONLESS on both the fitter and the tax CLI (price_field
  only, no cost / yield), so the tax isolates parameter-selection overfitting, not
  cost drag — exactly as the SMA tax does.
- Sibling CLI over a --strategy flag on overfitting_tax.py: that file is
  SMA-specific at every layer (hardcoded SMACrossoverStrategy(50,200),
  make_sma_optuna_fit_fn), so a flag would branch every line; a sibling keeps each
  CLI single-purpose, matching the momentum_walkforward.py precedent.
- Three commits split helper → fitter → CLI so the only change to verdict-central
  working code (the month_end_indices extraction) is its own
  trivially-verifiable no-op commit.

**Verification:**
- Full suite 197 passed (was 188), only the pre-existing websockets
  DeprecationWarning.  The no-op refactor kept all prior TSMOM tests green
  unchanged; the SMA fitter and its tests are untouched.
- A negative-control test proves the basis-matching is load-bearing: a close-basis
  engine paired with an adj_close strategy RAISES the Day-32 guard, on the same
  bars the matched run scores cleanly.
- CLI smoke on SPY (--n-trials 3): 15 folds, fixed=+0.67, fitted=+0.67, B&H=+0.78,
  in-sample=+1.03, tax=+0.36 — the fixed +0.67 and B&H +0.78 reproduce the fixed
  momentum verdict's SPY row exactly, confirming the sibling is on the same
  adj_close basis and 756/252 windows as the verdict.  Early read: tuning the
  lookback matched fixed TSMOM(12) OOS and bought nothing, the SMA overfitting
  story repeating — to be confirmed on the full basket.
- Three code commits plus this DEV_LOG entry.

**Blocked on:**
- Nothing.

**Next up:**
- Run the full-basket momentum tax (all 17 etf_basket symbols, the multi-minute
  Optuna sweep) and read whether tuning the lookback adds OOS edge or just overfits
  the way the SMA did — the completion of the momentum verdict.
- Optional: re-fetch the basket to un-stale it (last bar 2026-06-09); backfill the
  [fill in] time-spent placeholders in recent entries.
- Later: a true downside-deviation / Sortino metric (WalkForwardResult exposes
  MaxDD only today); cross-sectional sector / cross-asset rotation (needs a
  portfolio-level backtester).

## Day 32

**Worked on:**
- Added a price-basis consistency guard to walk_forward_validate (commit "Add
  price-basis consistency guard to walk-forward validation"). It raises ValueError
  when the engine's price_field disagrees with the running strategy's price_field,
  so a caller can no longer silently mix price-return signals with total-return
  scoring. It checks fold_strategy (NOT the passed-in strategy), so it covers BOTH
  the fit_fn=None path and the fit_fn path; it is hasattr-safe via
  getattr(fold_strategy, "price_field", None), so strategies without a price_field
  (SMACrossoverStrategy) are skipped, never crashed; and it sits immediately after
  per-fold strategy selection and before generate_signals - fail-fast, before any
  signal math runs. [+4 tests, including a fit_fn-path mismatch]
- Added the fixed time-series-momentum walk-forward verdict runner,
  scripts/momentum_walkforward.py (commit "Add fixed time-series-momentum
  walk-forward verdict runner"). This is the FIRST entry point that runs
  TimeSeriesMomentum through walk_forward_validate - grep confirmed none existed,
  the prior CLIs were SMA-only. It constructs exactly ONE Backtester and a
  TimeSeriesMomentumStrategy from a SINGLE price_field variable, so engine and
  strategy bases are equal by construction and the new guard is a backstop here,
  never the mechanism. It reads the built-in always-long buy-and-hold benchmark.
- The runner reports per symbol: OOS Sharpe, B&H Sharpe, MaxDD on both sides, and
  dd_reduction = bh_max_dd - oos_max_dd (POSITIVE = momentum had the SMALLER
  drawdown, i.e. cut risk), plus a cross-symbol MEAN row and bottom-line tallies
  ("beat B&H on Sharpe on X/N; cut max drawdown on Y/N"). Split into a pure
  summarize_verdict helper + a thin run_one_symbol for hermetic testing. [+4 tests]
- Flags: --lookback (12), --train (756), --test (252), --step (None),
  --price-field (adj_close), --cash-yield (0.0), --fee-bps (0.0), --slippage-bps
  (0.0), --symbols.
- 188 tests green (was 180).

**Why it matters:**
- This is the verdict-path wiring: the guard closes the footgun Days 30 and 31
  both deferred - price basis lived in two disconnected places (engine and
  strategy) with nothing forcing them to agree - and the runner is the first real
  caller that passes a basis to both at once. With a FIXED 12-month rule, the
  runner is already a genuine partial verdict on whether momentum beats holding,
  out-of-sample and after accounting, with no Optuna involved yet.

**Architectural note:**
- Verdict-first reorder: for momentum the FIXED 12-month rule is the headline
  question (unlike the SMA dummy, where no single parameter had a special claim),
  so the fixed runner is a real partial verdict without needing per-fold tuning.
  Tuning + the overfitting tax are the NEXT step, not this one.
- The guard was folded into Day 32 rather than shipped standalone because the
  runner is its first real caller - the first place a basis is passed to both
  engine and strategy - so the safety ships with the code that first needs it.
- Window sizing for a MONTHLY 12-month-lookback strategy: train=756 (~3y) warms
  the 12-month lookback with margin so no FLAT warmup bleeds into the test window
  and TSMOM's own ">lookback month-end observations" guard cannot trip; test=252
  (~1y, ~12 monthly rebalances per fold); step=None (non-overlapping). On a
  full-history basket symbol (~4,600 bars from 2008) this yields ~15 folds.
- --cash-yield defaults to 0.0 because B&H is always-long and so earns ZERO
  flat-yield by construction - a non-zero yield only lifts the strategy side. That
  is a legitimate effect (idle capital earns interest) but a cash-rate assumption,
  so the headline verdict makes none; bracket it later at 0.04.
- --price-field defaults to adj_close (the honest total-return basis): verified in
  DuckDB that adj_close is genuinely dividend-adjusted, not a copy of close -
  SPY/TLT/XLU/XLP show 2008 adj_close materially below close and the latest bar
  equal, the signature of correct back-adjustment.

**Verification:**
- Full suite 188 passed (was 180), only the pre-existing websockets
  DeprecationWarning. The prior 180 are unchanged, proving the guard is a clean
  no-op on every existing path and the new runner touched nothing shared.
- Guard tests: a fit_fn=None mismatch raises, a matched basis runs clean, a
  strategy without price_field (SMA) is skipped via the getattr-None branch, and a
  fit_fn-path mismatch raises - proving the guard reads fold_strategy, not the
  ignored passed-in strategy.
- Runner tests are fully hermetic (synthetic in-memory bars, no DuckDB):
  summarize_verdict sign conventions, run_one_symbol returns a WalkForwardResult
  end-to-end, a basis-matched run never trips the guard, and a positive
  --cash-yield never lowers OOS return (lifts it when bars go flat).
- Two code commits ("Add price-basis consistency guard to walk-forward
  validation", "Add fixed time-series-momentum walk-forward verdict runner") plus
  this DEV_LOG entry.

**Blocked on:**
- Nothing.

**Next up:**
- Run the fixed verdict against the basket (adj_close, then a --cash-yield 0.04
  bracket) and read it on MaxDD / drawdown reduction, not Sharpe alone -
  crisis-avoidance is what momentum is meant to deliver.
- Build make_tsmom_optuna_fit_fn (additive, optuna_fit.py only) and add a momentum
  overfitting-tax column: the FULL verdict on whether tuning the lookback adds edge
  or just overfits the way the SMA did.
- Later: a true downside-deviation / Sortino metric (WalkForwardResult exposes
  MaxDD only today); cross-sectional sector / cross-asset rotation (needs a
  portfolio-level backtester).

## Day 31

**Worked on:**
- Added cash-on-flat accounting to the Backtester: on bars where the held
  position is FLAT (out of the market), idle capital now earns interest instead
  of nothing. New annual_cash_yield parameter on Backtester.__init__, last
  positional, defaulting to 0.0; validated >= 0 with a ValueError and no silent
  fallback (mirrors the fee/slippage guards).
- The rate is an ANNUAL simple-interest fraction (0.04 = 4%/yr), converted
  internally to a per-bar log return: self.per_bar_cash_yield = log(1 +
  annual_cash_yield) / annualization_factor. Storing it annual (not per-bar)
  means a caller passes a familiar yearly percentage and is not off by a factor
  of ~252.
- New block 4c in run(), after the cost block (reuses its `held` array) and
  before the equity cumsum: flat_yield = where(held == 0, per_bar_cash_yield, 0),
  then flat_yield[0] = 0.0 to preserve the structural index-0 zero, then
  strategy_returns = strategy_returns + flat_yield. Added (mirroring how cost is
  subtracted) so every downstream metric reflects it.
- Reported the cumulative interest earned as cash_earned_pct on BacktestResult,
  the income mirror of total_cost_pct - same type, same default 0.0, placed
  immediately after it. [+6 tests]
- Buy-and-hold and the strategy layer required no change: B&H is always long (its
  only flat bar is the structural bar 0, which the mask zeroes), and cash yield is
  an engine accounting concept, not a signal concept.
- 180 tests green (was 174).

**Why it matters:**
- Cash-on-flat is the last of the three pre-verdict accounting items (transaction
  costs landed Day 28, total return Day 30, this is cash-on-flat). A momentum
  strategy sits in cash for long stretches; modeling zero return on that idle
  capital understates its true return, especially against a buy-and-hold
  benchmark that is always invested. This closes the last accounting gap before
  the verdict.

**Architectural note:**
- Built as a rate defaulting to 0.0, NOT an always-on charge, so all prior
  results stay bit-for-bit reproducible and every existing test stays green. The
  verdict run will opt into a real rate explicitly.
- One-place, engine-only change (plus one field on BacktestResult). The
  walk-forward validator passes the Backtester through unchanged, so the yield
  flows to both the strategy path and the B&H path with no validator edit.
- The flat_yield[0] = 0.0 line is load-bearing: held[0] is 0 by construction, so
  without it the mask would credit yield at index 0 and leak a spurious return
  into the equity curve's first step - corrupting the Sharpe [1:] slice and the
  walk-forward seam-stripping (fr.returns[1:]). A dedicated test pins that index-0
  stays exactly 0.0.
- Annualization is the footgun this guards against: log(1 + rate) /
  annualization_factor, using self.annualization_factor (not a hardcoded 252), so
  compounding the per-bar yield over a year sums back to the full annual rate -
  not the rate charged once per bar.

**Verification:**
- Full suite 180 passed (was 174), only the pre-existing websockets
  DeprecationWarning. Prior 174 unchanged, proving annual_cash_yield=0.0 is a true
  no-op and the new cash_earned_pct field broke no existing BacktestResult
  construction.
- test_all_flat_earns_annualized_yield: 252 flat bars (after the index-0 mask) at
  4%/yr produce total_return_pct ~ 0.04 (not 0.04 per bar) - the off-by-252 proof.
  test_index_zero_yield_stays_zero: returns[0] == 0.0 exactly with a 5%/yr yield.
  test_all_long_earns_no_cash_yield: a fully-invested run earns 0.0 and is
  bit-identical to a free run. test_mixed_earns_yield_only_on_flat_bars: signals
  [1,1,0,0,0] credits yield only on the two flat bars. test_cash_yield_default_is_no_op
  and test_negative_annual_cash_yield_raises round out the set.
- One code commit (engine + result field + tests): "Add cash-on-flat yield to
  backtester".

**Blocked on:**
- Nothing.

**Next up:**
- Wire price_field AND annual_cash_yield through the verdict path
  (walk_forward_validate / the overfitting-tax CLI), passing the SAME basis and
  the same yield to both the engine and the strategy explicitly - the footgun
  mitigation, and the last wiring before the verdict.
- Optional operational step still pending: re-fetch the basket to un-stale it
  (blocks nothing).
- Then momentum through the full harness (walk-forward + Optuna + overfitting tax
  + buy-and-hold) on adj_close, after costs, with cash-on-flat - the actual
  verdict. Folds sized against the 12-month lookback; judged on drawdown and
  downside, not just Sharpe.
- SMA crossover price_field for a consistent basis across strategies.

## Day 30

**Worked on:**
- Added a configurable price basis to the Backtester so returns can be computed
  from close (price return, the existing behavior) or adj_close (total return,
  dividends and splits folded in). New price_field parameter on
  Backtester.__init__, last positional, defaulting to "close"; validated against
  {"close","adj_close"} with a ValueError and no silent fallback; run() now
  builds the price array via getattr(b, self.price_field). _make_trade was not
  touched - it reads fill prices off the passed-in array, so trade fills inherit
  the basis automatically. [+4 tests]
- Mirrored the same change onto TimeSeriesMomentumStrategy: a price_field
  parameter (last, default "close", same validation), generate_signals now reads
  getattr(bar, self.price_field), and the name property appends the basis only
  for the non-default case - "TSMOM(12)" stays byte-identical, "TSMOM(12,
  adj_close)" for the adj_close basis. [+1 test]
- Buy-and-hold required no change: it runs an all-long signal through the same
  Backtester, so it inherits the engine's price basis for free and stays
  apples-to-apples with the strategy by construction.
- 174 tests green (was 169).

**Why it matters:**
- adj_close total-return accounting is one of the three pre-verdict items the
  momentum verdict is blocked on (transaction costs landed Day 28; this is total
  return; cash-on-flat is next). Using raw close ignores dividends, which
  understates a buy-and-hold investor's real return and biases any
  strategy-vs-buy-and-hold comparison. This adds the capability without yet
  flipping the basis.

**Architectural note:**
- Built as a configurable basis defaulting to "close", NOT a hard switch, so all
  prior results stay bit-for-bit reproducible and every existing test stays
  green. The verdict run will opt into "adj_close" explicitly.
- The basis is set in TWO disconnected places - the Backtester and the strategy
  - because the engine and strategy are intentionally decoupled (the engine does
  not know which strategy produced the signals). There is no single threading
  point; consistency is enforced at the CALL SITE that constructs both for a run.
  The footgun: a future call site that forgets to pass adj_close to both silently
  gets price return. Mitigation deferred to verdict day - the entry point will
  pass the same basis to both explicitly, and the proof tests ensure the
  adj_close path genuinely differs so a silent no-op would be caught.
- SMA crossover still reads close and would need the same parameter for a
  consistent basis across strategies; scoped as a small follow-on, not today, to
  keep this change on the momentum path heading for a verdict.

**Verification:**
- Engine: prior 169 plus 4 new = 173 green on the full suite. Default and
  explicit "close" produce bit-identical returns (np.array_equal); invalid
  price_field raises; diverging close vs adj_close data yields different
  total_return_pct (>1e-6); identical data through either basis yields identical
  returns (so the difference is data-driven, not flag-driven).
- Strategy: 173 plus 1 new = 174 green on the full suite. name property prints
  "TSMOM(12)" for the default and "TSMOM(12, adj_close)" for the adj_close basis
  (verified live). The proof test constructs an increasing close path (LONG
  basis) against a decreasing-but-positive adj_close path (FLAT basis) so every
  warmed-up month-end's trailing-return sign flips between bases, and asserts the
  two signal arrays are not array-equal.
- Two code commits, both two-file and additive: "Add configurable price basis to
  backtester" and "Add configurable price basis to time-series momentum".

**Blocked on:**
- Nothing.

**Next up:**
- Cash-on-flat accounting: model a short-term interest rate earned while the
  strategy is FLAT (out of the market), which currently earns nothing - the last
  pre-verdict accounting item.
- Then wire price_field through the verdict path (walk_forward_validate / the
  overfitting-tax CLI), passing the SAME basis to both the engine and the
  strategy explicitly - the footgun mitigation.
- Optional operational step still pending: re-fetch the basket to un-stale it
  (blocks nothing).
- Then momentum through the full harness (walk-forward + Optuna + overfitting tax
  + buy-and-hold) on adj_close after costs, folds sized against the 12-month
  lookback, judged on drawdown and downside, not just Sharpe.
- SMA crossover price_field for a consistent basis across strategies.

## Day 29

**Worked on:**
- Audited all 17 etf_basket ETFs for non-finite (NaN / inf) close and
  adj_close. Found exactly 17 bad bars: one per symbol, all the trailing
  2026-06-10 row, close=NaN and adj_close=NaN. No interior holes, no
  infinities - a stale whole-basket last-fetch (the close had not settled
  when it was pulled).
- Cleaned the store: scripts/migrate_drop_nonfinite_bars.py deletes every
  row with a non-finite close or adj_close. It probes DuckDB for
  isnan()/isinf() before building the predicate (both exist in 1.5.2), has a
  sanity gate that aborts unless exactly 17 rows match, and a backup guard.
  DB backed up to .bak-pre-nonfinite first. Pre-count 17, post-count 0,
  deleted 17; verified on disk with an independent count separate from the
  script's self-report.
- Compute boundary: Backtester.run now rejects any close that is non-finite
  or <= 0, raising a ValueError that names the count and the first bad bar's
  index and timestamp - instead of silently producing NaN metrics. The guard
  sits between building the closes array and the log-return math. [+5 tests]
- Write boundary: write_bars now skips-and-warns on any bar with a non-finite
  close or adj_close (math.isfinite, log.warning, no raise) so a bad bar can
  never be stored again. Returned count excludes skipped bars. [+1 test]
- 169 tests green (was 163).

**Why it matters:**
- Day 28's SPY sanity exposed a NaN close that poisoned full-history return
  and drawdown. The verdict on momentum is only as trustworthy as the data
  and the engine under it, so this had to be closed before any pre-verdict
  work.

**Architectural note:**
- Two boundaries, two correct responses. The WRITE boundary skips-and-warns:
  one bad trailing bar must not abort a multi-symbol ingest. The COMPUTE
  boundary raises: a NaN close means no valid result is possible, so fail
  loud.
- The store uses INSERT OR IGNORE keyed on (symbol, timestamp, timeframe)
  with daily bars floored to midnight UTC, so a re-fetch cannot overwrite an
  existing bad row - the bad row must be DELETED before a clean re-fetch can
  replace it. That ordering is why the migration deletes rather than updates.
- This is the Day 21 prevention-plus-detection pattern reused: prevent at the
  write boundary, detect at the compute boundary, clean the existing damage
  with a backed-up one-off migration.

**Verification:**
- migrate output: per-symbol breakdown 17x1, "Sanity check passed: 17 == 17",
  non-finite remaining 0, rows deleted 17. Independent on-disk count returned
  0 after the migration. Full suite 169 passed, only the pre-existing
  websockets DeprecationWarning.

**Blocked on:**
- Nothing.

**Next up:**
- Optional operational step: re-fetch the basket to refresh the now-deleted
  2026-06-10 row (the write-boundary guard now protects against re-storing a
  bad trailing bar; the basket is ~2 weeks stale).
- Pre-verdict accounting: total-return via adj_close (switch the signal AND
  the engine together, never one alone) and cash yield on flat periods.
- Then momentum through the full harness (walk-forward + Optuna + overfitting
  tax + buy-and-hold), folds sized against the 12-month lookback, judged on
  drawdown and downside, not just Sharpe.

## Day 28 — 2026-06-23

## Day 28

**Worked on:**
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

**Why it matters:**
- This is the friction that turns a backtest from a fantasy into something
  closer to honest. It is one of the three things (with adj_close total-return
  accounting and cash-on-flat) that have to be in before any TSMOM-vs-B&H
  verdict means anything.

**Architectural note:**
- Cost is charged as a linear log-return drag (turnover*cost_rate subtracted),
  not the multiplicative (1 - turnover*cost_rate). At basis-point magnitudes
  the gap is negligible (second order in the rate); the approximation's
  validity is bounded by a small cost_rate, which realistic costs respect.
- win_rate and Trade records stay GROSS; per-trade cost attribution is a
  deliberately deferred scope boundary. Only the aggregate metrics are net.

**Verification:**
- SPY TSMOM(12) gross-vs-net sanity (one-off, not committed) at 3 bps total:
  total return 3.1236 -> 3.1026, Sharpe 0.5300 -> 0.5281, max drawdown
  unchanged, total_cost_pct 0.0051 over 9 round trips. Net strictly worse on
  return and Sharpe, drawdown untouched, cost tiny - momentum's low turnover
  means costs barely bite, which is the point.

**Blocked on:**
- Nothing.

**Next up:**
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

## Day 27 — 2026-06-22

**Worked on:** Added CI via GitHub Actions (.github/workflows/ci.yml) — runs `uv run pytest -q -m "not integration"` on every push and PR to main in a clean Ubuntu environment, excluding the 4 live-API/network integration tests (152 hermetic tests run in CI). Added a project CLAUDE.md encoding the engineering conventions (Strategy contract, verify-on-disk discipline, two-commit git flow, untracked-docs rule, scope limits).

**Why it matters:** Moves test verification off conversational attestation onto machine-produced logs — the green check is machine truth, not a prose claim, and catches a lookahead or contract regression the moment it lands. CLAUDE.md makes every Claude Code session start aligned with the conventions instead of re-deriving them per prompt. Neither touches trading edge; both serve result integrity and portfolio credibility. Made the four load_bars_for_symbols tests in test_cli_common.py hermetic via a tmp_path DuckDB fixture seeded with synthetic SPY/QQQ/AAPL bars (monkeypatching cli_common.DuckDBStore), after the first CI run surfaced that three silently depended on the local ingested DB and a fourth passed only because CI's DB was empty. CI now runs 152 hermetic tests green.

**Architectural note:** No application or test code changed. CI is tests-only (no coverage gate, lint, or deploy). The `-m "not integration"` filter is the hermeticity boundary — live-API tests need secrets and would be a separate secret-gated job later. The first red CI run did its job: it caught three tests that were integration tests in disguise (depending on un-versioned local DB state) and fixed them by isolating the DB, not by tagging them out, so the logic stays covered in CI.

**Blocked on:** None.

**Next up:** Wire TimeSeriesMomentumStrategy into the walk-forward / Optuna / overfitting-tax / buy-and-hold harness; add a transaction-cost model before any verdict.


## Day 26 — 2026-06-21

**Worked on:** Added TimeSeriesMomentumStrategy (long/flat, monthly rebalance, 12-month default lookback) in src/strategies/time_series_momentum.py, matching the existing Strategy contract: list[OHLCVBar] → int8 np.ndarray, emitting only SIGNAL_LONG and SIGNAL_FLAT, with a FLAT warmup and no internal lag. Added 12 unit tests in tests/test_strategies.py (uptrend/downtrend/flat, warmup, output contract, long-flat-only, monthly cadence, no-lookahead, too-short raise, empty raise) — 156 tests green. SPY sanity check: flat through 2008–2009 and through 2022, long through the recoveries — the crisis-avoidance behavior TSMOM is supposed to show.


**Worked on:** Added CI via GitHub Actions (.github/workflows/ci.yml) — runs `uv run pytest` on every push and PR to main in a clean Ubuntu environment. Added a project CLAUDE.md encoding the engineering conventions (Strategy contract, verify-on-disk discipline, two-commit git flow, untracked-docs rule, scope limits).




**Why it matters:** First strategy with a real economic thesis (momentum), unlike the edgeless SMA dummy. It drops into the existing backtester/walk-forward harness with no contract change — the primitive the Optuna sweep, cost model, and B&H verdict all hang off.


**Why it matters:** Moves test verification off conversational attestation onto machine-produced logs — the green check is machine truth now, not a prose claim. CLAUDE.md makes every Claude Code session start aligned with the conventions instead of re-deriving them per prompt.

**Architectural note:** No internal shift: the backtester's signals[:-1] * returns[1:] is the only lag — each month-end's signal takes effect on its own month-end bar and is forward-filled across the following days. The final bar is forced to be a month-end by convention; this is inert for the backtest (the engine uses signals[:-1], so the last signal earns no return) but needs an exchange-calendar check before live execution, since "is the last bar a month-end" is undecidable from price data alone. It uses raw close, matching the engine, so the backtest is price-return, not total-return.

**Architectural note:** No application or test code changed. CI is tests-only (no coverage gate, lint, or deploy). CLAUDE.md is tracked, unlike the intentionally-untracked trading_explained.md and daily_prompt.md.


**Blocked on:** Nothing.

**Next up:** Before any TSMOM-vs-B&H verdict, make the comparison honest: (1) total-return accounting — switch the signal AND the engine to adj_close together, never just one; (2) credit cash yield on flat periods (currently 0); (3) the transaction-cost model. Raw-close plus zero-cash currently flatters TSMOM vs an always-invested B&H. Then run TSMOM through walk-forward + Optuna + overfitting-tax + B&H, sizing folds against the 12-month lookback (a test fold needs multiple years to clear the 12-month warmup with usable post-warmup signal). Cross-sectional rotation stays a separate later phase (it breaks the per-symbol contract).

---

## Day 25 — 2026-06-19

**Worked on:** Built scripts/overfitting_tax.py — the CLI that runs the per-fold Optuna fitter (make_sma_optuna_fit_fn) against the fixed SMA(50,200) baseline across the full etf_basket and reports the overfitting tax. It mirrors compare_walkforward's loading (build_symbol_list / load_bars_for_symbols), builds each symbol's (train=504, test=126) splits ONCE and scores them twice on byte-identical windows — fixed (static SMA, no fit_fn) and fitted (fit_fn re-tuning fast/slow per fold by warm-only in-sample Sharpe) — then prints per-symbol Fixed OOS / Fitted OOS / B&H / mean in-sample / tax, plus a basket aggregate and two beat-counts. The only non-glue logic, summarize_tax (tax = mean per-fold in-sample Sharpe − stitched fitted OOS), is a pure function unit-tested in tests/test_overfitting_tax.py (the formula + the empty-fold guard). Full suite 144. Verified the harness against the committed baseline: every Fixed OOS value reproduces the Day-24 baseline run exactly across all 17 symbols, so the fitted column is trustworthy.

**Why it matters:** This is the clean negative result the whole validation arc was built to produce, and it is unambiguous. Across the basket, per-fold tuning does NOT create out-of-sample edge: mean fitted OOS Sharpe −0.02 vs the fixed baseline's +0.04 — tuning is, if anything, slightly worse — and fitted beat the static baseline on only 6 of 17 symbols (a coin flip). The overfitting tax is enormous and universal: mean in-sample Sharpe +1.01 collapses to mean fitted OOS −0.02, a ~1.0-Sharpe-unit gap on EVERY symbol (range +0.67 to +1.41). The optimizer reliably finds ~1.0 in-sample Sharpe that evaporates entirely out-of-sample — textbook overfitting, measured. And buy-and-hold dominates both: mean B&H Sharpe +0.41 beats fixed (+0.04) and fitted (−0.02) by a wide margin, winning on 15 of 17 symbols. Fitted beat B&H on exactly 2 — TLT (+0.26 vs +0.01) and IEF (+0.16 vs +0.09), the two bond ETFs whose own buy-and-hold is ~flat, so the trend filter wins only by sidestepping the 2020–22 bond drawdown (crisis alpha against a near-zero benchmark, not deployable edge). A methodological note worth recording: a 2-symbol preview (SPY, TLT) had shown fitted beating fixed on both, which would have suggested tuning helps — the full basket flipped that, the same small-sample cherry-pick trap QQQ illustrated earlier.

**Architectural note:** The harness's value is that it makes the comparison structural and verifiable — the fixed control reproduces the prior committed baseline number-for-number across all 17 symbols, which is the proof the fitted measurement is honest rather than a coincidence of a new code path. summarize_tax takes plain scalars (not a WalkForwardResult) so the tax formula is unit-tested without a live run, following the same extract-the-pure-decision pattern as should_alert and resolve_window. The script is fixed to etf_basket (no --universe flag) because the tax question is specifically about this stable, long-history basket; --symbols still overrides for spot checks.

**Blocked on:** None.

**Next up:** The SMA crossover is now conclusively edgeless on this basket — both as a fixed rule and tuned per fold — and the pipeline has proven it can identify a no-edge strategy as no-edge while quantifying the overfitting tax. The methodology vehicle has done its job. The next phase is to test a strategy with an actual economic alpha hypothesis (not a moving-average rule with no reason to work), running it through the same seam + walk-forward + tax harness, which now exists and is validated.

---

## Day 24 — 2026-06-19

**Worked on:** Built the full fit_fn pipeline that turns the walk-forward validator from a fixed-strategy scorer into a per-fold optimizer, in three layered pieces — a benchmark to measure against, the seam to plug fitting in, and the Optuna fitter itself. (1) Buy-and-hold benchmark in src/research/walk_forward.py: extracted the fold-stitching + return-metrics logic into a shared `_stitch_oos(per_fold, annualization_factor)` used by BOTH the strategy path and the benchmark, so the apples-to-apples comparison is structural, not two copies that could drift. The benchmark runs an always-long position (np.full(len, SIGNAL_LONG)) through the identical backtester.run on the identical test_bars per fold — same windows, same warm-up exclusion, same seam-zero stripping — so only the position series differs. Added bh_return/bh_sharpe/bh_max_drawdown to the frozen WalkForwardResult; compare_walkforward.py prints OOS / B&H / Δ per symbol. An invariant test pins it: an always-long *strategy* yields a stitched OOS identical to the benchmark. (2) fit_fn seam: added `fit_fn: Callable[[list[OHLCVBar]], Strategy] | None = None` to walk_forward_validate; the single seam line is now `fold_strategy = strategy if fit_fn is None else fit_fn(train_bars)`, with everything downstream byte-identical, so a fitted strategy is warmed over train+test yet scored only on the test window. No lookahead — fit_fn sees TRAIN bars only and signals are causal; the linchpin test asserts the i-th call receives exactly splits[i][0]. (3) Optuna fitter in new src/research/optuna_fit.py: `make_sma_optuna_fit_fn(...)` returns a fit_fn that runs a seeded-TPE study per train window, maximizing WARM-ONLY in-sample Sharpe and returning the best SMACrossoverStrategy. Search is window-clamped so every trial is constructible — fast ≤ len−2, slow ∈ [max(slow_range[0], fast+1), min(slow_range[1], len−1)] — so fast < slow and slow < len both hold by construction, and slow_range[0] is a respected floor that keeps the search in the golden-cross regime. A record side-channel captures {fast, slow, in_sample_sharpe} per fold, aligned by index with per_fold[i], for the step-3 tax. Seven tests including reproducibility (seed → identical params), a direction test on a trend-reversal series (a minimize/constant objective fails it), and a warm-only test (recorded Sharpe == warm-only slice, ≠ full-window). uv add optuna (4.9.0). Full suite 142.

**Why it matters:** The benchmark is the honesty check the project hinges on: across etf_basket, buy-and-hold BEATS the fixed SMA(50,200) on 15 of 17 symbols. QQQ — whose 0.53 OOS Sharpe looked tempting last session — LOSES to simply holding QQQ (B&H 0.80, Δ −0.27), which quantifies and kills the cherry-pick. The only positive alpha is TLT (Δ +0.22, the filter sidestepped the 2020–22 bond drawdown) and IEF (Δ +0.04) — real but modest crisis-alpha, not deployable. The seam + fitter exist to ask the next question rigorously: does *tuning* the SMA per fold beat the fixed baseline out-of-sample? The warm-only objective is what makes that measurement honest — the recorded in-sample Sharpe is scored over the same active regime as the OOS window (warm-up dropped on both sides), so the in-sample-vs-OOS tax is apples-to-apples rather than flattered by leading zero-return bars.

**Architectural note:** _stitch_oos makes the OOS-vs-B&H comparison rest on byte-identical code rather than discipline. The seam is a true one-line substitution — the no-lookahead guarantee lives in the validator's slice (train prefix warms, test suffix scores), and the fitter respects it by never letting its objective touch a test bar. The window clamps and slow floor guarantee every Optuna trial is a valid SMACrossoverStrategy without reject-and-retry, and keep the search semantically meaningful. Warm-only is a better *objective*, not just a cleaner report: a full-window Sharpe would quietly reward shorter slow windows for having fewer warm-up zeros — a measurement artifact, not signal quality. The fitter is reusable infra: the make_*_fit_fn pattern generalizes to any future strategy through the same seam.

**Blocked on:** None.

**Next up:** Step 3 — wire make_sma_optuna_fit_fn into compare_walkforward (or a sibling script), run across all 17 etf_basket symbols, and pair record[i] (warm-only in-sample Sharpe) with per_fold[i] (realized OOS Sharpe) to print the overfitting tax alongside the fixed-baseline OOS and the buy-and-hold line. Honest expectation: fitted OOS lands back near the fixed-SMA baseline and stays under buy-and-hold — the clean non-result that proves tuning a no-edge signal does not manufacture OOS edge.

---

## Day 23 — 2026-06-17

**Worked on:** Added content-level validation to scripts/data_health.py — until now the health report only caught *structural* problems (stale, missing, short history); it never looked at the bar values themselves. Two pure, fully-tested helpers do the work: `find_zero_volume_bars` (flags `volume == 0`) and `find_extreme_jumps` (flags any consecutive close-to-close ratio outside [0.5, 2.0], i.e. a >+100% or >−50% single-session move). Both are wired into the per-symbol report via one parameterized query filtered to exactly the inspected universe (`AND symbol IN (?, …)`), which loads ~30x less data than scanning all 520 symbols when `--universe etf_basket` is given, and skips the query entirely if the universe resolves to no symbols. Also extracted `should_alert(stale, missing, zero_volume)` so the process exit policy is unit-testable without running `main()`: zero-volume escalates to exit 1 (a liquid name should never have a zero-volume day), but extreme-jump deliberately stays informational — it also fires on genuine extreme moves, so it cannot be an automatic verdict. Added tests/test_data_health.py with 9 tests: both helpers (including the strict upper/lower [0.5, 2.0] boundaries, a drop-to-zero, and the non-positive-prior-close skip that avoids a divide-by-zero) plus `should_alert` (all-clear → no alert, each flag individually → alert). Full suite now 130.

**Why it matters:** This validates bar *content*, not just freshness, closing the gap where the backtester could silently read a corrupt bar that passed every structural check. Running it on `etf_basket` came back content-clean (0 zero-volume, 0 extreme-jump), which is the useful confirmation that yfinance's `close` is split-adjusted for the basket. The full-DB scan earned its keep immediately: it surfaced real bad data outside the basket — SW carries 376 zero-volume bars — alongside a textbook false positive, GL's *real* −53% day on 2024-04-11 (a short-seller report, not a data error, ratio 0.469). That single false positive is the whole reason extreme-jump is informational rather than exit-1: the heuristic cannot distinguish an unadjusted split from a genuine crash, so it flags for a human to investigate instead of failing CI.

**Architectural note:** The two checks are pure functions over a `list[(date, close, volume)]` — no DB, no argparse — so they test with hand-built fixtures and the boundary behavior (`< 0.5` / `> 2.0`, strict) is pinned directly rather than inferred from a live run. `should_alert` follows the same extract-the-decision pattern as `resolve_window` from Day 22: the policy lives in a tiny pure function and `main()` just calls it. The asymmetry in the exit policy is the considered call here — unambiguous bad data (zero volume) hard-fails red, ambiguous heuristics (extreme jump) report but don't fail — so the report stays trustworthy as a CI gate without crying wolf on legitimate market events.

**Blocked on:** None.

**Next up:** Day 24 — the fit_fn seam integration that Day 22 earmarked for Day 23, pushed one slot by this content-validation work. Replace `fold_strategy = strategy` with `fold_strategy = fit_fn(train_bars)` running an Optuna sweep per train window, evaluated across the now content-validated etf_basket OOS sample to measure the overfitting tax against the fixed-strategy baseline.

---

## Day 22 — 2026-06-11

**Worked on:** Stabilized the data foundation before resuming Phase 2. Added a fixed `etf_basket` universe to config/universe.yaml — 17 liquid, long-history, non-return-selected ETFs (SPY, QQQ, IWM; sectors XLK/XLF/XLE/XLV/XLY/XLP/XLI/XLU/XLB; bonds TLT/IEF; GLD; international EFA/EEM). Deep-backfilled 2008-01-02 → present through the Day 21 fixed write path: ~4,639 bars/symbol, and verified on disk that every symbol has zero duplicate dates and all timestamps at midnight UTC. Hardened scripts/backfill_universe.py: added `--start`/`--end` with parse-time validation via `date.fromisoformat` (a malformed date now fails at argparse instead of deep in the fetcher) and an "--end requires --start" guard the mutually-exclusive group can't express; extracted the window-selection logic into a pure, importable `resolve_window(years, start, end)` helper that raises `ValueError` on the no-input misuse. The original `--years` path is byte-for-byte unchanged. Hardened src/data/universe.py: `load_universe` now raises on a ticker that parsed as a YAML boolean (e.g. an unquoted `ON`) rather than silently emitting `"True"`, since `bool` subclasses `int` and would slip past the existing `str()` cast. Added tests/test_backfill_window.py — 4 tests for `resolve_window` (start-only defaults end to today, start+end verbatim, years delegates to `compute_date_range`, no-input raises) — bringing the suite to 121. Confirmed by reading scripts/refresh_sp500_universe.py that the sp500 refresh does load-modify-dump (loads the full YAML, overwrites only `universes['sp500']['tickers']`, re-dumps the whole dict), so `etf_basket` and `test` survive a refresh untouched.

**Why it matters:** The strategy's trade base was one ticker. Walk-forward and the coming Optuna sweeps need enough independent trades to be statistically meaningful, on data that spans multiple regimes — so the basket goes back to 2008-01-02, through the 2008 crash, the 2020 COVID drawdown, and 2022's rate shock. ETFs were chosen over individual names specifically to sidestep survivorship bias by construction: the basket membership is fixed and the instruments don't get delisted out of the sample the way single stocks do. This is deliberately a data-foundation detour ahead of the originally-forecast Optuna seam work (now Day 23) — fit-on-train evaluation is only worth running once the OOS sample under it is broad and clean.

**Architectural note:** `resolve_window` was extracted as a pure function precisely so the window arithmetic is testable without argparse or the network — the CLI now just feeds it the three parsed values. The YAML-boolean guard in `load_universe` is a loud safety net, not a fix for a present bug: `'ON'` is correctly quoted in config today, but an unquoted edit would otherwise corrupt the universe silently. Note for backtest time: SPY and QQQ are shared across sp500/test/etf_basket and now carry 2008-onward history, while most sp500 names do not — so the sp500 universe has a per-symbol length asymmetry that downstream code must handle (e.g. via `min_bars` or per-symbol date alignment) rather than assuming uniform series lengths.

**Blocked on:** None.

**Next up:** The originally-forecast fit_fn seam integration (now Day 23) — replace `fold_strategy = strategy` with `fold_strategy = fit_fn(train_bars)` running an Optuna sweep per train window, now evaluated across the deeper, broader etf_basket OOS sample to measure the overfitting tax against the fixed-strategy baseline.

---

## Day 21 — 2026-06-08

**Worked on:** Discovered and fixed a timezone-duplication bug that was corrupting the entire bar dataset, then completed the planned walk-forward validator work. The bug surfaced during a sanity check on the new walk-forward CLI: SPY showed 1,848 bars for 2021-01-04 → 2026-05-08, impossible for a daily series (~1,350 trading days). Root cause: PRIMARY KEY (symbol, timestamp, timeframe) keys on the raw timestamp value, not the calendar date. Two ingest runs using different yfinance versions produced distinct UTC timestamps for the same trading date — an older version returned tz-naive pd.Timestamps, stored by the fetcher as 00:00 UTC; a newer version returned America/New_York-aware timestamps, stored as 04:00 UTC (summer/EDT) or 05:00 UTC (winter/EST). Both cleared the PK as numerically distinct keys; INSERT OR IGNORE never fired; and read_bars, which matches on CAST(timestamp AS DATE), returned both rows per affected date. SPY had 505 duplicated dates out of 1,343 unique ones — 27% of the series were spurious zero-return bars, deflating measured volatility and making annualization_factor=252 wrong (the data was effectively ~346 bars/year). Consequence stated plainly: every Sharpe computed before today, including Day 16's in-sample 0.22, was on corrupted data and is not trustworthy. The fix has two parts. Prevention: duckdb_store._bar_to_tuple now floors 1d bars to midnight UTC (ts.replace(hour=0, minute=0, second=0, microsecond=0) after the existing astimezone(utc).replace(tzinfo=None) line), so a 00:00 bar and an Eastern-midnight bar for the same date collapse to the same PK value and any future double-ingest deduplicates automatically. Regression test test_daily_bar_tz_duplicate_is_rejected: writes the same bar at 00:00 and 05:00 UTC, asserts exactly one stored row, confirmed it fails without the floor (INSERT OR IGNORE silently admitted the second row, returning 2 rows on read). Cleanup: one-time migration scripts/migrate_dedup_daily_bars.py backed up the DB, deduped on (symbol, date, timeframe) via ROW_NUMBER() OVER (PARTITION BY symbol, CAST(timestamp AS DATE), timeframe ORDER BY timestamp DESC) rn=1 — keeping the newer-ingest row — then re-stamped survivors to midnight UTC via CAST(CAST(timestamp AS DATE) AS TIMESTAMP), swapped the table atomically (CREATE staging + DROP original + RENAME). Verified: SPY 1,848 → 1,343 rows, zero dup dates store-wide, all timestamps at 00:00, SPY (date, close) sequence byte-identical to the pre-migration DISTINCT sequence, second symbol spot-check clean. On the planned work: built walk_forward_validate in src/research/walk_forward.py — generates signals over train+test bars so SMA indicators are warm at the test start, backtests only the test window, then stitches the disjoint test windows into one OOS track record (concatenated log-returns with structural index-0 zeros stripped at each seam, equity curve anchored at 1.0, OOS scalars: sharpe, total_return, max_drawdown, win_rate, n_folds_positive_sharpe, total_trades). Result is a frozen WalkForwardResult dataclass. Guards: explicit empty-splits ValueError, overlap guard rejecting shared test bars between folds. Per-fold strategy assignment is isolated to the single seam line fold_strategy = strategy for Day 22. Also pulled inline engine math into src/backtest/metrics.py (total_return, sharpe_ratio, max_drawdown, win_rate) shared by engine and validator; hardened the Sharpe zero-variance guard from == 0.0 to < 1e-12, because a constant series has residual std ~1e-18 from float rounding that slips past exact equality and produces a garbage Sharpe in the billions, ranking a loser as the best parameter. Golden-verified byte-identical on real SPY data. Built compare_walkforward.py — per-symbol per-fold table plus stitched OOS footer; header prints the actual loaded date range (bars[0]/bars[-1]) rather than the requested args.start/args.end, which would have surfaced the dup bug at a glance had it existed from day one.

**Why it matters:** First honest result on clean data: SPY, fixed SMA(50, 200), train=504 bars / test=126 bars, six folds — OOS Sharpe 0.36, total return +17.82%, 4/6 folds positive, including one −19% whipsaw fold the validator correctly surfaced without crashing. Nothing has been fitted to these windows, so this is the fixed-strategy baseline, not an overfitting-tax measurement — that comparison arrives Day 22 when Optuna fitting runs inside each train window. The data bug matters beyond the immediate numbers: every finding from Days 16–18 was on corrupted data, and the bug was invisible in the output because the CLI printed the requested date range rather than the actual loaded bar count. Closing both the root cause (write-path floor) and the observability gap (header now shows actual loaded range) means the same class of corruption cannot hide again.

**Architectural note:** The write-path floor is the single most important design property of today's fix: because _bar_to_tuple is the only entry point for every bar this project will ever store, any future fetcher inherits the dedup guarantee without modification — one choke point, one fix. The seam line pattern (fold_strategy = strategy as the single isolated assignment in the fold loop) keeps the validator flat and makes Day 22's optimizer integration a genuine one-line substitution rather than a structural refactor. Each fold re-enters flat at its first test bar, so one boundary bar's return per fold is not captured; this is a small conservative understatement that does not distort cross-strategy ranking and is preferable to the complexity of mid-fold position hand-off. Very small test windows yield degenerate OOS metrics rather than crashes — acceptable for a research tool where the caller controls window sizes.

**Blocked on:** None.

**Next up:** Day 22 — fit_fn seam integration: replace fold_strategy = strategy with fold_strategy = fit_fn(train_bars) where fit_fn runs an Optuna sweep over the train window and returns an optimized SMACrossoverStrategy. First genuine out-of-sample test: fit on train, evaluate on test, aggregate OOS Sharpe across folds, compare to today's fixed-strategy baseline to measure the overfitting tax.

---

## Day 20 — 2026-06-04

**Worked on:** Phase 2 — built the walk-forward splitter, the foundation of out-of-sample validation. Created src/research/walk_forward.py with one pure function, walk_forward_splits(bars, train_size, test_size, step=None) -> list[tuple[list[OHLCVBar], list[OHLCVBar]]], that slices a time-ordered bar list into successive (train, test) window pairs using a rolling/sliding window. Each fold's test window begins on the exact bar immediately after its train window (strict adjacency — no gap, no overlap), step defaults to test_size so test windows tile the post-training timeline exactly once, and leftover tail bars too few for a complete fold are dropped rather than emitted as a short fold (keeps every fold size-identical so fold-aggregated metrics stay comparable). Four validation guards raise before any slicing (train_size < 1, test_size < 1, step < 1 when provided, and len(bars) < train_size + test_size — the "not enough data for one fold" case, which raises rather than silently returning [], mirroring run_universe's all-skipped guard). The function is pure: no I/O, depends only on src.data.schema.OHLCVBar, assumes ascending-time input (guaranteed upstream by read_bars) and deliberately does not re-sort. Added tests/test_walk_forward.py with 11 tests built on a make_bars(n) helper that encodes each bar's index into its close value, so tests verify which bars land in which window by reading close rather than comparing timestamps. Tests cover: fold count on clean division, exact window sizes, strict train/test adjacency (the lookahead guard), no test-window overlap under default step, step-advance semantics, custom step > test_size, the four validation errors, and the exact-minimum-bars boundary (15 bars = train 10 + test 5 yields exactly one fold, proving the <= loop condition is inclusive). Test count grew from 80 to 91.

**Why it matters:** This is the overfitting defense the Day 16-18 research arc made concrete. Day 16 found a candidate winner on one symbol; Day 17 showed it was symbol-specific noise; Day 18 confirmed no SMA crossover has positive average edge on large-caps. All of those measured in-sample. Walk-forward is the machinery that will measure genuinely out-of-sample performance — train on one window, evaluate on the next, never let the strategy see its test data. The splitter is deliberately scoped to window generation alone: the validator that runs strategies through the folds and aggregates out-of-sample metrics is a later day. Splitting the boundary arithmetic (where an off-by-one silently reintroduces lookahead bias) into its own independently-tested unit means that when the validator is built, the windows underneath it are already proven correct.

**Architectural note:** The splitter has no dependency on the runner, the backtester, or DuckDB — it transforms one list into a list of sub-list pairs and nothing more. That isolation is why its 11 tests run in 0.02 seconds with no fixtures or real data. The adjacency property — test window starts at exactly start + train_size — is the single line where correctness lives, because a one-bar overlap there would leak a training bar into evaluation, the exact failure walk-forward exists to prevent; the test suite asserts it on bar position (via the close-encodes-index trick), not just timestamp ordering, so a gap or overlap cannot pass silently. The function works in bar-count terms rather than calendar terms (train_size=504 bars, not "2 years"), which is simpler and deterministic; calendar-aware splitting is a later refinement only if needed.

**Blocked on:** None.

**Next up:** The walk-forward validator — run a strategy (or the full parameter grid) through each fold via BacktestRunner, training/selecting on the train window and recording performance only on the test window, then aggregate out-of-sample metrics across folds. It reuses the splitter built today plus the existing runner and cli_common.load_bars_for_symbols, so it is orchestration over proven parts rather than new plumbing.

---

## Day 19 — 2026-06-04

**Worked on:** Phase 2 opener — a pure refactor with zero behavior change. Extracted the duplicated symbol-list-building and DuckDB read-loop logic that compare_universe.py and compare_matrix.py both carried into a new shared module src/research/cli_common.py with two pure functions: build_symbol_list(symbols_csv, universe, limit) -> list[str] (the --symbols CSV vs --universe/--limit branch, with whitespace and empty-token cleanup) and load_bars_for_symbols(symbols, start, end) -> dict[str, list[OHLCVBar]] (opens DuckDB once for the whole batch, skips missing symbols with log.warning + continue, preserves insertion order). Both take plain values rather than an argparse Namespace so they are unit-testable without a parser, and neither prints nor owns its error guards — the CLIs keep their own wording. Rewired both CLIs to call the helpers: compare_universe.py 375 to 336 lines, compare_matrix.py 557 to 501 lines, with DuckDBStore and load_universe imports removed (plus logging in matrix) once unused. compare_strategies.py was deliberately left untouched — it is single-symbol and shares no surface. Added tests/test_cli_common.py with 10 tests (6 pure for build_symbol_list, 4 DuckDB-backed for load_bars_for_symbols: known symbols, missing-symbol skip, input-order preservation, all-missing empty dict). Test count grew from 70 to 80.

**Why it matters:** The extraction was deferred on purpose until three call sites existed — at two it would have been premature abstraction. With three research CLIs now sharing the same load skeleton, the helper pays for itself and gives the upcoming walk-forward validator a tested bar-loading primitive to reuse instead of a fourth copy. The refactor was verified by capturing each CLI's stdout to golden files BEFORE any change, then diffing after every edit: all three CLIs produced byte-for-byte identical output, and all 80 tests stayed green. That discipline — golden-output capture, scope-before-cut, one CLI at a time, diff after each — is as much the point of the day as the code.

**Architectural note:** cli_common depends only on the data layer (duckdb_store, universe, schema), never on the sibling CLIs, so there is no circular import. The two functions deliberately stop short of owning user-facing output: each CLI still prints its own "Loading..."/"Loaded..." lines and its own empty-symbols / empty-bars error guards, because the wording diverges per CLI and belongs to the tool that knows its own context. This keeps the helper a pure data primitive (list in, dict out) and preserves the project-wide "raw data internally, formatting only at the edge" rule. A pre-existing minor quirk in compare_matrix (it prints the dimensions line before the empty-symbols guard) was left untouched — fixing it would be a behavior change and belongs in a separate commit.

**Blocked on:** None.

**Next up:** The walk-forward validator. The runner's stateless design plus the now-extracted load_bars_for_symbols helper mean the validator can slice each symbol's bars into train/test windows and call the existing runner methods inside a sliding-window loop, with no re-plumbing of data loading. This is the overfitting defense the Day 16-18 findings made concrete.

---

## Day 18 — 2026-06-02

**Worked on:** Extended src/research/runner.py with BacktestRunner.run_matrix(), the cross-product of run_many and run_universe: it takes bars_by_symbol (dict) plus a list of strategies and runs every (symbol, strategy) combination, returning a list[BacktestResult] in symbol-major, strategy-minor order. Same validation pattern as the other two methods (non-empty dict, non-empty strategies, isinstance check naming the bad index, non-empty bars per symbol, min_bars >= 1). min_bars filters at the SYMBOL level, not the per-cell level: a symbol below the threshold is skipped for all strategies, keeping the downstream pivot table rectangular rather than ragged. The all-skipped guard raises ValueError reporting the largest available bar count. The degenerate cases (one strategy → run_universe behavior, one symbol → run_many behavior) fall out of the nested loop without special-casing. Built src/research/compare_matrix.py — the third research CLI (python -m src.research.compare_matrix) — with a hardcoded six-pair SMA grid (same as compare_strategies.py for cross-tool comparability), --universe/--symbols/--limit/--start/--end/--metric flags, and a pivot-table view (symbols as rows, strategies as columns, chosen metric in cells) plus a strategy-ranking summary by mean metric. min_bars is derived from max(slow)+1 so every symbol contributes a full row or none. Pivot columns are reindexed to grid order (overriding pandas' alphabetical default); rows sort by row-mean with max_drawdown auto-inverted. Added 8 unit tests for run_matrix (empty-dict, empty-strategies, strategy-type, empty-bars-value, cross-product size, symbol-major ordering across all six positions, symbol-level skip, all-skipped raise). Test count grew from 62 to 70; suite runs in ~1.6 seconds.

**Why it matters:** This is the first tool that surfaces strategy-symbol fit, which neither per-axis CLI could. The 25-symbol Sharpe matrix produced three findings. First, SMA(50, 200) ranked best for the third independent time: highest mean Sharpe (-0.01, essentially flat) and the most symbols with positive Sharpe (14/25), while Day 16's SMA(10, 50) "winner" ranked second-to-last on mean Sharpe (-0.25) — the overfitting verdict from Day 17 holds under the fuller view. Second, EVERY strategy had negative mean Sharpe across the 25 large-caps: no simple SMA crossover configuration produced positive average risk-adjusted edge on individual stocks over 2021-2026. The "best" strategy is merely the least-bad. This is the honest ceiling of naive trend-following, and the framework reported it without flattery. Third, the best strategy varies by symbol — QQQ peaks at SMA(50, 200)=1.52, NVDA at SMA(50, 100)=1.00, while AIG and ABNB are negative across all six. Some symbols are structurally trend-friendly and others trend-hostile, which motivates per-symbol parameter selection as a Phase 2 research direction.

**Architectural note:** run_matrix completes the 2x2 of research axes: one/many symbols crossed with one/many strategies. The three CLIs (compare_strategies, compare_universe, compare_matrix) remain separate entry points rather than one tool with mode flags — each answers one well-defined question. With three call sites now sharing the same DuckDB-read / runner-call / compare-or-pivot / format-and-print skeleton, the shared core is finally worth extracting; that refactor (a common load-and-run helper) is now justified where at two call sites it would have been premature. The matrix CLI's pivot/summary display logic is the one genuinely new piece and stays local to compare_matrix.py.

**Blocked on:** None. VS Code's editor buffer again drifted from disk on the test file (the recurring 9+/M badge), but git add stages from disk so the commit captured the correct clean file; verified via git diff --cached --stat showing 141 insertions matching the 8 new tests.

**Next up:** Phase 2 begins — parameter optimization and walk-forward validation. The matrix is the data structure Optuna-driven sweeps will populate (hundreds of parameter combinations × symbols), and walk-forward validation will defend against the overfitting this week's findings made concrete. First Phase 2 task: extract the shared load-and-run helper now that three CLIs justify it, then build the walk-forward splitter on top of the runner.

---

## Day 17 — 2026-05-26

**Worked on:** Extended src/research/runner.py with a new BacktestRunner.run_universe() method that runs one strategy against many symbols and returns a list[BacktestResult] in dict insertion order. Same dependency-injected Backtester and isinstance validation as run_many(), but the iteration axis is symbols instead of strategies. Each result's strategy_name is composed as f"{strategy.name} on {symbol}" — injected via the strategy_name kwarg on Backtester.run() rather than mutating the strategy object — so the per-symbol label survives into compare() output without callers having to track it separately. Added an optional min_bars guard so symbols too short for the strategy's slow-window warmup are silently skipped (continue inside the loop), with a ValueError raised after the loop if every symbol ends up skipped (silent empty result would mask a config mistake). Built src/research/compare_universe.py — a second CLI entry point (python -m src.research.compare_universe) with --universe, --symbols, --limit, --fast, --slow, --start, --end, --sort flags. The CLI reads bars from DuckDB (single context-managed read, silently skipping symbols with no DB data), passes them to runner.run_universe with min_bars=slow+1, and prints the same sorted comparison table from Day 16 plus an aggregate footer counting positive Sharpe, Sharpe > 0.5, and positive return. Added 8 unit tests for run_universe covering empty-dict validation, strategy-type isinstance check, empty-bars-value validation, result-per-symbol invariant, dict insertion order preservation, min_bars skip behavior, all-symbols-skipped raise, and result type. Test count grew from 54 to 62; full suite runs in ~2 seconds.

**Why it matters:** Day 16 produced a "SMA(10, 50) beats SMA(50, 200) on SPY" result. That was suspicious — a single 5-year window on a single symbol can find any winner by chance. Today's multi-symbol axis was the first real check, and it inverted the conclusion: across the first 25 S&P 500 symbols, SMA(50, 200) had 14/25 positive Sharpe and 5/25 Sharpe > 0.5; SMA(10, 50) had only 9/25 positive Sharpe and 2/25 Sharpe > 0.5. The faster crossover that "won" on SPY underperformed on the broader universe — classic single-symbol overfitting that the framework now catches automatically. SMA(10, 50) also trades 5-10x more (e.g. 36 trades vs 13 on ABNB), which would compound into a transaction-cost drag the no-cost backtester currently hides. This is the smallest version of out-of-sample testing the system supports; formal walk-forward comes in Phase 2.

**Architectural note:** run_universe and run_many are deliberately separate methods, not one method with a "mode" flag. Each maps to a distinct research question ("which strategy is best on this symbol?" vs "which symbols does this strategy work on?"), and a unified API would force callers to construct degenerate single-element lists for the dimension they're not sweeping. Same reasoning applies to compare_strategies.py and compare_universe.py as separate CLIs rather than one with a mode toggle. Day 18 will add a third method (run_matrix) and a third CLI (compare_matrix.py) for the strategy × symbol cross-product — at three call sites the shared core will be ready to extract; at two it would be premature.

**Blocked on:** None. Tooling friction during the day (VS Code save behavior corrupted the test file mid-edit, requiring a git checkout and a clean re-paste of the Claude Code generated tests) cost time but did not affect the final artifact.

**Next up:** Day 18 — strategy × symbol cross-product matrix (run_matrix on BacktestRunner, compare_matrix.py CLI). Same primitives, one more layer of orchestration. This is the building block Phase 2's Optuna parameter optimization will feed on.

---

## Day 16 — 2026-05-15

**Worked on:** First research tooling. Built src/research/ package with two new files: src/research/runner.py defining BacktestRunner with run_many() (executes N strategies against one symbol's bars, returns ordered list[BacktestResult]) and a static compare() method (aggregates results into a pandas DataFrame, sorted by configurable metric). Built src/research/compare_strategies.py — a CLI entry point (python -m src.research.compare_strategies) that sweeps 6 SMA crossover parameter combinations on SPY by default, with --symbol/--start/--end/--sort flags. Output is a formatted terminal table with the best strategy on top, including a header line "Comparison sorted by X (descending)" and auto-inverted direction for max_drawdown. Added 10 unit tests covering runner validation, result ordering, DataFrame columns, sort direction, and synthetic-BacktestResult construction (make_result factory bypasses the engine so compare()'s sort logic is tested in isolation). Test count grew from 44 to 54.

**Why it matters:** this is the smallest unit of real research — comparing alternatives. Before today the only way to compare two strategies was to run two `python -c` scripts and eyeball the outputs. Now there's one command that runs the comparison, formats the table, and tells you which configuration wins on Sharpe (or any other metric). The runner is the foundation for Phase 2 parameter optimization: instead of 6 hand-picked SMA combos, the same code will eventually consume hundreds of (fast, slow) combinations generated by Optuna, or hundreds of stocks from the S&P 500 universe. Same shape, more rows.

**First real research finding:** on SPY 2021-2026, SMA(10, 50) beat canonical SMA(50, 200) on both total return (+38.34% vs +25.22%) AND max drawdown (20.83% vs 21.13%). Shorter-horizon variant won on a single window — but a single 5-year window is statistically thin, so this is a hypothesis to stress-test, not a conclusion. Day 17 (multi-symbol) and Phase 2 (walk-forward validation) will determine whether the finding holds out-of-sample.

**Architectural detail worth noting:** BacktestRunner.compare() is a @staticmethod that returns a DataFrame with raw numeric values; the CLI does its own percentage formatting on a copy. This keeps the runner's output usable for downstream consumers (CSV export, plotting, optimization loops) while letting the CLI present numbers human-readably. Same principle as the indicator/strategy/backtester separation — each layer produces a clean output and lets the next layer decide how to consume it.

**Blocked on / Bugs:** None. One thing worth flagging for future-me: the parameter grid is currently hardcoded in compare_strategies.py. That's fine for Day 16's "smallest research script" goal but won't scale to Phase 2 parameter sweeps. Day 17+ will probably extract grid generation into a separate config or generator function.

**Next up:** Day 17 — multi-symbol comparison. Same runner, but the dimension being compared shifts from "one symbol × many strategies" to "many symbols × one strategy." This is the first step toward portfolio-level analysis where you'll see which sectors/stocks the strategy actually works on.

---


## Day 15 — 2026-05-14

**Worked on:** First backtester. Built src/backtest/ package with two new files: src/backtest/result.py defining frozen dataclasses Trade (single round-trip record) and BacktestResult (equity curve, trades list, and precomputed metrics), and src/backtest/engine.py defining the Backtester class. The engine implements vectorized next-bar-execution simulation: signals[i-1] earns the return from bar i-1 to bar i, which makes lookahead bias structurally impossible. Computes equity curve via cumsum of log returns then exp, extracts trades from signal transitions (handling 0→nonzero entries, nonzero→0 exits, sign flips, and end-of-data force-close via a factored _make_trade helper), and pre-computes total return, annualized Sharpe ratio (252 trading days, ddof=1), max drawdown, win rate, and trade count. No transaction costs or position sizing — that's Phase 3 work; this measures raw strategy edge. Smoke-tested on 5 years of SPY SMA(50,200) signals from Day 14: +25.22% total return, 0.22 Sharpe, 21.13% max drawdown, 57.1% win rate, 7 trades. Underperforms SPY buy-and-hold (~+70%) as trend-followers do in choppy bull markets — this is the truth-telling layer working correctly. Added 15 unit tests including the critical test_no_lookahead_bias guard. Test count grew from 29 to 44.

**Why it matters:** this is the layer that judges strategies. Before today, the codebase could only produce strategies that LOOKED sensible. The backtester is what tells you whether they would actually make money — and in this case, told you that the canonical SMA crossover on SPY alone is mediocre. That's a useful finding: it sets the realistic baseline against which future strategies (RSI mean reversion, multi-factor ensembles, ML-based) will be compared. The result dataclasses being frozen means historical backtest results can't be silently mutated. The next-bar execution model means accidentally writing a lookahead-biased strategy is now structurally hard.

**Blocked on / Bugs:** One sharp edge caught and fixed during testing: numpy's std(ddof=1) is undefined for samples of size <= 1 and produces NaN, which the original `if std_return == 0.0` guard didn't catch. This surfaced as 8 RuntimeWarnings when running the 2-bar unit tests. Fixed by short-circuiting on len(active_returns) < 2 before computing std. Lesson: sample-size guards belong before the math, not after.

**Next up:** Day 16 — strategy comparison and reporting. Build a small CLI that runs multiple strategies (or one strategy across multiple parameter combos) and produces a comparison table. Foundation for the parameter optimization work in Phase 2.

---


## Day 14 — 2026-05-10

**Worked on:** First strategy. Created src/strategies/ package with two new files: src/strategies/base.py defining the Strategy abstract base class (mirrors the architectural pattern of src/brokers/base.py) and three module-level signal constants SIGNAL_LONG (+1), SIGNAL_FLAT (0), SIGNAL_SHORT (-1). Created src/strategies/sma_crossover.py with SMACrossoverStrategy(fast_window=50, slow_window=200) implementing the canonical Golden Cross / Death Cross rule. The strategy returns a numpy int8 array aligned 1:1 with input bars — SIGNAL_FLAT during warmup (positions where either SMA is NaN), SIGNAL_LONG when fast > slow, SIGNAL_SHORT when fast < slow. Smoke-tested on 5 years of SPY data (1848 bars, 2021-01-04 to 2026-05-08): 69.8% LONG, 19.5% SHORT, 10.8% FLAT, 7 regime transitions matching real market history (2022 bear market, recent April 2026 selloff). Added 11 unit tests in tests/test_strategies.py covering ABC enforcement, parameter validation, name property, length and dtype contracts, warmup behavior, uptrend/downtrend signals, qualitative crossover detection, and input validation. Test count grew from 18 to 29.

**Why it matters:** this is the conceptual transition from features to decisions. Up to today the code summarized prices into numbers; now it emits opinions about them. The Strategy ABC is what every future strategy will subclass — RSI mean reversion, momentum, pairs trading, ML-based — so getting the contract right today (numpy int8 output, length alignment, warmup-as-FLAT, pure-function semantics) saves the entire Phase 2+ refactor. The signal-as-position (not signal-as-trade) design lets the same strategy work for backtest and live execution without special cases — the backtester will derive trades by diffing consecutive positions.

**Blocked on / Bugs:** None. One discovery during smoke testing: the initial SPY data in DuckDB only had ~2 years of history because the Day 11 backfill skipped pre-existing 2024 data via INSERT OR IGNORE. Re-fetched explicitly with fetch_daily('SPY', '2021-01-01', '2026-05-10') to get the full 1343 bars (1848 with overlap from neighboring stocks). This is a documented limitation of the idempotent backfill approach — fresh universes get full history, but symbols pre-seeded with partial history don't get backfilled to the requested years. Acceptable trade-off; the explicit re-fetch is a clean workaround.

**Next up:** Day 15 — first backtester. Build src/backtest/ that takes (bars, signals) and simulates what the strategy would have earned over the historical period. Compute realistic metrics: total return, Sharpe ratio, max drawdown, win rate. This is where strategies prove (or disprove) edge.

---


## Day 13 — 2026-05-10

**Worked on:** First technical indicators. Created src/features/ package with src/features/indicators.py containing three vectorized functions: sma (simple moving average via pandas.rolling), log_returns (np.log of close ratio), and rsi (Wilder's RSI with explicit two-phase implementation — simple-mean seed at index period, then the recursive smoothing). All three operate on lists of OHLCVBar and return numpy arrays aligned 1:1 with input — NaN values where lookback is insufficient, never silently dropped. Single private helper _bars_to_close_series centralizes the OHLCVBar → pandas conversion. Added 10 unit tests in tests/test_indicators.py covering known values, edge cases (all-gains → 100, all-losses → 0), input validation (empty bars, invalid window/period), and length-alignment guarantees. Smoke-tested against real SPY data from DuckDB — SMA-50, RSI-14, and log returns all compute in milliseconds for 252 bars. Smoke-tested current SPY (2025-01-01 to today): RSI-14 = 75.47 (overbought), SMA-50 = 682.55.

**Why it matters:** Every strategy and every ML feature in Phase 2+ ultimately reduces to numerical arrays computed from OHLCV bars. Today's three functions are the simplest examples of the pattern — input contract (list of OHLCVBar), output contract (np.ndarray of len(bars) with NaN-padded prefix), implementation strategy (pandas internals, numpy boundary). Every future indicator follows the same shape.

**Blocked on / Bugs:** One subtle issue caught and fixed during development. The first RSI implementation used pandas.ewm(alpha=1/period, adjust=False), which runs the Wilder recursion from the first observation rather than seeding with a simple mean of the first `period` deltas. The two methods converge after ~3× period bars but diverge noticeably on the very first emitted RSI values — meaning values would not match TradingView, ta-lib, Bloomberg, or StockCharts on short series. Replaced with explicit two-phase implementation: simple arithmetic mean of first `period` gain/loss values at index `period`, then Wilder recursion for all subsequent indices. Verified against canonical Wilder reference series — agreement within 0.006 of published values [70.46, 66.25, 66.48, 69.35, 66.30, 57.92]. Test suite expanded from 8 to 18 tests.

**Next up:** Day 14 — first signal generation. Build src/strategies/ with a crossover strategy (e.g. SMA(50) vs SMA(200)) that takes bars, runs indicators, and emits buy/sell/hold signals as a numpy array aligned to the input bars.

---


## Day 12 — 2026-05-07

**Worked on:** Made the data layer production-ready for daily operations. Three major additions: (1) DuckDBStore.last_timestamp(symbol, timeframe="1d") — returns the most recent bar timestamp via SELECT MAX, with the same UTC re-attach guard as _tuple_to_bar. Powers incremental updates. (2) scripts/update_universe.py — daily incremental update CLI with --universe and --lookback-days flags. Per-symbol logic: if no stored data, skip with a hint to run backfill; otherwise compute start = max(last_ts.date() - lookback_days, 2010-01-01) and fetch only the tail. Tested on test universe (5/5 already current after re-runs proving double idempotency) and sp500 (503/503 in 68 seconds, 1500 new bars — ~17x speedup vs the 20-minute backfill). (3) scripts/data_health.py — read-only diagnostic that opens DuckDB with read_only=True, reports total bars, distinct symbols, date range, file size, stale symbols (sorted by most-stale-first, capped at 20 unless --verbose), short-history symbols (<252 bars, capped at 10), and (with --universe filter) symbols missing from DB. Exit 0 if clean, exit 1 if stale or missing — cron-friendly. Verified all four code paths via smoke tests. (4) Added 1 unit test (test_last_timestamp_returns_max) — pure unit, no network, uses tmp_path.

**Why it matters:** Backfill is a 20-minute one-time bootstrap; daily updates need to be 1-minute or cron jobs become impractical. Health diagnostics turn silent corruption (stale or missing tickers) into loud, actionable warnings before they poison a backtest.

**Blocked on / Bugs:** Discovered and fixed a latent timezone bug in DuckDBStore. The DuckDB Python driver was converting tz-aware datetimes to LOCAL time when binding to TIMESTAMP columns, then stripping tzinfo. The pipeline only worked because yfinance daily bars at 05:00 UTC happen to land on the correct calendar date when shifted to US/Eastern (UTC-5). This would have silently corrupted data on any machine in another timezone, or for any non-yfinance source, or for intraday bars. Fix: _bar_to_tuple now does bar.timestamp.astimezone(timezone.utc).replace(tzinfo=None) before binding, so the stored value is unambiguously UTC. The test_last_timestamp_returns_max test caught this — it's exactly the kind of bug a unit test with non-aligned timestamps surfaces. Existing 627k bars in DuckDB are unaffected as calendar dates (which is all daily bars need); future intraday data will be timezone-correct from day one.

**Documented limitation:** A delisted symbol's last_timestamp will keep growing stale silently — the update script will compute a start date relative to it and yfinance will return nothing or an error. The script handles this gracefully (skip + log warning) but doesn't actively detect "this symbol is dead." Will fix in Phase 2+ with point-in-time universe membership.

**Next up:** Day 13 — first technical indicators (SMA, RSI, returns). Build src/features/ with vectorized pandas/numpy implementations operating on lists of OHLCVBar.

---


## Day 11 — 2026-05-04

**Worked on:** Scaled the data layer to the S&P 500. Created config/universe.yaml with named universes ("test" with 5 tickers, "sp500" auto-populated). Built src/data/universe.py with load_universe() and list_universes() helpers. Built scripts/refresh_sp500_universe.py — scrapes Wikipedia constituents via pandas.read_html (with browser User-Agent header to bypass 403 and io.StringIO wrapping to avoid pandas printing HTML to stdout), normalizes ticker dots to dashes for Yahoo (BRK.B → BRK-B), updates the YAML. Added YFinanceFetcher.fetch_daily_batch() with on_error=skip|raise — partial failures don't kill the run. Built scripts/backfill_universe.py — argparse CLI taking --universe and --years, with tqdm progress bar and per-symbol error isolation. Tested end-to-end: backfilled 5 years of S&P 500 daily bars (~626k rows, 503/503 symbols, ~20 minutes). Verified idempotency at scale — re-running the test backfill produces 0 duplicate inserts. Added 3 unit tests for the universe loader (pure, no network).

**Why it matters:** The bot now has 5 years of daily price history for every S&P 500 stock stored locally — all 500+ companies, downloaded once and ready to query instantly. Before this, any strategy test would have to reach out to the internet every single time, which is slow and breaks if the data provider is unavailable. Now the bot can test a trading idea across hundreds of stocks in seconds, using data that's already on disk.

**Blocked on / Bugs:** Two pandas-related issues fixed: (1) Wikipedia returned 403 with default urllib User-Agent — fixed by passing a browser UA via urllib.request.Request. (2) pd.read_html(html_string) was printing the raw HTML to stdout as a side effect — fixed by wrapping in io.StringIO. Also needed to add lxml dependency (pandas.read_html requires it).

**Next up:** Day 12 — daily incremental update script (only fetch the latest N days, not full history), gap detection (find missing dates per symbol), and a "data health" diagnostic script.

---


## 2026-05-01 — Day 10
**Worked on:** Started Phase 1 (data infrastructure). Created `src/data/schema.py` with frozen `OHLCVBar` dataclass and `CREATE_TABLE_SQL`. Built `src/data/yfinance_fetcher.py` — thin adapter around `yfinance.Ticker.history` that returns `list[OHLCVBar]` sorted ascending. Built `src/data/duckdb_store.py` with `INSERT OR IGNORE`-based `write_bars` (idempotent), `read_bars` range query, and context-manager support. Added `tests/test_data_layer.py` with three integration tests covering fetcher output, roundtrip persistence, and idempotency. Verified end-to-end: fetched 252 SPY daily bars for 2024, wrote to DuckDB, re-ran the pipeline and confirmed 0 duplicate rows on second run.
**Why it matters:** Before this, the bot had no memory of what prices did in the past. This day gave it a database — a local file that stores the open, high, low, close, and volume for any stock, for any day. Think of it as building the library the bot will read before making any decision. The design is also "safe to re-run": if the daily download job runs twice by accident, it won't create duplicate entries or corrupt the numbers.
**Blocked on / Bugs:** None.
**Next up:** Day 11 — extend fetcher to support batch fetching across the S&P 500, add a "universe" config file listing tickers, build a backfill script that ingests N years of history for the full universe.

---

## 2026-04-29 — Day 9
**Worked on:** Refactored all four operational scripts (`first_order.py`, `limit_order.py`, `cancel_order.py`, `order_history.py`) to use the `AlpacaBroker` abstraction. Removed every `from alpaca.*` import from `scripts/` — verified with `grep -rn "from alpaca" scripts/` returning empty. Added `Broker.get_latest_price()` abstract method and Alpaca implementation to eliminate the temporary `_data` attribute leak — `grep -rn "broker._data" scripts/` also returns empty. Status comparisons now use plain strings (`"filled"`, `"canceled"`) instead of alpaca's `OrderStatus` enum. Combined script line count dropped roughly 50% across the four files. Confirmed all four scripts produce identical user-facing output. Smoke test still passes.
**Why it matters:** This confirmed that none of the trading scripts need to know they're using Alpaca anymore — they just say "place this order" and the broker layer handles the rest. That means if we ever want to switch to a different broker, only one file changes and every strategy works without modification. It also means we can plug in a fake "paper broker" for testing strategies without any internet connection at all.
**Blocked on / Bugs:** None.
**Next up:** Day 10 — start Phase 1 of the roadmap. Build `src/data/` layer: yfinance + Alpaca historical data fetcher, DuckDB schema, basic OHLCV storage and retrieval.

---

## 2026-04-27 — Day 8
**Worked on:** Built the broker abstraction layer. Created `src/brokers/base.py` with the `Broker` ABC plus typed dataclasses (`OrderRequest`, `OrderResult`, `AccountSnapshot`) and string Enums (`OrderSide`, `OrderType`, `TimeInForce`). Built `src/brokers/alpaca_broker.py` implementing the full interface against alpaca-py, with a hard assertion preventing `paper=False` and a `from_env()` classmethod. Added `tests/test_alpaca_broker.py` as a read-only integration smoke test — 1 passed. Added `pythonpath` and `integration` marker to `pyproject.toml`, created `conftest.py` at repo root for reliable imports. Refactored `scripts/hello_alpaca.py` to use `AlpacaBroker` — script shrank from 33 lines to 29 lines, all SDK code now behind the abstraction.
**Why it matters:** This is the foundation that everything else builds on. The bot now speaks a common language for placing orders — it says "buy 10 shares of SPY" and doesn't care how that gets executed. Alpaca is just one possible answer. This makes the bot portable: swap in a different broker, a simulator, or a backtester, and the strategies don't change at all. Without this layer, every strategy would be glued to Alpaca's specific code and impossible to test offline.
**Blocked on / Bugs:** None.
**Next up:** Day 9 — refactor `first_order.py`, `limit_order.py`, `cancel_order.py`, and `order_history.py` to use `AlpacaBroker`.

---

## 2026-04-26 — Day 7
**Worked on:** Built `scripts/limit_order.py` (submits an intentionally non-marketable limit BUY 20% below market, GTC, with a paranoia assertion preventing accidental marketable orders; saves order ID to `logs/last_limit_order_id.txt` for handoff). Built `scripts/cancel_order.py` (cancels by ID from CLI arg or from the handoff file, polls for terminal state up to 10 s). Built `scripts/order_history.py` (fetches last 50 orders across all statuses, prints fixed-width table with fill price and limit price columns, plus a grouped summary line). Verified full lifecycle end-to-end: limit order placed at $571.17, cancelled successfully, order history table confirmed `ACCEPTED: 1 | CANCELED: 1`.
**Why it matters:** Market orders just buy at whatever price is available — limit orders let the bot say "only buy if the price drops to X." That's essential for any real strategy. This day also added the ability to cancel an order that hasn't filled yet, and to look up a full history of what the bot has done. Without that history, there's no way to know if a trade actually went through or why it didn't.
**Blocked on / Bugs:** None.
**Next up:** Day 8 — design the broker abstraction layer in `src/brokers/`. Refactor shared credential loading and paper-account safety check out of scripts into a reusable module.

---

## 2026-04-25 — Day 6
**Worked on:** Built `scripts/first_order.py` with three safety checks: `paper=True` hardcoded (must never be changed), PA account prefix verification before any order is submitted, and hardcoded `SYMBOL`/`QTY` constants. Added dry-run preview with live SPY price fetch via `StockHistoricalDataClient`, market-hours detection with next-open timestamp, and double confirmation (`"yes"` typed explicitly) before submitting. Submitted first paper order — 1 share of SPY at $713.96, order ID `c5559e4b-bf39-42b4-9658-1e9da0639004`, status ACCEPTED, queued for Monday open. Verified order appeared correctly in Alpaca dashboard. Rotated API keys after accidental exposure in screenshot.
**Why it matters:** This was the first proof that the bot can actually do the one thing it exists to do — place a trade. Everything before this was setup; this was the moment it became real. The multiple safety checks (paper-mode lock, account verification, typing "yes" twice) are deliberate: one mistaken click in a live trading system can cost real money instantly, so the guards have to be there from the very first order.
**Blocked on / Bugs:** API key accidentally visible in a dashboard screenshot — immediately regenerated keys and updated `.env`.
**Next up:** Day 7 — limit orders, order cancellation, querying order history.

---

## 2026-04-25 — Days 1–5
**Worked on:** Project initialization — pyproject.toml, uv-managed virtual environment, folder scaffold (src/, tests/, notebooks/, scripts/, data/, logs/), .gitignore, .env.example, README.md. Alpaca paper account created and connected; $200k buying power confirmed. GitHub repo initialized and remote connected.
**Why it matters:** Getting the foundation right means not having to redo it later. Keeping API keys out of the code means they can never accidentally end up on GitHub. Separating strategy code from scripts and data means the project stays navigable as it grows. These decisions feel invisible when they're done right — and extremely painful to fix after the fact when they're skipped.
**Blocked on / Bugs:** None.
**Next up:** First commit. Begin Phase 1 — data ingestion layer (Alpaca bar fetch, DuckDB schema, yfinance backfill).

