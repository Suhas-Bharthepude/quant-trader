# Your Quant Trading System, Explained Like You've Never Traded

*A guide to the system you built, written as if you've never opened a brokerage account in your life.*

---

## Part 0: The Big Picture — What Is This Whole Thing?

Before any of the parts make sense, here's the whole thing in one page.

### What you're building

A **systematic swing-trading bot**: a program that decides — automatically, by rules, with no human in the loop — when to buy and sell stocks, holding each position for days to weeks at a time. "Systematic" means the decisions come from code, not gut feel. "Swing" means the holding period sits between day-trading (in and out within hours) and long-term investing (buy and hold for years).

The finished system will do four things end to end: pull market data, decide what to trade, simulate those decisions against history to check they'd have made money, and — eventually — place real orders through a broker (Alpaca). Right now it does the first three. The fourth (real money) is the final phase, staged carefully: paper trading first, then small real amounts, then scaled up.

It's also a **portfolio project** — something to show in technical interviews. That shapes every decision toward "would a senior engineer respect this?" rather than "what's the fastest hack?"

### The most important thing to understand

**You are not building a strategy. You are building the machine that finds and validates strategies.**

This is the idea that makes everything else click. The first strategy you implemented — the "SMA crossover," a 50-year-old trend-following rule — does **not** make money on individual stocks. You proved that (Part 10). So why pour weeks of engineering into infrastructure around a strategy that doesn't work?

Because SMA crossover is the **crash-test dummy**, not the car. It's a simple, well-understood rule whose only job is to exercise the machinery: the data pipeline, the backtester, the research tools. If the machine can honestly tell you "this strategy has no edge" — instead of flattering you with a fake-good number — then the machine is trustworthy. Later phases feed *better* strategies into the exact same machine, and you'll believe its verdict because you watched it deliver bad news without blinking.

Most amateur quant projects get this backwards. They fall in love with one strategy, build a backtest that accidentally cheats, see a beautiful (fake) result, and lose money in real life. You're building the validation machine *first*, deliberately, so that when a real edge shows up you can tell it apart from luck.

### Where you are right now

This is a **240-day build**. You're at **Day 31**. The foundation is done, Phase 2 is complete (walk-forward validation plus the parameter optimization on top of it), and you've now started Phase 3 and built the first strategy with a real reason to work. Six things happened after Day 25, and the second half of this document covers them: you built a **momentum** strategy — the first one with a genuine economic hypothesis behind it, unlike the deliberately-empty SMA crash-test dummy (Part 16); you added two engineering tools that make the whole project more trustworthy — automated testing in the cloud and a conventions file for your AI assistant (Part 17); you taught the backtester to charge **transaction costs**, so its results reflect the real cost of trading rather than a frictionless fantasy (Part 18); you hardened the data layer and the engine against corrupt prices, after a single bad bar was found poisoning a whole-history result (Part 19); you taught the backtest to measure **total return** — counting dividends, not just price changes — as a switch you can turn on for the eventual verdict (Part 20); and you closed the last pre-verdict accounting gap by teaching the backtest to credit **interest on idle cash** while the strategy sits out of the market (Part 21). What's still ahead: running that momentum strategy through the full honesty machine for a verdict, the rest of risk management (Phase 3), machine learning (Phase 4), and live deployment (Phase 5).

In other words: the chassis, the engine block, and the honesty-checking instrument were built, tested, and *proven* on the hardest case — a strategy that turned out to be empty. Now the first real engine part is in, and all the realism needed to judge it fairly — trading costs, total-return accounting, and interest on idle cash — is in too. The verdict on whether momentum actually works comes next — run through the same unflinching machine.

### The one hard truth about trading

Markets are *mostly* random. Nobody reliably predicts tomorrow's price. The entire game is finding tendencies that are *slightly* better than a coin flip and exploiting them consistently across thousands of trades until the small edge compounds. The hard part isn't finding patterns — patterns are everywhere, and most are noise. The hard part is telling a *real* edge from a lucky-looking accident. That single discipline — relentless skepticism toward your own results — is what separates quants who compound wealth from quants who blow up. Every design choice in this system serves it.

With that frame in place, the rest of this document walks through each piece of the machine, bottom to top.

---

## Part 1: The Ground Floor — What Is Any of This?

### What is trading, really?

Forget everything you've seen in movies. No yelling, no suspenders, no Wolf of Wall Street energy.

Trading is just this: **buying something with the intent to sell it later for more money than you paid.**

You can trade houses, baseball cards, sneakers, gold bars, foreign currencies, or shares of Apple. The mechanics are identical. You buy when you think it's worth more than the asking price, you sell when you think you've squeezed out as much profit as you reasonably can, and you try not to be wrong too often.

The only thing that makes financial trading different from flipping sneakers on StockX is that the items being traded are **slices of ownership in companies** (called shares of stock) or **bundles of those slices** (called ETFs, which are like pre-made fruit baskets of stocks).

When you buy one share of SPY, you're buying a tiny sliver of ownership in 500 of the largest American companies at once. Apple, Microsoft, Walmart, Coca-Cola — you own a microscopic piece of all of them, and when those companies do well, your share is worth more.

### What is a market, and what is a "price"?

A market is just a place where buyers and sellers find each other. The grocery store is a market. eBay is a market. The "stock market" you hear about on the news is really just the **NYSE** and **Nasdaq** — two giant computer systems where millions of people submit orders every second saying "I'll sell my Apple shares for $215" or "I'll buy 100 shares of Tesla at $390."

When two orders match, a trade happens, and that price becomes the **most recent price** of the stock. A millisecond later another trade happens at $215.02, and that becomes the price. All day long.

Here's the thing that confuses everyone at first: a stock price isn't decided by some authority. There's no price tag. The price of Apple is **whatever the most recent two people agreed it was worth.** If the last trade was $215.34, that's the price until the next trade happens.

This means prices change constantly — many times per second — based on what millions of people *think* the company is worth right now. Every news headline, earnings report, rumor, and mood swing is reflected in that price. This is the chaotic ocean your trading system is trying to navigate.

### Why does anyone think they can predict this?

Honest answer: most people can't, and most people don't make money trying.

But there's a key insight: even though prices are noisy and unpredictable in the short term, **patterns exist**. Stocks that have been rising tend to keep rising for a while (this is called "momentum"). Stocks that have crashed too hard tend to bounce a bit (this is called "mean reversion").

These patterns aren't laws of nature — they're tendencies. They don't work every day, and they don't work for every stock. But over thousands of trades, if you can find a tendency that's *slightly* better than 50/50 and trade it consistently, the math compounds. That's the whole game.

A "trading strategy" is just a **specific set of rules for spotting one of these tendencies and acting on it.** Your system, when finished, will be a robot that follows such rules without flinching, fear, or fatigue.

---

## Part 2: The Building Blocks — What Are You Actually Looking At?

### Bars: How prices are summarized

If you tried to look at every single trade Apple made in a day, you'd melt your computer — it trades millions of times daily. So traders summarize price activity into **bars**.

A daily bar tells you five things about a stock for that day: **Open** (first trade), **High** (highest trade), **Low** (lowest trade), **Close** (last trade), and **Volume** (how many shares changed hands). Together this is an **OHLCV bar**. One bar per day; a stock's history is just a long list of these going back years.

In your system, you defined this exact structure as the `OHLCVBar` Python object (Day 10). Every other piece of your system speaks the language of OHLCV bars. They're the atom of your stack.

### Why "Close" is the most important price

Of the four prices, the close gets disproportionate attention. During the day, prices flop around for dumb reasons — someone liquidating to pay rent, a rumor that circulated for an hour, a fat-fingered trade. But the close is the price the market settled on after a full day of digestion. It represents the consensus value at the end of the day.

That's why almost every indicator your system computes uses the close. It's the most signal-rich number in a bar. (The backtester also fills trades at the close, for the same reason.)

### Where bars come from

Your bars come from **Yahoo Finance** — a free public data source with decades of daily history for nearly every US stock and ETF (Day 10, via the `yfinance` library). You wrote code (`YFinanceFetcher`) that downloads bars for any symbol over any date range, then stores them in a fast database called **DuckDB**, which is like a spreadsheet program optimized for crunching billions of rows.

Why store them at all? Hitting Yahoo's servers every time you backtest would be slow and rate-limited. Storing bars locally means you can run a backtest over years of S&P 500 data in seconds without ever touching the internet. This local database is your **memory of the world** — the foundation everything else sits on.

---

## Part 3: Where the Data Comes From — Your Pipeline

A "pipeline" is just a series of steps that takes raw input on one end and produces useful output on the other, like an assembly line.

```
Yahoo Finance API → Fetcher → DuckDB Store → Universe → Backfill / Update → Health checks
```

### The Universe (Day 11)

Before trading anything, you have to decide **what stocks you'll consider** — your "universe." Trading every stock on Earth is chaotic (penny stocks, delistings, no-volume names), so professional systems narrow to a clean set.

You chose the **S&P 500** — the 500 largest US public companies. Liquid trading, reliable reporting, decades of clean data. The cleanest possible playground for a beginner's quant system. The list lives in `config/universe.yaml`, refreshed by scraping Wikipedia's S&P 500 page (a normal, reliable approach — that page updates within hours of any index change). (Later, on Day 22, you added a second, smaller universe alongside it — see Part 13.)

### Backfill, update, and health checks (Days 11–12)

**Backfill** is the one-time bootstrap: downloads several years of daily bars for the entire S&P 500 (~627,000 bars). ~20 minutes, limited by Yahoo's rate limits.

**Update** is the daily heartbeat: asks each stock for its latest stored bar and fetches only the days after. ~60 seconds. Fast enough for a daily cron job.

**Health check** (`data_health.py`) is a read-only diagnostic that reports stale symbols, missing symbols, and suspiciously short histories. A car's dashboard for your data. (Day 23 taught it to inspect the prices themselves, not just the dates — see Part 13.)

### Three bugs the data layer caught (Days 12, 14, and 21)

While writing a unit test on Day 12, you found DuckDB's Python driver was silently converting timezone-aware timestamps to local time, then stripping the timezone. Stored data was wrong by 5 hours — invisible for daily bars (5 hours doesn't change the calendar date), but it would have destroyed an intraday strategy later. Fixed by explicitly converting to UTC before storing.

On Day 14, smoke-testing your first strategy on SPY, you noticed only 505 bars when you'd asked for 5 years. The Day 11 backfill had skipped bars already in the database — and SPY had been pre-seeded with just 2024's data from a Day 10 test. One-line fix: explicit re-fetch. **Lesson: idempotency protects against duplication, not against under-coverage.**

The third bug, on Day 21, was the most consequential of the whole project — and timezone handling was the culprit *again*, in a subtler way. Day 21 started as "build the walk-forward validator" and turned into a data rescue when a new tool loaded **1,848 daily bars** for SPY over Jan 2021–May 2026 — impossible, since that span holds only ~1,350 trading days. The cause: a daily bar is identified in the database by (symbol, timestamp, timeframe) using the *exact* timestamp, and SPY had been ingested twice under two versions of the data library — an older one returning timestamps with no timezone (stored as midnight UTC) and a newer one returning New-York-aware timestamps (stored as 4 or 5 a.m. UTC, depending on daylight saving). Same trading day, two different timestamps, so the database saw two different bars and kept both. 505 of SPY's 1,343 trading days had a duplicate row.

Here's why it mattered. The two copies carried the *same* closing price, so the "return" from a bar to its duplicate was exactly zero — injecting ~505 fake zero-change days, about a quarter of the series, into every calculation. Total return and the equity curve barely noticed (multiplying by 1.0 changes nothing). But the **Sharpe ratio** is built from the *spread* of daily returns, and padding the series with hundreds of artificial zeros quietly shrinks that spread and corrupts the annualization (the math assumes ~252 trading days a year; the polluted series effectively had ~346). **Every Sharpe number computed before this fix — including the 0.22 in Part 7 and all the figures in Parts 8–10 — was computed on this padded data and should be treated as unreliable. The returns and drawdowns in those sections are essentially fine; the Sharpes are not.**

The fix had two halves. *Prevention*: the storage layer now snaps every daily bar to midnight UTC before saving, so one trading day can only ever produce one row no matter how its timezone is labelled — plus a regression test that reproduces the exact bug to ensure it can't return. *Cleanup*: a one-time migration (database backed up first) collapsed the existing duplicates, dropping SPY from 1,848 rows to a correct 1,343, with verification that not a single closing price changed in the process.

**Lesson: your metrics are only as trustworthy as the data underneath them.** This bug was invisible for weeks — every test passed, every backtest ran, the numbers looked plausible — because nothing was wrong with the *code* that computed the metrics; the rows it computed them on were doubled. It surfaced only because a new tool printed a bar count that failed a back-of-the-envelope check. The habit that caught it — "1,848 can't be right for five years of trading days" — is worth more than any single test.

---

## Part 4: Brokers — How Your System Actually Trades

### What is a broker?

You can't walk up to the NYSE and buy a share of Apple — the exchanges only deal with licensed members. So you go through a **broker**, a company with exchange access that places trades on your behalf and holds your money and shares.

For your system, you chose **Alpaca** — a broker built for algorithmic traders, whose whole business is letting code (not humans tapping buttons) place orders. Alpaca offers a free **paper trading** account where every trade is simulated with real market prices but no real money. Essential for early-stage development.

### Order types

**Market order** (Day 6): "Buy right now at whatever the price is." Fast, guaranteed to fill, but you don't know the exact price.

**Limit order** (Day 7): "Buy only if the price drops to $X or lower." You set a ceiling; the order waits, and may never fill. Slower, but gives price control.

Most professional strategies use limit orders for entries (avoid paying inflated prices in volatile moments) and market orders for exits (guarantee you can get out).

### The broker abstraction (Day 8)

Instead of writing code that talks directly to Alpaca, you defined an abstract `Broker` interface — a contract saying *"any broker must support: buy, sell, get_price, get_account_info."* Then `AlpacaBroker` is one implementation. Want to switch to Interactive Brokers or add crypto via Coinbase in two years? That's a new file; the rest of the system is untouched.

A safety detail: `AlpacaBroker`'s constructor has a hard `assert paper is True`. Accidentally instantiate it with real-money credentials before you're ready, and the program crashes immediately. Defensive coding for the case where being wrong costs real money. (This abstraction earned its keep on Day 20, when Alpaca deprecated several day-trading API fields — none of which your code touched, because the broker layer keeps that surface in one swappable place.)

---

## Part 5: Indicators — Turning Prices Into Signals

This is where the real intellectual work begins. Until now your system could fetch and store prices. Now it has to start **understanding** them.

### What is an indicator?

An indicator is a function that takes a list of bars and outputs a number — or a series of numbers — summarizing some aspect of the price history. Daily prices are noisy and hard to read; an indicator squeezes that noise into a cleaner signal you can build rules around. You built three on Day 13.

### SMA — Simple Moving Average

Take the last N closing prices, average them, move forward one day, repeat. A moving average smooths out the day-to-day noise so you can see the underlying trend.

The most famous trading rule on Earth uses two of them:
- **Golden Cross**: when the 50-day SMA crosses above the 200-day SMA, the trend is turning up. Buy signal.
- **Death Cross**: the reverse. Sell signal.

It's *literally a one-line rule*: `if sma_50 > sma_200: buy`. Variants have generated real returns for 50+ years. The "window" controls sensitivity — a 10-day window reacts fast but is noisy; a 200-day window is slow but stable. Days 16–18 will show you *exactly* how much that choice matters, and that the answer is "more than you'd think."

### Log returns — How prices change

Instead of "what's the price?", log returns ask "**how much did the price change?**" For each bar this computes `ln(close_today / close_yesterday)` — a slightly different way of measuring percentage change. For small moves it's nearly identical to a regular percent change.

Why use logs? Three reasons:

1. **They add up.** A stock that goes up 50% then down 50% has actually lost 25% ($100 → $150 → $75). With log returns you can just *add* daily returns to get the total, instead of multiplying — which avoids floating-point drift over thousands of bars.
2. **They're symmetric.** A $100→$110 move and a $110→$100 move have equal magnitude in log space (opposite signs). In percent terms they don't (+10% vs −9.1%). Symmetry matters for statistical models.
3. **They're approximately bell-curve distributed.** Most quant math assumes normally-distributed inputs. Daily log returns fit that far better than simple returns.

Returns are also the universal language of investing — *"SPY went up 15%"* is meaningful; *"SPY went to $682"* is meaningful only if you know where it started. Returns let you compare a $5 stock to a $5,000 stock fairly.

### RSI — Relative Strength Index

The most famous momentum indicator. It asks *"recently, has this stock been winning more days than losing?"* — looking at the last 14 days, separating up-days from down-days, and producing a single number from 0 to 100. Above ~70 is "overbought" (a pullback is more likely than usual); below ~30 is "oversold" (a bounce is more likely). These aren't guarantees, but over thousands of trades an "RSI < 30, buy" rule has historically produced positive expected returns.

**Why the exact formula matters**: there's no single official RSI formula. TradingView, MetaTrader, Bloomberg, and ta-lib all use Wilder's 1978 formula; dozens of approximations exist. If yours differs, your signals fire on different days than every chart the world looks at, and you'll waste days debugging "is it my code or theirs?" You implemented Wilder's exact two-phase formula (a simple-average seed, then a recursive smoothing that weights the prior average 13/14 and the new day 1/14). Your RSI now matches TradingView and Bloomberg within 0.006. Forever.

### The NaN-padding contract — the most important decision in your indicator module

A 50-day SMA can't be computed for the first 49 bars. RSI-14 can't for the first 14. Log returns can't for the first 1. Two options:

**Option A**: Drop the warmup values, return shorter arrays. Now `sma_50` returns 203 numbers, `rsi_14` returns 238, `log_returns` returns 251. To use them together you track three different offsets — and **one mistake silently misaligns the wrong RSI with the wrong price**, producing a backtest that looks profitable but isn't. This is "lookahead bias," the #1 reason backtested strategies fail in real trading.

**Option B (what you built)**: Always return arrays the same length as input, with NaN where the indicator can't compute yet. Now `sma_50[i]`, `rsi_14[i]`, `log_returns[i]`, and `bars[i]` always refer to the same point in time. Strategy code becomes trivially correct:

```python
if not isnan(rsi_14[i]) and rsi_14[i] < 30:
    buy()
```

NaN is the function's polite way of saying *"I don't know yet — skip this day."* This single discipline separates production-quality quant infrastructure from spaghetti scripts. The strategy layer inherits it, the backtester inherits it, the runner inherits it. **One shape contract, applied recursively, across every layer of the system.**

---

## Part 6: Strategies — Turning Signals Into Decisions

Indicators describe the world. Strategies have **opinions** about it. This was the conceptual leap on Day 14: your code stopped just summarizing prices and started taking sides.

### What is a strategy, really?

A strategy is the rulebook. It takes indicator outputs and produces, for every day, one of three answers:

- **+1 (LONG)**: I want to be holding this stock today. Bet prices rise.
- **0 (FLAT)**: No position. Out of the market.
- **−1 (SHORT)**: Bet prices fall.

In your code these are constants `SIGNAL_LONG`, `SIGNAL_FLAT`, `SIGNAL_SHORT` — readable names instead of bare numbers.

**Going long** is normal investing: buy at $100, sell at $110, pocket $10. **Going short** is the inverse: borrow shares, sell at $100, buy back at $90, return them, pocket the difference. Shorting is riskier (a long can only lose 100%; a short can theoretically lose infinity) but essential — without it, your strategy can only profit during uptrends.

### Position signal vs. trade signal — a crucial distinction

A naïve strategy emits *events*: "BUY on day 50, SELL on day 200." Intuitive, but clunky — every downstream system must remember whether it's in a position, the entry price, whether the next signal is an entry or exit.

Your strategy instead emits **position signals**: for every bar, what do I want to hold? `[FLAT, FLAT, ..., LONG, LONG, FLAT, SHORT, ...]`. This is **stateless** — each bar's signal stands alone. A trade happened on bar 50 if `signals[49] != signals[50]`. Position signals describe **desired state**; trades are just **transitions**. The backtester derives every entry and exit by walking the array once.

### The Strategy contract

Every future strategy follows the same rules: input `list[OHLCVBar]`, output `np.ndarray` of int8 the same length, values in {−1, 0, 1}, FLAT during warmup, pure function (no I/O, no broker calls, no global state). Get the contract right once and every future strategy — RSI mean reversion, momentum, an ML model — slots into the same backtester, risk module, and live executor without inventing its own conventions.

(A nerdy detail: signal arrays use `int8`, one byte each instead of eight. Across a parameter sweep of 100 combinations × 503 stocks × 5 years, that's ~630 MB instead of ~5 GB. The right call early, when it costs nothing.)

### Your first strategy: SMA Crossover

The Golden Cross — long when fast SMA > slow SMA, short when fast < slow, flat during warmup. Five lines of logic, generating returns since the 1970s. Run on 5 years of SPY it produced 7 signal transitions (trend-followers trade infrequently — a feature) with 69.8% of bars LONG (SPY mostly trended up). But signals alone don't tell you whether you'd have made money. For that, you needed the backtester.

---

## Part 7: Backtesting — The Truth-Telling Layer

If strategies are opinions, the backtester is the judge.

A strategy can produce sensible-looking signals and still lose money. Maybe its entries are late, its exits early, its few wrong calls catastrophic. You can't tell from looking at signals — you have to *simulate* against history. That's Day 15.

### What a backtest computes

Take `bars` and `signals`, produce a frozen `BacktestResult` with three layers:

**The equity curve** — same length as bars, showing what $1 would have grown to following the signals exactly. **The trades** — one record per round-trip (entry, exit, prices, direction, hold length, return). **The metrics** — five numbers compressing the whole run into one comparable row:

- **Total return**: did you make money? (`0.25` = +25%)
- **Sharpe ratio**: return per unit of risk, annualized.
- **Max drawdown**: worst peak-to-trough decline, as a positive fraction.
- **Win rate**: fraction of trades that were profitable.
- **Number of trades**: how many round-trips.

### Sharpe ratio, briefly

The most-quoted number in quant finance. It asks: *"How much return per unit of volatility?"* A 20% return sounds great until you learn it bounced between +200% and −50% along the way. Sharpe penalizes that. Roughly `(mean daily return) / (volatility of daily returns) × sqrt(252)`, where `sqrt(252)` annualizes from daily to yearly. (As Part 3's Day 21 story shows, this number is only as good as the daily-return series feeding it — pad that series with junk and the Sharpe lies while the return looks fine.)

Rules of thumb (single asset): below 0 is losing on a risk-adjusted basis (worse than cash); 0–0.5 is weak/noise; 0.5–1 is decent; 1–2 is very good; above 2 is rare and you should suspect a bug; above 3 is almost certainly lookahead bias.

### Next-bar execution: the most important rule

The single design decision that matters more than all the others combined.

When a strategy emits a signal at bar `i`, it knows everything through bar `i`'s close — so the earliest it can act is *at* that close, earning the return from bar `i` to bar `i+1`. In code, one line:

```python
strategy_returns[1:] = signals[:-1] * asset_returns[1:]
```

*Yesterday's signal earns today's return.* A signal at bar `i` never earns the return at bar `i`. The opposite — `signals[i]` earning `asset_returns[i]` — is **lookahead bias**, the #1 reason retail backtest results don't replicate live. Your engine makes it structurally impossible. Your `test_no_lookahead_bias` unit test constructs a 100% price jump and verifies the strategy can't capture it by going long *after* the jump already happened. **That test is the most important assertion in your codebase.**

### Frozen results: reproducibility over convenience

`Trade` and `BacktestResult` are **frozen dataclasses** — immutable after construction. Backtest results are historical artifacts; if someone asks "what was the Sharpe?" a month later, that number must be the same. Found a bug? You don't patch old results, you re-run with the fixed engine. The cost is recomputation; the benefit is that every result you ever look at is internally consistent.

### Your first backtest on SPY

SMA(50, 200) over 5 years of SPY: **+25.22% total return, 0.22 Sharpe, 21.13% max drawdown, 57.1% win rate, 7 trades.**

Read it carefully. +25% sounds positive — until you learn SPY itself returned roughly **+70%** buy-and-hold over the same window. The strategy *underperformed badly*. Sharpe 0.22 is near the bottom of "weak." The 21% drawdown means it didn't even protect you from the 2022 bear market. Win rate 57% is modest, but one trade earned +29.71% over ~3 years — **trend-followers have low win rates but big winners; a few huge wins fund many small losses.**

Why does trend-following underperform a bull market? The Golden Cross fires *after* the run-up has already started, and the Death Cross flips to short *after* the bear market is underway. It enters and exits late. In long smooth bull markets it captures most of the upside; in choppy ones it whipsaws.

**The most important sanity check passed silently.** Sharpe was 0.22, not 3.0. If the backtester had lookahead bias, the result would have looked *amazing* and been completely fake. The unimpressive number is the trustworthy *kind* of number — low, not suspiciously high. (Its exact value was later corrected by the Day 21 data fix; the point that it's "low, not 3.0" survives regardless.)

### A bug caught during testing

Writing the backtester tests, you saw 15 passes but 8 numpy `RuntimeWarning`s. Computing Sharpe for two-bar test cases, the returns array had length 1, where sample standard deviation is undefined — numpy returned NaN, and the guard `if std_return == 0.0:` didn't catch it. A NaN Sharpe would have silently poisoned every downstream comparison. Fix: check sample size *before* the math, not after. **Lesson: validate that inputs are mathematically defined before computing on them.** (Day 21 hardened this same guard further — see Part 11 — when a constant-but-not-exactly-zero series slipped past the exact-equality check.)

---

## Part 8: Research, Part One — Comparing Strategies on One Symbol (Day 16)

If the backtester is the judge, the research layer is the courtroom.

Through Day 15 your system ran *one* backtest at a time — enough to validate the engine, not enough to do research. Research is comparing alternatives. A backtest in isolation is just a number; *"SMA(50, 200) returned +25.22%"* is meaningless until you ask "compared to what?" A backtest *next to other backtests* is information.

### BacktestRunner — comparing N strategies on one symbol

A tiny class, two methods. `run_many(bars, strategies)` runs each strategy through the backtester and returns results in input order. `compare(results, sort_by=...)` aggregates them into a sorted pandas DataFrame, one row per result. Zero state, zero I/O — the runner doesn't read DuckDB or print anything; it just orchestrates.

That minimalism matters because the runner is the bottleneck through which *every* future research workflow flows: Phase 2's parameter optimization, the walk-forward validator, Phase 4's ML pipeline. They all get the same clean primitive.

### Two design choices that pay off later

**Raw numbers in, raw numbers out.** `compare()` returns `0.2522`, not `"+25.22%"`. The CLI formats a *copy* afterward. Phase 2's optimization loop needs to sort by Sharpe as a number, not a string. Each layer produces clean raw output and lets the next decide how to consume it.

**Top of the table = best, always.** Most metrics are "higher is better," so descending sort works. Max drawdown is the exception (smaller is better), handled with one line: `ascending = args.sort == "max_drawdown"`. The convention is encoded in code, not in the user's head.

### Your first research finding

Six SMA parameter combinations on SPY, sorted by Sharpe:

```
strategy        total_return  sharpe  max_drawdown  win_rate  n_trades
SMA(10, 50)     +38.34%       0.31    20.83%        33.3%     39
SMA(50, 200)    +25.22%       0.22    21.13%        57.1%     7
SMA(50, 100)    +19.36%       0.17    32.04%        60.0%     15
SMA(10, 30)     +17.15%       0.15    25.22%        42.1%     57
SMA(20, 100)    -18.44%      -0.20    46.32%        32.0%     25
SMA(20, 50)     -19.83%      -0.21    44.52%        33.3%     33
```

*(These Sharpe figures predate the Day 21 data fix — see Part 3. The relative ranking and the lesson below hold; treat the exact Sharpe values as approximate.)*

Three things to read. **The canonical Golden Cross isn't the winner** — SMA(10, 50) beat SMA(50, 200) on Sharpe, return, and drawdown. **Win rate misleads alone** — the Sharpe winner had only a 33% win rate, because trend-followers make money on a few huge wins, not hit rate (Sharpe respects this; win rate doesn't). **Two configurations lost money** on a period when SPY returned +70% buy-and-hold — parameter choice matters as much as strategy choice.

### Why this was a hypothesis, not a conclusion

Easy to miss: you ran *one* comparison, on *one* symbol, over *one* window. SMA(10, 50) won. You might conclude "SMA(10, 50) is the best config for SPY." **Don't.** You've shown it had the best Sharpe among six pairs on this specific 5-year SPY window — a statement about one sample, not underlying truth. Is it structurally better, or did 2021–2026 happen to suit it? Would it win on AAPL, MSFT, GOOG, or only on broad-index ETFs?

This is the central paradox of empirical quant research: **the more you optimize on past data, the more likely you find something that won't work in the future.** It's called **overfitting**, and it's the discipline-killer of every quant who's ever existed. The defense is **out-of-sample validation** — train on one period or symbol, test on another. Day 17 systematizes it across symbols; walk-forward (Phase 2, now built — Part 12) systematizes it across time.

---

## Part 9: Research, Part Two — Testing the Day 16 Hypothesis (Day 17)

Day 16 asked "which strategy is best for *this* symbol?" Day 17 flips the axis: "which symbols does *this* strategy work on?"

Day 16 ended with a candidate winner (SMA(10, 50)) and a deliberate warning that one symbol over one window isn't enough to trust. Day 17's job was to resolve that. The logic is a clean hypothesis test: if SMA(10, 50) is *genuinely* better than SMA(50, 200), it should win across many symbols, not just SPY. If the canonical (50, 200) instead has higher average Sharpe across a basket, that's strong evidence the Day 16 result was symbol-specific noise.

### run_universe — the dual of run_many

A new method, `run_universe(bars_by_symbol, strategy, min_bars=None)`. It runs one strategy across many symbols, returning results in dict insertion order. For each symbol: if it has fewer than `min_bars` bars, skip it silently (`continue`); otherwise generate signals, run the backtester, append. The result label is composed as `f"{strategy.name} on {symbol}"` — injected via the backtester's kwarg, never by mutating the (stateless, reusable) strategy object. If *every* symbol is skipped, it raises rather than returning an empty list that would mask a misconfiguration.

**Why skip silently instead of raising?** A universe isn't homogeneous — recent IPOs have short histories, some symbols are delisted, some aren't in the database yet. A 200-day strategy genuinely can't run on a 100-bar symbol; that's a fact about the symbol, not an error. Raising on the first short symbol would abort the whole scan, defeating its purpose. The all-skipped guard catches the *genuine* misconfiguration (you set the threshold too high for everything).

### How multi-symbol backtesting caught an overfitting mistake

The finding that justified the whole day. Both candidate strategies across the first 25 S&P 500 symbols:

| Strategy     | Positive Sharpe | Sharpe > 0.5 | Positive Return |
|--------------|-----------------|--------------|-----------------|
| SMA(50, 200) | **14/25 (56%)** | **5/25 (20%)** | **14/25 (56%)** |
| SMA(10, 50)  | 9/25 (36%)      | 2/25 (8%)    | 9/25 (36%)      |

**The Day 16 winner lost.** SMA(10, 50), which beat SMA(50, 200) on SPY, underperformed it on every aggregate metric across the broader universe. The faster windows happened to fit SPY's specific path in 2021–2026 — but "fits one path well" is not "captures real edge." On 24 other paths, the slower windows were better. (A second nail: SMA(10, 50) trades 5–10× more — 36 trades vs 13 on ABNB — so once Phase 3 adds transaction costs, the fast version bleeds even more.)

**This is not a failure of the strategy. It's a success of the methodology.** Had you skipped this step and traded SMA(10, 50) live on the Day 16 result, you'd have bet on a strategy that loses on 64% of the universe. A portfolio-ready system catches that before it costs money — and on Day 17 you demonstrated it does. That's the difference between a quant who learns and one who blows up.

### The footer is a stand-in for a generalization metric

`compare_universe.py` prints one row per symbol plus an aggregate footer like `14/25 positive Sharpe, 5/25 Sharpe > 0.5, 14/25 positive return`. That footer is doing more work than it looks. Until walk-forward validation arrives (now built — Part 12), it's the closest thing the system has to a **generalization metric** — a single readout of whether an edge holds broadly or only in a couple of lucky spots. A strategy with Sharpe 0.5 on one symbol and ~0.1 on the other 24 *averages* to something meaningful-looking, but is almost worthless in practice; the footer's hit-count surfaces that distinction at a glance, where a mean alone would hide it.

### One important caveat

Even the "winner" wasn't *good* in absolute terms — best Sharpe 0.81 (AJG), median around 0.10 (basically zero), with names like ABBV at −0.83 Sharpe and 70% drawdown. **"Better than SMA(10, 50)" is not the same as "good enough to deploy."** A second tell that neither configuration is a generalized winner: the top-5 symbols for the two strategies barely overlap (only ALB appears in both). They aren't capturing the *same* edge — they're capturing different noise on different symbols. And the 25-symbol slice was alphabetically biased (every symbol starting with A) — directionally robust enough to invalidate the Day 16 hypothesis, but not a final verdict.

Architecturally, Day 17 filled in the second of four research cells:

| | One strategy | Many strategies |
|---|---|---|
| **One symbol** | A single backtest (Day 15) | `compare_strategies` (Day 16) |
| **Many symbols** | `compare_universe` (Day 17) | `compare_matrix` (Day 18) |

---

## Part 10: Research, Part Three — The Full Matrix (Day 18)

The first three cells of that table covered one axis at a time. Day 18 filled the fourth: **every strategy against every symbol at once.**

### run_matrix — the cross-product

`run_matrix(bars_by_symbol, strategies, min_bars=None)` runs the full grid: an outer loop over symbols, an inner loop over strategies, returning results in symbol-major order (all strategies for symbol 1, then all for symbol 2). The two degenerate cases fall out of the loop structure with no special-casing — one strategy makes it behave like `run_universe`, one symbol like `run_many`.

One deliberate choice: `min_bars` filters at the **symbol** level, not the per-cell level. Bar count is a property of the symbol's history, not of any one strategy. So a symbol below the threshold is skipped for *all* strategies — keeping the output a clean rectangle (every symbol contributes a full row of results or none) rather than a ragged table with holes.

### compare_matrix.py — the pivot view

The third research CLI displays results as a **pivot table**: symbols as rows, strategies as columns, the chosen metric in each cell, plus a strategy-ranking summary. The layout is deliberate — humans scan variation-across-strategies horizontally within a row, and variation-across-symbols vertically. Since symbols (25) vastly outnumber strategies (6), the tall dimension belongs on the rows.

The `min_bars` floor is derived from `max(slow) + 1` across the grid — the *slowest* strategy sets the bar. If a symbol is too short for SMA(50, 200), it's dropped from the whole matrix even though SMA(10, 30) could have run on it. That's intentional: a uniform floor keeps the pivot rectangular and readable.

### What the full matrix revealed

The strategy ranking, by mean Sharpe across the 25 symbols:

```
strategy      mean_sharpe  positive_sharpe
SMA(50, 200)     -0.01          14
SMA(50, 100)     -0.14           9
SMA(20, 100)     -0.15          10
SMA(10, 30)      -0.20           9
SMA(10, 50)      -0.25           9
SMA(20, 50)      -0.31           4
```

*(As with Part 8, these Sharpe figures predate the Day 21 data fix — the qualitative findings below hold; the exact values are approximate.)*

Three findings, each more important than the last.

**1. SMA(50, 200) wins for the third independent time.** Best mean Sharpe, most symbols positive (14/25). Day 16's SMA(10, 50) "winner" ranks second-to-last. The overfitting verdict from Day 17 holds under the fullest view available.

**2. Every strategy has negative mean Sharpe.** Read that carefully. Across 25 large-caps over 2021–2026, *no* SMA crossover configuration produced positive average risk-adjusted returns. The "best" is merely the least-bad (flat, −0.01). This is the honest, unglamorous ceiling of naive trend-following on individual stocks — and the framework reported it without flattery. A system that flattered you here would be worse than useless; it would lose you money.

**3. The best strategy varies by symbol.** QQQ peaks at SMA(50, 200)=1.52, NVDA at SMA(50, 100)=1.00, while AIG and ABNB are negative across all six configurations. Some symbols are structurally trend-friendly (momentum names where trends persist), others trend-hostile (choppy names that whipsaw crossovers). No single column dominates every row. **This is a portfolio-construction signal you literally could not see from either per-axis tool** — and it's a concrete Phase 2 research question: should you run different parameters on different stocks based on their character?

### The three-day research arc

Put the findings in sequence and you have the shape of real quant research:

1. **Day 16:** SMA(10, 50) beats SMA(50, 200) on SPY. *(A single sample.)*
2. **Day 17:** That reverses across 25 symbols — SMA(50, 200) generalizes better. *(Overfitting caught.)*
3. **Day 18:** The full matrix confirms SMA(50, 200) is most robust, reveals that no SMA crossover has positive average edge on individual large-caps, and shows strategy-symbol fit varies enough to motivate per-symbol parameter selection. *(The honest ceiling, plus the next direction.)*

A researcher who builds the tool, uses it to invalidate their own earlier conclusion, and then uses the richer tool to map the actual edge landscape and decide where to go next. That arc *is* the skill.

---

## Part 11: A Note on the Codebase Itself (Days 19 and 21)

By Day 18 the three research CLIs each carried their own copy of the same plumbing — resolve the list of symbols, open the database, load each symbol's bars. Day 19 pulled that shared logic into a single small module (`cli_common.py`) that the CLIs now call instead of duplicating.

The interesting part isn't the code — it's the *timing*. The duplication existed after two CLIs, but extracting then would have been premature: two copies aren't yet a pattern, and a wrong guess about what's "shared" creates a tangled helper with five switches. By three call sites the shape was unambiguous, so the extraction was safe. And because a refactor must change *structure without changing behavior*, the work was verified by capturing each tool's exact output beforehand and confirming it was byte-for-byte identical afterward, with all tests still green. That discipline — refactor only when the duplication has earned it, and prove you changed nothing — is the kind of judgment that separates a maintainable codebase from a brittle one.

Day 21 applied the exact same move one level down. The walk-forward validator needed to score *slices* of a backtest, which meant the five metric formulas (return, Sharpe, drawdown, win rate) had to be callable on their own — but they lived inline inside the backtester. So they were extracted into a shared `metrics.py` that both the engine and the validator now call: one definition, reused, no chance of the two drifting apart. Same byte-identical golden-output check guarded the extraction. (It also fixed a latent bug along the way: the Sharpe "is the volatility zero?" guard used exact equality, which a constant-but-not-*exactly*-zero series slips past, producing a garbage astronomical Sharpe; it now uses a small tolerance instead.)

This is the same minimal-correct-change ethos that runs through everything below: touch only what the change requires, and prove on disk that nothing else moved.

---

## Part 12: Walk-Forward Validation — The First Honest Out-of-Sample Test (Days 20–21)

This is the overfitting defense the Day 16–18 arc made concrete — and it's built.

**What walk-forward validation is.** Every result in Parts 8–10 was measured *in-sample* — the strategy was scored on the same data a human used to pick it. That flatters results: SMA(10, 50) looked best on SPY precisely because it was chosen on SPY. Walk-forward fixes this by splitting the timeline into successive **(train, test)** windows. The strategy is fit or selected on a *training* window, then scored only on the *test* window that immediately follows — data it never saw. Slide forward and repeat, then stitch all those unseen test windows end-to-end into one continuous **out-of-sample** track record: the closest thing to "what an account actually following this would have done."

**The one trick that makes it work (Days 20–21).** You can't score a test window by running the strategy on those test bars alone. A 200-day moving average needs 200 days of history just to *start* producing a signal, so a test window run cold would be all warmup and emit nothing. The fix is *warm up on train, score on test*: generate the signals over the train-plus-test span together so the indicators are fully warmed by the time the test window begins, then keep only the test-window portion of the result. Day 20 built the window-cutting (with the fragile boundary arithmetic tested in isolation); Day 21 built the validator that runs each fold this way and stitches the test windows into the aggregate.

**The first honest out-of-sample number.** Fixed SMA(50, 200) on SPY, two-year train / six-month test, six folds: an out-of-sample Sharpe of **0.36**, return +17.82%, with four of six folds positive — including one fold the validator correctly showed losing ~19% in a choppy stretch. One caveat on what this is *not* yet: nothing is being optimized here — the training window only *warms* the indicators, the parameters stay fixed. So this is an honest *baseline*, not yet the "overfitting tax" measurement. That measurement is exactly what the next chapters build (Parts 14–15) — but first the data foundation needed leveling up (Part 13).

**A documented limitation, honestly.** Each fold's simulation starts flat and re-enters on the test window's first bar, so one day's return per fold isn't captured — a small, conservative *under*statement that hits every strategy equally and so doesn't distort which strategy looks best. And very small test windows produce statistically meaningless metrics (not crashes); test windows should be sized for enough trades to mean something.

---

## Part 13: The Data Foundation, Leveled Up (Days 22–23)

Before you could fairly test whether *optimizing* a strategy creates real edge, you had to fix two things about the ground that test would run on: *what* you test on, and whether you can *trust* the numbers in it.

### Why you switched playgrounds: from 500 stocks to 17 ETFs (Day 22)

Through Day 18 your research ran on the S&P 500 — 500 individual company stocks. Great for a first pass, but for the serious out-of-sample work coming next, individual stocks carry a quiet poison: **survivorship bias.** Companies get acquired, go bankrupt, or fall out of the index and vanish from your data. If your universe only contains the companies that *survived* to today, you're unknowingly studying a rigged sample — the winners — and any "edge" you find is partly just "stocks that didn't die tend to have gone up."

So you built a new, deliberately small universe: the **`etf_basket`** — 17 exchange-traded funds chosen to sidestep that problem by construction. Three broad-market indices (SPY, QQQ, IWM), the nine S&P 500 sector funds (XLK technology, XLF financials, XLE energy, XLV health care, XLY consumer discretionary, XLP consumer staples, XLI industrials, XLU utilities, XLB materials), two bond funds (TLT, IEF), gold (GLD), and two international funds (EFA, EEM). ETFs are baskets, not companies — they don't go bankrupt and get deleted the way a single stock does, so the membership is *fixed* and the sample isn't secretly filtered to winners.

Then you **deep-backfilled** them: instead of 5 years, you pulled history all the way back to **January 2008** — roughly 4,600 daily bars per fund. The reason is regimes. A strategy that looks good only because it was tested during one long bull market hasn't really been tested. 2008-onward forces every fund through the 2008 financial crisis, the 2020 COVID crash, and the 2022 interest-rate shock. If an edge survives all three, that means something; if it only survives the calm years, it's a mirage.

You also hardened the download script while you were in there — date inputs are now validated the moment you type them (a typo'd date fails instantly instead of deep inside the fetcher), and the universe loader now refuses a malformed ticker rather than silently turning it into garbage. Small armor, but the kind that saves an afternoon later.

(Worth noting: this didn't *replace* the S&P 500 work — that research stands. It added a cleaner, longer, regime-spanning playground specifically for the validation work ahead.)

### Why you taught the health check to read prices, not just dates (Day 23)

Your data health check (Part 3) could already catch *structural* problems — a symbol that stopped updating, one that's missing, one with suspiciously little history. But it had a blind spot: it never looked at the actual *numbers* in the bars. A bar could carry a completely corrupt price and sail through every structural check.

So you added two **content** checks. The first flags any **zero-volume** day — a day where supposedly not a single share traded. For a liquid fund that's impossible; it means the data is broken. The second flags an **extreme jump** — any day where the price more than doubled or more than halved versus the day before. That's the fingerprint of either a real crash or a data error (often an unadjusted stock split, where the price *looks* like it fell 50% overnight but actually just got divided).

Here's the interesting design decision. A zero-volume day is *unambiguously* bad data, so it **hard-fails** the check — red light, stop. But an extreme jump is *ambiguous* — it might be broken data, or it might be a real, violent market move — so it only **flags for a human** rather than failing automatically. The proof that this distinction matters showed up the moment you ran it across the whole database: it correctly caught a stock (SW) with 376 genuinely-broken zero-volume bars, *and* it flagged a −53% single-day drop in another stock (GL) on April 11, 2024 — which turned out to be **completely real**, the day a short-seller report cratered the stock. If the extreme-jump check hard-failed, it would have screamed "bad data!" about a true historical event. The check can't tell a crash from a glitch, so it's honest about that and asks you to look.

The basket itself came back spotless — zero bad bars — which was its own useful confirmation: the prices Yahoo gives you for these funds are properly split-adjusted, so the backtester is reading clean history. (A note for Part 18: "spotless" was true for the *history*; a single corrupt *final* bar slipped in later, during a routine update, and the Day 28 cost-model work is what caught it.)

**Lesson: "the data loaded without errors" and "the data is correct" are different claims.** Structural checks verify the first; content checks start on the second. Neither is optional once real money is the eventual destination.

---

## Part 14: Teaching the Machine to Tune Itself (Day 24)

Everything up to now used **fixed** strategies — SMA(50, 200), a parameter choice a human made. The whole promise of Phase 2 was to let the machine *choose* the parameters itself, on each training window, and then see whether that choice survived out-of-sample. Day 24 built that, in three layers.

### Layer 1: a fair yardstick — buy-and-hold

Before asking "is the tuned strategy good?", you need something to compare it against. The honest benchmark for any stock strategy is the laziest possible alternative: **just buy the thing and hold it.** No signals, no trading, no cleverness. If your sophisticated strategy can't beat sitting on your hands, it isn't worth running.

So you taught the validator to compute, alongside every strategy result, the buy-and-hold result *over the exact same windows* — same time periods, same warm-up handling, same math — differing only in that buy-and-hold is always invested. Making it the same code path matters: the comparison is then honest *by construction*, not two separate calculations that might quietly disagree. The first thing this revealed was sobering — across the basket, **buy-and-hold beat the fixed SMA strategy on 15 of 17 funds.** The strategy you'd spent weeks building infrastructure around loses, on most things, to doing nothing. (Which is exactly what you expected — remember, it's the crash-test dummy.)

### Layer 2: the seam — one line that changes everything

The walk-forward validator was built (Day 21) with a single line deliberately left as a placeholder: *"use this fixed strategy on every fold."* Day 24 activated it. Now that line reads: *"if no tuner was given, use the fixed strategy as before; if a tuner was given, call it with this fold's training data and use whatever strategy it hands back."*

That's the entire change. Everything around it — warm up over train, score only on test, never peek at the future — stays identical. The crucial property: the tuner only ever sees the **training** window. It never touches the test data it'll be graded on. This is the same anti-cheating discipline as the rest of the system, now extended to the tuning step itself: you can't overfit to data you're not allowed to look at.

### Layer 3: the tuner itself — Optuna

The tuner uses **Optuna**, a standard library for searching parameter combinations intelligently — it doesn't try every option blindly; it learns which regions look promising and focuses there. For each training window it tries ~30 combinations of the fast and slow moving-average lengths, keeps the one with the best in-sample score, and hands that strategy back to the validator to be graded on the untouched test window that follows.

Two careful details made the measurement trustworthy. First, **the search is fenced in** so every combination it tries is valid (fast always shorter than slow, slow always short enough to fit the window) and stays in the sensible "golden-cross" range rather than wandering into nonsense. Second — and this is subtle — when scoring a combination in-sample, the tuner **ignores the warm-up period** (the early days where a long moving average doesn't yet have enough history to mean anything). Why? Because the test window it'll be compared against is already warmed up. Score the training period *including* its warm-up but the test period *without* it, and you'd be comparing two different things — and the headline number, *how much the optimizer flatters itself*, would come out wrong. Matching the two makes it fair.

A small but important side effect: the tuner *records* the in-sample score it achieved on every window. That record is the raw material for the single most revealing number in the whole project, which Day 25 finally computed.

**Lesson: the benchmark and the anti-cheating rule are not afterthoughts — they're the load-bearing parts.** Anyone can build a thing that optimizes a number. The discipline is in making sure the number it optimizes is measured honestly and compared against the lazy alternative.

---

## Part 15: The Overfitting Tax, Measured (Day 25)

This is the day the machine delivered the verdict it was built to deliver.

### What "the overfitting tax" actually is

Here's the trap that destroys most amateur quants, stated plainly. You take a strategy, tune its parameters until it looks fantastic on historical data, see a beautiful number, and conclude you've found an edge. But you didn't find an edge — you found the parameters that best *memorized the noise* in that specific stretch of history. Run those same parameters on new data the tuner never saw, and the beautiful number collapses. **The gap between how good it looked while you were tuning it (in-sample) and how it actually performs on unseen data (out-of-sample) is the "overfitting tax"** — the portion of the apparent edge that was never real, just the optimizer fooling itself.

Day 25 measured that gap, for the tuned SMA strategy, across all 17 funds. The script runs two things on each fund over identical windows — the fixed SMA(50, 200), and the per-fold Optuna-tuned version — and prints, side by side: how the tuned version looked in-sample, how it actually did out-of-sample, how the fixed version did, and how plain buy-and-hold did.

### The result

The numbers, averaged across the basket (Sharpe ratio — return per unit of risk, where higher is better and roughly 0 means "no skill"):

| | Average Sharpe |
|---|---|
| In-sample (how the tuner saw it) | **+1.01** |
| Out-of-sample (what tuning actually delivered) | **−0.02** |
| Fixed SMA(50, 200), out-of-sample | +0.04 |
| Buy-and-hold | **+0.41** |

Read those four numbers slowly, because together they tell the entire story of the project.

**The tuner reliably produced a Sharpe of ~1.0 in-sample.** That's a *genuinely good-looking* number — the kind that, on a screenshot, says "I found something." It showed up on *every single fund*, ranging from +0.67 to +1.41.

**Out-of-sample, that ~1.0 collapsed to roughly zero — slightly negative.** The overfitting tax was about a full point of Sharpe, on every fund. Almost everything the optimizer "found" was noise it had memorized. This is overfitting, caught in the act and measured.

**Tuning didn't even beat *not* tuning.** The fixed, un-tuned SMA scored +0.04 out-of-sample; the tuned one scored −0.02. Letting an optimizer work hard on each window produced a *slightly worse* result than leaving the parameters alone. Tuning a strategy with no real edge doesn't create edge — it just adds a way to fool yourself.

**And buy-and-hold beat all of it,** by a wide margin (+0.41), winning on 15 of the 17 funds. The two it didn't win — TLT and IEF, the bond funds — are the exception that proves the rule: bonds went roughly nowhere over this period, so their buy-and-hold Sharpe is near zero, and a trend filter "wins" only by sidestepping their 2020–2022 slide. Beating a flat benchmark by playing defense is real, but it's a low bar, and it isn't the kind of edge you build a business on.

### Two things that make this trustworthy

First, a correctness check that's easy to miss: the fixed-strategy column reproduced your earlier committed baseline **number-for-number across all 17 funds.** That's the proof the new code is measuring honestly — the control matches what you already knew, so the new (tuned) column can be believed.

Second, a lesson in miniature. Earlier you'd run a quick 2-fund preview (SPY and TLT) where the tuned version happened to beat the fixed one on both — which, taken alone, would have suggested "tuning helps!" The full 17-fund run flipped that completely. **It's the same cherry-picking trap from the Day 16–18 arc, one more time: two data points can say anything; the broad test tells the truth.** You've now been bitten by — and caught — that exact illusion three separate times. That reflex *is* the skill.

### Why a "nothing works" result is a triumph

It would be easy to read this day as a failure: you built an elaborate tuning machine and it found no edge. That's exactly backwards. **The machine was built to tell you the truth, and the truth here is "no edge" — and it said so, loudly, with a number, instead of flattering you.** A worse system would have shown you that +1.0 in-sample Sharpe and let you believe it. Most retail quant projects *are* that worse system, which is why most retail quants lose money. Yours measured the tax and refused to pretend. When a strategy with an actual reason to work comes along, you'll run it through this same machine and you'll believe its verdict — because you watched it deliver bad news without blinking.

**Lesson: the goal was never "make the SMA strategy work." It was "build something that can tell whether *any* strategy works." That's done — and proven on the hardest possible case, a strategy you already knew was empty.**

---

## Part 16: The First Strategy With a Real Reason to Work (Day 26)

Every result up to here tested the SMA crossover — and the whole point of Day 25 was that it has no edge, *because there was never a reason for it to.* A moving-average crossover is a shape on a chart; nothing about the world makes that shape pay. Day 26 builds the first strategy with an actual *hypothesis* behind it: **momentum.**

### What momentum is, and why it might actually work

**Momentum** is the tendency of things that have been going up to keep going up for a while — and things that have been going down to keep going down. You met the word back in Part 1; this is the strategy that bets on it directly.

The difference from the SMA dummy is that momentum has a *reason* economists actually argue about. People are slow to react to news, so a stock that just turned good keeps drifting up as the crowd catches on. Winners attract more buyers (herding). Fund managers pile into what's already working. None of these are laws — they're behavioral tendencies — but they're a genuine *story* for why the pattern could persist, and momentum is one of the most-studied, most-documented effects in all of finance. That's the bar a real strategy has to clear: not "this line crossed that line," but "here is why a tendency exists, and here is how I capture it."

### Two flavors of momentum — and which one you built

There are two ways to measure momentum, and the distinction matters because it decides how much of your machine you can reuse.

**Time-series momentum** (also called *absolute* momentum): judge each thing **against its own past.** "Has SPY gone up over the last year? Then hold SPY. Has it gone down? Then sit out." One symbol at a time, in isolation. It never compares SPY to QQQ — only SPY-today to SPY-a-year-ago. **This is what you built on Day 26.**

**Cross-sectional momentum** (also called *relative* momentum): **rank a group of things against each other** and hold the strongest few. "Of my 17 ETFs, which 3 went up the most this past year? Hold those three, rebalance monthly." This compares symbols to *each other*, not to their own history.

Why time-series first? Because it's a **drop-in.** Recall the Strategy contract from Part 6: a strategy takes one symbol's bars and emits one position per bar (+1/0/−1). Time-series momentum fits that exactly — it looks at one symbol and says "hold it / don't" — so it slots into the existing backtester, the walk-forward validator, the tuning harness, *everything*, with zero rework. Cross-sectional momentum breaks the contract: ranking requires looking at all symbols at once and producing **target weights** ("put 33% in each of the top 3"), which needs a different, portfolio-level backtester you haven't built. So time-series is roughly 80% of the payoff for 20% of the work, and cross-sectional rotation is deliberately deferred to a later phase (Part 23).

### The rule, concretely

The strategy is **`TimeSeriesMomentumStrategy`**, and it has exactly one knob and two timing rules.

- **Lookback** — how far back you measure the trend. The default is **12 months**, the most-studied horizon in the momentum literature. It's a constructor argument so you can later sweep it (3, 6, 12 months are all common) without touching the code.
- **Monthly rebalance** — the strategy is only allowed to change its mind at **month-ends**, not every day. "Rebalance" just means "reconsider and adjust the position." Why monthly? Momentum is a *slow* signal — a year-long trend doesn't reverse on a random Tuesday. Checking daily would just generate noise and churn (and, once Part 18's costs are in, pay fees for nothing).
- **The decision** — at each month-end, look at the symbol's **own return over the trailing 12 months.** If that return is positive, hold the symbol for the coming period (LONG, +1). If it's zero or negative, stay out (FLAT, 0).

Crucially, this strategy is **long/flat only** — it emits LONG or FLAT and *never* SHORT. (Recall shorting from Part 6: betting prices fall.) Long/flat is the conservative choice: when the trend turns down, the strategy simply steps aside to cash rather than taking the riskier bet that the decline continues. It can protect you from a crash without exposing you to the unique dangers of a short position. (The code makes this structural — it doesn't even import the SHORT constant, so a short literally cannot leak out by accident.)

### How it honors the contract — and the one subtle correctness detail

Everything about the output matches the contract from Part 6: one position per bar, same length as the input, values only in {0, 1}, FLAT during warmup. **Warmup** here is the first 12 months: there's no "12 months earlier" to compare against yet, so the trend is undefined and the position is FLAT — exactly the NaN-padding philosophy from Part 5, expressed as "no position until I have enough history."

And — echoing the single most important rule in the whole system (Part 7) — the strategy adds **no internal lag of its own.** The one-bar "yesterday's signal earns today's return" delay lives *only* in the backtester. The strategy emits, for each bar, the position implied by data through that bar; the backtester's next-bar-execution rule supplies the single, correct lag. Adding a second lag inside the strategy would double-count it and quietly corrupt the results, so it deliberately doesn't.

One subtle detail worth understanding, because it's the kind of precision that keeps a backtest honest. The strategy detects month-ends by **real trading dates** — the last actual bar that exists in each calendar month — not by the calendar's official last day. Why? Because the 31st of a month often falls on a weekend when markets are closed, so there's no bar on that date. If the code keyed off calendar dates, the exact day a signal changed would become unpredictable and occasionally land on a day with no price. Keying off "the last bar that actually traded that month" makes the rebalance dates real, concrete trading days, every time.

### The test that matters most, again

Just like the backtester (Part 7), the strategy's most important test is the **no-lookahead** check — and it's done cleverly here. Build a price history, compute the signals, then change **only the very last day's price** and recompute. Every signal *before* the final month must come out byte-for-byte identical. The logic: changing today's price can only affect today's own decision; if it somehow changed a signal from months ago, the strategy would be reaching into the future. It passed. (Twelve unit tests in total cover this plus uptrend, downtrend, the flat-price boundary, warmup, the output shape, the "never short" guarantee, the monthly-only cadence, and the error cases — bringing the suite to 156.)

### The payoff: what it does on real SPY history

This is the satisfying part. Run 12-month time-series momentum on SPY's full history and look at *when* it switches between LONG and FLAT:

- It stayed **FLAT — out of the market — through almost the entire 2008–2009 financial crisis** and through the **2022 bear market.**
- It was **LONG through the recoveries and the 2021 bull run.**
- Its very first LONG signal didn't fire until late 2009 — about 12 months into the data — because of the warmup.

That is *exactly* what trend-following is supposed to do: ride the up-trends and step aside when the trend rolls over. Seeing it emerge on real data — flat through the two worst declines of the era, without anyone hand-coding "avoid 2008 and 2022" — is the first sign this strategy is doing something structurally sensible, in a way the SMA dummy never did.

### Two honest caveats, flagged not hidden

**It treats the final bar as a month-end even if the month isn't over.** For the backtest this is completely harmless — remember next-bar execution: the engine never acts on the *last* signal (there's no "next bar" for it to earn a return on), so a slightly-early final rebalance changes no result. But for *live* trading someday, "is today the last trading day of the month?" genuinely can't be answered from price data alone — it needs a real exchange calendar (which days the market is open). That's noted for the live phase, not papered over.

**It uses raw closing prices, so it measures *price* return, not *total* return.** Total return includes dividends — the cash a fund pays out — and ignoring them understates how much a buy-and-hold investor actually earned (they collect every dividend; this strategy collects them only while it's holding). That biases any eventual strategy-vs-buy-and-hold comparison; Day 30 added the ability to switch the strategy and the backtester to total return — defaulted off, to be turned on for the verdict (Part 20).

### What "winning" will mean for momentum

A framing that matters for when the verdict comes: time-series momentum's classic claim to fame is **not higher returns** than buy-and-hold — it's **smaller drawdowns.** It's closer to insurance: by sidestepping the worst declines (as the SPY behavior above shows), it aims to deliver a smoother ride, not a bigger number. So when you eventually judge it, the honest things to look at are **max drawdown** and downside risk, *alongside* Sharpe and return — not return alone. Judge it only on total return and you'd likely call it a loss while completely missing the thing it's actually good at.

(Two commits, per the project's standing discipline: the strategy code first, then a separate development-log entry. The signal primitive is built and tested — but note what today did *not* do: it did not run momentum through the full out-of-sample machine. That verdict is still pending; today just built the trustworthy building block.)

---

## Part 17: Two Tools That Keep the Project Honest (Day 27)

Day 27 wasn't a trading concept — it was a pair of *engineering* tools that make the whole project more trustworthy and easier to work on. (Part 11 already showed the doc cares about codebase craft, not just trading; this is more of that.)

### Continuous Integration (CI) — letting a robot run your tests

Up to now, "the tests pass" rested on *you* remembering to run them and reporting back. That's fine until it isn't — you forget, or they pass on your laptop because of something only your laptop has.

**Continuous Integration (CI)** fixes that. CI is an automatic robot that runs your *entire* test suite every single time you push code to GitHub. A fresh, clean computer in the cloud checks out your code, runs the tests, and posts a green check (all passed) or a red X (something broke) right on the commit. You wire this up with **GitHub Actions** — GitHub's built-in automation service, which runs a script you define in response to events like "code was pushed." (Your script is a small configuration file living at `.github/workflows/` in the repo.)

Why this matters *specifically for this project*: the entire value proposition here is that the machine tells the truth and you **verify on disk rather than trusting a claim.** CI extends that exact discipline to the tests themselves. A passing CI run is *machine-produced proof*, posted publicly on the commit, not "trust me, it worked." For a portfolio project you'll show in interviews, that green check is the difference between *saying* "I wrote tests" and *demonstrating* that they pass on every change, automatically, on a machine that isn't yours.

One careful detail. Your test suite contains a handful of **integration tests** — an "integration test" is one that exercises real *external* systems (a live network call, the actual Alpaca broker API) rather than pure in-memory logic. Those can't run on a clean cloud machine: there are no broker credentials there, and you wouldn't *want* an automated test firing real API calls on every push. So the CI is told to **skip them** — it runs only the tests marked "not integration," which are the fast, self-contained ones that depend on nothing outside the code. (In the test command this is the `-m "not integration"` filter.) The result: CI runs the ~150-plus pure tests in seconds, every push, and stays green without ever needing the outside world.

### CLAUDE.md — house rules for your AI assistant

The second tool is a plain text file named **`CLAUDE.md`**, sitting in the repo. It encodes the project's *conventions* — the things you've been enforcing by hand all along: how commit messages should read, the verify-on-disk discipline, the "code first, then a separate dev-log entry" two-commit pattern, which files must stay untracked, the exact command to run the tests.

What makes it special is *who reads it*: it's loaded automatically at the start of every session with your AI coding assistant (Claude Code), so the assistant follows the project's house rules without being re-told each time. It's documentation written for an AI collaborator instead of a human one — a way to make "the way we do things here" stick across every session, so standards don't quietly drift.

(Both were committed as the Day 27 work.)

---

## Part 18: The Cost of Trading, Modeled (Day 28)

Every backtest number in this entire document, up to here, quietly assumed one false thing: **that trading is free.** It isn't. Day 28 added the friction — the first piece of Phase 3 — so the machine's results reflect the real cost of moving money around, not a frictionless fantasy.

### The two costs

When you trade, you lose a little money to two things:

- **Fees** (also called commissions): what the broker charges to execute an order. A direct, named cost.
- **Slippage**: the gap between the price you *expected* and the price you *actually got.* When your order hits the market, the price can move slightly against you before it fills — more so for large orders or in fast-moving markets. Nobody "charges" you slippage; it's just lost to the mechanics of how orders fill, but it's every bit as real as a fee.

### Basis points — how costs are measured

Costs are small, so traders measure them in **basis points (bps).** A basis point is **one hundredth of one percent** — so 1 bp = 0.01% = 0.0001 as a fraction, and 10 bps = 0.10% = 0.001. The model takes a fee in bps and a slippage in bps, and simply **adds them into one total cost rate.** (Three bps total — 1 bp fee plus 2 bps slippage — means every unit of trading costs you 0.0003 of the amount traded.)

### Turnover — the thing costs are charged on

Here's the key idea that makes the model correct: **you pay for *trading*, not for *holding*.** The measure of "how much trading you did" is **turnover** — the size of the change in your position.

- Entering a full position = **1 unit** of turnover.
- Exiting it later = **1 unit.** (So a complete round-trip — buy then sell — is 2 units.)
- Flipping straight from fully-long to fully-short = **2 units** at once (you close the long *and* open the short).
- **Holding a position you already have = 0 units.** No change, no cost.

The cost charged is turnover multiplied by the cost rate. So a strategy that sits in a position for months pays almost nothing; a strategy that flips in and out constantly pays a lot. This is precisely the warning from Part 10, now made real: the fast-trading SMA configurations that traded 5–10× more will *bleed* under costs, while a patient strategy barely notices.

### How it's wired into the backtester

The backtester now accepts a fee rate and a slippage rate, **both defaulting to zero.** On each bar it works out how much the position changed (turnover), multiplies by the cost rate, and **subtracts that from that bar's return.** From there, *everything downstream* — the equity curve, the Sharpe ratio, the total return, the max drawdown — automatically reflects the cost, with no other change needed, because they're all computed from the now-after-cost returns. The engine also reports the **total cost paid** over the whole run, as a fraction of your capital, so you can see exactly how much friction cost you.

Two engineering choices worth calling out, because they're the kind of care this project is built on:

**Default to zero, so nothing old breaks.** With both rates set to zero, the new code produces results *bit-for-bit identical* to before. Every earlier finding and every existing test still holds unchanged; costs are something you deliberately *opt into*, never a silent rewrite of past numbers. (All 156 prior tests stayed green, plus 7 new cost tests, for 163.)

**The cost is applied as a tiny, exactly-additive drag.** There's a more mathematically elaborate way to fold in costs, but the difference at realistic basis-point levels is microscopic, and the simple version has a real virtue: you can *add up the per-trade costs by hand* and check the engine's total against your arithmetic. Auditable beats clever. (And a deliberate scope line, flagged not hidden: for now only the *aggregate* numbers — Sharpe, return, drawdown — are after-cost; the per-trade win-rate is still measured before cost. That's a noted limitation to tighten later, not an accident.)

### The result — and why it barely moved

Run the 12-month momentum strategy on SPY twice, once free and once at 3 bps total cost:

| Metric | Free (gross) | After 3 bps cost (net) |
|---|---|---|
| Total return | 3.1236 | 3.1026 |
| Sharpe ratio | 0.5300 | 0.5281 |
| Max drawdown | 0.3410 | 0.3410 (unchanged) |
| Total cost paid | — | ~0.0051 (≈0.5% of capital) |

The cost barely dented the result. The drawdown didn't move at all (cost shifts the whole curve down slightly but doesn't deepen the worst trough). Why so small? Because **momentum trades rarely** — about 9 round-trips across roughly 18 years of history. This is exactly the point flagged back in Part 10: the "after-cost" number is the honest test of whether a thin edge *survives friction*, and a patient strategy like momentum survives nearly intact, whereas a hyperactive one would have been gutted.

(Important: this is *not* yet a verdict that momentum has edge — it only shows that costs don't *kill* it. The real verdict still comes later, through the full out-of-sample machine.)

### The bug it caught — the project's recurring lesson, one more time

While running that sanity check, the inspection turned up a **data defect**: SPY's most recent stored bar (June 10, 2026) had a **missing closing price** — a `NaN`, the computer's marker for "not a number," an undefined value — probably left behind by a stale or half-finished last-day download. That single bad value **poisoned** the full-history return and drawdown: any arithmetic that touches a NaN produces NaN, so it spread through the equity curve and turned the headline numbers into "not a number." (The clean table above came from re-running on the history minus that one corrupt final bar.)

It was a *data* problem, not a cost-model problem — the cost numbers themselves came out clean, because they don't depend on prices. But it surfaced two things that became the whole of the next day (Part 19): the bad data had to be cleaned — and the other 16 ETFs checked for the same corrupt-final-bar issue — and the backtester had to **refuse to run on a non-finite price** rather than silently emit NaN results.

This is the **Day 21 duplication lesson, all over again** (Part 3): *"the data loaded" is not "the data is correct."* The numbers were computed perfectly — on a row that was garbage. It surfaced only because someone actually looked at the output and a NaN where a return should be failed the smell test. Sanity-check the numbers, not just whether the code ran.

(Two commits, per the standing discipline: the engine and result changes first — together with a small fix to the engine's own description, which had still claimed costs weren't modeled — then the separate dev-log entry.)

---

## Part 19: Sealing the Pipeline Against Corrupt Data (Day 29)

Day 28 ended on a discovery, not a fix: a single missing closing price — a `NaN`, the computer's marker for "not a number" — sitting on SPY's most recent stored bar, quietly turning a whole-history return and drawdown into "not a number." Day 29 is the day that hole got closed, in three moves: find the full extent of the damage, delete it, and make it structurally impossible for the same kind of garbage to either get stored again or be silently computed on.

### First, find out how bad it really is

The Day 28 finding was on SPY alone, because SPY is the symbol that happened to get looked at. The first move on Day 29 was a read-only audit of *all 17 ETFs* — because a defect you've only seen in one place might be one place or might be everywhere, and you don't clean anything until you know which.

A quick definition the audit leans on: a number is **non-finite** if it's either `NaN` or infinity (positive or negative). Those are the values that have no business in a price series and that poison any arithmetic they touch. The audit scanned every bar of every ETF — the close *and* the dividend-adjusted close — for any non-finite value.

The result was clean to read and slightly worse than feared: **every one of the 17 ETFs had exactly one bad bar, and in every case it was the same one** — the trailing 2026-06-10 row, with both close and adjusted close `NaN`. No bad values buried earlier in any history (a mid-series hole would have been a far nastier problem); no infinities; just one corrupt bar at the very end of each of the 17 series. The shared date and shared signature point straight at the cause: a single data update fetched the whole basket on a day before the market's close had settled, so the source handed back a placeholder `NaN` for every symbol at once, and all 17 got stored. It wasn't a SPY fluke — it was one stale fetch hitting the entire basket. That diagnosis is *why* the rest of the day mattered: a systematic cause will recur on the next update unless something stops it.

### Second, delete the damage

Cleaning stored data is done with a **migration** — a one-off script that changes what's already in the database, kept in the repo (not run by hand and forgotten) so the fix is documented, reviewable, and repeatable. This one deletes every row with a non-finite close or adjusted close.

Three pieces of care went into it, the same instincts as the Day 21 deduplication fix (Part 3):

- **Back up first.** The database was copied to a backup file before a single row was touched. A destructive change you can't undo is a change you don't make.
- **Verify the tools, don't assume them.** The script needed two specific database functions to test for `NaN` and infinity. Rather than assume they existed and hope, it *probed* the database to confirm both were available and behaved correctly before building the deletion on top of them. (They were.) Small thing — but assuming a function exists and being wrong is exactly how a "cleanup" silently deletes nothing, or the wrong thing.
- **A sanity gate.** The audit said 17 bad rows. The script counted the rows it was about to delete and *refused to proceed unless that count was exactly 17* — if the data didn't match what the audit had described, it would stop, untouched, rather than delete a number of rows nobody had eyeballed. It matched: 17 found, 17 deleted, 0 non-finite rows remaining.

And the result was confirmed **on disk independently**, not just taken from the script's own success message — a separate one-line count, run by hand against the database file, returned zero. (The recurring discipline of this whole project: a tool reporting success is a claim; a number you check yourself is evidence.)

### Third — and most important — make it impossible to recur

Deleting the bad rows fixes *today*. It does nothing about tomorrow's update doing the exact same stale-fetch trick. So the heart of Day 29 was adding two guards, at the two different places bad data can do harm — and the two guards behave *differently*, on purpose, because the right response depends on where you are.

**At the write boundary — where data gets stored — skip and warn.** The storage layer now inspects every incoming bar and silently drops any whose close or adjusted close is non-finite, logging a warning that names the symbol and date it skipped. The key choice here is that it does *not* crash. Data gets stored in big batches — the whole 17-ETF basket at once — and one bad trailing bar must not abort the entire ingest and throw away the sixteen good symbols that came with it. So the bad bar is quietly dropped, a warning is recorded, and the good data flows through. This is the *prevention*: the same stale fetch that caused the mess can never again deposit a `NaN` into the store.

**At the compute boundary — where a backtest runs — refuse outright.** The backtester now checks, before it does any return math, that every close is finite *and* strictly positive; if any close is non-finite, zero, or negative, it raises a loud error that says how many bad values there are and points at the first one by date. Here the choice is the opposite of the write boundary: it *does* stop, hard. Why the difference? Because at compute time there is no safe way to continue — a `NaN` close has no meaningful return, and quietly producing `NaN` metrics is precisely the Day 28 failure that started all this. Better to fail loudly with a message that names the culprit than to hand back a number that's secretly garbage. (The "strictly positive" part catches a subtler corruption: a price of zero or below is not just wrong, it's un-loggable — the return math takes the logarithm of a price ratio, and the log of zero is negative infinity, the log of a negative is undefined. So zero and negative prices are rejected alongside `NaN` and infinity.)

There's a quiet reason *both* guards were needed, not just one. The store is built so that re-fetching a date it already has is a no-op — it won't overwrite an existing row (this is the deduplication safety from Part 3). That's normally a virtue, but it has a sharp edge here: a bad row already sitting in the database can't be fixed just by re-downloading that day; the bad row has to be *deleted first*, and only then can a clean re-fetch put a good bar in its place. That's the whole reason Day 29 needed a one-off deletion *and* a write-time guard — the migration removes the damage that's already there, the guard stops new damage from arriving. Two problems, two tools.

### The shape of the fix, and what it cost

Step back and this is the **Day 21 pattern, one level richer.** The duplication bug was fixed with prevention (stop it at the write path) plus cleanup (a one-off migration of the existing damage). The `NaN` bug got the same two — prevention at the write boundary, a one-off cleanup migration — plus a third layer the duplication bug never had: *detection at the compute boundary*, so that even if some future corruption slips past everything else, it surfaces as a clear error instead of a silent `NaN`. Prevent, clean, and detect.

It came with six new tests — five pinning down that the engine rejects `NaN`, infinity, zero, and negative closes while still running fine on a clean series, and one proving the store drops a non-finite bar and keeps the good ones — taking the suite from 163 to **169**, all green.

What Day 29 deliberately did *not* do: it didn't re-fetch the now-missing recent bars (that's optional housekeeping, and the basket is a couple of weeks stale either way), it didn't touch the dividend/total-return accounting still owed before any verdict, and it ran no strategy comparison. It was pure foundation work. But it's the foundation every later number stands on: when the momentum verdict finally comes, it will be computed on data the system actively refuses to corrupt, by an engine that refuses to run on garbage. That's worth a day.

---

## Part 20: Total Return — Teaching the Backtest to Count Dividends (Day 30)

Back in Part 16 there was a caveat flagged but not fixed: the momentum strategy reads raw closing prices, so it measures *price* return — how much the price moved — and quietly ignores **dividends**, the cash a fund pays out to the people holding it. Day 30 is the day that gap got closed. It is the second of three "make the comparison honest before you judge anything" items: transaction costs landed on Day 28 (Part 18), this is total-return accounting, and cash-on-the-sidelines is still to come.

### Price return vs total return — and why the difference is not small

Two ways to measure how much you made holding something:

- **Price return** looks only at the price: buy at 100, it's 110 later, that's +10%. Dividends are invisible to it.
- **Total return** counts the price move *and* the dividends you collected along the way. If that same fund also paid you $3 per share in dividends while you held it, your real gain was +13%, not +10%.

For a single fast trade the difference is tiny. But over the long horizons this system tests — years of history — dividends compound into a large chunk of an investor's actual return. A broad US stock fund pays roughly 1.5-2% a year in dividends; over a decade that is a meaningful slice of total profit. Ignoring it doesn't just make the numbers slightly low — it systematically *understates* what a buy-and-hold investor really earned.

And that is exactly why it matters *here*, for the verdict that's coming. A buy-and-hold investor is always holding, so they collect *every* dividend. The momentum strategy is only invested *some* of the time (it steps aside to cash when the trend rolls over), so it collects dividends only while it happens to be holding. If the backtest measures both on price return alone, it strips dividends out of *both* — but it strips more out of the always-invested benchmark than out of the sometimes-invested strategy, quietly tilting the comparison in the strategy's favor. To judge momentum fairly against buy-and-hold, both have to be measured on total return.

### The fix: adjusted close

The data already contains what's needed. Recall from Part 2 that each bar carries several prices. One of them, the **adjusted close** (`adj_close`), is the closing price with dividends and stock splits already folded back in — a version of the price series engineered so that its bar-to-bar changes *are* total return. (Yahoo computes it for you; this is the same `adj_close` field the Day 29 audit was checking for NaNs.) Switch the math from reading `close` to reading `adj_close`, and price return becomes total return with no other change. Raw `close` gives price return; `adj_close` gives total return — that is the entire mechanism.

### Built as a switch that defaults to off

Here is the important design decision, and it is the same instinct that runs through the whole project. Day 30 did **not** simply rip out `close` and hard-wire `adj_close` everywhere. Instead it added a **configurable knob** — a setting called `price_field` — that chooses which price to read, and **defaults to `close`** (the existing behavior).

Why default to the old behavior? Because the project's iron rule is that a change to shared code must not silently alter a single past result. With the knob left at its default, every line of math reads `close` exactly as before, so all 169 prior tests stay **bit-for-bit identical** — a "no-op," a change that changes nothing until you ask it to. Total return is something you now deliberately *opt into*; it is never a silent rewrite of numbers you already recorded. The eventual verdict run will turn the knob to `adj_close` on purpose, knowing the numbers are meant to move.

The alternative — flipping everything to `adj_close` in one shot — would have re-written every historical figure at once, which is precisely the kind of large, all-at-once change where a wrong number can hide unnoticed. Off-by-default is the safer, more reversible path, and it keeps the day's commits clean and additive.

### Why the knob had to go in two places, not one

This is the subtle part, and it is worth slowing down for, because it shapes a real risk.

The two pieces of the system that read a price are **decoupled on purpose** (Part 6's separation of concerns):

- The **backtester** reads prices to compute the profit-and-loss.
- The **strategy** reads prices to decide its signal (momentum's trailing-12-month return).

The backtester doesn't know which strategy produced the signals it's scoring; the strategy doesn't know which backtester will score it. That decoupling is a virtue — it's why any strategy slots into the same engine — but it means there is **no single switch** that flips the basis everywhere. The knob had to be added in *both* places: one on the backtester, one on the momentum strategy. For the two to agree, the code that *builds both for a run* has to hand them the same setting.

There is one happy exception. The **buy-and-hold benchmark** needs no separate change at all. Recall from Part 14 that buy-and-hold is computed by running an always-long signal through the *same* backtester — so whatever price basis the backtester is set to, the benchmark automatically inherits it. Strategy and benchmark move together by construction, which is exactly the apples-to-apples property the verdict depends on.

### The footgun, flagged not hidden

Because the basis lives in two places and defaults to `close`, there is a real trap waiting at verdict time: if the code that runs the verdict remembers to switch the *strategy* to `adj_close` but forgets the *backtester* (or vice versa), nothing errors — it just silently computes a mismatched, half-total-return result. The default that protects today's reproducibility is the same default that will quietly do the wrong thing if a future caller is careless.

This isn't left to chance. The mitigation is deliberate and deferred to the day it's actually needed: the verdict's entry point will pass the same basis to both the engine and the strategy explicitly, in one place, so they cannot drift apart. And the tests written today (below) guarantee that the `adj_close` path genuinely produces *different* numbers than `close` — so a silent no-op, where the switch did nothing, would be caught rather than mistaken for a real result.

### How do you test a switch that's off by default?

This is a neat little testing problem. If the knob defaults to `close` and the test data has `adj_close` equal to `close` (as all the old synthetic test bars did), then turning the knob to `adj_close` changes *nothing* — and a test that "passes" proves only that the switch does nothing, which is worthless.

So the proof tests were built around data where the two price series **deliberately diverge**. The engine tests construct bars whose `close` path rises while their `adj_close` path takes a different shape, run the same bars through both settings, and assert the two total returns come out **different** by a real margin — proving the switch actually changes the math. A companion test does the opposite: feed in bars where `adj_close` equals `close`, and assert the two settings produce *identical* results — proving the difference is driven by the *data* diverging, not by the flag doing something spooky on its own. The momentum strategy got the same treatment: a rising `close` path (which says "be LONG") against a falling-but-still-positive `adj_close` path (which says "be FLAT"), so the two settings produce genuinely different signal arrays — the basis flips the actual decision, not merely a number's last digit.

That's five new tests in all (four on the engine, one on the strategy), taking the suite from 169 to **174**, all green — the prior 169 untouched (proving the default is a true no-op) plus the five that prove the opt-in works.

### What Day 30 did not do

It did not flip the basis — by default the system still measures price return, exactly as before. It did not add cash-on-the-sidelines accounting (the idle-cash interest, which is the next day's job). And it ran no strategy comparison and produced no verdict. This was pre-verdict plumbing: the *capability* to measure total return now exists and is proven to work, switched off, waiting to be turned on when the verdict is run on honest, dividend-inclusive numbers.

With this, two of the three pre-verdict accounting items are done — transaction costs (Day 28) and total return (Day 30). Cash-on-flat is the last one (Part 21). After that, the basis gets wired through the verdict path and momentum finally faces the full machine.

(Two code commits, per the standing discipline — the engine change first, then the matching strategy change, each with its tests — followed by the separate development-log entry.)

---

## Part 21: Cash on the Sidelines — Paying Interest on Idle Money (Day 31)

Two of the three "make the comparison honest before you judge anything" items were already in: transaction costs (Day 28, Part 18) and total return (Day 30, Part 20). Day 31 closed the third and last one — **cash-on-flat** — so the machine is finally ready to judge momentum on fully honest numbers.

### The gap: idle cash that earns nothing

Recall what the momentum strategy actually does (Part 16): when the trend is up it holds the fund (LONG), and when the trend rolls over it steps aside to **cash** (FLAT). The backtest, up to now, treated those out-of-the-market stretches as earning *exactly nothing* — flat cash, zero return, a dead patch on the equity curve.

That isn't how cash works in the real world. Money sitting in a brokerage account isn't stuffed in a mattress; it earns interest — the **risk-free rate**, the return you get for taking essentially no risk at all (think a Treasury bill, or the interest a broker pays on an idle balance). Over the last couple of years that rate has been around 4-5% a year. A strategy that spends months in cash is, in reality, quietly collecting that interest the whole time — and a backtest that books zero for those stretches is *understating* what the strategy truly earned.

### Why this matters most for a strategy that sits out

Here's why this isn't a rounding-error detail, and why it lands squarely on the verdict that's coming. The whole *point* of momentum is to step aside during the bad times — it was FLAT through almost all of 2008 and 2022 (Part 16). That's a lot of time in cash. Compare it to buy-and-hold, which is *always* invested and so never has idle cash to earn interest on. If the backtest credits neither of them with cash interest, it is penalizing the strategy precisely for the defensive behavior that's supposed to be its strength: every month momentum spends safely in cash is a month it earns nothing in the model, while in reality it would be earning the risk-free rate. Crediting that interest is what makes the eventual momentum-vs-buy-and-hold comparison fair to *both* sides.

### The one genuinely tricky part: a yearly rate is not a daily rate

This is the piece worth slowing down for, because it's the kind of mistake that produces a number that looks plausible and is wildly wrong.

The backtest works in **daily** steps — one bar per trading day. The interest rate you'd naturally want to type in, though, is an **annual** one: "cash earns 4% a year." Those are not the same number, and confusing them is a disaster. There are about **252** trading days in a year, so 4% *per year* is only about **0.016% per day**. If you carelessly fed "4%" into the engine as a *daily* rate, you'd be crediting 4% every single day — compounding to something absurd like a *ten-thousand-percent* annual return. An error of a factor of 252, hiding inside a number that, at a glance, looks like a perfectly reasonable interest rate.

So Day 31 made the setting explicitly an **annual** rate, and the engine does the conversion itself: it takes the yearly rate you give it and divides it down to the correct per-day amount, in the same units the rest of the return math uses. The property that makes it correct is checkable by hand — credit that tiny per-day amount on every one of the ~252 trading days in a year, add them up, and you get back exactly the annual rate you started with: 4% over the year, not 4% a day. You pass in a familiar yearly percentage; the engine guarantees it doesn't get multiplied into nonsense.

(A small note for the curious: the conversion is done in the same "log return" units explained in Part 5 — the form where daily numbers *add up* cleanly over time — which is why a year of daily pieces sums back to the annual figure exactly rather than approximately.)

### Where the interest gets applied — and one subtle edge

The rule is simple: on every bar where the strategy is **FLAT** (holding cash), credit the little per-day interest amount; on every bar where it's holding the fund, credit nothing extra (it's earning the fund's return instead, not cash interest). The backtest already tracks, bar by bar, whether a position is held, so spotting the flat bars is free.

There's one subtle edge that had to be handled with care — the same "first bar is special" detail that runs through the whole engine (Part 7). The very first bar of any run is structurally flat: there's no prior signal yet, so the strategy is technically "in cash" on day one by construction, not by choice. That first slot is a deliberate placeholder the engine relies on elsewhere — the Sharpe-ratio math and the walk-forward stitching both depend on it being exactly zero. So the interest credit is explicitly forced to skip that first bar: it applies to genuinely-flat days *after* the start and leaves the structural first-day zero untouched. Miss that, and you'd quietly leak a phantom day of interest into a slot other calculations assume is empty — corrupting the very risk metric the verdict will be read on. A dedicated test pins it: on an all-cash run, day one's return comes out exactly zero, every time.

### Buy-and-hold, untouched again

Just like the total-return change (Part 20), the buy-and-hold benchmark needs no special handling. Buy-and-hold is *always* invested — it's never flat (except that structural first bar, which the skip above zeroes out anyway) — so it never has idle cash and earns no cash interest, which is exactly right. The strategy collects interest while it sits out; the always-invested benchmark doesn't. The comparison stays honest by construction.

### Off by default, opt-in, and reported

The same iron rule as every shared-code change in this project: the new interest rate **defaults to zero.** Leave it alone and the backtest behaves bit-for-bit as it did before — all 174 prior tests stay green, untouched. Cash interest is something you deliberately turn on for the verdict, never a silent rewrite of past numbers. (The engine also rejects a *negative* rate outright — cash can't lose money just by sitting there — the same way it already rejects negative trading costs.)

And, mirroring how the cost model reports the total fees paid (Part 18), the engine now also reports the **total interest earned** over a run, as a fraction of capital — so when the verdict comes, you can see at a glance how much of the strategy's return came from sitting in cash versus from the trades themselves. An income line to sit beside the cost line.

### The test that proves it

The most important of the six new tests is the one that guards the annual-vs-daily trap directly. It builds a full year of trading days — 252 of them — on a flat, do-nothing position, turns on a 4%-a-year cash rate, and checks that the whole year earns **about 4% total**, not 4% per day. If the conversion were wrong by that factor of 252, this test would show a preposterous number and fail loudly. It's the arithmetic equivalent of the no-lookahead test (Part 7): a single check positioned exactly where the dangerous mistake would land. (The other five cover the no-op default, the skipped first bar, an all-invested run earning zero interest, a mixed run earning interest only on its flat stretch, and the negative-rate rejection — taking the suite from 174 to **180**, all green.)

### What Day 31 did not do, and the milestone it reached

It modeled the interest as a clean, exactly-additive credit — the same audit-friendly choice as the cost model, where you can add the pieces up by hand. It did **not** turn the rate on for any real run, did not wire it into the verdict path, and produced no verdict. Pure pre-verdict plumbing, switched off, waiting.

But it's a milestone: **all three pre-verdict accounting items are now done** — transaction costs, total return, and cash-on-flat. The backtest can now, when asked, charge realistic trading friction, count dividends, *and* pay interest on idle cash — the three things standing between a frictionless fantasy and an honest account of what a strategy would really have earned. What's left before the verdict is no longer accounting; it's *wiring* — handing all these switches to the verdict run together, in one place, so none of them gets forgotten (Part 23).

(Two commits, per the standing discipline: the engine change and its tests first, then the separate development-log entry.)

---

## Part 22: Where the System Stands

### The full vertical slice (through Day 31)

```
Yahoo Finance (the world's price data)
       ↓
[YFinanceFetcher] downloads bars
       ↓
[DuckDBStore] saves them locally  (midnight-UTC floor — Day 21;
              non-finite price bars skipped on write — Day 29)
       ↓
[Universe loader]  S&P 500  +  etf_basket (17 ETFs, backfilled to 2008)  ← Day 22
[Backfill / Update / Health Check]  now with content checks              ← Day 23
       (zero-volume = hard-fail, extreme-jump = flag-for-human)
       ↓
[OHLCVBar list] pulled from the database
       ↓
[Indicators] turn bars into numpy feature arrays           ← Day 13
       ↓
[Strategies] turn features into position signals           ← Day 14
   • SMACrossoverStrategy (the crash-test dummy)            ← Day 14
   • TimeSeriesMomentumStrategy (first real thesis)         ← Day 26
     (configurable price basis: close / adj_close)          ← Day 30
       ↓
[Backtester] turns signals into a P&L curve                ← Day 15
   (metric math extracted to metrics.py)                   ← Day 21
   (transaction costs: fees + slippage on turnover)        ← Day 28
   (rejects non-finite / non-positive closes)              ← Day 29
   (configurable price basis: close / adj_close)           ← Day 30
   (cash-on-flat: interest on idle capital while FLAT)     ← Day 31
       ↓
[BacktestRunner] run_many / run_universe / run_matrix      ← Days 16–18
       ↓
[compare_strategies / compare_universe / compare_matrix]   ← Days 16–18
   (shared loading logic factored into cli_common)         ← Day 19
       ↓
[walk_forward_splits]   cuts bars into train/test folds    ← Day 20
[walk_forward_validate] warm-up-on-train, score-on-test,
                        stitched out-of-sample track record ← Day 21
   + buy-and-hold benchmark over identical windows         ← Day 24
   + fit_fn seam: optionally tune a strategy per fold       ← Day 24
[compare_walkforward]   per-fold table + OOS / B&H / Δ      ← Days 21, 24
       ↓
[make_sma_optuna_fit_fn] per-fold Optuna tuning,
                         warm-only objective                ← Day 24
       ↓
[overfitting_tax]  fixed vs tuned vs buy-and-hold,
                   measured across the basket               ← Day 25
       ↓
[CI on GitHub Actions]  runs the test suite on every push  ← Day 27
[CLAUDE.md]             project conventions for the AI      ← Day 27
       ↓
   ─────── Coming up ───────
[basis + cash-yield wired through the verdict path]         ← verdict-day wiring
[Momentum, run through the full harness]  the real verdict  ← next
[Cross-sectional rotation]  rank-and-hold, target weights   ← later phase
[Risk module] position sizing, stops                        ← Phase 3 (rest)
[Machine learning] LightGBM, ensemble models               ← Phase 4
[Live executor + monitoring] real orders via Alpaca        ← Phase 5
```

### What you actually have now

1. **A clean, deduplicated, regime-spanning data pipeline.** Yahoo Finance → DuckDB, refreshed daily, health-checked automatically — deduplicated and timezone-canonicalized (Day 21), spanning a fixed 17-ETF basket back to 2008 (Day 22) with both structural *and* content checks (Day 23). (The corrupt trailing bar found on Day 28 was cleaned on Day 29, and the store now refuses to write a non-finite price, so it cannot recur.)
2. **A broker abstraction.** Clean interface to Alpaca paper trading with hard safety guards against going live by accident.
3. **A feature layer.** Three vectorized indicators matching industry-standard implementations, all NaN-aligned to bars.
4. **A decision layer with two strategies.** The SMA crossover (the deliberately-empty crash-test dummy) *and*, now, time-series momentum — the first strategy with a genuine economic hypothesis — both emitting position signals in the same alignment contract. (As of Day 30, momentum can compute its signal from price return or total return, to match the backtester's basis.)
5. **A truth-telling backtester.** Structurally-impossible lookahead, frozen results, all five core metrics living in a shared `metrics.py` — and now charging realistic transaction costs (fees + slippage on turnover), defaulting to zero so every prior result stands, and, as of Day 29, refusing to run at all on a non-finite or non-positive close rather than silently emitting NaN metrics. As of Day 30 it can also compute returns from the dividend-adjusted close (total return) instead of the raw close (price return), via a price-basis setting that defaults to the raw close so every prior result stays unchanged. And as of Day 31 it pays a configurable interest rate on idle capital during the bars the strategy sits in cash (FLAT) — again defaulting to zero, so prior results are unchanged.
6. **A research layer with all four axes filled** — single backtest, strategies-on-one-symbol, one-strategy-across-symbols, and the full matrix. Three CLIs sharing a common data-loading core.
7. **Walk-forward validation, end to end** — a tested splitter (Day 20), a validator that warms-on-train and scores-on-test into one out-of-sample track record (Day 21), a buy-and-hold benchmark on identical windows (Day 24), and a per-fold tuning seam (Day 24).
8. **Per-fold parameter optimization** — an Optuna tuner that re-fits the strategy on each training window without ever seeing the test data, with a warm-only objective (Day 24), plus the CLI that measures the overfitting tax across the basket (Day 25).
9. **The headline finding, measured honestly** — the SMA crossover has no out-of-sample edge, *fixed or tuned*, and a ~1.0-Sharpe overfitting tax; buy-and-hold beats it on 15 of 17 funds.
10. **Automated, machine-verified testing** — a CI workflow on GitHub Actions runs the suite on every push (Day 27), and a CLAUDE.md conventions file keeps every AI-assisted session on the project's house rules.
11. **A test suite of 180 tests** running in seconds, including the critical `test_no_lookahead_bias` guards in *both* the backtester and the momentum strategy, the cost-model tests, the finite-close and write-boundary guards from Day 29, the price-basis tests from Day 30 (proving total return differs from price return only when the data diverges), the cash-on-flat tests from Day 31 (including the proof that a yearly interest rate is spread correctly across the ~252 trading days in a year, not charged every day), and the reproducibility / warm-only guards on the tuner. Five bugs caught before they could mislead — two separate timezone problems (the Day 21 one had been silently corrupting every Sharpe), a NaN-Sharpe edge case, a partial-history gap, and the corrupt-final-bar data defect surfaced on Day 28 (cleaned, and guarded against at both the write and compute boundaries, on Day 29).

### The principles that guided all of it

**Separation of concerns.** Each part does one job — the fetcher fetches, the store stores, the indicator computes, the strategy decides, the backtester simulates, the runner orchestrates, the CLI presents. None know each other's internals. This is why the test suite is fast (any layer tests in isolation) and why swapping any component is a one-file change.

**Raw numbers everywhere, formatting only at the edge.** Internal layers pass numbers; only the CLI formats them. Every layer composes with the next without parsing strings.

**Idempotency.** Running backfill twice produces the same state as once — restartable pipelines. (With the Day 14 caveat: it protects against duplication, not under-coverage.)

**Reproducibility over convenience.** Frozen results, pure functions, stateless runners. A computation done today reproduces exactly a month from now.

**Top of the table = best, always.** Whatever metric you sort by, the best row is on top. Convention in the code, not in the user's head.

**Abstract only once the pattern is real.** Day 19 and Day 21 both waited for a third call site before extracting a shared helper — premature abstraction guesses wrong and creates more complexity than it removes.

**Match the existing contract before inventing a new one.** Time-series momentum (Day 26) was chosen partly *because* it fits the position-signal contract unchanged — reuse over rework. The shape you got right once keeps paying off.

**Isolate the part where correctness is fragile.** The Day 20 splitter, and the momentum no-lookahead test, both target exactly the spot where a silent off-by-one would reintroduce lookahead bias. One hard thing at a time.

**Default new behavior to off.** The Day 28 cost model defaults to zero so it can't silently change a single prior result. New capability is opt-in; old findings stay reproducible.

**Trust the data only after you've checked it.** The Day 21 duplication bug and the Day 28 corrupt-final-bar both passed every test and ran in every backtest — they surfaced only because a number failed a human smell test. Numbers computed correctly on bad rows are still wrong. (Day 23's content checks and Day 29's write-boundary skip are this principle built into the pipeline; Day 29's finite-close guard extends it into the backtester itself.)

**One shape contract, applied recursively.** Bars → indicators → strategies → backtester → runner. Each layer transforms data without breaking alignment. The single most important architectural decision in the project.

**A negative result, delivered honestly, is the product.** The machine's job was never to make a strategy win — it was to tell the truth about whether *anything* wins. Measuring "no edge, with a ~1.0 overfitting tax" cleanly, instead of hiding it behind a flattering in-sample number, is the whole point. A framework that only ever delivered good news would be worthless — and dangerous.

---

## Part 23: What's Coming Next

The state of play: you now have the first strategy with a real reason to work (momentum), built as a trustworthy signal, *and* all the accounting realism needed to judge it fairly — trading costs, total return, and interest on idle cash. What you do **not** yet have is the verdict — momentum has not been run through the full out-of-sample machine. The immediate road is about earning the right to that verdict honestly.

**The last step before the verdict is wiring, not accounting.** All three pre-verdict accounting items are now built — transaction costs (Day 28), total return (Day 30), and cash-on-flat (Day 31). What's left is to connect them to the verdict run. Two of them are switches that default *off* and live in more than one place: the total-return basis sits on *both* the strategy and the backtester (they must agree), and the cash-yield rate sits on the engine. So the code that stands up the verdict has to hand the *same* settings to every piece explicitly, in one spot — otherwise it silently mixes total return with price return, or forgets to credit the idle cash, and the result looks plausible while being subtly wrong. That call-site wiring — threading the basis and the cash rate through the walk-forward path together — is the last thing to build before momentum faces the machine.

**Re-fetch to un-stale the basket (optional, operational).** Day 29 already did the hard part — it audited all 17 ETFs, cleaned the corrupt trailing bars, and hardened both the write boundary and the engine against non-finite prices (Part 19). What remains is purely operational: the deleted 2026-06-10 bars and the roughly two weeks of fresh data since then should be re-fetched so the basket is current. The new write-boundary guard now protects that re-fetch from re-storing a bad trailing bar, so it's a low-risk catch-up — not a prerequisite for anything below.

**Then run momentum through the full harness — the actual verdict.** Walk-forward validation, per-fold Optuna tuning, the overfitting tax, and the buy-and-hold benchmark — the exact machine that judged the SMA dummy — turned on momentum, after costs. Two things to get right: the walk-forward windows must be sized against the 12-month lookback (a test window needs to span multiple years to have enough warmed-up signal to mean anything), and the verdict must be read on **drawdown and downside risk**, not return alone — because, as Part 16 explained, crisis-avoidance is the thing momentum is actually supposed to deliver.

**After that — cross-sectional rotation.** The relative-momentum flavor from Part 16: rank the 17 ETFs, hold the strongest few, rebalance monthly. This is its own phase because it breaks the one-symbol contract — it needs ranking across symbols and target weights, which means a portfolio-level backtester.

**Phase 3 (the rest) — risk management.** Beyond costs, the parts that keep a real edge from blowing up: **position sizing** (how much to bet on each trade) and **stop-losses** (automatic exits that cap a loss). *"The best traders aren't the ones with the best entry signals — they're the ones with the best risk management."*

**Phase 4 — machine learning.** Instead of hand-written rules, train models (LightGBM, ensembles) to learn from dozens of indicators at once. Your NaN-alignment contract pays off here — aligned indicators snap into a clean feature matrix — and your `Strategy` base class means an ML model slots into the same backtester and the same tuning/validation harness you've now proved out twice.

**Phase 5 — live execution.** Real money, eventually, staged carefully: paper → small real → scaled. The way every responsible team deploys.

---

## Part 24: The Takeaway

You've now spent 31 days building something real:

- A clean, deduplicated, regime-spanning data pipeline; a broker abstraction with safety guards; industry-standard indicators; a strategy layer with abstract base classes — now holding *two* strategies, the empty crash-test dummy and the first one with a real economic thesis; a backtester where lookahead is structurally impossible and trading costs are modeled; a research layer with all four axes filled; walk-forward validation; per-fold parameter optimization with the overfitting tax measured; and automated, machine-verified testing on every push — over a data layer and backtester that now actively refuse to store or compute on corrupt prices, and that can measure total return, dividends included, and pay interest on idle cash, not just price change, when asked.
- 180 tests running in seconds.
- A research arc that landed in the most valuable place it could — a strategy proven empty, and a machine that proved it without flinching — and has now turned to a strategy with an actual reason to work, with the verdict still honestly pending.

Most retail quant projects fail because the data is broken, the backtest cheats, the indicators don't match reality, or the research is a tangle of copy-pasted scripts. You handled all of that. Then you did the thing almost nobody does: you took a strategy, tuned it as hard as an optimizer can, ran it honestly on data it had never seen, and let the machine tell you it was worth nothing — a ~1.0 Sharpe of pure self-deception, beaten by buying and holding. You didn't argue with the result. You recorded it and moved on.

That single act — building a thing that can find candidate winners *and* willingly disbelieve them until they survive out-of-sample — is the entire difference between quant research and data-mining theater. The chassis is done, the honesty instrument is installed and calibrated, and the first real engine part — a strategy with a genuine reason to work — is now bolted in, along with the cost realism needed to judge it fairly. Whether momentum actually earns its keep, you don't know yet. But you've built the one thing that can tell you the truth about it — and you've already watched it refuse to lie.

You're not flipping sneakers anymore. And you're not fooling yourself either — which, in this game, is most of the battle.