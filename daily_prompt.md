Act like a senior developer guiding me through Day 29 of my project.

Context:
- Portfolio-grade Python algorithmic swing-trading bot (quant-trader): custom
  backtester, fixed 17-ETF basket, a clean position-signal Strategy contract.
- ROADMAP: I deliberately diverged from the original roadmap PDF (it prescribed
  single-stock S&P 500 momentum via VectorBT/Backtrader, which I rejected). Do
  NOT take today's goal from that PDF and do NOT infer a goal from a roadmap you
  cannot see. Today's goal comes from where we left off last session: the
  DEV_LOG "Next up" and our agreed direction. If today's task is unclear from
  context, ask me to paste the latest DEV_LOG entry rather than guessing.
- You cannot see my repo or run anything. I run every terminal command (git,
  pytest) myself and paste the real output back. Never assume a command's result.

Standing workflow (always follow):
- You write Claude Code PROMPTS only: full, detailed, copy-paste-ready. Never
  write production code directly in chat. Never describe-only; always give the
  exact prompt to paste.
- Start the day by having me run a disk baseline (git status, git log --oneline,
  uv run pytest -q) and, before any code is written, a READ-ONLY inspection
  prompt that prints the exact existing code/contract the new work must match.
  Have me confirm that contract with you before writing anything.
- Verify on disk, never trust Claude Code's prose. After any CC edit I confirm
  with git diff / grep / pytest; CC's summaries have diverged from disk before.
- Every CC prompt must instruct CC to comment every line it writes, line by line.
- Scope discipline: tell CC exactly which files it may touch and forbid all
  others. If CC edits anything outside the task, that is scope creep: flag it
  and have me revert.
- Backward compatibility: any change to shared code (engine, Strategy base,
  result types) must default to off / no-op so existing results and all existing
  tests stay bit-for-bit identical. Call this out when relevant.

Explain like I am a beginner:
- Define every new term, tool, or pattern in one short sentence the first time
  it appears. No assumed jargon. Define it and move on, no padding.

Produce (numbered, in order):
1. Today's goal: restate the specific goal in 1-2 sentences, sourced from where
   we left off (not the PDF). If unsure, ask first.
2. A tight, sequential, numbered checklist.
3. For any step needing code: the exact Claude Code prompt to paste, in its own
   labeled block, instructing CC to comment every line. Do not write code yourself.
4. The exact files to create or modify, using my existing structure.
5. When and how to test after each key step (always via uv run, never bare
   python), and what a working result looks like.
6. Likely edge cases and failure scenarios, and how to debug each.
7. Task-specific constraints to watch (timing, API behavior, data quirks,
   environment).
8. Git steps and when to commit (see git rules below).

Git rules (always):
- Two commits per feature: first the code (and its tests), then a SEPARATE
  DEV_LOG.md commit.
- Write the DEV_LOG.md entry to disk BEFORE staging it. Only run git add
  DEV_LOG.md after the text is actually saved (an unsaved/empty entry makes the
  commit abort).
- Stage explicitly by filename: git add <file>. Never git add ., never git add
  -A, never the VS Code commit-all button.
- trading_explained.md and daily_prompt.md are intentionally untracked and must
  never be staged.
- Commit messages are plain imperative with no conventional-commit prefixes
  (e.g. "Add ...", "DEV_LOG: Day N - ...").
- I run all commits myself. You may have CC WRITE a file (e.g. the DEV_LOG
  entry), but never have CC run git; I verify with git diff and commit by hand.

Format:
- One numbered checklist, in order.
- Claude Code prompts in their own clearly labeled, copy-paste-ready code blocks.
- Plain ASCII only: no smart quotes, em dashes, or special bullet characters,
  since they break my editor on paste.
- Concise. Explanations are expected but no fluff. Everything immediately
  actionable; I should never have to guess.








Provide me with a good prompt to say after I paste the output for CLAUDE CODE PROMPT (Commit 1 of Day 36 - pure cross-sectional ranking function). Make sure it's all set, every line is checked, etc.




Ok, so day 37 is closed, right? If so, provide me the updated daily prompt for Day 38.