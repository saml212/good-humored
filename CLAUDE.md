# CLAUDE.md — good-humored

> RL environments for humor. Specialized from rockie-claude's
> `claude-md/ml-research.md` template.

## Workflow: Plan → Research → Build → Audit → Run → Assess → Codify

Every cycle should make the next cycle better.

- **Plan:** Talk with the user. Understand the goal before touching code.
- **Research:** Send agents to verify claims and check novelty. Never assert facts without evidence.
- **Build:** Write code. Keep it clean. Comment the non-obvious.
- **Audit:** Send a separate agent to review code before running. Check shapes, gradients, stability. The implementer does not review their own work.
- **Run:** Use the hardware. Parallel experiments when possible.
- **Assess:** Be honest. Negative results are data. Don't spin.
- **Codify:** Update `STATE.md` and `EXPERIMENT_LOG.md`. If you learned a lesson, emit a `[LEARN]` block so it auto-saves to the learnings DB.

## Learnings DB

A SQLite DB at `.claude/memory/workflow.db` persists durable rules, corrections, and gotchas across sessions. Relevant rules auto-inject at prompt time via the `load-relevant-rules.sh` hook.

**When you learn something worth persisting, emit a `[LEARN]` block in your response:**

```
[LEARN] <category>: <one-line rule>
Mistake: <what went wrong>
Correction: <what the right approach is>
```

When you definitively kill a research direction, emit a `[DEAD-END]` block:

```
[DEAD-END] <direction>: <reason>
Evidence: <path to run dir or paper>
```

The brainstorm agent auto-loads relevant dead-ends on future "what should we try" prompts.

## Pre-Experiment Checklist (MANDATORY before every experiment)

1. **State the hypothesis in one sentence.** If you can't, don't run it.
2. **Predict the metric delta.** Record `predicted_delta` with `calibration.py add` — forces you to have a quantitative prior, and lets the harness track your calibration over time.
3. **Compute FLOPs, memory, and param count on paper.** 10 minutes. No exceptions.
4. **Try to disprove it in 5 minutes.** "Could a simpler baseline do this?"
5. **Check the literature first.** Send a research agent BEFORE building.
6. **Design the comparison before the experiment.** What's the baseline? Are params matched?
7. **Define success criteria.** What metric improvement justifies the compute cost?
8. **Verify the claim is novel.** Don't claim uniqueness without checking.

After the run, `calibration.py close <run_id> <hypothesis> <actual_delta>` closes the loop. Periodically `calibration.py report` to see if your priors are improving.

## Waterfall for new ideas

1. **Brainstorm agent** — Generate ideas.
2. **Research agent** — Validate against literature.
3. **Attack agent** — Find fatal flaws.
4. **Validation agent** — Confirm or deny each attack with evidence.

Only build what survives all four stages.

## Hard Rules

- Verify before claiming. Use web search or research agents.
- Audit code with a separate agent before running.
- Smoke test every model (forward, backward, gradient check) before training.
- Use standard benchmarks for publishable claims.
- Save the exact script alongside experiment results for reproducibility.
- Log everything to a file. Produce a human-readable summary at the end.
- Smoke test must include EVAL batch size, not just training — eval can OOM even if training fits.
- Use the same dataset for ALL experiments in a comparison.
- HF cache defaults to container disk. Symlink `HF_HOME` to persistent volume immediately.
- The param-matched baseline ablation blocks ALL downstream decisions. Run it first.
- Humor-specific: any generation eval MUST include a novelty check against a memorized-joke corpus — mode collapse onto memorized jokes is the documented failure mode (>90% of ChatGPT jokes were 25 templates).
- Humor-specific: LLM-judge-only rewards get hacked fast (documented: GRPO + GPT-4.1 judge → classic-joke regurgitation). Rewards need decomposition: novelty + comprehensibility + human preference.

## Repo Layout

- `STATE.md` — Current project state, what's running, dead ends
- `EXPERIMENT_LOG.md` — Every experiment and result
- `references.md` — Paper references library
- `experiment-runs/` — Archived exact scripts from each experiment
- `archive/` — Dead ends and superseded docs

## Hardware

- Local dev box (Apple Silicon) — quick iteration, no training
- Rockie platform (`rockie experiment submit`) — managed GPU compute, tenant `t-24174c0a69ce`
- Spot-first GPU policy via harness `gpu.py` router if BYO providers are configured in `.env`

## Data

Code lives in this repo. Data and checkpoints live elsewhere (gitignored; pointer file at repo root documents locations).

Candidate reward/eval datasets identified in the literature review:
- **Oogiri-Corpus / Oogiri-Master** (Dec 2025) — ~100 candidates per prompt, each rated by ~100 independent judges; cleanest humor RM data, popularity-bias-free.
- **New Yorker Caption Contest** — ~250M human ratings on 2.2M captions; known negative result: RLHF/DPO underperform top humans here.
- **LOL Arena** (Good Start Labs) — live humor preference pipeline from Bad Cards players.

## Research Direction

RL environments for humor generation. Three-pass verified literature review (July 2026) found three open gaps, in priority order: **(1) Reverse transfer** — HumorBench showed STEM-reasoning training transfers to humor *comprehension*; nobody has tested whether training on humor transfers back to general reasoning/taste (clean novel paper). **(2) Multi-turn conversational humor environments** — banter, callbacks, timing; nothing exists, and RLVR demonstrably *damages* multi-turn conversational skill, which is a built-in publication hook. **(3) Diversity-preserving RL against live human humor preferences** — naive versions tried and failed publicly (HumorGen: DPO/O-GRPO no better than curated SFT; NYCC paper: RLHF/DPO limitations on creative tasks), so the opportunity is the fix: online human feedback + anti-mode-collapse machinery (novelty penalties vs. a joke corpus, diversity-aware GRPO variants ported from math/image domains).

## User Context

Sam Larson (samlarson@pebbleml.com) — researcher; builds/runs Rockie (rockielab.com). Working thesis: humor ≈ familiar-but-expectation-breaking, correlates with intelligence/wisdom (r = .29–.40 in the psych literature), and may be a generalizable RL training signal the way code/math are. Prefers cheap subagents (Haiku/Sonnet) for bulk research fan-out. Wants verifiable rewards; skeptical of engagement metrics (likes ≠ funny).
