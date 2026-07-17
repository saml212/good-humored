# STATE

_Last updated: 2026-07-16_

## Current focus

Rejection-cascade benchmark, Phase 1: **rejector validation** (the
instrument-calibration experiment). Nothing downstream runs until the rejector
is shown to label topics (not jokes) consistently.

## Master plan (agreed 2026-07-16)

1. Benchmark frontier + open-weight models on the cascade → **ill-humored
   leaderboard** (exhaustion depth, path divergence, cross-model overlap).
   Neutral framing, brutal numbers — credibility is what makes it sellable.
   Roster: newest frontier first — Fable 5, Opus 4.8, Grok 4.5, GPT (latest),
   Gemini (latest), "Sol" (Sam's mention — identify provider), plus open
   weights: Llama, Qwen, DeepSeek. **GATE: Sam reads/approves docs/BENCHMARK.md
   before any model runs** (2026-07-16). Rejector validation is exempt
   (instrument calibration, not benchmarking).
2. Pick post-training target: strong base + weak cascade score (headroom) +
   trainable size (7–32B) — not necessarily the top scorer.
3. GRPO with the humor reward stack (`.claude/skills/humor-rl/`) vs. curated
   SFT baseline (HumorGen's bar). Show cascade score move. **Episode format is
   conversational (banter), not single-joke** — callback bonuses +
   context-ablation scoring (docs/BENCHMARK.md §1b). Sam: "humor is
   contextualized."
4. **Reverse transfer** (the pitch's spine): MMLU/GPQA before/after humor RL
   vs. compute-matched control. "Does gaining humor gain wisdom?" — untested
   by anyone.

## Novelty status (attack agent, 2026-07-16 — survives, crowded)

Mandatory related work: Denial Prompting/NEOCODER (2407.09007, technique-denial
for code, T≈5), MUTATE (2605.28465, objective failure memory in text-adventure,
closest), NoveltyBench §4.3 (2504.05228, 8-turn regeneerate, set metric not
path). Differentiator we own: **subjective content-agnostic rejection on an
open creative task, ~50 turns, topic sequence as the measured object, compared
across runs AND across models.** Cross-model comparison exists nowhere.

## Commercial status (landscape scan, 2026-07-16)

Topic-trajectory entropy collapse axis: unoccupied. Only humor vendor: Good
Start Labs (LOL Arena — preference data, complementary not competing). Clearest
buyer: Anthropic (>$1B/yr on RL environments reportedly). Biggest threats:
Good Start Labs moving fast; Meta DARLING as clone-the-method risk. Details:
`docs/private/commercial-landscape.md` (gitignored).

## Running / in flight

- **rejector-validation-v1** — Haiku labeler vs keyword baseline, 32-item
  hand-built fixture, 3 repeats. Audit agent reviewing code before run.
- Harness maintenance PRs upstream (rockie-claude #33 + backport/loop agents)
  — separate workstream, do not block research.

## Dead ends

_(none yet — negative results from the literature are in
`references/negative-results.md`)_

## Key constraints

- No GPU_API_KEY in .env yet → cascade pilot runs via `claude` CLI providers;
  paper-grade runs need real APIs (temperature control, true multi-turn).
- Corpus licensing: SocialGrep (CC-BY) safe; taivop + Oogiri research-only —
  keep commercial/reference corpora separated (`references/corpus-sources.md`).
- Repo is public, all rights reserved; `docs/private/` and `CLAUDE.md` are
  gitignored on purpose.
