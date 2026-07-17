# References

Reference corpus for the good-humored humor-RL project, compiled July 2026
across five parallel research passes (arXiv, ACL Anthology, Semantic
Scholar, OpenReview, HuggingFace, GitHub, PapersWithCode, LessWrong/
AlignmentForum). Every claim below was checked against a primary source
where one exists; anything that couldn't be confirmed is marked
`[UNVERIFIED]` in the relevant file rather than dropped or inflated.

**Files in this corpus:**
- `papers.bib` — BibTeX for everything confirmed to exist, ~90 entries.
- `negative-results.md` — **read this first.** Every documented failure of
  naive humor generation and RL-on-creative-tasks, with exact numbers.
- `datasets.md` — what data exists, size, license, access, known bias.
- `code-and-models.md` — repos and weights, audited for whether they
  actually run.
- `psychology.md` — the humor/intelligence and humor-theory literature.
- `trajectory-grounding.md` — semantic foraging / verbal-fluency metrics
  (Troyer clustering/switching, Hills-Jones-Todd patch models), humor-topic
  sociology, and what each humor theory predicts about topic exhaustion.
  The scientific basis of the cascade benchmark's metrics.
- `related-work-cascade.md` — adversarial novelty audit of the rejection
  cascade: Denial Prompting/NEOCODER, MUTATE, NoveltyBench and the exact
  differentiation the cascade can honestly claim. Read before writing any
  novelty sentence.
- `corpus-sources.md` — downloadable joke corpora (~3.1M jokes) with
  licenses; feeds the memorized-joke novelty penalty.
- `pdfs/` — local-only, gitignored. Never committed; see "PDF policy" below.

---

## Start here — the ~10 highest-value entries

1. **HumorGen** (`ajayi2026humorgen`, arXiv:2604.09629) — the single
   cleanest negative result in the corpus: neither DPO nor offline-GRPO
   beats a well-curated SFT baseline for humor generation, with exact,
   verified numbers. Also introduces the Humor Transfer Bench and a
   persona-distillation SFT recipe worth reading even though no code
   shipped. Any project plan that assumes "add preference optimization"
   will help needs to argue why it beats this.

2. **The ChatGPT joke probe** (`jentzsch2023chatgpt`, WASSA 2023,
   arXiv:2306.04563) — the load-bearing mode-collapse citation, verified
   exactly (>90% of 1,008 jokes were the same 25 templates). Comes with
   working code (`github.com/DLR-SC/JokeGPT-WASSA23`). This is the number
   every downstream mode-collapse claim should trace back to.

3. **The LessWrong GRPO + GPT-4.1-judge post** (`agg2025funniestjoke`) —
   a real, documented, in-the-open reward-hacking experiment showing an
   LLM-judge reward gets gamed in *two different directions* depending on
   the rubric. Directly validates the `humor-rl` skill's insistence on
   decomposed, non-judge-only rewards.

4. **New Yorker Caption Contest — "Humor in AI"** (`zhang2024humorinai`,
   NeurIPS 2024 D&B, arXiv:2406.10522) — the largest usable
   humor-preference dataset found (250M+ ratings, 2.2M+ captions), with a
   documented RLHF/DPO-limitations finding and a counterintuitive nuance
   (DPO *increased* diversity here, contra the usual expectation). Dataset
   is downloadable now.

5. **HumorBench** (`narad2025humorbench`, arXiv:2507.21476) — the one
   confirmed transfer direction in the whole literature: STEM-reasoning
   training transfers into humor *comprehension*. This is exactly what
   makes gap #1 (reverse transfer: humor → reasoning) a clean, untested,
   novel question rather than a guess.

6. **DARLING** (`li2025darling`, Meta, arXiv:2509.02534) — the most
   directly reusable diversity-preserving RL codebase found, explicitly
   validated on creative writing, built on verl, with a full training
   pipeline (not just scripts to derive from).

7. **RAGEN / StarPO** (`wang2025ragen`, arXiv:2504.20073) — the most
   mature multi-turn trajectory-level RL codebase found, actively
   maintained (commits as recent as today), with a named failure mode
   ("Echo Trap") directly relevant to gap #2.

8. **Greengross & Miller (2011) + Christensen et al. (2018)**
   (`greengross2011humor`, `christensen2018clever`) — the primary-source,
   full-text-confirmed empirical basis for the project's r≈.29–.40
   intelligence-humor thesis. Read these before citing the range anywhere
   else — the exact numbers (.38, .49, .25, .22) are more precise and more
   defensible than the rounded range.

9. **McGraw & Warren (Benign Violation Theory) + Suls (incongruity-
   resolution)** (`mcgraw2010benign`, `suls1972twostage`) — the theoretical
   backbone for reward architecture. Both are dual-appraisal,
   simultaneity-gated models arrived at independently — strong convergent
   suggestion that a humor reward should be a *product* of two signals
   (violation × benign-ness, or incongruity × resolution), not a single
   scalar.

10. **Oogiri-Master/Oogiri-Corpus** (`murakami2025oogirimaster`,
    arXiv:2512.21494) **+ CLoT-Oogiri-GO dataset** (`zhong2024clot`) —
    together, the popularity-bias-free rating *methodology* (Oogiri-Master)
    and the one immediately-downloadable large humor-preference dataset
    (Oogiri-GO, usable today on HuggingFace, ~130K samples). Read both:
    they are not the same rating methodology despite the shared name — see
    `datasets.md`.

**Honorable mentions that almost made the cut:** the Google DeepMind
comedian workshop study (`mirowski2024comedians`) — 20 professional
comedians calling LLM output "the most bland, boring thing" and attributing
it to safety-tuning, direct qualitative evidence for a real risk in this
project's approach; and "Engagement Undermines Safety"
(arXiv:2510.18454) — stereotypical/toxic jokes scoring 10–21% higher on
humor metrics, the single most concrete reward-hacking mechanism found for
judge design specifically.

---

## Organized by theme

### Humor generation with LLMs
HumorBench, HumorGen, HumorRank, CLoST, CLoT, IRS, SemEval-2026 MWAHAHA,
the Dad Jokes generalization paper (detection, not generation — see
correction in `negative-results.md`), Oogiri-Master, Oogiri-GO, LoTbench,
the DeepMind comedian workshop study, the three New Yorker Caption Contest
papers, and the CHum workshop corpus (a small but standout ACL-affiliated
venue producing exactly this project's target research, largely invisible
to RL-keyword search since most CHum papers don't mention RL at all — see
`papers.bib` §1–2 and full paper list in the research notes). Full details
and the three-gap synthesis (reverse transfer, multi-turn banter, diversity-
preserving RL) in `negative-results.md` and this file's "gaps" section
below.

### Negative results
`negative-results.md` — read in full. Covers the ChatGPT joke probe, the
NYCC RLHF/DPO limitations (two distinct papers, disambiguated), HumorGen's
SFT-beats-DPO/GRPO finding (with a caught-and-corrected fabricated number),
the LessWrong reward-hacking experiment, RLVR's documented damage to
multi-turn conversation, the corrected κ=0.49→0.41 annotation-agreement
citation, and general RL diversity-collapse mechanics from adjacent domains
(math reasoning, image generation, music, story generation) that transfer
even though they aren't humor-specific.

### Datasets
`datasets.md` — tiered by actual usability. Tier 1 (usable now, large):
NYCC ranking data, NYCC matching/ranking/explanation splits, Oogiri-GO.
Tier 2 (real, bias-controlled, not yet freely downloadable): Oogiri-Master/
Corpus (builder-code-only), LoTbench (code/data "coming soon"). Tier 3
(usable now, smaller): ColBERT, Weller & Seppi Reddit data, StandUp4AI
(real audience-laughter labels), HaHackathon, HAHA, Chumor. Tier 4 (raw
material, real popularity-bias confound): Short Jokes, One Million Reddit
Jokes, and older scraped corpora.

### Code and models
`code-and-models.md` — ready-to-build-on-Monday: DARLING, RAGEN/StarPO,
verl, verl-agent/GiGPO, Agent Lightning, SynthesizeMe, Personalized
RewardBench, PAD, OpenRLHF. Usable-with-friction: DPH-RL, DRA-GRPO, GEM,
LMRL Gym (JAX-locked, stale). Thin/unrunnable: PersRM-R1, the Oogiri
builder repo, the Dad Jokes transfer repo. Paper-only, no code found: DQO,
Info-GRPO, EDGE-GRPO, LoTbench, and — notably — **every core
humor-generation-with-LLMs paper in this corpus** (HumorBench, HumorGen,
HumorRank, CLoST, IRS). This last point is the biggest gap; see below.

### Psychology and humor theory
`psychology.md` — the intelligence-correlation literature (Greengross &
Miller, Christensen et al., Feingold & Mazzella, Howrigan & MacDonald,
Kellner & Benedek, the Arslan/Sak/Ateşgöz children's study, the Greengross
twin-heritability study), the classical theories (Benign Violation,
incongruity-resolution, superiority, relief, play-mirth), the Humor Styles
Questionnaire, the Greengross/Martin/Miller stand-up-comedians study, and
the audience-dependence / rater-disagreement psychometrics literature
(Uma et al., Davani et al., Zhang et al., Cao et al.) that bears directly
on reward-model design.

### Adjacent RL machinery
Diversity-aware GRPO variants (GEM, DPH-RL, DRA-GRPO, DARLING, DQO,
Info-GRPO, EDGE-GRPO), RLVR diversity-collapse characterization papers,
multi-turn RL infrastructure (LMRL Gym, RAGEN/StarPO, verl-agent/GiGPO,
Agent Lightning), and personalized reward modeling (PAD, PersRM-R1,
SynthesizeMe, Personalized RewardBench). Full usability audit in
`code-and-models.md`.

---

## The three open gaps, as this research left them

1. **Reverse transfer (humor → general reasoning/taste).** Confirmed
   untested. Every transfer result found in this corpus runs the other
   direction (HumorBench: STEM → humor comprehension) or stays within humor
   (Humor Transfer Bench: across input-prompt domains; the Dad Jokes paper:
   across humor-detection datasets). No paper trains on humor and measures
   general-capability gains. Still a clean, novel research question.

2. **Multi-turn conversational humor (banter, callbacks, timing).**
   Confirmed open. The closest thing found — "Multi-Agent Comedy Club"
   (`hong2026comedyclub`) — conditions a single monologue generator on
   stored community/critic feedback between rounds; it is not live
   agent-to-agent banter. No dedicated benchmark or dataset targeting
   comedic callback/timing/banter was found despite specific searches
   (MT-Bench-101-style, EYT-Bench-style queries). RLVR's documented damage
   to multi-turn conversational skill generally (`negative-results.md` §5)
   is a real risk to design against, and RAGEN/StarPO and verl-agent/GiGPO
   are the best available infrastructure for the trajectory-level
   optimization this will need.

3. **Diversity-preserving RL against live human humor preferences.** The
   naive versions have been run and published as negative results
   (HumorGen: DPO/O-GRPO no better than SFT; NYCC: RLHF/DPO underperform top
   humans). What hasn't been tried: online human feedback combined with a
   diversity-preservation mechanism actually validated on creative text
   (DARLING is the strongest candidate mechanism; nothing combines it with
   a live human-preference loop for humor specifically yet). Also
   unaddressed: the identity-of-speaker confound in humor judgment
   (`kim2026counterfactual` — jokes attributed to "privileged" speakers
   judged more harshly for identical content) that any live-feedback
   pipeline will need to correct for, not just the mode-collapse mechanics.

---

## PDF policy

This repo is public with no license (all rights reserved) — third-party
paper PDFs and datasets are not ours to redistribute. If a paper PDF was
downloaded for reading during this research, it lives under `references/pdfs/`,
which is listed in `.gitignore` and must never be committed. Citations,
links, and summaries are recorded in this corpus instead; go to the original
source (arXiv/publisher/DOI link in `papers.bib`) to read the paper itself.

## What to distrust and re-verify yourself

- Any number attributed only to "search synthesis" in `negative-results.md`
  or `code-and-models.md` (flagged inline) — these came from a search
  engine's own generated answer text, which was independently caught
  fabricating a full results table during this research (see
  `negative-results.md` §3) before being caught by a raw-text re-fetch.
  Trust the primary-source-confirmed numbers; re-verify the rest before
  using them in anything that matters.
- OpenReview review threads for humor-adjacent NeurIPS/ICLR/ACL submissions
  were largely inaccessible during this research pass (Cloudflare-gated).
  If review-buried negative results are valuable to this project, a manual
  pass from a normal browser session (not an automated fetch) is worth
  doing — this corpus did not manage it.
- Several `[UNVERIFIED]` entries throughout are real papers whose *existence*
  is confirmed but whose specific claimed numbers weren't independently
  re-derived from the primary text (see especially the "Advantage Collapse
  in GRPO" percentage and the DivPO diversity percentages in
  `negative-results.md`).
