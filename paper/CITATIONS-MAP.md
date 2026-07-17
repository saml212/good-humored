# Citations Map — audit trail for `paper/DRAFT.md`

Every non-trivial claim in the draft, its citation, and the exact
`references/` (or repo-internal) file that verifies it. Rule followed:
**no claim in the draft that isn't traceable to a line in this table.**
Where the underlying reference file itself flags a number as
`UNVERIFIED`/`SUGGESTIVE`/`SPECULATIVE`, that flag is carried into this
table's Notes column — none of those flagged items were promoted to
firmer language in the draft.

Bibkeys refer to `references/papers.bib` where an entry exists. Several
citations here (marked *no bib entry*) exist only in the narrative
reference files, compiled in a research pass after `papers.bib` was
written — see the "Bibliographic hygiene" note at the end.

---

## Section 2 — Introduction

| Claim | Citation | Verifying file | Notes |
|---|---|---|---|
| 90.2% (909/1,008) of ChatGPT jokes were 1 of 25 recurring templates | Jentzsch & Kersting, 2023, WASSA (`jentzsch2023chatgpt`) | `references/negative-results.md` §1 | Primary-text confirmed, code public |
| A lookup table sampled at temperature can mimic a distribution's shape (motivating claim, not a cited finding) | — (argument, not a citation) | `docs/BENCHMARK.md` "Why this is different from everything published" | Stated as our own argument, correctly uncited |
| HumorBench: STEM-reasoning training transfers to humor comprehension | Narad et al., 2025, arXiv:2507.21476 (`narad2025humorbench`) | `references/humor-honesty-beauty.md` §3.3(1); `references/README.md` | ESTABLISHED per source's own tag |
| ToM-HCAT / ToMBench as evidence humor requires theory of mind | ToM-HCAT, *Frontiers in Psychology* 2018 (PMC6099116); Chen et al. 2024, ACL, arXiv:2402.15052 | `references/humor-honesty-beauty.md` §3.3(2) | ESTABLISHED (human side); LLM ToM measurement flagged as "active, real research area," not solved — draft does not claim it's solved |
| Benign Violation Theory: humor requires norm-violation + benign appraisal | McGraw & Warren, 2010, *Psychological Science* 21(8):1141-1149 (`mcgraw2010benign`) | `references/humor-honesty-beauty.md` §3.3(3); `references/trajectory-grounding.md` §3.4 | ESTABLISHED |
| The three links are independently established; causal transfer (training on humor → general capability) is untested | — (synthesis) | `references/humor-honesty-beauty.md` §3.3 closing paragraph + "What the README can honestly say" | Source file explicitly states this is *not* established — draft matches that hedge exactly, does not overclaim |
| Humor correlates with intelligence, r ≈ .29–.40 (Future Work, §7) | Greengross & Miller, 2011, *Intelligence* 39(4):188-192 (`greengross2011humor`); Christensen et al., 2018, *Psych. Aesthetics, Creativity & Arts* 12(2):136-143 (`christensen2018clever`) | `references/psychology.md`; `references/humor-honesty-beauty.md` §1.2 | ESTABLISHED but explicitly noted (in source) as largely one research program (Greengross/Miller/Martin cluster); draft does not claim independent replication |

**Deliberately excluded:** the humor↔honesty (HEXACO Honesty-Humility) and
humor↔mathematical-beauty (Zeki mOFC) material in
`references/humor-honesty-beauty.md` §1 and §2 is tagged SUGGESTIVE/
SPECULATIVE by its own source file (no study has tested humor and
math-beauty appreciation in the same paradigm; "funny = honest" is not a
tested finding). The task scoped the alignment framing to world-models +
ToM + norm-awareness specifically, and that scoping also keeps the
Introduction inside the ESTABLISHED-tagged material. Not used anywhere in
the draft.

---

## Section 3.1 — Humor generation and its documented failures

| Claim | Citation | Verifying file | Notes |
|---|---|---|---|
| 909/1,008 (90.2%) jokes were 1 of 25 templates; top 4 templates > 50% of output | Jentzsch & Kersting, 2023 (`jentzsch2023chatgpt`) | `references/negative-results.md` §1 | Same citation as above; repeated because it anchors both the motivation and the related-work section |
| GRPO + GPT-4.1-judge reward hacking, two distinct collapses, rubric-hardening relocated rather than fixed the hack | LessWrong post, 2025-05-16 (`agg2025funniestjoke`) | `references/negative-results.md` §1 | Explicitly labeled in source as "a real experiment, not a paper" — draft carries the same caveat ("documented, non-peer-reviewed write-up") |
| HumorGen: neither DPO nor offline-GRPO beats a well-curated SFT baseline; "a data quality ceiling" | Ajayi & Mitra, 2026, arXiv:2604.09629 (`ajayi2026humorgen`) | `references/negative-results.md` §3 | Numbers cross-checked against raw arXiv HTML after an earlier fetch fabricated a different table (documented in source) — draft uses only the re-verified numbers |
| NYCC: 250M+ ratings / 2.2M+ captions; SFT regressed below zero-shot; frontier models trail top humans | Zhang et al., 2024, NeurIPS D&B, arXiv:2406.10522 (`zhang2024humorinai`) | `references/negative-results.md` §2a | Verbatim abstract quote confirmed in source |
| Persona-prompting for audience-adapted humor has "minimal impact"; small-scale SFT on preferences closes the gap | Zhou et al., 2025, EMNLP Findings, arXiv:2502.20356 (`zhou2025bridging`) | `references/negative-results.md` §2b | Exact quote confirmed in source |

---

## Section 3.2 — Diversity and novelty evaluation

| Claim | Citation | Verifying file | Notes |
|---|---|---|---|
| Denial Prompting / NEOCODER protocol (technique-denial, T≈5, code domain, correctness signal) | Lu et al., 2024/2025, NAACL, arXiv:2407.09007 | `references/related-work-cascade.md` §1 | *No bib entry* — verified in this file only (HTTP 200 + HTML full-text read), post-dates `papers.bib` compilation |
| NoveltyBench §4.3: in-context regeneration, 8 turns, no denial framing, Distinct-k set metric | Zhang et al., 2025, arXiv:2504.05228 | `references/related-work-cascade.md` §2 | *No bib entry*, same as above |
| MUTATE: per-object failure memory, objective task failure, Path Discovery / Divergence Momentum metrics, no cross-model comparison | Park et al., 2026, arXiv:2605.28465 | `references/related-work-cascade.md` §3 | *No bib entry*, same as above; source calls this "the single closest paper found" |
| Differentiation sentence (subjective/content-agnostic vs. objective/technique rejection; global vs. per-object memory; depth; cross-model+across-run path identity as the novel measured object) | — (our own synthesis, attributed) | `references/related-work-cascade.md`, "Sharpest honest differentiation sentence" (final section) | Quoted near-verbatim from the source file, as instructed |

---

## Section 3.3 — Cognitive-science lineage

| Claim | Citation | Verifying file | Notes |
|---|---|---|---|
| Clustering (run-length) / switching (transition count) as dissociable fluency-scoring components; dissociates clinical populations independent of raw item count | Troyer, Moscovitch & Winocur, 1997, *Neuropsychology* 11(1):138-146 | `references/trajectory-grounding.md` §1.2 | Confirmed via PubMed (PMID 9055277) + Google Scholar in source; Troyer 2000 normative-data follow-up is flagged UNVERIFIED-SECONDARY in source and is **not** cited in the draft |
| Optimal foraging in semantic memory; Marginal Value Theorem patch-departure timing predicts recall performance | Hills, Jones & Todd, 2012, *Psychological Review* 119(2):431-440 | `references/trajectory-grounding.md` §1.1, §1.3 | Confirmed via 4+ independent hosts in source |
| Category fluency ported to an LLM (Llama 2 7B), path-dependent/n-gram-overlap scoring vs. human sequences; unforced, single-model, single-category | Heineman, Koenen & Varma, 2024, arXiv:2405.06714 | `references/trajectory-grounding.md` §1.4; `references/related-work-cascade.md` §7 | *No bib entry*; HTTP 200 verified in both source files |

**Deliberately excluded:** the "random walk vs. optimal foraging" debate is
cited in the draft's foraging discussion only as a general framing (patch
structure could be an artifact of embedding-space geometry, not a
computed policy) and is **not** attributed to a single paper — the source
file itself flags a single-paper version of this claim as
UNVERIFIED-SECONDARY and recommends citing it as an open debate rather
than a specific citation. Draft follows that guidance. Troyer (2000),
Fine & DeSoucey, and Dundes's "Dead Baby Joke Cycle" (all
UNVERIFIED-SECONDARY in `references/trajectory-grounding.md` §1.2/§2.2/
§2.3) are likewise not cited in the draft.

---

## Section 3.4 — Cross-model homogeneity

| Claim | Citation | Verifying file | Notes |
|---|---|---|---|
| LLM responses more similar to each other than human responses are to each other, across standardized creativity tests | Wenger & Kenett, 2025, arXiv:2501.19361 | `references/related-work-cascade.md` §4 | *No bib entry* |
| Five frontier LLMs selecting Cards Against Humanity winners agree with each other far more than with humans (9,894 rounds) | Fettach, Bied, Toivonen & De Bie, 2026, arXiv:2604.08757 (`fettach2026cardsagainstllms`) | `references/related-work-cascade.md` §5 | Author given names explicitly flagged "not confirmed" in `papers.bib` note — draft cites surnames only, matching that caveat |

---

## Section 4.1 — Cascade protocol

| Claim | Citation | Verifying file | Notes |
|---|---|---|---|
| Rejector labels the topic explicitly; rejections accumulate (no sliding window); ~turns/N runs/M models design | Design spec (repo-internal, not external literature) | `docs/BENCHMARK.md` §1 "Design decisions (made)" | Internal design decision, correctly cited to the spec, not to a paper |

---

## Section 4.2 — Metric families and psychological lineage

| Claim | Citation | Verifying file | Notes |
|---|---|---|---|
| Within-model divergence: identical human fluency runs would be pathological, not healthy | Troyer et al., 1997 (interpretation) | `references/trajectory-grounding.md` §1.2, §6.2 | Draft's inference from the source's clustering/switching framing, not a verbatim claim from Troyer et al. |
| Cross-model overlap as the most diagnostic, least-precedented metric | — (our claim) | `docs/BENCHMARK.md` §1 "The three metrics" table; `references/related-work-cascade.md` overall verdict | Matches source's own framing ("Cross-model overlap is the most interesting of the three") |
| Depth-to-degradation as an LLM analog of MVT patch-departure/small-well behavior | Hills, Jones & Todd, 2012 | `references/trajectory-grounding.md` §1.1, §6.3 | Interpretive port, flagged as such ("This is our port of...") |

---

## Section 4.3 — Instrument validation arc

All claims in this subsection are internal experimental results, not
literature citations. Verifying source for every number: `EXPERIMENT_LOG.md`,
cross-checked against the underlying data.

| Claim | Verifying file(s) |
|---|---|
| EXP-001 fixture design, pre-registered bars, keyword-baseline disproof attempt, result (ARI 0.620/0.271, invariance 0.600), diagnosed failure modes, zero punchline-mechanism contamination | `EXPERIMENT_LOG.md` EXP-001; `experiment-runs/2026-07-16-rejector-validation/report.json` |
| EXP-002 LABEL_PROMPT v2, ARI 0.837, invariance 0.800, residual `cat`/`pet` and `health`/`medicine` misses | `EXPERIMENT_LOG.md` EXP-002; `experiment-runs/2026-07-16-rejector-validation-v2/report.json` |
| EXP-003 semantic re-scoring: union-find invariance 0.900/ARI crash to 0.659 (hypernym hub-bridging); complete-linkage ARI 0.697/invariance 0.800 | `EXPERIMENT_LOG.md` EXP-003 |
| EXP-003b Sonnet-as-rejector negative result (ARI 0.633, invariance 0.700 vs. Haiku's 0.837/0.800) | `EXPERIMENT_LOG.md` EXP-003b; `experiment-runs/2026-07-17-rejector-validation-v3-sonnet/report.json` |
| Instrument decision: Haiku + v2 + raw string scoring is pilot-grade instrument; conservative bias-direction argument | `EXPERIMENT_LOG.md` "Instrument decision (2026-07-17, pilot grade)" |

---

## Section 4.4 — Memorized-joke novelty check

| Claim | Citation | Verifying file | Notes |
|---|---|---|---|
| 25 templates / 90.2% figure as minimum viable memorized-joke corpus | Jentzsch & Kersting, 2023 (`jentzsch2023chatgpt`) | `docs/BENCHMARK.md` §3; `references/negative-results.md` §1 | Same citation, reused per its stated role as "the minimum viable memorized-joke corpus" |
| Novelty-penalty mechanism is inert without a real scraped corpus; target ~3.1M jokes | — (design status, not a literature claim) | `docs/BENCHMARK.md` §3; `references/corpus-sources.md` (per `references/README.md` description) | Stated honestly in draft as "a design requirement, not yet a completed feature" |

---

## Section 5 — Experiments

| Claim | Citation | Verifying file | Notes |
|---|---|---|---|
| 12 models, 4 access lanes (claude/codex/api/grok), depth 30, N=4 | — (internal experiment record) | `EXPERIMENT_LOG.md` EXP-004; **verified directly against** `experiment-runs/2026-07-17-cascade-pilot/lane-{claude,codex,api,grok}/*.jsonl` | See "Discrepancy found" below — the 12-model, 4-lane count was confirmed from the run directories, not taken as given from EXP-004's prose, which undercounts |
| Pre-launch audit fixes: canon-vs-raw primary-metric bug (BLOCKER-1), key-fragment scrubbing, UNPARSEABLE sentinel | — (internal audit record) | `EXPERIMENT_LOG.md` EXP-004 | Internal QA, not external literature |
| Predicted deltas: cross-model Jaccard ≈0.35, within-model Jaccard ≈0.55, ≥1/3 of roster degrading by turn 30 | — (pre-registered prediction) | `EXPERIMENT_LOG.md` EXP-004 "Predicted deltas" | The original text says "≥4/10 models" — draft restates as "≥1/3," a judgment-call normalization now that the roster is 12, not 10; flagged explicitly in the report to the user, not silently substituted |
| Wrapper confound: CLI lanes lack temperature control, encode multi-turn as transcript-in-prompt; API lanes are native | — (internal validity note) | `EXPERIMENT_LOG.md` EXP-004 "Known validity limits"; `STATE.md` "Key constraints" | Both files independently state this; draft cites the phenomenon, not a specific number |
| Results pending; uneven per-lane run counts (e.g., grok lane has 1 of 4 runs) | — (run-directory state) | `experiment-runs/2026-07-17-cascade-pilot/lane-grok/` (one file present, 17 lines vs. 30 expected for a complete depth-30 run) | Verified by direct file listing/line count, not asserted from the log text |

---

## Section 6 — Discussion and Limitations

All five limitations trace to the same two internal sources:
`EXPERIMENT_LOG.md` (EXP-004 "Known validity limits" and the "Instrument
decision" block) and `STATE.md` ("Key constraints": no `GPU_API_KEY` yet,
so CLI providers are the only way to run several lanes). No external
literature citation applies to this section — it is an honesty section
about our own protocol, by design.

---

## Section 7 — Future Work (reverse transfer)

| Claim | Citation | Verifying file | Notes |
|---|---|---|---|
| Reverse transfer (train on humor, measure general reasoning) is untested by anyone; HumorBench confirms only the forward direction | Narad et al., 2025 (`narad2025humorbench`) | `docs/BENCHMARK.md` §6 gap 1; `references/README.md` "three open gaps" §1; `CLAUDE.md` "Research Direction" | Repeated framing across three repo docs, all consistent |
| Humor production ability correlates with intelligence, r ≈ .29–.40 | Greengross & Miller, 2011; Christensen et al., 2018 | `references/psychology.md`; `references/humor-honesty-beauty.md` §1.2 | Same citation as Introduction |

---

## Claims found in repo docs that lack citation support (reported, not papered over)

1. **`CLAUDE.md` "User Context": "correlates with intelligence/wisdom (r = .29–.40 in the psych literature)."**
   The r ≈ .29–.40 figure is real and well-cited (Greengross & Miller 2011;
   Christensen et al. 2018) — for **intelligence**. No file in
   `references/` reports any correlation, at that magnitude or any other,
   between humor and **wisdom**. The draft does not use the word "wisdom"
   anywhere; this is flagged here as a claim in the project's own
   `CLAUDE.md` that outruns its cited evidence, not as something
   reproduced in the paper.

2. **`STATE.md` "Commercial status": "Anthropic (>$1B/yr on RL environments reportedly)."**
   Self-flagged with "reportedly" in the source, and not traceable to any
   file in `references/`. Not used in the draft (out of scope for a D&B
   paper), flagged here only because the task asked for an unpapered-over
   list.

3. **`README.md` roadmap step 3 vs. actual `EXP-004`/pilot design.**
   Not a citation gap but a factual inconsistency worth surfacing: the
   README's roadmap describes the cascade pilot as "2 models, depth 15,
   N=3 runs," but the pilot that was actually pre-registered and launched
   (`EXPERIMENT_LOG.md` EXP-004, `experiment-runs/2026-07-17-cascade-pilot/`)
   is 12 models, depth 30, N=4 — substantially larger on every axis. The
   draft uses the real, verified pilot design (12/30/4); README's roadmap
   text appears stale and should be updated to match, independent of this
   paper.

4. **`EXPERIMENT_LOG.md` EXP-004 internal model-count inconsistency.**
   The hypothesis line says "frontier and open-weight models" generically;
   the pre-registration line says "≥ 4/10 models hit degradation"; the
   itemized setup lists "Models (11)"; the actual run directories
   (`experiment-runs/2026-07-17-cascade-pilot/lane-*/`) show 12, because a
   `grok` lane was added after the log entry was written (mirroring how
   `kimi` was itself a late addition within that same entry). The draft
   uses 12 (directly verified from the filesystem) and flags the
   discrepancy here rather than silently picking one of the log's own
   internally inconsistent numbers.

5. **`references/trajectory-grounding.md`'s UNVERIFIED-SECONDARY items** —
   Troyer (2000) normative data paper, the Fine & DeSoucey "Joking
   Cultures" exact venue/year, Dundes's "Dead Baby Joke Cycle" exact
   bibliographic details, and the single-paper version of the
   "optimal-foraging-vs-random-walk" debate — are all flagged by the
   source file itself as unconfirmed at the level of detail some framing
   would want. None of the five are cited in the draft. Listed here so a
   reviewer of this map can see they were considered and excluded, not
   overlooked.

6. **`references/humor-honesty-beauty.md`'s SUGGESTIVE/SPECULATIVE material**
   (HEXACO Honesty-Humility ↔ humor style; shared neural locus for
   mathematical beauty and humor appreciation) is real, verified-to-exist
   literature, but explicitly tagged by its own source file as not rising
   to the same evidentiary level as the world-models/ToM/norm-awareness
   chain. Excluded from the draft for the same reason given in the
   Section 2 table above.

---

## Bibliographic hygiene note

`references/papers.bib` was compiled in an earlier research pass than
`references/related-work-cascade.md` and `references/trajectory-grounding.md`
(same day, earlier timestamp). Several citations load-bearing for Sections
3.2–3.4 of the draft — Denial Prompting/NEOCODER (2407.09007), NoveltyBench
(2504.05228), MUTATE (2605.28465), Wenger & Kenett (2501.19361), and the
category-fluency-on-LLMs paper (2405.06714) — are verified (HTTP 200 +
content read) in the narrative `.md` files but have **no corresponding
entry in `papers.bib`**. This doesn't affect the draft's honesty (every
claim is still traced to a verifying file above), but `papers.bib` should
be extended with these five entries before this draft moves toward a
camera-ready bibliography — flagging it now so it doesn't get missed.
