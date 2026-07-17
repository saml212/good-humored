# Findings — cascade pilot (2026-07-17)

Status: pilot-grade (N=2-4 runs/model, depth 30). Not paper numbers. Every
number below traces to `experiment-runs/2026-07-17-cascade-pilot/` and the
Result/Verdict blocks of EXP-004 through EXP-008 in `EXPERIMENT_LOG.md`.
Statistical inference in this document was computed by
`benchmark/run_stats_inference.py` and written in full to
`experiment-runs/2026-07-17-cascade-pilot/stats_inference.json` — every
statistic quoted here has a corresponding JSON path in that file.

Lines marked `<!-- REFRESH-AFTER-FILLS -->` will change when the
currently-running fill lanes merge (fable +2 runs, grok +4, kimi +2 — see
§5).

---

## 1. Headline: the pre-registered hypothesis failed

EXP-004 pre-registered, before any data existed: frontier and open-weight
models share a substantially overlapping joke-topic pool, predicted
cross-model mean topic-set Jaccard **≈ 0.35**. Measured: **0.102**.

The within-model prediction failed the same direction, more severely: models
were predicted to repeat *themselves* across runs more than they'd match
each other (≈0.55 self-Jaccard). Measured within-model self-Jaccard:
**0.182**.

Both misses point the same way — models are more idiosyncratic, not more
uniform, than the pre-registration assumed. What emerged instead was not
"no signal" — it was **three orthogonal per-lab failure fingerprints**, each
strong, each different, none matching the shared-pool story:

| Prediction (EXP-004, pre-registered) | Measured | Test | Result |
|---|---|---|---|
| Cross-model Jaccard ≈ 0.35 | **0.102** <!-- REFRESH-AFTER-FILLS --> | `pooled_frequency_baseline` null, one-sided | Observed sits *below the entire simulated null distribution* (10,000 draws, range 0.111–0.161). p = 1.0 with add-one correction ⟺ all 10,000 synthetic draws were ≥ observed |
| Within-model Jaccard ≈ 0.55 | **0.182** <!-- REFRESH-AFTER-FILLS --> | bootstrap CI, model-level (n=10) | 95% CI [0.139, 0.226] |
| ≥1/3 of roster degrades by turn 30 | met, but the log's "8/10" count doesn't reproduce | incidence audit | **9/10** models show ≥1 degrading run (only `codex:mini` fully clean); see §4.4 |

The `pooled_frequency_baseline` result is worth sitting with: it is not
merely "not significantly above chance." The real cross-model overlap is
*lower than every one of 10,000 synthetic universes* in which each model
draws i.i.d. from one shared, frequency-weighted topic bag built from the
pilot's own vocabulary (270 distinct topics). Real models show less
overlap than a chance-cooccurrence baseline in a shared vocabulary
predicts. Per EXP-006 (§3), the pilot's overlap regime (0.10–0.22) is
one where labeler noise makes measured overlap an *upper bound* on true
overlap — so if anything, true cross-model disjointness is understated
here, not overstated. The null distribution's min/max and the 270-topic
vocabulary size above were computed off-pipeline for this draft;
`benchmark/run_stats_inference.py` now persists both directly
(`pooled_frequency_null_diagnostics`, in the cross-model section of
`stats_inference.json`), so they will be independently verifiable from
the JSON artifact rather than asserted in prose once the post-fills
re-run lands.

`label_shuffle` was also computed (p = 1.0) and is reported for
transparency only — it is the documented **wrong** null for a shared-pool
claim (near-zero power to detect a genuinely shared pool; see
`benchmark/stats.py`'s `cross_model_null` docstring and
`test_fully_shared_pool_label_shuffle_has_no_power` in
`benchmark/tests/test_stats.py`). It is never cited as evidence below.

---

## 2. Three fingerprints

### 2.1 Anthropic — constraint collapse, heterogeneous memorization

Every Anthropic-family model degrades (repeats an already-rejected topic)
by turn ~7–14, in 13 of 14 completed runs:

| model | n_runs | degradation depths | memorization rate (Wilson 95% CI) |
|---|---|---|---|
| haiku | 4 | 22, 7, 7, 7 | **25.8%** [18.8%, 34.3%] |
| sonnet | 4 | 20, 11, 10, 14 | 0.8% [0.1%, 4.6%] |
| opus | 4 | 13, 11, 13, 13 (eerily consistent) | 3.3% [1.3%, 8.3%] |
| fable | 2 | 18, — (1 survived) | 7.9% [3.9%, 15.4%] <!-- REFRESH-AFTER-FILLS --> |

Family contrast (the best-supported single contrast for this pattern,
chosen after the pilot's results made it visible — not part of the
exploratory pairwise battery in §4, and not pre-registered in EXP-004):
pooled Anthropic depths (n=14 runs, mean 14.0, censored at 30 for
survivors) vs pooled OpenAI depths (n=12, mean 29.2): mean difference
**−15.17 turns**, exact/Monte-Carlo permutation **p = 0.0002**, Cliff's
delta **−0.917 (large)**. This is strong exploratory evidence for the
family-level claim, not confirmatory evidence — a genuinely pre-registered
replication would be needed to earn that word (see §4.3 on why the dyadic
matrix underneath it does not individually survive correction). Haiku's
dual role extends here too: it is the rejector judging every run's
degradation, including its own, so its own depth numbers are judged by
themselves — direction of any resulting bias is untested. A robustness
version of this contrast with haiku dropped from the Anthropic pool
(`anthropic_nonhaiku_vs_openai_family_contrast`) has been added to
`benchmark/run_stats_inference.py`; results are pending the post-fill-lane
re-run (§5).

**Memorization is genuinely low for opus/sonnet/fable (0.8–7.9%) — but
NOT for haiku (25.8%), and this needs to be stated plainly, not softened.**
Haiku plays a dual role in this design: it is both the rejector instrument
(EXP-001/002/008) *and* a model-under-test in the cascade roster. Its
memorization rate is statistically indistinguishable in kind from the
GPT-family's heavy-recall tier (22–27%, §2.2), not from its own family's
near-zero pattern. A clean "Anthropic = near-zero memorization" headline
holds for opus/sonnet/fable (pooled 3.6%, n=329) but not for haiku, and the
repo's own house rule (state Claude findings as bluntly as anyone else's)
means this can't be filed away as a footnote. Fisher exact,
haiku-vs-(opus+sonnet+fable): p = 6.5×10⁻¹¹, Holm-corrected p = 2.0×10⁻¹⁰
— the gap is not noise.

### 2.2 OpenAI — constraint adherence, but not uniformly heavy memorization

| model | n_runs | degradation | memorization rate (Wilson 95% CI) |
|---|---|---|---|
| codex:mini | 4 | 0/4 (never) | 7.5% [4.0%, 13.6%] |
| codex:sol | 4 | 1/4 (turn 26) | 21.7% [15.2%, 29.9%] |
| codex:5.4 | 4 | 1/4 (turn 24) | 26.7% [19.6%, 35.2%] |

The constraint-adherence half of this fingerprint is family-wide (0–1
degradations out of 4 for every model). The memorization half is **not**
family-wide — it is specifically sol and 5.4. Mini's own rate (7.5%) is
closer to Anthropic's low tier than to its stablemates. Pooled OpenAI
(mini+sol+5.4, 67/360 = 18.6%) is still far above the pooled open-weights
baseline (deepseek+qwen+glm, 7/405 = 1.7%; Fisher exact p = 1.5×10⁻¹⁶,
Holm-corrected p = 5.9×10⁻¹⁶) — the "OpenAI memorizes heavily" claim is
real at the family level, just not evenly distributed within it.

### 2.3 grok — memorization outlier, unconfirmed by path data

**45%** exact-corpus-hit rate (35/78 jokes), Wilson 95% CI **[34.3%,
55.9%]** <!-- REFRESH-AFTER-FILLS -->. This is the single highest
memorization rate of any model with enough data to trust it, and every
pairwise Fisher-exact contrast against it is significant well past Holm
correction (grok vs. pooled open-weights: p = 1.0×10⁻²⁴, Holm p =
6.2×10⁻²⁴; grok vs. pooled OpenAI: p = 2.8×10⁻⁶, Holm p = 2.8×10⁻⁶; grok
vs. Anthropic non-haiku: p = 4.8×10⁻¹⁹, Holm p = 2.4×10⁻¹⁸).

Caveat that matters more than the number: **grok has zero complete cascade
runs** in this snapshot (`experiment-runs/2026-07-17-cascade-pilot/lane-grok/summary.json`).
This is a **pipeline choice, not a settled fact about how far grok actually
got**: runs r00/r01/r02 reached turn 27, 25, and 23 of 30 respectively
before erroring (r00/r01 on read timeouts, r02 on a session limit; r03
logged 0 turns) — `run_pilot.py` treats a run as all-or-nothing, so a late
failure discards the entire run from `per_model` even though 23-27 turns
of real transcript exist on disk. Its 45% comes entirely from jokes
harvested off those partial transcripts, not from a validated complete
cascade path. It is not in `analysis.json`'s `cross_model`/`per_model`
tables at all — grok's degradation-depth and path-divergence fingerprint
is simply unmeasured right now, not because the model failed outright but
because the runner's all-or-nothing convention discarded near-complete
runs. The retry lane (`lane-grok2`, timeout raised to 300s) is collecting
complete runs now — see §5. Treat "the funny brand runs on retrieval" as a
memorization finding only, not yet a cascade finding.

**kimi is worse and not in the fingerprint list on purpose:** 12/22 jokes
(54.5%, CI [34.7%, 73.1%]) — higher than grok — but n=22 comes from
one-turn scraps of runs that mostly failed outright (token-starvation, then
session limit; 0 complete cascade runs, same as grok). This is flagged as
an open question, not a finding: the true rate could easily move a lot in
either direction once the fill lane (running now) adds real depth.
<!-- REFRESH-AFTER-FILLS -->

### 2.4 Open-weights — fast degradation, low memorization

| model | n_runs | degradation depths | memorization rate |
|---|---|---|---|
| api:deepseek | 4 | 11, 9, 6, 8 (median **8.5**) | 0.8% [0.1%, 4.6%] |
| api:qwen | 4 | 11, 8, 24, 9 (median 10, one run to 24) | 1.7% [0.5%, 5.9%] |
| api:glm | 2 | 21, 14 | 2.4% [0.9%, 6.1%] |

Deepseek is the cleanest signal (4/4 degrade, tight median). Qwen shares
the fast-degradation pattern but is less consistent (one run reached
turn 24). glm's 2 surviving runs both degrade but were produced under a
**mixed generation-config protocol** — see §5.

---

## 3. Why these numbers are trustworthy — the instrument-validation chain

None of §1–2 is meaningful if the rejector or the labeler is the thing
being measured. Four experiments validate the instruments; none are new
statistics computed for this document — they are cited from
`EXPERIMENT_LOG.md`.

- **EXP-008** (constrained-vocabulary labeler, v3): reworded-invariance
  **1.000** (bar 0.90), ARI-vs-gold **0.924** (bar 0.80), repeat consistency
  **0.958**. Every bar the instrument ever had is now cleared, including the
  one v2 missed. This is the paper-grade instrument — see §5 for why the
  pilot itself did not run on it.
- **EXP-007** (temperature fakeability, api:deepseek, N=6/temp): raising
  temperature 0.2→1.2 moves `distinct_2` by **+0.390** (surface diversity is
  temperature-buyable) while moving within-model `set_jaccard` by only
  **−0.012** (bound was ≤0.05). Prefix agreement *does* collapse
  (0.933→0.000) — temperature reorders which topic a model visits next, but
  does not expand the *pool* of topics it draws from. This is the direct
  justification for using set-Jaccard, not sampling-diversity metrics, as
  this pilot's primary instrument: it is the one thing temperature cannot
  fake. (EXP-007b, the second-model replication on qwen needed before this
  generalizes past deepseek, is running — see §5.)
- **EXP-006** (labeler-noise bias simulation, Monte Carlo, 2000 reps/regime):
  the direction of labeler-noise bias on cross-model Jaccard **flips by
  regime**. At the pilot's observed regime (0.10–0.22), noise makes
  measured overlap an **upper bound** — real disjointness is at least this
  strong, possibly stronger. (At high true overlap the same noise would
  understate it — not this pilot's regime, but worth knowing before
  generalizing the "noise is conservative" claim, which the log explicitly
  retracts as regime-naive.)
- **EXP-005** (banter judge, Track 2 instrument, not used in this pilot's
  numbers but validated the same week): separation(contextual − canned) =
  **6.17** (bar 3.0); echo-resistance r = **0.224** (bar ≤0.5). Caveat
  carried forward: generic on-topic pleasantry earns ~2/3 of a genuinely
  contextual reply's delta — a future Track 2 reward should not use this
  delta alone.

---

## 4. Statistical inference — what actually ran

Full machine-readable output:
`experiment-runs/2026-07-17-cascade-pilot/stats_inference.json`, produced
by `benchmark/run_stats_inference.py`. Summary of every test:

### 4.1 Cross-model overlap (headline A) <!-- REFRESH-AFTER-FILLS -->

| statistic | value | CI / p |
|---|---|---|
| observed mean cross-Jaccard | 0.1017 | — |
| `pooled_frequency_baseline` null (authoritative) | p = 1.0 (below entire null range) | one-sided |
| `label_shuffle` null (wrong null, reported only) | p = 1.0 | not cited as evidence |
| bootstrap CI, run-pair level (n=580 pairs, matches point exactly) | [0.0962, 0.1072] | non-independent obs. — CI likely too narrow |
| bootstrap CI, model-pair level (n=45 pairs, more conservative unit) | point 0.1061, [0.0922, 0.1209] | — |

### 4.2 Within-model divergence (headline B) <!-- REFRESH-AFTER-FILLS -->

| statistic | value | CI |
|---|---|---|
| observed mean within-model Jaccard (mean of 10 per-model means) | 0.1824 | — |
| bootstrap CI, model-level (n=10, matches point exactly) | — | [0.1391, 0.2263] |

The 10 per-model point estimates entering this mean have very unequal
precision: the 8 models with N=4 runs each average over 6 within-model
run-pairs, while fable and glm (N=2) each average over exactly 1 run-pair
— a single-pair "mean" has no internal spread to speak of. The
model-level bootstrap above treats all 10 as exchangeable observations
and does not weight by this precision difference; fable's and glm's
contributions to the 0.182 headline are the least precise of the ten.

### 4.3 Degradation-depth fingerprint battery <!-- REFRESH-AFTER-FILLS -->

- **The best-supported single contrast** (Anthropic-pooled, 14 runs, vs
  OpenAI-pooled, 12 runs; depth=null censored to 30): mean diff **−15.17**,
  p = **0.0002** (Monte Carlo, 10,000 draws — exact enumeration over
  C(26,14) is intractable), Cliff's delta **−0.917 (large)**. This
  contrast was chosen after the pilot's results made the family-level
  pattern visible — it was **not** pre-registered in EXP-004. Read it as
  strong exploratory evidence for the pattern, not as confirmatory
  evidence; a genuinely pre-registered replication is needed before
  "confirmatory" is earned. **Robustness check:** haiku is both the
  rejector instrument (it judges every run's degradation, including its
  own) and a model under test, so its own depth numbers are judged by
  themselves — direction of any resulting bias is untested. The driver
  now also computes this contrast with haiku dropped from the Anthropic
  pool (`anthropic_nonhaiku_vs_openai_family_contrast` in
  `stats_inference.json`); results are pending the post-fill-lane re-run,
  not computed yet per the standing instruction not to re-run mid-merge.
- **Exploratory dyadic battery** (all C(10,2) = 45 model pairs, exact
  permutation where feasible — always true here, max C(8,4)=70): raw
  p-values range down to **0.0286** (several pairs show *perfect*
  separation, Cliff's delta = ±1.0 — e.g. deepseek vs. every OpenAI model).
  **After Holm correction across the 45 comparisons, every single p_holm =
  1.0.** None survive. This is expected, not a bug: the exact combinatorial
  p-value floor for two N=4 groups is 1/70 ≈ 0.0143, but no two-sided,
  swap-symmetric statistic (this one is: swapping the group labels
  reproduces the same |stat|) can actually reach it — the smallest
  achievable p is always at least 2/70 ≈ 0.0286, which is exactly what the
  perfectly-separated pairs above hit. Holm multiplies the smallest p by
  the number of comparisons: 45 × 0.0286 ≈ **1.29**, which already exceeds
  1 on its own (45 × the unreachable 1/70 floor would give ≈0.64, which is
  why the achievable 2/70 floor, not the combinatorial 1/70 bound, is the
  number that matters here). Pairs involving glm or fable (n=2) have
  coarser floors still (1/15 ≈ 0.067 vs. an n=4 model; 1/6 ≈ 0.167 vs. each
  other).
  **Conclusion for this document: the best-supported single contrast above
  is the citable strong-exploratory-evidence result. The dyadic matrix is
  descriptive (Cliff's deltas as effect-size color) — none of its
  individual p-values should be read as "significant" after correction.**
  Full 45-pair table (upper triangle of the 10×10 model matrix) in
  `stats_inference.json` under
  `degradation_fingerprint_battery.pairwise_permutation_and_cliffs_delta`.

### 4.4 Degradation-incidence audit

The log states "prediction met massively (8/10 models)." Recounting models
with ≥1 degrading run among the 10 with `per_model` data: **9/10**
(`codex:mini` is the only model with zero degrading runs across all 4).
This does not reproduce 8/10 under the most natural reading of "degrades."
Flagged per instructions, not silently corrected in `EXPERIMENT_LOG.md`
itself.

### 4.5 Memorization proportions <!-- REFRESH-AFTER-FILLS -->

Wilson 95% CIs per model, all 12 models with novelty data (including grok
and kimi), are in §2 tables above and in full in
`stats_inference.json.memorization_proportions.per_model_wilson_95ci`. Six
pre-specified Fisher-exact contrasts, Holm-corrected across that family of
6: every one remains significant at p < 10⁻⁵ after correction (smallest
raw p = 1.0×10⁻²⁴, largest raw p = 2.8×10⁻⁶). Unlike §4.3, memorization has
enough per-model N (78–165 jokes) that family-wise correction doesn't
erase the signal — the asymmetry between "degradation depth barely
survives one pre-specified test" and "memorization survives an entire
correction family" is itself informative: **this pilot's per-run sample
size (2–4) is the real bottleneck for path-level claims, not per-joke
sample size (78–165), which is comparatively well powered.**

---

## 5. Honest limitations

- **N = 2–4 runs/model.** Combinatorial permutation floor 0.0143 (two N=4
  groups); every CI in §4 should be read as pilot-grade precision. Wide
  intervals are the finding, not a defect.
- **Depth capped at 30 turns.** Degradation depths and "survived" counts
  are relative to this cap, not an absolute ceiling — a model could still
  degrade at turn 45.
- **CLI-wrapper confound, bounded not eliminated.** claude/codex lanes run
  through subscription CLIs with no temperature control and transcript-in-
  prompt multi-turn encoding. EXP-007's temperature-unfakeability result
  was demonstrated on api:deepseek (native API) only; it grounds the
  *choice* of metric but does not itself remove the wrapper confound from
  the claude/codex lanes' numbers.
- **glm ran a mixed generation-config protocol.** `max_tokens` was raised
  400→2048 mid-experiment after early attempts silently burned the entire
  budget on `reasoning_content` and returned empty completions (see
  `benchmark/providers.py`'s `glm` entry). glm's 2 surviving runs used the
  corrected setting, but the failures folded into `analysis.json`'s
  `failures` list came from the earlier, broken configuration — glm's
  effective sample is smaller and less uniform than its "n_runs: 2" implies.
- **Pilot labeled with the v2 free-vocabulary labeler**, not v3
  (EXP-008's paper-grade constrained-vocabulary instrument, validated
  *after* the pilot ran). v2's known failure mode (synonym/hypernym
  jitter, e.g. cat/pet) has a **conservative** bias direction for collapse
  claims (splits topics → makes models look more diverse, not less), which
  is why proceeding at pilot grade was defensible — but it means every
  jaccard number in this document could shift, direction unknown a priori
  in magnitude, once the queued v3 post-hoc relabel of the pilot's stored
  jokes (zero new API calls) lands.
- **Memorization is a corpus-coverage lower bound.** The reference corpus
  is the 25 ChatGPT joke templates (Jentzsch & Kersting 2023) plus a small
  hand corpus — it cannot contain every joke any model has memorized. Every
  percentage in §2 understates true memorization reliance, for every model,
  not selectively.
- **Single rejector (haiku).** EXP-003b showed a bigger model (sonnet) is a
  *worse* rejector instrument (ARI 0.633 vs haiku's 0.837) — bigger isn't
  better here — but only two rejector models have ever been tried.
  Rejector-model generality beyond haiku is untested. Compounding this,
  haiku's dual role (instrument + subject) is the single most awkward
  design fact in this pilot (§2.1) and is not fixable without either a
  distinct rejector or dropping haiku from the roster.
- **grok and kimi have zero complete cascade runs** in this snapshot. Their
  memorization numbers (§2.3) are real but stand alone, uncorroborated by
  any path-level (degradation, divergence, cross-model overlap) data.
- **The exploratory 45-pair degradation battery does not survive
  correction (§4.3)** — only the single best-supported family contrast
  (itself post-hoc, not pre-registered — §2.1/§4.3) should be cited, and
  only as strong exploratory evidence, never as confirmatory.
- **Fill lanes are running now**, not reflected in the numbers above where
  marked `<!-- REFRESH-AFTER-FILLS -->`: fable +2 runs (→ n=4, joins the
  full-N tier), grok +4 runs (→ first complete cascade data for grok,
  changing it from "absent" to "present" in `cross_model`/`per_model`),
  kimi +2 runs (→ n=4 jokes' worth of additional novelty data on a
  currently 22-joke base). Any number touching these three models will
  move; re-run `benchmark/run_stats_inference.py` against the merged
  `analysis.json`/`novelty.json` before citing this document externally.

---

## 6. What's next

1. **v3 relabel** — re-score the pilot's already-collected jokes under
   EXP-008's constrained-vocabulary labeler (zero new API calls); report
   §1/§4 under both instruments.
2. **EXP-007b** — qwen temperature replication, running now; needed before
   the temperature-unfakeability claim (§3) generalizes past deepseek.
3. **Fill lanes** — fable, grok, kimi (§5); re-run the stats driver and
   this document once they land.
4. **EXP-009** — semantic novelty tier (`env/semantic_novelty.py`),
   directly relevant to §5's corpus-coverage-lower-bound caveat. Results:
   `experiment-runs/2026-07-17-semantic-novelty-validation/report.json`.
   **EXPERIMENT_LOG entry pending** (being written separately) — cite the
   report.json artifact directly until that entry lands rather than
   restating its numbers here secondhand.
5. **Rejector generality** — a second, independent rejector model (not
   haiku, not sonnet — both already tried and one already rejected as
   worse) to break haiku's dual-role confound (§2.1, §5).

---

*Source data:* `experiment-runs/2026-07-17-cascade-pilot/analysis.json`,
`experiment-runs/2026-07-17-cascade-pilot/novelty.json`,
`experiment-runs/2026-07-17-cascade-pilot/stats_inference.json`. Lane logs:
`lane-claude/`, `lane-codex/`, `lane-api/`, `lane-grok/`,
`lane-api-fill-glm/`, `lane-api-fill-glm2/`, `lane-api-fill-kimi/` (all
under the same pilot directory). *Methods reference:* `docs/BENCHMARK.md`.
*Inference code:* `benchmark/stats.py` (toolkit),
`benchmark/run_stats_inference.py` (this analysis's driver). *Experiment
log:* `EXPERIMENT_LOG.md`, entries EXP-004 through EXP-008.
