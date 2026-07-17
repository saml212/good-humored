# Findings — cascade pilot (2026-07-17)

Status: pilot-grade, refreshed after fill lanes merged. Roster is now
**11 models**, N=4 runs/model except `api:glm` (N=2 — see §5). Not paper
numbers. Every number below traces to
`experiment-runs/2026-07-17-cascade-pilot/` and the Result/Verdict blocks
of EXP-004 through EXP-009 in `EXPERIMENT_LOG.md` (including the EXP-004
kimi-drop addendum and the EXP-007b/c temperature-replication entries).
Statistical inference in this document was computed by
`benchmark/run_stats_inference.py` and written in full to
`experiment-runs/2026-07-17-cascade-pilot/stats_inference.json` — every
statistic quoted here has a corresponding JSON path in that file, and the
reconstruction was re-verified against the refreshed `analysis.json`
(11/11 models, diffs 0.0 — see `stats_inference.json.data_integrity_check`).

This revision resolves every `REFRESH-AFTER-FILLS` marker from the prior
draft; none remain. §5 records exactly what changed when the fill lanes
landed.

---

## 1. Headline: the pre-registered hypothesis failed

EXP-004 pre-registered, before any data existed: frontier and open-weight
models share a substantially overlapping joke-topic pool, predicted
cross-model mean topic-set Jaccard **≈ 0.35**. Measured (11-model roster):
**0.113**.

The within-model prediction failed the same direction, more severely: models
were predicted to repeat *themselves* across runs more than they'd match
each other (≈0.55 self-Jaccard). Measured within-model self-Jaccard:
**0.208** — up from the 10-model pilot's 0.182, driven almost entirely by
grok's newly-landed complete cascade data (self-Jaccard 0.443, the highest
of any model; §2.3).

Both misses point the same way — models are more idiosyncratic, not more
uniform, than the pre-registration assumed. What emerged instead was not
"no signal" — it was **four orthogonal per-lab failure fingerprints**, each
strong, each different, none matching the shared-pool story:

| Prediction (EXP-004, pre-registered) | Measured | Test | Result |
|---|---|---|---|
| Cross-model Jaccard ≈ 0.35 | **0.113** | `pooled_frequency_baseline` null, one-sided | Observed sits *below the entire simulated null distribution* (10,000 draws, range 0.119–0.157). p = 1.0 with add-one correction ⟺ all 10,000 synthetic draws were ≥ observed |
| Within-model Jaccard ≈ 0.55 | **0.208** | bootstrap CI, model-level (n=11) | 95% CI [0.151, 0.271] |
| ≥1/3 of roster degrades by turn 30 | met, but the log's "8/10" count doesn't reproduce | incidence audit | **9/11** models show ≥1 degrading run (`codex:mini` and, now, `grok` both fully clean); see §4.4 |

The `pooled_frequency_baseline` result is worth sitting with: it is not
merely "not significantly above chance." The real cross-model overlap is
*lower than every one of 10,000 synthetic universes* in which each model
draws i.i.d. from one shared, frequency-weighted topic bag built from the
pilot's own vocabulary (281 distinct topics — up from 270 as grok's real
vocabulary joined the pool). Real models show less overlap than a
chance-cooccurrence baseline in a shared vocabulary predicts. Per EXP-006
(§3), the pilot's overlap regime (0.10–0.22) is one where labeler noise
makes measured overlap an *upper bound* on true overlap — so if anything,
true cross-model disjointness is understated here, not overstated. The
null distribution's min/max and the vocabulary size above are now
persisted directly in `stats_inference.json` under
`headline_cross_model_overlap.null_test.pooled_frequency_baseline.diagnostics`
(`pooled_frequency_null_diagnostics`) — cited from there, not computed
off-pipeline as in the prior draft of this document.

`label_shuffle` was also computed (p = 1.0) and is reported for
transparency only — it is the documented **wrong** null for a shared-pool
claim (near-zero power to detect a genuinely shared pool; see
`benchmark/stats.py`'s `cross_model_null` docstring and
`test_fully_shared_pool_label_shuffle_has_no_power` in
`benchmark/tests/test_stats.py`). It is never cited as evidence below.

---

## 2. Four fingerprints

(Terminology note: EXP-004's original verdict said "three" — at that
point grok had no path data and its recall pattern was folded in with
OpenAI's. grok's complete cascade profile (§2.3) is distinct from
OpenAI's on two of three axes, so the count is now four.)

### 2.1 Anthropic — constraint collapse, one outlier, heterogeneous memorization

haiku, sonnet, and opus degrade (repeat an already-rejected topic) in
**12 of 12 runs, uniformly** — every single run, no exceptions. Family
degradation across all four Anthropic models is 13 of 16 runs, and the
missing 3 are not scattered — they are all **fable**:

| model | n_runs | degradation depths | memorization rate (Wilson 95% CI) |
|---|---|---|---|
| haiku | 4 | 22, 7, 7, 7 (4/4) | **25.8%** [18.8%, 34.3%] |
| sonnet | 4 | 20, 11, 10, 14 (4/4) | 0.8% [0.1%, 4.6%] |
| opus | 4 | 13, 11, 13, 13 (4/4, eerily consistent) | 3.3% [1.3%, 8.3%] |
| fable | 4 | 18, —, —, — (**1/4**) | 4.7% [2.3%, 9.4%] |

**Fable breaks the family pattern, and this is a finding, not noise.**
With the fill lane's extra 2 runs landed (n_runs 2→4), fable now
degrades in only 1 of 4 runs — the same 0-or-1-per-4 adherence rate as
the OpenAI family (§2.2: mini 0/4, sol 1/4, 5.4 1/4), not the 4/4 pattern
every other Anthropic model shows. Its memorization stayed low and got
more precise with the extra data (7.9%→**4.7%**, n=89→149, tighter CI).
The newest Anthropic model in this roster is trending toward OpenAI-style
constraint adherence while keeping Anthropic-style low memorization — a
combination none of the other three models show. This is exactly the kind
of within-family heterogeneity this document's house rule requires
surfacing, not burying under the family-level headline.

Family contrast (the best-supported single contrast for this pattern,
chosen after the pilot's results made it visible — not part of the
exploratory pairwise battery in §4, and not pre-registered in EXP-004):
pooled Anthropic depths (n=16 runs — haiku+sonnet+opus+fable, mean 16.0,
censored at 30 for survivors) vs pooled OpenAI depths (n=12, mean 29.2):
mean difference **−13.17 turns**, exact/Monte-Carlo permutation
**p = 0.0002**, Cliff's delta **−0.781 (large)**. Both the mean difference
and the effect size **weakened** from the 10-model pilot's −15.17/−0.917 —
fable's 3 newly-landed survivals are exactly why, and that is the correct,
expected direction for adding a model that partially breaks the pattern.
This remains strong exploratory evidence for the family-level claim, not
confirmatory evidence — a genuinely pre-registered replication would be
needed to earn that word (see §4.3 on why the dyadic matrix underneath it
does not individually survive correction).

Haiku's dual role extends here too: it is the rejector judging every
run's degradation, including its own, so its own depth numbers are judged
by themselves — direction of any resulting bias is untested. The
robustness version of this contrast with haiku dropped from the Anthropic
pool is now computed: sonnet+opus+fable (n=12 runs, mean 17.75) vs OpenAI
(n=12, mean 29.2): mean difference **−11.42 turns**, **p = 0.0005**, Cliff's delta
**−0.708 (large)**. **The confound does not carry the result** — dropping
haiku shrinks the effect somewhat (−13.17→−11.42, −0.781→−0.708) but it
stays large and stays significant. The family-level pattern is not an
artifact of haiku judging itself.

**Memorization is genuinely low for opus/sonnet/fable (0.8–4.7%) — but
NOT for haiku (25.8%), and this needs to be stated plainly, not softened.**
Haiku plays a dual role in this design: it is both the rejector instrument
(EXP-001/002/008) *and* a model-under-test in the cascade roster. Its
memorization rate is statistically indistinguishable in kind from the
GPT-family's heavy-recall tier (22–27%, §2.2), not from its own family's
near-zero pattern. A clean "Anthropic = near-zero memorization" headline
holds for opus/sonnet/fable (pooled 3.1%, 12/389) but not for haiku, and
the repo's own house rule (state Claude findings as bluntly as anyone
else's) means this can't be filed away as a footnote. Fisher exact,
haiku-vs-(opus+sonnet+fable): p = 1.7×10⁻¹², Holm-corrected p = 3.8×10⁻¹²
— the gap is not noise, and got more significant, not less, with fable's
extra data.

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

### 2.3 grok — constraint adherence, fixed retrieval repertoire, top memorization

The retry lane (`lane-grok2`, timeout raised to 300s after the original
lane's read-timeout failures) landed **4/4 complete cascade runs**. The
"unconfirmed by path data" framing from the prior draft of this document
is now obsolete — grok has a complete profile, and it is its own
fingerprint, distinct from all three others:

- **Zero degradations in 4 runs** — grok never repeats an already-rejected
  topic and never refuses, matching OpenAI's constraint-adherence pattern
  (§2.2: mini 0/4) rather than any Anthropic model.
- **set_jaccard = 0.443** — the **highest within-model overlap of any
  model in the roster** (next: codex:5.4 at 0.281; the 11-model mean is
  0.208, §4.2). grok tells itself the same small set of topics run after
  run far more than any other model — a fixed retrieval repertoire, not a
  wide distribution.
- **Memorization 40.9%** [34.3%, 47.9%] (81/198 jokes — tighter than the
  prior draft's 78-joke estimate). Every pairwise Fisher-exact contrast
  against it is significant well past Holm correction: grok vs. pooled
  open-weights p = 9.4×10⁻³⁷ (Holm 5.7×10⁻³⁶); grok vs. pooled OpenAI
  p = 2.5×10⁻⁸ (Holm 2.5×10⁻⁸); grok vs. Anthropic non-haiku p = 1.6×10⁻³¹
  (Holm 8.0×10⁻³¹).

Put together: grok adheres to the constraint like OpenAI, but the topics
it retreats to under that constraint are drawn from an unusually small,
fixed, heavily-memorized set. "The funny brand runs on retrieval" is now a
cascade-path finding as well as a memorization finding — the fixed
repertoire (high self-Jaccard) and the heavy memorized-joke reliance are
two views of the same underlying behavior, not independent facts.

**kimi is DROPPED from the roster, not merely flagged.** Per the EXP-004
addendum: kimi-k2.5 is a reasoning model whose `reasoning_content` burn
grows with the cascade's accumulating rejection list, so no fixed
`max_tokens` survived depth 30 — three escalating attempts all failed
(400 → empty at turn 1; 2048 → died turns 6/12; 4096 → died turns 20/18,
the pre-committed "4096 or drop" threshold). Zero complete cascade runs,
ever. Its memorization number is real but remains a scrap-based flag, not
a path finding: **40.0%** [28.6%, 52.6%] (24/60 jokes, tighter than the
prior 22-joke estimate but still built entirely from partial/failed-run
scraps). Notably close to grok's 40.9% — but with no path data to
corroborate it and a design-level reason (the max_tokens failure mode)
that it will likely never get any, this repo does not include kimi in the
per-lab fingerprint claims.

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
- **EXP-007/007b/007c** (temperature fakeability): EXP-007 (api:deepseek,
  N=6/temp) found raising temperature 0.2→1.2 moves `distinct_2` by
  **+0.390** (surface diversity is temperature-buyable) while moving
  within-model `set_jaccard` by only **−0.012** (bound was ≤0.05) —
  temperature reorders which topic a model visits next (prefix agreement
  collapses 0.933→0.000) but does not expand the *pool* it draws from.
  **EXP-007b (qwen) is DISQUALIFIED, not a replication**: qwen's endpoint
  silently ignores the temperature parameter (byte-identical outputs at
  1.2 across all 6 runs) — no evidence for or against fakeability where
  temperature never reached the model. **EXP-007c (glm) REPLICATED the
  pattern** on a second honored-endpoint model: distinct_2 **+0.143**
  (predicted +0.15 — near-exact), set_jaccard **−0.037** (within EXP-007's
  ≤0.05 bound), ratio **3.9×** (bar ≥3×), manipulation-check gate 6/6
  distinct turn-1 jokes at temp 0.95 (vs. qwen's 1/6 — the gate that
  qwen failed). The differentiator claim now stands on two honored-endpoint
  native-API models (deepseek, glm), not one; this is the direct
  justification for using set-Jaccard, not sampling-diversity metrics, as
  this pilot's primary instrument.
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

### 4.1 Cross-model overlap (headline A)

| statistic | value | CI / p |
|---|---|---|
| observed mean cross-Jaccard | 0.1126 | — |
| null diagnostics (`n_distinct_topics`, range) | 281 topics; null range [0.1189, 0.1567] | persisted in `stats_inference.json` |
| `pooled_frequency_baseline` null (authoritative) | p = 1.0 (below entire null range) | one-sided |
| `label_shuffle` null (wrong null, reported only) | p = 1.0 | not cited as evidence |
| bootstrap CI, run-pair level (n=800 pairs, matches point exactly) | [0.1072, 0.1178] | non-independent obs. — CI likely too narrow |
| bootstrap CI, model-pair level (n=55 pairs, more conservative unit) | point 0.1133, [0.0983, 0.1296] | — |

### 4.2 Within-model divergence (headline B)

| statistic | value | CI |
|---|---|---|
| observed mean within-model Jaccard (mean of 11 per-model means) | 0.2075 | — |
| bootstrap CI, model-level (n=11, matches point exactly) | — | [0.1507, 0.2706] |

Precision across the 11 per-model point estimates is much more even now
than in the 10-model pilot: **10 of 11 models** have N=4 runs (6
within-model run-pairs each); only **`api:glm` remains at N=2** (1
run-pair — see §5). glm's contribution to the 0.208 headline is
accordingly the single least precise of the eleven. The headline itself
moved up substantially from the 10-model pilot's 0.182, and the reason is
identifiable, not diffuse: grok's self-Jaccard (0.443, §2.3) is nearly
double the next-highest model's (codex:5.4, 0.281) and is now one of
eleven equally-weighted points in the mean — a single outlier model pulls
a model-level average by construction, which is exactly why per-model
point estimates (`stats_inference.json`'s
`headline_within_model_divergence.per_model_point_estimates`, and §2.3's
grok row above) matter alongside the 0.208 average, not instead of it.

### 4.3 Degradation-depth fingerprint battery

- **The best-supported single contrast** (Anthropic-pooled — haiku,
  sonnet, opus, fable — 16 runs, vs OpenAI-pooled, 12 runs; depth=null
  censored to 30): mean diff **−13.17**, p = **0.0002** (Monte Carlo,
  10,000 draws — exact enumeration is intractable at this n), Cliff's
  delta **−0.781 (large)**. Both weakened from the 10-model pilot's
  −15.17/−0.917 — fable's 3 newly-landed survivals (§2.1) pull the
  Anthropic pool's mean depth up. This contrast was chosen after the
  pilot's results made the family-level pattern visible — it was **not**
  pre-registered in EXP-004. Read it as strong exploratory evidence for
  the pattern, not as confirmatory evidence; a genuinely pre-registered
  replication is needed before "confirmatory" is earned.
  **Robustness check (now computed):** haiku is both the rejector
  instrument (it judges every run's degradation, including its own) and a
  model under test, so its own depth numbers are judged by themselves —
  direction of any resulting bias is untested. Dropping haiku from the
  Anthropic pool (`anthropic_nonhaiku_vs_openai_family_contrast` in
  `stats_inference.json`: sonnet+opus+fable, 12 runs, vs OpenAI, 12 runs):
  mean diff **−11.42**, p = **0.0005**, Cliff's delta **−0.708 (large)**.
  The effect shrinks somewhat without haiku but stays large and
  significant — **the confound does not carry the result.**
- **Exploratory dyadic battery** (all C(11,2) = 55 model pairs, exact
  permutation where feasible — always true here, max C(8,4)=70): raw
  p-values still range down to **0.0286** (several pairs show *perfect*
  separation, Cliff's delta = ±1.0 — e.g. deepseek vs. grok/OpenAI models).
  **After Holm correction across the 55 comparisons, every single p_holm =
  1.0.** None survive. This is expected, not a bug: the exact combinatorial
  p-value floor for two N=4 groups is 1/70 ≈ 0.0143, but no two-sided,
  swap-symmetric statistic (this one is: swapping the group labels
  reproduces the same |stat|) can actually reach it — the smallest
  achievable p is always at least 2/70 ≈ 0.0286, which is exactly what the
  perfectly-separated pairs above hit. Holm multiplies the smallest p by
  the number of comparisons: 55 × 0.0286 ≈ **1.57**, which already exceeds
  1 on its own (55 × the unreachable 1/70 floor would give ≈0.79, which is
  why the achievable 2/70 floor, not the combinatorial 1/70 bound, is the
  number that matters here). Now that fable and grok are both N=4, only
  `api:glm` (N=2) has a coarser floor: C(6,2)=15, combinatorial 1/15 ≈
  0.067, practically achievable ≈2/15 ≈ 0.133 against any N=4 model — the
  N=2-vs-N=2 case from the prior draft no longer arises (glm is the only
  N=2 model left).
  **Conclusion for this document: the best-supported single contrasts
  above are the citable strong-exploratory-evidence results. The dyadic
  matrix is descriptive (Cliff's deltas as effect-size color) — none of
  its individual p-values should be read as "significant" after
  correction.** Full 55-pair table (upper triangle of the 11×11 model
  matrix) in `stats_inference.json` under
  `degradation_fingerprint_battery.pairwise_permutation_and_cliffs_delta`.

### 4.4 Degradation-incidence audit

The log states "prediction met massively (8/10 models)" — against the
original 10-model pilot, before fills. Recounting against the current
**11-model** roster: models with ≥1 degrading run is **9/11**. The
zero-degradation set changed shape, not just size: it used to be
`codex:mini` alone; it is now **`codex:mini` AND `api:grok`** (§2.3) —
grok's complete cascade data adds a second model to the
never-degrades set rather than breaking it. Neither 8/10 nor 9/11 is a
clean reproduction of the log's exact count under the most natural
reading of "degrades" (≥1 run degrades); flagged per instructions, not
silently corrected in `EXPERIMENT_LOG.md` itself.

### 4.5 Memorization proportions

Wilson 95% CIs per model, all 12 models with novelty data (11 in the
cascade roster plus kimi, dropped from path-level analysis but retained
for novelty — §2.3), are in §2 tables above and in full in
`stats_inference.json.memorization_proportions.per_model_wilson_95ci`. Six
pre-specified Fisher-exact contrasts, Holm-corrected across that family of
6: every one remains significant at p < 10⁻⁷ after correction (smallest
raw p = 9.4×10⁻³⁷, largest raw p = 2.5×10⁻⁸ — both tightened from the
10-model pilot as grok's and fable's larger samples landed). Unlike §4.3,
memorization has enough per-model N (60–198 jokes) that family-wise
correction doesn't erase the signal — the asymmetry between "degradation
depth barely survives one pre-specified test" and "memorization survives
an entire correction family" is itself informative: **this pilot's
per-run sample size (2–4) is the real bottleneck for path-level claims,
not per-joke sample size (60–198), which is comparatively well powered.**

---

## 5. Honest limitations

- **Fill-lane changelog (fable, grok, kimi merged 2026-07-17):** fable
  2→4 runs (fills completed the pair that failed on session limit in the
  original lane); grok went from **absent entirely** (0 complete runs, all
  4 original-lane attempts timed out) to **4/4 complete runs** via a retry
  lane with timeout raised 120s→300s; kimi was **dropped from the cascade
  roster** after three escalating `max_tokens` attempts (400/2048/4096)
  all failed on the reasoning-content burn described in the EXP-004
  addendum — it remains in novelty.json (memorization only, §2.3) but will
  not get path-level data without a design change (streaming +
  reasoning-budget control, or a non-reasoning kimi variant). Roster is
  now 11 models for all path-level analysis (§1, §4.1–4.4), 12 for
  memorization (§4.5).
- **N = 4 runs/model for 10 of 11 models; `api:glm` remains at N=2.**
  Combinatorial permutation floor 0.0143 for any two N=4 groups (practically
  achievable ≈0.0286, §4.3); coarser for any pair involving glm. Every CI
  in §4 should be read as pilot-grade precision. Wide intervals are the
  finding, not a defect.
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
- **kimi has zero complete cascade runs and is dropped from the roster**
  (see the changelog bullet above). Its memorization number (§2.3) is real
  but stands alone, uncorroborated by any path-level data, and — per the
  EXP-004 addendum's diagnosis of *why* it fails (reasoning-content burn
  that scales with cascade depth) — is unlikely to get any without a
  design change. Grok, which shared this limitation in the prior draft of
  this document, no longer does: it now has full path-level data (§2.3).
- **The exploratory 55-pair degradation battery does not survive
  correction (§4.3)** — only the two best-supported family contrasts
  (with and without haiku; both post-hoc, not pre-registered — §2.1/§4.3)
  should be cited, and only as strong exploratory evidence, never as
  confirmatory.
- **A handful of descriptive text fields inside `stats_inference.json`
  itself were stale until this refresh's driver fixes.** Auditing the
  refreshed JSON against this document surfaced five places where
  `benchmark/run_stats_inference.py` had hardcoded a value that was only
  correct for the original 10-model roster (a "45 model pairs" unit
  string now describing 55-pair data; the within-model `claim` field
  literally saying "observed 0.182" next to a JSON field reading 0.208;
  an "n=10" bootstrap-CI unit label on n=11 data; a "10 with per_model
  data" phrase in the incidence audit; and a haiku-memorization warning
  hardcoding fable's *old* 7.9%%/8%% rate next to its *new* 4.7%% figure).
  All five are now computed from the data rather than hardcoded — but the
  fix landed in the driver's source after `stats_inference.json` was last
  generated, so a handful of prose strings inside that JSON (not any
  numeric field — every number was re-verified directly against
  `analysis.json`/`novelty.json` for this document) will only read
  correctly again once the driver is re-run. This document's numbers come
  from the JSON's numeric fields and from direct recomputation, not from
  the stale prose.

---

## 6. What's next

1. **v3 relabel** — re-score the pilot's already-collected jokes under
   EXP-008's constrained-vocabulary labeler (zero new API calls); report
   §1/§4 under both instruments. (Lane activity visible under
   `experiment-runs/2026-07-17-cascade-pilot-v3-relabel/` at time of
   writing, but no completed analysis to cite yet — not claimed here.)
2. **EXP-007b/c — DONE.** qwen (EXP-007b) is disqualified: its endpoint
   silently ignores the temperature parameter. glm (EXP-007c) replicated
   the differentiator claim on a second honored-endpoint model: distinct_2
   +0.143 (predicted +0.15), set_jaccard −0.037, ratio 3.9×, manipulation
   gate 6/6 (§3). The claim now stands on two models, not one.
3. **kimi replacement lane** — per the EXP-004 addendum's own
   recommendation: streaming with reasoning-budget control, or a
   non-reasoning kimi variant, not another fixed `max_tokens` bump (three
   escalating attempts already failed — §5).
4. **EXP-009 — DONE.** Semantic novelty tier
   (`env/semantic_novelty.py`), directly relevant to §5's
   corpus-coverage-lower-bound caveat. Full paraphrase detection 1.000 at
   FPR≤0.05 (recommended threshold 0.38), vs. 0.0 for the n-gram baseline
   on the identical paraphrase set — closes the 2-word-reskin evasion the
   n-gram tier provably missed. One MAJOR carried forward as a documented,
   unfixed limitation: padding/dilution (a verbatim memorized joke behind
   filler sentences) evades every novelty tier, n-gram and semantic alike.
   `EXPERIMENT_LOG.md`'s EXP-009 entry (with the full validation-artifact
   story — a first run's 0.0 detection turned out to be a reference-set
   mismatch, not a finding, per its own `[LEARN]` block) is no longer
   pending; it has landed. Full numbers:
   `experiment-runs/2026-07-17-semantic-novelty-validation/report.json`.
5. **Rejector generality** — a second, independent rejector model (not
   haiku, not sonnet — both already tried and one already rejected as
   worse) to break haiku's dual-role confound (§2.1, §5).

---

*Source data:* `experiment-runs/2026-07-17-cascade-pilot/analysis.json`,
`experiment-runs/2026-07-17-cascade-pilot/novelty.json`,
`experiment-runs/2026-07-17-cascade-pilot/stats_inference.json`. Lane logs
(all under the same pilot directory): `lane-claude/`, `lane-codex/`,
`lane-api/`, `lane-grok/`, `lane-grok2/`, `lane-claude-fill-fable/`,
`lane-api-fill-glm/`, `lane-api-fill-glm2/`, `lane-api-fill-kimi/`,
`lane-api-fill-kimi2/`. *Methods reference:* `docs/BENCHMARK.md`.
*Inference code:* `benchmark/stats.py` (toolkit),
`benchmark/run_stats_inference.py` (this analysis's driver). *Experiment
log:* `EXPERIMENT_LOG.md`, entries EXP-004 through EXP-009, including the
EXP-004 kimi-drop addendum and the EXP-007b/c temperature-replication
entries.
