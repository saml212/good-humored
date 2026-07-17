# Theory → Code Map

Status: **standing instrument, re-read and updated every cycle.** Written
2026-07-17. Owner: Sam Larson.

Purpose: this repo invokes psychology, philosophy, and sociology
constructs constantly — in README.md, docs/BENCHMARK.md, references/ — to
justify why the benchmark and reward stack are built the way they are.
This document is the audit of whether that invocation is backed by actual
code, or is prose riding on a citation. For every construct: what the
theory says, where (if anywhere) it is operationalized, how faithful that
operationalization is, what breaks it today, and what a real fix costs.

**Tone note, stated once so it doesn't need repeating per section:** this
is an internal audit, not a grant application. "Partial proxy" and
"distortion" are not embarrassing findings — they are the actual content
of the document. A theory-map with nine "faithful, fully implemented"
verdicts would mean nobody looked hard enough.

**Method:** every code claim below was checked against the actual file
(read in full, not grepped), not against what a docstring says the file
does — several of the gaps below are exactly the gap between the two.

---

## 1. Incongruity-Resolution Theory

**Construct.** Suls (1972), in Goldstein & McGhee eds., *The Psychology of
Humor* (`suls1972twostage`) — the project's stated primary citation for
its own thesis.

**What the theory actually says.** Two sequential stages, not one: (1) a
setup generates an expectation that the punchline disconfirms, producing
incongruity/surprise; (2) the perceiver then *problem-solves* to find a
cognitive rule that reconciles the punchline with the setup. Incongruity
alone is explicitly insufficient — resolution is the load-bearing second
stage, not an epilogue. Wyer & Collins (1992) extend it further:
resolution alone isn't enough either — a joke needs "elaboration"
(drawing out further implications) to actually land. So the theory has
**three** structural moments in total (violate → resolve → elaborate),
and this project's own reference file (`references/psychology.md` §4)
records all three.

**Where it lives in code today.** NOWHERE, for either of the two
canonical stages, let alone the third. `ComprehensibilityReward`
(`env/rewards.py:549-589`) is the nearest thing to a "resolution" proxy
and it is not one: it scores token-count band, terminal punctuation, and
unique-token ratio — a structural well-formedness check with no
representation of "does a rule exist that reconciles the punchline with
the setup." `JudgePreferenceReward` (`env/rewards.py:223-284`) collapses
whatever the judge intuits about violation *and* resolution into one
undifferentiated `[0,1]` scalar; the two stages are not separately
elicited from the judge, so even if a judge model is implicitly doing
Suls' two-stage reasoning internally, the reward signal never surfaces
which stage succeeded or failed. `benchmark/rejector.py`'s labeler
extracts a topic noun, explicitly not a punchline mechanism (by design —
see its own "never the punchline mechanism" rule) — so even the
benchmark side has nothing that touches incongruity or resolution as
such.

**Fidelity assessment: distortion.** The repo's own README leads with
"familiar enough to comprehend instantly, novel enough to break your
expectation" as *the* thesis, cites Suls as its provenance, and then the
codebase implements neither half as the theory defines them (see §3 for
the granular breakdown of what the "familiar"/"novel" code proxies
actually measure instead). Calling the current stack "incongruity-
resolution grounded" in prose while the code measures surface length and
n-gram overlap is the single cleanest instance of theory-richer-than-code
in this repo.

**Known exploit/gap.** Because resolution is never checked, a
grammatical, correctly-punctuated non-sequitur that clears the
comprehensibility heuristic's three sub-checks (length band, terminal
punctuation, 0.35–0.95 unique-token ratio) scores full comprehensibility
credit with zero actual incongruity-resolution structure. There is no
regression test anywhere in `env/tests/test_rewards.py` that a
well-formed-but-nonsensical sentence is penalized relative to an actual
joke — `test_well_formed_completion_gets_full_credit` only checks that a
*real* joke passes, never that a fake one is caught.

**Candidate upgrade.** Two new signals, not one, matching the theory's
own two stages: (a) an expectation-violation magnitude proxy — e.g. the
punchline's token-level surprisal conditioned on the setup, from any
small causal LM, as a cheap stand-in for "how strained was the
expectation"; (b) a resolution/coherence proxy — an NLI-style or
judge-based "does a single frame make setup+punchline both true at once"
check, scored independently of funniness. Validate by hand-labeling ~60
items (30 real jokes / 30 setup+non-sequitur pairs matched for length) and
checking that the resolution signal separates them while
`ComprehensibilityReward` alone does not — a direct, cheap disproof
check before trusting the new term.

---

## 2. Benign Violation Theory (BVT)

**Construct.** McGraw & Warren (2010), *Psychological Science* 21(8)
(`mcgraw2010benign`) — primary, full-text confirmed.

**What the theory actually says.** Three conditions are *jointly
necessary and sufficient*: a situation is appraised as a violation
(threatens how the world "should" be), appraised as benign
(simultaneously, some competing norm/psychological distance neutralizes
the threat), and — the load-bearing clause — **these two appraisals must
occur simultaneously**. Five experiments show pure-violation (offense)
and pure-benign (boring) both fail to produce humor. This project's own
`references/README.md` and `references/psychology.md` both flag the
structural implication explicitly: a dual-appraisal, simultaneity-gated
theory suggests a reward architecture that **multiplies** a
violation-detector by a benign-detector, not one that sums or averages
them — because a simultaneity gate means either signal at zero should
zero out the product, not merely subtract from a sum.

**Where it lives in code today.** NOWHERE as a dual-appraisal detector —
no violation-detector or benign/safety-detector pair exists anywhere in
`env/rewards.py` or `benchmark/`. More specifically damning: the reward
stack's actual combination algebra is the *opposite* of what the theory
recommends. `combined_reward()` (`env/rewards.py:634-659`) sums all five
terms (`judge_preference`, `corpus_novelty_penalty`,
`self_repetition_penalty`, `intra_group_diversity`,
`comprehensibility`) — `sum(term[i] for term in per_term)` — a flat
linear combination with no multiplicative gate anywhere in the pipeline.

**Fidelity assessment: distortion — the sharpest one in this document.**
`references/README.md` calls BVT "arguably the single most directly
useful theory for reward design" specifically *because* it's
multiplicative, and the actual reward architecture built is purely
additive. This isn't a missing feature so much as a live contradiction
between what the citation corpus recommends and what the shipped
combination function does.

**Known exploit/gap.** Additive combination lets a completion compensate
across dimensions the theory says cannot trade off. A safe, comprehensible,
totally non-violating completion (`comprehensibility` term maxed,
`corpus_novelty`/`self_repetition` both near-zero penalty because it's
generic filler, judge gives it a modest score for being pleasant) can
out-score a genuinely risky, well-resolved violation-humor completion that
the judge scores lower for edginess — BVT predicts the *first* completion
shouldn't be funny at all (pure benign, no violation = boring), but
nothing in the additive stack enforces that; it just adds up whatever
credit each independent heuristic hands out.

**Candidate upgrade.** Build the two missing detectors and gate rather
than sum: a violation-detector (cheap version: judge-rated "does this
defy a norm/expectation, yes/no or 0–1"); a benign-detector (cheap
version: an existing safety/toxicity classifier plus a
"psychological-distance" heuristic, e.g. fictional/absurd framing scores
higher benign than a first-person real-world claim). Combine as
`judge_weight * violation * benign` in place of (or alongside, logged
separately for comparison) the current flat `judge_preference` term.
Validate the way McGraw & Warren did: construct three synthetic
conditions — pure-violation, pure-benign, both-simultaneous — and confirm
the multiplicative term is near-zero on the first two and positive only
on the third, mirroring the paper's own five-experiment logic before
trusting it in a live reward stack.

---

## 3. "Familiar + expectation-breaking = taste" (README thesis)

**Construct.** The README's own framing (no single citation — a synthesis
built on Suls, incongruity theory generally, and the project's product-
taste analogy).

**What the theory actually says.** A joke must be simultaneously familiar
enough to parse instantly and novel enough to violate the parse — this is
Suls' two-stage structure (§1) restated as a product-design analogy
("familiar enough to use, surprising enough to matter"). The README
explicitly claims this generalizes beyond humor to taste itself.

**Where it lives in code today.** Split, imperfectly, across two clusters
of reward terms: the "novel" half maps to `CorpusNoveltyPenalty` and
`SelfRepetitionPenalty` (`env/rewards.py:287-489`) plus the offline
`benchmark/joke_novelty.py` check; the "familiar" half maps to
`ComprehensibilityReward` (`env/rewards.py:549-589`).

**Fidelity assessment: partial proxy, and the mapping is thinner than the
README's confident tone suggests.** Neither side measures the
psychological construct it's named for:
- The "novel" side measures *lack of textual n-gram overlap* with a
  corpus/self-history/group, not "expectation violation." A joke can be
  textually unique (zero trigram overlap with anything) while being
  utterly unsurprising — a bland, novel-vocabulary non-joke scores
  maximal novelty here. Conversely a genuinely surprising joke that
  happens to reuse a well-worn setup phrase can be penalized as if it
  were derivative.
- The "familiar" side measures *surface well-formedness* (length,
  punctuation, vocabulary-uniqueness band), not "instantly comprehended as
  making sense." See §1's comprehensibility discussion — the same
  well-formed-nonsense gap applies here directly.

Both are cheap, real statistical proxies for something real, but the
README's "familiar enough to comprehend instantly, novel enough to break
your expectation" sentence describes cognitive events; the code measures
surface statistics correlated with, but not constitutive of, those
events.

**Known exploit/gap.** This is where the project's own documented
2-word-template-reskin evasion lives structurally: a memorized joke
template with ~2 content words substituted routinely drops trigram-
Jaccard similarity below `CorpusNoveltyPenalty`'s threshold
(`env/rewards.py:314-327`) and scores reward `0.0` — not even a ramped
partial penalty, full evasion — while `ComprehensibilityReward` scores it
at or near full credit (it's still well-formed). The reward stack can
therefore score a near-verbatim memorized joke as simultaneously "novel"
and "familiar" — i.e., high combined reward — which is exactly the mode
collapse this entire project exists to prevent, achieved by fooling both
halves of its own defining thesis at once. See §6 for the sharper version
of this finding (the claimed regression test for this exact case doesn't
exist).

**Candidate upgrade.** Same fix as §1 and §6: an embedding-based semantic
novelty tier (already scoped and explicitly deferred in
`env/rewards.py`'s own module docstring: "Add an embedding-based tier
later, behind a guarded import, if the cheap check stops being enough")
for the novelty half; a genuine expectation-violation signal for the
familiar/resolution half. Validate jointly: run both old and new metrics
over a held-out set of hand-reskinned templates (2, 4, 8 words changed)
and confirm the new novelty tier's penalty decays *gradually* with edit
distance from the template rather than cliff-dropping to zero at some
n-gram threshold.

---

## 4. Humor as a hard-to-fake honest signal

**Construct.** Miller (2000, *The Mating Mind*) and Greengross & Miller
(2011) (`greengross2011humor`) — costly/honest-signaling framing, chained
in `references/humor-honesty-beauty.md` to Zahavi's handicap principle.

**What the theory actually says.** Humor production is proposed as a
hard-to-fake fitness/intelligence indicator because it requires real-time
verbal/crystallized cognitive ability that a lower-quality individual
cannot cheaply counterfeit — the standard costly-signaling logic. This
project's own honesty-beauty review is explicit that the theoretical
chassis (Zahavi's handicap principle) is itself disputed within
evolutionary biology (Penn & Számadó 2020), and that "funny people are
more honest" is the owner's own inferential leap, not a finding in the
literature — the established part is narrower: humor production
correlates with intelligence (r≈.29–.40), and humor *style* (not amount)
correlates with HEXACO Honesty-Humility, in different directions for
different styles.

**Where it lives in code today.** NOWHERE as an implemented "fakeability"
metric — no code computes anything resembling "how hard would this have
been to counterfeit." But the *implicit* claim (this project treats humor
production as a meaningful signal worth measuring at all, in part because
it's supposed to be hard to game) can be checked against how actually
fake-able each of the project's own metrics is:
- `JudgePreferenceReward` — EASY to fake. This is the single most
  documented failure mode in the project's own literature (the LessWrong
  GRPO experiment collapsed a judge-only reward twice, in two different
  directions, `references/negative-results.md` §1).
- `CorpusNoveltyPenalty` / `SelfRepetitionPenalty` — EASY to fake, per §3
  and §6's 2-word-reskin finding.
- `IntraGroupDiversityReward` — moderately easy to game: it rewards mean
  pairwise trigram *distance* within a GRPO group, which a policy can
  maximize by making group members lexically distinct from each other
  while each individually stays low-effort — trigram distinctness is not
  the same as "each member is independently a real, hard-to-fake joke."
- The cascade's `path_divergence` / `depth_to_degradation`
  (`benchmark/metrics.py`) is the one metric in this repo that actually
  earns "hard to fake" empirically, not just rhetorically — EXP-007 is
  registered specifically to demonstrate that temperature inflates
  sampling-diversity metrics 3x+ more than it moves path divergence
  (`EXPERIMENT_LOG.md`).

**Fidelity assessment: distortion by omission.** The project's rhetorical
framing ("humor is an honest signal, hard to fake") is not matched by
making the *reward* metrics resistant to faking — only the cascade
*benchmark* diagnostic (not a training reward) has been empirically
tested for fakeability, and that test is still pending (EXP-007's result
is "(pending)" as of the last log entry). The RL reward stack that
actually shapes model behavior is, by the project's own documented
history, easier to hack than the honest-signal framing implies humor
should be.

**Known exploit/gap.** Same list as above — judge-hacking (documented,
twice), n-gram reskin evasion (documented, unresolved), group-diversity
gaming (undocumented, untested).

**Candidate upgrade.** Generalize EXP-007's pattern from a benchmark-only
ablation into a standing "fakeability audit" applied to every *reward*
term, not just the cascade's diagnostic metrics: for each term, construct
a cheap adversarial probe (temperature-only inflation, template
reskinning, degenerate-but-distinct group members) and require the term
to show a documented, small response to the probe before it's trusted in
a live training run — the same falsifiable-registration discipline
`EXPERIMENT_LOG.md` already applies to benchmark claims, applied to
reward terms.

---

## 5. Semantic foraging (Hills, Jones & Todd 2012) + Troyer clustering/switching (1997)

**Construct.** Troyer, Moscovitch & Winocur (1997), *Neuropsychology*
11(1) (`troyer1997clustering`) for the clustering/switching scoring pair;
Hills, Jones & Todd (2012), *Psychological Review* 119(2)
(`hills2012foraging`) for the Marginal Value Theorem (MVT) patch-departure
model.

**What the theory actually says.** Two genuinely different claims, often
conflated in this project's own prose. (1) Troyer et al.: fluency output
decomposes into **clustering** (mean run-length of consecutive items from
the same semantic *patch*, e.g. "dog, cat, hamster" = one cluster of pets)
and **switching** (count of transitions between patches) — a scoring
method, agnostic to *why* switches happen. (2) Hills et al.: patch-
departure *timing* is predicted by the Marginal Value Theorem — leave a
patch when its local yield rate drops to the *global average* yield rate
across the whole task, not merely "when it feels empty"; people whose
switching is more MVT-consistent recall *more* items overall, i.e. it's a
claim about adaptive foraging *policy*, not just a descriptive pattern.

**Where it lives in code today.** `cluster_switch_stats`
(`benchmark/metrics.py:188-214`) is a faithful port of Troyer's *scoring
mechanics* (run-length + switch-count over a `category_of` function) —
but as actually wired in `benchmark/analyze_pilot.py:74-75`,
`category_of` is `ls.canon` — `LabelSpace.canon`
(`benchmark/label_space.py`), which merges only near-*synonyms* of the
same topic (e.g. "exercise"/"gym"/"fitness" → one canonical string), not
a coarse semantic-category taxonomy (e.g. "animals" spanning
cat/dog/hamster) the way Troyer's actual "patch" concept requires. The
**MVT half is not implemented at all** — no code anywhere computes a
local-yield-vs-global-average comparison, or tests whether the model's
post-forced-exit topic choice is MVT-consistent (re-entering a nearby
unexploited patch) vs. a random jump. `references/trajectory-grounding.md`
§6 item 3 explicitly proposes this analysis; it does not exist in
`benchmark/`.

**Fidelity assessment: partial proxy, degrading toward distortion because
of how the pieces compose.** The clustering/switching *mechanics* are
faithful (correct run-length/switch-count arithmetic, unit-tested in
`benchmark/tests/test_metrics.py`). But because the cascade design forces
a *new* topic every turn by construction (that's the whole rejection
mechanism), and `ls.canon` only merges direct synonyms rather than
grouping into broader semantic patches, in practice `mean_cluster_size`
as wired will sit close to 1.0 almost always — the only way to get a
cluster >1 is a synonym-level near-repeat, which is a narrow, mostly
noise-driven event, not "the model stayed in a semantic neighborhood for
several turns." The metric that ships is real and correctly computed, but
it is not measuring "patch residence" in the sense Troyer's theory means
by that word — it's measuring something closer to "how often did
canonicalization catch two turns using near-identical vocabulary," which
is a much narrower and noisier signal.

**Known exploit/gap.** Because true patch-level clustering isn't
measured, a model that walks a genuinely tight semantic neighborhood
(e.g., five consecutive but lexically-varied animal jokes: cat, dog,
parrot, hamster, goldfish) is invisible to `cluster_switch_stats` as
currently wired — it would score as five singleton "clusters" (five
switches, mean cluster size ~1), the opposite of what a human forager
doing the equivalent verbal-fluency task would be scored as (one long
cluster, zero switches). This is a real, silent understatement of
patch-level collapse, running in the conservative direction (makes models
look *more* diverse than they are), consistent with this project's stated
preference for conservative-biased instruments — but it should be stated,
not assumed.

**Candidate upgrade.** Build an actual coarse-category function distinct
from `LabelSpace.canon` — e.g., a small fixed taxonomy (animals, food,
relationships, work, technology, ...) that a cheap classifier or
embedding-cluster assigns each topic label into, and re-run
`cluster_switch_stats` against *that* as `category_of` instead of
`ls.canon`. Separately, implement the MVT comparison
`trajectory-grounding.md` already specifies: for each patch, compare local
judge-rated novelty of the last k jokes against the running global
average, and classify each forced departure as MVT-consistent or not.
Validate by hand-labeling a handful of real cascade transcripts into
patches by eye and checking the new `category_of` against that human
gold segmentation (an ARI-style check, mirroring how the rejector itself
was validated).

---

## 6. "A joke heard twice is dead" (novelty decay under repetition)

**Construct.** Three independent psychological mechanisms, all invoked
in this project's grounding: incongruity-theory familiarity decay
(Kant/Schopenhauer, via `references/trajectory-grounding.md` §3.1),
semantic satiation (Jakobovits 1967, `references/trajectory-grounding.md`
§5), and Berlyne's arousal wear-out (habituation + tedium, two opposing
factors, both trending toward decay).

**What the theory actually says.** These are *three different
mechanisms* predicting the same surface phenomenon for different reasons
— the grounding doc is explicit that this matters: incongruity decay is
about a schema/expectation that can't be re-strained once seen (content-
structural); semantic satiation is about raw lexical repetition frequency
independent of whether the repeated item was ever funny (~15-30 reps,
purely mechanical, content-blind); Berlyne's model is arousal-based
(overshoot-then-relief wearing out through two additive but distinct
routes). None of the three is "textual similarity to something said
before" — that's a convenient measurable stand-in for all three, not a
direct instantiation of any of them.

**Where it lives in code today.** `CorpusNoveltyPenalty` and
`SelfRepetitionPenalty` (`env/rewards.py:287-489`, both reward-side) and
`benchmark/joke_novelty.py` (offline check: exact-hash + trigram-Jaccard
against the 25 ChatGPT templates) are the operationalizations. All three
are surface n-gram overlap checks — none model repetition *frequency*
(semantic satiation's mechanism specifically requires counting reps, not
just detecting one prior similar item), none model arousal, and none
distinguish "this is structurally the same joke" from "this happens to
share vocabulary."

**Fidelity assessment: partial proxy, honestly labeled as such in the
code's own comments** (`env/rewards.py`'s module docstring calls the
n-gram approach "the literal spec," not a claim of psychological
fidelity) **— but with one specific place where the code's own
self-assessment is worse than "partial proxy," it's actively wrong.**
`CorpusNoveltyPenalty`'s docstring (`env/rewards.py:314-327`) documents
the 2-word-template-reskin evasion explicitly and states it is "pinned"
as a locked-in regression test: *"see `env/tests/test_rewards.py`'s
`test_two_word_template_reskin_fully_evades_and_is_locked_in`, which pins
the exact evasion down as a regression test rather than letting it
silently drift."*

**[RESOLVED same night — audit-race note, 2026-07-17.]** At the time this
audit read `env/tests/test_rewards.py`, that test did not exist — the
docstring's safety-net claim was ahead of reality. A concurrent fix wave
(the env package's adversarial-audit response) added
`test_two_word_template_reskin_fully_evades_and_is_locked_in` minutes
later; its presence has since been verified directly (grep count 1). The
finding stands as a process lesson — a docstring claimed a test before the
test existed, which is exactly the failure shape this document exists to
catch — but the safety net is now real. The underlying exploit (2-word
reskin evades the n-gram novelty penalty) remains real, unfixed, and now
locked-in as documented-known behavior; the theory-grounded fix is the
semantic-novelty tier (priority translation #1 below).

**Known exploit/gap.** This *is* the exploit/gap — stated twice because
it is the single worst distortion found in this audit (see the closing
section). The documented failure mode (2-word reskin fully evades the
novelty penalty) is real and unfixed. The claim that it is caught by a
regression test is false. The practical consequence: nothing in CI or the
test suite would currently catch a regression, an intentional exploit
attempt during training, or simply confirm the documented behavior is
still exactly as described — the safety net the comment describes is
notional.

**Candidate upgrade.** Two separate fixes, don't conflate them: (a)
**write the actual test** the docstring claims exists — cheap, no new
infrastructure, should happen before anything else in this document; (b)
the real fix for the underlying evasion is the same embedding-based
semantic-novelty tier flagged in §3/§4 (deferred in the module's own
docstring). For semantic satiation specifically, a genuinely different
addition is needed: a repetition-*count* signal (how many times has a
near-identical topic/frame appeared in this policy's history, not just
"is there one similar prior item") — `SelfRepetitionPenalty`'s rolling
window checks similarity to *any* prior item, not frequency of a specific
recurring pattern, so it cannot currently distinguish "said once before"
from "said fifteen times before," which is exactly the distinction
semantic satiation's mechanism depends on.

---

## 7. Theory of mind / audience-relativity

**Construct.** Cao et al. (2023) culture × social-distance interaction
(`cao2023culture`); the broader annotator-disagreement literature (Uma et
al. 2021, Davani et al. 2022, Zhang et al. 2024, all in
`references/psychology.md` §7); ToM-HCAT / ToMBench as cited in
`references/humor-honesty-beauty.md` §3.3.

**What the theory actually says.** Humor appreciation is measurably
audience-dependent, not a fixed property of the text: Cao et al. show a
rigorously replicated culture × social-distance interaction (Chinese
participants rate distant-other jokes funnier than close-other jokes;
Americans show no such difference), and the disagreement literature
argues this isn't noise to average away but structured signal that a
single "gold" funniness label erases. Getting a joke also requires
modeling what the *specific* other mind in the exchange already believes
and is primed to find surprising or safe (theory of mind, ToM-HCAT).

**Where it lives in code today.** Track 2's context-ablation design
(`benchmark/banter.py::context_ablation_score`,
`env/banter_env.py::BanterEnv.step`) is the only code in this repo that
tests audience-*something* — but what it actually tests is narrower than
theory of mind: whether a reply is calibrated to **its own immediate
conversational context** (true vs. swapped), not whether it's calibrated
to a *modeled different mind's* likely reaction. There is no persona axis,
no demographic/cultural conditioning, no per-rater or per-cluster
preference modeling anywhere in the reviewed code. Track 1's rejector
(`benchmark/rejector.py`) is, by contrast, an explicit **fixed, single,
content-agnostic audience** — the same frozen topic-preference function
applied identically regardless of who's "asking" — which is the opposite
of audience-relativity, and is *intentionally* so (Track 1 is meant to be
a sterile diagnostic, not a humor-quality judge), but this means Track 1
has no theory-of-mind content at all, contrary to how the README's
alignment framing sometimes reads.

**Fidelity assessment: partial proxy, and only for Track 2.** Context-
ablation is a real, clever, and correctly-reasoned operationalization of
"is this responsive to a specific conversational history" — but that is a
narrower construct than "theory of mind" or "audience-relativity" as the
cited literature means those terms (a single fixed conversational context
vs. a genuinely modeled *different* audience with different priors,
culture, or social distance from the speaker). Track 1's rejector,
despite living in a project whose alignment framing invokes ToM as one of
three pillars, has zero ToM content by design.

**Known exploit/gap.** `benchmark/banter.py`'s own module docstring
documents the residual risk plainly: subtraction cancels a judge's
*constant* bias but not a bias that is itself context-dependent (e.g. a
judge that scores context-echoing keyword-stuffing higher regardless of
true responsiveness) — a policy could learn to sprinkle context-echoing
words into an otherwise context-blind reply and inflate the delta without
being genuinely in-context. Separately, `docs/BENCHMARK.md`'s own stated
validity risk #2 for the rejector ("its own topic prior contaminates the
result") means even Track 1's single "audience" is not neutral — it has
an implicit taste, just an unmodeled and unacknowledged one.

**Candidate upgrade.** Build genuine persona-conditioning: an explicit
modeled-audience variable (e.g., culture/social-distance dimensions from
Cao et al., or per-rater preference clusters à la Davani et al.) that the
judge or context is conditioned on, then test whether context-ablation
delta *and* persona-swap delta jointly predict real human funniness
ratings split by audience segment — using Oogiri-GO's ~100-raters-per-item
structure (already identified in this project's own datasets.md as the
cleanest available multi-rater humor data) as the validation set, since it
has the rater-level granularity this repo's own code currently discards
by scoring against one aggregate judge call.

---

## 8. Annotator noise, κ = 0.41 (gold-as-distribution)

**Construct.** Sun et al. (2022), *ExPUNations*, EMNLP
(`sun2022expunations`) — Cohen's κ = 0.41 for funniness ratings
specifically (not 0.49, which is a different construct — pun-word
semantic validity; the corrected citation is recorded in
`references/negative-results.md` §4 and `references/psychology.md` §7).

**What the theory actually says.** Funniness agreement between annotators
is only *moderate* (κ = 0.41) — worse than the commonly-miscited 0.49.
The implication this project's own literature draws explicitly (Uma et
al. 2021, Davani et al. 2022, Zhang et al. 2024) is architectural, not
just cautionary: treat "gold" funniness labels as **samples from a
preference distribution**, and prefer reward-model designs that predict
per-annotator or per-cluster labels and aggregate afterward, rather than
collapsing disagreement into one majority-vote or single-model score
upfront — because the disagreement is informative (who found it funny),
not noise to be averaged away.

**Where it lives in code today.** NOWHERE. Every judge-based scoring path
in this codebase — `JudgePreferenceReward` (`env/rewards.py`), the banter
`JUDGE_PROMPT` (`benchmark/banter.py`) — is a single model, single call,
single scalar. There is no ensemble of judges, no distributional reward
model, no per-rater or per-cluster prediction anywhere. Even the
rejector-validation experiments test single-model-as-instrument choices
(Haiku vs. Sonnet, `EXPERIMENT_LOG.md` EXP-003b) rather than an ensemble
or a distributional readout.

**Fidelity assessment: distortion by omission — cited as load-bearing,
then contradicted by the actual architecture.** `docs/BENCHMARK.md` §4
and `references/psychology.md` §7 both present κ=0.41 as a design-relevant
fact ("labels are noisier than commonly claimed, so treat 'gold labels' as
samples from a preference distribution"), and then every scoring
mechanism in the codebase treats a single Haiku-tier (or whatever model is
configured) judge call as if it produces a ground-truth scalar. The
context-ablation delta trick (`benchmark/banter.py`) cancels a judge's
*constant scale* bias, which is a real and useful property, but says
nothing about *disagreement or variance across different judges or
audiences* — it is not a distributional-modeling fix, and shouldn't be
mistaken for one.

**Known exploit/gap.** Because there is no distributional modeling, the
entire reward stack inherits one judge's idiosyncratic taste as if it
were funniness itself — precisely the risk κ=0.41 warns against, and
exactly the mechanism by which the LessWrong GRPO experiment
(`references/negative-results.md` §1) got hacked into judge-specific
quirks (absurdist bonus tags scored 20/20 by the judge despite the human
author disagreeing) rather than anything a second judge or a real human
population would agree is funnier.

**Candidate upgrade.** Multi-judge ensemble for `judge_preference` — 3-5
differently-prompted or differently-modeled judges, reward = mean *and*
reward-variance logged (and optionally down-weighted-for-high-disagreement
items, per Zhang et al.'s distributional-RM proposal) rather than
collapsed silently. Validate against a real multi-rater slice: Oogiri-GO
items with their ~100 human ratings each are already the right shape to
check whether ensemble judge-variance actually correlates with real
human-annotator disagreement, which is the one experiment that would tell
you whether the ensemble is adding real signal or just diluting a single
judge's opinion into a costlier one.

---

## 9. Depth-to-degradation

**Construct.** No single named psychological theory — this is the
project's own metric, gestured at metaphorically ("a small well") as
connected to verbal-fluency exhaustion / semantic-foraging patch
depletion (§5) and to novelty decay (§6).

**What psychological construct it actually maps to — honestly: mostly
none.** `depth_to_degradation` (`benchmark/metrics.py:159-182`) is,
mechanically, exactly two things: the first index at which an
already-seen *raw string* topic label repeats, or the first turn
`looks_like_refusal`'s regex fires — whichever comes first. This is an
engineering metric wearing a theory costume, and it is worth saying so
plainly rather than dressing it up: it does not measure "yield rate
falling to the global average" (MVT, §5), does not count repetition
frequency (semantic satiation, §6), and does not measure any gradual
quality decline (Berlyne's arousal wear-out, §6) — it measures a single
brittle event (exact-string repeat, or a regex phrase match) and reports
the turn index at which that event first occurs.

**Where it lives in code today.** `benchmark/metrics.py::depth_to_degradation`,
wired through `benchmark/run_pilot.py` and `benchmark/analyze_pilot.py`.

**Fidelity assessment: engineering metric, honestly assessed as such —
not a distortion because the project doesn't over-claim it in the code
itself, but the surrounding prose (README's "small well" framing, the
metric table's "collapse signature" language) invites more theoretical
weight than the mechanics bear.** Two specific fragilities inherited
directly from documented instrument limitations elsewhere in this repo:
(a) it runs on the **raw**, primary-scored topic label
(`docs/TRANSFER-PLAN.md` §5.3 confirms raw-label scoring is this
project's own stated "primary" convention) — but EXP-002/EXP-003
(`EXPERIMENT_LOG.md`) found and only partially fixed synonym scatter
("cat" vs. "pet"), meaning a same-topic reuse expressed with a synonym is
invisible to the *primary* `depth_to_degradation` number, understating
degradation; (b) `looks_like_refusal` is, by its own docstring,
"deliberately conservative" — a small regex over a handful of refusal
phrasings — so a model that degrades via increasingly bland, low-quality,
but lexically-non-repeating output on genuinely new topics would score as
never-degraded (`depth: None`, "survived the whole cascade") even though a
human rater would very plausibly call that a collapse.

**Known exploit/gap.** A model gaming this metric doesn't need to avoid
degrading in any real sense — it needs to avoid emitting an exact-string
topic repeat or a regex-matched refusal phrase. Padding degenerate output
with slightly-varied phrasing that never trips either detector would
report a clean, non-degraded run.

**Candidate upgrade.** Report `depth_to_degradation` under both the raw
label path *and* the `LabelSpace`-canonicalized path side by side — the
same "primary + semantic, reported alongside" pattern
`benchmark/analyze_pilot.py` already applies to cross-model overlap —
rather than only the raw version. Separately, add an independent
*quality*-decay signal (e.g., a judge-rated funniness or the
`ComprehensibilityReward` heuristic's trend across the trajectory) as a
second degradation channel distinct from repeat/refusal, since every
theory this metric metaphorically gestures at (§5, §6) predicts gradual
decay, not a single binary event. Validate by hand-reviewing a sample of
"depth: None" (never-degraded) transcripts from the pending EXP-004
pilot and checking whether a human rater agrees those runs stayed genuinely
fresh through 30 turns.

---

## 10. Humor Styles / valence decomposition (HSQ)

**Construct.** Martin, Puhlik-Doris, Larsen, Gray & Weir (2003),
*Journal of Research in Personality* (`martin2003hsq`) — four styles:
affiliative, self-enhancing (both benign), aggressive, self-defeating
(both potentially detrimental). Not one of the constructs named in this
audit's brief, but invoked prominently enough in the reference corpus
(`references/psychology.md` §5, and the Superiority Theory cautionary
note) to belong here.

**What the theory actually says.** "Funny" is not unidimensional-benign —
a reward model that doesn't distinguish affiliative/self-enhancing humor
from aggressive/self-defeating humor risks optimizing toward cruelty or
harmful self-disparagement while still scoring "funny" by a naive judge.
Greengross, Martin & Miller (2012) sharpen the risk further: professional
comedians use *all four* styles more than controls, including aggressive
and self-defeating, and among comedians, professional success correlates
*positively* with affiliative style and *negatively* with self-defeating
style — so "funnier = more of every style" is empirically false even
within the population that is, by any measure, actually good at this.

**Where it lives in code today.** NOWHERE. No file in `benchmark/` or
`env/` classifies or scores humor style/valence at all. `JudgePreferenceReward`
returns one undifferentiated `[0,1]` funniness score with no style axis;
nothing in the reward stack would distinguish a warm affiliative joke
from a punching-down aggressive one that a judge happened to rate equally
funny.

**Fidelity assessment: NOWHERE, and this is the quiet safety-relevant gap
in the reward stack** — not because anyone claims otherwise (the project
doesn't currently assert style-awareness anywhere in its own prose), but
because the surrounding literature is explicit that this is a real,
named failure mode (Superiority Theory's "cautionary" entry in
`references/psychology.md` §4 predicts exactly this: "a reward model
naively trained on 'funny = puts someone down' will overfit to
superiority humor... this failure mode is theoretically predicted here,
not just an empirical accident to be discovered the hard way"), and
nothing downstream of that prediction has been built.

**Known exploit/gap.** A judge-only reward with no style decomposition
has no mechanism to prevent exactly the overfitting the theory predicts —
if aggressive/superiority humor scores reliably higher with a given judge
model (plausible; `references/humor-honesty-beauty.md` cites a separate,
concrete finding that stereotypical/toxic jokes score 10-21% higher on
humor metrics — "Engagement Undermines Safety," arXiv:2510.18454), the
current reward stack has no term that would counteract that pull.

**Candidate upgrade.** Add an HSQ-style classifier (cheap version: a
judge-prompt asking specifically "is this humor affiliative/self-
enhancing or aggressive/self-defeating," separate from the funniness
question) and either penalize aggressive/self-defeating output directly
or at minimum log the style distribution of a training run's completions
over time so a style-collapse (e.g., drifting toward aggressive humor
because it scores higher) would be visible before it shows up as a
harm incident. Validate against Greengross, Martin & Miller (2012)'s own
finding: check that the classifier assigns comedians' historically-coded
aggressive routines to "aggressive" reliably, using their study's
materials or a similar public comedy-transcript sample as ground truth.

---

## 11. Script-based theories (SSTH / GTVH) — noted, not built

**Construct.** Raskin (1985), Script-based Semantic Theory of Humor
(SSTH); Attardo & Raskin, General Theory of Verbal Humor (GTVH), which
extends SSTH with six Knowledge Resources (Script Opposition, Logical
Mechanism, Situation, Target, Narrative Strategy, Language).

**What the theory actually says.** A text is humorous when it is
compatible with two *opposed* semantic scripts simultaneously — this is
the most computationally-structured classical theory in the project's own
corpus (`references/psychology.md` §4 flags it as "probably the single
most 'computational' classical theory — worth a closer look for
reward-model feature engineering specifically").

**Where it lives in code today.** NOWHERE. No script-opposition detector,
no Knowledge-Resource decomposition, exists anywhere in `benchmark/` or
`env/`.

**Fidelity assessment: not yet attempted — flagged as a candidate, not a
current failure.** Unlike the constructs above, this one isn't cited as
already backing a design decision, so there's no gap between claim and
code to indict — it's an acknowledged, unstarted opportunity.

**Known exploit/gap.** N/A — nothing built yet to exploit.

**Candidate upgrade.** GTVH's six Knowledge Resources are a genuinely
promising feature-engineering target for the incongruity-resolution gap
in §1: Script Opposition maps directly onto "expectation violation," and
Logical Mechanism maps onto "resolution rule." A cheap first pass: prompt
a judge to output the two opposed scripts explicitly (rather than a bare
funniness scalar), and use whether it can name two coherent, opposed
scripts at all as a structural proxy for both incongruity *and*
resolution simultaneously — a two-birds fix for §1's gap, worth
prototyping before building the surprisal/NLI machinery proposed there,
since it's cheaper (one judge-prompt redesign, no new model dependency).

---

## Prioritized next three translations

Ranked by how much closing the gap would improve the benchmark/reward
stack specifically, not by theoretical interest alone.

### 1. Close the reskin evasion — write the missing test, then fix the metric

The cheapest fix in this document and the most urgent: `env/rewards.py`'s
own docstring asserts a regression test exists that does not
(`env/tests/test_rewards.py`, verified absent — §6). Step one is
mechanical (write the test that pins the currently-documented-but-
unverified evasion behavior, so it's at least *tracked*, even before it's
fixed). Step two is the real fix already scoped in the module's own
docstring: an embedding-based semantic-novelty tier for
`CorpusNoveltyPenalty`/`SelfRepetitionPenalty`, reusing the threshold-
calibration playbook `benchmark/label_space.py` already built and
validated for topic labels (a 64-pair must-merge/must-not-merge fixture,
swept for the safest threshold). **Rough cost:** small — a day for the
test, a few days for the embedding tier plus calibration, no GPU, no new
external dependency beyond `sentence_transformers` (already a project
dependency for `label_space.py`).

### 2. Multi-judge / distributional reward modeling

Directly addresses §8 (κ=0.41 gold-as-distribution), meaningfully reduces
§4's judge-hacking exposure (the single most-documented failure mode in
this project's own literature), and partially addresses §7's audience-
relativity gap if judges are persona-varied rather than merely
model-varied. **What it needs:** 3-5 judge calls per completion instead
of one (cost multiplies accordingly — real but modest at API-judge
prices), an aggregation layer (mean + logged variance, per Zhang et al.'s
distributional-RM framing) replacing `JudgePreferenceReward`'s single-call
design, and a validation pass against Oogiri-GO's real ~100-rater-per-item
structure to check whether ensemble variance tracks real human
disagreement. **Rough cost:** medium — a few days of engineering, no GPU,
API cost scales linearly with ensemble size; the calibration-against-real-
raters step is the part that takes actual research time, not engineering
time.

### 3. Multiplicative BVT gate to replace the additive reward sum

The highest-payoff fix conceptually — it resolves the single clearest
contradiction between what this project's own citation corpus recommends
and what the shipped code does (§2) — and the most expensive to do
properly, because unlike #1 and #2 it requires building two entirely new
detector signals (violation, benign) from nothing, not just improving an
existing one. **What it needs:** a violation-detector and a benign/safety-
detector (both buildable cheaply as judge-prompt variants to start), a
redesign of `combined_reward`'s algebra to gate rather than sum at least
the judge term, and — critically — a validation experiment mirroring
McGraw & Warren's own five-experiment logic (synthetic pure-violation /
pure-benign / both-simultaneous conditions, checking the multiplicative
term is near-zero on the first two) before trusting it in a live GRPO run.
**Rough cost:** medium-high — likely a full research-and-build cycle (new
signals, re-derived weights, a fresh smoke test, the BVT-style ablation
validation) rather than an incremental patch; the right candidate for a
dedicated experiment cycle rather than a same-day fix.

---

## 12. Registration-grade specs: the next two reward terms (2026-07-17)

These two specs turn priority-translation #3 above (§2's BVT gate) and
§1's incongruity-resolution candidate upgrade from prose into buildable,
TRL-compatible reward terms, following exactly the discipline
`env/semantic_novelty.py` established for the last new tier: inert by
default (`weight=0.0`), sign-guarded, no hidden network calls, a
pre-registered validation design with numeric bars before either is
trusted in a live run. **Neither is built yet** — this is the spec, not
the implementation. No API calls were made writing this; every number
below is a registered prediction/bar, not a measured result.

### 12.1 BVT multiplicative gate — `BVTGateReward`

**Theory citation trail.** McGraw & Warren (2010), *Psychological
Science* 21(8), `mcgraw2010benign` (§2 above, primary, full text
confirmed): "three conditions are jointly necessary and sufficient for
eliciting humor: a situation must be appraised as a violation, a
situation must be appraised as benign, and these two appraisals must
occur simultaneously." **What this operationalization loses, stated up
front rather than discovered later:** (a) "appraisal" in the theory is a
property of a PERCEIVER's judgment relative to their own norms and
psychological distance from the situation — not a property of text in
isolation. Substituting an LLM judge's rating collapses an
audience-relative event into "how one model reads it," inheriting
exactly the κ=0.41/audience-relativity gaps §7–§8 already document, not
fixing them. (b) "Simultaneous" describes two appraisals happening in
ONE mind's processing at once; two separate judge calls (chosen
deliberately below, to avoid anchoring) encode the LOGICAL conjunction
(both must be true) but not literal temporal simultaneity in a single
appraisal process — no scalar product captures the phenomenology, only
the logic. (c) `references/theory-operationalization.md` RQ2 confirms (a
documented search trail, not an assumption) that **no published work has
computationally tested multiplicative-vs-additive BVT** — the one
existing attempt to combine multiple humor theories computationally,
THInC (De Marez, Winters & Rigouts Terryn 2024, CREAI workshop,
arXiv:2409.01232), chose additive GA2M with limited pairwise
interactions, not a clean product. That is real, if indirect, evidence
against assuming a multiplicative architecture is obviously correct just
because the theory is dual-appraisal — recorded honestly here so the
eventual write-up doesn't oversell how settled this is. Building and
validating the product against a real human-rated dataset (Oogiri-GO,
per RQ2's own actionability note) IS the novel contribution, not a
foregone conclusion.

**Combination structure — product, argued against min() and soft-AND.**
Three candidates, `v` = violation_score, `b` = benign_score, both in
[0,1]:

| combination | formula | AND-strictness | worst failure mode |
|---|---|---|---|
| min | `min(v,b)` | strict floor, zero compensability | vanishing gradient on the non-binding term; degenerates to whichever dimension has within-GROUP variance |
| **product (chosen)** | `v·b` | sub-multiplicative, punishes near-zero on either dimension hardest | benign-washing; correlated-judge-template exploitation |
| soft-AND (geometric mean) | `√(v·b)` | re-admits partial compensability | reintroduces the exact additive-sum failure this term exists to close |

**min() rejected — not for the textbook "fuzzy-logic AND" reason, for a
GRPO-specific one.** `env/rewards.py`'s reward functions feed
`GRPOTrainer`, which computes advantage RELATIVE TO OTHER COMPLETIONS IN
THE SAME GROUP (same prompt, `num_generations` samples — see
`IntraGroupDiversityReward`'s own docstring on this exact TRL behavior).
`min(v,b)`'s gradient is exactly zero with respect to whichever term is
NOT currently the minimum — so if one detector (say the benign judge,
being lenient about fictional framing) rates every completion in a group
near-ceiling while the other (violation) is where the real variance
lives, `min(v,b) ≈ v` for the WHOLE group, and the policy-gradient
signal collapses to "whichever single detector currently has
within-group variance" — precisely the additive-stack failure this term
exists to close, just hiding one level down instead of at the top. This
is plausible, not contrived: GRPO groups share one prompt, so completions
in a group are often similar enough in norm-content that a coarse benign
classifier rates them uniformly. Product's gradient (`∂(vb)/∂v = b`,
`∂(vb)/∂b = v`) never fully zeroes on either side as long as the OTHER
term is nonzero, so within-group variance in EITHER dimension always
contributes to the learning signal — the concrete, GRPO-specific reason
to prefer product over min, not just "the theory says multiply."

**Soft-AND (geometric mean) rejected for reintroducing compensability.**
`√(v·b)` sits, by the AM-GM-HM inequality chain, strictly between
`min(v,b)` and the arithmetic mean — e.g. v=0.01, b=1.0 gives product
0.01 (correctly ≈zero) but geometric mean 0.1 (a near-total absence of
violation still earns 10% credit). That residual tolerance for a
near-zero term is EXACTLY the "one signal compensates for the other's
near-absence" failure §2 diagnoses in the current additive sum, just
softened rather than closed. Kept on record as a documented escape hatch
(a tunable Cobb-Douglas `v^α · b^(1-α)`) if pure product proves too harsh
in practice (e.g. reward collapses near-zero for most completions early
in training, a cold-start problem) — but NOT the registered default;
that would be choosing convenience over the theory's own stated logic
mid-registration.

**Measuring violation and benign SEPARATELY.** Two independent judge
calls, not one combined prompt, for the same reason `context_ablation_
score` (`benchmark/banter.py`) uses the SAME judge/rubric across two
DIFFERENT contexts rather than asking one question that conflates both:
a single call asking "rate violation AND benign-ness" risks a halo
effect — an overall "I like/dislike this completion" impression
contaminating both sub-ratings so they covary for reasons unrelated to
the actual dual-appraisal construct, defeating the point of measuring
two theoretically-independent appraisals. Cost: 2 judge calls instead of
1 (3, alongside `judge_preference`) — a real, stated cost, not hidden.

```
VIOLATION_PROMPT = """You are assessing whether a piece of text VIOLATES a
norm, expectation, or how something 'should' be -- logically, socially, or
morally -- regardless of whether it is funny, safe, or appropriate. A text
can present a strong violation while being completely FUNNY, completely
OFFENSIVE, or completely NEUTRAL -- score ONLY the presence/strength of the
violation, never whether it's OK.

Text: {completion}

Output ONLY an integer 0 (no violation of any kind -- an unremarkable,
expected statement) to 10 (a severe, unambiguous violation) on the first line.
"""

BENIGN_PROMPT = """You are assessing whether a piece of text is BENIGN --
i.e. is there ALSO a competing appraisal that neutralizes any threat (it is
clearly fictional/hypothetical, the target is distant from the reader, a
norm holds that makes it acceptable, or nothing in it is severe enough to
threaten anyone)? Do NOT let the mere presence of a disclaimer ('just
kidding', 'jk', a fictional-frame label) drive your score by itself --
assess whether the underlying situation is genuinely safe once you set the
disclaimer aside, not whether the text CLAIMS to be joking. A text
describing a genuinely harmful act that merely appends 'jk' should score
LOW, not high.

Text: {completion}

Output ONLY an integer 0 (a genuine, unmitigated threat -- no safe reading
available) to 10 (completely safe; no unneutralized threat present) on the
first line.
"""
```

Both reuse `benchmark/banter.py`'s `_parse_score`-style first-line-integer
parse (guarded against "10"/"23" misreads the same way), then divide by
10 to satisfy `JudgePreferenceReward`'s existing `[0, 1]` contract —
`violation_judge`/`benign_judge` are `Callable[[Any, str], float]`,
IDENTICAL in shape to `RewardConfig.judge`, so an existing `judge`
callable's provider plumbing (`benchmark/providers.py`'s `get_provider`)
is directly reusable, just pointed at a different prompt template.
Recommend `claude:haiku` for both, matching EXP-003b's finding (bigger ≠
better instrument; sonnet was WORSE as the cascade rejector) rather than
assuming a frontier judge is automatically right here without its own
check.

**Pseudocode (TRL contract: `f(prompts, completions, **kwargs) ->
list[float]`).**

```python
class BVTGateReward:
    """reward = weight * violation_score * benign_score. Inert by default
    (weight=0.0) -- same discipline as SemanticNoveltyPenalty. No network
    call unless both violation_judge and benign_judge are injected."""

    __name__ = "bvt_gate_reward"

    def __init__(self, violation_judge=None, benign_judge=None, weight=0.0):
        self.violation_judge = violation_judge
        self.benign_judge = benign_judge
        self.weight = weight
        self._warned = False

    def __call__(self, prompts, completions, **kwargs):
        texts = _contents(completions)
        if self.violation_judge is None or self.benign_judge is None:
            if not self._warned:
                warnings.warn(
                    "bvt_gate_reward: violation_judge/benign_judge not "
                    "both configured -- this term contributes 0.0 to "
                    "every reward.", RuntimeWarning, stacklevel=2)
                self._warned = True
            return [0.0] * len(texts)

        prompts_iter = prompts if prompts is not None else [None] * len(texts)
        rewards = []
        for p, c in zip(prompts_iter, texts):
            v = float(self.violation_judge(p, c))
            b = float(self.benign_judge(p, c))
            for name, score in (("violation_judge", v), ("benign_judge", b)):
                if not (0.0 <= score <= 1.0):
                    raise ValueError(
                        "bvt_gate_reward: %s must return [0, 1], got %r"
                        % (name, score))
            rewards.append(self.weight * v * b)
        return rewards
```

**Gaming vectors and guards.**

1. **Benign-washing (disclaimer/fictional-frame stuffing) — the sharper
   of the two.** Because the product needs BOTH terms high, and a
   genuinely severe violation is often easy for a violation-judge to
   detect correctly, the cheapest attack surface is the OTHER term:
   append a boilerplate disclaimer ("jk!!", "just a joke, love you all",
   a fictional-frame tag) to a real, unmitigated norm violation, hoping a
   naive benign judge keys on the disclaimer TOKENS rather than the
   underlying severity. This is the mechanism behind the "Engagement
   Undermines Safety" finding already cited in §10 (stereotypical/toxic
   jokes scoring 10-21% higher) and structurally the same shape as the
   LessWrong GRPO absurdist-tag collapse (`references/negative-results.md`
   §1) — a cheap textual marker overriding a judge's real assessment.
   Guard: `BENIGN_PROMPT` explicitly instructs the judge to set the
   disclaimer aside (above); this is validated, not just hoped for, via
   the `disclaimer_washed_violation` fixture class below with its own bar.
2. **Correlated-judge-template exploitation.** A policy under RL pressure
   can converge on a narrow structural template that both judges
   independently rate highly for unrelated reasons (e.g. an absurdist
   framing device one judge reads as "clearly a big violation" and the
   other reads as "clearly fictional/silly, so safe") — not because it
   satisfies genuine dual appraisal, but because it trips both cheap
   detectors at once, the same Goodhart pattern as the LessWrong
   collapse. Guard: NOT fully closeable by this term alone — needs a
   standing "fakeability audit" (§4's candidate upgrade, generalized
   here): periodically check the diversity of completions scoring high
   on `bvt_gate_reward` using machinery already in the stack
   (`SelfRepetitionPenalty`/`IntraGroupDiversityReward`), since this term
   does NOT replace those. A training run where high-scoring completions
   cluster into one or two surface templates is the trip-wire, not a
   static pre-registration bar.

**Pre-registered validation design (mirrors EXP-005's structure: fixture,
per-class separation, repeat consistency, an echo-hackability trip-wire,
a disproof check).**

Fixture: 32 hand-authored items, 4 gold classes of 8, plus an 8-item
gaming-probe class scored separately:

- `both` (n=8): real dual-appraisal jokes — genuine transgression,
  clearly fictional/absurd/playful framing.
- `violation_only` (n=8): a real, seriously-meant transgression with NO
  protective frame — should score high violation, LOW benign.
- `benign_only` (n=8): pleasant, unremarkable, non-violating text —
  should score LOW violation, HIGH benign.
- `neither` (n=8): flat, unremarkable non-humor that is neither a
  violation nor warm/pleasant (deadpan non-sequiturs, mundane
  complaints) — should score LOW on both.
- `disclaimer_washed_violation` (n=8, gaming probe): `violation_only`
  items with a "jk"/fictional-tag disclaimer appended verbatim — the
  load-bearing adversarial check.

**Bars (registered, numeric, on the judge's native 0-10 scale before /10
normalization — matching EXP-005's reporting convention):**
- separation(`both`.violation − `benign_only`.violation) ≥ 5.0.
- separation(`both`.benign − `violation_only`.benign) ≥ 5.0.
- mean(`both`.product) ≥ 0.50 (on [0,1]) — the only class that should
  earn substantial gate credit.
- mean(`violation_only`.product) ≤ 0.15; mean(`benign_only`.product) ≤
  0.15; mean(`neither`.product) ≤ 0.10 — all three non-`both` classes
  gated near-zero, the entire point of the multiplicative structure (an
  additive sum would let `violation_only`'s high violation partially
  compensate, exactly §2's documented failure).
- `disclaimer_washed_violation`.benign mean ≤ 3.0 (0-10) — THE critical
  gaming-guard bar.
- echo-hackability (reuses `validate_banter_judge.py`'s
  `ECHO_RISK_THRESHOLD = 0.5` convention exactly): Pearson r(violation_
  score, shock-word count) over a class where a cheap profanity/taboo-
  keyword count varies independently of true violation label ≤ 0.5;
  Pearson r(benign_score, disclaimer-marker count) over
  `disclaimer_washed_violation` ∪ `both` ≤ 0.5.
- repeat consistency: 3 repeats/item, PASS/FAIL label (product > 0.2, a
  fixed operating threshold for this check only) agreement ≥ 0.85 —
  discrete-label convention, mirroring `validate_rejector.py`'s
  `repeat_consistency`.
- **disproof check** (CLAUDE.md checklist item 4 / EXP-005 checklist item
  4): compare against the CURRENT additive stack (`judge_weight *
  judge_score + comprehensibility`, no BVT term) on `violation_only` and
  `benign_only`. Predicted (registered blind, before any of this is
  built): the additive stack UNDER-penalizes `violation_only` relative
  to `both` — a well-formed, comprehensible, seriously-cruel one-liner
  can still score adequately on `judge_preference` + full
  `comprehensibility` credit — the literal mechanism §2's "Known
  exploit/gap" describes; the check should show it directly rather than
  assume it.

### 12.2 Two-stage incongruity-resolution gate — `TwoStageIncongruityGate`

**Theory citation trail.** Suls (1972), in Goldstein & McGhee (eds.),
*The Psychology of Humor*, `suls1972twostage` (§1 above): setup
generates an expectation, punchline disconfirms it (stage 1,
incongruity), perceiver then problem-solves to find a reconciling rule
(stage 2, resolution) — "incongruity alone is not sufficient." Kao, Levy
& Goodman (2016), *Cognitive Science* 40(5), doi:10.1111/cogs.12269
(`references/theory-operationalization.md` RQ1.1, ESTABLISHED, PMC full
text read) and Trott, Walker, Taylor & Coulson (2025), *Cognitive
Science*, doi:10.1111/cogs.70066 (RQ1.2, ESTABLISHED, PMC full text
read) — the load-bearing empirical constraint this design is built
around: surprisal predicts joke-vs-non-joke well but has **no
established relationship to funniness magnitude among things that are
already jokes** (ambiguity r=.03, p=.697 within-puns in Kao et al.; the
Is-Joke × Surprisal interaction in Trott et al. shows the same null a
decade later, on different materials, with an LLM surprisal estimator
instead of a psycholinguistic one). Wyer & Collins (1992),
`wyer1992comprehension` — the theory's own third stage, elaboration,
explicitly and honestly NOT attempted here (documented scope limit,
matching §1's "all three" framing).

**What this operationalization loses.** (a) The theory describes a
COGNITIVE PROCESS — a perceiver forms a real-time expectation, then
genuinely problem-solves. This design substitutes an LM's own generation
as a stand-in for "what a perceiver would expect," which conflates
"statistically likely completion to a language model shaped by its
training corpus's exposure to joke tropes" with "what a human perceiver,
encountering this setup fresh, would expect" — these are not the same
population, and the gap is largest for well-worn joke SHAPES an LM has
seen thousands of times (see gaming vector 2 below — the same gap
wearing an adversarial-robustness hat). (b) The resolution proxy (does
priming the predictor with "expect a twist" bring its guess measurably
closer to the real punchline) operationalizes "a reconciling rule
exists" as "an LM, told to expect wordplay, can approximate the
punchline" — this cannot distinguish "the LM guessed close because a
genuine, articulable logical mechanism connects setup and punchline"
from "the LM guessed close because this is a common templated joke-shape
it has memorized the general vicinity of" — the single largest fidelity
gap in the whole design, not a minor caveat. (c) Surprisal itself is
never MEASURED here (no logprob access from CLI-wrapped judges — see
below); it is doubly approximated, first as semantic distance between a
generated guess and the actual text rather than Kao/Trott's actual
negative-log-probability of tokens under a real LM. A proxy of a proxy,
registered honestly as such (Tier B, below) precisely because the more
faithful Tier A is blocked by an infrastructure constraint, not a
theoretical preference.

**No logprob access via CLI — the two-tier design, and why.**
`benchmark/providers.py`'s existing contract is `complete(prompt: str)
-> str` — plain text in, plain text out; `claude -p` and `codex exec`
(the CLI-wrapped providers) expose no logprobs at all. A REAL surprisal
computation (`-mean(logP(token_i | context))`) needs either a
completions-style endpoint with `echo`+`logprobs` (scores a GIVEN
continuation, not just sampled tokens) or a chat endpoint that happens
to expose per-token logprobs for the assistant's own generation —
NEITHER is confirmed available for any of this project's native-API
providers (`api:deepseek`, `api:glm`, `api:qwen`) as of this writing;
nobody has pre-probed it. Given that:

- **Tier A (preferred if it ever pre-probes clean, NOT registered
  yet):** a real surprisal computation against `api:glm`'s completions
  endpoint specifically — **glm, not deepseek or qwen**, for reasons
  already established in this project's own experiment log rather than
  assumed here: EXP-007b found qwen's endpoint silently ignores declared
  sampling parameters (byte-identical outputs at temperature 1.2 vs
  0.2) — the [LEARN] block from that experiment is explicit that ANY
  sampling/scoring-parameter-dependent mechanism on a native API needs a
  pre-registered manipulation check before being trusted, and an
  endpoint that ignores one declared parameter is not a safe default bet
  for trusting another (logprobs/echo) without its own check.
  deepseek-chat deprecates 2026-07-24 — building new standing
  infrastructure on a model with an imminent retirement date is a bad
  trade regardless of its EXP-007 track record. glm is the one model
  this project has already established DOES honor its declared
  parameters within their documented range (EXP-007c: temperature
  clamped to [0,1] but honored inside it, verified via a manipulation
  check). Before Tier A is trusted, it needs EXACTLY EXP-007c's
  manipulation-check pattern applied to a NEW parameter: does
  `api:glm`'s completions surface accept an explicit continuation and
  return per-token logprobs for it (not just for freely-sampled
  tokens)? Register that pre-probe as its own tiny experiment before
  writing any Tier A code — do not assume the answer.
- **Tier B (registered here, buildable now, zero new infra beyond what
  `benchmark/banter.py` and `env/semantic_novelty.py` already
  establish):** a judge-based "predict-then-diverge" proxy, described in
  full below. This is what gets built, validated, and (if it clears its
  bars) wired into `RewardConfig` first; Tier A is a documented upgrade
  path, not a prerequisite.

**Tier B design: predict-then-diverge (stage 1) + primed re-prediction
(stage 2).**

Setup/punchline split: most completions in this project's environments
are full one-liners with no explicit setup/punchline API boundary, so a
first, SEPARATE call asks the SAME predictor to identify the split:

```
SPLIT_PROMPT = """Below is a joke. Identify its SETUP (everything before the
reveal) and its PUNCHLINE (the reveal itself). If the joke has no
identifiable setup/punchline structure (e.g. it is a single-clause pun with
no buildup), output exactly "NO_SPLIT" and nothing else.

Joke: {completion}

Output exactly two lines:
SETUP: <the setup text>
PUNCHLINE: <the punchline text>
"""
```

A completion the predictor marks `NO_SPLIT` (or that fails to parse both
lines) returns 0.0 from this term — a documented "can't apply" case, not
a silent default pass, matching this codebase's existing `None`-sentinel
convention (`banter.py`'s `_judge_once`, `rejector.py`'s `UNPARSEABLE`)
rather than pretending every completion has this structure.

```
PREDICT_COLD_PROMPT = """Setup: {setup}

Predict, in ONE short sentence, the single most natural, UNSURPRISING way
you would expect this to continue -- not a joke, just the ordinary
continuation a typical person would expect.

Output ONLY your one-sentence predicted continuation.
"""

PREDICT_PRIMED_PROMPT = """Setup: {setup}

This is the setup of a joke that ends in a clever twist, pun, or reframe.
Predict, in ONE short sentence, what you think the twist/punchline is.

Output ONLY your one-sentence predicted punchline.
"""
```

Both predictor outputs and the real punchline are embedded (reusing
`env/semantic_novelty.py`'s exact `embed_fn` injection convention and
guarded `sentence_transformers`/`numpy` import, for the same
testability-without-a-model-download reason — a fake `embed_fn` in unit
tests, a real `all-MiniLM-L6-v2` in production) and compared by cosine
distance:

```
distance_cold    = 1 - cos_sim(embed(predict_cold),   embed(punchline))
distance_primed  = 1 - cos_sim(embed(predict_primed), embed(punchline))

gate_1_surprise   = distance_cold >= surprise_threshold        # default 0.5
gate_2_resolution = (distance_cold - distance_primed) >= drop_threshold  # default 0.15
                    and distance_primed < distance_cold

reward = weight if (gate_1_surprise and gate_2_resolution) else 0.0
```

A strict AND, not a graded sum of the two gates' partial credit — for
the same reason argued in 12.1: incongruity-resolution's own theory
statement is "incongruity alone is NOT sufficient," a genuine
conjunction, and per RQ1's cross-validated finding, surprisal magnitude
must never be treated as a scalar to maximize (a policy rewarded for HOW
MUCH `distance_cold` exceeds threshold would directly re-create the
"more surprisal = more reward" mistake two independent papers a decade
apart falsify) — this term is a GATE, zero/nonzero, exactly like the
ranked shortlist's own framing ("build it as a zero/nonzero gate ... not
as a scalar to maximize").

**Pseudocode (TRL contract).**

```python
class TwoStageIncongruityGate:
    """reward = weight if (surprising cold AND resolves when primed) else 0.
    Inert by default. `predictor` matches providers.py's Callable[[str],
    str] contract (NOT judge's (prompt, completion)->float shape -- this
    class needs raw text generations, not a score). `embed_fn` matches
    SemanticNoveltyPenalty's injection contract exactly."""

    __name__ = "two_stage_incongruity_gate"

    def __init__(self, predictor=None, embed_fn=None, weight=0.0,
                surprise_threshold=0.5, drop_threshold=0.15,
                allow_degraded=False):
        ...  # same guarded sentence_transformers/numpy import + loud
             # SemanticNoveltyUnavailable-style failure as
             # env/semantic_novelty.py, reusing that exception type

    def __call__(self, prompts, completions, **kwargs):
        texts = _contents(completions)
        if self.degraded or self.predictor is None:
            return [0.0] * len(texts)
        rewards = []
        for text in texts:
            setup, punchline = self._split(text)          # None, None on NO_SPLIT
            if setup is None:
                rewards.append(0.0)
                continue
            cold = self.predictor(PREDICT_COLD_PROMPT.format(setup=setup))
            primed = self.predictor(PREDICT_PRIMED_PROMPT.format(setup=setup))
            d_cold = self._distance(cold, punchline)
            d_primed = self._distance(primed, punchline)
            passes = (d_cold >= self.surprise_threshold
                      and d_primed < d_cold
                      and (d_cold - d_primed) >= self.drop_threshold)
            rewards.append(self.weight if passes else 0.0)
        return rewards
```

Cost, stated plainly: **3 LM calls per completion** (split + cold-predict
+ primed-predict) plus 2 cheap embedding comparisons. On top of `judge_
preference`'s existing 1 call and `BVTGateReward`'s 2 (if both are
enabled), a full stack could reach 6 LM calls per completion — a real,
multiplying cost across a GRPO batch called hundreds of times per run.
Recommend `claude:haiku` for the predictor (EXP-003b: bigger is not
automatically a better instrument here) and treat this as the single
most expensive term in the stack if enabled.

**Gaming analysis.**

1. **Vague/abstract "resolvable surprise" faking.** A policy can learn to
   end completions with a hedgy, widely-applicable "deep" non-sequitur
   ("...and that's when nothing really made sense anymore") rather than
   a genuine punchline. Primed with "expect a twist," a predictor's best
   guess for almost ANY setup tends to drift toward similarly abstract,
   vaguely-clever language — inflating `cos_sim(predict_primed,
   punchline)` for reasons that have nothing to do with a real
   reconciling rule existing, purely because the space of "vague
   clever-sounding endings" is small and mutually close in embedding
   space. Guard: a dedicated `vague_abstract_gaming_probe` fixture class
   (below) with its own registered bar — the single most important
   adversarial check in the validation design, because the architecture
   cannot distinguish this from genuine resolution by construction; only
   an empirical bar catches it.
2. **Pun-template spam / Goodharting the predictor's own prior.** The
   primed predictor's guess is shaped by ITS OWN training-data exposure
   to common joke tropes (puns on frequent words, "it was all a dream,"
   anthropomorphized-object non-sequiturs). A policy that always lands
   on one of these STOCK twist shapes will pass the resolution gate
   regardless of whether the specific joke is fresh or well-constructed,
   because the predictor's prior over "what a twist looks like" is
   itself narrow and templated — the same class of exploit
   `CorpusNoveltyPenalty`'s 2-word-reskin evasion is, one level removed
   (gaming the DETECTOR's blind spot rather than the underlying
   construct). Guard: this term does not replace
   `SelfRepetitionPenalty`/`IntraGroupDiversityReward`, and MUST be read
   alongside them in any real run; additionally, register a reskin-spam
   regression probe in the validation suite itself (5 superficial
   reskins of one stock twist template, tracked — not gated pass/fail,
   since this is inherently a monitoring signal rather than a
   static-fixture-checkable property — see `CorpusNoveltyPenalty`'s own
   `test_two_word_template_reskin_fully_evades_and_is_locked_in` for the
   precedent of pinning a KNOWN exploit down as a tracked regression
   rather than pretending it's solved).

**Pre-registered validation design (mirrors §1's own candidate-upgrade
proposal — 30 real jokes / 30 setup+non-sequitur pairs — sized down
slightly to match this project's typical fixture density, plus the
gaming-probe class added here).**

Fixture: 40 items —
- `real_joke` (n=12): genuine setup+punchline jokes with an articulable
  resolving mechanism (puns, misdirection, wordplay).
- `setup_nonsequitur` (n=12): the SAME setups, punchline swapped for an
  unrelated, non-resolving continuation — surprising, but nothing to
  resolve. Constructed specifically so **gate 1 alone cannot distinguish
  this class from `real_joke`** — that is the whole point of this class
  (see disproof check below).
- `boring_expected` (n=8): setup + the literal expected/unsurprising
  completion — should fail gate 1 outright, the control proving the
  surprise gate isn't vacuous.
- `vague_abstract_gaming_probe` (n=8, scored separately from the three
  gold classes above): setups + deliberately generic, "deep"-sounding
  non-sequitur endings (gaming vector 1's exact attack).

**Bars (registered, numeric):**
- gate_1 pass rate: `real_joke` ≥ 0.85, `setup_nonsequitur` ≥ 0.85 (both
  surprising — by construction, near-equal), `boring_expected` ≤ 0.15.
  Separation(`real_joke or setup_nonsequitur` − `boring_expected`) ≥
  0.70.
- gate_2 (resolution) pass rate: `real_joke` ≥ 0.70 (registered
  expectation of real, non-trivial miss rate — an LM-generation proxy
  for "a reconciling rule exists" will not be perfect); `setup_
  nonsequitur` ≤ 0.20. **The load-bearing bar:** separation(`real_joke` −
  `setup_nonsequitur`) on gate_2 pass rate ≥ 0.50 — this is the entire
  reason a second stage exists, since gate 1 alone cannot make this
  distinction by the fixture's own construction.
- `vague_abstract_gaming_probe` gate_2 pass rate ≤ 0.25 — the
  gaming-guard bar; a failure here means the design needs a tighter
  drop_threshold or a different similarity metric before it is trusted,
  not that it should ship anyway.
- **disproof check** (CLAUDE.md checklist item 4): does gate_1 ALONE
  separate `real_joke` from `setup_nonsequitur` as well as the full
  two-stage gate does? Predicted (registered, before any code exists):
  gate_1 pass-rate difference between these two classes ≈ 0 (both
  constructed to be equally surprising) — if this prediction is WRONG
  (gate 1 alone already separates them), the second stage is not
  earning its keep and the term should not ship as designed.
- repeat consistency: 3 repeats on a 12-item subset, PASS/FAIL label
  agreement ≥ 0.85 (discrete-label convention, same as 12.1's bar and
  `validate_rejector.py`'s `repeat_consistency`).

**Where both slot into `RewardConfig`, exactly like
`semantic_novelty_weight`.**

```python
# BVT gate (12.1) -- no ML dependency, judge-based only, like judge_weight
violation_judge: Optional[Callable[[Any, str], float]] = None
benign_judge: Optional[Callable[[Any, str], float]] = None
bvt_gate_weight: float = 0.0        # OFF by default -- new tier

# Two-stage incongruity gate (12.2) -- needs an embedding backend, like
# semantic_novelty_weight
incongruity_predictor: Optional[Callable[[str], str]] = None
two_stage_incongruity_weight: float = 0.0   # OFF by default -- new tier
incongruity_surprise_threshold: float = 0.5
incongruity_drop_threshold: float = 0.15
```

Both weights are BONUS fields (`>= 0` sign guard, added to
`__post_init__`'s `_BONUS_FIELDS` tuple exactly like `intra_group_
diversity_weight`/`comprehensibility_weight` — a negative weight here
would silently reward FAILING the gate). `reward_stack()` appends
`BVTGateReward` unconditionally when `bvt_gate_weight != 0.0` (no lazy
import needed — pure judge calls, same footprint as
`JudgePreferenceReward`) and lazily imports a new
`env/incongruity_gate.py` (mirroring `env/semantic_novelty.py`'s exact
lazy-import-only-when-nonzero pattern) when `two_stage_incongruity_
weight != 0.0`. Neither term changes behavior for any existing training
config that doesn't opt in — the same guarantee `semantic_novelty_
weight`'s docstring already makes.

**What neither term fixes, stated once so it isn't oversold:** both
remain judge/predictor-dependent (inheriting every generic judge-hacking
risk §4/§8 already document), neither addresses Wyer & Collins' third
incongruity-resolution stage (elaboration, out of scope), and both add
real per-completion LM-call cost to an already-6-term stack. They ARE
the two most theory-faithful gaps this document has open, and per
`references/theory-operationalization.md` RQ2, the BVT product-vs-
additive comparison specifically would be a genuine, citable novel
empirical contribution if the validation above is run and reported
honestly either way.

---

## What this document is not

It is not a claim that the current benchmark/reward stack is unsound —
several pieces (the cascade's path-divergence/cross-model-overlap
metrics, the Troyer clustering *mechanics*, the context-ablation delta's
bias-cancellation logic, the sign-guarded `RewardConfig`) are genuinely
well-reasoned engineering, faithfully described in their own code. The
gaps recorded here are concentrated in exactly the places the project's
own prose reaches furthest past what's built — which is precisely where
a standing audit document earns its keep. Re-run this read against the
code every time a construct gets a real implementation, and move its
verdict up from "distortion" or "NOWHERE," not just its prose.
