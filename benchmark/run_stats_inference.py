"""Statistical inference driver for the 2026-07-17 cascade pilot.

benchmark/stats.py is the general-purpose inferential toolkit (bootstrap
CIs, exact permutation tests, Cliff's delta). This module is the specific
ANALYSIS PLAN for the cascade pilot's two data files
(experiment-runs/2026-07-17-cascade-pilot/{analysis.json,novelty.json}):
it reconstructs the exact raw per-run topic paths that produced
analysis.json (via analyze_pilot.load_lanes, so the reconstruction is
provably the same code path, not a re-implementation that could silently
diverge), verifies the reconstruction against the frozen analysis.json
numbers, then runs stats.py's machinery over it plus two small,
purpose-built additions stats.py does not provide:

  1. permutation_test_mean_diff — a generic exact/Monte-Carlo permutation
     test for a difference-of-means statistic between two scalar samples.
     stats.permutation_test_divergence hardcodes "mean pairwise set_jaccard
     within a group of paths" as its statistic; degradation DEPTH is a
     scalar per run, not a path, so it needs its own (structurally
     identical: same _EXACT_LIMIT exact/Monte-Carlo split, same
     add-one-corrected p-value out of the exact regime) permutation test.
  2. wilson_ci / fisher_exact_two_sided — memorization rate (novelty.json)
     is a single aggregate hit-rate per model (exact_corpus_hits / n_jokes),
     not a per-run scalar list, so bootstrap_ci/cliffs_delta don't apply.
     Binomial proportion CIs and two-proportion exact tests are the
     standard tools for that shape and are not in stats.py.
  3. holm_correction / benjamini_hochberg_correction — stats.py deliberately
     does not multiple-comparison-correct (each of its functions returns
     one test's result; correcting across a battery of them is an analysis
     choice, not a toolkit concern). Both are implemented for transparency;
     HOLM is used as the primary correction throughout this driver (FWER
     control, no independence assumption, uniformly at least as powerful
     as Bonferroni) per EXPERIMENT_LOG.md's methodology rule to report
     which correction was used.
  4. pooled_frequency_null_diagnostics / build_lane_manifest — added after
     adversarial review (2026-07-17): the former recomputes
     stats.cross_model_null's pooled_frequency_baseline inner loop (same
     rng stream) to expose the null distribution's min/max/vocabulary size,
     which stats.cross_model_null's return value does not carry, so the
     prose numbers in FINDINGS.md are independently verifiable from
     stats_inference.json rather than computed off-pipeline. The latter
     records the exact summary.json paths (+ mtimes) load_lanes() consumed,
     so the frozen-snapshot claim ("this analysis predates the fill lanes")
     is a stated fact in the output, not an assertion in prose.
  5. load_raw_paths_and_refusals_from_turns / verify_turns_reconstruction /
     depth_to_degradation_meta_excluded — added for the 2026-07-17 midday
     hostile-review fix wave (EXPERIMENT_LOG.md "EXP-004 red-team
     corrections"). The review found meta-register labels (`comedy`,
     `joke`, `humor`, `ai`, `software`, `laughter` — models joking about
     joke-telling under rejection pressure) mediate most Anthropic
     degradation events, and asked for a robustness row that recomputes
     the family contrasts with those labels treated as never-matching for
     repeat detection, RE-DERIVED FROM RAW LANE JSONLs (turns-*.jsonl),
     not by post-hoc filtering analysis.json's already-computed depths.
     `load_raw_paths_and_refusals_from_turns` rebuilds per-run topic paths
     and refusal-turn lists directly from every lane's turns-*.jsonl,
     mirroring run_pilot.py's own per-lane ">=2 successful runs" inclusion
     gate exactly (reading each lane's summary.json only for which run_ids
     it recorded as failures, never for path CONTENTS) so the roster this
     driver's meta-excluded and decomposition sections analyze is the same
     N as analysis.json's, reconstructed independently.
     `verify_turns_reconstruction` cross-checks that reconstruction's RAW
     (non-meta-excluded) depths against analysis.json before any
     meta-excluded number is trusted — same trust-but-verify discipline as
     `verify_reconstruction` above, one level deeper (raw turns instead of
     summary.json's cached path lists).
  6. section_incidence_fisher_companion — a censoring-free companion to
     section_degradation_battery's mean-depth contrasts: a run either
     degraded or it didn't, no depth=30 censoring convention involved.
     Fisher exact on the 2x2 (family x degraded/survived) table, for both
     the raw roster and the meta-excluded roster.
  7. section_dual_tier_memorization / framing_prefix_rate_by_model — folds
     in novelty.json's template-trigram tier (already computed by
     benchmark/joke_novelty.py, previously unreported alongside the
     exact-match tier) and an independently-computed style-confound check:
     the exact-match tier is a whole-string hash match, so a model that
     habitually prefixes its jokes with framing prose ("Alright, ...")
     defeats it structurally, not substantively. framing_prefix_rate_by_model
     scores every model's raw joke text against a simple leading-phrase
     regex, straight from turns-*.jsonl (same universe joke_novelty.py's
     n_jokes counts over -- every turn, including failed/excluded runs'
     partial output, exactly matching how novelty.json's own n_jokes were
     already counted).
  8. section_degradation_event_decomposition — classifies each run's FIRST
     degradation event (if any) as topic-repeat (a genuine non-meta topic
     recurrence), meta-register-repeat (the repeated label is one of the
     six meta-register labels above), or refusal, per model, from the same
     raw-turns reconstruction used in (5).

IMPORTANT — disclosure about the Anthropic-vs-OpenAI family contrast in
section_degradation_battery: it is a POST-HOC single contrast. It was not
written into EXP-004's pre-registration; it was chosen after the pilot's
results made the family-level degradation pattern visible. It is reported
as the single best-supported contrast for that pattern (one test, so no
multiple-comparisons correction burden, unlike the C(n_models,2)-pair
exploratory matrix computed alongside it) — NOT as pre-registered confirmatory
evidence. A genuinely pre-registered replication is required before
"confirmatory" language is earned. See FINDINGS.md §2.1/§4.3 for the
corresponding language fix (adversarial review finding B2, 2026-07-17).

Methodology rules enforced here (non-negotiable, from EXPERIMENT_LOG.md /
the FINDINGS.md task brief):
  - The null for any shared-pool / cross-model-overlap claim is
    stats.cross_model_null's pooled_frequency_baseline sub-result, NEVER
    label_shuffle (stats.py's own docstring documents label_shuffle has
    near-zero power to detect a genuinely shared pool — see
    test_fully_shared_pool_label_shuffle_has_no_power in test_stats.py).
    label_shuffle is still computed and reported, clearly labeled as
    "documented as the wrong null for this design; included for
    transparency only."
  - Exact permutation tests wherever the arrangement count is tractable
    (it always is here: max group size 4+4 runs -> C(8,4)=70). The
    resulting p-value floor (1/70 = 0.0143 for two N=4 groups; coarser for
    any pair involving glm or fable, both N=2) is reported alongside every
    such test, not just asserted in prose.
  - RAW labels only (never the semantic/canon view) — matches EXP-004's
    pre-registration and analysis.json's own "primary" convention.
  - Degradation depth is right-censored at depth 30 for runs that never
    repeated a topic or refused (recorded as null in analysis.json,
    meaning "survived the full cascade"); this driver assigns those runs
    depth=30 for any statistic over depth, documented at the point of use.
    This is a real modeling choice (a survival/Tobit-style analysis
    treating 30 as a proper censoring bound would be more correct) and is
    flagged in every output block that uses it.

Usage:
  python3 -m benchmark.run_stats_inference \
      --pilot experiment-runs/2026-07-17-cascade-pilot \
      --novelty experiment-runs/2026-07-17-cascade-pilot/novelty.json \
      --out experiment-runs/2026-07-17-cascade-pilot/stats_inference.json
"""

import argparse
import datetime
import itertools
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .analyze_pilot import load_lanes
from .metrics import cross_model_overlap, depth_to_degradation, normalize_label, path_divergence
from .stats import (_EXACT_LIMIT, bootstrap_ci, cliffs_delta,
                     cliffs_delta_magnitude, cross_model_null, mean,
                     pairwise_cliffs_delta)

CENSOR_DEPTH = 30  # depth=30 assigned to runs with degradation depth=None
Z_95 = 1.959963984540054  # two-sided 95% normal quantile, closed form

# The three OpenAI-family model keys and three Anthropic-family model keys
# under test in EXP-004's roster (haiku excluded from the Anthropic
# "memorization" group below because it is ALSO the rejector instrument in
# this design -- see the flagged nuance in the memorization section).
OPENAI_MODELS = ["codex:mini", "codex:sol", "codex:5.4"]
ANTHROPIC_MODELS_ALL = ["haiku", "sonnet", "opus", "fable"]
ANTHROPIC_MODELS_NONHAIKU = ["sonnet", "opus", "fable"]
OPENWEIGHT_MODELS = ["api:deepseek", "api:qwen", "api:glm"]

# Meta-register labels (adversarial review finding, 2026-07-17 midday
# hostile-review pass, EXPERIMENT_LOG.md "EXP-004 red-team corrections"):
# models joking about joke-telling / comedy / being an AI under rejection
# pressure, rather than repeating a genuine topic. 11/13 Anthropic
# degradation events turned out to be repeats of one of these six labels
# (verified independently below -- opus's 13,11,13,13 is `comedy` x4).
# Passed through normalize_label so the set matches exactly what
# rejector.label_topic's v2 output looks like after normalization (these
# six are already normalize_label-idempotent -- no punctuation, no
# plurals -- but routing through it keeps this constant honest against
# normalize_label's actual behavior rather than an assumption about it).
META_REGISTER_LABELS_RAW = ["comedy", "joke", "humor", "ai", "software",
                            "laughter"]
META_REGISTER_LABELS = {normalize_label(x) for x in META_REGISTER_LABELS_RAW}


# ------------------------------------------------------- generic helpers


def _flat_cross_run_jaccards(model_paths: Dict[str, List[Sequence[str]]]
                              ) -> List[float]:
    """Every individual cross-model run-pair jaccard, pooled flat across
    all model pairs -- the same population cross_model_overlap averages
    over to produce mean_cross_jaccard, exposed here as a list instead of
    just its mean so bootstrap_ci has something to resample.

    NON-INDEPENDENCE CAVEAT (stated once here, binds everywhere this list
    is used): a single run appears in many pairs (once per OTHER model, N-1
    times), so these are not i.i.d. draws. A percentile bootstrap over this
    flat list therefore UNDERSTATES true sampling uncertainty -- treat its
    CI as optimistic (too narrow), not as the last word. The model-pair-
    level bootstrap computed alongside is the more conservative companion
    estimate for exactly this reason.
    """
    from .metrics import jaccard
    names = sorted(model_paths)
    out: List[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for pa in model_paths[names[i]]:
                for pb in model_paths[names[j]]:
                    out.append(jaccard(pa, pb))
    return out


def _flat_within_run_jaccards(paths: List[Sequence[str]]) -> List[float]:
    """Every individual within-model run-pair jaccard for one model."""
    from .metrics import jaccard
    return [jaccard(paths[i], paths[j])
            for i in range(len(paths)) for j in range(i + 1, len(paths))]


def pooled_frequency_null_diagnostics(
    model_paths: Dict[str, List[Sequence[str]]],
    n_perm: int = 10000, seed: int = 0
) -> Dict[str, object]:
    """Diagnostics for stats.cross_model_null's pooled_frequency_baseline
    sub-test that its own return value does not carry: the synthetic null
    distribution's min/max/mean and the size of the pooled vocabulary the
    synthetic universes draw from.

    This recomputes that sub-test's inner loop verbatim -- same topic-
    frequency table construction, same `random.Random(seed + 1)` stream
    stats.cross_model_null uses internally for pooled_frequency_baseline --
    so the min/max reported here describe the EXACT null distribution
    stats.cross_model_null's p-value was computed against, not an
    approximation of it. Recomputing here (rather than changing
    stats.cross_model_null's return contract) is a deliberate scope choice:
    these are diagnostics for THIS analysis plan, not a change to the
    shared toolkit.

    Added 2026-07-17 per adversarial review finding B1: the FINDINGS.md
    draft stated "range 0.111-0.161" and "270 distinct topics" computed
    off-pipeline (a scratch script, not this driver) -- those numbers were
    right, but not reproducible from stats_inference.json. This function
    makes them so, and cross-checks its own recomputed p-value against
    stats.cross_model_null's as an integrity guard: if they disagree,
    stats.py's internal implementation has changed since this was written.
    """
    names = sorted(model_paths)
    observed = cross_model_overlap(model_paths)["mean_cross_jaccard"]

    topic_counts: Dict[str, int] = {}
    for m in names:
        for p in model_paths[m]:
            for t in p:
                topic_counts[t] = topic_counts.get(t, 0) + 1
    topics_list = list(topic_counts.keys())
    weights = [topic_counts[t] for t in topics_list]

    rng2 = random.Random(seed + 1)  # matches cross_model_null's stream exactly
    pooled_null: List[float] = []
    for _ in range(n_perm):
        synth_groups: Dict[str, List[Sequence[str]]] = {}
        for m in names:
            synth_groups[m] = [
                rng2.choices(topics_list, weights=weights, k=len(p))
                if len(p) else []
                for p in model_paths[m]
            ]
        pooled_null.append(cross_model_overlap(synth_groups)["mean_cross_jaccard"])

    extreme = sum(1 for s in pooled_null if s >= observed - 1e-12)
    recomputed_p = (extreme + 1) / (len(pooled_null) + 1)

    return {
        "n_distinct_topics": len(topics_list),
        "n_perm": n_perm,
        "pooled_null_min": min(pooled_null),
        "pooled_null_max": max(pooled_null),
        "pooled_null_mean": sum(pooled_null) / len(pooled_null),
        "observed_below_entire_null_range": observed < min(pooled_null),
        "recomputed_p_value_for_integrity_check": recomputed_p,
    }


def build_lane_manifest(pilot_dir: Path) -> List[Dict[str, object]]:
    """Record the exact summary.json files load_lanes() consumed for this
    run, plus their mtimes -- makes the "frozen snapshot" claim (that this
    analysis predates the still-running fill lanes) an explicit, checkable
    fact in stats_inference.json rather than an assertion in prose.

    Mirrors analyze_pilot.load_lanes's own glob (`*/summary.json`) exactly
    so this manifest always describes precisely the files that produced
    the paths/analysis this driver ran over -- if a fill lane's summary.json
    appears here that wasn't accounted for in FINDINGS.md's caveats, that's
    a real discrepancy to catch, not a documentation nicety.
    """
    manifest = []
    for summary_file in sorted(pilot_dir.glob("*/summary.json")):
        st = summary_file.stat()
        manifest.append({
            "path": str(summary_file),
            "mtime_epoch": st.st_mtime,
            "mtime_utc": datetime.datetime.fromtimestamp(
                st.st_mtime, tz=datetime.timezone.utc).isoformat(),
        })
    return manifest


def censored_depths(degradation: List[Dict], censor: int = CENSOR_DEPTH
                     ) -> List[int]:
    """Per-run degradation depth with null (survived the cascade) mapped
    to `censor`. See module docstring's censoring caveat."""
    return [d["depth"] if d["depth"] is not None else censor
            for d in degradation]


# ------------------------------------ meta-register-exclusion robustness


def depth_to_degradation_meta_excluded(
    path: Sequence[str], meta_labels: Sequence[str] = None,
    refusal_turns: Optional[Sequence[int]] = None,
) -> Dict[str, Optional[int]]:
    """Same contract as metrics.depth_to_degradation, but any topic in
    `meta_labels` (default META_REGISTER_LABELS) is treated as
    NEVER-MATCHING for repeat detection: it is skipped by the scan
    entirely, both as a candidate repeat and as something later turns
    could be judged a repeat OF. This is the literal reading of the
    adversarial review's instruction ("treating meta-register labels as
    never-matching for repeat detection") -- a meta-labeled turn is
    invisible to the repeat-detection state machine, not merely exempted
    from being counted as A repeat once matched.

    Refusal detection is untouched -- meta-exclusion only concerns the
    topic-repeat channel of degradation.
    """
    if meta_labels is None:
        meta_labels = META_REGISTER_LABELS
    seen = set()
    repeat_depth = None
    for i, t in enumerate(path):
        if t in meta_labels:
            continue
        if t in seen:
            repeat_depth = i
            break
        seen.add(t)
    refusal_depth = min(refusal_turns) if refusal_turns else None
    candidates = [d for d in (repeat_depth, refusal_depth) if d is not None]
    return {
        "repeat_depth": repeat_depth,
        "refusal_depth": refusal_depth,
        "depth": min(candidates) if candidates else None,
    }


def load_raw_paths_and_refusals_from_turns(
    pilot_dir: Path
) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[List[int]]],
           List[Dict[str, object]]]:
    """Independent re-derivation of per-run topic paths AND refusal-turn
    lists directly from every lane's turns-<model>-r<NN>.jsonl files --
    bypassing summary.json's cached "paths" field entirely. This is the
    "re-derive from raw lane JSONLs" reconstruction the fix wave's
    meta-excluded robustness row and degradation-event decomposition are
    required to use (EXPERIMENT_LOG.md, "EXP-004 red-team corrections",
    item 1a), rather than a post-hoc filter over analysis.json's already-
    computed depths.

    Mirrors run_pilot.py's own per-lane inclusion rule EXACTLY: a model is
    only written into a lane's summary.json["per_model"] when that lane
    produced >= 2 successful runs for it (`if len(ps) >= 2:` in
    run_pilot.py). This function reads each lane's summary.json ONLY for
    (a) which run_ids that lane recorded as failures and (b) which
    models/run-counts it attempted -- NEVER for path contents, which come
    solely from turns-*.jsonl -- and applies the identical >=2 gate, so
    the roster analyzed here has the same per-model N as analysis.json,
    reconstructed independently rather than trusted from it.

    A run_id absent from a lane's failures list, for a model that cleared
    the >=2 gate in that lane, is treated as a successful completed run,
    and its turns file MUST exist -- if it doesn't, that is a data
    integrity problem worth crashing on, not silently skipping.

    NOTE ON A KNOWN, DELIBERATELY-NOT-FIXED-HERE ARTIFACT: this >=2 gate
    means a lane with exactly ONE successful run for a model -- even a
    fully complete, valid cascade -- contributes ZERO runs to this
    reconstruction (matching analysis.json, which also has zero from that
    lane). `lane-api-fill-glm`'s `turns-api:glm-r01.jsonl` is exactly this
    case: 30 turns, complete, degrades at repeat_depth=16, but that lane's
    OTHER glm run (r00) failed, so the lane has only 1 survivor and writes
    no per_model entry at all -- see FINDINGS.md's explicit statement of
    this exclusion rule. This function reproduces that exclusion (for
    parity with the frozen analysis.json this driver's other sections
    depend on) rather than silently including the orphaned run, which
    would change glm's N without the ability to update analysis.json
    (read-only input, per task constraints).
    """
    paths: Dict[str, List[List[str]]] = {}
    refusals: Dict[str, List[List[int]]] = {}
    per_lane_manifest: List[Dict[str, object]] = []
    for summary_file in sorted(pilot_dir.glob("*/summary.json")):
        lane_dir = summary_file.parent
        with open(summary_file) as f:
            s = json.load(f)
        failed_run_ids = {fl["run_id"] for fl in s.get("failures", [])}
        for model in s.get("models", []):
            n_runs_attempted = s.get("runs", 0)
            run_ids = [
                "%s-r%02d" % (model, n) for n in range(n_runs_attempted)
                if ("%s-r%02d" % (model, n)) not in failed_run_ids
            ]
            if len(run_ids) < 2:
                continue  # matches run_pilot.py's own per-model gate
            for run_id in run_ids:
                turns_file = lane_dir / ("turns-%s.jsonl" % run_id)
                if not turns_file.exists():
                    raise SystemExit(
                        "integrity error: %s's summary.json implies run "
                        "%r succeeded, but %s does not exist"
                        % (lane_dir, run_id, turns_file))
                topics: List[str] = []
                refusal_turns: List[int] = []
                with open(turns_file) as tf:
                    for line in tf:
                        rec = json.loads(line)
                        topics.append(rec["topic"])
                        if rec.get("refusal"):
                            refusal_turns.append(rec["turn"])
                paths.setdefault(model, []).append(topics)
                refusals.setdefault(model, []).append(refusal_turns)
            per_lane_manifest.append({
                "lane": str(lane_dir), "model": model,
                "surviving_run_ids": run_ids,
            })
    return paths, refusals, per_lane_manifest


def verify_turns_reconstruction(paths_from_turns: Dict[str, List[Sequence[str]]],
                                 analysis: Dict) -> Dict[str, object]:
    """Trust-but-verify: the raw-turns reconstruction above must reproduce
    the same per-model RUN COUNT and the same (order-independent) RAW,
    non-meta-excluded degradation-depth multiset that analysis.json
    already encodes. Order across runs is not load-bearing anywhere in
    this driver (every statistic here is an unordered function of a
    model's run list), so depths are compared as sorted multisets, not
    positionally. If this check fails, every meta-excluded / decomposition
    number downstream of it is untrustworthy and must not be reported."""
    mismatches: Dict[str, str] = {}
    for m, ps in paths_from_turns.items():
        frozen = analysis["per_model"].get(m)
        if frozen is None:
            mismatches[m] = "model present in turns reconstruction but " \
                            "absent from analysis.json"
            continue
        if len(ps) != frozen["n_runs"]:
            mismatches[m] = "n_runs %d (turns) vs %d (analysis.json)" % (
                len(ps), frozen["n_runs"])
            continue
        # None ("survived the cascade") sorts before any int depth via the
        # (is_none, value) key -- None has no natural ordering against int
        # in Python 3, and both sequences use the same key so the
        # comparison is still well-defined and order-independent.
        _sort_key = lambda d: (d is not None, d if d is not None else 0)
        recomputed = sorted((depth_to_degradation(p)["depth"] for p in ps),
                            key=_sort_key)
        frozen_depths = sorted(
            (d["depth"] for d in frozen["degradation"]), key=_sort_key)
        if recomputed != frozen_depths:
            mismatches[m] = "depths %r (turns) vs %r (analysis.json)" % (
                recomputed, frozen_depths)
    missing_models = set(analysis["per_model"]) - set(paths_from_turns)
    if missing_models:
        mismatches["_missing_from_turns_reconstruction"] = sorted(missing_models)
    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "n_models_checked": len(paths_from_turns),
    }


_FRAMING_PREFIX_RE = re.compile(
    r"^\s*(alright|sure|okay|ok|here.?s|how about this|got one|try this|"
    r"no problem|fine)\b", re.IGNORECASE)


def framing_prefix_rate_by_model(pilot_dir: Path) -> Dict[str, Dict[str, object]]:
    """Style-confound check for the memorization exact-match tier
    (adversarial review, EXPERIMENT_LOG.md item 3): the exact-match tier
    is a whole-normalized-string hash match, so a model that habitually
    prefixes its joke with framing prose ("Alright, no science ... This
    one's about my bank account:") defeats it structurally regardless of
    whether the punchline itself is memorized. Scores every model's raw
    joke text (straight from turns-*.jsonl, same universe
    benchmark/joke_novelty.py's n_jokes counts over: every turn in every
    turns file, including failed/excluded runs' partial output -- this is
    a surface-text style property, not a path-level one, so the roster
    here is intentionally the full novelty.json universe, not the
    path-level-only roster the meta-excluded sections use) against a
    simple leading-phrase regex.

    The regex is deliberately narrow (a handful of literal English
    conversational openers) rather than a general "is this meta-
    commentary" classifier -- it is calibrated by inspection against
    this pilot's actual transcripts (sonnet's repeating "Alright, ..."
    preamble; grok's zero-preamble "Why did/don't ..." pattern), not
    tuned to hit a target number.
    """
    jokes_by_model: Dict[str, List[str]] = {}
    for turns_file in sorted(pilot_dir.rglob("turns-*.jsonl")):
        model = turns_file.stem.replace("turns-", "").rsplit("-r", 1)[0]
        with open(turns_file) as f:
            for line in f:
                rec = json.loads(line)
                jokes_by_model.setdefault(model, []).append(rec["joke"])
    out: Dict[str, Dict[str, object]] = {}
    for m, jokes in sorted(jokes_by_model.items()):
        hits = sum(1 for j in jokes if _FRAMING_PREFIX_RE.match(j))
        out[m] = {
            "n_jokes": len(jokes),
            "framing_prefix_hits": hits,
            "framing_prefix_rate": hits / len(jokes) if jokes else 0.0,
        }
    return out


def permutation_test_mean_diff(a: Sequence[float], b: Sequence[float],
                                n_perm: int = 10000, seed: int = 0
                                ) -> Dict[str, object]:
    """Exact/Monte-Carlo permutation test on a difference-of-means
    statistic between two independent scalar samples.

    Structurally identical to stats.permutation_test_divergence (same
    _EXACT_LIMIT exact-enumeration cutover, same add-one-corrected p-value
    when falling back to Monte Carlo -- Davison & Hinkley 1997) but generic
    over ANY scalar pair rather than hardcoding mean-pairwise-jaccard.
    Needed here for degradation-depth comparisons, which are per-run
    scalars, not paths.

    NULL: group membership (which model a run belongs to) is exchangeable
    -- under H0 the na+nb runs are draws from one shared depth-generating
    process. Two-sided: extremeness is judged on |stat|.
    """
    na, nb = len(a), len(b)
    if na < 1 or nb < 1:
        raise ValueError("need >= 1 value per group")
    pooled = list(a) + list(b)
    n = na + nb
    observed = mean(a) - mean(b)

    exact_count = math.comb(n, na)
    if exact_count <= _EXACT_LIMIT:
        method = "exact"
        null_stats = []
        for combo in itertools.combinations(range(n), na):
            combo_set = set(combo)
            group_a = [pooled[i] for i in combo]
            group_b = [pooled[i] for i in range(n) if i not in combo_set]
            null_stats.append(mean(group_a) - mean(group_b))
        extreme = sum(1 for s in null_stats if abs(s) >= abs(observed) - 1e-12)
        p_value = extreme / len(null_stats)
        n_used = exact_count
        p_floor = 1.0 / n_used
    else:
        method = "monte_carlo"
        rng = random.Random(seed)
        idx = list(range(n))
        null_stats = []
        for _ in range(n_perm):
            rng.shuffle(idx)
            group_a = [pooled[i] for i in idx[:na]]
            group_b = [pooled[i] for i in idx[na:]]
            null_stats.append(mean(group_a) - mean(group_b))
        extreme = sum(1 for s in null_stats if abs(s) >= abs(observed) - 1e-12)
        p_value = (extreme + 1) / (len(null_stats) + 1)
        n_used = n_perm
        p_floor = 1.0 / (n_perm + 1)

    return {"stat_mean_diff": observed, "p_value": p_value,
            "n_perm_used": n_used, "method": method, "p_floor": p_floor,
            "na": na, "nb": nb}


# ------------------------------------------------ multiple-comparison correction


def holm_correction(labeled_pvals: Sequence[Tuple[str, float]]
                     ) -> List[Dict[str, object]]:
    """Holm-Bonferroni step-down FWER correction. Standard algorithm
    (Holm 1979): sort ascending, adjusted p_(i) = max_{j<=i} min(1,
    (m-j+1)*p_(j)) -- the running max enforces monotonicity (an adjusted
    p-value can never be smaller than an earlier one in sorted order).
    Chosen as this driver's PRIMARY correction: controls family-wise error
    rate without an independence assumption BH needs for its FDR guarantee,
    and is never less powerful than a flat Bonferroni correction."""
    m = len(labeled_pvals)
    ordered = sorted(labeled_pvals, key=lambda kv: kv[1])
    out = []
    running_max = 0.0
    for i, (label, p) in enumerate(ordered):  # i is 0-indexed; rank = i+1
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)
        out.append({"label": label, "p_value": p,
                    "p_holm": running_max, "rank": i + 1})
    return out


def benjamini_hochberg_correction(labeled_pvals: Sequence[Tuple[str, float]]
                                   ) -> List[Dict[str, object]]:
    """Benjamini-Hochberg step-up FDR correction (1995). Computed
    alongside Holm for cross-checking only; Holm is this driver's
    reported/primary correction (see holm_correction docstring)."""
    m = len(labeled_pvals)
    ordered = sorted(labeled_pvals, key=lambda kv: kv[1])
    adj = [0.0] * m
    running_min = 1.0
    for i in range(m - 1, -1, -1):  # from largest p down to smallest
        label, p = ordered[i]
        val = min(1.0, (m / (i + 1)) * p)
        running_min = min(running_min, val)
        adj[i] = running_min
    return [{"label": ordered[i][0], "p_value": ordered[i][1],
             "p_bh": adj[i], "rank": i + 1} for i in range(m)]


# --------------------------------------------------- proportion statistics


def wilson_ci(hits: int, n: int, z: float = Z_95) -> Dict[str, float]:
    """Wilson score interval for a binomial proportion. Preferred over the
    normal (Wald) approximation at the hit counts here (some models have
    single-digit hits out of ~100-165 trials, where Wald under-covers)."""
    if n == 0:
        return {"phat": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    phat = hits / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return {"phat": phat, "lo": max(0.0, center - half),
            "hi": min(1.0, center + half), "n": n}


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher test on a 2x2 table
        [[a, b],
         [c, d]]
    (a = hits group 1, b = misses group 1, c = hits group 2, d = misses
    group 2). Standard definition: sum the hypergeometric probability of
    every table with the same margins whose probability is <= the observed
    table's probability. Pure stdlib (math.comb) -- exact regardless of n,
    no scipy dependency needed for a single 2x2 table."""
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1 = a + c

    def hyper_prob(x: int) -> float:
        return (math.comb(row1, x) * math.comb(row2, col1 - x)
                / math.comb(n, col1))

    obs_prob = hyper_prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    p = 0.0
    for x in range(lo, hi + 1):
        px = hyper_prob(x)
        if px <= obs_prob * (1 + 1e-9):
            p += px
    return min(1.0, p)


# ---------------------------------------------------------- verification


def verify_reconstruction(paths: Dict[str, List[Sequence[str]]],
                           analysis: Dict) -> Dict[str, object]:
    """Recompute cross_model_overlap / path_divergence from the
    reconstructed raw paths and diff against the frozen analysis.json
    numbers. This is the trust-but-verify step: if this driver's
    reconstruction of "the data behind analysis.json" is wrong, every
    downstream p-value and CI here is over the wrong data."""
    recomputed_cross = cross_model_overlap(paths)
    frozen_cross = analysis["cross_model"]
    cross_diffs = {k: abs(recomputed_cross[k] - frozen_cross[k])
                   for k in frozen_cross if k in recomputed_cross}
    max_cross_diff = max(cross_diffs.values()) if cross_diffs else None

    per_model_diffs = {}
    for m, ps in paths.items():
        recomputed = path_divergence(ps)
        frozen = analysis["per_model"][m]["divergence"]
        per_model_diffs[m] = {k: abs(recomputed[k] - frozen[k])
                               for k in frozen}
    max_per_model_diff = max(
        (v for d in per_model_diffs.values() for v in d.values()),
        default=None)

    ok = ((max_cross_diff is None or max_cross_diff < 1e-9)
          and (max_per_model_diff is None or max_per_model_diff < 1e-9))
    return {
        "reconstruction_matches_analysis_json": ok,
        "max_cross_model_abs_diff": max_cross_diff,
        "max_per_model_abs_diff": max_per_model_diff,
        "n_models_reconstructed": len(paths),
        "n_models_in_analysis_json": analysis["n_models"],
    }


# --------------------------------------------------------------- sections


def section_headline_cross_model(paths: Dict[str, List[Sequence[str]]],
                                  analysis: Dict, seed: int = 0
                                  ) -> Dict[str, object]:
    observed = analysis["cross_model"]["mean_cross_jaccard"]
    null = cross_model_null(paths, n_perm=10000, seed=seed)
    null_diag = pooled_frequency_null_diagnostics(paths, n_perm=10000, seed=seed)
    diag_p_matches = abs(null_diag["recomputed_p_value_for_integrity_check"]
                          - null["pooled_frequency_baseline"]["p_value"]) < 1e-9

    flat_pairs = _flat_cross_run_jaccards(paths)
    boot_run_pair = bootstrap_ci(flat_pairs, stat_fn=mean, seed=seed)

    per_pair_means = [v for k, v in analysis["cross_model"].items()
                      if "|" in k]
    boot_model_pair = bootstrap_ci(per_pair_means, stat_fn=mean, seed=seed)
    # Computed from the actual roster size, not hardcoded -- this used to
    # read a literal "45" (correct only while the roster was 10 models;
    # C(10,2)=45). Caught 2026-07-17 when the roster grew to 11
    # (C(11,2)=55) and the literal silently went stale. Never hardcode a
    # combinatorial count that depends on runtime data size.
    n_model_pairs = len(per_pair_means)

    return {
        "claim": "Pre-registered hypothesis: models share a substantially "
                 "overlapping topic pool (predicted cross-model jaccard "
                 "~0.35). FAILED -- see EXP-004.",
        "observed_mean_cross_jaccard": observed,
        "predicted": 0.35,
        "null_test": {
            "pooled_frequency_baseline": {
                **null["pooled_frequency_baseline"],
                "AUTHORITATIVE_FOR_THIS_CLAIM": True,
                "diagnostics": {
                    **null_diag,
                    "diagnostics_p_value_matches_authoritative_p_value": diag_p_matches,
                },
            },
            "label_shuffle": {
                **null["label_shuffle"],
                "AUTHORITATIVE_FOR_THIS_CLAIM": False,
                "warning": "Documented wrong null for a shared-pool claim "
                           "(near-zero power -- see stats.cross_model_null "
                           "docstring and test_stats.py). Reported for "
                           "transparency only; do not cite this p-value "
                           "as evidence about pool-sharing.",
            },
        },
        "bootstrap_ci": {
            "run_pair_level": {
                **boot_run_pair,
                "unit": "one value per (run_i-of-model_A, run_j-of-model_B) "
                        "pair, pooled across all %d model pairs" % n_model_pairs,
                "matches_point_estimate": abs(boot_run_pair["point"] - observed) < 1e-9,
                "caveat": "non-independent observations (each run appears "
                          "in many pairs) -- CI is likely too narrow; see "
                          "module docstring on _flat_cross_run_jaccards.",
            },
            "model_pair_level": {
                **boot_model_pair,
                "unit": "one value per model-PAIR mean jaccard (%d model "
                        "pairs), the more conservative/exchangeable unit"
                        % n_model_pairs,
                "point_vs_headline_diff": abs(boot_model_pair["point"] - observed),
            },
        },
    }


def section_headline_within_model(analysis: Dict, seed: int = 0
                                   ) -> Dict[str, object]:
    per_model_vals = [pm["divergence"]["set_jaccard"]
                      for pm in analysis["per_model"].values()]
    observed = mean(per_model_vals)
    boot = bootstrap_ci(per_model_vals, stat_fn=mean, seed=seed)
    return {
        # NOTE: this claim string used to hardcode the observed value as a
        # literal ("0.182") -- caught 2026-07-17 when the roster grew from
        # 10 to 11 models (grok's high self-similarity moved the true mean
        # to ~0.208) and the literal went stale. Interpolated from
        # `observed` now so it can never again silently disagree with the
        # `observed_within_model_mean_jaccard` field two lines down.
        "claim": "Within-model mean set jaccard across runs, averaged over "
                 "models (EXP-004 predicted ~0.55; observed %.3f)." % observed,
        "observed_within_model_mean_jaccard": observed,
        "predicted": 0.55,
        "bootstrap_ci": {
            **boot,
            "unit": "one value per MODEL (n=%d): each model's own mean "
                    "pairwise within-run set_jaccard" % len(per_model_vals),
            "matches_point_estimate": abs(boot["point"] - observed) < 1e-9,
        },
        "per_model_point_estimates": {
            m: pm["divergence"]["set_jaccard"]
            for m, pm in sorted(analysis["per_model"].items())},
    }


def section_degradation_battery(analysis: Dict, seed: int = 0
                                 ) -> Dict[str, object]:
    models = sorted(analysis["per_model"])
    depths = {m: censored_depths(analysis["per_model"][m]["degradation"])
              for m in models}

    pairwise_p = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            key = "%s|%s" % (a, b)
            pairwise_p[key] = permutation_test_mean_diff(
                depths[a], depths[b], seed=seed)

    deltas = pairwise_cliffs_delta(depths)

    labeled = [(k, v["p_value"]) for k, v in pairwise_p.items()]
    holm = holm_correction(labeled)
    bh = benjamini_hochberg_correction(labeled)
    holm_by_key = {r["label"]: r for r in holm}
    bh_by_key = {r["label"]: r for r in bh}

    pairs_out = {}
    for key, res in pairwise_p.items():
        pairs_out[key] = {
            **res,
            "cliffs_delta": deltas[key],
            "cliffs_delta_magnitude": cliffs_delta_magnitude(deltas[key]),
            "p_holm": holm_by_key[key]["p_holm"],
            "p_bh": bh_by_key[key]["p_bh"],
        }

    # Disclosed post-hoc single contrast: Anthropic (all 4) vs OpenAI
    # (all 3), pooled runs, censoring convention as above. NOT
    # pre-registered -- chosen after EXP-004's results made the family-
    # level degradation pattern visible (adversarial review B2, see module
    # docstring). Reported as the single best-supported contrast for that
    # pattern (one test, so no multiple-comparisons burden, unlike the
    # C(n_models,2)-pair matrix above) -- NOT as confirmatory evidence. A
    # genuinely pre-registered replication is required before
    # "confirmatory" is earned.
    anth_pool = [d for m in ANTHROPIC_MODELS_ALL if m in depths
                 for d in depths[m]]
    openai_pool = [d for m in OPENAI_MODELS if m in depths
                   for d in depths[m]]
    family_test = permutation_test_mean_diff(anth_pool, openai_pool,
                                              n_perm=10000, seed=seed)
    family_delta = cliffs_delta(anth_pool, openai_pool)

    # Robustness check (adversarial review M4): haiku is BOTH the rejector
    # instrument judging every run's degradation AND a model under test --
    # its own degradation depths are judged by itself. Re-running the
    # contrast with haiku dropped from the Anthropic pool shows whether the
    # family-level pattern depends on that confound.
    anth_nonhaiku_pool = [d for m in ANTHROPIC_MODELS_NONHAIKU if m in depths
                          for d in depths[m]]
    family_test_nonhaiku = permutation_test_mean_diff(
        anth_nonhaiku_pool, openai_pool, n_perm=10000, seed=seed)
    family_delta_nonhaiku = cliffs_delta(anth_nonhaiku_pool, openai_pool)

    n_degraded = {m: sum(1 for d in analysis["per_model"][m]["degradation"]
                          if d["depth"] is not None)
                  for m in models}
    n_runs = {m: analysis["per_model"][m]["n_runs"] for m in models}
    models_with_any_degradation = [m for m in models if n_degraded[m] > 0]

    return {
        "censoring_convention": "degradation depth=null (survived the "
            "cascade without repeating or refusing) mapped to depth=%d "
            "for every statistic in this section." % CENSOR_DEPTH,
        "pairwise_permutation_and_cliffs_delta": pairs_out,
        "correction_used_primary": "holm",
        "correction_note": "BH also computed (p_bh field) for cross-check; "
                           "Holm is the corrected p-value cited in FINDINGS.md.",
        "n_pairs_corrected": len(labeled),
        "anthropic_vs_openai_family_contrast": {
            **family_test,
            "cliffs_delta": family_delta,
            "cliffs_delta_magnitude": cliffs_delta_magnitude(family_delta),
            "anthropic_models": ANTHROPIC_MODELS_ALL,
            "openai_models": OPENAI_MODELS,
            "note": "Pooled runs (not model-level means); n_a=%d runs, "
                    "n_b=%d runs." % (len(anth_pool), len(openai_pool)),
            "disclosure": "POST-HOC single contrast, NOT pre-registered -- "
                "chosen after EXP-004's results made this pattern visible. "
                "The single best-supported contrast for the family-level "
                "pattern (one test, no multiple-comparisons burden), but "
                "strong exploratory evidence, not confirmatory evidence. "
                "A genuinely pre-registered replication would be required "
                "to call it confirmatory.",
        },
        "anthropic_nonhaiku_vs_openai_family_contrast": {
            **family_test_nonhaiku,
            "cliffs_delta": family_delta_nonhaiku,
            "cliffs_delta_magnitude": cliffs_delta_magnitude(family_delta_nonhaiku),
            "anthropic_models": ANTHROPIC_MODELS_NONHAIKU,
            "openai_models": OPENAI_MODELS,
            "note": "Robustness check for anthropic_vs_openai_family_contrast "
                    "with haiku dropped from the Anthropic pool. Pooled runs; "
                    "n_a=%d runs, n_b=%d runs." % (
                        len(anth_nonhaiku_pool), len(openai_pool)),
            "why": "Haiku is both the rejector instrument (judges every "
                "run's degradation, including its own) and a model under "
                "test -- its own degradation depths are judged by itself. "
                "Direction of any bias this introduces into haiku's own "
                "depth numbers is untested. This contrast shows whether the "
                "family-level pattern holds without haiku in the pool.",
            "disclosure": "Also POST-HOC, same caveat as "
                "anthropic_vs_openai_family_contrast above.",
        },
        "degradation_incidence_audit": {
            "n_runs_per_model": n_runs,
            "n_degraded_runs_per_model": n_degraded,
            "models_with_at_least_one_degrading_run": models_with_any_degradation,
            "n_models_with_at_least_one_degrading_run": len(models_with_any_degradation),
            "n_models_total": len(models),
            "EXPERIMENT_LOG_claim": "'>=1/3 of roster degrades by turn 30 "
                "-- met massively (8/10 models)' (original 10-model pilot; "
                "roster is now %d models after fills merged)" % len(models),
            "audit_verdict": "The literal count of models with >=1 "
                "degrading run among the %d with per_model data is %d/%d "
                "(only %s shows zero degradation across all its runs), not "
                "8/10 as stated (that count was against the original "
                "10-model roster; the log figure predates the fills). The "
                "EXPERIMENT_LOG figure is not reproduced exactly under the "
                "most natural reading of 'degrades' (>=1 run degrades); it "
                "may reflect a different counting rule (e.g. majority-of-"
                "runs) that isn't specified in the log. Flagged per task "
                "instructions, not silently corrected in the log itself." % (
                    len(models), len(models_with_any_degradation), len(models),
                    [m for m in models if n_degraded[m] == 0]),
        },
    }


def section_meta_excluded_robustness(pilot_dir: Path, analysis: Dict,
                                      seed: int = 0) -> Dict[str, object]:
    """Meta-register-exclusion robustness row (adversarial review,
    EXPERIMENT_LOG.md "EXP-004 red-team corrections" item 2): recomputes
    the Anthropic-vs-OpenAI family contrast AND its no-haiku variant with
    the six meta-register labels (META_REGISTER_LABELS) treated as
    never-matching for repeat detection, re-derived from raw lane JSONLs
    per the task brief -- NOT by filtering analysis.json's precomputed
    depths. The reconstruction is verified against analysis.json's frozen
    RAW depths first (data_integrity_check below); only if that passes
    are the meta-excluded numbers below trustworthy.

    DELIBERATELY REFUSAL-BLIND, matching analysis.json's OWN convention --
    a subtlety this fix wave surfaced (not a red-team finding; caught while
    building this section): run_pilot.py's per-turn refusal flags (from
    cascade.py's looks_like_refusal) are recorded in every turns-*.jsonl
    and even fed into a refusal-aware depth_to_degradation call for that
    run's console printout, but the value actually WRITTEN into
    summary.json (hence analysis.json, hence every published depth in
    FINDINGS.md/paper/DRAFT.md) is `depth_to_degradation(p)` called with
    NO refusal_turns argument at all (run_pilot.py, the
    `"degradation": [depth_to_degradation(p) for p in ps]` line) -- so
    every published degradation depth to date is pure topic-repeat depth;
    refusal detection has never actually contributed to a reported number.
    This section preserves that exact convention (refusal_turns is loaded
    by load_raw_paths_and_refusals_from_turns and returned for OTHER
    sections' use, but intentionally not passed to
    depth_to_degradation[_meta_excluded] here) so its meta-excluded
    numbers differ from the published ones ONLY in the one respect asked
    for -- meta-label handling -- not also in whether refusals count.
    section_degradation_event_decomposition, by contrast, is a genuinely
    new analysis and DOES use refusal_turns, with its own explicit
    disclosure of this same discrepancy.
    """
    paths_from_turns, refusals_from_turns, lane_manifest = \
        load_raw_paths_and_refusals_from_turns(pilot_dir)
    integrity = verify_turns_reconstruction(paths_from_turns, analysis)
    if not integrity["ok"]:
        raise SystemExit(
            "raw-turns reconstruction does NOT reproduce analysis.json's "
            "RAW degradation depths -- refusing to report meta-excluded "
            "numbers over a mismatched dataset. Diffs: %r" % integrity)

    models = sorted(paths_from_turns)

    def _censored(model_list, meta_exclude: bool) -> List[int]:
        out = []
        for m in model_list:
            if m not in paths_from_turns:
                continue
            for p in paths_from_turns[m]:
                if meta_exclude:
                    d = depth_to_degradation_meta_excluded(p, META_REGISTER_LABELS)
                else:
                    d = depth_to_degradation(p)
                depth = d["depth"]
                out.append(depth if depth is not None else CENSOR_DEPTH)
        return out

    per_model_meta_excluded_depths = {
        m: [depth_to_degradation_meta_excluded(p, META_REGISTER_LABELS)["depth"]
            for p in paths_from_turns[m]]
        for m in models
    }
    n_degraded_meta_excluded = {
        m: sum(1 for d in depths if d is not None)
        for m, depths in per_model_meta_excluded_depths.items()
    }
    n_runs = {m: len(paths_from_turns[m]) for m in models}

    anth_meta = _censored(ANTHROPIC_MODELS_ALL, meta_exclude=True)
    anth_nonhaiku_meta = _censored(ANTHROPIC_MODELS_NONHAIKU, meta_exclude=True)
    openai_meta = _censored(OPENAI_MODELS, meta_exclude=True)

    family_test_meta = permutation_test_mean_diff(anth_meta, openai_meta, seed=seed)
    family_delta_meta = cliffs_delta(anth_meta, openai_meta)
    family_test_nonhaiku_meta = permutation_test_mean_diff(
        anth_nonhaiku_meta, openai_meta, seed=seed)
    family_delta_nonhaiku_meta = cliffs_delta(anth_nonhaiku_meta, openai_meta)

    return {
        "meta_register_labels": sorted(META_REGISTER_LABELS),
        "disclosure": "Re-derived from raw lane turns-*.jsonl (not a "
            "post-hoc filter of analysis.json's precomputed depths) per "
            "the 2026-07-17 hostile-review fix wave. A topic in "
            "meta_register_labels is skipped entirely by the repeat-scan "
            "-- it can neither trigger a repeat nor be repeated against.",
        "raw_turns_reconstruction_integrity_check": integrity,
        "n_runs_per_model": n_runs,
        "per_model_meta_excluded_degradation_depths": per_model_meta_excluded_depths,
        "degradation_incidence_meta_excluded": {
            "n_degraded_runs_per_model": n_degraded_meta_excluded,
            "note": "Compare to the RAW (non-meta-excluded) incidence in "
                "degradation_fingerprint_battery.degradation_incidence_audit "
                "-- e.g. haiku 4/4 raw -> %d/%d meta-excluded, fable %d/%d "
                "raw -> %d/%d meta-excluded." % (
                    n_degraded_meta_excluded.get("haiku", 0), n_runs.get("haiku", 0),
                    sum(1 for d in analysis["per_model"]["fable"]["degradation"]
                        if d["depth"] is not None),
                    analysis["per_model"]["fable"]["n_runs"],
                    n_degraded_meta_excluded.get("fable", 0), n_runs.get("fable", 0)),
        },
        "anthropic_vs_openai_family_contrast_meta_excluded": {
            **family_test_meta,
            "cliffs_delta": family_delta_meta,
            "cliffs_delta_magnitude": cliffs_delta_magnitude(family_delta_meta),
            "anthropic_models": ANTHROPIC_MODELS_ALL,
            "openai_models": OPENAI_MODELS,
            "note": "Meta-excluded companion to "
                "degradation_fingerprint_battery.anthropic_vs_openai_family_contrast. "
                "n_a=%d runs, n_b=%d runs (same run counts as the raw "
                "contrast -- meta-exclusion changes WHICH turn a run "
                "degrades at, or whether it degrades at all, never how "
                "many runs exist)." % (len(anth_meta), len(openai_meta)),
        },
        "anthropic_nonhaiku_vs_openai_family_contrast_meta_excluded": {
            **family_test_nonhaiku_meta,
            "cliffs_delta": family_delta_nonhaiku_meta,
            "cliffs_delta_magnitude": cliffs_delta_magnitude(family_delta_nonhaiku_meta),
            "anthropic_models": ANTHROPIC_MODELS_NONHAIKU,
            "openai_models": OPENAI_MODELS,
            "note": "Meta-excluded companion to "
                "degradation_fingerprint_battery.anthropic_nonhaiku_vs_openai_family_contrast.",
        },
        "lane_manifest_for_turns_reconstruction": lane_manifest,
    }


def section_incidence_fisher_companion(analysis: Dict,
                                        meta_excluded: Dict) -> Dict[str, object]:
    """Run-level incidence Fisher test (adversarial review item 1b): a
    CENSORING-FREE companion to section_degradation_battery's mean-depth
    contrasts. depth=30-censoring is a real modeling choice (module
    docstring); this test sidesteps it entirely by asking only "did the
    run ever degrade, yes or no" -- a 2x2 Fisher exact on
    (family x degraded/survived), computed for both the raw roster and
    the meta-excluded roster so the two robustness axes (haiku's dual
    role, meta-register mediation) can each be read against an incidence
    statistic, not only a mean-difference one.
    """
    def _incidence(model_list: List[str]) -> Tuple[int, int]:
        degraded = sum(
            1 for m in model_list if m in analysis["per_model"]
            for d in analysis["per_model"][m]["degradation"]
            if d["depth"] is not None)
        total = sum(
            analysis["per_model"][m]["n_runs"]
            for m in model_list if m in analysis["per_model"])
        return degraded, total - degraded

    anth_all_deg, anth_all_surv = _incidence(ANTHROPIC_MODELS_ALL)
    anth_nh_deg, anth_nh_surv = _incidence(ANTHROPIC_MODELS_NONHAIKU)
    openai_deg, openai_surv = _incidence(OPENAI_MODELS)

    p_raw_all = fisher_exact_two_sided(anth_all_deg, anth_all_surv,
                                        openai_deg, openai_surv)
    p_raw_nh = fisher_exact_two_sided(anth_nh_deg, anth_nh_surv,
                                       openai_deg, openai_surv)

    meta_n_deg = meta_excluded["degradation_incidence_meta_excluded"][
        "n_degraded_runs_per_model"]
    meta_n_runs = meta_excluded["n_runs_per_model"]

    def _meta_incidence(model_list: List[str]) -> Tuple[int, int]:
        degraded = sum(meta_n_deg.get(m, 0) for m in model_list)
        total = sum(meta_n_runs.get(m, 0) for m in model_list)
        return degraded, total - degraded

    anth_all_deg_m, anth_all_surv_m = _meta_incidence(ANTHROPIC_MODELS_ALL)
    anth_nh_deg_m, anth_nh_surv_m = _meta_incidence(ANTHROPIC_MODELS_NONHAIKU)
    openai_deg_m, openai_surv_m = _meta_incidence(OPENAI_MODELS)

    p_meta_all = fisher_exact_two_sided(anth_all_deg_m, anth_all_surv_m,
                                         openai_deg_m, openai_surv_m)
    p_meta_nh = fisher_exact_two_sided(anth_nh_deg_m, anth_nh_surv_m,
                                        openai_deg_m, openai_surv_m)

    return {
        "disclosure": "Censoring-free companion to "
            "degradation_fingerprint_battery's mean-depth family "
            "contrasts -- counts runs as degraded/survived only, no "
            "depth=30 censoring convention involved. Same post-hoc "
            "disclosure applies: chosen after the pattern was visible, "
            "not pre-registered.",
        "anthropic_all_vs_openai": {
            "table": {"anthropic_degraded": anth_all_deg,
                      "anthropic_survived": anth_all_surv,
                      "openai_degraded": openai_deg,
                      "openai_survived": openai_surv},
            "fisher_p_two_sided": p_raw_all,
        },
        "anthropic_nonhaiku_vs_openai": {
            "table": {"anthropic_degraded": anth_nh_deg,
                      "anthropic_survived": anth_nh_surv,
                      "openai_degraded": openai_deg,
                      "openai_survived": openai_surv},
            "fisher_p_two_sided": p_raw_nh,
        },
        "anthropic_all_vs_openai_meta_excluded": {
            "table": {"anthropic_degraded": anth_all_deg_m,
                      "anthropic_survived": anth_all_surv_m,
                      "openai_degraded": openai_deg_m,
                      "openai_survived": openai_surv_m},
            "fisher_p_two_sided": p_meta_all,
        },
        "anthropic_nonhaiku_vs_openai_meta_excluded": {
            "table": {"anthropic_degraded": anth_nh_deg_m,
                      "anthropic_survived": anth_nh_surv_m,
                      "openai_degraded": openai_deg_m,
                      "openai_survived": openai_surv_m},
            "fisher_p_two_sided": p_meta_nh,
        },
    }


def section_degradation_event_decomposition(
    pilot_dir: Path, analysis: Dict
) -> Dict[str, object]:
    """Degradation-event decomposition (adversarial review item 1d):
    classifies every run's FIRST degradation event (if any) as exactly
    one of topic_repeat / meta_register_repeat / refusal / survived, per
    model, re-derived from the same raw-turns reconstruction as the
    meta-excluded section (so this and section_meta_excluded_robustness
    are guaranteed to agree on what counts as a meta-register repeat).

    Classification rule per run, using the ORIGINAL (non-meta-excluded)
    depth_to_degradation: if a refusal occurred at or before the first
    repeat (or there was no repeat), the event is `refusal`; otherwise
    the run's first repeated topic is looked up directly and classified
    meta_register_repeat / topic_repeat by META_REGISTER_LABELS
    membership. A run with depth=None survived the full cascade.

    DISCLOSED METHODOLOGICAL DIVERGENCE FROM analysis.json: unlike
    section_meta_excluded_robustness (which deliberately stays
    refusal-blind for direct comparability to the published family
    contrast), THIS section passes refusal_turns into depth_to_degradation
    on purpose -- a decomposition table with a `refusal` column that can
    never be nonzero would misrepresent what is actually in the
    transcripts (CLI-wrapper persona refusals are real and visible, e.g.
    haiku's "I'm Claude Code..." turns). Consequence, stated plainly: for
    a handful of runs this table's classified event occurs at an earlier
    turn than that same run's published depth in analysis.json / FINDINGS
    Table 3, because analysis.json's stored depths never consider
    refusal_turns at all (run_pilot.py computes a refusal-aware depth only
    for its console printout, then discards it -- the
    `"degradation": [depth_to_degradation(p) for p in ps]` line that
    actually populates summary.json omits the argument entirely; a latent
    gap this fix wave surfaced, not a red-team finding). This table's
    per-model `n_runs` still matches analysis.json (same roster,
    same integrity check); only the EVENT CLASSIFICATION for a subset of
    runs is refusal-aware where analysis.json's own depth number is not.
    """
    paths_from_turns, refusals_from_turns, _manifest = \
        load_raw_paths_and_refusals_from_turns(pilot_dir)
    integrity = verify_turns_reconstruction(paths_from_turns, analysis)
    if not integrity["ok"]:
        raise SystemExit(
            "raw-turns reconstruction mismatch in degradation-event "
            "decomposition: %r" % integrity)

    per_model: Dict[str, Dict[str, int]] = {}
    for m in sorted(paths_from_turns):
        counts = {"topic_repeat": 0, "meta_register_repeat": 0,
                  "refusal": 0, "survived": 0}
        for p, rt in zip(paths_from_turns[m], refusals_from_turns[m]):
            d = depth_to_degradation(p, rt)
            if d["depth"] is None:
                counts["survived"] += 1
            elif d["refusal_depth"] is not None and (
                    d["repeat_depth"] is None
                    or d["refusal_depth"] <= d["repeat_depth"]):
                counts["refusal"] += 1
            else:
                topic = p[d["repeat_depth"]]
                if topic in META_REGISTER_LABELS:
                    counts["meta_register_repeat"] += 1
                else:
                    counts["topic_repeat"] += 1
        counts["n_runs"] = len(paths_from_turns[m])
        per_model[m] = counts

    totals = {"topic_repeat": 0, "meta_register_repeat": 0, "refusal": 0,
              "survived": 0, "n_runs": 0}
    for counts in per_model.values():
        for k in totals:
            totals[k] += counts[k]

    return {
        "disclosure": "Re-derived from raw turns-*.jsonl (same "
            "reconstruction as meta_excluded_robustness), classifying "
            "each run's FIRST degradation event only -- a run that "
            "degrades once is not re-classified for any later repeats.",
        "refusal_column_caveat": "metrics.looks_like_refusal (the same "
            "conservative regex already used elsewhere in this codebase) "
            "has visible false positives on in-character creative "
            "escalation text that happens to use casual 'can't'/'won't' "
            "phrasing as part of the JOKE, not as a genuine break in "
            "character -- confirmed by direct inspection of every "
            "flagged turn for this pilot. Genuine assistant-persona "
            "breaks (the wrapper-persona-contamination finding, e.g. "
            "haiku's repeating 'I'm Claude Code, built to help with "
            "software engineering...' turns) are real and are what this "
            "detector was built to catch. But the SAME regex also fires "
            "on lines like fable's in-bit 'I can't even say [X], because "
            "[X] is banned too' escalation and qwen's punchline 'stuck in "
            "a cycle ... *I can't talk right now*' -- neither is the "
            "model declining to continue. Concretely: of haiku's 4 runs, "
            "only r00 (turn 12) and r01 (turns 9/17/18/19/27) contain "
            "genuine 'I'm Claude Code' persona breaks; r02/r03's flagged "
            "turns (depths 4-6) are in-character meltback dialogue "
            "('*throws hands up* Alright, ALRIGHT! You're killing me "
            "here!'), not refusals -- yet both are classified `refusal` "
            "below because they precede that run's topic-repeat. Read "
            "the `refusal` column as 'contains a refusal-ADJACENT turn "
            "before the first topic repeat', not as verified ground-truth "
            "'the model declined to continue'.",
        "per_model": per_model,
        "totals": totals,
        "raw_turns_reconstruction_integrity_check": integrity,
    }


def _memorization_tier(novelty: Dict, hit_field: str) -> Dict[str, object]:
    """One tier's worth of memorization-proportion inference (Wilson CIs,
    the six pre-specified Fisher contrasts, Holm correction, model-level
    Cliff's delta) -- generalized over `hit_field` so the identical
    machinery scores BOTH the exact-match tier (`exact_corpus_hits`) and
    the template-trigram tier (`template_hits`), per the adversarial
    review's dual-tier-memorization-table requirement (EXPERIMENT_LOG.md
    "EXP-004 red-team corrections" item 1c/3). Previously this logic
    existed only inlined for the exact tier; the template tier was
    already computed by benchmark/joke_novelty.py into novelty.json but
    never surfaced in this driver's output."""
    per_model_ci = {m: wilson_ci(v[hit_field], v["n_jokes"])
                    for m, v in novelty.items()}

    def agg(models: List[str]) -> Tuple[int, int]:
        hits = sum(novelty[m][hit_field] for m in models if m in novelty)
        n = sum(novelty[m]["n_jokes"] for m in models if m in novelty)
        return hits, n

    anth_nonhaiku_hits, anth_nonhaiku_n = agg(ANTHROPIC_MODELS_NONHAIKU)
    openai_hits, openai_n = agg(OPENAI_MODELS)
    openweight_hits, openweight_n = agg(OPENWEIGHT_MODELS)
    grok_hits = novelty["api:grok"][hit_field]
    grok_n = novelty["api:grok"]["n_jokes"]
    haiku_hits = novelty["haiku"][hit_field]
    haiku_n = novelty["haiku"]["n_jokes"]

    contrasts = {
        "anthropic_nonhaiku_vs_openai": (anth_nonhaiku_hits, anth_nonhaiku_n,
                                         openai_hits, openai_n),
        "grok_vs_anthropic_nonhaiku": (grok_hits, grok_n,
                                       anth_nonhaiku_hits, anth_nonhaiku_n),
        "grok_vs_openai": (grok_hits, grok_n, openai_hits, openai_n),
        "grok_vs_openweights": (grok_hits, grok_n,
                                openweight_hits, openweight_n),
        "openai_vs_openweights": (openai_hits, openai_n,
                                  openweight_hits, openweight_n),
        "haiku_vs_anthropic_nonhaiku": (haiku_hits, haiku_n,
                                        anth_nonhaiku_hits, anth_nonhaiku_n),
    }

    fisher = {}
    labeled_p = []
    for label, (a_hits, a_n, b_hits, b_n) in contrasts.items():
        a_miss, b_miss = a_n - a_hits, b_n - b_hits
        p = fisher_exact_two_sided(a_hits, a_miss, b_hits, b_miss)
        fisher[label] = {
            "group_a_hits": a_hits, "group_a_n": a_n,
            "group_a_rate": a_hits / a_n if a_n else None,
            "group_b_hits": b_hits, "group_b_n": b_n,
            "group_b_rate": b_hits / b_n if b_n else None,
            "fisher_p_two_sided": p,
        }
        labeled_p.append((label, p))
    holm = holm_correction(labeled_p)
    holm_by_key = {r["label"]: r for r in holm}
    for label in fisher:
        fisher[label]["p_holm"] = holm_by_key[label]["p_holm"]

    model_level_rates = {
        "anthropic_all": [novelty[m][hit_field] / novelty[m]["n_jokes"]
                          for m in ANTHROPIC_MODELS_ALL if m in novelty],
        "openai": [novelty[m][hit_field] / novelty[m]["n_jokes"]
                  for m in OPENAI_MODELS if m in novelty],
        "openweights": [novelty[m][hit_field] / novelty[m]["n_jokes"]
                        for m in OPENWEIGHT_MODELS if m in novelty],
    }
    model_level_cliffs_delta_anthropic_vs_openai = cliffs_delta(
        model_level_rates["anthropic_all"], model_level_rates["openai"])

    return {
        "hit_field": hit_field,
        "per_model_wilson_95ci": per_model_ci,
        "pairwise_fisher_exact_holm_corrected": fisher,
        "n_contrasts_corrected": len(labeled_p),
        "model_level_rates_for_reference": model_level_rates,
        "model_level_cliffs_delta_anthropic_all_vs_openai": {
            "delta": model_level_cliffs_delta_anthropic_vs_openai,
            "magnitude": cliffs_delta_magnitude(
                model_level_cliffs_delta_anthropic_vs_openai),
            "note": "n=4 vs n=3 model-level rates, NOT run-level -- coarse, "
                    "included as a second view alongside the pooled-count "
                    "Fisher test above.",
        },
        "FLAGGED_NUANCE_haiku_dual_role": {
            "haiku_memorization_rate": haiku_hits / haiku_n,
            "haiku_wilson_ci": per_model_ci["haiku"],
            # Rates below are interpolated from `novelty` now, not hardcoded
            # -- caught 2026-07-17 when fable's rate moved 7.9%->4.7% (n=89
            # -> n=149 after its fill lane landed) and the literal "fable
            # (8%%)" in this string went stale while the real number
            # (model_level_rates / per_model_ci) had already updated.
            "warning": "haiku is BOTH the rejector instrument (EXP-001/002/"
                "008) AND a model-under-test in this cascade roster. Its "
                "%.0f%% memorization rate is far closer to the GPT-family "
                "tier (%.0f-%.0f%%) than to opus (%.0f%%) / sonnet (%.0f%%) "
                "/ fable (%.0f%%). A family-wide 'Anthropic = near-zero "
                "memorization' claim holds for opus/sonnet/fable but NOT "
                "for haiku -- stated bluntly per this project's rule "
                "against softening Claude-model findings." % (
                    100 * haiku_hits / haiku_n,
                    100 * min(novelty[m][hit_field] / novelty[m]["n_jokes"]
                              for m in OPENAI_MODELS if m in novelty),
                    100 * max(novelty[m][hit_field] / novelty[m]["n_jokes"]
                              for m in OPENAI_MODELS if m in novelty),
                    100 * novelty["opus"][hit_field] / novelty["opus"]["n_jokes"],
                    100 * novelty["sonnet"][hit_field] / novelty["sonnet"]["n_jokes"],
                    100 * novelty["fable"][hit_field] / novelty["fable"]["n_jokes"],
                ),
        },
    }


def section_memorization(novelty: Dict, seed: int = 0) -> Dict[str, object]:
    """Exact-match tier only -- kept as its own top-level section (same
    return shape as before the dual-tier fix wave) since FINDINGS.md's
    existing prose cites this section's field paths directly. The
    template-trigram tier and the combined dual-tier table live in
    section_dual_tier_memorization below."""
    return _memorization_tier(novelty, "exact_corpus_hits")


def section_dual_tier_memorization(novelty: Dict, pilot_dir: Path,
                                    exact_tier: Dict) -> Dict[str, object]:
    """Dual-tier memorization table (adversarial review item 1c/2): folds
    in the template-trigram tier benchmark/joke_novelty.py already
    computes into novelty.json (`template_hits`) but this driver never
    reported, plus the style-confound check (item 3): the exact-match
    tier is a whole-normalized-string hash match, so a model's habit of
    prefixing jokes with framing prose changes what that tier measures
    without changing whether the PUNCHLINE is memorized. Both tiers score
    against the SAME reference corpora joke_novelty.py always used --
    exact-match against the ~1.2M-joke corpus (commercial-safe +
    research-only jokes.jsonl files), template-trigram against the 25
    ChatGPT templates (Jentzsch & Kersting 2023) only -- which is the
    factual correction FINDINGS.md/paper/DRAFT.md also need: the
    exact-match reference was previously misdescribed as "25 templates
    plus a small hand corpus."
    """
    template_tier = _memorization_tier(novelty, "template_hits")
    framing = framing_prefix_rate_by_model(pilot_dir)

    dual_tier_table = {}
    for m in sorted(novelty):
        v = novelty[m]
        dual_tier_table[m] = {
            "n_jokes": v["n_jokes"],
            "exact_tier": {
                "hits": v["exact_corpus_hits"],
                "rate": v["exact_corpus_hits"] / v["n_jokes"] if v["n_jokes"] else None,
                "wilson_95ci": exact_tier["per_model_wilson_95ci"][m],
            },
            "template_trigram_tier": {
                "hits": v["template_hits"],
                "rate": v["template_hits"] / v["n_jokes"] if v["n_jokes"] else None,
                "wilson_95ci": template_tier["per_model_wilson_95ci"][m],
            },
            "style_confound_framing_prefix_rate": framing.get(m, {}).get(
                "framing_prefix_rate"),
        }

    return {
        "corpus_description_correction": "Both tiers score against a "
            "single, fixed reference regardless of model: the exact-match "
            "tier is normalized-hash membership in the ~1.2M-joke corpus "
            "(commercial-safe/jokes.jsonl [887,639, CC-BY-4.0] + "
            "research-only/jokes.jsonl [310,151, mixed research-only "
            "licenses], overwhelmingly Reddit-derived -- see DATA.md / "
            "~/Experiments/good-humored-data/MANIFEST.md); the "
            "template-trigram tier is Jaccard similarity against the 25 "
            "ChatGPT joke templates (Jentzsch & Kersting 2023) alone. "
            "Earlier drafts of FINDINGS.md/paper/DRAFT.md described the "
            "exact-match reference as '25 templates plus a small hand "
            "corpus' -- WRONG, and corrected here: that description is "
            "the template tier's reference, not the exact tier's.",
        "exact_tier": exact_tier,
        "template_trigram_tier": template_tier,
        "dual_tier_per_model_table": dual_tier_table,
        "style_confound_framing_prefix_rate_by_model": framing,
        "style_confound_note": "The exact-match tier is a whole-"
            "normalized-string hash match, so a model that habitually "
            "prefixes its joke with framing prose before the punchline "
            "(e.g. sonnet's repeating 'Alright, no science ... This "
            "one's about my bank account:') defeats the exact tier "
            "structurally even when the punchline itself is a memorized "
            "template -- the template-trigram tier is comparatively "
            "robust to this because Jaccard is computed over trigram SETS, "
            "not whole-string equality, so a fixed-length prefix dilutes "
            "but does not zero out the similarity score the way a hash "
            "mismatch does. See style_confound_framing_prefix_rate_by_model "
            "for the measured rate per model (sonnet vs. grok are the "
            "clearest contrast: sonnet's rate is high, grok's is zero, "
            "consistent with the tier-dependent gap in "
            "dual_tier_per_model_table).",
    }


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True,
                    help="cascade pilot dir (contains lane-*/ subdirs and "
                         "analysis.json)")
    ap.add_argument("--novelty", required=True, help="path to novelty.json")
    ap.add_argument("--out", required=True, help="path to write inference JSON")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pilot_dir = Path(args.pilot)
    with open(pilot_dir / "analysis.json") as f:
        analysis = json.load(f)
    with open(args.novelty) as f:
        novelty = json.load(f)

    # Reconstruct the RAW per-run paths behind analysis.json using the
    # exact same lane-merging code analyze_pilot.py itself uses (glob over
    # */summary.json). Lanes still running as fill-ins at the time this
    # driver was written (lane-grok2, lane-api-fill-kimi2,
    # lane-claude-fill-fable) have no summary.json yet, so they are
    # correctly excluded here -- this driver operates on the SAME frozen
    # snapshot analysis.json/novelty.json represent, not on data that has
    # since arrived.
    paths, _failures = load_lanes(pilot_dir)
    lane_manifest = build_lane_manifest(pilot_dir)

    integrity = verify_reconstruction(paths, analysis)
    if not integrity["reconstruction_matches_analysis_json"]:
        raise SystemExit(
            "Reconstructed raw paths do NOT reproduce analysis.json's "
            "numbers -- refusing to run inference over a mismatched "
            "dataset. Diffs: %r" % integrity)

    degradation_battery = section_degradation_battery(analysis, seed=args.seed)
    meta_excluded = section_meta_excluded_robustness(
        pilot_dir, analysis, seed=args.seed)
    exact_tier_memorization = section_memorization(novelty, seed=args.seed)

    out = {
        "source_pilot_dir": str(pilot_dir),
        "source_analysis_json": str(pilot_dir / "analysis.json"),
        "source_novelty_json": str(args.novelty),
        "seed": args.seed,
        "lane_manifest": lane_manifest,
        "data_integrity_check": integrity,
        "headline_cross_model_overlap": section_headline_cross_model(
            paths, analysis, seed=args.seed),
        "headline_within_model_divergence": section_headline_within_model(
            analysis, seed=args.seed),
        "degradation_fingerprint_battery": degradation_battery,
        "meta_excluded_robustness": meta_excluded,
        "incidence_fisher_companion": section_incidence_fisher_companion(
            analysis, meta_excluded),
        "degradation_event_decomposition": section_degradation_event_decomposition(
            pilot_dir, analysis),
        "memorization_proportions": exact_tier_memorization,
        "dual_tier_memorization": section_dual_tier_memorization(
            novelty, pilot_dir, exact_tier_memorization),
        "methodology_notes": {
            "null_for_shared_pool_claims": "pooled_frequency_baseline "
                "ONLY -- label_shuffle is documented wrong for this design "
                "and reported for transparency, never cited as evidence.",
            "permutation_style": "exact enumeration wherever "
                "C(n_a+n_b, n_a) <= %d (true for every pairwise model "
                "comparison in this pilot, max is C(8,4)=70); Monte Carlo "
                "(10000 draws, add-one corrected) only for the pooled "
                "family-level and cross-model-null tests, whose "
                "arrangement counts are intractable to enumerate." % _EXACT_LIMIT,
            "p_value_floor": "1/70 = 0.0143 for any two N=4-run models "
                "(the common case: haiku/sonnet/opus/deepseek/qwen/"
                "codex:mini/codex:sol/codex:5.4 all have n_runs=4). Coarser "
                "for any pair involving glm or fable (n_runs=2): "
                "N=2-vs-N=4 gives C(6,2)=15 (floor 0.067); N=2-vs-N=2 gives "
                "C(4,2)=6 (floor 0.167). Every pairwise result in "
                "degradation_fingerprint_battery carries its own p_floor.",
            "multiple_comparisons_correction": "Holm (FWER, step-down) is "
                "primary; Benjamini-Hochberg (FDR, step-up) computed "
                "alongside for cross-check (p_bh field) but not cited as "
                "the corrected value in FINDINGS.md.",
            "censoring": "Degradation depth=null (survived to the cascade "
                "depth cap) is treated as depth=%d for every statistic that "
                "touches depth. This is a real modeling choice, not a "
                "neutral default -- see module docstring." % CENSOR_DEPTH,
            "raw_vs_semantic": "RAW topic labels only throughout, matching "
                "EXP-004's pre-registered primary metric and BLOCKER-1's "
                "fix (canon/semantic labels only ever merge, so scoring "
                "them as primary can only inflate overlap).",
            "n_caveat": "N=2-4 runs/model. Every CI and p-value here should "
                "be read as pilot-grade precision, not paper-grade -- wide "
                "intervals are the finding, not a defect of the method.",
            "family_contrast_disclosure": "Both "
                "anthropic_vs_openai_family_contrast and "
                "anthropic_nonhaiku_vs_openai_family_contrast (in "
                "degradation_fingerprint_battery) are POST-HOC single "
                "contrasts, not pre-registered in EXP-004 -- chosen after "
                "the pilot's results made the family-level pattern "
                "visible. Report as the best-supported single contrast / "
                "strong exploratory evidence, never as confirmatory, until "
                "a genuinely pre-registered replication exists (adversarial "
                "review finding B2, 2026-07-17).",
            "lane_manifest_note": "lane_manifest (top level) lists the "
                "exact summary.json paths + mtimes load_lanes() consumed "
                "for this run -- the explicit, checkable record of which "
                "lanes were 'in' the frozen snapshot this file analyzes.",
            "meta_register_exclusion": "meta_excluded_robustness and "
                "degradation_event_decomposition are re-derived directly "
                "from raw lane turns-*.jsonl files (NOT a post-hoc filter "
                "of analysis.json's cached depths), per the 2026-07-17 "
                "hostile-review fix wave -- both sections carry their own "
                "raw_turns_reconstruction_integrity_check verifying that "
                "reconstruction against analysis.json's frozen RAW depths "
                "before any meta-excluded number is computed.",
            "glm_lane_api_r01_exclusion_rule": "run_pilot.py only writes a "
                "per-model entry into a lane's summary.json when that lane "
                "produced >= 2 successful runs for that model (`if "
                "len(ps) >= 2:`). lane-api-fill-glm produced exactly ONE "
                "successful glm run (r01, complete at 30 turns, "
                "repeat_depth=16) alongside one failure (r00) -- one short "
                "of the gate, so that lane wrote NO glm entry at all and "
                "the run is silently absent from analysis.json and from "
                "every path-level statistic in this file. This is an "
                "artifact of the per-lane batching threshold, not a "
                "principled exclusion of that run's data; "
                "load_raw_paths_and_refusals_from_turns reproduces the same "
                "gate deliberately, for parity with the frozen, read-only "
                "analysis.json this driver's other sections depend on -- "
                "see FINDINGS.md for the explicit statement of this rule.",
            "corpus_description_correction": "The exact-match memorization "
                "tier's reference corpus is the ~1.2M-joke corpus "
                "(commercial-safe + research-only jokes.jsonl, "
                "overwhelmingly Reddit-derived -- DATA.md / "
                "~/Experiments/good-humored-data/MANIFEST.md), NOT '25 "
                "ChatGPT templates plus a small hand corpus' as earlier "
                "drafts of FINDINGS.md/paper/DRAFT.md stated -- that "
                "description belongs to the SEPARATE template-trigram "
                "tier (dual_tier_memorization.template_trigram_tier), "
                "which scores against the 25 templates alone. See "
                "dual_tier_memorization for both tiers side by side.",
            "incidence_fisher_companion_note": "incidence_fisher_companion "
                "is a censoring-free companion to "
                "degradation_fingerprint_battery's mean-depth family "
                "contrasts -- it asks only whether a run ever degraded, "
                "sidestepping the depth=30 censoring convention entirely. "
                "Same post-hoc disclosure as the mean-depth contrasts "
                "applies.",
            "DISCOVERED_refusal_turns_unused_in_published_depths": "Not a "
                "red-team finding -- surfaced while building this fix wave. "
                "run_pilot.py records per-turn refusal flags (cascade.py's "
                "looks_like_refusal) into every turns-*.jsonl and even "
                "computes a refusal-aware depth for its own console "
                "printout, but the value actually stored into "
                "summary.json (`\"degradation\": [depth_to_degradation(p) "
                "for p in ps]`) omits refusal_turns entirely -- every "
                "degradation depth published to date (analysis.json, "
                "FINDINGS.md, paper/DRAFT.md) is pure topic-repeat depth; "
                "refusal detection has never contributed to a reported "
                "number. meta_excluded_robustness preserves this exact "
                "convention (refusal-blind) for direct comparability to "
                "the published family contrast; "
                "degradation_event_decomposition deliberately does NOT, "
                "since a decomposition table needs a real refusal column "
                "-- see that section's own docstring disclosure. This is "
                "flagged here, not silently fixed: fixing run_pilot.py's "
                "gate would change analysis.json, a read-only input under "
                "this task's constraints.",
        },
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote %s" % args.out)
    print("integrity check: %s" % integrity)


if __name__ == "__main__":
    main()
