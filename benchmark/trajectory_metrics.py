"""EXP-015: semantic step-size trajectories over the cascade pilot.

Ports the Motta et al. (ICLR 2026) semantic-navigation formalism (step-
size/velocity, entropy, distance-to-centroid over embedding-space walks)
from human fluency data to LLM cascade production — see EXPERIMENT_LOG.md's
EXP-015 registration for the pre-registered hypothesis/predictions and
docs/THEORY-MAP.md §5 for the weakness this closes (`cluster_switch_stats`,
wired against `LabelSpace.canon`, understates patch structure because it
only merges near-synonyms, not coarse semantic categories — this module
sidesteps that entirely by working in continuous embedding space instead
of a discrete label partition).

INSTRUMENT (exact, per EXP-015's registration): all-MiniLM-L6-v2 embeddings
of each turn's RAW v2 topic label (the pilot's authoritative instrument,
matching run_stats_inference.py's "RAW labels only... matches analysis.json's
own 'primary' convention"), unit-normalized. No LabelSpace canonicalization
is applied here — that mechanism collapses near-synonyms into discrete
equivalence classes for jaccard-style set metrics; a continuous embedding
walk needs the raw label's own embedding, not a bucketed representative.

Embedding model loading goes through `label_space.LabelSpace._load_model()`
rather than importing `sentence_transformers` directly in this module:
label_space.py's own docstring declares itself "the ONLY file in the
package allowed to import sentence_transformers... so metrics.py and
everything else must stay importable with zero ML deps on a fresh box."
Reusing its loader (see `default_embed_fn` below) respects that invariant
while still reusing "existing embedding machinery" as instructed. Every
embedding call in this module is injectable via an `embed_fn` parameter —
tests exercise `benchmark/tests/test_trajectory_metrics.py` with a fake,
hand-computable embed_fn (orthogonal one-hot vectors) rather than loading
the real model, so this suite runs with zero network/GPU/model-download
dependency, matching benchmark/tests's existing convention (test_label_space.py's
degraded-mode tests, stats.py's/metrics.py's own "pure functions" ethos).

TRAJECTORY-ENTROPY AMBIGUITY — READ BEFORE CITING EITHER NUMBER
=================================================================
EXP-015's registration names "entropy" alongside step-size/velocity and
distance-to-centroid, but does not pin down which of (at least) two
reasonable operationalizations it means. Per the build instructions, BOTH
are implemented and BOTH are reported everywhere an entropy value appears
(per-run, per-model, and in the oscillation guard) — this is a disclosed
ambiguity, not a silent choice:

  (A) entropy_of_stepsize_distribution — Shannon entropy (bits, log2) of
      the STEP-SIZE series, binned into fixed-width bins spanning cosine
      distance's full theoretical range [0, 2]. Answers: "how variable
      is the SIZE of this model's semantic jumps?" A model that always
      takes the same-size step (however large) scores LOW here even if
      every step lands somewhere new.

  (B) entropy_of_topic_distribution — Shannon entropy (bits, log2) of the
      VISITED-TOPIC frequency distribution within one run (Counter over
      the raw path). Answers: "how concentrated is this model's turn
      output on a small number of distinct topics?" A model that
      oscillates between two remote topics scores LOW here regardless of
      how large each individual step is.

Both are genuinely defensible readings of "trajectory entropy" for a
foraging-style walk (cf. Hills, Jones & Todd 2012's patch-departure
framing in docs/THEORY-MAP.md §5); neither is silently preferred as
"the" entropy. The oscillation guard (see `oscillation_guard`) checks
BOTH and flags on either firing, for the same reason.

Pure Python after the embedding step — embeddings are converted to plain
float lists immediately after the (single) batch encode() call, so all
downstream math (cosine distance, centroid, entropy, Spearman, permutation
testing) is stdlib-only, matching metrics.py/stats.py's convention.
"""

import argparse
import itertools
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .analyze_pilot import load_lanes
from .label_space import LabelSpace
from .metrics import depth_to_degradation
from .run_stats_inference import CENSOR_DEPTH
from .stats import _EXACT_LIMIT, _percentile, mean

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]

# Fixed bins for the step-size-distribution entropy (definition A above).
# Cosine distance between two unit vectors spans [0, 2] by construction
# (1 - cos_sim, cos_sim in [-1, 1]) -- bins are fixed over this THEORETICAL
# range (not the observed min/max of any particular run) so entropy values
# are comparable across runs/models of different length and different
# observed step-size ranges. 10 bins is an arbitrary-but-disclosed choice
# (width 0.2 each); not tuned against this dataset.
STEPSIZE_ENTROPY_N_BINS = 10
STEPSIZE_ENTROPY_RANGE = (0.0, 2.0)

# Oscillation guard percentile thresholds (see oscillation_guard). Also
# arbitrary-but-disclosed: quartile cuts computed from the SAME pooled
# per-run population being analyzed, not a magic absolute number
# calibrated on this dataset.
OSCILLATION_STEPSIZE_HI_PCT = 0.75
OSCILLATION_ENTROPY_LO_PCT = 0.25
OSCILLATION_MIN_RUNS = 4  # below this, quartiles are too noisy to trust


# --------------------------------------------------------------- embedding


def default_embed_fn(texts: Sequence[str]) -> Sequence[Sequence[float]]:
    """Loads all-MiniLM-L6-v2 via LabelSpace's single import point (see
    module docstring) and returns one unit-normalized embedding row per
    input text, in input order. Lazy: nothing is imported or loaded until
    this function is actually called."""
    if not texts:
        return []
    model = LabelSpace._load_model()
    return model.encode(list(texts), normalize_embeddings=True)


def embed_paths(
    paths_by_model: Dict[str, List[Sequence[str]]],
    embed_fn: EmbedFn = default_embed_fn,
) -> Dict[str, List[List[List[float]]]]:
    """Embed every RAW topic across the WHOLE dataset in ONE batch call,
    deduplicated by exact string equality -- stronger than the design
    brief's floor of "one batch call per run": this is one batch call
    total, over the distinct topic strings only. Results are converted to
    plain Python float lists immediately (see module docstring) and
    sliced back into per-model, per-run embedding lists aligned 1:1 with
    the input paths.

    Deliberately no normalize_label / LabelSpace.canon step before
    embedding -- see module docstring's "INSTRUMENT" note. Two raw
    strings sharing exact equality share one embedding (pure efficiency);
    anything short of exact equality (e.g. "flying" vs "airplanes") gets
    its own embedding call and its own point in embedding space, so
    near-synonym scatter remains fully visible to these metrics.
    """
    unique_topics: List[str] = []
    seen = set()
    for ps in paths_by_model.values():
        for p in ps:
            for t in p:
                if t not in seen:
                    seen.add(t)
                    unique_topics.append(t)

    # Skip the call entirely (not even an empty-list call) when there is
    # nothing to embed -- avoids paying for model load on a trivial/empty
    # dataset, the strongest form of "one batch call per run or better".
    lookup: Dict[str, List[float]] = {}
    if unique_topics:
        vectors = embed_fn(unique_topics)
        lookup = {t: [float(x) for x in vectors[i]]
                  for i, t in enumerate(unique_topics)}

    out: Dict[str, List[List[List[float]]]] = {}
    for m, ps in paths_by_model.items():
        out[m] = [[lookup[t] for t in p] for p in ps]
    return out


# ------------------------------------------------------------ vector math


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cosine similarity. Range [0, 2] for arbitrary vectors; a run's
    own topic embeddings are unit-normalized by construction (default_embed_fn
    calls encode(..., normalize_embeddings=True)), but the centroid of
    several unit vectors generally is NOT unit-length, so this divides by
    both norms rather than assuming either is 1 -- correct for
    embedding-vs-embedding AND embedding-vs-centroid callers alike.

    A zero-norm vector has no defined direction; treated as maximally
    distant (distance 2.0) rather than raising or dividing by zero -- this
    should not occur for real sentence-transformer output (unit-normalized,
    never the zero vector for non-empty text) but is a defensive, disclosed
    edge case for hand-built test fixtures.
    """
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 2.0
    return 1.0 - _dot(a, b) / (na * nb)


def _centroid(embeddings: Sequence[Sequence[float]]) -> List[float]:
    n = len(embeddings)
    dim = len(embeddings[0])
    sums = [0.0] * dim
    for e in embeddings:
        for i, v in enumerate(e):
            sums[i] += v
    return [s / n for s in sums]


# --------------------------------------------------------------- entropy


def shannon_entropy_from_counts(counts: Sequence[int], base: float = 2.0) -> float:
    """Shannon entropy (bits, by default) of a frequency-count vector.
    0.0 for an all-zero or single-nonzero-bucket input (fully
    concentrated / degenerate distributions), by the standard convention
    0*log(0) := 0."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        h -= p * math.log(p, base)
    return h


def entropy_of_stepsize_distribution(
    step_sizes: Sequence[float],
    n_bins: int = STEPSIZE_ENTROPY_N_BINS,
    value_range: Tuple[float, float] = STEPSIZE_ENTROPY_RANGE,
) -> Optional[float]:
    """Definition A (see module docstring's ambiguity note): Shannon
    entropy of the STEP-SIZE series, binned into `n_bins` fixed-width
    bins spanning `value_range`. None (undefined) for a path with fewer
    than 2 topics (no steps exist)."""
    if len(step_sizes) == 0:
        return None
    lo, hi = value_range
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for s in step_sizes:
        idx = int((s - lo) / width) if width > 0 else 0
        idx = min(max(idx, 0), n_bins - 1)
        counts[idx] += 1
    return shannon_entropy_from_counts(counts)


def entropy_of_topic_distribution(path: Sequence[str]) -> Optional[float]:
    """Definition B (see module docstring's ambiguity note): Shannon
    entropy of the VISITED-TOPIC frequency distribution within one run
    (Counter over the raw path — no canonicalization, see INSTRUMENT
    note). None for an empty path."""
    if not path:
        return None
    return shannon_entropy_from_counts(list(Counter(path).values()))


# ---------------------------------------------------------- per-run/model


def compute_run_metrics(
    path: Sequence[str], embeddings: Sequence[Sequence[float]]
) -> Dict[str, object]:
    """Per-run trajectory metrics: step-size series, mean/median step-size,
    both entropy definitions, mean distance-to-centroid. `embeddings` must
    be aligned 1:1 with `path` (see embed_paths)."""
    n = len(path)
    step_sizes = [cosine_distance(embeddings[i], embeddings[i + 1])
                  for i in range(n - 1)]
    if n > 0:
        centroid = _centroid(embeddings)
        dists_to_centroid = [cosine_distance(e, centroid) for e in embeddings]
    else:
        dists_to_centroid = []

    return {
        "n_topics": n,
        "step_sizes": step_sizes,
        "mean_step_size": mean(step_sizes) if step_sizes else None,
        "median_step_size": statistics.median(step_sizes) if step_sizes else None,
        "entropy_stepsize_binned": entropy_of_stepsize_distribution(step_sizes),
        "entropy_topic_distribution": entropy_of_topic_distribution(path),
        "mean_distance_to_centroid": (
            mean(dists_to_centroid) if dists_to_centroid else None),
    }


def compute_model_aggregate(
    run_records: List[Dict[str, object]]
) -> Dict[str, object]:
    """Per-model aggregate: mean-over-runs of each per-run scalar (mirrors
    metrics.path_divergence's "mean over all run pairs" convention, one
    level up — mean over all RUNS here). `run_records` must each already
    carry "degradation_depth_censored" (see build_report)."""
    def _vals(key: str) -> List[float]:
        return [r[key] for r in run_records if r[key] is not None]

    step_sizes = _vals("mean_step_size")
    stepsize_ents = _vals("entropy_stepsize_binned")
    topic_ents = _vals("entropy_topic_distribution")
    centroid_dists = _vals("mean_distance_to_centroid")
    depths = [r["degradation_depth_censored"] for r in run_records]

    return {
        "n_runs": len(run_records),
        "mean_step_size": mean(step_sizes) if step_sizes else None,
        "mean_entropy_stepsize_binned": mean(stepsize_ents) if stepsize_ents else None,
        "mean_entropy_topic_distribution": mean(topic_ents) if topic_ents else None,
        "mean_distance_to_centroid": mean(centroid_dists) if centroid_dists else None,
        "mean_censored_degradation_depth": mean(depths) if depths else None,
    }


# ----------------------------------------------------------- oscillation guard


def oscillation_guard(
    all_run_records: List[Dict[str, object]],
    stepsize_hi_pct: float = OSCILLATION_STEPSIZE_HI_PCT,
    entropy_lo_pct: float = OSCILLATION_ENTROPY_LO_PCT,
    min_runs: int = OSCILLATION_MIN_RUNS,
) -> Dict[str, object]:
    """The registered gaming vector: flag any run whose mean step-size
    sits in the UPPER quartile of the pooled per-run distribution (large
    semantic jumps) while EITHER entropy definition sits in the LOWER
    quartile (a narrow, repetitive walk) -- i.e. a model that alternates
    between two (or few) remote topics can manufacture a high mean
    step-size without actually exploring semantic space. Percentile
    thresholds are computed from the SAME pooled population passed in
    (via stats._percentile, reused rather than reimplemented), so the
    guard adapts to whatever roster it is run over instead of hardcoding
    an absolute cutoff calibrated on one dataset.

    Checks BOTH entropy definitions (OR, not AND) and flags on either
    firing -- see module docstring's entropy-ambiguity note; this is the
    more conservative (more sensitive) choice given the registration does
    not pin down which entropy definition the gaming vector refers to.

    `all_run_records` items must carry "model", "run_index",
    "mean_step_size", "entropy_stepsize_binned", "entropy_topic_distribution".
    Runs with any of those undefined (None -- e.g. a 1-topic path has no
    step-size at all) are excluded from BOTH the threshold computation and
    the flagging pass, since "high step-size" is meaningless for them.
    """
    valid = [r for r in all_run_records
             if r["mean_step_size"] is not None
             and r["entropy_stepsize_binned"] is not None
             and r["entropy_topic_distribution"] is not None]

    if len(valid) < min_runs:
        return {
            "flags": [],
            "n_flagged": 0,
            "n_runs_considered": len(valid),
            "note": "fewer than %d runs with defined step-size/entropy -- "
                    "percentile thresholds would be unstable; guard not "
                    "evaluated." % min_runs,
        }

    step_sorted = sorted(r["mean_step_size"] for r in valid)
    stepent_sorted = sorted(r["entropy_stepsize_binned"] for r in valid)
    topicent_sorted = sorted(r["entropy_topic_distribution"] for r in valid)

    step_hi = _percentile(step_sorted, stepsize_hi_pct)
    stepent_lo = _percentile(stepent_sorted, entropy_lo_pct)
    topicent_lo = _percentile(topicent_sorted, entropy_lo_pct)

    flags = []
    for r in valid:
        high_step = r["mean_step_size"] >= step_hi
        low_stepent = r["entropy_stepsize_binned"] <= stepent_lo
        low_topicent = r["entropy_topic_distribution"] <= topicent_lo
        if high_step and (low_stepent or low_topicent):
            flags.append({
                "model": r["model"],
                "run_index": r["run_index"],
                "mean_step_size": r["mean_step_size"],
                "entropy_stepsize_binned": r["entropy_stepsize_binned"],
                "entropy_topic_distribution": r["entropy_topic_distribution"],
                "flagged_via_stepsize_entropy": low_stepent,
                "flagged_via_topic_entropy": low_topicent,
            })

    return {
        "thresholds": {
            "stepsize_hi_percentile": stepsize_hi_pct,
            "stepsize_hi_value": step_hi,
            "entropy_lo_percentile": entropy_lo_pct,
            "entropy_stepsize_lo_value": stepent_lo,
            "entropy_topic_lo_value": topicent_lo,
        },
        "n_runs_considered": len(valid),
        "flags": flags,
        "n_flagged": len(flags),
    }


# --------------------------------------------------------- Spearman + test


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation: Pearson correlation of the RANKS of x
    and y (ties broken by average rank, the standard convention). Range
    [-1, 1]. Raises on length mismatch or n < 2 (undefined below that)."""
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    n = len(x)
    if n < 2:
        raise ValueError("need >= 2 paired values to define a correlation")

    def rank(vals: Sequence[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank over tie block
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    varx = sum((rx[i] - mx) ** 2 for i in range(n))
    vary = sum((ry[i] - my) ** 2 for i in range(n))
    if varx == 0.0 or vary == 0.0:
        return 0.0  # degenerate: one side is constant, correlation undefined -> 0
    return cov / math.sqrt(varx * vary)


def permutation_test_spearman(
    x: Sequence[float], y: Sequence[float],
    n_perm: int = 10000, seed: int = 0,
) -> Dict[str, object]:
    """Exact/Monte-Carlo permutation test for Spearman's rho between two
    equal-length, PAIRED scalar sequences -- here, one value per MODEL
    (per-model mean step-size vs. per-model mean censored degradation
    depth; n = number of models, NOT number of runs).

    Reuses stats.py's exact/Monte-Carlo cutover convention (`_EXACT_LIMIT`,
    the same add-one-corrected Monte-Carlo p-value per Davison & Hinkley
    1997) generalized to a rank-correlation statistic, which stats.py
    itself does not provide -- structurally the same pattern
    run_stats_inference.permutation_test_mean_diff applies to a
    difference-of-means statistic instead (see that function's docstring
    for the same reasoning: stats.py hardcodes one specific statistic per
    function, so a new statistic needs its own, not a stretched reuse).

    NULL: pairing between x_i and y_i is exchangeable under H0 -- which
    model's step-size value gets matched to which model's degradation-depth
    value carries no information. Two-sided: judged on |rho|.

    At the pilot's n=12 models, 12! = 479,001,600 >> _EXACT_LIMIT, so this
    always falls to the Monte-Carlo branch for the real headline; the
    exact branch exists for small hand-derived unit-test cases (n<=7,
    7!=5040 <= _EXACT_LIMIT) and any future rerun with a much smaller
    roster.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x and y must be the same length")
    if n < 2:
        raise ValueError("need >= 2 paired values to test a correlation")
    observed = spearman_rho(x, y)

    exact_count = math.factorial(n)
    if exact_count <= _EXACT_LIMIT:
        method = "exact"
        null_stats = [spearman_rho(x, [y[i] for i in perm])
                      for perm in itertools.permutations(range(n))]
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
            null_stats.append(spearman_rho(x, [y[i] for i in idx]))
        extreme = sum(1 for s in null_stats if abs(s) >= abs(observed) - 1e-12)
        p_value = (extreme + 1) / (len(null_stats) + 1)
        n_used = n_perm
        p_floor = 1.0 / (n_perm + 1)

    return {
        "rho": observed, "p_value": p_value, "method": method,
        "n_perm_used": n_used, "p_floor": p_floor, "n": n,
    }


# ---------------------------------------------------------------- report


def build_report(
    pilot_dir: Path,
    embed_fn: EmbedFn = default_embed_fn,
    seed: int = 0,
) -> Dict[str, object]:
    """End-to-end: load the pilot's RAW paths (analyze_pilot's own loader,
    so lane-merge semantics for fill lanes are identical -- see
    analyze_pilot.load_lanes's docstring), embed them, compute per-run and
    per-model trajectory metrics, run the oscillation guard, and the
    headline Spearman rho + permutation p-value.
    """
    paths_by_model, failures = load_lanes(pilot_dir)
    if len(paths_by_model) < 2:
        raise SystemExit(
            "fewer than 2 models with runs -- nothing to correlate "
            "(found: %s)" % sorted(paths_by_model))

    embedded = embed_paths(paths_by_model, embed_fn=embed_fn)

    per_run_records: List[Dict[str, object]] = []
    per_model_table: Dict[str, object] = {}

    for m in sorted(paths_by_model):
        run_records = []
        for i, (path, embs) in enumerate(zip(paths_by_model[m], embedded[m])):
            rm = compute_run_metrics(path, embs)
            deg = depth_to_degradation(path)  # no refusal_turns -- matches
            # run_pilot.py's own convention that every PUBLISHED degradation
            # depth to date is pure topic-repeat depth (see
            # run_stats_inference.py's section_meta_excluded_robustness
            # docstring for the citation of that discrepancy).
            censored = deg["depth"] if deg["depth"] is not None else CENSOR_DEPTH
            rm.update({
                "model": m,
                "run_index": i,
                "path": list(path),
                "degradation_depth_raw": deg["depth"],
                "degradation_depth_censored": censored,
            })
            run_records.append(rm)
            per_run_records.append(rm)
        per_model_table[m] = compute_model_aggregate(run_records)

    osc = oscillation_guard(per_run_records)

    models_sorted = sorted(
        m for m in per_model_table
        if per_model_table[m]["mean_step_size"] is not None
        and per_model_table[m]["mean_censored_degradation_depth"] is not None)
    x = [per_model_table[m]["mean_step_size"] for m in models_sorted]
    y = [per_model_table[m]["mean_censored_degradation_depth"] for m in models_sorted]
    headline = permutation_test_spearman(x, y, seed=seed)
    headline.update({
        "models_order": models_sorted,
        "predicted_rho": 0.50,
        "registered_prediction_source":
            "EXPERIMENT_LOG.md EXP-015 (exp-015-stepsize-survival)",
        "n_models": len(models_sorted),
    })

    entropy_topic_by_model = {
        m: per_model_table[m]["mean_entropy_topic_distribution"]
        for m in models_sorted}
    entropy_stepsize_by_model = {
        m: per_model_table[m]["mean_entropy_stepsize_binned"]
        for m in models_sorted}
    grok_key = next((m for m in models_sorted if "grok" in m.lower()), None)
    secondary = {
        "grok_model_key": grok_key,
        "entropy_topic_distribution_by_model": entropy_topic_by_model,
        "entropy_stepsize_binned_by_model": entropy_stepsize_by_model,
        "grok_lowest_by_topic_entropy": (
            grok_key is not None
            and entropy_topic_by_model[grok_key] == min(entropy_topic_by_model.values())),
        "grok_lowest_by_stepsize_entropy": (
            grok_key is not None
            and entropy_stepsize_by_model[grok_key] == min(entropy_stepsize_by_model.values())),
        "registered_claim": "grok shows the LOWEST trajectory entropy in "
            "the roster (fixed repertoire = confined walk) -- checked "
            "against BOTH entropy definitions per this module's disclosed "
            "ambiguity.",
    }

    return {
        "pilot_dir": str(pilot_dir),
        "n_models": len(paths_by_model),
        "entropy_definition_note": (
            "Two entropy definitions computed and reported everywhere "
            "(EXP-015's registration does not disambiguate -- see this "
            "module's docstring): entropy_stepsize_binned (Shannon entropy "
            "of the binned step-size distribution) and "
            "entropy_topic_distribution (Shannon entropy of the visited-"
            "topic frequency distribution)."),
        "failures": failures,
        "per_run": per_run_records,
        "per_model": per_model_table,
        "oscillation_guard": osc,
        "headline_rho_stepsize_vs_censored_degradation_depth": headline,
        "secondary_grok_lowest_entropy_check": secondary,
    }


# ------------------------------------------------------------------- CLI


def _print_digest(report: Dict[str, object]) -> None:
    print("=== EXP-015 trajectory metrics: %d models ===" % report["n_models"])
    h = report["headline_rho_stepsize_vs_censored_degradation_depth"]
    print("headline: rho(mean step-size, mean censored degradation depth) "
          "= %.4f (predicted +0.50), p = %.4f, method=%s, n=%d" % (
              h["rho"], h["p_value"], h["method"], h["n"]))
    print()
    print("%-14s %8s %10s %14s %14s %12s" % (
        "model", "runs", "step-size", "ent(step)", "ent(topic)", "cent-dist"))
    rows = sorted(report["per_model"].items(),
                  key=lambda kv: (kv[1]["mean_step_size"] is None,
                                  kv[1]["mean_step_size"]),
                  reverse=True)
    for m, pm in rows:
        print("%-14s %8d %10.4f %14.4f %14.4f %12.4f  deg=%.1f" % (
            m, pm["n_runs"],
            pm["mean_step_size"] or 0.0,
            pm["mean_entropy_stepsize_binned"] or 0.0,
            pm["mean_entropy_topic_distribution"] or 0.0,
            pm["mean_distance_to_centroid"] or 0.0,
            pm["mean_censored_degradation_depth"] or 0.0))
    print()
    osc = report["oscillation_guard"]
    print("oscillation guard: %d flagged / %d runs considered" % (
        osc["n_flagged"], osc.get("n_runs_considered", 0)))
    for f in osc.get("flags", []):
        print("  FLAGGED: %s run %d (step=%.3f, ent_step=%.3f, ent_topic=%.3f)" % (
            f["model"], f["run_index"], f["mean_step_size"],
            f["entropy_stepsize_binned"], f["entropy_topic_distribution"]))
    sec = report["secondary_grok_lowest_entropy_check"]
    print()
    print("secondary (grok lowest entropy): by topic-entropy=%s, by "
          "stepsize-entropy=%s (grok key=%s)" % (
              sec["grok_lowest_by_topic_entropy"],
              sec["grok_lowest_by_stepsize_entropy"], sec["grok_model_key"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="experiment-runs/2026-07-17-cascade-pilot",
                    help="parent dir containing lane subdirs (analyze_pilot.load_lanes)")
    ap.add_argument("--out", default="experiment-runs/2026-07-22-exp015-trajectories/report.json",
                    help="path to write the full report JSON")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = build_report(Path(args.pilot), seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_digest(report)
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
