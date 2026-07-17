"""Inferential layer over benchmark/metrics.py's descriptive path metrics.

metrics.py answers "what happened" (divergence, overlap, degradation).
This module answers "how sure are we" — permutation tests, bootstrap CIs,
and small-N-appropriate effect sizes over the same path data. Pure
functions, stdlib only, Python 3.9 compatible (per repo convention: see
label_space.py's "zero ML deps on a fresh box" note — numpy is importable
in this dev environment (checked: numpy 2.0.2) but deliberately NOT used
here, so this module stays importable anywhere run_pilot.py/analyze_pilot.py
are, and so results are reproducible byte-for-byte off a stock Python 3.9).
At the data sizes this benchmark produces (N=4 runs/model, <=12 models),
stdlib loops are fast enough that numpy would only buy style, not speed.

Method lineage:
- exact permutation tests: Fisher (1935); preferred over Monte Carlo
  whenever the arrangement count is small enough to enumerate (it is,
  here — see _EXACT_LIMIT).
- percentile bootstrap: Efron & Tibshirani (1993, ch. 13).
- Cliff's delta / rank-biserial correlation: Cliff (1993); Romano et al.
  (2006) for the magnitude bands used in cliffs_delta_magnitude. Chosen
  over Cohen's d because d is biased upward at n < 20 per group (this
  repo's own statistical-analysis skill notes this) and assumes
  normality this benchmark's N=4 samples cannot support checking.

Honesty notes threaded through every docstring below, per project
convention: state what each null hypothesis does and does NOT cover.
Small N is a real limitation of the pilot design (see EXP-004, N=4
runs/model) — these functions report that honestly (wide CIs, coarse
discrete p-value grids) rather than dressing it up.
"""

import itertools
import math
import random
from typing import Callable, Dict, List, Optional, Sequence

from .metrics import cross_model_overlap, jaccard

# Enumerate exactly below this many arrangements; sample above it. At
# N=4 runs/model (EXP-004's design), a two-model comparison is C(8,4)=70
# and a bootstrap over one model's 4 runs is 4**4=256 — both trivially
# exact. This constant exists for when someone reruns this with a
# different N and the arrangement count blows up.
_EXACT_LIMIT = 100_000


# ---------------------------------------------------------------- helpers


def mean(xs: Sequence[float]) -> float:
    """Arithmetic mean. Default stat_fn for bootstrap_ci."""
    if not xs:
        raise ValueError("mean of empty sequence")
    return sum(xs) / len(xs)


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (matches numpy's default 'linear'
    method) of an already-sorted sequence. q in [0, 1]."""
    if not sorted_vals:
        raise ValueError("empty sequence has no percentile")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[int(idx)]
    frac = idx - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _mean_pairwise_jaccard(paths: List[Sequence[str]]) -> float:
    """Mean pairwise set-Jaccard within one group of runs (>=2 required).

    Deliberately leaner than metrics.path_divergence for use inside a
    permutation loop that may run up to _EXACT_LIMIT times: path_divergence
    also computes norm_edit_distance, which is O(len_a * len_b) per pair
    (vs. jaccard's O(len)) and would be pure waste multiplied over
    thousands of permutations we only need set_jaccard from.
    """
    if len(paths) < 2:
        raise ValueError("need >= 2 runs to define within-group divergence")
    vals = [jaccard(paths[i], paths[j])
            for i in range(len(paths)) for j in range(i + 1, len(paths))]
    return sum(vals) / len(vals)


# --------------------------------------------------------- bootstrap CIs


def bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[Sequence[float]], float] = mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, object]:
    """Percentile bootstrap CI for any scalar statistic over `values`
    (one value per independent run — e.g. per-run divergence, per-run
    degradation depth).

    HONEST SMALL-N CAVEAT: this benchmark's pilot design (EXP-004) has
    N=4 runs/model. A bootstrap resample of 4 points drawn from only 4
    distinct values is a genuinely coarse approximation of the true
    sampling distribution — there are only 4**4 = 256 distinct ways to
    resample 4 items with replacement from 4 items, so the "10000
    resamples" default is mostly redundant draws of those 256 outcomes,
    not 10000 independent pieces of evidence. This function enumerates
    those 256 (or however many n**n is) EXACTLY when n**n <= _EXACT_LIMIT,
    which is strictly more defensible than Monte Carlo sampling the same
    small population with replacement. Above that threshold (e.g.
    bootstrapping over 12 models' worth of per-model scalars, 12**12 is
    intractable) it falls back to n_boot Monte Carlo draws.

    Report these intervals to show precision is limited, not to imply
    N=4 supports a tight estimate — a wide CI here is the finding, not a
    bug. Do not hide them in favor of point estimates alone.
    """
    values = list(values)
    n = len(values)
    if n < 2:
        raise ValueError("bootstrap needs >= 2 values")
    point = stat_fn(values)

    exact_count = n ** n
    if exact_count <= _EXACT_LIMIT:
        method = "exact"
        boot_stats = [stat_fn([values[i] for i in combo])
                      for combo in itertools.product(range(n), repeat=n)]
        n_used = exact_count
    else:
        method = "monte_carlo"
        rng = random.Random(seed)
        boot_stats = [stat_fn([values[rng.randrange(n)] for _ in range(n)])
                      for _ in range(n_boot)]
        n_used = n_boot

    boot_stats.sort()
    lo = _percentile(boot_stats, alpha / 2)
    hi = _percentile(boot_stats, 1 - alpha / 2)
    return {"point": point, "lo": lo, "hi": hi, "n_boot": n_used,
            "n": n, "alpha": alpha, "method": method}


# ------------------------------------------------- permutation: 2 models


def permutation_test_divergence(
    paths_a: List[Sequence[str]],
    paths_b: List[Sequence[str]],
    n_perm: int = 10000,
    seed: int = 0,
) -> Dict[str, object]:
    """Are model A's and model B's WITHIN-model divergences distinguishable?

    Statistic: (mean pairwise set_jaccard within A's runs) minus (mean
    pairwise set_jaccard within B's runs) — i.e. the same "self-jaccard"
    metrics.path_divergence reports, differenced across models. Positive
    stat = A repeats itself more across runs (lower diversity) than B.

    NULL / WHAT THIS TESTS: run-to-model assignment is exchangeable —
    under H0, A's and B's runs are draws from the same divergence-
    generating process, so which runs are labeled "A" vs "B" carries no
    information. We shuffle that label and recompute the statistic.

    WHAT THIS DOES NOT TEST: whether A and B visit the SAME topics as
    each other (that's cross-model overlap — see metrics.cross_model_overlap
    and cross_model_null below). Two models can have identical,
    indistinguishable within-model divergence while walking completely
    disjoint topic sets; this test is blind to that axis entirely.

    With na, nb runs there are C(na+nb, na) distinct label splits. The
    pilot's design (EXP-004: N=4 runs/model) makes na=nb=4, so
    C(8,4)=70 — small enough to enumerate EXACTLY rather than sample,
    which is strictly more defensible (no Monte Carlo error, and the
    result is deterministic regardless of `seed`). Above _EXACT_LIMIT
    splits, n_perm splits are sampled instead (seeded, so still
    reproducible) and a standard add-one correction is applied to the
    p-value (Davison & Hinkley 1997) so it can never read exactly zero
    from a finite sample.
    """
    if len(paths_a) < 2 or len(paths_b) < 2:
        raise ValueError(
            "need >= 2 runs per model to define within-model divergence")

    na, nb = len(paths_a), len(paths_b)
    pooled = list(paths_a) + list(paths_b)
    n = na + nb
    observed = _mean_pairwise_jaccard(paths_a) - _mean_pairwise_jaccard(paths_b)

    exact_count = math.comb(n, na)
    if exact_count <= _EXACT_LIMIT:
        method = "exact"
        null_stats = []
        for combo in itertools.combinations(range(n), na):
            combo_set = set(combo)
            group_a = [pooled[i] for i in combo]
            group_b = [pooled[i] for i in range(n) if i not in combo_set]
            null_stats.append(
                _mean_pairwise_jaccard(group_a) - _mean_pairwise_jaccard(group_b))
        extreme = sum(1 for s in null_stats if abs(s) >= abs(observed) - 1e-12)
        p_value = extreme / len(null_stats)
        n_perm_used = exact_count
    else:
        method = "monte_carlo"
        rng = random.Random(seed)
        idx = list(range(n))
        null_stats = []
        for _ in range(n_perm):
            rng.shuffle(idx)
            group_a = [pooled[i] for i in idx[:na]]
            group_b = [pooled[i] for i in idx[na:]]
            null_stats.append(
                _mean_pairwise_jaccard(group_a) - _mean_pairwise_jaccard(group_b))
        extreme = sum(1 for s in null_stats if abs(s) >= abs(observed) - 1e-12)
        p_value = (extreme + 1) / (len(null_stats) + 1)
        n_perm_used = n_perm

    return {"stat": observed, "p_value": p_value, "n_perm": n_perm_used,
            "method": method}


# ---------------------------------------------- cross-model shared-pool null


def cross_model_null(
    model_paths: Dict[str, List[Sequence[str]]],
    n_perm: int = 10000,
    seed: int = 0,
) -> Dict[str, object]:
    """Is observed cross-model topic overlap higher than chance? Runs TWO
    genuinely different null models, because that question is ambiguous
    until "chance" is pinned down — read both docstrings below, not just
    the p-values, before citing either in the paper.

    (1) label_shuffle — permute which MODEL each RUN is attributed to
        (group sizes preserved), recompute mean_cross_jaccard on the
        reshuffled grouping, repeat n_perm times. p_value = fraction of
        reshuffled statistics >= the true observed statistic (one-sided:
        the question is whether observed is HIGHER than shuffled).

        WHAT THIS TESTS: whether the true run-to-model assignment carries
        information beyond an arbitrary regrouping of the same runs —
        i.e. model-IDENTITY effects / heterogeneity across models.

        WHAT THIS DOES NOT TEST (read this before using it to argue for
        a shared pool): if every model already draws from ONE shared,
        model-independent topic pool, label-shuffling barely moves the
        statistic. mean_cross_jaccard under the true grouping and under
        any reshuffled grouping are both just "average pairwise jaccard
        over most run pairs, minus whichever subset happens to land in
        the same group" — under genuine homogeneity that exclusion has
        little systematic effect on the mean, so this test has LOW POWER
        to detect a shared pool. A non-significant label_shuffle p-value
        is NOT evidence against pool-sharing; it mostly just means no
        single model behaves distinctively differently from the rest.
        Demonstrated in test_stats.py: fully-duplicated-path data (the
        most extreme possible shared pool) still yields label_shuffle
        p ~= 1.0, because shuffling identical data changes nothing.

    (2) pooled_frequency_baseline — the actual shared-pool test. Null:
        each model independently emits a run by sampling its topics i.i.d.
        (with replacement, run-length matched to the real run) from ONE
        pooled empirical topic-FREQUENCY table built from every topic
        every model actually used. This is the "bag of topics, one shared
        jar, no per-model correlation" baseline. p_value = fraction of
        n_perm synthetic universes whose mean_cross_jaccard >= observed
        (one-sided — is observed higher than this baseline predicts).

        ASSUMPTIONS, stated plainly (this is the "simplest defensible
        version" the design brief asked for, not a validated generative
        model of the cascade process):
          (a) treats a run's topics as an unordered i.i.d. sample — drops
              all sequential/semantic structure (a real cascade's next
              topic is correlated with the rejection history; this
              baseline has no notion of that), so it is a deliberately
              crude "bag of topics" null;
          (b) the frequency table is built FROM the observed data itself,
              so this is a conditional/resampling-style null, not an
              independent prior — it answers "given this exact topic
              vocabulary and its observed overall frequency, does
              cross-model overlap exceed what random co-occurrence in
              that vocabulary predicts", not "would independently-trained
              models invent the same topics from scratch";
          (c) sampling is with replacement, so within-run repeats are
              possible in the synthetic data even though the real
              rejector cascade discourages them — again, simplest
              defensible baseline, not a generative model to defend
              beyond that.

    Returns both sub-results plus the observed statistic they're each
    compared against.
    """
    names = sorted(model_paths)
    if len(names) < 2:
        raise ValueError("need >= 2 models")

    observed = cross_model_overlap(model_paths)["mean_cross_jaccard"]

    # ---- (1) label_shuffle ----
    runs: List[Sequence[str]] = []
    sizes: List[int] = []
    for m in names:
        runs.extend(model_paths[m])
        sizes.append(len(model_paths[m]))

    def _grouped_cross_jaccard(flat_runs: List[Sequence[str]],
                                group_sizes: List[int]) -> float:
        groups: Dict[str, List[Sequence[str]]] = {}
        idx = 0
        for gi, s in enumerate(group_sizes):
            groups["g%d" % gi] = flat_runs[idx: idx + s]
            idx += s
        return cross_model_overlap(groups)["mean_cross_jaccard"]

    rng = random.Random(seed)
    label_null = []
    for _ in range(n_perm):
        shuffled = runs[:]
        rng.shuffle(shuffled)
        label_null.append(_grouped_cross_jaccard(shuffled, sizes))
    label_extreme = sum(1 for s in label_null if s >= observed - 1e-12)
    label_p = (label_extreme + 1) / (len(label_null) + 1)

    # ---- (2) pooled_frequency_baseline ----
    topic_counts: Dict[str, int] = {}
    for m in names:
        for p in model_paths[m]:
            for t in p:
                topic_counts[t] = topic_counts.get(t, 0) + 1
    topics_list = list(topic_counts.keys())
    weights = [topic_counts[t] for t in topics_list]

    rng2 = random.Random(seed + 1)  # independent stream, still seed-derived
    pooled_null = []
    for _ in range(n_perm):
        synth_groups: Dict[str, List[Sequence[str]]] = {}
        for m in names:
            synth_groups[m] = [
                rng2.choices(topics_list, weights=weights, k=len(p))
                if len(p) else []
                for p in model_paths[m]
            ]
        pooled_null.append(cross_model_overlap(synth_groups)["mean_cross_jaccard"])
    pooled_extreme = sum(1 for s in pooled_null if s >= observed - 1e-12)
    pooled_p = (pooled_extreme + 1) / (len(pooled_null) + 1)

    return {
        "observed_mean_cross_jaccard": observed,
        "label_shuffle": {
            "p_value": label_p,
            "n_perm": n_perm,
            "hypothesis": "model-identity effect (heterogeneity across "
                          "models); NOT a test of shared-pool sharing — "
                          "see docstring",
        },
        "pooled_frequency_baseline": {
            "p_value": pooled_p,
            "n_perm": n_perm,
            "hypothesis": "observed overlap higher than a bag-of-topics "
                          "shared-pool baseline predicts — see docstring "
                          "for assumptions",
        },
    }


# --------------------------------------------------------- effect sizes


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta: P(X > Y) - P(X < Y), computed over all n1*n2 pairs.
    Range [-1, 1]; 0 = stochastic equality, +1 = every x exceeds every y.

    Chosen (per the design brief and this repo's own statistical-analysis
    skill reference, effect_sizes_and_power.md: "Cohen's d has slight
    upward bias with small samples (n < 20)") over Cohen's d for pairwise
    model comparisons at N=4 runs/model: it makes no distributional-shape
    assumption metrics.py's data (jaccard/depth scalars, not remotely
    guaranteed normal at n=4) can't support checking anyway.

    Algebraically identical to the rank-biserial correlation for two
    independent samples (rank_biserial_correlation below is a thin
    documented alias). Magnitude bands (Romano et al. 2006, matched to
    Cohen's d via the same effect at each threshold): negligible < 0.147,
    small < 0.33, medium < 0.474, else large — see cliffs_delta_magnitude.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        raise ValueError("need >= 1 value per group")
    more = less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1
    return (more - less) / (n1 * n2)


def cliffs_delta_magnitude(delta: float) -> str:
    """Romano et al. (2006) magnitude bands for Cliff's delta."""
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def rank_biserial_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank-biserial correlation for two independent samples. Algebraically
    identical to cliffs_delta (both equal P(X>Y) - P(X<Y); see e.g. the
    Mann-Whitney-U identity r_rb = 1 - 2U/(n1*n2)). Exposed under both
    names because different subfields cite one or the other — use
    whichever your target venue expects, the number is the same."""
    return cliffs_delta(x, y)


def pairwise_cliffs_delta(
    model_values: Dict[str, Sequence[float]]
) -> Dict[str, float]:
    """Cliff's delta for every pair of models over some per-run scalar
    (e.g. per-run set_jaccard, or degradation depth). Keyed "modelA|modelB"
    — same convention as metrics.cross_model_overlap's per-pair keys.
    Positive value: modelA's values tend to exceed modelB's."""
    names = sorted(model_values)
    if len(names) < 2:
        raise ValueError("need >= 2 models")
    out: Dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out["%s|%s" % (names[i], names[j])] = cliffs_delta(
                model_values[names[i]], model_values[names[j]])
    return out
