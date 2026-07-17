"""Trajectory metrics for the rejection-cascade benchmark.

The unit of measurement is a *path*: the ordered list of normalized topic
labels a model walked under accumulating rejections. Jokes are not scored
here — the trajectory is the measurement (docs/BENCHMARK.md).

Metric lineage, on purpose:
- cluster/switch statistics port Troyer, Moscovitch & Winocur (1997),
  the canonical scoring of human verbal-fluency trajectories;
- the patch-walk framing follows Hills, Jones & Todd (2012), optimal
  foraging in semantic memory;
- ARI is used only by rejector validation (instrument calibration).

Pure functions, stdlib only, Python 3.9 compatible.
"""

import math
import re
import string
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

# ---------------------------------------------------------------- labels

_ARTICLES = {"a", "an", "the"}


def normalize_label(label: str) -> str:
    """Canonicalize a free-text topic label: lowercase, strip punctuation
    and articles, naive singularization. 'The Cats!' -> 'cat'."""
    label = label.strip().lower()
    label = label.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in label.split() if w not in _ARTICLES]
    out = []
    for w in words:
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
            w = w[:-2]  # crunches -> crunch, glasses -> glass
        elif (len(w) > 3 and w.endswith("s")
              and not w.endswith(("ss", "us", "is", "as"))):
            # exempt -us/-is/-as endings: octopus, tennis, christmas are
            # not plurals (audit W2 — naive strip broke real topic words)
            w = w[:-1]
        out.append(w)
    return " ".join(out)


# ------------------------------------------------------ pairwise helpers


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Jaccard similarity of the *sets* of topics in two paths."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def prefix_agreement_depth(a: Sequence[str], b: Sequence[str]) -> int:
    """Number of leading positions where two paths agree exactly.
    A model reading a memorized list produces deep prefix agreement."""
    d = 0
    for x, y in zip(a, b):
        if x != y:
            break
        d += 1
    return d


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance over topic labels (order-sensitive)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pairwise(fn, paths: List[Sequence[str]]) -> List[float]:
    vals = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            vals.append(fn(paths[i], paths[j]))
    return vals


# ------------------------------------------------- within-model metrics


def path_divergence(paths: List[Sequence[str]]) -> Dict[str, float]:
    """How differently does ONE model walk topic space across runs?

    Low divergence = the "distribution" is a lookup table. Returns means
    over all run pairs:
      set_jaccard        — overlap of topics visited (1.0 = same topics)
      prefix_depth       — leading turns identical (list-reading signature)
      norm_edit_distance — order-sensitive dissimilarity in [0, 1]
    """
    if len(paths) < 2:
        raise ValueError("need >= 2 runs to measure divergence")
    max_len = max(len(p) for p in paths) or 1
    return {
        "set_jaccard": _mean(_pairwise(jaccard, paths)),
        "prefix_depth": _mean(_pairwise(
            lambda a, b: float(prefix_agreement_depth(a, b)), paths)),
        "norm_edit_distance": _mean(_pairwise(
            lambda a, b: edit_distance(a, b) / max(len(a), len(b), 1), paths)),
        "n_pairs": float(len(paths) * (len(paths) - 1) // 2),
    }


# -------------------------------------------------- cross-model metrics


def cross_model_overlap(
    model_paths: Dict[str, List[Sequence[str]]]
) -> Dict[str, float]:
    """Do DIFFERENT models walk the same path? (ecosystem-level collapse)

    Compares every run of model A against every run of model B, for all
    model pairs. High cross-model jaccard with high within-model jaccard
    means the industry shares one humor prior.
    """
    names = sorted(model_paths)
    if len(names) < 2:
        raise ValueError("need >= 2 models")
    per_pair: Dict[str, float] = {}
    all_j: List[float] = []
    all_prefix: List[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            js = [jaccard(pa, pb)
                  for pa in model_paths[names[i]]
                  for pb in model_paths[names[j]]]
            per_pair["%s|%s" % (names[i], names[j])] = _mean(js)
            all_j.extend(js)
            all_prefix.extend(
                float(prefix_agreement_depth(pa, pb))
                for pa in model_paths[names[i]]
                for pb in model_paths[names[j]])
    out = {"mean_cross_jaccard": _mean(all_j),
           "mean_cross_prefix_depth": _mean(all_prefix)}
    out.update(per_pair)
    return out


# ----------------------------------------------------- degradation depth


def depth_to_degradation(
    path: Sequence[str],
    refusal_turns: Optional[Sequence[int]] = None,
) -> Dict[str, Optional[int]]:
    """First turn (0-indexed) at which the model degrades.

    repeat_depth  — first reuse of an already-rejected topic
    refusal_depth — first refusal turn, if the runner recorded any
    depth         — min of the two (None = survived the whole cascade)
    """
    seen = set()
    repeat_depth = None
    for i, t in enumerate(path):
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


# ------------------------------------- Troyer-style cluster/switch stats


def cluster_switch_stats(
    path: Sequence[str],
    category_of: Callable[[str], str],
) -> Dict[str, float]:
    """Troyer et al. (1997) clustering/switching, ported to topic paths.

    In human fluency, subjects emit runs within a semantic patch, then
    switch. Here `category_of` maps a topic label to a coarse category
    (e.g. 'cat' -> 'animals'); runs of consecutive same-category topics
    are clusters. Long clusters + few switches = local trudging; a healthy
    forager balances both (Hills, Jones & Todd 2012).
    """
    if not path:
        return {"mean_cluster_size": 0.0, "n_switches": 0.0,
                "n_categories": 0.0}
    cats = [category_of(t) for t in path]
    runs: List[int] = [1]
    for prev, cur in zip(cats, cats[1:]):
        if cur == prev:
            runs[-1] += 1
        else:
            runs.append(1)
    return {
        "mean_cluster_size": _mean([float(r) for r in runs]),
        "n_switches": float(len(runs) - 1),
        "n_categories": float(len(set(cats))),
    }


# ------------------------------------------------- instrument calibration


def adjusted_rand_index(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """ARI between two labelings of the same items. Used to score the
    rejector's induced partition against the fixture's gold partition."""
    if len(labels_a) != len(labels_b):
        raise ValueError("labelings must cover the same items")
    n = len(labels_a)
    if n == 0:
        return 1.0
    cont: Counter = Counter(zip(labels_a, labels_b))
    a_marg: Counter = Counter(labels_a)
    b_marg: Counter = Counter(labels_b)
    sum_ij = sum(math.comb(v, 2) for v in cont.values())
    sum_a = sum(math.comb(v, 2) for v in a_marg.values())
    sum_b = sum(math.comb(v, 2) for v in b_marg.values())
    total = math.comb(n, 2)
    if total == 0:
        return 1.0
    expected = sum_a * sum_b / total
    max_index = (sum_a + sum_b) / 2
    if max_index == expected:
        return 1.0
    return (sum_ij - expected) / (max_index - expected)


REFUSAL_PATTERN = re.compile(
    r"\b(i (can.?t|won.?t|am unable)|as an ai|i.?m sorry,? but"
    r"|cannot comply|no more jokes|out of (jokes|ideas))\b",
    re.IGNORECASE,
)


def looks_like_refusal(text: str) -> bool:
    """Cheap refusal detector for degradation depth. Deliberately
    conservative; the runner logs full text so misses are recoverable."""
    return bool(REFUSAL_PATTERN.search(text))
