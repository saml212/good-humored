#!/usr/bin/env python3
"""EXP-014 -- validates `env/incongruity_gate.py`'s `TwoStageIncongruityGate`
against the pre-registered fixture + bars (`docs/THEORY-MAP.md` §12.2,
EXPERIMENT_LOG.md's EXP-013/014 registration), using a REAL claude:haiku
predictor for the cold/primed calls and real all-MiniLM-L6-v2 embeddings
for the distance computation.

Shape mirrors env/validate_bvt_gate.py (EXP-013)'s own shape: fixture ->
real judge/predictor calls -> per-class metric -> separation/consistency/
disproof checks; LabelCache + `--max-calls` budget guard (cache/resume
-safe).

Design note -- the "replay split" trick. §12.2's registered budget is "2-3
calls" per item ("cold + primed calls" -- SPLIT is not mentioned in the
registration's compute estimate), and the fixture already carries GOLD
`setup`/`punchline` fields (that IS the fixture's whole construction: e.g.
`setup_nonsequitur` shares its `setup` with the matching `real_joke` item
and swaps only the punchline). Re-deriving the split via a REAL SPLIT_PROMPT
call would let split-accuracy noise contaminate a validation whose bars are
about the cold/primed GATING logic specifically -- a different concern the
registration doesn't ask this run to test. So this script builds a hybrid
predictor: `SPLIT_PROMPT` calls are answered by a zero-cost, zero-network
REPLAY of the fixture's own gold `setup`/`punchline` (formatted exactly as
a real predictor's `SETUP: ...\\nPUNCHLINE: ...` answer), while
`PREDICT_COLD_PROMPT`/`PREDICT_PRIMED_PROMPT` calls are forwarded to the
REAL haiku CLI. The gate is then run through its OWN real `_split` (so the
real parsing code executes against a realistically-formatted string) and
its own real `_distance` (so the real embedding backend + cosine-similarity
code executes) -- only the call that would need a genuine judgment (would
this text split into setup/punchline this way) is replaced by a replay of
already-known-correct data; the two calls this validation is actually
about are always real. `_split`/`_distance` are accessed directly (not
only `__call__`) because the registered bars need gate_1 and gate_2 pass
rates SEPARATELY, and the AND/OR/strict-gate reward `__call__` itself
returns has already collapsed that distinction away -- exactly the
"class's own plumbing is exercised by fake-embedder unit tests, a
validator's job is producing the real numbers" principle
env/validate_semantic_novelty.py's module docstring states, applied here
to a class with two internal helper methods instead of one.

Usage (real embeddings need the sentence_transformers backend -- see
scratchpad note in this project's session log if system python lacks it;
run with a python that has sentence_transformers+numpy installed):
  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 python3 -m env.validate_incongruity_gate \\
      --judge claude:haiku --repeats 3 --max-calls 900 \\
      --out experiment-runs/2026-07-17-incongruity-gate-validation

EXP-014b addendum (2026-07-22, --k-samples) -- multi-sample centroid
gating. EXP-014's diagnosis was single-guess sampling noise (repeat
consistency 0.556); EXP-014b's registered fix is K cold + K primed
predictor draws per item per repeat, reduced to a per-condition CENTROID
embedding before the identical (unchanged) surprise_threshold/
drop_threshold arithmetic applies -- see EXPERIMENT_LOG.md's EXP-014b
registration for the full design and predictions.

DESIGN DECISION, FLAGGED per this build's own instructions: multi
-sampling lives ENTIRELY IN THIS RUNNER, not as a constructor param on
`TwoStageIncongruityGate` -- env/incongruity_gate.py is UNTOUCHED by this
addendum (zero lines changed, its shipped k=1 `__call__` semantics are
bit-identical to before EXP-014b). Reasoning: this validator has never
driven the gate through `__call__` in the first place (see the "replay
split" design note above -- `_split`/`_distance` are called directly
because the registered bars need gate_1/gate_2 pass rates SEPARATELY,
a distinction `__call__`'s strict-AND collapses). A `k_samples`
constructor param on the gate would only be exercised by `__call__`,
which this validator does not use -- adding it would be untested-by-
this-experiment surface area purely for a hypothetical future training
-time reward change, not something EXP-014b's own data would validate.
Keeping it runner-only also makes the "zero change to shipped k=1
semantics" hard constraint trivially, unconditionally true (nothing in
incongruity_gate.py moved) rather than merely "tested and believed
true." `score_repeat_k`/`run_validation_k` below are NEW functions
(score_repeat/run_validation are untouched and still used verbatim for
k_samples=1 -- see their k=1 delegation, which is how this addendum
guarantees the k=1 CLI path is bit-identical to EXP-014's own numbers,
not just "should produce equivalent output").
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# Make the repo root importable regardless of how this script is invoked
# (same convention as env/smoke.py / env/certify_judge_oogiri.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.providers import get_provider  # noqa: E402
from benchmark.relabel import LabelCache  # noqa: E402
from env.incongruity_gate import (PREDICT_COLD_PROMPT, PREDICT_PRIMED_PROMPT,
                                  SPLIT_PROMPT, TwoStageIncongruityGate)  # noqa: E402

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "incongruity_gate_fixture.jsonl"
CLASSES = ("real_joke", "setup_nonsequitur", "boring_expected",
          "vague_abstract_gaming_probe")

PROMPT_VERSION = "exp014-v1"
COLD_LABELER = "incongruity-cold-%s" % PROMPT_VERSION
PRIMED_LABELER = "incongruity-primed-%s" % PROMPT_VERSION

_SEP = "\x1f"

# The literal prefix SPLIT_PROMPT.format(...) always starts with -- used
# by the hybrid predictor below to route a rendered prompt back to its
# template without needing to un-format it.
_SPLIT_PREFIX = "Below is a joke."
_COLD_MARKER = "UNSURPRISING"       # only PREDICT_COLD_PROMPT contains this
_PRIMED_MARKER = "clever twist"     # only PREDICT_PRIMED_PROMPT contains this

# Repeat-consistency subset: the one class sized exactly 12 (§12.2:
# "3 repeats on a 12-item subset").
REPEAT_CONSISTENCY_CLASS = "real_joke"


# ------------------------------------------------------------- fixture I/O


def load_fixture() -> List[Dict]:
    with open(FIXTURE) as f:
        items = [json.loads(line) for line in f if line.strip()]
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fixture ids in %s" % FIXTURE)
    for i in items:
        if i["gold_class"] not in CLASSES:
            raise ValueError("item %r has unknown gold_class %r"
                             % (i["id"], i["gold_class"]))
        if not i.get("setup") or not i.get("punchline"):
            raise ValueError("item %r missing setup/punchline" % i["id"])
    return items


# ------------------------------------------------------------- budget guard


class BudgetExceeded(Exception):
    """Raised once --max-calls underlying REAL predictor calls have been
    made (replayed SPLIT calls never count against budget)."""


def make_budgeted_complete(complete: Callable[[str], str], stats: Dict,
                           max_calls: int) -> Callable[[str], str]:
    def wrapped(prompt: str) -> str:
        if stats["calls"] >= max_calls:
            raise BudgetExceeded(
                "validate_incongruity_gate: hit --max-calls=%d underlying "
                "predictor calls" % max_calls)
        stats["calls"] += 1
        return complete(prompt)
    return wrapped


# ---------------------------------------------------------- hybrid predictor


class PredictorEmpty(Exception):
    """Raised when a cold/primed call returns a blank response even after
    retries -- the documented 'can't apply' sentinel for this validation
    (mirrors env/incongruity_gate.py's own NO_SPLIT convention, one level
    down: here it's an empty PREDICTION, not an empty split)."""


def make_hybrid_predictor(item: Dict, repeat: int, complete: Callable[[str], str],
                          cache: LabelCache, stats: Dict, retries: int,
                          sample_idx: Optional[int] = None
                          ) -> Callable[[str], str]:
    """predictor(prompt) -> str routed by which of the three §12.2 prompt
    templates the rendered `prompt` came from. SPLIT_PROMPT calls are
    answered by a zero-cost replay of `item['setup']`/`item['punchline']`
    (see module docstring); PREDICT_COLD_PROMPT/PREDICT_PRIMED_PROMPT calls
    are real, cached (keyed by stage+repeat+item id, so repeats stay
    genuinely independent) and budgeted.

    `sample_idx` (EXP-014b's --k-samples addendum): None (the default)
    reproduces the ORIGINAL cache-key format byte-for-byte -- every
    pre-existing call site (positional, 6 args, no knowledge of this
    param) resolves through the exact same key as before this param
    existed, so EXP-014's own cache remains hit-compatible and score_repeat's
    k=1 behavior stays bit-identical. Passing an int gives that guess its
    OWN cache slot (keyed by repeat+sample_idx+item id) so K independent
    cold/primed draws for the same (item, repeat) never collide with, or
    silently reuse, each other's cached answer -- the entire point of
    K-sampling is K genuinely independent predictor calls."""
    def predictor(prompt: str) -> str:
        if prompt.startswith(_SPLIT_PREFIX):
            return "SETUP: %s\nPUNCHLINE: %s" % (item["setup"], item["punchline"])
        stage = "cold" if _COLD_MARKER in prompt else "primed"
        assert _PRIMED_MARKER in prompt or stage == "cold", \
            "unrecognized predictor prompt: %r" % prompt
        labeler = COLD_LABELER if stage == "cold" else PRIMED_LABELER
        if sample_idx is None:
            key = "%d%s%s" % (repeat, _SEP, item["id"])
        else:
            key = "%d%s%d%s%s" % (repeat, _SEP, sample_idx, _SEP, item["id"])
        cached = cache.get(labeler, key)
        if cached is not None:
            stats["cache_hits"] = stats.get("cache_hits", 0) + 1
            return cached
        raw = ""
        for _ in range(retries + 1):
            raw = complete(prompt)
            if raw and raw.strip():
                raw = raw.strip()
                break
        if not raw or not raw.strip():
            stats["unparseable"] = stats.get("unparseable", 0) + 1
            raise PredictorEmpty(stage)
        cache.put(labeler, key, raw)
        return raw
    return predictor


# --------------------------------------------------------------- score_repeat


def score_repeat(item: Dict, repeat: int, complete: Callable[[str], str],
                 cache: LabelCache, stats: Dict, retries: int, embed_fn,
                 surprise_threshold: float, drop_threshold: float) -> Dict:
    """One repeat's (d_cold, d_primed, gate_1, gate_2) for one fixture
    item, computed by exercising the REAL TwoStageIncongruityGate's own
    `_split`/`_distance` methods (see module docstring for why `_split`
    is called at all when the answer is a replay: it validates the real
    parsing code, not just the gating arithmetic)."""
    predictor = make_hybrid_predictor(item, repeat, complete, cache, stats,
                                      retries)
    gate = TwoStageIncongruityGate(predictor=predictor, embed_fn=embed_fn,
                                   weight=1.0,
                                   surprise_threshold=surprise_threshold,
                                   drop_threshold=drop_threshold)
    try:
        setup, punchline = gate._split(item["text"])
        if setup is None:
            return {"d_cold": None, "d_primed": None, "gate_1": None,
                   "gate_2": None, "passes": None, "unparseable": True}
        cold = gate.predictor(PREDICT_COLD_PROMPT.format(setup=setup))
        primed = gate.predictor(PREDICT_PRIMED_PROMPT.format(setup=setup))
    except PredictorEmpty:
        return {"d_cold": None, "d_primed": None, "gate_1": None,
               "gate_2": None, "passes": None, "unparseable": True}
    d_cold = gate._distance(cold, punchline)
    d_primed = gate._distance(primed, punchline)
    gate_1 = d_cold >= gate.surprise_threshold
    gate_2 = (d_primed < d_cold) and ((d_cold - d_primed) >= gate.drop_threshold)
    return {"d_cold": d_cold, "d_primed": d_primed, "gate_1": bool(gate_1),
           "gate_2": bool(gate_2), "passes": bool(gate_1 and gate_2),
           "unparseable": False}


# ----------------------------------------------------------- score_repeat_k
# EXP-014b (--k-samples) -- multi-sample centroid gating. See this
# module's docstring for the FLAGGED design decision (runner-only, gate
# file untouched).


def _centroid_and_dispersion(vecs) -> "tuple":
    """Reduce K assumed-UNIT-NORMALIZED guess embeddings to (a) a single
    unit-renormalized centroid direction and (b) a bounded [0, 1]
    dispersion diagnostic.

    dispersion = 1 - R, where R = ||mean(vecs)|| is the "resultant
    length" of the K unit vectors -- the standard circular/spherical
    -variance statistic for directional data. R=1 (dispersion=0) means
    all K guesses landed on the EXACT same embedding direction (zero
    diversity -- averaging can only cancel embedding noise, not real
    disagreement, because there was none to cancel). R->0 (dispersion
    ->1) means the K guesses point in unrelated directions -- genuine
    sampling diversity for the centroid to average over. Renormalizing
    the raw mean back onto the unit sphere before returning it keeps the
    centroid-to-punchline dot product on the exact same cosine
    -similarity scale `_distance`'s single-guess version uses, so
    surprise_threshold/drop_threshold (UNCHANGED from EXP-014) stay
    meaningful against a centroid instead of a lone guess.
    """
    import numpy as np
    arr = np.asarray(vecs, dtype=float)
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    dispersion = 1.0 - norm
    if norm < 1e-12:
        # Degenerate case: K guesses cancel out almost exactly (maximally
        # dispersed, e.g. two antipodal guesses at K=2). Returning the
        # un-renormalized near-zero vector as-is gives a centroid-to
        # -punchline dot product of ~0 -- i.e. maximal distance, the
        # honest "no consensus direction exists" answer -- rather than
        # dividing by ~0 and amplifying floating-point noise into a
        # meaningless direction.
        return mean, dispersion
    return mean / norm, dispersion


def score_repeat_k(item: Dict, repeat: int, complete: Callable[[str], str],
                   cache: LabelCache, stats: Dict, retries: int, embed_fn,
                   surprise_threshold: float, drop_threshold: float,
                   k_samples: int) -> Dict:
    """K-sampled generalization of `score_repeat` (EXP-014b, --k-samples).

    k_samples == 1 delegates BYTE-FOR-BYTE to the untouched `score_repeat`
    -- same cache keys (sample_idx=None), same 2-item `_distance` embed
    batch, same call count -- guaranteeing the k=1 path is bit-identical
    to EXP-014's own numbers, not merely "should produce equivalent
    output" (see TestScoreRepeatKOneMatchesScoreRepeat in
    env/tests/test_validate_incongruity_gate.py).

    k_samples > 1 draws K independent cold guesses and K independent
    primed guesses (each its own real predictor call, its own cache slot
    keyed by sample_idx via `make_hybrid_predictor`), embeds all 2K
    guesses plus the gold punchline in ONE `embed_fn` batch call, reduces
    each condition's K guesses to a single centroid via
    `_centroid_and_dispersion`, and applies the IDENTICAL, unchanged
    gate-1/gate-2 threshold arithmetic `score_repeat` uses. Returns the
    same core keys as `score_repeat` (so every downstream aggregator --
    class_gate_rates, pooled_gate1_rate, repeat_consistency, build_bars --
    is reused UNCHANGED) plus four new diagnostic keys (cold_dispersion,
    primed_dispersion, n_unique_cold, n_unique_primed) that are always
    None on the delegated k_samples==1 path.
    """
    if k_samples == 1:
        result = score_repeat(item, repeat, complete, cache, stats, retries,
                              embed_fn, surprise_threshold, drop_threshold)
        return {**result, "cold_dispersion": None, "primed_dispersion": None,
               "n_unique_cold": None, "n_unique_primed": None}
    if k_samples < 1:
        raise ValueError("k_samples must be >= 1, got %r" % (k_samples,))

    import numpy as np

    # sample_idx=None here is deliberate and harmless: this predictor is
    # used ONLY for the SPLIT replay (never cached/never budgeted, see
    # make_hybrid_predictor), so which cache key format it would use for
    # cold/primed calls never matters -- it never makes one.
    split_predictor = make_hybrid_predictor(item, repeat, complete, cache,
                                            stats, retries)
    gate = TwoStageIncongruityGate(predictor=split_predictor, embed_fn=embed_fn,
                                   weight=1.0,
                                   surprise_threshold=surprise_threshold,
                                   drop_threshold=drop_threshold)
    setup, punchline = gate._split(item["text"])
    if setup is None:
        return {"d_cold": None, "d_primed": None, "gate_1": None,
               "gate_2": None, "passes": None, "unparseable": True,
               "cold_dispersion": None, "primed_dispersion": None,
               "n_unique_cold": None, "n_unique_primed": None}

    cold_guesses: List[str] = []
    primed_guesses: List[str] = []
    try:
        for k_idx in range(k_samples):
            sample_predictor = make_hybrid_predictor(
                item, repeat, complete, cache, stats, retries,
                sample_idx=k_idx)
            cold_guesses.append(
                sample_predictor(PREDICT_COLD_PROMPT.format(setup=setup)))
            primed_guesses.append(
                sample_predictor(PREDICT_PRIMED_PROMPT.format(setup=setup)))
    except PredictorEmpty:
        return {"d_cold": None, "d_primed": None, "gate_1": None,
               "gate_2": None, "passes": None, "unparseable": True,
               "cold_dispersion": None, "primed_dispersion": None,
               "n_unique_cold": None, "n_unique_primed": None}

    embeddings = np.asarray(embed_fn(cold_guesses + primed_guesses + [punchline]))
    cold_vecs = embeddings[:k_samples]
    primed_vecs = embeddings[k_samples:2 * k_samples]
    punchline_vec = embeddings[2 * k_samples]

    cold_centroid, cold_dispersion = _centroid_and_dispersion(cold_vecs)
    primed_centroid, primed_dispersion = _centroid_and_dispersion(primed_vecs)

    d_cold = float(1.0 - np.dot(cold_centroid, punchline_vec))
    d_primed = float(1.0 - np.dot(primed_centroid, punchline_vec))
    gate_1 = d_cold >= surprise_threshold
    gate_2 = (d_primed < d_cold) and ((d_cold - d_primed) >= drop_threshold)
    return {
        "d_cold": d_cold, "d_primed": d_primed,
        "gate_1": bool(gate_1), "gate_2": bool(gate_2),
        "passes": bool(gate_1 and gate_2), "unparseable": False,
        "cold_dispersion": cold_dispersion, "primed_dispersion": primed_dispersion,
        "n_unique_cold": len(set(cold_guesses)),
        "n_unique_primed": len(set(primed_guesses)),
    }


def run_validation(items: List[Dict], complete: Callable[[str], str],
                   cache: LabelCache, max_calls: int, repeats: int,
                   retries: int, embed_fn, surprise_threshold: float,
                   drop_threshold: float, raw_log=None) -> Dict:
    stats: Dict = {"calls": 0, "cache_hits": 0, "unparseable": 0}
    budgeted = make_budgeted_complete(complete, stats, max_calls)
    per_item: Dict[str, List[Dict]] = {}
    budget_exhausted = False
    for item in items:
        per_item[item["id"]] = []
        for r in range(repeats):
            try:
                result = score_repeat(item, r, budgeted, cache, stats,
                                      retries, embed_fn, surprise_threshold,
                                      drop_threshold)
            except BudgetExceeded:
                budget_exhausted = True
                break
            per_item[item["id"]].append(result)
            if raw_log is not None:
                raw_log.write(json.dumps({
                    "id": item["id"], "repeat": r,
                    "gold_class": item["gold_class"], **result}) + "\n")
                raw_log.flush()
        if budget_exhausted:
            break
    return {"per_item": per_item, "stats": stats,
           "budget_exhausted": budget_exhausted}


def run_validation_k(items: List[Dict], complete: Callable[[str], str],
                     cache: LabelCache, max_calls: int, repeats: int,
                     retries: int, embed_fn, surprise_threshold: float,
                     drop_threshold: float, k_samples: int,
                     raw_log=None) -> Dict:
    """K-sampling generalization of `run_validation` (EXP-014b). Identical
    resume/budget/loop structure -- the only difference is delegating
    per-repeat scoring to `score_repeat_k` (itself a k_samples==1
    passthrough to the untouched `score_repeat` when k_samples==1, so
    THIS function's k=1 output is bit-identical to `run_validation`'s own;
    see TestRunValidationKOneMatchesRunValidation)."""
    stats: Dict = {"calls": 0, "cache_hits": 0, "unparseable": 0}
    budgeted = make_budgeted_complete(complete, stats, max_calls)
    per_item: Dict[str, List[Dict]] = {}
    budget_exhausted = False
    for item in items:
        per_item[item["id"]] = []
        for r in range(repeats):
            try:
                result = score_repeat_k(item, r, budgeted, cache, stats,
                                        retries, embed_fn, surprise_threshold,
                                        drop_threshold, k_samples)
            except BudgetExceeded:
                budget_exhausted = True
                break
            per_item[item["id"]].append(result)
            if raw_log is not None:
                raw_log.write(json.dumps({
                    "id": item["id"], "repeat": r,
                    "gold_class": item["gold_class"], "k_samples": k_samples,
                    **result}) + "\n")
                raw_log.flush()
        if budget_exhausted:
            break
    return {"per_item": per_item, "stats": stats,
           "budget_exhausted": budget_exhausted}


# --------------------------------------------------------------- metrics


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def class_gate_rates(items: List[Dict], per_item: Dict[str, List[Dict]],
                     cls: str) -> Dict:
    """Pooled (item x repeat) observation fractions -- matches this
    project's other 'rate' metrics (e.g. EXP-010's escape_rate,
    env/validate_semantic_novelty.py's detection rate), NOT the item-then
    -class averaging env/validate_bvt_gate.py uses for its continuous
    MEANS. Rates and means are different established conventions in this
    codebase; see this script's module docstring / the final report for
    the explicit note."""
    ids = [i["id"] for i in items if i["gold_class"] == cls]
    obs = [r for iid in ids for r in per_item.get(iid, []) if not r["unparseable"]]
    n = len(obs)
    return {
        "n_items": len(ids), "n_observations": n,
        "gate_1_pass_rate": (sum(1 for o in obs if o["gate_1"]) / n) if n else None,
        "gate_2_pass_rate": (sum(1 for o in obs if o["gate_2"]) / n) if n else None,
        "both_gates_pass_rate": (sum(1 for o in obs if o["passes"]) / n) if n else None,
        "mean_d_cold": _mean([o["d_cold"] for o in obs]),
        "mean_d_primed": _mean([o["d_primed"] for o in obs]),
    }


def class_dispersion_stats(items: List[Dict], per_item: Dict[str, List[Dict]],
                           cls: str) -> Dict:
    """EXP-014b's cold/primed dispersion diagnostic (see
    `_centroid_and_dispersion`), pooled item x repeat over one fixture
    class -- the honest "did K=5 buy real guess diversity, or five
    near-identical guesses" measurement the registration calls for.
    None fields (k_samples=1 observations carry no dispersion; unparseable
    repeats) are excluded via `_mean`, matching `class_gate_rates`'
    established filtering convention. Values are None (not 0.0) when
    every observation is a k_samples=1 passthrough -- "no dispersion data
    collected" is a different fact than "zero measured dispersion" and
    must not be conflated."""
    ids = [i["id"] for i in items if i["gold_class"] == cls]
    obs = [r for iid in ids for r in per_item.get(iid, []) if not r["unparseable"]]
    return {
        "n_observations": len(obs),
        "mean_cold_dispersion": _mean([o["cold_dispersion"] for o in obs]),
        "mean_primed_dispersion": _mean([o["primed_dispersion"] for o in obs]),
        "mean_n_unique_cold": _mean([o["n_unique_cold"] for o in obs]),
        "mean_n_unique_primed": _mean([o["n_unique_primed"] for o in obs]),
    }


def pooled_gate1_rate(items: List[Dict], per_item: Dict[str, List[Dict]],
                      classes: Sequence[str]) -> Optional[float]:
    ids = [i["id"] for i in items if i["gold_class"] in classes]
    obs = [r for iid in ids for r in per_item.get(iid, []) if not r["unparseable"]]
    return (sum(1 for o in obs if o["gate_1"]) / len(obs)) if obs else None


def repeat_consistency(items: List[Dict], per_item: Dict[str, List[Dict]],
                       cls: str = REPEAT_CONSISTENCY_CLASS) -> float:
    """Pooled pairwise PASS/FAIL agreement across repeats, over the
    REPEAT_CONSISTENCY_CLASS subset only (§12.2: '3 repeats on a 12-item
    subset') -- mirrors env/validate_bvt_gate.py's repeat_consistency /
    benchmark/validate_rejector.py's repeat_consistency (pooled agreement,
    not item-then-mean). PASS/FAIL is the gate's own combined `passes`
    (gate_1 AND gate_2), the discrete label the class actually emits as a
    reward at inference time."""
    ids = [i["id"] for i in items if i["gold_class"] == cls]
    agrees, total = 0, 0
    for iid in ids:
        labels = [r["passes"] for r in per_item.get(iid, []) if not r["unparseable"]]
        for x in range(len(labels)):
            for y in range(x + 1, len(labels)):
                agrees += labels[x] == labels[y]
                total += 1
    return agrees / total if total else 1.0


# ------------------------------------------------------------------ bars


def build_bars(rates: Dict[str, Dict], sep_gate1: Optional[float],
               sep_gate2: Optional[float], consistency: float) -> List[Dict]:
    def bar(name, value, op, threshold):
        if value is None:
            passed = False
        elif op == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        return {"name": name, "value": value, "op": op, "threshold": threshold,
               "passed": bool(passed)}

    rj, sn, be, vg = (rates["real_joke"], rates["setup_nonsequitur"],
                      rates["boring_expected"],
                      rates["vague_abstract_gaming_probe"])
    return [
        bar("real_joke_gate1_pass_rate", rj["gate_1_pass_rate"], ">=", 0.85),
        bar("setup_nonsequitur_gate1_pass_rate", sn["gate_1_pass_rate"], ">=", 0.85),
        bar("boring_expected_gate1_pass_rate", be["gate_1_pass_rate"], "<=", 0.15),
        bar("separation_surprising_minus_boring_gate1", sep_gate1, ">=", 0.70),
        bar("real_joke_gate2_pass_rate", rj["gate_2_pass_rate"], ">=", 0.70),
        bar("setup_nonsequitur_gate2_pass_rate", sn["gate_2_pass_rate"], "<=", 0.20),
        bar("separation_real_joke_minus_nonsequitur_gate2", sep_gate2, ">=", 0.50),
        bar("vague_abstract_gaming_probe_gate2_pass_rate",
           vg["gate_2_pass_rate"], "<=", 0.25),
        bar("repeat_consistency_real_joke", consistency, ">=", 0.85),
    ]


# --------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="claude:haiku",
                    help="predictor provider spec, resolved via "
                         "get_provider() (benchmark/providers.py) -- "
                         "neutral-cwd CLI pattern, real calls only.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--max-calls", type=int, default=900,
                    help="hard cap on underlying REAL predictor calls "
                         "(replayed split calls never count); stop "
                         "cleanly with a partial report if hit.")
    ap.add_argument("--surprise-threshold", type=float, default=None,
                    help="override TwoStageIncongruityGate's default "
                         "(0.5) surprise_threshold. Default: use the "
                         "class's own default.")
    ap.add_argument("--drop-threshold", type=float, default=None,
                    help="override TwoStageIncongruityGate's default "
                         "(0.15) drop_threshold. Default: use the "
                         "class's own default.")
    ap.add_argument("--k-samples", type=int, default=1,
                    help="EXP-014b addendum: K cold + K primed predictor "
                         "draws per item per repeat, reduced to a "
                         "per-condition centroid embedding before the "
                         "UNCHANGED surprise/drop threshold arithmetic "
                         "applies (see run_validation_k/score_repeat_k). "
                         "Default 1 reproduces EXP-014's original single "
                         "-guess behavior bit-identically (env/incongruity"
                         "_gate.py itself is untouched by this flag -- see "
                         "module docstring's FLAGGED design decision).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.repeats < 2:
        ap.error("--repeats must be >= 2: consistency needs repetition "
                 "(mirrors validate_rejector.py's audit N4)")
    if args.k_samples < 1:
        ap.error("--k-samples must be >= 1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    command = ("python3 -m env.validate_incongruity_gate --judge %s "
              "--repeats %d --retries %d --max-calls %d --k-samples %d "
              "--out %s" %
              (args.judge, args.repeats, args.retries, args.max_calls,
               args.k_samples, args.out))
    print("command: %s" % command)

    items = load_fixture()
    complete = get_provider(args.judge)
    cache = LabelCache(out / "label_cache.jsonl")

    # Real embedding backend, model cached locally (HF_HUB_OFFLINE=1 in
    # the environment avoids a network round-trip). Loaded ONCE, shared
    # across every gate instance this run constructs (constructing a
    # TwoStageIncongruityGate with embed_fn already supplied is cheap --
    # it skips guarded backend construction entirely).
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from env.semantic_novelty import _MODEL_NAME
    t_load0 = time.time()
    model = SentenceTransformer(_MODEL_NAME)
    model_load_seconds = time.time() - t_load0

    def embed_fn(texts):
        return np.asarray(model.encode(list(texts), normalize_embeddings=True,
                                       show_progress_bar=False))

    # Pull the class's OWN defaults unless explicitly overridden -- avoids
    # a hardcoded threshold drifting from env/incongruity_gate.py's
    # registered defaults.
    _defaults_probe = TwoStageIncongruityGate(embed_fn=embed_fn)
    surprise_threshold = (args.surprise_threshold
                          if args.surprise_threshold is not None
                          else _defaults_probe.surprise_threshold)
    drop_threshold = (args.drop_threshold if args.drop_threshold is not None
                      else _defaults_probe.drop_threshold)

    run_stamp = time.strftime("%Y%m%dT%H%M%S")
    raw_path = out / ("incongruity_raw.%s.jsonl" % run_stamp)
    t0 = time.time()
    try:
        with open(raw_path, "a") as raw_log:
            result = run_validation_k(items, complete, cache, args.max_calls,
                                      args.repeats, args.retries, embed_fn,
                                      surprise_threshold, drop_threshold,
                                      args.k_samples, raw_log=raw_log)
    finally:
        cache.close()
    wall = time.time() - t0

    rates = {cls: class_gate_rates(items, result["per_item"], cls)
            for cls in CLASSES}
    sep_gate1 = None
    surprising_rate = pooled_gate1_rate(items, result["per_item"],
                                        ("real_joke", "setup_nonsequitur"))
    if surprising_rate is not None and rates["boring_expected"]["gate_1_pass_rate"] is not None:
        sep_gate1 = surprising_rate - rates["boring_expected"]["gate_1_pass_rate"]
    sep_gate2 = None
    if (rates["real_joke"]["gate_2_pass_rate"] is not None and
       rates["setup_nonsequitur"]["gate_2_pass_rate"] is not None):
        sep_gate2 = (rates["real_joke"]["gate_2_pass_rate"] -
                    rates["setup_nonsequitur"]["gate_2_pass_rate"])
    gate1_disproof_diff = None
    if (rates["real_joke"]["gate_1_pass_rate"] is not None and
       rates["setup_nonsequitur"]["gate_1_pass_rate"] is not None):
        gate1_disproof_diff = (rates["real_joke"]["gate_1_pass_rate"] -
                              rates["setup_nonsequitur"]["gate_1_pass_rate"])
    consistency = repeat_consistency(items, result["per_item"])
    bars = build_bars(rates, sep_gate1, sep_gate2, consistency)
    dispersion = {cls: class_dispersion_stats(items, result["per_item"], cls)
                 for cls in CLASSES}

    # EXP-014b addendum: the two registered predictions are specific to
    # the multi-sample (k_samples > 1) design; a k_samples=1 invocation of
    # this SAME script is just a fresh run of EXP-014 itself and keeps
    # EXP-014's own registered numbers/experiment name, unconditionally --
    # this is what makes the k=1 CLI path a genuine rerun of EXP-014, not
    # a relabeled variant of it.
    is_multisample = args.k_samples > 1
    experiment_name = ("EXP-014b-multisample-incongruity-gate"
                      if is_multisample else "EXP-014-incongruity-gate-validation")
    predicted_pass_rate = 0.60 if is_multisample else 0.65
    predicted_consistency = 0.82 if is_multisample else None

    # Read-only comparison against EXP-014's own (untouched) frozen
    # report.json, if present -- avoids hardcoding the baseline's numbers
    # a second time and risking them drifting out of sync with the file
    # they're supposed to describe.
    comparison_to_exp014 = None
    baseline_path = Path("experiment-runs/2026-07-17-incongruity-gate-validation/report.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        comparison_to_exp014 = {
            "baseline_report_path": str(baseline_path),
            "baseline_real_joke_pass_rate": baseline.get("real_joke_pass_rate"),
            "baseline_repeat_consistency": baseline.get("repeat_consistency"),
            "baseline_bars_passed": sum(1 for b in baseline.get("bars", []) if b["passed"]),
            "baseline_bars_total": len(baseline.get("bars", [])),
            "this_run_bars_passed": sum(1 for b in bars if b["passed"]),
            "this_run_bars_total": len(bars),
        }

    report = {
        "experiment": experiment_name,
        "command": command,
        "run_stamp": run_stamp,
        "judge": args.judge,
        "repeats": args.repeats,
        "retries": args.retries,
        "k_samples": args.k_samples,
        "n_items": len(items),
        "wall_seconds": round(wall, 1),
        "model_load_seconds": round(model_load_seconds, 2),
        "embedding_model": _MODEL_NAME,
        "surprise_threshold": surprise_threshold,
        "drop_threshold": drop_threshold,
        "split_prompt": SPLIT_PROMPT,
        "predict_cold_prompt": PREDICT_COLD_PROMPT,
        "predict_primed_prompt": PREDICT_PRIMED_PROMPT,
        "split_note": ("SPLIT_PROMPT calls are answered by a zero-cost "
                      "replay of the fixture's gold setup/punchline, not "
                      "a real predictor call -- see module docstring. "
                      "Only cold/primed calls are real and count against "
                      "--max-calls."),
        "per_class_rates": rates,
        "dispersion_diagnostic": dispersion,
        "dispersion_note": ("EXP-014b's cold-guess-dispersion diagnostic: "
                            "mean_cold_dispersion = 1 - ||mean(K unit cold-"
                            "guess embeddings)||, bounded [0, 1]. ~0 means "
                            "the K guesses landed on the same embedding "
                            "direction (no real diversity -- a centroid "
                            "can only cancel embedding noise, there was no "
                            "disagreement to average over); higher means "
                            "genuine sampling diversity. mean_n_unique_* is "
                            "the literal-text-identity cross-check (k=1 "
                            "observations report None -- no dispersion "
                            "was collected, not zero dispersion)."),
        "separation_surprising_minus_boring_gate1": sep_gate1,
        "separation_real_joke_minus_nonsequitur_gate2": sep_gate2,
        "real_joke_pass_rate": rates["real_joke"]["both_gates_pass_rate"],
        "predicted_real_joke_pass_rate_registered": predicted_pass_rate,
        "vague_probe_gate2_pass_rate": rates["vague_abstract_gaming_probe"]["gate_2_pass_rate"],
        "disproof_check_gate1_alone_diff_real_joke_minus_nonsequitur": gate1_disproof_diff,
        "disproof_check_predicted_registered": 0.0,
        "repeat_consistency": consistency,
        "repeat_consistency_class": REPEAT_CONSISTENCY_CLASS,
        "predicted_repeat_consistency_registered": predicted_consistency,
        "comparison_to_exp014": comparison_to_exp014,
        "bars": bars,
        "all_bars_pass": all(b["passed"] for b in bars),
        "call_stats": {**result["stats"], "max_calls": args.max_calls,
                      "budget_exhausted": result["budget_exhausted"]},
        "cache_path": str(out / "label_cache.jsonl"),
        "raw_log_path": str(raw_path),
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=== %s (%s, %d repeats, k_samples=%d, %d items) ===" %
         (experiment_name, args.judge, args.repeats, args.k_samples, len(items)))
    for cls in CLASSES:
        r = rates[cls]
        d = dispersion[cls]
        print("%-32s n_obs=%d  gate1=%s  gate2=%s  both=%s  cold_disp=%s" % (
            cls, r["n_observations"],
            "None" if r["gate_1_pass_rate"] is None else "%.3f" % r["gate_1_pass_rate"],
            "None" if r["gate_2_pass_rate"] is None else "%.3f" % r["gate_2_pass_rate"],
            "None" if r["both_gates_pass_rate"] is None else "%.3f" % r["both_gates_pass_rate"],
            "None" if d["mean_cold_dispersion"] is None else "%.3f" % d["mean_cold_dispersion"]))
    print("real_joke_pass_rate = %s (predicted %.2f)" %
         ("None" if report["real_joke_pass_rate"] is None
          else "%.3f" % report["real_joke_pass_rate"], predicted_pass_rate))
    if is_multisample:
        print("real_joke cold-guess dispersion diagnostic: mean=%s  "
             "mean_unique_of_%d=%s (0=identical guesses -- averaging bought "
             "nothing; higher=real diversity)" %
             ("None" if dispersion["real_joke"]["mean_cold_dispersion"] is None
              else "%.3f" % dispersion["real_joke"]["mean_cold_dispersion"],
              args.k_samples,
              "None" if dispersion["real_joke"]["mean_n_unique_cold"] is None
              else "%.2f" % dispersion["real_joke"]["mean_n_unique_cold"]))
    if comparison_to_exp014 is not None:
        print("vs EXP-014 baseline: pass_rate %s -> %s | consistency %s -> %s | bars %d/%d -> %d/%d" % (
            "None" if comparison_to_exp014["baseline_real_joke_pass_rate"] is None
            else "%.3f" % comparison_to_exp014["baseline_real_joke_pass_rate"],
            "None" if report["real_joke_pass_rate"] is None
            else "%.3f" % report["real_joke_pass_rate"],
            "None" if comparison_to_exp014["baseline_repeat_consistency"] is None
            else "%.3f" % comparison_to_exp014["baseline_repeat_consistency"],
            "%.3f" % consistency,
            comparison_to_exp014["baseline_bars_passed"], comparison_to_exp014["baseline_bars_total"],
            comparison_to_exp014["this_run_bars_passed"], comparison_to_exp014["this_run_bars_total"]))
    print("separation(surprising - boring) gate1 = %s (bar >= 0.70)" %
         ("None" if sep_gate1 is None else "%.3f" % sep_gate1))
    print("separation(real_joke - nonsequitur) gate2 = %s (bar >= 0.50)" %
         ("None" if sep_gate2 is None else "%.3f" % sep_gate2))
    print("disproof: gate1-alone diff(real_joke - nonsequitur) = %s (predicted ~0)" %
         ("None" if gate1_disproof_diff is None else "%.3f" % gate1_disproof_diff))
    print("repeat_consistency(%s) = %.3f (bar >= 0.85%s)" %
         (REPEAT_CONSISTENCY_CLASS, consistency,
          "" if predicted_consistency is None
          else ", predicted %.2f" % predicted_consistency))
    print()
    print("=== bars ===")
    for b in bars:
        status = "PASS" if b["passed"] else "FAIL"
        print("[%s] %-48s value=%s %s %s" %
             (status, b["name"],
              "None" if b["value"] is None else "%.3f" % b["value"],
              b["op"], b["threshold"]))
    print()
    print("calls=%d cache_hits=%d unparseable=%d wall=%.1fs budget_exhausted=%s" %
         (report["call_stats"]["calls"], report["call_stats"]["cache_hits"],
          report["call_stats"]["unparseable"], wall,
          result["budget_exhausted"]))
    print("report: %s" % (out / "report.json"))


if __name__ == "__main__":
    main()
