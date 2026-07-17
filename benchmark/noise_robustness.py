"""EXP-006 — offline Monte-Carlo noise-robustness simulator.

Zero API calls. This is a pure simulation, but it is grounded, not invented:
the noise MODEL (the three per-call error rates below) is measured from the
real rejector's repeat-labeling logs, and the SCORING uses the actual
`benchmark.metrics` functions (jaccard / path_divergence / cross_model_overlap
/ normalize_label) rather than reimplementations, so the numbers this
produces are about the real pipeline, not a toy stand-in for it.

Why this exists (docs/BENCHMARK.md, EXPERIMENT_LOG.md "Instrument decision"):
the paper currently ASSERTS that rejector label noise is conservative for
collapse claims — "noise splits topics, making models look MORE diverse, so
any collapse we find survives the noise." An adversarial reviewer correctly
flagged this as asserted-not-demonstrated. Worse, there's a known mechanism
that could push the OTHER way: LABEL_PROMPT v2's generalize-up instruction
sometimes maps distinct specifics onto one hypernym (documented: flamingo ->
animal in EXP-002). If model A jokes about cats and model B about dogs and
BOTH get labeled "animal", cross-model overlap is INFLATED — a manufactured
collapse finding. This module quantifies both effects and their net, per
true-overlap regime, with a component-level decomposition.

A calibration prediction is already registered (noise-robustness-v1, metric
net_bias_on_cross_model_jaccard, predicted -0.06). This code does not look at
that number and nothing here is tuned toward or away from it.

Usage:
  python3 -m benchmark.noise_robustness --reps 2000 --seed 20260717 \\
      --out experiment-runs/2026-07-17-noise-robustness/results.json

Pure stdlib, Python 3.9 compatible.
"""

import argparse
import json
import random
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .metrics import cross_model_overlap, path_divergence

# ======================================================================
# 1. EMPIRICAL NOISE RATES
# ======================================================================
#
# Source: the three repeat-labeling runs against benchmark/fixtures/
# rejector_validation.jsonl (32 items, gold_topic known for every
# non-"ambiguous" item):
#   haiku_v1   - experiment-runs/2026-07-16-rejector-validation/
#                labels_raw.20260716T231429.jsonl   (EXP-001, LABEL_PROMPT v1)
#   haiku_v2   - experiment-runs/2026-07-16-rejector-validation-v2/
#                labels_raw.20260716T232758.jsonl   (EXP-002, LABEL_PROMPT v2)
#   sonnet_v2  - experiment-runs/2026-07-17-rejector-validation-v3-sonnet/
#                labels_raw.20260716T234904.jsonl   (EXP-003b, sonnet, v2 prompt)
#
# Every one of the 3 x 32 x 3-repeats = 288 raw calls was read and classified
# by hand against its item's gold_topic (after benchmark.metrics.normalize_label,
# which the raw logs already apply at write time) into one of:
#   match       - identical to gold
#   synonym     - different word, same concept/specificity (documented
#                 precedent: EXP-001's own language for fitness/exercise/gym
#                 is "all correct, different words" — that is exactly this
#                 bucket, and it is reused verbatim for analogous cases:
#                 office/work, programming/programmer, cold/weather,
#                 caffeine/coffee, relationship/marriage, teamwork/work)
#   generalize  - label is a hypernym of gold that plausibly extends to
#                 sibling topics too (documented: EXP-002's marriage ->
#                 flamingo -> "animal"; "travel" for every airplane item is
#                 the prompt's OWN few-shot hypernym example firing at
#                 near-100% — both counted here)
#   other       - anything else: the joke's surface object captured instead
#                 of its topic (flamingo, broken arm, luggage, pilot, arm,
#                 boss), over-specification / hyponym (decaf, and one-off
#                 wrong-domain aberrations seen only in the sonnet log), or a
#                 genuine parse failure (the v1 work-c meta-commentary
#                 sentence — this predates the shape-guard fix documented in
#                 rejector.py's "audit WARN-3" note, which is why the raw
#                 sentence is in the log rather than the UNPARSEABLE
#                 sentinel the current code would emit)
#
# Two exclusions, both documented in EXPERIMENT_LOG.md and applied
# identically across all three logs:
#   - cats-c: a documented dual-topic fixture defect (EXP-002: "cats-c (cat +
#     cooking joke) is another dual-topic fixture item"). "cooking" is
#     arguably the CORRECT label for the joke's actual pivot — counting it as
#     rejector noise would misattribute a fixture-authoring flaw to the
#     instrument. Excluded from every log's tally.
#   - ambig-1, ambig-2: the fixture's own "ambiguous" pair_type, deliberately
#     multi-valid-topic items. validate_rejector.py itself excludes these
#     from ARI / pair-match scoring for the identical reason — a "miss"
#     against ONE of several defensible gold topics isn't unambiguously an
#     instrument error.
#
# One asymmetric exclusion: weather-a/b are dropped from the haiku_v1 tally
# ONLY. EXP-002's setup note records that weather-a/b were REPLACED before
# v2 ran because the original v1 text was itself dual-topic ("cold setup,
# politician butt"). The v1 log's "politician"/"politic" labels are for that
# now-gone joke text, not comparable to the CURRENT fixture's gold ("weather")
# — they are a fixture defect, not a rejector error, and the item no longer
# exists in the form the model saw. weather-a/b ARE included in the haiku_v2
# and sonnet_v2 tallies, which ran against the current (repaired) text.
#
# Counts below are exact tallies over the surviving items (27 for haiku_v1,
# 29 apiece for haiku_v2 / sonnet_v2, x3 repeats each).
_LOG_TALLIES = {
    "haiku_v1": dict(match=41, synonym=18, generalize=4, other=18, n=81),
    "haiku_v2": dict(match=49, synonym=13, generalize=24, other=1, n=87),
    "sonnet_v2": dict(match=50, synonym=10, generalize=15, other=12, n=87),
}


def _rates_from_tally(tally: Dict[str, int]) -> Dict[str, float]:
    n = tally["n"]
    return {
        "match": tally["match"] / n,
        "synonym": tally["synonym"] / n,
        "generalize": tally["generalize"] / n,
        "other": tally["other"] / n,
    }


def _pooled_tally() -> Dict[str, int]:
    pooled = dict(match=0, synonym=0, generalize=0, other=0, n=0)
    for t in _LOG_TALLIES.values():
        for k in pooled:
            pooled[k] += t[k]
    return pooled


EMPIRICAL_NOISE_RATES = {
    name: _rates_from_tally(t) for name, t in _LOG_TALLIES.items()
}
EMPIRICAL_NOISE_RATES["pooled_all"] = _rates_from_tally(_pooled_tally())

# The DEPLOYED instrument is haiku + LABEL_PROMPT v2 (EXPERIMENT_LOG.md's
# "Instrument decision" section: "Haiku + LABEL_PROMPT v2, raw string
# scoring is the instrument"). That is the model+prompt combination the real
# cascade pilot actually runs, so it is the PRIMARY source for this
# simulation's noise parameters. haiku_v1 used an abandoned prompt version;
# sonnet_v2 is not the deployed rejector (EXP-003b's own verdict: "bigger !=
# better instrument... Haiku stays"). Both are reported for context/
# sensitivity via --rate-source but neither is the default.
DEFAULT_RATE_SOURCE = "haiku_v2"


# ======================================================================
# 2. TOPIC ONTOLOGY
# ======================================================================
#
# ~30 common joke topics, drawn from the rejector_validation fixture's own
# groups (cat, dog, marriage, work, doctor, programmer, airplane, coffee,
# exercise, weather) plus obvious cascade topics in the same families. Each
# topic carries 2-4 synonyms (same concept, different word — the synonym-
# swap noise draws from this list) and ONE hypernym. Hypernyms are
# deliberately SHARED across 3 sibling topics each (10 hypernym families x 3
# topics = 30) — that sharing is exactly the mechanism the reviewer flagged:
# if generalize-up noise fires on two DIFFERENT true topics that share a
# hypernym, their labels collide and cross-model (or within-model)
# overlap is manufactured, not measured.
TOPIC_ONTOLOGY = {
    # --- pet family ---
    "cat":        {"synonyms": ["feline", "kitty", "kitten"], "hypernym": "pet"},
    "dog":        {"synonyms": ["canine", "puppy", "pooch"], "hypernym": "pet"},
    "parrot":     {"synonyms": ["bird", "parakeet"], "hypernym": "pet"},
    # --- drink family ---
    "coffee":     {"synonyms": ["espresso", "caffeine", "latte"], "hypernym": "drink"},
    "tea":        {"synonyms": ["chai", "herbaltea"], "hypernym": "drink"},
    "beer":       {"synonyms": ["ale", "brew"], "hypernym": "drink"},
    # --- relationship family ---
    "marriage":   {"synonyms": ["spouse", "wedding", "relationship"], "hypernym": "relationship"},
    "dating":     {"synonyms": ["romance", "courtship"], "hypernym": "relationship"},
    "divorce":    {"synonyms": ["breakup", "separation"], "hypernym": "relationship"},
    # --- labor family ---
    "work":       {"synonyms": ["job", "office", "career"], "hypernym": "labor"},
    "boss":       {"synonyms": ["manager", "supervisor"], "hypernym": "labor"},
    "meeting":    {"synonyms": ["standup", "conferencecall"], "hypernym": "labor"},
    # --- tech family ---
    "programmer": {"synonyms": ["coder", "developer", "programming"], "hypernym": "tech"},
    "software":   {"synonyms": ["app", "code"], "hypernym": "tech"},
    "computer":   {"synonyms": ["laptop", "pc"], "hypernym": "tech"},
    # --- health family ---
    "doctor":     {"synonyms": ["physician", "medic"], "hypernym": "health"},
    "nurse":      {"synonyms": ["caregiver"], "hypernym": "health"},
    "hospital":   {"synonyms": ["clinic", "er"], "hypernym": "health"},
    # --- wellness family ---
    "exercise":   {"synonyms": ["fitness", "gym", "workout"], "hypernym": "wellness"},
    "yoga":       {"synonyms": ["stretching", "meditation"], "hypernym": "wellness"},
    "diet":       {"synonyms": ["nutrition", "dieting"], "hypernym": "wellness"},
    # --- travel family ---
    "airplane":   {"synonyms": ["flight", "plane", "travel"], "hypernym": "travel"},
    "train":      {"synonyms": ["railway", "subway"], "hypernym": "travel"},
    "hotel":      {"synonyms": ["motel", "resort"], "hypernym": "travel"},
    # --- nature family ---
    "weather":    {"synonyms": ["forecast", "climate", "cold"], "hypernym": "nature"},
    "rain":       {"synonyms": ["storm", "drizzle"], "hypernym": "nature"},
    "snow":       {"synonyms": ["blizzard", "frost"], "hypernym": "nature"},
    # --- education family ---
    "school":     {"synonyms": ["college", "homework"], "hypernym": "education"},
    "teacher":    {"synonyms": ["professor", "tutor"], "hypernym": "education"},
    "exam":       {"synonyms": ["quiz", "test"], "hypernym": "education"},
}
assert len(TOPIC_ONTOLOGY) == 30
TOPIC_UNIVERSE = sorted(TOPIC_ONTOLOGY)


# ======================================================================
# 3. TRUE-OVERLAP REGIMES
# ======================================================================
#
# Each regime is a combinatorial construction, not a hand-picked jaccard
# number: `s` topics are SHARED across all n_models pools, `q` further
# topics are PRIVATE to each model (disjoint slices of the remaining
# universe). A model's assigned pool is shared ∪ private (size s+q). For
# any two models under this construction, the pool-level jaccard is exactly
# s / (s + 2q) — reported below as "pool_jaccard" (the analytic DESIGN
# target). The realized clean-path jaccard measured by metrics.cross_model_
# overlap will sit close to, but not exactly at, this number, because
# every RUN of a model further subsamples its pool at `keep_prob` (see
# sample_visited_set) to create run-to-run within-model variation — except
# in the "full" regime, where keep_prob=1.0 on purpose: "full collapse"
# means every run of every model visits the literal same fixed set, so
# clean cross- AND within-model jaccard are both exactly 1.0, not merely
# close to it.
REGIMES = {
    "full":     {"s": 10, "q": 0, "keep_prob": 1.0},   # pool_jaccard = 1.000
    "high":     {"s": 9,  "q": 3, "keep_prob": 0.75},  # pool_jaccard = 0.600
    "moderate": {"s": 4,  "q": 4, "keep_prob": 0.75},  # pool_jaccard = 0.333 (~0.35 target)
    "low":      {"s": 1,  "q": 5, "keep_prob": 0.75},  # pool_jaccard = 0.091 (~0.1 target)
    "disjoint": {"s": 0,  "q": 7, "keep_prob": 0.75},  # pool_jaccard = 0.000
}
REGIME_ORDER = ["full", "high", "moderate", "low", "disjoint"]


def _pool_jaccard(s: int, q: int) -> float:
    denom = s + 2 * q
    return (s / denom) if denom else 1.0


# ======================================================================
# 4. GROUND-TRUTH PATH GENERATION
# ======================================================================


def assign_pools(rng, s: int, q: int, n_models: int = 4) -> List[List[str]]:
    """Shuffle the 30-topic universe; first `s` topics are shared by every
    model, the next n_models*q topics are sliced into disjoint per-model
    private pools. Requires s + n_models*q <= len(TOPIC_UNIVERSE) (checked
    by the REGIMES table: worst case is disjoint's 4*7=28 <= 30)."""
    order = list(TOPIC_UNIVERSE)
    rng.shuffle(order)
    if s + n_models * q > len(order):
        raise ValueError("regime pool sizes exceed the topic universe")
    shared = order[:s]
    rest = order[s:]
    pools = []
    idx = 0
    for _ in range(n_models):
        private = rest[idx: idx + q]
        idx += q
        pools.append(shared + private)
    return pools


def sample_visited_set(rng, pool: Sequence[str], keep_prob: float) -> List[str]:
    """One run's true topic set: independently keep each pool topic with
    probability keep_prob. This is what makes within-model runs of the SAME
    model diverge even though they share a pool (real models don't repeat
    an identical topic list every run) — never returns empty."""
    kept = [t for t in pool if rng.random() < keep_prob]
    if not kept:
        kept = [rng.choice(list(pool))]
    return kept


def make_path(rng, visited: Sequence[str], depth: int) -> List[str]:
    """Expand a visited SET into a depth-length ordered path by concatenating
    shuffled copies of the set until `depth` is reached. set(path) ==
    set(visited) exactly, which is what every jaccard-based metric below
    actually measures; the resulting order-sensitive artifacts (prefix_depth,
    edit_distance) are reported for completeness but are not this
    experiment's primary readout (see REPORT BACK in the task spec)."""
    path: List[str] = []
    visited = list(visited)
    while len(path) < depth:
        order = list(visited)
        rng.shuffle(order)
        path.extend(order)
    return path[:depth]


# ======================================================================
# 5. NOISE MODEL
# ======================================================================


def noise_free_rates() -> Dict[str, float]:
    return {"match": 1.0, "synonym": 0.0, "generalize": 0.0, "other": 0.0}


def component_only_rates(rates: Dict[str, float], component: str) -> Dict[str, float]:
    """Isolate ONE noise component at its empirical rate; everything that
    would have been a different noise type becomes a clean match instead.
    Used for the synonym-only / generalize-only / other-only decomposition
    — each answers "how much bias would THIS mechanism alone produce, at
    its own measured rate, with nothing else confounding it."""
    out = {"match": 1.0 - rates[component], "synonym": 0.0,
           "generalize": 0.0, "other": 0.0}
    out[component] = rates[component]
    return out


def apply_noise(rng, path: Sequence[str], rates: Dict[str, float],
                other_counter: List[int]) -> List[str]:
    """Transform one true path into a noisy observed path, one call per
    turn, independently. Three failure directions, each modeled to match
    what the empirical logs actually showed for it:

    - synonym: replaces the true topic with a random member of its OWN
      synonym list (a different STRING, same concept) — this is the
      "noise splits topics" mechanism: two occurrences of the same true
      topic can now land in different Jaccard buckets, which can only
      make apparent overlap go DOWN (conservative for collapse claims).

    - generalize: replaces the true topic with its HYPERNYM string, which
      by construction (Section 2) is shared with 2 sibling topics. This is
      the reviewer's flagged mechanism: it can make apparent overlap go UP
      when it fires on genuinely different true topics that share a
      hypernym (inflationary — a manufactured collapse).

    - other: replaces the true topic with a fresh globally-unique token.
      Modeled as maximally dispersive on purpose: the haiku_v2 calibration
      data (the deployed instrument) shows its "other" bucket is 1/87 and
      that one instance ("injury" for a doctor joke) is a semantic
      near-miss, not a parse failure — zero instances of the current
      shape-guard's shared "unparseable" sentinel appear in the deployed
      instrument's own validation run. A caveat, not swept under the rug:
      IF genuine parse failures occurred at nonzero rate, rejector.py's
      actual UNPARSEABLE sentinel is a single FIXED string, which would be
      inflationary (two failed calls collide), not dispersive, the
      opposite of what's modeled here. That risk is real in principle but
      not supported by the calibration data at hand, so it is flagged here
      rather than fabricated a rate for.
    """
    r_match = rates["match"]
    r_syn = rates["synonym"]
    r_gen = rates["generalize"]
    out = []
    for true_topic in path:
        u = rng.random()
        if u < r_match:
            out.append(true_topic)
        elif u < r_match + r_syn:
            syns = TOPIC_ONTOLOGY[true_topic]["synonyms"]
            out.append(rng.choice(syns) if syns else true_topic)
        elif u < r_match + r_syn + r_gen:
            out.append(TOPIC_ONTOLOGY[true_topic]["hypernym"])
        else:
            other_counter[0] += 1
            out.append("other::%d" % other_counter[0])
    return out


# ======================================================================
# 6. ONE REPLICATE
# ======================================================================


def run_replicate(rng, regime: Dict, rates: Dict[str, float],
                   depth: int = 30, n_models: int = 4,
                   n_runs: int = 4) -> Dict[str, float]:
    """Generate one seeded synthetic world (n_models models x n_runs runs x
    depth turns), score it CLEAN and NOISY with the real benchmark.metrics
    functions, and return both plus their paired difference. Pairing clean
    against noisy derived from the SAME true paths (not independently
    resampled) is what isolates the noise contribution from ordinary
    Monte-Carlo sampling variance in the ground truth."""
    pools = assign_pools(rng, regime["s"], regime["q"], n_models)
    model_names = ["model_%d" % i for i in range(n_models)]

    true_paths: Dict[str, List[List[str]]] = {}
    for name, pool in zip(model_names, pools):
        runs = []
        for _ in range(n_runs):
            visited = sample_visited_set(rng, pool, regime["keep_prob"])
            runs.append(make_path(rng, visited, depth))
        true_paths[name] = runs

    other_counter = [0]
    noisy_paths = {
        name: [apply_noise(rng, p, rates, other_counter) for p in runs]
        for name, runs in true_paths.items()
    }

    clean_cross = cross_model_overlap(true_paths)
    noisy_cross = cross_model_overlap(noisy_paths)

    clean_within = [path_divergence(runs) for runs in true_paths.values()]
    noisy_within = [path_divergence(runs) for runs in noisy_paths.values()]

    def _avg(dicts, key):
        return statistics.mean(d[key] for d in dicts)

    return {
        "clean_cross_jaccard": clean_cross["mean_cross_jaccard"],
        "noisy_cross_jaccard": noisy_cross["mean_cross_jaccard"],
        "clean_within_jaccard": _avg(clean_within, "set_jaccard"),
        "noisy_within_jaccard": _avg(noisy_within, "set_jaccard"),
        "clean_within_prefix": _avg(clean_within, "prefix_depth"),
        "noisy_within_prefix": _avg(noisy_within, "prefix_depth"),
        "clean_within_edit": _avg(clean_within, "norm_edit_distance"),
        "noisy_within_edit": _avg(noisy_within, "norm_edit_distance"),
    }


# ======================================================================
# 7. REGIME-LEVEL AGGREGATION
# ======================================================================


def _derive_seed(base_seed: int, *parts: str) -> int:
    """Deterministic (not Python's randomized-per-process hash()) seed
    derivation so a given (base_seed, regime, variant) always reproduces
    identically across machines and runs."""
    key = "|".join(parts).encode()
    return (base_seed + zlib.crc32(key)) & 0xFFFFFFFF


def _mean_sd(xs: List[float]) -> Tuple[float, float]:
    mean = statistics.mean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return mean, sd


def simulate_variant(regime_name: str, regime: Dict, rates: Dict[str, float],
                      reps: int, base_seed: int, variant_name: str,
                      depth: int = 30, n_models: int = 4,
                      n_runs: int = 4) -> Dict:
    """Run `reps` seeded replicates of one (regime, noise-variant)
    combination and summarize clean/noisy/bias as mean +/- population sd."""
    rng = random.Random(_derive_seed(base_seed, regime_name, variant_name))
    field_names = ["clean_cross_jaccard", "noisy_cross_jaccard",
                   "clean_within_jaccard", "noisy_within_jaccard",
                   "clean_within_prefix", "noisy_within_prefix",
                   "clean_within_edit", "noisy_within_edit"]
    series = {f: [] for f in field_names}
    bias_cross = []
    bias_within_jaccard = []
    bias_within_prefix = []
    bias_within_edit = []

    for _ in range(reps):
        r = run_replicate(rng, regime, rates, depth, n_models, n_runs)
        for f in field_names:
            series[f].append(r[f])
        bias_cross.append(r["noisy_cross_jaccard"] - r["clean_cross_jaccard"])
        bias_within_jaccard.append(
            r["noisy_within_jaccard"] - r["clean_within_jaccard"])
        bias_within_prefix.append(
            r["noisy_within_prefix"] - r["clean_within_prefix"])
        bias_within_edit.append(
            r["noisy_within_edit"] - r["clean_within_edit"])

    out = {"n_reps": reps}
    for f in field_names:
        m, sd = _mean_sd(series[f])
        out[f + "_mean"] = round(m, 6)
        out[f + "_sd"] = round(sd, 6)
    for label, series_vals in (
        ("bias_cross_jaccard", bias_cross),
        ("bias_within_jaccard", bias_within_jaccard),
        ("bias_within_prefix", bias_within_prefix),
        ("bias_within_edit", bias_within_edit),
    ):
        m, sd = _mean_sd(series_vals)
        out[label + "_mean"] = round(m, 6)
        out[label + "_sd"] = round(sd, 6)
    return out


def simulate_all(rate_source: str, reps: int, base_seed: int,
                  regime_names: List[str], depth: int = 30,
                  n_models: int = 4, n_runs: int = 4) -> Dict:
    rates = EMPIRICAL_NOISE_RATES[rate_source]
    variants = {
        "net": rates,
        "synonym_only": component_only_rates(rates, "synonym"),
        "generalize_only": component_only_rates(rates, "generalize"),
        "other_only": component_only_rates(rates, "other"),
        "noise_free_sanity": noise_free_rates(),
    }
    results = {}
    for regime_name in regime_names:
        regime = REGIMES[regime_name]
        regime_out = {
            "design": {
                "s": regime["s"], "q": regime["q"],
                "keep_prob": regime["keep_prob"],
                "pool_jaccard_target": round(
                    _pool_jaccard(regime["s"], regime["q"]), 6),
            },
            "net": None,
            "decomposition": {},
        }
        for variant_name, variant_rates in variants.items():
            summary = simulate_variant(
                regime_name, regime, variant_rates, reps, base_seed,
                variant_name, depth, n_models, n_runs)
            if variant_name == "net":
                regime_out["net"] = summary
            else:
                regime_out["decomposition"][variant_name] = summary
        results[regime_name] = regime_out
    return results


# ======================================================================
# 8. CLI
# ======================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2000,
                    help="seeded replicates per (regime, noise-variant); "
                         "must be >= 1000 (spec requirement)")
    ap.add_argument("--seed", type=int, default=20260717,
                    help="base seed; regime/variant seeds are derived from "
                         "it deterministically (crc32, not hash())")
    ap.add_argument("--regimes", default=",".join(REGIME_ORDER),
                    help="comma-separated subset of: " +
                         ",".join(REGIME_ORDER))
    ap.add_argument("--rate-source", default=DEFAULT_RATE_SOURCE,
                    choices=sorted(EMPIRICAL_NOISE_RATES),
                    help="which repeat-labeling log parameterizes the noise "
                         "model (default: haiku_v2, the deployed rejector)")
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--n-models", type=int, default=4)
    ap.add_argument("--n-runs", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.reps < 1000:
        ap.error("--reps must be >= 1000 (spec: >=1000 seeded replicates "
                 "per regime)")

    regime_names = [r.strip() for r in args.regimes.split(",") if r.strip()]
    for r in regime_names:
        if r not in REGIMES:
            ap.error("unknown regime %r (valid: %s)" %
                     (r, ", ".join(REGIME_ORDER)))

    t0 = time.time()
    results = simulate_all(args.rate_source, args.reps, args.seed,
                           regime_names, args.depth, args.n_models,
                           args.n_runs)
    wall = round(time.time() - t0, 2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "EXP-006-noise-robustness",
        "command": "python3 " + " ".join(sys.argv),
        "base_seed": args.seed,
        "reps_per_variant": args.reps,
        "rate_source": args.rate_source,
        "depth": args.depth,
        "n_models": args.n_models,
        "n_runs": args.n_runs,
        "wall_seconds": wall,
        "empirical_noise_rates": EMPIRICAL_NOISE_RATES,
        "log_tallies": _LOG_TALLIES,
        "topic_ontology_n_topics": len(TOPIC_ONTOLOGY),
        "topic_ontology_n_hypernym_families": len(
            set(v["hypernym"] for v in TOPIC_ONTOLOGY.values())),
        "regimes": results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=== EXP-006 noise-robustness (%s, %d reps/variant) ===" %
          (args.rate_source, args.reps))
    for regime_name in regime_names:
        net = results[regime_name]["net"]
        print("%-10s clean_cross=%.3f noisy_cross=%.3f bias_cross=%+.4f"
              " (sd=%.4f)  bias_within=%+.4f" % (
                  regime_name, net["clean_cross_jaccard_mean"],
                  net["noisy_cross_jaccard_mean"],
                  net["bias_cross_jaccard_mean"],
                  net["bias_cross_jaccard_sd"],
                  net["bias_within_jaccard_mean"]))
    print("report: %s" % out_path)


if __name__ == "__main__":
    main()
