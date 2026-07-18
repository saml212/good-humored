#!/usr/bin/env python3
"""EXP-013 -- validates `env/bvt_gate.py`'s `BVTGateReward` against the
pre-registered fixture + bars (`docs/THEORY-MAP.md` §12.1,
EXPERIMENT_LOG.md's EXP-013/014 registration), using REAL claude:haiku
judges via `benchmark/providers.py`'s neutral-cwd CLI pattern.

Shape mirrors `benchmark/validate_banter_judge.py` (fixture -> judge ->
per-class metric -> separation/consistency/echo checks) and
`env/certify_judge_oogiri.py` (LabelCache + `--max-calls` budget guard,
cache/resume-safe). `VIOLATION_PROMPT`/`BENIGN_PROMPT` are imported
VERBATIM from `env.bvt_gate` -- this script never redefines them, so a
future prompt edit in the gate module is automatically what gets
validated next run, never a stale copy.

Design note on how `BVTGateReward` is actually exercised: this script does
its OWN judge-call orchestration (cache lookup, retry, budget guard,
per-repeat independence) OUTSIDE the class -- exactly how every other
validator in this repo treats the class/instrument split
(`env/certify_judge_oogiri.py`'s `score_candidate` sits outside
`JudgePreferenceReward` the same way; `env/validate_semantic_novelty.py`'s
module docstring states the principle directly: the class's own
scoring/ramping logic is exercised by its unit tests with fake
embedders/judges, a VALIDATOR script's job is producing the real NUMBERS).
Once both real (violation, benign) scores for one repeat are in hand, this
script wraps them in trivial replay lambdas and calls the REAL
`BVTGateReward.__call__` on them -- so the actual production multiplication
+ `[0, 1]` bounds-check code path is what gets scored, not a
reimplementation of `v * b`.

Cache keys include the repeat index (`"<repeat>\\x1f<text>"`) -- repeats
must be genuinely independent live calls for repeat-consistency to mean
anything; caching by text alone would make repeat 2/3 free replays of
repeat 1 and manufacture perfect consistency.

Usage:
  PYTHONNOUSERSITE=1 python3 -m env.validate_bvt_gate \\
      --judge claude:haiku --repeats 3 --max-calls 900 \\
      --out experiment-runs/2026-07-17-bvt-gate-validation
"""

import argparse
import json
import statistics
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
from benchmark.validate_banter_judge import ECHO_RISK_THRESHOLD, _pearson  # noqa: E402
from env.bvt_gate import BENIGN_PROMPT, VIOLATION_PROMPT, BVTGateReward  # noqa: E402
from env.rewards import ComprehensibilityReward, JudgePreferenceReward  # noqa: E402

FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "bvt_gate_fixture.jsonl"
GOLD_CLASSES = ("both", "violation_only", "benign_only", "neither")
GAMING_CLASS = "disclaimer_washed_violation"
ALL_CLASSES = GOLD_CLASSES + (GAMING_CLASS,)

PROMPT_VERSION = "exp013-v1"
VIOLATION_LABELER = "bvt-violation-%s" % PROMPT_VERSION
BENIGN_LABELER = "bvt-benign-%s" % PROMPT_VERSION
DISPROOF_LABELER = "bvt-disproof-judge-%s" % PROMPT_VERSION

# Discrete PASS/FAIL operating threshold for the repeat-consistency check
# ONLY (docs/THEORY-MAP.md §12.1's registered bar) -- NOT the reward
# itself, which stays a continuous product everywhere else in this script.
REPEAT_PASS_THRESHOLD = 0.2

# Unit separator, same convention as env/certify_judge_oogiri.py's
# cache_key -- item text may contain any printable character.
_SEP = "\x1f"


# ------------------------------------------------------------- fixture I/O


def load_fixture() -> List[Dict]:
    with open(FIXTURE) as f:
        items = [json.loads(line) for line in f if line.strip()]
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fixture ids in %s" % FIXTURE)
    for i in items:
        if i["gold_class"] not in ALL_CLASSES:
            raise ValueError("item %r has unknown gold_class %r"
                             % (i["id"], i["gold_class"]))
    return items


# ------------------------------------------------------------- budget guard


class BudgetExceeded(Exception):
    """Raised once --max-calls underlying judge calls have been made.
    Mirrors env/certify_judge_oogiri.py's BudgetExceeded exactly."""


def make_budgeted_complete(complete: Callable[[str], str], stats: Dict,
                           max_calls: int) -> Callable[[str], str]:
    def wrapped(prompt: str) -> str:
        if stats["calls"] >= max_calls:
            raise BudgetExceeded(
                "validate_bvt_gate: hit --max-calls=%d underlying judge "
                "calls" % max_calls)
        stats["calls"] += 1
        return complete(prompt)
    return wrapped


# ------------------------------------------------------------- score parse

import re  # noqa: E402

# 0-10, NOT 1-10 -- VIOLATION_PROMPT/BENIGN_PROMPT explicitly allow 0
# ("no violation of any kind" / "a genuine, unmitigated threat"), unlike
# env/certify_judge_oogiri.py's 1-10 funniness scale. (?<!\d)/(?!\d) guards
# keep "100"/"23" from being misread as "10"/"2" -- same pattern as
# benchmark/banter.py's _parse_score.
_SCORE_RE = re.compile(r"(?<!\d)(10|[0-9])(?!\d)")


def _parse_score(raw: str) -> Optional[int]:
    if not raw or not raw.strip():
        return None
    first_line = raw.strip().splitlines()[0]
    m = _SCORE_RE.search(first_line)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------- judge calls


def judge_raw_score(direction: str, text: str, repeat: int,
                    prompt_template: str, complete: Callable[[str], str],
                    cache: LabelCache, stats: Dict, retries: int
                    ) -> Optional[int]:
    """Cache-checked, budgeted raw 0-10 score for one (direction, text,
    repeat) triple. Returns None (never cached) if unparseable even after
    `retries` retries -- mirrors env/certify_judge_oogiri.py's
    score_candidate convention (an unparseable call is dropped, not
    zeroed)."""
    labeler = VIOLATION_LABELER if direction == "violation" else BENIGN_LABELER
    key = "%d%s%s" % (repeat, _SEP, text)
    cached = cache.get(labeler, key)
    if cached is not None:
        stats["cache_hits"] = stats.get("cache_hits", 0) + 1
        return cached
    prompt = prompt_template.format(completion=text)
    raw = None
    for _ in range(retries + 1):
        response = complete(prompt)
        raw = _parse_score(response)
        if raw is not None:
            break
    if raw is None:
        stats["unparseable"] = stats.get("unparseable", 0) + 1
        return None
    cache.put(labeler, key, raw)
    return raw


def score_repeat(item: Dict, repeat: int, complete: Callable[[str], str],
                 cache: LabelCache, stats: Dict, retries: int) -> Dict:
    """One repeat's (violation_raw, benign_raw, product) for one fixture
    item. `product` is computed by calling the REAL BVTGateReward.__call__
    on trivial replay judges wrapping the two already-obtained scores (see
    module docstring) -- product is None if either score was unparseable."""
    text = item["text"]
    v_raw = judge_raw_score("violation", text, repeat, VIOLATION_PROMPT,
                            complete, cache, stats, retries)
    b_raw = judge_raw_score("benign", text, repeat, BENIGN_PROMPT,
                            complete, cache, stats, retries)
    if v_raw is None or b_raw is None:
        return {"violation_raw": v_raw, "benign_raw": b_raw, "product": None}
    v_norm, b_norm = v_raw / 10.0, b_raw / 10.0
    gate = BVTGateReward(violation_judge=lambda p, c: v_norm,
                         benign_judge=lambda p, c: b_norm, weight=1.0)
    product = gate(prompts=[None], completions=[text])[0]
    return {"violation_raw": v_raw, "benign_raw": b_raw, "product": product}


def run_validation(items: List[Dict], complete: Callable[[str], str],
                   cache: LabelCache, max_calls: int, repeats: int,
                   retries: int, raw_log=None) -> Dict:
    """Runs `repeats` repeats over every fixture item, in order, stopping
    cleanly (no exception escapes) if --max-calls is exhausted partway
    through. The repeat in progress when the budget is hit is dropped;
    everything already cached on disk is reused free on the next
    invocation."""
    stats: Dict = {"calls": 0, "cache_hits": 0, "unparseable": 0}
    budgeted = make_budgeted_complete(complete, stats, max_calls)
    per_item: Dict[str, List[Dict]] = {}
    budget_exhausted = False
    for item in items:
        per_item[item["id"]] = []
        for r in range(repeats):
            try:
                result = score_repeat(item, r, budgeted, cache, stats, retries)
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


# --------------------------------------------------------------- metrics


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def item_means(per_item: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Item-level mean-of-valid-repeats for violation_raw/benign_raw/
    product -- mirrors benchmark/validate_banter_judge.py's
    item_mean_deltas (item-then-class averaging, not raw pooled
    averaging, so items with more successfully-parsed repeats don't get
    extra weight in the class mean)."""
    out = {}
    for item_id, repeats in per_item.items():
        v = [r["violation_raw"] for r in repeats if r["violation_raw"] is not None]
        b = [r["benign_raw"] for r in repeats if r["benign_raw"] is not None]
        p = [r["product"] for r in repeats if r["product"] is not None]
        out[item_id] = {
            "violation_raw_mean": _mean(v), "benign_raw_mean": _mean(b),
            "product_mean": _mean(p), "n_valid_repeats": len(repeats),
            "n_scored_product": len(p),
        }
    return out


def per_class_stats(items: List[Dict], means: Dict[str, Dict]) -> Dict:
    out = {}
    for cls in ALL_CLASSES:
        ids = [i["id"] for i in items if i["gold_class"] == cls]
        v = [means[i]["violation_raw_mean"] for i in ids
            if means[i]["violation_raw_mean"] is not None]
        b = [means[i]["benign_raw_mean"] for i in ids
            if means[i]["benign_raw_mean"] is not None]
        p = [means[i]["product_mean"] for i in ids
            if means[i]["product_mean"] is not None]
        out[cls] = {
            "n_items": len(ids), "n_scored": len(p),
            "violation_raw_mean": _mean(v), "benign_raw_mean": _mean(b),
            "product_mean": _mean(p),
        }
    return out


# Deliberately crude, hand-picked lexicons -- same spirit as
# benchmark/validate_rejector.py's keyword_baseline ("deliberately crude:
# it is the disproof bar, not a serious labeler"). Not a sentiment model;
# a cheap surface-token counter, which is exactly what the echo-
# hackability check needs to stand in for ("a cheap profanity/taboo-
# keyword count", §12.1). Built by reading violation_only's own 8 items
# (see env/tests/fixtures/bvt_gate_fixture.jsonl) for their recurring
# negatively-charged vocabulary.
_SHOCK_WORDS = (
    "worthless", "hate", "hopes you fail", "fail", "disappear", "pity",
    "keyed", "angry", "fired", "quit", "resent", "rumor", "spreading",
    "sabotage", "threat", "cruel", "hurt", "damage", "betray",
)

# Disclaimer/fictional-frame markers -- built by reading
# disclaimer_washed_violation's own 8 items for their recurring appended
# tokens.
_DISCLAIMER_MARKERS = (
    "jk", "kidding", "lol", "haha", "just a joke", "hypothetical",
    "fictional", "not real", "silly", "good fun", "promise", "no offense",
    "😂",
)


def _keyword_count(text: str, lexicon: Sequence[str]) -> int:
    t = text.lower()
    return sum(t.count(w) for w in lexicon)


def shock_word_count(text: str) -> int:
    return _keyword_count(text, _SHOCK_WORDS)


def disclaimer_marker_count(text: str) -> int:
    return _keyword_count(text, _DISCLAIMER_MARKERS)


def echo_hackability_check(items: List[Dict], means: Dict[str, Dict]) -> Dict:
    """Two Pearson-r trip-wires, reusing validate_banter_judge.py's
    ECHO_RISK_THRESHOLD=0.5 convention and _pearson helper exactly (no
    reimplementation).

    1. violation_score vs shock_word_count, over `violation_only` (n=8):
       this is the one gold class with a uniformly HIGH, roughly-constant
       TRUE violation level and no protective frame -- the class where
       true label doesn't vary, so any judge-score variation the shock-
       word count explains is exactly the "cheap keyword count is doing
       the work, not a holistic read of severity" signature the check is
       for. (§12.1's own text names "a class where a cheap profanity/
       taboo-keyword count varies independently of true violation label"
       without naming which committed class satisfies that -- this is the
       documented interpretive choice; see this script's module docstring
       header note and the final report for the flag.)
    2. benign_score vs disclaimer_marker_count, over
       disclaimer_washed_violation UNION both (n=16) -- unambiguous per
       the registration text.
    """
    def _by_ids(ids, score_key):
        xs, ys = [], []
        for i in ids:
            m = means[i["id"]]
            if m[score_key] is None:
                continue
            text = i["text"]
            xs.append(m[score_key])
            ys.append(shock_word_count(text) if score_key == "violation_raw_mean"
                     else disclaimer_marker_count(text))
        return xs, ys

    violation_only_items = [i for i in items if i["gold_class"] == "violation_only"]
    xs1, ys1 = _by_ids(violation_only_items, "violation_raw_mean")
    r_shock = _pearson(xs1, ys1)

    washed_and_both = [i for i in items
                       if i["gold_class"] in ("disclaimer_washed_violation", "both")]
    xs2, ys2 = _by_ids(washed_and_both, "benign_raw_mean")
    r_disclaimer = _pearson(xs2, ys2)

    return {
        "violation_vs_shock_word_count": {
            "class": "violation_only", "n": len(xs1), "pearson_r": r_shock,
            "risk_detected": bool(r_shock is not None and r_shock > ECHO_RISK_THRESHOLD),
        },
        "benign_vs_disclaimer_marker_count": {
            "class": "disclaimer_washed_violation ∪ both", "n": len(xs2),
            "pearson_r": r_disclaimer,
            "risk_detected": bool(r_disclaimer is not None and r_disclaimer > ECHO_RISK_THRESHOLD),
        },
        "threshold": ECHO_RISK_THRESHOLD,
    }


def repeat_consistency(per_item: Dict[str, List[Dict]]) -> float:
    """Pooled pairwise PASS/FAIL agreement across repeats, over the FULL
    fixture (all 40 items) -- mirrors benchmark/validate_rejector.py's
    repeat_consistency exactly (pooled agreement, not item-then-mean).
    PASS/FAIL is `product > REPEAT_PASS_THRESHOLD` (a fixed 0.2 operating
    point for THIS check only, per §12.1's registration). Items with < 2
    valid (parseable-product) repeats contribute 0 pairs, same "no
    fabricated agreement from a lone value" discipline as
    validate_banter_judge.py's _stdev."""
    agrees, total = 0, 0
    for item_id, repeats in per_item.items():
        labels = [r["product"] > REPEAT_PASS_THRESHOLD for r in repeats
                 if r["product"] is not None]
        for x in range(len(labels)):
            for y in range(x + 1, len(labels)):
                agrees += labels[x] == labels[y]
                total += 1
    return agrees / total if total else 1.0


# --------------------------------------------------------------- disproof


DISPROOF_JUDGE_PROMPT = """Rate how funny and well-crafted this piece of text is on its own.

Text: {completion}

On a scale from 1 (not funny / poorly crafted) to 10 (extremely funny / well-crafted), output ONLY a single integer from 1 to 10, nothing else.
"""


def make_disproof_judge(complete: Callable[[str], str], cache: LabelCache,
                        stats: Dict, retries: int
                        ) -> Callable[[object, str], float]:
    """Callable[[Any, str], float] matching JudgePreferenceReward's judge
    contract. On repeated parse failure, falls back to a neutral 0.5
    (recorded in stats) rather than raising -- JudgePreferenceReward.
    __call__ does `float(self.judge(p, c))` with no None-sentinel path,
    unlike this script's own judge_raw_score, so there is no way to
    signal "unparseable" back through that contract without either
    raising (killing the whole disproof pass on one flaky call) or
    substituting a stated neutral default. Chosen: neutral default,
    logged loudly, not silently."""
    def judge(prompt, completion):
        cached = cache.get(DISPROOF_LABELER, completion)
        if cached is not None:
            stats["cache_hits"] = stats.get("cache_hits", 0) + 1
            return cached
        p = DISPROOF_JUDGE_PROMPT.format(completion=completion)
        raw = None
        for _ in range(retries + 1):
            raw = _parse_score(complete(p))
            if raw is not None:
                break
        if raw is None:
            stats["disproof_unparseable_fallback"] = \
                stats.get("disproof_unparseable_fallback", 0) + 1
            normalized = 0.5
        else:
            normalized = raw / 10.0
        cache.put(DISPROOF_LABELER, completion, normalized)
        return normalized
    return judge


def run_disproof_check(items: List[Dict], complete: Callable[[str], str],
                       cache: LabelCache, stats: Dict, max_calls: int,
                       retries: int) -> Dict:
    """CLAUDE.md checklist item 4 / EXP-005-style disproof check: does the
    CURRENT additive stack (judge_weight * judge_score + comprehensibility,
    RewardConfig's own defaults: judge_weight=1.0,
    comprehensibility_weight=0.3) under-penalize `violation_only` relative
    to `both`, the failure §12.1's own text predicts? Includes `both` as
    the comparison anchor (not itself part of the registered bar, but the
    disproof's own prose invokes it directly: "...can still score
    adequately... relative to both").

    ONE pass (not repeated across `repeats`) -- this is a supplementary
    comparison metric, not part of the main consistency measurement.

    `max_calls` is the ABSOLUTE run-wide cap (not a remaining-calls
    delta) -- `stats` is the SAME dict `run_validation` already
    incremented, so `make_budgeted_complete`'s `stats["calls"] >=
    max_calls` check must compare against the same absolute number the
    main run used, or this call under-budgets itself by exactly however
    many calls the main run already spent."""
    budgeted = make_budgeted_complete(complete, stats, max_calls)
    judge = make_disproof_judge(budgeted, cache, stats, retries)
    jp = JudgePreferenceReward(judge=judge, weight=1.0)
    comp = ComprehensibilityReward(weight=0.3)

    out = {}
    try:
        for cls in ("violation_only", "benign_only", "both"):
            its = [i for i in items if i["gold_class"] == cls]
            texts = [i["text"] for i in its]
            jp_scores = jp(prompts=[None] * len(texts), completions=texts)
            comp_scores = comp(prompts=[None] * len(texts), completions=texts)
            additive = [a + b for a, b in zip(jp_scores, comp_scores)]
            out[cls] = {
                "n_items": len(its),
                "judge_preference_mean": _mean(jp_scores),
                "comprehensibility_mean": _mean(comp_scores),
                "additive_stack_mean": _mean(additive),
            }
    except BudgetExceeded:
        out["budget_exhausted"] = True
    return out


# ------------------------------------------------------------------ bars


def build_bars(per_class: Dict, disclaimer_class_stats: Dict,
               echo: Dict, consistency: float) -> List[Dict]:
    def bar(name, value, op, threshold):
        if value is None:
            passed = False
        elif op == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        return {"name": name, "value": value, "op": op, "threshold": threshold,
               "passed": bool(passed)}

    both, vio, ben, nei = (per_class["both"], per_class["violation_only"],
                           per_class["benign_only"], per_class["neither"])
    sep_violation = (both["violation_raw_mean"] - ben["violation_raw_mean"]
                     if both["violation_raw_mean"] is not None and
                     ben["violation_raw_mean"] is not None else None)
    sep_benign = (both["benign_raw_mean"] - vio["benign_raw_mean"]
                 if both["benign_raw_mean"] is not None and
                 vio["benign_raw_mean"] is not None else None)

    bars = [
        bar("separation_both_violation_minus_benign_only_violation",
           sep_violation, ">=", 5.0),
        bar("separation_both_benign_minus_violation_only_benign",
           sep_benign, ">=", 5.0),
        bar("both_product_mean", both["product_mean"], ">=", 0.50),
        bar("violation_only_product_mean", vio["product_mean"], "<=", 0.15),
        bar("benign_only_product_mean", ben["product_mean"], "<=", 0.15),
        bar("neither_product_mean", nei["product_mean"], "<=", 0.10),
        bar("disclaimer_washed_benign_raw_mean",
           disclaimer_class_stats["benign_raw_mean"], "<=", 3.0),
        bar("echo_violation_vs_shock_word_r_not_risky",
           0.0 if not echo["violation_vs_shock_word_count"]["risk_detected"] else 1.0,
           "<=", 0.0),
        bar("echo_benign_vs_disclaimer_marker_r_not_risky",
           0.0 if not echo["benign_vs_disclaimer_marker_count"]["risk_detected"] else 1.0,
           "<=", 0.0),
        bar("repeat_consistency", consistency, ">=", 0.85),
    ]
    return bars


# --------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="claude:haiku",
                    help="provider spec, resolved via get_provider() "
                         "(benchmark/providers.py) -- neutral-cwd CLI "
                         "pattern, real calls only.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--max-calls", type=int, default=900,
                    help="hard cap on underlying judge calls (including "
                         "retries and the disproof-check pass); stop "
                         "cleanly with a partial report if hit.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.repeats < 2:
        ap.error("--repeats must be >= 2: consistency needs repetition "
                 "(mirrors validate_rejector.py's audit N4)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    command = ("python3 -m env.validate_bvt_gate --judge %s --repeats %d "
              "--retries %d --max-calls %d --out %s" %
              (args.judge, args.repeats, args.retries, args.max_calls,
               args.out))
    print("command: %s" % command)

    items = load_fixture()
    complete = get_provider(args.judge)
    cache = LabelCache(out / "label_cache.jsonl")

    run_stamp = time.strftime("%Y%m%dT%H%M%S")
    raw_path = out / ("bvt_raw.%s.jsonl" % run_stamp)
    t0 = time.time()
    try:
        with open(raw_path, "a") as raw_log:
            result = run_validation(items, complete, cache, args.max_calls,
                                    args.repeats, args.retries,
                                    raw_log=raw_log)
            # args.max_calls is the ABSOLUTE cap, not a remaining-calls
            # delta -- result["stats"] is the same dict run_validation
            # already incremented, see run_disproof_check's docstring.
            disproof = run_disproof_check(items, complete, cache,
                                          result["stats"], args.max_calls,
                                          args.retries)
    finally:
        cache.close()
    wall = time.time() - t0

    means = item_means(result["per_item"])
    per_class = per_class_stats(items, means)
    margin = None
    if per_class["both"]["product_mean"] is not None:
        others = [per_class[c]["product_mean"] for c in
                 ("violation_only", "benign_only", "neither")
                 if per_class[c]["product_mean"] is not None]
        if others:
            margin = per_class["both"]["product_mean"] - max(others)
    echo = echo_hackability_check(items, means)
    consistency = repeat_consistency(result["per_item"])
    bars = build_bars(per_class, per_class[GAMING_CLASS], echo, consistency)

    report = {
        "experiment": "EXP-013-bvt-gate-validation",
        "command": command,
        "run_stamp": run_stamp,
        "judge": args.judge,
        "repeats": args.repeats,
        "retries": args.retries,
        "n_items": len(items),
        "wall_seconds": round(wall, 1),
        "violation_prompt": VIOLATION_PROMPT,
        "benign_prompt": BENIGN_PROMPT,
        "per_class": per_class,
        "margin_both_minus_max_other": margin,
        "predicted_margin_registered": 0.40,
        "echo_hackability_check": echo,
        "repeat_consistency": consistency,
        "disproof_check_additive_stack": disproof,
        "bars": bars,
        "all_bars_pass": all(b["passed"] for b in bars),
        "call_stats": {**result["stats"], "max_calls": args.max_calls,
                      "budget_exhausted": result["budget_exhausted"]},
        "cache_path": str(out / "label_cache.jsonl"),
        "raw_log_path": str(raw_path),
        "item_means": means,
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=== EXP-013 BVT gate validation (%s, %d repeats, %d items) ===" %
         (args.judge, args.repeats, len(items)))
    for cls in ALL_CLASSES:
        pc = per_class[cls]
        print("%-28s n=%d  violation=%s  benign=%s  product=%s" % (
            cls, pc["n_scored"],
            "None" if pc["violation_raw_mean"] is None else "%.2f" % pc["violation_raw_mean"],
            "None" if pc["benign_raw_mean"] is None else "%.2f" % pc["benign_raw_mean"],
            "None" if pc["product_mean"] is None else "%.3f" % pc["product_mean"]))
    print("margin (both - max other) = %s (predicted +0.40)" %
         ("None" if margin is None else "%.3f" % margin))
    print("repeat_consistency = %.3f (bar >= 0.85)" % consistency)
    print("echo risk: violation/shock=%s benign/disclaimer=%s" %
         (echo["violation_vs_shock_word_count"]["risk_detected"],
          echo["benign_vs_disclaimer_marker_count"]["risk_detected"]))
    print()
    print("=== bars ===")
    for b in bars:
        status = "PASS" if b["passed"] else "FAIL"
        print("[%s] %-55s value=%s %s %s" %
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
