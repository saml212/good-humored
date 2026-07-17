"""Banter judge validation — the instrument-validation harness for Track 2
(docs/BENCHMARK.md Section 1b).

Track 2's measuring instrument is an LLM judge doing context-ablation
scoring: context_ablation_score() in benchmark/banter.py, delta =
score(reply | true context) - score(reply | swapped context). Before any
Track 2 pilot runs, this script checks that instrument against a
hand-built fixture with known structure — mirrors validate_rejector.py,
Track 1's template (EXPERIMENT_LOG.md EXP-001 through EXP-003b):

  1. per-class mean delta + spread — does the ordering the design predicts
     actually show up (contextual > generic_responsive > canned)?
  2. separation(contextual - canned) and separation(contextual -
     generic_responsive) — the load-bearing numbers. Canned should show
     ~0 delta (context-blind by construction: a verbatim classic joke
     from the memorized-joke corpus, benchmark/fixtures/... drawn from
     ~/Experiments/good-humored-data/corpus/chatgpt-25-templates.jsonl).
     Contextual should show a large positive delta. generic_responsive
     (on-topic but substantively interchangeable) should sit in between —
     this class exists specifically to test whether the delta tracks real
     contextual dependence or just topical keyword overlap, the residual
     risk BENCHMARK.md Section 1b states plainly.
  3. repeat consistency of deltas — same item, K repeats, same judge: how
     much does the delta itself jitter? (Continuous analogue of
     validate_rejector's repeat_consistency; that one is a [0,1]
     agreement fraction over discrete labels, this one is a spread over a
     continuous score, so it is reported directly in points, not
     normalized — see repeat_delta_stdev_mean's docstring.)
  4. keyword_echo_check — the residual-risk test, made mechanical: is the
     judge just rewarding literal keyword overlap between reply and
     context, rather than genuine contextual fit? If generic_responsive
     items whose reply happens to share more surface vocabulary with
     their context get deltas that approach contextual-sized deltas, the
     judge is echo-hackable exactly as BENCHMARK.md Section 1b warns a
     policy could learn to exploit.

Fixture-authoring failure class to avoid (EXPERIMENT_LOG.md audit-W6,
carried over from Track 1): an item that accidentally tests TWO things at
once. Here that is a "contextual" reply that would also read as funny
generically, or a "canned" reply that happens to fit its context by
coincidence. Every fixture item's `notes` field states its single
purpose; benchmark/fixtures/banter_judge_validation.jsonl's contexts are
also kept keyword-disjoint from each other so swap partners are
genuinely unrelated conversations, not just superficially different.

Usage:
  python3 -m benchmark.validate_banter_judge --judge claude:haiku \
      --repeats 2 --out experiment-runs/2026-07-17-banter-judge-validation
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .banter import (JUDGE_PROMPT_VERSION, _content_words, _format_turns,
                     context_ablation_score, swap_partner)
from .providers import get_provider

FIXTURES = Path(__file__).parent / "fixtures" / "banter_judge_validation.jsonl"
GOLD_CLASSES = ("contextual", "generic_responsive", "canned")

# Pearson r above this on the generic_responsive-only keyword-overlap-vs-
# delta correlation counts as "the residual risk is live" — a judge that
# is really tracking contextual fit should show LOW correlation on this
# class specifically, because generic_responsive replies are on-topic
# (so overlap varies) but never substantively responsive (so a sound
# judge's delta should stay flat regardless of overlap). Threshold is a
# coarse trip-wire, not a precision instrument — see keyword_echo_check's
# docstring for what a positive reading does and does not prove.
ECHO_RISK_THRESHOLD = 0.5


def load_fixtures() -> List[Dict]:
    with open(FIXTURES) as f:
        items = [json.loads(line) for line in f if line.strip()]
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        dupes = [i for i in ids if ids.count(i) > 1]
        raise ValueError("duplicate fixture ids: %s" % sorted(set(dupes)))
    for i in items:
        if i["gold_class"] not in GOLD_CLASSES:
            raise ValueError("item %r has unknown gold_class %r"
                             % (i["id"], i["gold_class"]))
    return items


def _context_text(item: Dict) -> str:
    """Render a fixture item's context turns with banter.py's own
    formatter (not reimplemented) so the judge sees exactly the same
    shape of context block a real pilot run would produce."""
    return _format_turns(item["context"])


def keyword_overlap(reply: str, context_text: str) -> float:
    """Jaccard overlap of >=5-char content words between a reply and a
    context block, reusing banter.py's _content_words tokenizer (the same
    stopword/length filtering detect_callback uses) instead of
    reimplementing tokenization. Purely a surface-form signal — exactly
    the kind of shortcut an echo-hackable judge would be tracking instead
    of genuine contextual fit."""
    r = _content_words(reply)
    c = _content_words(context_text)
    u = r | c
    return len(r & c) / len(u) if u else 0.0


def run_fixture_with_judge(
    items: List[Dict],
    judge_complete,
    repeats: int,
    raw_log=None,
) -> Dict[str, List[Dict]]:
    """Score every fixture item `repeats` times against context_ablation_score.

    Swap partner for item i is swap_partner(i, len(items)) — banter.py's
    deterministic, index-based pairing rule (no randomness), applied to
    the fixture's own file order, so a re-run is reproducible from the
    fixture file alone with no separate pairing manifest needed. Returns
    {item_id: [ablation_dict, ...]}, one dict per repeat as returned by
    context_ablation_score (true_score/swapped_score/delta/
    judge_prompt_version)."""
    n = len(items)
    contexts = [_context_text(it) for it in items]
    ablations: Dict[str, List[Dict]] = {}
    for i, item in enumerate(items):
        swap_idx = swap_partner(i, n)
        true_ctx = contexts[i]
        swapped_ctx = contexts[swap_idx]
        ablations[item["id"]] = []
        for k in range(repeats):
            result = context_ablation_score(item["reply"], true_ctx,
                                            swapped_ctx, judge_complete)
            ablations[item["id"]].append(result)
            if raw_log is not None:
                raw_log.write(json.dumps({
                    "id": item["id"], "repeat": k,
                    "gold_class": item["gold_class"],
                    "swap_partner_id": items[swap_idx]["id"],
                    **result,
                }) + "\n")
                raw_log.flush()
    return ablations


def _stdev(xs: Sequence[float]) -> float:
    """Sample standard deviation. 0.0 for n < 2 (no spread is definable,
    and a lone value should never look "perfectly consistent" by
    omission — callers must gate on n before trusting this)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pure-stdlib Pearson correlation coefficient (no numpy dependency —
    same "importable anywhere" convention as stats.py). None when either
    series has zero variance: a constant judge (all deltas identical, or
    all overlaps identical) makes the correlation mathematically
    undefined, and reporting 0.0 in that case would misleadingly read as
    "no echo risk" rather than "cannot tell from this data" — callers
    must treat None as "inconclusive," not "safe."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def item_mean_deltas(ablations: Dict[str, List[Dict]]) -> Dict[str, Optional[float]]:
    """Mean delta per item across repeats, ignoring unparseable
    (None-delta) repeats. None if every repeat for that item failed to
    parse — mirrors banter.py's own None-sentinel convention rather than
    silently defaulting to 0, which would fabricate a fake "no context
    dependence" reading for an item the judge simply never answered."""
    out: Dict[str, Optional[float]] = {}
    for item_id, results in ablations.items():
        deltas = [r["delta"] for r in results if r["delta"] is not None]
        out[item_id] = sum(deltas) / len(deltas) if deltas else None
    return out


def keyword_echo_check(
    items: List[Dict],
    mean_deltas: Dict[str, Optional[float]],
    overlaps: Dict[str, float],
) -> Dict:
    """Is the judge rewarding keyword overlap instead of genuine
    contextual fit? (BENCHMARK.md Section 1b's stated residual risk, made
    measurable.)

    Two views, because a whole-fixture correlation is ambiguous on its
    own: contextual replies are BOTH genuinely responsive AND likely to
    share more vocabulary with their context (real responsiveness often
    looks like keyword reuse), so a positive overall correlation is
    expected even from a good judge and proves nothing by itself.
    generic_responsive items break that confound by design — they are
    on-topic (so keyword overlap varies across them) but never
    substantively responsive (so a sound judge's delta should stay flat
    regardless of overlap). A positive correlation WITHIN
    generic_responsive specifically — or, stated the way the design brief
    frames it, generic_responsive items with high overlap scoring
    contextual-sized deltas — is the operational definition of "the judge
    is echo-hackable" used here.

    risk_detected is a coarse trip-wire (ECHO_RISK_THRESHOLD on the
    generic_responsive-only correlation), not a precision test: it can
    miss a subtler echo effect and it can occasionally fire on
    genuinely-noisy small-N data (10 generic_responsive items). Treat a
    positive reading as "go look at the raw per-item log," not as a
    standalone verdict.
    """
    generic_ids = [it["id"] for it in items
                   if it["gold_class"] == "generic_responsive"
                   and mean_deltas.get(it["id"]) is not None]
    scored_ids = [it["id"] for it in items if mean_deltas.get(it["id"]) is not None]

    overall_r = _pearson([overlaps[i] for i in scored_ids],
                        [mean_deltas[i] for i in scored_ids])
    generic_r = _pearson([overlaps[i] for i in generic_ids],
                        [mean_deltas[i] for i in generic_ids])

    # Literal split test: within generic_responsive, do the high-overlap
    # replies score like contextual replies? Split at the median overlap
    # value (ties broken by list order — fine at n=10, not a statistic
    # that needs to be exact).
    sorted_generic = sorted(generic_ids, key=lambda i: overlaps[i])
    half = len(sorted_generic) // 2
    low_ids, high_ids = sorted_generic[:half], sorted_generic[half:]

    def _class_mean(ids: List[str]) -> Optional[float]:
        vals = [mean_deltas[i] for i in ids]
        return sum(vals) / len(vals) if vals else None

    risk_detected = bool(generic_r is not None and generic_r > ECHO_RISK_THRESHOLD)

    return {
        "overall_pearson_r": overall_r,
        "generic_responsive_pearson_r": generic_r,
        "generic_responsive_low_overlap_mean_delta": _class_mean(low_ids),
        "generic_responsive_high_overlap_mean_delta": _class_mean(high_ids),
        "risk_detected": risk_detected,
        "threshold": ECHO_RISK_THRESHOLD,
        "n_generic_responsive_scored": len(generic_ids),
    }


def score(items: List[Dict], ablations: Dict[str, List[Dict]]) -> Dict:
    """Compute all validation metrics for one judge's raw ablation
    results. Pure function (no I/O) so tests can feed it hand-built or
    fake-judge-produced `ablations` directly, mirroring validate_rejector
    .py's score()."""
    mean_deltas = item_mean_deltas(ablations)
    overlaps = {it["id"]: keyword_overlap(it["reply"], _context_text(it))
               for it in items}

    per_class: Dict[str, Dict] = {}
    for cls in GOLD_CLASSES:
        ids = [it["id"] for it in items if it["gold_class"] == cls]
        vals = [mean_deltas[i] for i in ids if mean_deltas[i] is not None]
        per_class[cls] = {
            "n_items": len(ids),
            "n_scored": len(vals),
            "mean_delta": (sum(vals) / len(vals)) if vals else None,
            "stdev_delta": _stdev(vals) if len(vals) >= 2 else 0.0,
            "min_delta": min(vals) if vals else None,
            "max_delta": max(vals) if vals else None,
        }

    def _separation(a: str, b: str) -> Optional[float]:
        ma, mb = per_class[a]["mean_delta"], per_class[b]["mean_delta"]
        return (ma - mb) if (ma is not None and mb is not None) else None

    # Repeat consistency: for each item with >= 2 parseable repeats, the
    # stdev of ITS OWN per-repeat deltas; report the mean across items.
    # Lower = more consistent. NOT a [0,1] agreement fraction like
    # validate_rejector's repeat_consistency (that metric is over
    # discrete labels; delta is continuous, so spread is reported in
    # scale points directly — see the pre-registration proposal for the
    # bar this is checked against).
    per_item_repeat_stdev = []
    n_unparseable_repeats = 0
    for results in ablations.values():
        deltas = [r["delta"] for r in results if r["delta"] is not None]
        n_unparseable_repeats += len(results) - len(deltas)
        if len(deltas) >= 2:
            per_item_repeat_stdev.append(_stdev(deltas))
    repeat_delta_stdev_mean = (
        sum(per_item_repeat_stdev) / len(per_item_repeat_stdev)
        if per_item_repeat_stdev else 0.0)

    return {
        "per_class": per_class,
        "separation_contextual_minus_canned": _separation("contextual", "canned"),
        "separation_contextual_minus_generic_responsive":
            _separation("contextual", "generic_responsive"),
        "repeat_delta_stdev_mean": round(repeat_delta_stdev_mean, 4),
        "n_unparseable_repeats": n_unparseable_repeats,
        "keyword_echo_check": keyword_echo_check(items, mean_deltas, overlaps),
        "item_mean_deltas": mean_deltas,
        "item_keyword_overlap": {k: round(v, 4) for k, v in overlaps.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="claude:haiku",
                    help="provider spec, e.g. claude:haiku, codex:mini, "
                         "api:deepseek (benchmark/providers.py get_provider)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.repeats < 2:
        ap.error("--repeats must be >= 2: delta consistency needs "
                 "repetition (mirrors validate_rejector.py's audit N4 — "
                 "repeats=1 fakes perfect consistency)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_fixtures()
    n_by_class = Counter(it["gold_class"] for it in items)
    judge_complete = get_provider(args.judge)

    # Timestamped per invocation — re-runs must never interleave into one
    # indistinguishable file (mirrors validate_rejector.py's audit B2).
    run_stamp = time.strftime("%Y%m%dT%H%M%S")
    raw_path = out / ("banter_judge_raw.%s.jsonl" % run_stamp)
    t0 = time.time()
    with open(raw_path, "a") as raw_log:
        ablations = run_fixture_with_judge(items, judge_complete,
                                           args.repeats, raw_log=raw_log)

    report = {
        "experiment": "banter-judge-validation",
        "run_stamp": run_stamp,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge": args.judge,
        "repeats": args.repeats,
        "n_items": len(items),
        "n_by_class": dict(n_by_class),
        "wall_seconds": round(time.time() - t0, 1),
        **score(items, ablations),
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=== banter judge validation (%s, %d repeats) ===" %
          (args.judge, args.repeats))
    for cls in GOLD_CLASSES:
        pc = report["per_class"][cls]
        print("%-20s n=%d  mean_delta=%s  stdev=%.3f" %
              (cls, pc["n_scored"],
               "None" if pc["mean_delta"] is None else "%.3f" % pc["mean_delta"],
               pc["stdev_delta"]))
    print("separation(contextual - canned)             = %s" %
          report["separation_contextual_minus_canned"])
    print("separation(contextual - generic_responsive) = %s" %
          report["separation_contextual_minus_generic_responsive"])
    print("repeat_delta_stdev_mean = %.3f (lower = more consistent)" %
          report["repeat_delta_stdev_mean"])
    print("keyword_echo_check: risk_detected=%s (generic_responsive r=%s)" %
          (report["keyword_echo_check"]["risk_detected"],
           report["keyword_echo_check"]["generic_responsive_pearson_r"]))
    print("report: %s" % (out / "report.json"))


if __name__ == "__main__":
    main()
