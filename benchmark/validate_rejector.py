"""Rejector validation — the first experiment in this repo, on purpose.

The rejector is the cascade's measurement instrument. If it labels
inconsistently, or labels jokes rather than topics, every downstream
number measures the instrument instead of the model. So before any
cascade runs, this script checks, against a hand-built fixture with
known topic structure:

  1. repeat consistency  — same joke labeled K times -> same label?
  2. reworded invariance — same joke, new words -> same label?
     (this is the topic-vs-joke discrimination test)
  3. same-topic cohesion — different joke, same topic -> same label?
  4. cross-topic separation — different topics -> different labels?
  5. ARI vs gold partition (unambiguous items only)
  6. a deliberately crude keyword baseline — if the model rejector
     doesn't clearly beat it, use the cheaper thing (pre-experiment
     checklist item 4: try to disprove with a simpler baseline).

Usage:
  python3 -m benchmark.validate_rejector --model haiku --repeats 3 \
      --out experiment-runs/2026-07-16-rejector-validation
"""

import argparse
import json
import string
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .label_space import LabelSpace
from .metrics import adjusted_rand_index, normalize_label
from .providers import make_claude_cli
from .rejector import LABEL_PROMPT_VERSION, label_topic

FIXTURES = Path(__file__).parent / "fixtures" / "rejector_validation.jsonl"

_STOPWORDS = set(
    "i me my we our you your he she his her it its they them their the a an "
    "and or but so to of in on at for with about was were is are be been am "
    "do did does have has had this that these those there here now then out "
    "up down off over under again once only just very too more most other "
    "some such no nor not own same than can will dont cant wont im ive id "
    "said says told tell asked went got get".split()
)


def keyword_baseline(joke: str) -> str:
    """Most frequent non-stopword (ties -> longest). Deliberately crude:
    it is the disproof bar, not a serious labeler."""
    words = [w.strip(string.punctuation).lower() for w in joke.split()]
    words = [w for w in words if w and w not in _STOPWORDS and len(w) > 2]
    if not words:
        return "unknown"
    counts = Counter(words)
    best = sorted(counts, key=lambda w: (-counts[w], -len(w)))[0]
    return normalize_label(best)


def load_fixtures() -> List[Dict]:
    with open(FIXTURES) as f:
        return [json.loads(line) for line in f if line.strip()]


def majority(labels: List[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def has_strict_majority(labels: List[str]) -> bool:
    """False when repeats split with no true winner (1-1-1, 2-2) — a
    fabricated 'majority' would otherwise leak into every downstream
    metric (audit W3)."""
    top = Counter(labels).most_common(1)[0][1]
    return top * 2 > len(labels)


def score(items: List[Dict], labels: Dict[str, List[str]],
          label_space: Optional[LabelSpace] = None) -> Dict:
    """Compute all validation metrics for one labeler's outputs.

    label_space, if given, canonicalizes every label before any
    comparison below (design doc: label_space.py) — synonym scatters
    (fitness/exercise/gym) then count as agreement, not disagreement.
    Deliberately NOT wired in by default here: this script's whole job
    is exposing raw rejector inconsistency (EXP-001/002), and silently
    smoothing that over with semantic merging would hide the exact
    instrument defect it exists to catch. Callers opt in explicitly."""
    if label_space is not None:
        labels = {i: [label_space.canon(l) for l in ls]
                  for i, ls in labels.items()}
    maj = {i["id"]: majority(labels[i["id"]]) for i in items}
    unambig = [i for i in items if i["pair_type"] != "ambiguous"]

    # 1. repeat consistency: mean pairwise agreement across K repeats
    agrees, total = 0, 0
    for i in items:
        ls = labels[i["id"]]
        for x in range(len(ls)):
            for y in range(x + 1, len(ls)):
                agrees += ls[x] == ls[y]
                total += 1
    repeat_consistency = agrees / total if total else 1.0

    # 2-3. pair matches within groups
    def pair_match(pair_type: str) -> float:
        hits, n = 0, 0
        by_group: Dict[str, Dict[str, str]] = {}
        for i in unambig:
            by_group.setdefault(i["group"], {})[i["pair_type"]] = maj[i["id"]]
        for g in by_group.values():
            if "original" in g and pair_type in g:
                hits += g["original"] == g[pair_type]
                n += 1
        return hits / n if n else 0.0

    # 4. cross-topic separation: cross-group pairs must get different
    # labels. NOTE (audit W1): this is a sanity FLOOR, not a
    # discriminator — any labeler emitting idiosyncratic per-joke strings
    # saturates it. It is excluded from pass/fail criteria.
    sep_hits, sep_n = 0, 0
    for a in range(len(unambig)):
        for b in range(a + 1, len(unambig)):
            ia, ib = unambig[a], unambig[b]
            if ia["group"] != ib["group"]:
                sep_hits += maj[ia["id"]] != maj[ib["id"]]
                sep_n += 1

    # 5. ARI of induced partition vs gold groups
    gold = [i["group"] for i in unambig]
    pred = [maj[i["id"]] for i in unambig]

    no_majority_ids = [i["id"] for i in items
                       if not has_strict_majority(labels[i["id"]])]

    return {
        "repeat_consistency": round(repeat_consistency, 4),
        "reworded_invariance": round(pair_match("reworded"), 4),
        "same_topic_cohesion": round(pair_match("same_topic"), 4),
        "cross_topic_separation": round(sep_hits / sep_n, 4) if sep_n else 0.0,
        "ari_vs_gold": round(adjusted_rand_index(gold, pred), 4),
        "no_majority_ids": no_majority_ids,
        "majority_labels": maj,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.repeats < 2:
        ap.error("--repeats must be >= 2: consistency needs repetition "
                 "(audit N4 — repeats=1 fakes a perfect score)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_fixtures()
    complete = make_claude_cli(args.model)

    # Timestamped per invocation — re-runs must never interleave into one
    # indistinguishable file (audit B2).
    run_stamp = time.strftime("%Y%m%dT%H%M%S")
    model_labels: Dict[str, List[str]] = {}
    raw_log = open(out / ("labels_raw.%s.jsonl" % run_stamp), "a")
    t0 = time.time()
    for i in items:
        model_labels[i["id"]] = []
        for k in range(args.repeats):
            lab = label_topic(i["joke"], complete)
            model_labels[i["id"]].append(lab)
            raw_log.write(json.dumps(
                {"id": i["id"], "repeat": k, "label": lab}) + "\n")
            raw_log.flush()
    raw_log.close()

    baseline_labels = {i["id"]: [keyword_baseline(i["joke"])] * args.repeats
                       for i in items}

    report = {
        "experiment": "rejector-validation",
        "run_stamp": run_stamp,
        "label_prompt_version": LABEL_PROMPT_VERSION,
        "model": args.model,
        "repeats": args.repeats,
        "n_items": len(items),
        "wall_seconds": round(time.time() - t0, 1),
        "rejector": score(items, model_labels),
        "keyword_baseline": score(items, baseline_labels),
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    r, b = report["rejector"], report["keyword_baseline"]
    print("=== rejector validation (%s, %d repeats) ===" %
          (args.model, args.repeats))
    for key in ("repeat_consistency", "reworded_invariance",
                "same_topic_cohesion", "cross_topic_separation",
                "ari_vs_gold"):
        print("%-24s rejector=%.3f  baseline=%.3f" % (key, r[key], b[key]))
    print("report: %s" % (out / "report.json"))


if __name__ == "__main__":
    main()
