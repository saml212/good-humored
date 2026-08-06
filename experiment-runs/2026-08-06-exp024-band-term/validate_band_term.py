#!/usr/bin/env python3
"""EXP-024 -- validates env/band_term.py against the two blind-author
fixtures: threshold sweep on author D (dev), single certification pass
on author E with dev-chosen thresholds. Real MiniLM embeddings.

Usage:
  python3 -m env.validate_band_term --out experiment-runs/2026-08-06-exp024-band-term
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from env.band_term import BandGate  # noqa: E402

DEV_FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "band_term_authorD.jsonl"
BLIND_FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "band_term_authorE.jsonl"
IN_BAND = "witty_anchored"
FLOOR_GRID = [round(0.10 + 0.05 * i, 2) for i in range(9)]   # 0.10..0.50
CEIL_GRID = [round(0.30 + 0.05 * i, 2) for i in range(9)]    # 0.30..0.70


def load(path: Path) -> List[Dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        for f in ("id", "gold_class", "context", "reply"):
            if f not in r:
                raise ValueError("fixture row missing %r" % f)
    return rows


def make_embed_fn():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts):
        return [list(map(float, v)) for v in
                model.encode(list(texts), normalize_embeddings=True)]
    return embed


def evaluate(rows: List[Dict], gate: BandGate) -> Dict:
    per_class: Dict[str, Dict] = {}
    for r in rows:
        s = gate.score(r["context"], r["reply"])
        c = per_class.setdefault(r["gold_class"],
                                 {"n": 0, "passed": 0, "items": []})
        c["n"] += 1
        c["passed"] += 1 if s["band_pass"] else 0
        c["items"].append({"id": r["id"], **{k: round(v, 3) if
                          isinstance(v, float) else v for k, v in s.items()}})
    pos = per_class.get(IN_BAND, {"n": 0, "passed": 0})
    neg_n = sum(c["n"] for k, c in per_class.items() if k != IN_BAND)
    neg_rejected = sum(c["n"] - c["passed"] for k, c in per_class.items()
                       if k != IN_BAND)
    sens = pos["passed"] / pos["n"] if pos["n"] else 0.0
    spec = neg_rejected / neg_n if neg_n else 0.0
    return {"balanced_accuracy": (sens + spec) / 2,
            "in_band_recall": sens, "violation_rejection": spec,
            "per_class": {k: {"n": v["n"], "passed": v["passed"]}
                          for k, v in per_class.items()},
            "detail": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dev, blind = load(DEV_FIXTURE), load(BLIND_FIXTURE)
    t0 = time.time()
    embed_fn = make_embed_fn()

    # Dev sweep (author D only)
    best = None
    for af in FLOOR_GRID:
        for wc in CEIL_GRID:
            r = evaluate(dev, BandGate(embed_fn, a_floor=af, w_ceil=wc))
            key = (r["balanced_accuracy"], af)  # tie-break: higher floor
            if best is None or key > best[0]:
                best = (key, af, wc, r)
    _, a_floor, w_ceil, dev_result = best

    # Single blind pass with frozen thresholds
    blind_result = evaluate(blind, BandGate(embed_fn, a_floor=a_floor,
                                            w_ceil=w_ceil))
    # Sensitivity: does +/-0.05 on either threshold flip the blind verdict?
    sensitivity = {}
    for da, dw in ((0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)):
        r = evaluate(blind, BandGate(embed_fn, a_floor=a_floor + da,
                                     w_ceil=w_ceil + dw))
        sensitivity["af%+.2f_wc%+.2f" % (da, dw)] = round(
            r["balanced_accuracy"], 3)

    results = {"a_floor": a_floor, "w_ceil": w_ceil,
               "dev": {k: dev_result[k] for k in
                       ("balanced_accuracy", "in_band_recall",
                        "violation_rejection", "per_class")},
               "blind": {k: blind_result[k] for k in
                         ("balanced_accuracy", "in_band_recall",
                          "violation_rejection", "per_class")},
               "blind_detail": blind_result["detail"],
               "blind_threshold_sensitivity": sensitivity,
               "wall_seconds": round(time.time() - t0, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    shutil.copy2(__file__, args.out / Path(__file__).name)

    print("dev-chosen thresholds: a_floor=%.2f w_ceil=%.2f" % (a_floor, w_ceil))
    for split in ("dev", "blind"):
        r = results[split]
        print("%-5s balanced_acc=%.3f  recall=%.2f  rejection=%.2f  %s"
              % (split, r["balanced_accuracy"], r["in_band_recall"],
                 r["violation_rejection"],
                 {k: "%d/%d" % (v["passed"], v["n"])
                  for k, v in r["per_class"].items()}))
    print("blind sensitivity:", results["blind_threshold_sensitivity"])
    print("results -> %s" % (args.out / "results.json"))


if __name__ == "__main__":
    main()
