"""Cascade pilot orchestrator: M models x N runs, isolated failures.

Each run is individually fenced (audit W5): one API failure loses one
run, not the sweep. Per-turn JSONL from cascade.py means even a lost
run leaves its partial trajectory on disk.

Usage:
  python3 -m benchmark.run_pilot --models haiku,sonnet --runs 3 \
      --depth 15 --rejector haiku --out experiment-runs/<date>-pilot

--models / --rejector take provider specs ("prefix:alias"), not just
claude aliases — "codex:mini", "api:deepseek", or a bare "haiku" (no
prefix defaults to claude, so old invocations are unaffected).
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List

from .cascade import run_cascade
from .label_space import LabelSpace
from .metrics import cross_model_overlap, depth_to_degradation, path_divergence
from .providers import get_provider


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="comma-separated provider specs under test, e.g. "
                         "haiku,codex:mini,api:deepseek")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--rejector", default="haiku",
                    help="provider spec for the rejector (default: claude)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rejector = get_provider(args.rejector)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    paths: Dict[str, List[List[str]]] = {}
    failures: List[Dict] = []
    for model in models:
        complete = get_provider(model)
        paths[model] = []
        for n in range(args.runs):
            run_id = "%s-r%02d" % (model, n)
            t0 = time.time()
            try:
                rec = run_cascade(
                    complete, rejector, args.depth, run_id,
                    log_path=out / ("turns-%s.jsonl" % run_id))
            except Exception as e:  # fence per run, keep sweeping
                failures.append({"run_id": run_id, "error": repr(e)})
                print("FAILED %s: %r" % (run_id, e))
                traceback.print_exc()
                continue
            paths[model].append(rec["path"])
            deg = depth_to_degradation(rec["path"], rec["refusal_turns"])
            print("%s: %d turns in %.0fs, degradation depth=%s" %
                  (run_id, rec["depth_completed"], time.time() - t0,
                   deg["depth"]))

    # PRIMARY metrics on RAW labels — pre-registered in EXP-004, enforced
    # after adversarial audit BLOCKER-1: canon() only ever merges, so
    # canonicalized jaccard/prefix can only move UP. Scoring canon paths
    # as primary structurally inflates the headline cross-model number
    # (with a known ~8% false-merge rate at threshold 0.70). Semantic
    # views ship alongside under *_semantic keys, never as the headline.
    label_space = LabelSpace().fit(
        [t for ps in paths.values() for p in ps for t in p])
    canon_paths = {m: label_space.canonize_paths(ps) for m, ps in paths.items()}

    summary: Dict = {"models": models, "runs": args.runs,
                     "depth": args.depth, "rejector": args.rejector,
                     "label_space_degraded": label_space.degraded,
                     "failures": failures, "per_model": {}}
    for model, ps in paths.items():
        if len(ps) >= 2:
            cps = canon_paths[model]
            summary["per_model"][model] = {
                "divergence": path_divergence(ps),
                "degradation": [depth_to_degradation(p) for p in ps],
                "divergence_semantic": path_divergence(cps),
                "degradation_semantic": [depth_to_degradation(p) for p in cps],
                "paths": ps,
                "paths_semantic": cps,
            }
    survivors = {m: ps for m, ps in paths.items() if ps}
    if len(survivors) >= 2:
        summary["cross_model"] = cross_model_overlap(survivors)
        summary["cross_model_semantic"] = cross_model_overlap(
            {m: canon_paths[m] for m in survivors})

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("summary: %s" % (out / "summary.json"))


if __name__ == "__main__":
    main()
