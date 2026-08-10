#!/usr/bin/env python3
"""EMULATOR-TRANSFER: score banter policy turns with the NYCC-trained
emulator and correlate with reaction_L within provocation strata.

Pins live in EXPERIMENT_LOG (registered before this file was written).

Usage (box, GPU 0):
  CUDA_VISIBLE_DEVICES=0 python3 -m rm.eval_emulator_transfer \
      --checkpoint /data/good-humored/runs/emulator_v1/best \
      --runs-dir /data/good-humored/runs --min-batch 300 \
      --max-turns 20000 --out /data/good-humored/runs/emulator_transfer.json
"""

import argparse
import collections
import glob
import json
from pathlib import Path

import torch

from rm.train_emulator import spearman  # same tie-corrected implementation


def collect_turns(runs_dir, min_batch, max_turns):
    """(context=preceding partner turn, text=policy turn, reaction_L,
    stratum=after_provocation) from recent policy-lane batches."""
    rows = []
    for f in sorted(glob.glob(str(Path(runs_dir) / "*_stream_*.scored.json")),
                    reverse=True):
        name = Path(f).name
        if name.startswith(("contrast", "glmself")):
            continue
        if int(name.split("_")[-1].split(".")[0]) < min_batch:
            continue
        stem = name.replace(".scored.json", "")
        transcripts = {}
        for line in (Path(runs_dir) / (stem + ".jsonl")).open():
            r = json.loads(line)
            if "turns" in r:
                transcripts[r["session_id"]] = r["turns"]
        d = json.loads(Path(f).read_text())
        for s in d["sessions"]:
            turns = transcripts.get(s["session_id"])
            if not turns:
                continue
            # policy turns sit at odd indices (partner first each round)
            by_pos = {}
            for i, t in enumerate(turns):
                if t["role"] == "policy":
                    by_pos[(t["turn"], t["text"])] = (
                        turns[i - 1]["text"] if i else "")
            for pt in s["per_turn"]:
                if "reaction_L" not in pt:
                    continue
                ctx = by_pos.get((pt["turn"], pt["text"]))
                if ctx is None:
                    continue
                rows.append({"context": ctx[:400], "text": pt["text"],
                             "reaction_L": pt["reaction_L"],
                             "stratum": pt.get("after_provocation") or "none"})
                if len(rows) >= max_turns:
                    return rows
    return rows


@torch.no_grad()
def emulator_scores(checkpoint, rows, batch_size=256):
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint).to("cuda").eval()
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = tok([r["context"] for r in chunk],
                  [r["text"] for r in chunk], truncation=True,
                  max_length=128, padding=True, return_tensors="pt")
        enc = {k: v.to("cuda") for k, v in enc.items()}
        out.extend(model(**enc).logits.squeeze(-1).float().cpu().tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--min-batch", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = collect_turns(args.runs_dir, args.min_batch, args.max_turns)
    print("collected %d policy turns" % len(rows))
    scores = emulator_scores(args.checkpoint, rows)
    for r, sc in zip(rows, scores):
        r["emulator"] = sc

    strata = collections.defaultdict(list)
    for r in rows:
        strata[r["stratum"]].append(r)
    result = {"n_turns": len(rows), "strata": {}}
    rhos = []
    for k, v in sorted(strata.items()):
        rho = spearman([r["emulator"] for r in v],
                       [r["reaction_L"] for r in v])
        result["strata"][k] = {"n": len(v), "rho": round(rho, 4)}
        rhos.append(rho)
        print("stratum %-12s n=%-6d rho=%.4f" % (k, len(v), rho))
    result["stratified_mean_rho"] = round(sum(rhos) / len(rhos), 4)
    result["pooled_rho"] = round(spearman(
        [r["emulator"] for r in rows], [r["reaction_L"] for r in rows]), 4)
    print("STRATIFIED MEAN rho=%.4f  pooled=%.4f"
          % (result["stratified_mean_rho"], result["pooled_rho"]))

    # top/bottom-10 unprovoked turns for the human read (pin: the read
    # must agree directionally or wiring waits regardless of rho)
    none_rows = sorted(strata.get("none", []), key=lambda r: -r["emulator"])
    result["read_top10"] = [{"emulator": round(r["emulator"], 3),
                             "text": r["text"]} for r in none_rows[:10]]
    result["read_bottom10"] = [{"emulator": round(r["emulator"], 3),
                                "text": r["text"]} for r in none_rows[-10:]]
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("->", args.out)


if __name__ == "__main__":
    main()
