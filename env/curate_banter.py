#!/usr/bin/env python3
"""Rolling curation over all scored banter batches.

Aggregates every *.scored.json into (a) a curation master -- top-K
sessions across the whole stream by curation_score, with the batch
config attached so config comparisons are empirical, (b) a per-batch
config table (mean floor pass / repetition / reaction by policy model,
temperature, provocation rate -- the data that picks env defaults), and
(c) a top-5 FULL-TRANSCRIPT text file for the human read, because
"really truly funny" is certified by reading, not by these numbers.

Usage (box, invoked by score_keeper.sh after each scored batch):
  python3 -m env.curate_banter --scored-glob '/data/good-humored/runs/*_stream_*.scored.json' \
      --transcripts-dir /data/good-humored/runs \
      --out-master curation_master.json --out-top curation_top5.txt
"""

import argparse
import glob
import json
from pathlib import Path

TOP_K = 50


def load_batch(scored_path, transcripts_dir):
    """Scored sessions + the generation config from the twin .jsonl."""
    data = json.loads(Path(scored_path).read_text())
    stem = Path(scored_path).name.replace(".scored.json", "")
    config = {}
    jsonl = Path(transcripts_dir) / (stem + ".jsonl")
    if jsonl.exists():
        first = json.loads(jsonl.read_text().split("\n", 1)[0])
        config = first.get("config", {})
    for s in data.get("sessions", []):
        s["batch"] = stem
        s["config"] = {k: config.get(k) for k in
                       ("model", "temperature", "provocation_rate")}
    return data.get("sessions", [])


def full_transcript(batch_stem, session_id, transcripts_dir):
    jsonl = Path(transcripts_dir) / (batch_stem + ".jsonl")
    if not jsonl.exists():
        return None
    for line in jsonl.open():
        r = json.loads(line)
        if r.get("session_id") == session_id and "turns" in r:
            return r
    return None


def render(rec, scored):
    lines = ["=" * 72,
             "batch=%s session=%s  curation=%.3f  task=%s" %
             (scored["batch"], scored["session_id"],
              scored["curation_score"], rec["task"]),
             "config=%s" % json.dumps(scored["config"]), "-" * 72]
    for t in rec["turns"]:
        tag = t["role"].upper()
        if t.get("provocation"):
            tag += "[%s]" % t["provocation"]
        lines.append("%-22s %s" % (tag + ":", t["text"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-glob", required=True)
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--out-master", required=True)
    ap.add_argument("--out-top", required=True)
    args = ap.parse_args()

    sessions, batch_stats = [], {}
    for path in sorted(glob.glob(args.scored_glob)):
        batch = load_batch(path, args.transcripts_dir)
        sessions.extend(batch)
        if batch:
            n = len(batch)
            batch_stats[batch[0]["batch"]] = {
                "config": batch[0]["config"], "n_scored": n,
                "mean_curation": round(
                    sum(s["curation_score"] for s in batch) / n, 3),
                "mean_floor_pass": round(
                    sum(s["floor_pass_rate"] for s in batch) / n, 3),
                "mean_reaction_L": round(
                    sum(s["mean_reaction_L"] for s in batch) / n, 3),
            }
    sessions.sort(key=lambda s: -s["curation_score"])
    top = sessions[:TOP_K]
    Path(args.out_master).write_text(json.dumps(
        {"n_sessions_scored": len(sessions), "n_batches": len(batch_stats),
         "batch_stats": batch_stats,
         "top": [{k: s[k] for k in ("batch", "session_id", "task", "config",
                                    "curation_score", "floor_pass_rate",
                                    "max_self_repetition", "mean_reaction_L")}
                 for s in top]}, indent=2))

    blocks = []
    for s in top[:5]:
        rec = full_transcript(s["batch"], s["session_id"],
                              args.transcripts_dir)
        if rec:
            blocks.append(render(rec, s))
    Path(args.out_top).write_text("\n\n".join(blocks) + "\n")
    print("curated %d sessions from %d batches -> %s (top5 -> %s)"
          % (len(sessions), len(batch_stats), args.out_master, args.out_top))


if __name__ == "__main__":
    main()
