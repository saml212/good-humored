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
import collections
import glob
import json
import re
from pathlib import Path

TOP_K = 50

# cross-session motif stats: the read found phrase attractors recurring
# ACROSS sessions ("bermuda triangle", haunted-office register) that
# per-session self_repetition cannot see -- the 25-template ChatGPT
# failure mode at motif level. Measured here per batch; NOT yet used in
# curation_score (measure first, penalize once quantified).
_STOP = set("the a an and or but of to in on at for with is are was were "
            "be it this that i you we they he she my your our just so".split())


def motif_stats(batch_sessions):
    """Distinct-trigram ratio + top recurring content bigrams, pooled
    over the batch's policy turns."""
    tri = collections.Counter()
    big = collections.Counter()
    for s in batch_sessions:
        for t in s.get("per_turn", []):
            toks = re.findall(r"[a-z']+", t["text"].lower())
            tri.update(zip(toks, toks[1:], toks[2:]))
            big.update(b for b in zip(toks, toks[1:])
                       if b[0] not in _STOP and b[1] not in _STOP)
    n_tri = sum(tri.values())
    top = [(" ".join(b), c) for b, c in big.most_common(5)]
    return {"trigram_diversity": round(len(tri) / n_tri, 3) if n_tri else 0,
            "top_motifs": top}


# policy agreement-opener rate: the RLHF sycophancy attractor showing
# up as conversational risk-aversion (policy yes-ands everything, the
# partner does the comedic lifting). ENV-CHARACTERIZATION metric, not
# a bug to prompt away -- the neutral policy prompt exists precisely
# so the env exposes this attractor for RL training to fix.
_AGREE = re.compile(
    r"^(absolutely|agreed|totally|exactly|perfect|good call|nice|right\??"
    r"|yes|yeah|deal|definitely|for sure|great idea|love it|haha,? (yes|yeah|true))\b",
    re.I)


def agreement_rate(batch_sessions):
    n = h = 0
    for s in batch_sessions:
        for t in s.get("per_turn", []):
            n += 1
            if _AGREE.search(t["text"].strip()):
                h += 1
    return round(h / n, 3) if n else 0


_ASTERISK = re.compile(r"\*[^*]{3,80}\*")


def asterisk_rate(batch_sessions):
    """RP stage-direction register health (policy turns)."""
    n = h = 0
    for s in batch_sessions:
        for t in s.get("per_turn", []):
            n += 1
            if _ASTERISK.search(t["text"]):
                h += 1
    return round(h / n, 4) if n else 0


# CJK leakage: GLM policy emits mixed-language turns, strongly
# temperature-correlated (0.06% at T=0.9 -> 1.13% at T=1.2, measured
# 2026-08-07) -- a hard product defect for an English banter dataset,
# invisible to every other metric and too rare for reads to catch
_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")


def cjk_rate(batch_sessions):
    n = h = 0
    for s in batch_sessions:
        for t in s.get("per_turn", []):
            n += 1
            if _CJK.search(t["text"]):
                h += 1
    return round(h / n, 4) if n else 0


def session_has_cjk(s):
    return any(_CJK.search(t["text"]) for t in s.get("per_turn", []))


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
    # prompt revisions shift the curation scale (v0.3.1 register fix
    # moved it -0.07..-0.13), so the human-read file draws from the
    # most RECENT batches only — last 20 PER POLICY LANE by batch
    # number, not global mtime: the contrast lane scores 3-4x faster,
    # and a global-mtime window silently went 40/40 contrast, which
    # the lane-scoped shortlist then excluded entirely (empty read
    # file). The all-time master keeps everything.
    paths = sorted(glob.glob(args.scored_glob))
    by_prefix = {}
    for p in paths:
        name = Path(p).name
        if name.startswith("contrast"):
            continue
        pref = name.split("_stream")[0]
        by_prefix.setdefault(pref, []).append(p)
    recent_paths = set()
    for pref, plist in by_prefix.items():
        plist.sort(key=lambda p: int(Path(p).name.split("_")[-1].split(".")[0]))
        recent_paths.update(plist[-20:])
    recent = []
    for path in paths:
        batch = load_batch(path, args.transcripts_dir)
        sessions.extend(batch)
        if path in recent_paths:
            recent.extend(batch)
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
                "motifs": motif_stats(batch),
                "policy_agreement_rate": agreement_rate(batch),
                "policy_asterisk_rate": asterisk_rate(batch),
                "policy_cjk_rate": cjk_rate(batch),
            }
    sessions.sort(key=lambda s: -s["curation_score"])
    top = sessions[:TOP_K]
    top_by_task = {}
    for s in sessions:
        top_by_task.setdefault(s["task"], s)  # first hit = best (sorted)
    slim = lambda s: {k: s[k] for k in ("batch", "session_id", "task",
                                        "config", "curation_score",
                                        "floor_pass_rate",
                                        "max_self_repetition",
                                        "mean_reaction_L")}
    Path(args.out_master).write_text(json.dumps(
        {"n_sessions_scored": len(sessions), "n_batches": len(batch_stats),
         "batch_stats": batch_stats,
         "top": [slim(s) for s in top],
         "top_by_task": {t: slim(s) for t, s in top_by_task.items()}},
        indent=2))

    # human-read file: cap 2 per task -- an uncapped top-N collapses to
    # one high-affordance task's trajectory family (observed: 9/10
    # supply-closet), and the read must sample the env's breadth.
    # Contrast-lane sessions are EXCLUDED: with 50x the sample count,
    # tail-luck under the multiplicative gate metric beat typical
    # strong-model sessions (an 8B session topped the table while
    # reading clearly worse) -- the lane is negative training
    # contrast, not demo material, and demo shortlists are lane-scoped
    recent.sort(key=lambda s: -s["curation_score"])
    picked, per_task = [], {}
    for s in recent:
        # contrast = negative training data; glmself = utilization
        # filler + partner-A/B arm whose scores are audience-inflated
        # (GLM judging GLM x GLM -- read-verified: a 1.633 self-play
        # session read a full tier below its 235B-partner score twin).
        # The shortlist is the DEMO channel: product config (strong
        # partner) only.
        if s["batch"].startswith(("contrast", "glmself")):
            continue
        if session_has_cjk(s):  # language defect: never demo material
            continue
        if per_task.get(s["task"], 0) >= 2:
            continue
        picked.append(s)
        per_task[s["task"]] = per_task.get(s["task"], 0) + 1
        if len(picked) == 5:
            break
    blocks = []
    for s in picked:
        rec = full_transcript(s["batch"], s["session_id"],
                              args.transcripts_dir)
        if rec:
            blocks.append(render(rec, s))
    Path(args.out_top).write_text("\n\n".join(blocks) + "\n")
    print("curated %d sessions from %d batches -> %s (top5 -> %s)"
          % (len(sessions), len(batch_stats), args.out_master, args.out_top))


if __name__ == "__main__":
    main()
