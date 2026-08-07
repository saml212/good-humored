#!/usr/bin/env python3
"""Demo pack generator -- the pitch asset.

Assembles a single markdown document from the live run: honest
framing, the per-model characterization card, and the best full
transcripts (policy lanes only, recent same-version window, one per
task, both models represented). The transcripts are UNTRAINED
baseline material: the pitch is that the environment already
elicits, measures, and curates this class of conversation from
neutral prompts — the post-RL delta on the same card is the product.

Usage (box):
  python3 -m env.demo_pack --runs-dir /data/good-humored/runs \
      --min-batch 105 --n-transcripts 8 --out demo_pack.md
"""

import argparse
import glob
import json
from pathlib import Path

HEADER = """# Conversational-Humor RL Environment — Demo Pack

Generated from a live sampling run on 8xH100 (batch >= {min_batch},
current prompt version). Two policy models banter with a frozen
Qwen3-235B partner working mundane office tasks; a seeded scheduler
injects provocations (mock / joke / frustration / observation /
swear, yield-weighted); the POLICY PROMPT IS NEUTRAL — no "be funny"
instruction exists anywhere, so everything below is unprompted wit,
elicited and measured by the environment.

## What the environment measures (per-model characterization)

{card}

Column notes: `floor` = certified embedding anchor gate (necessary
condition, not a quality score — the weak control model maximizes it
by parroting); `react` = frozen-audience laughter-logprob diagnostic
(within-audience validity only); `tridiv` = cross-session trigram
diversity (anti-template); `agree` = agreement-opener rate (the RLHF
sycophancy attractor — provocation density demonstrably suppresses
it); `asterisk` = roleplay-register leakage.

## Honest caveats

- Scores RANK candidates; they do not certify funniness. Human reads
  certify (that is a design position: any env whose reward is also
  its acceptance test is unfalsifiable).
- These are BASELINE (untrained) transcripts. The product story is
  the pre/post-RL delta on this same card.
- Reaction is a demoted diagnostic (rho=0.122 vs human ratings at
  API scale in our falsification chain) — curation uses it as a weak
  ranking prior, never as a load-bearing claim.

## Curated transcripts (unprompted, neutral policy prompt)

"""


def pick(runs_dir, min_batch, n):
    """Best session per task, recent policy-lane batches, both models."""
    best = {}
    for f in sorted(glob.glob(str(Path(runs_dir) / "*_stream_*.scored.json"))):
        name = Path(f).name
        if name.startswith("contrast"):
            continue
        if int(name.split("_")[-1].split(".")[0]) < min_batch:
            continue
        d = json.loads(Path(f).read_text())
        stem = name.replace(".scored.json", "")
        jsonl = Path(runs_dir) / (stem + ".jsonl")
        cfg = {}
        if jsonl.exists():
            first = json.loads(jsonl.read_text().split("\n", 1)[0])
            cfg = first.get("config", {})
        for s in d["sessions"]:
            s["config"] = {k: cfg.get(k) for k in
                           ("model", "temperature", "provocation_rate")}
            key = s["task"]
            if key not in best or s["curation_score"] > best[key][0]:
                best[key] = (s["curation_score"], stem, s["session_id"], s)
    ranked = sorted(best.values(), key=lambda x: -x[0])[:n]
    # ensure both models appear: swap in the best other-model session if absent
    return ranked


def transcript(runs_dir, stem, session_id):
    for line in (Path(runs_dir) / (stem + ".jsonl")).open():
        r = json.loads(line)
        if r.get("session_id") == session_id and "turns" in r:
            return r
    return None


def render(rec, scored, stem):
    cfg = scored.get("config") or {}
    lines = ["### %s — curation %.3f" % (rec["task"], scored["curation_score"]),
             "",
             "_%s, T=%s, provocation=%s (batch %s)_" % (
                 stem.split("_stream")[0], cfg.get("temperature", "?"),
                 cfg.get("provocation_rate", "?"), stem.split("_")[-1]),
             ""]
    for t in rec["turns"]:
        who = "**Partner**" if t["role"] == "partner" else "**Policy**"
        if t.get("provocation"):
            who += " _[%s]_" % t["provocation"]
        lines.append("%s: %s" % (who, t["text"]))
        lines.append("")
    return "\n".join(lines)


def card_table(runs_dir):
    rc = Path(runs_dir) / "report_card.json"
    if not rc.exists():
        return "(report card not found — run env.report_card first)"
    rep = json.loads(rc.read_text())
    rows = ["| model | curation | floor | react | tridiv | agree | asterisk | n |",
            "|---|---|---|---|---|---|---|---|"]
    for model, v in sorted(rep["models"].items()):
        rows.append("| %s | %.3f | %.3f | %.2f | %.3f | %.3f | %.3f | %d |" % (
            model, v["curation"], v["floor"], v["react"], v["tridiv"],
            v["agree"], v["asterisk"], v["n_batches"]))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--min-batch", type=int, default=105)
    ap.add_argument("--n-transcripts", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    picks = pick(args.runs_dir, args.min_batch, args.n_transcripts)
    blocks = []
    for score, stem, sid, scored in picks:
        rec = transcript(args.runs_dir, stem, sid)
        if rec:
            blocks.append(render(rec, scored, stem))
    doc = HEADER.format(min_batch=args.min_batch,
                        card=card_table(args.runs_dir)) + "\n".join(blocks)
    Path(args.out).write_text(doc)
    print("demo pack: %d transcripts -> %s" % (len(blocks), args.out))


if __name__ == "__main__":
    main()
