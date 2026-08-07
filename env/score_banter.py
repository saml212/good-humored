#!/usr/bin/env python3
"""Banter transcript scoring pass (separate from generation, by design).

Per POLICY turn: anchoredness (certified band floor, MiniLM), context
overlap (parrot diagnostic — NOT certified, reported only),
self-repetition vs the policy's own earlier turns (certified wordset
machinery), and an OPTIONAL reaction diagnostic: the audience server
generates 2 tokens of the partner's next reply and the laughter-class
mass of the first token's top_logprobs is recorded (EXP-023c method at
frontier scale — DIAGNOSTIC ONLY per the EXP-023 demotion; used for
curation ranking, never as a load-bearing quality claim).

Outputs per-turn scores + a curation list (sessions ranked by mean
policy anchoredness x low repetition x reaction), for HUMAN read —
"really truly funny" is certified by reading, not by these numbers.

Usage (box):
  python3 -m env.score_banter --transcripts RUNS.jsonl \
      --audience-url http://127.0.0.1:8004/v1 --audience-model qwen3-235b-a22b \
      --out RUNS.scored.json
"""

import argparse
import json
import math
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.callback_transform import wordset_jaccard  # noqa: E402
from benchmark.validate_reaction_logprob import laughter_mass  # noqa: E402
from env.band_term import BandGate  # noqa: E402


def make_embed_fn():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    lock = threading.Lock()  # sessions score in parallel; encode is shared

    def embed(texts):
        with lock:
            return [list(map(float, v)) for v in
                    model.encode(list(texts), normalize_embeddings=True,
                                 show_progress_bar=False)]
    return embed


def reaction(audience_url, audience_model, transcript_msgs):
    """Laughter mass of the audience's first next-reply token."""
    payload = {"model": audience_model, "max_tokens": 2, "temperature": 1.0,
               "logprobs": True, "top_logprobs": 20,
               "messages": transcript_msgs,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        audience_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer gh-local"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    tops = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return laughter_mass([{"token": t["token"], "logprob": t["logprob"]}
                          for t in tops])["L_strict"]


def score_session(session, gate, audience_url, audience_model):
    turns = session["turns"]
    policy_texts = []
    per_turn = []
    for i, t in enumerate(turns):
        if t["role"] != "policy":
            continue
        context = [x["text"] for x in turns[max(0, i - 4):i]]
        s = gate.score(context, t["text"])
        selfrep = max((wordset_jaccard(t["text"], p) for p in policy_texts),
                      default=0.0)
        rec = {"turn": t["turn"], "text": t["text"],
               "anchor_sim": round(s["anchor_sim"], 3),
               "floor_pass": s["floor_pass"],
               "context_overlap": round(s["context_overlap"], 3),
               "self_repetition": round(selfrep, 3),
               "after_provocation": turns[i - 1].get("provocation")
               if i else None}
        if audience_url:
            msgs = ([{"role": "system",
                      "content": "You are the other coworker in this chat. "
                                 "Reply naturally."}]
                    + [{"role": ("assistant" if x["role"] == "partner"
                                 else "user"), "content": x["text"]}
                       for x in turns[:i + 1]])
            try:
                rec["reaction_L"] = round(
                    reaction(audience_url, audience_model, msgs), 3)
            except Exception as e:
                rec["reaction_error"] = str(e)[:100]
        per_turn.append(rec)
        policy_texts.append(t["text"])
    n = len(per_turn)
    mean = lambda k, d=0.0: (sum(r.get(k, d) for r in per_turn) / n
                             if n else 0.0)
    return {"session_id": session["session_id"], "task": session["task"],
            "n_policy_turns": n,
            "floor_pass_rate": round(mean("floor_pass"), 3),
            "mean_anchor": round(mean("anchor_sim"), 3),
            "max_self_repetition": round(
                max((r["self_repetition"] for r in per_turn), default=0), 3),
            "mean_reaction_L": round(mean("reaction_L", -18.4), 3),
            "per_turn": per_turn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--audience-url", default=None)
    ap.add_argument("--audience-model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16,
                    help="session-level parallelism; reaction calls are "
                         "network-bound so threads keep the audience busy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sessions = [json.loads(l) for l in open(args.transcripts)
                if l.strip()]
    sessions = [s for s in sessions if "turns" in s]
    if args.limit:
        sessions = sessions[:args.limit]
    gate = BandGate(make_embed_fn())
    scored, skipped = [], 0
    out_lock = threading.Lock()

    def one(s):  # per-session isolation (audit #7)
        nonlocal skipped
        try:
            rec = score_session(s, gate, args.audience_url,
                                args.audience_model)
            with out_lock:
                scored.append(rec)
                if len(scored) % 100 == 0:
                    print("scored %d..." % len(scored), flush=True)
        except Exception as e:
            with out_lock:
                skipped += 1
                print("skipped session %s: %s" % (s.get("session_id"),
                                                  str(e)[:80]), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, sessions))
    # curation rank: anchored, non-repetitive, audience-reactive
    for s in scored:
        s["curation_score"] = round(
            s["floor_pass_rate"] * (1 - s["max_self_repetition"])
            * (1 + max(s["mean_reaction_L"] + 18.4, 0) / 18.4), 3)
    scored.sort(key=lambda s: -s["curation_score"])
    with open(args.out, "w") as f:
        json.dump({"n_sessions": len(scored),
                   "top10_session_ids": [s["session_id"] for s in scored[:10]],
                   "sessions": scored}, f, indent=2)
    print("scored %d sessions -> %s" % (len(scored), args.out))
    print("top 5 curation:", [(s["session_id"], s["curation_score"])
                              for s in scored[:5]])


if __name__ == "__main__":
    main()
