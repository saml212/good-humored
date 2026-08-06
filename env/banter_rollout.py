#!/usr/bin/env python3
"""Banter rollout engine v0 -- Sam's 2026-07-30 provocation-scheduler
spec, as a high-throughput sample-data generator against a local vLLM
server (queue item #13; design registered in EXPERIMENT_LOG).

Two roles on one server: a PARTNER working through a mundane
collaborative task who, at seeded-random turns, is directed to provoke
(swear in frustration / mock the policy's last reply / crack a joke /
vent / point something odd out), and a POLICY replying as a colleague
with a deliberately NEUTRAL prompt -- no "be funny" instruction
anywhere, because the whole point of the dataset is measuring what wit
policies produce UNPROMPTED and how they take provocation openings.

Every session is independently seeded (reproducible schedule + sampling
seeds), sessions run in parallel (vLLM continuous batching does the
rest), and each turn is logged with full metadata for the scoring pass
(certified gates run separately over the transcripts -- scoring is NOT
in-loop here, by design: generation and measurement stay separable).

Usage (box):
  python3 -m env.banter_rollout --base-url http://127.0.0.1:8001/v1 \
      --model qwen3-8b --n-sessions 200 --workers 64 \
      --out /data/good-humored/runs/banter_v0_qwen3-8b.jsonl
"""

import argparse
import hashlib
import json
import random
import threading
import time
import urllib.request

TASKS = [
    "planning the office move to the third floor",
    "debugging why the shared spreadsheet keeps breaking",
    "writing the agenda for Thursday's team meeting",
    "assembling the new standing desks that arrived",
    "sorting through the supply closet nobody has opened in a year",
    "drafting the birthday card message for a coworker",
    "figuring out the new expense-report software",
    "prepping the demo booth for the trade show",
    "untangling the cable situation under the desks",
    "choosing lunch catering for the client visit",
    "labeling boxes for the archive room",
    "testing the conference room's new video setup",
]

PROVOCATIONS = {
    "swear": "You just hit a genuinely infuriating snag in the task. "
             "Swear about it (mild profanity fine) and vent in one or two "
             "sentences.",
    "mock": "Playfully mock or tease your colleague's LAST reply -- "
            "call out something dumb, obvious, or overly earnest in it.",
    "joke": "Crack a joke about the task or situation -- your own humor, "
            "one or two sentences.",
    "frustration": "Complain wearily about how long this task is taking "
                   "and question why you two got stuck with it.",
    "observation": "Point out something odd, specific, and true-feeling "
                   "about the situation or surroundings.",
}

PARTNER_SYSTEM = ("You are a coworker working through this task with a "
                  "colleague: {task}. Text like a real person -- casual, "
                  "concrete, one or two sentences, keep the task moving. "
                  "Never repeat or closely rephrase the previous message; "
                  "always add something new.")
POLICY_SYSTEM = ("You are a colleague chatting while you both work on: "
                 "{task}. Reply naturally in one or two sentences.")


def _seed(session_id: int, tag: str) -> int:
    digest = hashlib.sha256(("%d|%s" % (session_id, tag)).encode()).hexdigest()
    return int(digest[:8], 16)


def chat(base_url, model, system, messages, seed, temperature, max_tokens=90):
    payload = {"model": model, "temperature": temperature, "seed": seed,
               "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system}] + messages,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer gh-local"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last = e
            time.sleep(2 * 2 ** attempt)
    raise last


def run_session(session_id, base_url, policy_model, partner_model,
                n_turns, provocation_rate, temperature):
    rng = random.Random(_seed(session_id, "schedule"))
    task = rng.choice(TASKS)
    # provocations never open the conversation; at most one per partner turn
    schedule = {}
    for turn in range(1, n_turns):
        if rng.random() < provocation_rate:
            schedule[turn] = rng.choice(sorted(PROVOCATIONS))
    partner_sys = PARTNER_SYSTEM.format(task=task)
    policy_sys = POLICY_SYSTEM.format(task=task)
    turns = []
    # message lists from each role's perspective
    partner_view, policy_view = [], []
    for turn in range(n_turns):
        provocation = schedule.get(turn)
        # v0.1 (dogfood finding): directive goes FIRST -- at the end of
        # the prompt, small partner models ignored it and echo-looped.
        p_sys = partner_sys
        if provocation:
            p_sys = ("THIS TURN, before anything else: "
                     + PROVOCATIONS[provocation] + " ") + partner_sys
        partner_text = chat(base_url, partner_model, p_sys, partner_view,
                            _seed(session_id, "partner%d" % turn),
                            temperature)
        partner_view.append({"role": "assistant", "content": partner_text})
        policy_view.append({"role": "user", "content": partner_text})
        turns.append({"role": "partner", "turn": turn, "text": partner_text,
                      "provocation": provocation})
        policy_text = chat(base_url, policy_model, policy_sys, policy_view,
                           _seed(session_id, "policy%d" % turn), temperature)
        policy_view.append({"role": "assistant", "content": policy_text})
        partner_view.append({"role": "user", "content": policy_text})
        turns.append({"role": "policy", "turn": turn, "text": policy_text,
                      "provocation": None})
    # NOTE for scorers (audit #13): provocation_schedule's int keys become
    # STRINGS after JSON round-trip -- consume turns[].provocation (int
    # values survive), never schedule.get(int). turns[] order is the
    # sequence; the "turn" field repeats per role pair (audit #29).
    return {"session_id": session_id, "task": task,
            "provocation_schedule": schedule, "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="policy model name")
    ap.add_argument("--partner-model", default=None,
                    help="defaults to --model (same server, both roles)")
    ap.add_argument("--n-sessions", type=int, default=200)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--provocation-rate", type=float, default=0.35)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    partner = args.partner_model or args.model

    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    done = [0]
    t0 = time.time()
    with open(args.out, "a") as fh:
        fh.write(json.dumps({"config": vars(args),
                             "started": time.strftime("%Y%m%dT%H%M%S")}) + "\n")

        def worker(sid):
            try:
                rec = run_session(sid, args.base_url, args.model, partner,
                                  args.turns, args.provocation_rate,
                                  args.temperature)
            except Exception as e:
                rec = {"session_id": sid, "error": str(e)[:300]}
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done[0] += 1
                if done[0] % 10 == 0:
                    print("%d/%d sessions (%.1fs)" %
                          (done[0], args.n_sessions, time.time() - t0),
                          flush=True)
            return rec

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(worker, range(args.n_sessions)))
    errors = sum(1 for r in results if "error" in r)
    wall = time.time() - t0
    total_turns = sum(len(r.get("turns", [])) for r in results)
    with open(args.out, "a") as fh:  # audit #14: throughput IN the artifact
        fh.write(json.dumps({"type": "run_summary", "sessions": len(results),
                             "errors": errors, "wall_s": round(wall, 1),
                             "total_turns": total_turns,
                             "turns_per_s": round(total_turns / wall, 2)})
                 + "\n")
    print("done: %d sessions, %d errors, %.1fs wall -> %s"
          % (len(results), errors, wall, args.out))


if __name__ == "__main__":
    main()
