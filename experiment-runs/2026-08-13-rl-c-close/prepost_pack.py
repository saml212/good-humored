"""Pre/post demo pack from the eval4 matched-seed A/B.

Selection is stated in the output: top-4 certified-delta pairs where
the trained arm is screen-clean, 2 median-delta pairs for
calibration, 1 random pair. Same seed = same task, same provocation
schedule, same partner model — the reply difference is the product.
"""
import json
import random
import re

_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
_AST = re.compile(r"\*[^*]{3,80}\*")


def cert(s):
    ts = s["per_turn"]
    n = len(ts)
    if not n:
        return 0.0
    fr = sum(1.0 for t in ts if t["floor_pass"]) / n
    sr = max(t["self_repetition"] for t in ts)
    ta = sum(max(t["reaction_L"] + 18.4, 0) / 18.4 for t in ts) / n
    ok = all(not (_CJK.search(t["text"]) or _AST.search(t["text"])) for t in ts)
    return fr * (1 - sr) * (1 + 0.5 * ta) * (1.0 if ok else 0.0)


scored, raw = {}, {}
for arm, fname in [("base", "eval4_base"), ("rlc", "eval4_rlc")]:
    d = json.load(open(f"/data/good-humored/runs/{fname}.scored.json"))
    scored[arm] = {s["session_id"]: s for s in d["sessions"]}
    raw[arm] = {}
    with open(f"/data/good-humored/runs/{fname}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if "session_id" in r and "turns" in r:
                raw[arm][r["session_id"]] = r

common = sorted(set(scored["base"]) & set(scored["rlc"]))
deltas = {sid: cert(scored["rlc"][sid]) - cert(scored["base"][sid]) for sid in common}
clean_top = [sid for sid in sorted(common, key=lambda x: -deltas[x])
             if cert(scored["rlc"][sid]) > 0][:4]
mid = sorted(common, key=lambda x: deltas[x])[len(common) // 2 - 1:len(common) // 2 + 1]
random.seed(4)
rand = [random.choice(common)]

out = []
out.append("# Pre/post demo pack — RL-C vs base (matched seeds)")
out.append("")
out.append("Eval4 A/B, 2026-08-13: 500 paired sessions, seed base 10M,")
out.append("identical 235B partner + provocation schedule per pair. Certified")
out.append("objective: **base 0.637 -> trained 0.696, delta +0.059, t=2.87**")
out.append("(pre-registered bar +0.03 at t>=2; measured noise floor +-0.01).")
out.append("Honest decomposition: ~2/3 fewer product-defect zeros (screen")
out.append("fails 106->81 sessions), ~1/3 topical grounding (floor 0.862->")
out.append("0.884). Audience laughter UNCHANGED (reaction_L -14.27 -> -14.25):")
out.append("cleaner and more grounded, not yet funnier. Selection below is")
out.append("stated per section; judge with the median/random pairs, not just")
out.append("the showcase.")
out.append("")


def render(sid, label):
    out.append(f"## {label} — session {sid}")
    t = raw["base"][sid]
    out.append(f"task: {t.get('task', '?')}  |  certified: base "
               f"{cert(scored['base'][sid]):.3f} vs trained {cert(scored['rlc'][sid]):.3f}")
    out.append("")
    for arm, tag in [("base", "BASE"), ("rlc", "TRAINED")]:
        out.append(f"**{tag}**")
        for turn in raw[arm][sid]["turns"][:8]:
            who = "P" if turn["role"] == "partner" else ">"
            prov = f" [prov:{turn['provocation']}]" if turn.get("provocation") else ""
            out.append(f"- {who}{prov} {turn['text']}")
        out.append("")


for sid in clean_top:
    render(sid, "SHOWCASE (top certified delta, trained screen-clean)")
for sid in mid:
    render(sid, "CALIBRATION (median delta)")
for sid in rand:
    render(sid, "CALIBRATION (random pair, seed 4)")

open("/data/good-humored/runs/demo_pack_prepost.md", "w").write("\n".join(out))
print("WROTE /data/good-humored/runs/demo_pack_prepost.md", len(out), "lines")
