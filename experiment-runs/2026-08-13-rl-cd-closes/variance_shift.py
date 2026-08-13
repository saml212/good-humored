"""RL-D pre-registration measurement: within-group smooth-reward
variance decomposition, early vs late RL-C training.

Groups = 8 GRPO rollouts sharing a session_id. Early = records in the
first 10% of each dump file, late = last 10%. Writes subset jsonl
files for env.score_banter, or (with --decompose) reads the scored
outputs and prints the per-component variance shares.
"""
import glob
import json
import sys
from collections import defaultdict


def build():
    early, late = defaultdict(list), defaultdict(list)
    for path in glob.glob("/data/good-humored/runs/rl_c_sessions/sessions.*.jsonl"):
        recs = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "session_id" in r and "turns" in r:
                    recs.append(r)
        n = len(recs)
        if n < 20:
            continue
        for i, r in enumerate(recs):
            frac = i / n
            if frac < 0.10:
                early[r["session_id"]].append(r)
            elif frac > 0.90:
                late[r["session_id"]].append(r)

    for name, pool in [("early", early), ("late", late)]:
        groups = {sid: rs for sid, rs in pool.items() if len(rs) >= 6}
        keep = sorted(groups)[:50]
        out = f"/data/good-humored/runs/rlc_groups_{name}.jsonl"
        with open(out, "w") as f:
            for sid in keep:
                for k, r in enumerate(groups[sid][:8]):
                    # score_banter keys sessions by session_id; group members
                    # share one, so disambiguate but keep the group recoverable
                    r2 = dict(r)
                    r2["group_id"] = sid
                    r2["session_id"] = sid * 100 + k
                    f.write(json.dumps(r2) + "\n")
        print(f"{name}: {len(keep)} groups -> {out}")


def decompose():
    import math
    W_F, W_S, W_T, W_SCR = 0.4, 0.2, 0.3, 0.5

    def soft_floor(sim):
        return 1.0 / (1.0 + math.exp(-(sim - 0.30) / 0.06))

    for name in ["early", "late"]:
        d = json.load(open(f"/data/good-humored/runs/rlc_groups_{name}.scored.json"))
        groups = defaultdict(list)
        for s in d["sessions"]:
            groups[s["session_id"] // 100].append(s)
        comp_vars = defaultdict(list)
        for sid, members in groups.items():
            if len(members) < 6:
                continue
            per = {"floor": [], "selfrep": [], "taste": [], "screen": [], "total": []}
            for s in members:
                ts = s["per_turn"]
                if not ts:
                    continue
                f_ = W_F * sum(soft_floor(t["anchor_sim"]) for t in ts) / len(ts)
                sr = W_S * (1.0 - sum(t["self_repetition"] for t in ts) / len(ts))
                ta = W_T * sum(max(t["reaction_L"] + 18.4, 0) / 18.4 for t in ts) / len(ts)
                import re
                cjk = re.compile(r"[一-鿿぀-ヿ가-힯]")
                ast = re.compile(r"\*[^*]{3,80}\*")
                scr = -W_SCR * sum(
                    1.0 for t in ts if cjk.search(t["text"]) or ast.search(t["text"])
                ) / len(ts)
                per["floor"].append(f_)
                per["selfrep"].append(sr)
                per["taste"].append(ta)
                per["screen"].append(scr)
                per["total"].append(f_ + sr + ta + scr)
            for k, vals in per.items():
                if len(vals) >= 6:
                    m = sum(vals) / len(vals)
                    comp_vars[k].append(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
        print(f"\n=== {name} (n_groups={len(comp_vars['total'])}) "
              "mean within-group variance by weighted component ===")
        tot = sum(sum(v) / len(v) for k, v in comp_vars.items() if k != "total")
        for k in ["floor", "selfrep", "taste", "screen", "total"]:
            v = sum(comp_vars[k]) / len(comp_vars[k])
            share = f"  share {v / tot:.0%}" if k != "total" else ""
            print(f"  {k:8s} var {v:.5f}{share}")


if __name__ == "__main__":
    if "--decompose" in sys.argv:
        decompose()
    else:
        build()
