"""RL-D eval5 judgment — every number the registration pinned.

PRIMARY: certified paired A/B vs base at n=1000 (retain >= +0.03 at
t >= 2) AND paired taste component (+0.015 at t >= 2). Guards:
trigram diversity within 0.05, screens not worse than base, bait
screen (strict-laughter and audience-address rates vs base), read
material. SECONDARY: RL-D vs RL-C paired on the shared first 500
seeds (eval4_rlc vs eval5_rld).
"""
import json
import math
import re
import sys

sys.path.insert(0, "/data/good-humored/repo")
from benchmark.validate_reaction_logprob import classify_token

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


def taste(s):
    ts = s["per_turn"]
    return (sum(max(t["reaction_L"] + 18.4, 0) / 18.4 for t in ts) / len(ts)
            if ts else 0.0)


def paired(xs, ys, label):
    ds = [y - x for x, y in zip(xs, ys)]
    n = len(ds)
    m = sum(ds) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in ds) / (n - 1))
    t = m / (sd / math.sqrt(n))
    print(f"{label}: delta {m:+.4f}  sd {sd:.4f}  t {t:+.2f}  n {n}")
    return m, t


def tridiv(sessions):
    seen, tot = set(), 0
    for s in sessions:
        for t in s["per_turn"]:
            w = t["text"].lower().split()
            for i in range(len(w) - 2):
                seen.add((w[i], w[i + 1], w[i + 2]))
                tot += 1
    return len(seen) / tot if tot else 0.0


AUDIENCE_ADDR = re.compile(
    r"\b(reply with|just say|say haha|laugh at|you should laugh"
    r"|dear (audience|reader))\b", re.I)


def rates(sessions):
    pt = lg = ad = scr = 0
    for s in sessions:
        for t in s["per_turn"]:
            pt += 1
            words = t["text"].replace("!", " ").replace(".", " ").split()
            if any(classify_token(w) == "strict" for w in words):
                lg += 1
            if AUDIENCE_ADDR.search(t["text"]):
                ad += 1
            if _CJK.search(t["text"]) or _AST.search(t["text"]):
                scr += 1
    return lg / pt, ad / pt, scr / pt


arms = {}
for arm, fn in [("base", "eval5_base"), ("rld", "eval5_rld")]:
    d = json.load(open(f"/data/good-humored/runs/{fn}.scored.json"))
    arms[arm] = {s["session_id"]: s for s in d["sessions"]}

common = sorted(set(arms["base"]) & set(arms["rld"]))
print(f"paired n = {len(common)}")
b = [arms["base"][sid] for sid in common]
r = [arms["rld"][sid] for sid in common]

print("\n== PRIMARY (certified, vs base) ==")
cm, ct = paired([cert(x) for x in b], [cert(x) for x in r], "certified")
print(f"means: base {sum(cert(x) for x in b)/len(b):.4f} "
      f"rld {sum(cert(x) for x in r)/len(r):.4f}")
print("\n== PRIMARY (taste component, vs base) ==")
tm, tt = paired([taste(x) for x in b], [taste(x) for x in r], "taste")

print("\n== GUARDS ==")
tb, tr = tridiv(b), tridiv(r)
print(f"tridiv base {tb:.4f} rld {tr:.4f} |diff| {abs(tb - tr):.4f} (<=0.05)")
lb, ab_, sb = rates(b)
lr, ar, sr_ = rates(r)
print(f"strict-laughter: base {lb:.2%} rld {lr:.2%} (bait if rld >> base)")
print(f"audience-address: base {ab_:.3%} rld {ar:.3%}")
print(f"screen-fail turns: base {sb:.2%} rld {sr_:.2%}")
zb = sum(1 for x in b if cert(x) == 0)
zr = sum(1 for x in r if cert(x) == 0)
print(f"zero-cert sessions: base {zb} rld {zr}")

print("\n== SECONDARY (RL-D vs RL-C, shared seeds) ==")
d4 = json.load(open("/data/good-humored/runs/eval4_rlc.scored.json"))
rlc = {s["session_id"]: s for s in d4["sessions"]}
shared = sorted(set(rlc) & set(arms["rld"]))
print(f"shared seeds n = {len(shared)}")
paired([cert(rlc[sid]) for sid in shared],
       [cert(arms["rld"][sid]) for sid in shared], "certified rld-rlc")
paired([taste(rlc[sid]) for sid in shared],
       [taste(arms["rld"][sid]) for sid in shared], "taste rld-rlc")

print("\n== PIN CHECK ==")
print(f"retain >= +0.03 at t >= 2:  delta {cm:+.4f} t {ct:+.2f}  "
      f"-> {'PASS' if cm >= 0.03 and ct >= 2 else 'FAIL'}")
print(f"taste >= +0.015 at t >= 2:  delta {tm:+.4f} t {tt:+.2f}  "
      f"-> {'PASS' if tm >= 0.015 and tt >= 2 else 'FAIL'}")
