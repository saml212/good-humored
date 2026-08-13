import json, math, re

_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
_AST = re.compile(r"\*[^*]{3,80}\*")
FLOOR = -18.4

def certified(sess):
    turns = sess["per_turn"]
    n = len(turns)
    if n == 0: return 0.0, {}
    floor_rate = sum(1.0 for t in turns if t["floor_pass"]) / n
    max_sr = max(t["self_repetition"] for t in turns)
    tastes = [max(t["reaction_L"] - FLOOR, 0.0) / -FLOOR for t in turns]
    taste = sum(tastes) / len(tastes)
    screens_ok = all(not (_CJK.search(t["text"]) or _AST.search(t["text"])) for t in turns)
    total = floor_rate * (1.0 - max_sr) * (1.0 + 0.5 * taste) * (1.0 if screens_ok else 0.0)
    return total, {"floor": floor_rate, "sr": max_sr, "taste": taste, "scr": screens_ok}

def tridiv(sessions):
    seen, tot = set(), 0
    for s in sessions:
        for t in s["per_turn"]:
            w = t["text"].lower().split()
            for i in range(len(w) - 2):
                seen.add((w[i], w[i+1], w[i+2])); tot += 1
    return len(seen) / tot if tot else 0.0

arms = {}
for arm in ["base", "rlc"]:
    d = json.load(open(f"/data/good-humored/runs/eval4_{arm}.scored.json"))
    arms[arm] = {s["session_id"]: s for s in d["sessions"]}

common = sorted(set(arms["base"]) & set(arms["rlc"]))
print("paired n =", len(common))

deltas, comp = [], {"base": [], "rlc": []}
for sid in common:
    tb, cb = certified(arms["base"][sid])
    tr, cr = certified(arms["rlc"][sid])
    deltas.append(tr - tb)
    comp["base"].append((tb, cb)); comp["rlc"].append((tr, cr))

n = len(deltas)
mean = sum(deltas) / n
sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
t = mean / (sd / math.sqrt(n))
for arm in ["base", "rlc"]:
    tot = [x[0] for x in comp[arm]]
    fl = sum(x[1]["floor"] for x in comp[arm]) / n
    sr = sum(x[1]["sr"] for x in comp[arm]) / n
    ta = sum(x[1]["taste"] for x in comp[arm]) / n
    scr = sum(1 for x in comp[arm] if not x[1]["scr"])
    print(f"{arm}: certified mean {sum(tot)/n:.4f}  floor {fl:.4f}  max_sr {sr:.4f}  taste {ta:.4f}  screen_fail_sessions {scr}")
print(f"PAIRED DELTA (rlc - base): {mean:+.4f}  sd {sd:.4f}  t = {t:+.2f}")
tb_ = tridiv(list(arms["base"].values())); tr_ = tridiv(list(arms["rlc"].values()))
print(f"trigram_diversity base {tb_:.4f}  rlc {tr_:.4f}  |diff| {abs(tb_-tr_):.4f}")
print("PIN CHECK: success requires delta >= +0.03 AND t >= 2 AND tridiv|diff| <= 0.05 AND screens clean")
