#!/usr/bin/env python3
"""The GRPO reward stack — every component evidence-placed.

reward(session) = floor_rate            # certified band gate (hard)
                x (1 - max_self_rep)    # certified anti-parrot
                x (1 + w_taste * taste) # reaction_L taste term
                x screens               # CJK / asterisk hard zeros

Evidence chain (EXPERIMENT_LOG): band floor certified blind
cross-author 0.90; self-rep machinery certified; reaction_L
human-adjudicated 32/50 vs the emulator's 18/50 on blind
disagreement pairs (2026-08-09) and wired at MODEST weight
(w_taste=0.5) partly because that pass sat exactly on the bar; the
NYCC emulator anti-correlates in-context (stratified −0.09) and
serves as the DECOUPLING MONITOR, never a reward. Screens are hard
zeros: a CJK-leaked or asterisk-RP turn is a product defect
regardless of wit.

Pure functions over a session dict (the banter_rollout record
shape); network calls (audience reaction) are injected, so the stack
unit-tests offline and wires into verl's reward manager unchanged.
"""

from benchmark.callback_transform import wordset_jaccard
from env.band_term import BandGate

import re

W_TASTE = 0.5
REACTION_FLOOR = -18.4          # laughter_mass floor (EXP-023c)
_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
_ASTERISK = re.compile(r"\*[^*]{3,80}\*")


def policy_turns_with_context(session, window=4):
    """Yield (context_texts, policy_text, prior_policy_texts)."""
    turns = session["turns"]
    prior = []
    for i, t in enumerate(turns):
        if t["role"] != "policy":
            continue
        ctx = [x["text"] for x in turns[max(0, i - window):i]]
        yield ctx, t["text"], list(prior)
        prior.append(t["text"])


def screens_pass(text):
    """Hard product-defect screens; a zero here zeroes the turn."""
    return not (_CJK.search(text) or _ASTERISK.search(text))


def session_reward(session, gate, reaction_fn=None):
    """Compute the stacked reward for one rollout session.

    gate: BandGate (embed fn injected by caller).
    reaction_fn: optional callable(transcript_msgs) -> laughter_mass L
      (network-injected; None disables the taste term for offline
      tests — the multiplicative structure keeps the rest intact).
    Returns dict with total and components (components logged for the
    decoupling monitor — never collapse them before logging).
    """
    floors, selfreps, tastes, screens = [], [], [], []
    turns = session["turns"]
    for i, t in enumerate(turns):
        if t["role"] != "policy":
            continue
        ctx = [x["text"] for x in turns[max(0, i - 4):i]]
        s = gate.score(ctx, t["text"])
        floors.append(1.0 if s["floor_pass"] else 0.0)
        prior = [x["text"] for x in turns[:i] if x["role"] == "policy"]
        selfreps.append(max((wordset_jaccard(t["text"], p) for p in prior),
                            default=0.0))
        screens.append(1.0 if screens_pass(t["text"]) else 0.0)
        if reaction_fn is not None:
            msgs = [{"role": ("assistant" if x["role"] == "partner"
                              else "user"), "content": x["text"]}
                    for x in turns[:i + 1]]
            L = reaction_fn(msgs)
            tastes.append(max(L - REACTION_FLOOR, 0.0) / -REACTION_FLOOR)
    n = len(floors)
    if n == 0:
        return {"total": 0.0, "n_policy_turns": 0}
    floor_rate = sum(floors) / n
    max_self_rep = max(selfreps)
    screen_rate = sum(screens) / n
    taste = sum(tastes) / len(tastes) if tastes else 0.0
    total = (floor_rate
             * (1.0 - max_self_rep)
             * (1.0 + W_TASTE * taste)
             * (1.0 if screen_rate == 1.0 else 0.0))
    return {"total": round(total, 5), "n_policy_turns": n,
            "floor_rate": round(floor_rate, 4),
            "max_self_rep": round(max_self_rep, 4),
            "taste": round(taste, 4),
            "screen_rate": round(screen_rate, 4)}


# ---------------------------------------------------------------------
# SMOOTH TRAINING OBJECTIVE (validated offline 2026-08-12): additive,
# dense, bounded -- no products, cliffs, or max-terms. Same-context
# relative sd 0.10 vs the certified objective's 0.52 (5x smoother);
# rank agreement with certified rho=0.769 on 500 eval sessions. The
# certified stack above remains the EVAL/curation instrument; this is
# the TRAINING signal only. Train smooth, judge certified.
import math as _math

W_SM_FLOOR, W_SM_SELFREP, W_SM_TASTE, W_SM_SCREEN = 0.4, 0.2, 0.3, 0.5
_SM_MARGIN = 0.06  # sigmoid width around the certified 0.30 threshold


def _soft_floor(sim):
    return 1.0 / (1.0 + _math.exp(-(sim - 0.30) / _SM_MARGIN))


def smooth_session_reward(session, gate, reaction_fn=None):
    """Per-turn additive reward; session value = mean over policy turns.

    Same injected dependencies as session_reward; components returned
    for the decoupling monitors (never collapse before logging).
    """
    turns = session["turns"]
    per_turn = []
    comp = {"floor": 0.0, "selfrep": 0.0, "taste": 0.0, "screens": 0.0}
    for i, t in enumerate(turns):
        if t["role"] != "policy":
            continue
        ctx = [x["text"] for x in turns[max(0, i - 4):i]]
        sim = gate.score(ctx, t["text"])["anchor_sim"]
        prior = [x["text"] for x in turns[:i] if x["role"] == "policy"]
        sr = max((wordset_jaccard(t["text"], p) for p in prior), default=0.0)
        scr = 0.0 if screens_pass(t["text"]) else 1.0
        taste = 0.0
        if reaction_fn is not None:
            msgs = [{"role": ("assistant" if x["role"] == "partner"
                              else "user"), "content": x["text"]}
                    for x in turns[:i + 1]]
            L = reaction_fn(msgs)
            taste = max(L - REACTION_FLOOR, 0.0) / -REACTION_FLOOR
        sf = _soft_floor(sim)
        r = (W_SM_FLOOR * sf + W_SM_SELFREP * (1.0 - sr)
             + W_SM_TASTE * taste - W_SM_SCREEN * scr)
        per_turn.append(r)
        comp["floor"] += sf
        comp["selfrep"] += sr
        comp["taste"] += taste
        comp["screens"] += scr
    n = len(per_turn)
    if n == 0:
        return {"total": 0.0, "n_policy_turns": 0}
    return {"total": round(sum(per_turn) / n, 5), "n_policy_turns": n,
            "floor_soft_mean": round(comp["floor"] / n, 4),
            "mean_self_rep": round(comp["selfrep"] / n, 4),
            "taste": round(comp["taste"] / n, 4),
            "screen_turns": int(comp["screens"])}
