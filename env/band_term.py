"""Familiar/novel embedding BAND gate -- EXP-024 (Sam's 2026-07-30 spec).

A reply is IN BAND when it is anchored to the conversation (familiar
enough to land) without restating it (novel enough to say something):

    anchor_sim      = max cosine(reply, turn) over context turns
    context_overlap = max wordset_jaccard(reply, turn) over context turns
    band_pass       = anchor_sim >= a_floor  AND  context_overlap < w_ceil

Floor edge rejects off-topic / generic-filler / word-salad replies;
ceiling edge rejects parrots. NECESSARY-CONDITION GATE, never a
funniness rating (the EXP-019/020/021 falsifications are about rating
claims; this term makes no such claim). Intended wiring: multiplicative
gate in the banter-env reward stack, weight decided at env assembly --
this module is inert until then (zero env/ reward-path imports).

Reuses the certified pieces: `wordset_jaccard` from
benchmark.callback_transform (EXP-016b's hyphen-safe content-word
similarity) and the injectable unit-normalized `embed_fn` contract
shared by env/semantic_novelty.py / env/incongruity_gate.py (vectors
pre-normalized; cosine = dot).
"""

from typing import Callable, Dict, List, Sequence

from benchmark.callback_transform import wordset_jaccard

DEFAULT_A_FLOOR = 0.30   # calibrated by EXP-024's dev sweep (author D)
DEFAULT_W_CEIL = 0.50    # -- both values FROZEN by the registered run;
                         # see EXPERIMENT_LOG.md EXP-024 before changing.


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


class BandGate:
    """Callable gate over (context_turns, reply).

    `embed_fn`: batch callable returning UNIT-NORMALIZED vectors (the
    package-standard contract). Injected, never imported here -- tests
    use fakes; production wires the shared MiniLM encoder once.
    """

    def __init__(self, embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]],
                 a_floor: float = DEFAULT_A_FLOOR,
                 w_ceil: float = DEFAULT_W_CEIL):
        if not (0.0 <= a_floor <= 1.0 and 0.0 < w_ceil <= 1.0):
            raise ValueError("BandGate thresholds out of range: "
                             "a_floor=%r w_ceil=%r" % (a_floor, w_ceil))
        self.embed_fn = embed_fn
        self.a_floor = a_floor
        self.w_ceil = w_ceil

    def score(self, context_turns: List[str], reply: str) -> Dict:
        if not context_turns:
            raise ValueError("BandGate needs at least one context turn")
        if not reply or not reply.strip():
            return {"anchor_sim": 0.0, "context_overlap": 0.0,
                    "floor_pass": False, "ceil_pass": True,
                    "band_pass": False}
        vectors = self.embed_fn(list(context_turns) + [reply])
        reply_vec = vectors[-1]
        anchor_sim = max(_dot(reply_vec, v) for v in vectors[:-1])
        context_overlap = max(wordset_jaccard(reply, t)
                              for t in context_turns)
        floor_pass = anchor_sim >= self.a_floor
        ceil_pass = context_overlap < self.w_ceil
        return {"anchor_sim": anchor_sim,
                "context_overlap": context_overlap,
                "floor_pass": floor_pass, "ceil_pass": ceil_pass,
                "band_pass": floor_pass and ceil_pass}

    def __call__(self, context_turns: List[str], reply: str) -> float:
        """Gate value: 1.0 in band, 0.0 out -- multiplicative wiring."""
        return 1.0 if self.score(context_turns, reply)["band_pass"] else 0.0
