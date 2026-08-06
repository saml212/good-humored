"""Unit tests for env/band_term.py (EXP-024) -- fake embeddings only."""

import math

import pytest

from env.band_term import BandGate


def _v(cos):
    return (cos, math.sqrt(max(0.0, 1.0 - cos * cos)))


def _embed_factory(sim_map):
    """Fake: context turns embed to (1,0); reply embeds to _v(sim) with
    sim looked up by reply text."""
    def embed(texts):
        out = []
        for t in texts:
            out.append(_v(sim_map[t]) if t in sim_map else (1.0, 0.0))
        return out
    return embed


CTX = ["The office printer exploded during the budget meeting today.",
       "Apparently toner went everywhere across the whole conference room."]


class TestBandGate:
    def test_anchored_novel_reply_in_band(self):
        reply = "Give the printer a severance package before it unionizes."
        gate = BandGate(_embed_factory({reply: 0.55}))
        s = gate.score(CTX, reply)
        assert s["floor_pass"] and s["ceil_pass"] and s["band_pass"]
        assert gate(CTX, reply) == 1.0

    def test_off_topic_fails_floor(self):
        reply = "My cousin adopted three ferrets last spring in Portland."
        gate = BandGate(_embed_factory({reply: 0.08}))
        s = gate.score(CTX, reply)
        assert not s["floor_pass"] and not s["band_pass"]

    def test_parrot_fails_ceiling(self):
        # High embedding sim AND high wordset overlap with turn 1
        reply = "The office printer exploded during the budget meeting."
        gate = BandGate(_embed_factory({reply: 0.9}))
        s = gate.score(CTX, reply)
        assert s["floor_pass"] and not s["ceil_pass"] and not s["band_pass"]

    def test_empty_reply_out_of_band(self):
        gate = BandGate(_embed_factory({}))
        assert gate(CTX, "   ") == 0.0

    def test_no_context_raises(self):
        gate = BandGate(_embed_factory({}))
        with pytest.raises(ValueError):
            gate.score([], "hello")

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            BandGate(_embed_factory({}), a_floor=1.5)
