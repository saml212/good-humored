"""Unit tests for benchmark/validate_reaction_logprob.py (EXP-023c) --
classification + mass math only; no API calls."""

import math

from benchmark.validate_reaction_logprob import (classify_token,
                                                 laughter_mass,
                                                 normalize_token)


class TestClassify:
    def test_strict_forms(self):
        for t in ("Haha", ' "H', "*laughs", "lol", "LMAO", "😂", "ha",
                  "hehe", '"Haha'):
            # '"H' normalizes to 'h' -- NOT strict; the others are
            if t == ' "H':
                assert classify_token(t) is None
            else:
                assert classify_token(t) == "strict", t

    def test_loose_stubs(self):
        assert classify_token("*l") == "loose"
        assert classify_token("l") == "loose"

    def test_non_laughter(self):
        for t in ("That", "I", "Okay", "hat", "hardly", "*smiles", "Nice"):
            assert classify_token(t) is None, t

    def test_normalize(self):
        assert normalize_token('*"Haha') == "haha"
        assert normalize_token("  lol") == "lol"


class TestMass:
    def test_strict_and_loose_bounds(self):
        tops = [{"token": "Haha", "logprob": math.log(0.3)},
                {"token": "*l", "logprob": math.log(0.1)},
                {"token": "That", "logprob": math.log(0.5)}]
        m = laughter_mass(tops)
        assert math.isclose(math.exp(m["L_strict"]), 0.3, rel_tol=1e-6)
        assert math.isclose(math.exp(m["L_loose"]), 0.4, rel_tol=1e-6)

    def test_zero_mass_floor(self):
        m = laughter_mass([{"token": "That", "logprob": math.log(0.9)}])
        assert m["L_strict"] == math.log(1e-8)
