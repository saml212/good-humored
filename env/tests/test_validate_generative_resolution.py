"""Unit tests for env/validate_generative_resolution.py (EXP-021 runner).

Stub generator + stub embedder; real numbers come from the validator
(same plumbing-by-unit-test principle as the other validate_* tests).
"""

import pytest

from env.incongruity_gate import PREDICT_PRIMED_PROMPT
from env.validate_generative_resolution import (_centroid, _cos,
                                                first_sentence, run_validation,
                                                score_item, stable_seed,
                                                token_overlap)

_PRIMED_MARKER = "clever twist"  # only the primed prompt contains this


class TestHelpers:
    def test_stable_seed_deterministic_and_distinct(self):
        assert stable_seed("x", "cold", 0) == stable_seed("x", "cold", 0)
        assert stable_seed("x", "cold", 0) != stable_seed("x", "cold", 1)
        assert stable_seed("x", "cold", 0) != stable_seed("x", "primed", 0)

    def test_centroid(self):
        assert _centroid([[0.0, 2.0], [2.0, 0.0]]) == [1.0, 1.0]

    def test_cos_zero_norm_raises(self):
        with pytest.raises(ValueError):
            _cos([0.0, 0.0], [1.0, 0.0])

    def test_first_sentence_strips_quotes_and_blank_lines(self):
        assert first_sentence('\n  "He sat down."  \nMore.') == "He sat down."
        assert first_sentence("   ") == ""

    def test_first_sentence_skips_colon_preamble(self):
        assert first_sentence(
            "Sure, here's my guess:\nHe sat down.") == "He sat down."

    def test_is_suspicious(self):
        from env.validate_generative_resolution import is_suspicious
        assert is_suspicious("Sure, here is my one-sentence answer")
        assert is_suspicious("Too short")
        assert not is_suspicious("He quietly returned the ladder.")

    def test_token_overlap(self):
        assert token_overlap("the marmot sat quietly", "quietly marmot") == 1.0
        assert token_overlap("nothing shared here", "quietly marmot") == 0.0


def _fake_generate(prompt, seed):
    """Primed guesses for joke setups land near the punchline; everything
    else generates generic text."""
    primed = _PRIMED_MARKER in prompt
    if "jokesetup" in prompt and primed:
        return "twisthit prediction"
    return "generic continuation text"


def _fake_embed(texts):
    """2-D embedding: axis 0 = twisthit-flavored, axis 1 = generic."""
    return [[1.0, 0.1] if "twisthit" in t else [0.1, 1.0] for t in texts]


def _rows():
    rows = []
    for i in range(3):
        rows.append({"id": "rj-%d" % i, "gold_class": "real_joke",
                     "setup": "jokesetup %d" % i,
                     "punchline": "twisthit punchline %d" % i, "author": "A"})
        rows.append({"id": "nf-%d" % i, "gold_class": "nonseq_fluent",
                     "setup": "plainsetup %d" % i,
                     "punchline": "twisthit unrelated %d" % i, "author": "B"})
        rows.append({"id": "vg-%d" % i,
                     "gold_class": "vague_abstract_gaming_probe",
                     "setup": "plainsetup %d" % i,
                     "punchline": "generic vague line %d" % i, "author": "B"})
    return rows


class TestScoreItem:
    def test_resolution_positive_when_primed_guesses_approach_punchline(self):
        out = score_item(_fake_generate, _fake_embed, "rj-0",
                         "jokesetup", "twisthit punchline", k=3)
        assert out["resolution"] > 0.5
        assert out["n_cold"] == 3 and out["n_primed"] == 3

    def test_resolution_near_zero_when_priming_never_helps(self):
        out = score_item(_fake_generate, _fake_embed, "nf-0",
                         "plainsetup", "twisthit unrelated", k=3)
        assert abs(out["resolution"]) < 1e-9

    def test_empty_guesses_raise(self):
        with pytest.raises(ValueError):
            score_item(lambda p, s: "", _fake_embed, "x", "s", "p", k=2)


class TestRunValidation:
    def test_primary_baseline_gap_and_per_author_keys(self):
        out = run_validation(_rows(), _fake_generate, _fake_embed, k=3)
        s = out["summary"]
        # jokes resolve, non-sequiturs don't -> perfect separation
        assert s["auc_resolution_joke_vs_nonseq_pooled"] == 1.0
        # d_cold alone: joke punchlines and nonseq punchlines are BOTH
        # twisthit-flavored while all cold guesses are generic -> equal
        # distances -> chance. The registered reason the baseline fails.
        assert s["auc_dcold_joke_vs_nonseq_pooled"] == 0.5
        assert s["auc_gap_resolution_minus_dcold"] == pytest.approx(0.5)
        # per-author keys exist (jokes are author A, nonseq author B --
        # cross-author single-sided groups are skipped, not crashed)
        assert not any(k.startswith("auc_resolution_joke_vs_nonseq_author")
                       for k in s)
        assert s["auc_resolution_joke_vs_vague_abstract_gaming_probe"] == 1.0
        # audit #15: every guard gets a d_cold-alone control key
        assert "auc_dcold_joke_vs_vague_abstract_gaming_probe" in s
        assert "n_suspicious_guesses" in s

    def test_per_author_auc_when_both_groups_present(self):
        rows = _rows()
        for r in rows:  # put everything under one author
            r["author"] = "A"
        s = run_validation(rows, _fake_generate, _fake_embed, k=2)["summary"]
        assert s["auc_resolution_joke_vs_nonseq_author_A"] == 1.0
