"""Unit tests for env/validate_pivot_differential.py (EXP-019 runner).

Real-model numbers come from the validator run; these tests cover the
pure logic only -- rank AUC, seam-safe continuation extraction, and the
aggregation/guard plumbing -- with stub tokenizers/scorers, so they run
offline in milliseconds (same principle as the other validate_* tests:
plumbing by unit test, real numbers by validator).
"""

import math

import pytest

from env.validate_pivot_differential import (CUE, extract_continuation_ids,
                                             rank_auc, run_validation,
                                             score_item)


class TestRankAuc:
    def test_perfect_separation(self):
        assert rank_auc([3.0, 4.0], [1.0, 2.0]) == 1.0

    def test_reversed(self):
        assert rank_auc([1.0, 2.0], [3.0, 4.0]) == 0.0

    def test_all_ties(self):
        assert rank_auc([1.0, 1.0], [1.0, 1.0]) == 0.5

    def test_mixed(self):
        # pairs: (2>1)=1, (2<3)=0, (4>1)=1, (4>3)=1 -> 3/4
        assert rank_auc([2.0, 4.0], [1.0, 3.0]) == 0.75

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rank_auc([], [1.0])


def _word_tokenizer(text):
    """Deterministic stub: one token per whitespace-delimited word."""
    return [hash(w) % 100003 for w in text.split()]


class TestExtractContinuationIds:
    def test_clean_seam(self):
        full, start, seam_ok = extract_continuation_ids(
            _word_tokenizer, "the setup here", " the punchline")
        assert seam_ok
        assert start == 3
        assert len(full) == 5

    def test_merged_seam_falls_back_to_divergence(self):
        # Tokenizer whose last context token merges with the
        # continuation: tokenizing ctx alone gives [1, 2]; tokenizing
        # ctx+cont gives [1, 9, 9] (divergence at index 1).
        def merging(text):
            if text == "a b":
                return [1, 2]
            return [1, 9, 9]
        full, start, seam_ok = extract_continuation_ids(merging, "a b", " c")
        assert not seam_ok
        assert start == 1  # conservative: score from first divergence

    def test_empty_continuation_raises(self):
        def degenerate(text):
            return [1, 2]  # same ids regardless of continuation
        with pytest.raises(ValueError):
            extract_continuation_ids(degenerate, "a b", "")

    def test_zero_context_tokens_raises(self):
        # start==0 would index logits[-1] (wraparound) -- must raise.
        with pytest.raises(ValueError):
            extract_continuation_ids(_word_tokenizer, "", " c d")


class FakeScorer:
    """Canned NLLs: jokes resolve under the cue, non-sequiturs don't.

    Cold NLL is high for anything whose punchline isn't 'expected';
    primed NLL drops by `drop` only when the punchline contains its
    class marker 'resolves'.
    """

    def score(self, ctx, cont):
        primed = CUE in ctx
        if "expected" in cont:
            nll = 1.0
        elif "resolves" in cont:
            nll = 6.0 - (3.0 if primed else 0.0)
        else:  # non-sequitur / vague: cue doesn't help
            nll = 6.0 - (0.2 if primed else 0.0)
        return {"mean_nll": nll, "first3_nll": nll, "max_nll": nll,
                "mean_entropy": 2.0 if "expected" not in cont else 0.5,
                "n_tokens": max(len(cont.split()), 1), "seam_ok": True}


def _rows():
    rows = []
    for i in range(3):
        rows.append({"id": "real_joke-%d" % i, "gold_class": "real_joke",
                     "setup": "setup %d" % i, "punchline": "resolves %d" % i})
        rows.append({"id": "nonseq-%d" % i, "gold_class": "setup_nonsequitur",
                     "setup": "setup %d" % i, "punchline": "unrelated %d" % i})
        rows.append({"id": "boring-%d" % i, "gold_class": "boring_expected",
                     "setup": "setup %d" % i, "punchline": "expected %d" % i})
        rows.append({"id": "vague-%d" % i,
                     "gold_class": "vague_abstract_gaming_probe",
                     "setup": "setup %d" % i, "punchline": "nothing %d" % i})
    return rows


class TestScoreItem:
    def test_delta_s_positive_only_when_cue_helps(self):
        joke = score_item(FakeScorer(), "setup", "resolves it")
        nonseq = score_item(FakeScorer(), "setup", "unrelated thing")
        assert joke["delta_s"] == pytest.approx(3.0)
        assert nonseq["delta_s"] == pytest.approx(0.2)


class TestRunValidation:
    def test_aggregation_and_registered_metrics(self):
        out = run_validation(_rows(), FakeScorer())
        s = out["summary"]
        assert s["n_items"] == 12
        assert s["seam_mismatches"] == 0
        # jokes resolve, non-sequiturs don't -> perfect deltaS separation
        assert s["auc_deltaS_joke_vs_nonseq"] == 1.0
        # cold surprisal is IDENTICAL for jokes and non-sequiturs (both
        # spike at 6.0) -> AUC 0.5: the registered reason the baseline
        # should fail where deltaS succeeds.
        assert s["auc_surprisal_joke_vs_nonseq"] == 0.5
        assert s["auc_gap_deltaS_minus_surprisal"] == pytest.approx(0.5)
        # anti-gaming guard: vague probes don't resolve either
        assert s["auc_deltaS_joke_vs_vague"] == 1.0
        assert s["class_means"]["real_joke"]["delta_s"] == pytest.approx(3.0)

    def test_missing_class_omits_auc_keys(self):
        rows = [r for r in _rows() if r["gold_class"] == "real_joke"]
        out = run_validation(rows, FakeScorer())
        assert "auc_deltaS_joke_vs_nonseq" not in out["summary"]

    def test_generic_per_class_auc_keys(self):
        # EXP-020: arbitrary class names get scold + deltaS AUCs vs joke.
        rows = _rows() + [
            {"id": "rare-%d" % i, "gold_class": "nonseq_rareword",
             "setup": "setup %d" % i, "punchline": "unrelated rare %d" % i}
            for i in range(3)]
        out = run_validation(rows, FakeScorer())
        s = out["summary"]
        assert "auc_scold_joke_vs_nonseq_rareword" in s
        assert "auc_deltaS_joke_vs_nonseq_rareword" in s
        assert "auc_scold_joke_vs_boring_expected" in s
        # cold surprisal: jokes 6.0 vs boring 1.0 -> AUC 1.0
        assert s["auc_scold_joke_vs_boring_expected"] == 1.0
