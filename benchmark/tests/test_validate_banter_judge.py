"""Unit tests for validate_banter_judge.py — Track 2's instrument
validation harness (docs/BENCHMARK.md Section 1b). Fake judge callables
only, no network or CLI calls. Run:
  python3 -m unittest discover benchmark/tests -v

Three fake judges probe the scoring pipeline itself (not any real
model): a PERFECT judge (contextual scores high only under its own true
context; canned is context-blind; generic_responsive shows a small,
overlap-independent bump) must show clean class separation and no
echo-risk reading. A CONSTANT judge (same score regardless of context)
must show every delta at exactly 0. An ECHO judge (scores by literal
keyword overlap between reply and context, blind to whether that overlap
reflects genuine responsiveness) must trip keyword_echo_check — this is
the mechanical demonstration of BENCHMARK.md Section 1b's stated
residual risk.
"""

import re
import unittest

from benchmark.validate_banter_judge import (ECHO_RISK_THRESHOLD,
                                             GOLD_CLASSES, _context_text,
                                             _pearson, _stdev,
                                             item_mean_deltas,
                                             keyword_echo_check,
                                             keyword_overlap, load_fixtures,
                                             run_fixture_with_judge, score)

# ------------------------------------------------------------- fakes

_OUTPUT_MARKER = "\n\nOutput ONLY"


def _extract_reply(prompt: str) -> str:
    """Recover the exact `reply` text a fake judge was asked to score,
    by splitting on the literal tail JUDGE_PROMPT always appends. Safe
    even when the reply itself contains internal blank lines (our canned
    jokes have a setup/punchline blank line) because the JUDGE_PROMPT
    scaffold never puts the literal string 'Output ONLY' anywhere except
    its own trailing instruction line."""
    context_part, _, reply_part = prompt.partition("Reply to score:")
    return reply_part.split(_OUTPUT_MARKER)[0].strip()


def make_perfect_judge(items):
    """A judge that measures EXACTLY what banter.py's context-ablation
    design claims to measure, and nothing else: contextual replies score
    high (9) only when shown their own true context, low (2) otherwise;
    canned replies score a flat 4 regardless of context (context-blind,
    delta 0 by construction); generic_responsive replies get a small,
    overlap-independent bump for their own topic (6 vs 4, delta 2) —
    "on-topic helps a little, but it isn't the same as being responsive."
    Built from a reply-text -> (gold_class, true_context) lookup over the
    real fixture, so it needs no knowledge beyond what the fixture itself
    encodes."""
    lookup = {it["reply"]: {"gold_class": it["gold_class"],
                            "true_context": _context_text(it)}
              for it in items}

    def judge(prompt: str) -> str:
        context_part, _, _reply_part = prompt.partition("Reply to score:")
        reply_text = _extract_reply(prompt)
        info = lookup[reply_text]
        is_true_context = info["true_context"] in context_part
        if info["gold_class"] == "contextual":
            return "9" if is_true_context else "2"
        if info["gold_class"] == "generic_responsive":
            return "6" if is_true_context else "4"
        return "4"  # canned: context-blind by construction

    return judge


def constant_judge(_prompt: str) -> str:
    """Always the same score. A judge this degenerate should produce a
    delta of exactly 0 everywhere -- no class separation, no echo-risk
    reading (the correlation is undefined, not zero, when nothing
    varies)."""
    return "7"


_WORD_RE = re.compile(r"[a-zA-Z]{5,}")


def echo_judge(prompt: str) -> str:
    """Scores purely by literal >=5-char word overlap between the reply
    and whatever context block it was shown -- blind to whether that
    overlap reflects genuine responsiveness. This is the failure mode
    BENCHMARK.md Section 1b warns about: a policy that sprinkles
    context-echoing keywords into an otherwise context-blind reply could
    inflate its context-ablation delta without truly being in-context.
    Deliberately independent of validate_banter_judge.keyword_overlap's
    implementation (different tokenizer, counts instead of Jaccard) so
    the test isn't just checking that a function agrees with itself."""
    context_part, _, _reply_part = prompt.partition("Reply to score:")
    reply_text = _extract_reply(prompt)
    reply_words = {w.lower() for w in _WORD_RE.findall(reply_text)}
    context_words = {w.lower() for w in _WORD_RE.findall(context_part)}
    overlap = len(reply_words & context_words)
    return str(min(10, max(1, 1 + 2 * overlap)))


# ------------------------------------------------------- fixture shape


class TestLoadFixtures(unittest.TestCase):
    def test_size_and_balance(self):
        items = load_fixtures()
        self.assertGreaterEqual(len(items), 24)
        self.assertLessEqual(len(items), 30)
        counts = {c: sum(1 for it in items if it["gold_class"] == c)
                 for c in GOLD_CLASSES}
        # "roughly balanced": no class off by more than 3 from an even split
        target = len(items) / len(GOLD_CLASSES)
        for c in GOLD_CLASSES:
            self.assertLessEqual(abs(counts[c] - target), 3, msg=counts)

    def test_unique_ids(self):
        items = load_fixtures()
        ids = [it["id"] for it in items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_item_has_notes_and_context_shape(self):
        items = load_fixtures()
        for it in items:
            self.assertTrue(it.get("notes"), it["id"])
            ctx = it["context"]
            self.assertGreaterEqual(len(ctx), 3)
            self.assertLessEqual(len(ctx), 9)  # generous upper bound
            self.assertEqual(ctx[0]["role"], "partner", it["id"])
            self.assertEqual(ctx[-1]["role"], "partner", it["id"])
            roles = [t["role"] for t in ctx]
            # strict partner/model alternation
            for i, r in enumerate(roles):
                self.assertEqual(r, "partner" if i % 2 == 0 else "model",
                                it["id"])

    def test_canned_replies_are_verbatim_corpus_jokes(self):
        # Spot-check: every canned reply contains the two-part
        # setup/punchline shape ("\n\n") the chatgpt-25-templates corpus
        # uses verbatim (DATA.md) -- distinguishes it from a paraphrase.
        items = load_fixtures()
        canned = [it for it in items if it["gold_class"] == "canned"]
        self.assertGreater(len(canned), 0)
        for it in canned:
            self.assertIn("\n\n", it["reply"], it["id"])


# --------------------------------------------------------- keyword_overlap


class TestKeywordOverlap(unittest.TestCase):
    def test_identical_text_full_overlap(self):
        self.assertEqual(keyword_overlap("hello world today", "hello world today"), 1.0)

    def test_disjoint_text_zero_overlap(self):
        self.assertEqual(keyword_overlap("purple gardening", "quantum orbital"), 0.0)

    def test_partial_overlap_between_zero_and_one(self):
        v = keyword_overlap("triple shot americano", "triple shot latte please")
        self.assertGreater(v, 0.0)
        self.assertLess(v, 1.0)

    def test_short_words_ignored_like_content_words(self):
        # "a", "to", "of" etc. are below the 5-char floor _content_words
        # applies -- overlap should stay 0 even though these words repeat.
        self.assertEqual(keyword_overlap("a to of it is", "of a it to is"), 0.0)


# ------------------------------------------------------- score() unit tests


class TestScorePureFunction(unittest.TestCase):
    """Exercise score() directly on hand-built ablation data (no judge,
    no fixture loop) -- mirrors validate_rejector.py's score() tests,
    which feed constructed `labels` dicts rather than running a labeler."""

    def _items(self):
        return [
            {"id": "c1", "gold_class": "contextual", "reply": "r c1",
             "context": [{"role": "partner", "content": "unique alpha bravo"}]},
            {"id": "g1", "gold_class": "generic_responsive", "reply": "r g1",
             "context": [{"role": "partner", "content": "unique charlie delta"}]},
            {"id": "n1", "gold_class": "canned", "reply": "r n1",
             "context": [{"role": "partner", "content": "unique echo foxtrot"}]},
        ]

    def test_unparseable_repeats_counted_and_excluded(self):
        items = self._items()
        ablations = {
            "c1": [{"delta": 8}, {"delta": None}],  # one parse failure
            "g1": [{"delta": 3}, {"delta": 3}],
            "n1": [{"delta": None}, {"delta": None}],  # total failure
        }
        result = score(items, ablations)
        self.assertEqual(result["per_class"]["contextual"]["mean_delta"], 8.0)
        self.assertEqual(result["per_class"]["canned"]["n_scored"], 0)
        self.assertIsNone(result["per_class"]["canned"]["mean_delta"])
        # 1 failure from c1 + 2 from n1 = 3
        self.assertEqual(result["n_unparseable_repeats"], 3)

    def test_separation_none_when_a_class_totally_unparseable(self):
        items = self._items()
        ablations = {
            "c1": [{"delta": 5}, {"delta": 5}],
            "g1": [{"delta": 1}, {"delta": 1}],
            "n1": [{"delta": None}, {"delta": None}],
        }
        result = score(items, ablations)
        self.assertIsNone(result["separation_contextual_minus_canned"])
        self.assertEqual(result["separation_contextual_minus_generic_responsive"], 4.0)

    def test_repeat_delta_stdev_mean_reflects_real_jitter(self):
        items = self._items()
        ablations = {
            "c1": [{"delta": 4}, {"delta": 8}],   # spread 2 either side of 6
            "g1": [{"delta": 2}, {"delta": 2}],   # no spread
            "n1": [{"delta": 0}, {"delta": 0}],   # no spread
        }
        result = score(items, ablations)
        # mean of [stdev([4,8])=2.828..., 0, 0] over the 3 items with >=2
        # parseable repeats
        self.assertAlmostEqual(result["repeat_delta_stdev_mean"],
                               _stdev([4, 8]) / 3, places=4)


class TestPearsonAndStdevHelpers(unittest.TestCase):
    def test_pearson_perfect_positive(self):
        self.assertAlmostEqual(_pearson([1, 2, 3], [2, 4, 6]), 1.0)

    def test_pearson_none_when_no_variance(self):
        self.assertIsNone(_pearson([1, 1, 1], [5, 2, 9]))
        self.assertIsNone(_pearson([1, 2, 3], [5, 5, 5]))

    def test_stdev_zero_for_single_value(self):
        self.assertEqual(_stdev([4.0]), 0.0)


# ---------------------------------------------- fake-judge integration


class TestPerfectJudge(unittest.TestCase):
    """The instrument, if it behaved exactly as designed, must show
    clean separation and no false echo-risk alarm."""

    def setUp(self):
        self.items = load_fixtures()
        judge = make_perfect_judge(self.items)
        self.ablations = run_fixture_with_judge(self.items, judge, repeats=2)
        self.result = score(self.items, self.ablations)

    def test_class_ordering_contextual_gt_generic_gt_canned(self):
        pc = self.result["per_class"]
        self.assertGreater(pc["contextual"]["mean_delta"],
                           pc["generic_responsive"]["mean_delta"])
        self.assertGreater(pc["generic_responsive"]["mean_delta"],
                           pc["canned"]["mean_delta"])

    def test_canned_delta_is_zero(self):
        self.assertEqual(self.result["per_class"]["canned"]["mean_delta"], 0.0)

    def test_separations_clearly_positive(self):
        self.assertGreater(self.result["separation_contextual_minus_canned"], 3.0)
        self.assertGreater(
            self.result["separation_contextual_minus_generic_responsive"], 2.0)

    def test_no_echo_risk_flagged(self):
        self.assertFalse(self.result["keyword_echo_check"]["risk_detected"])

    def test_perfectly_repeatable_deltas(self):
        # deterministic judge -> zero jitter across repeats
        self.assertEqual(self.result["repeat_delta_stdev_mean"], 0.0)


class TestConstantJudge(unittest.TestCase):
    """A judge with an absolute-scale bias but zero context sensitivity
    must show every delta at exactly 0 -- the subtraction in
    context_ablation_score is designed to cancel exactly this."""

    def setUp(self):
        self.items = load_fixtures()
        self.ablations = run_fixture_with_judge(self.items, constant_judge,
                                                repeats=2)
        self.result = score(self.items, self.ablations)

    def test_all_class_deltas_zero(self):
        for cls in GOLD_CLASSES:
            self.assertEqual(self.result["per_class"][cls]["mean_delta"], 0.0)

    def test_all_separations_zero(self):
        self.assertEqual(self.result["separation_contextual_minus_canned"], 0.0)
        self.assertEqual(
            self.result["separation_contextual_minus_generic_responsive"], 0.0)

    def test_echo_check_undefined_not_falsely_safe_looking(self):
        echo = self.result["keyword_echo_check"]
        # Correlation is mathematically undefined (zero variance both
        # sides) -- must read None, not 0.0, and must not be flagged as
        # risky from an undefined reading.
        self.assertIsNone(echo["generic_responsive_pearson_r"])
        self.assertFalse(echo["risk_detected"])


class TestEchoJudge(unittest.TestCase):
    """A judge that scores by literal keyword overlap, blind to genuine
    responsiveness, must trip keyword_echo_check -- the mechanical
    demonstration of BENCHMARK.md Section 1b's residual risk."""

    def setUp(self):
        self.items = load_fixtures()
        self.ablations = run_fixture_with_judge(self.items, echo_judge,
                                                repeats=2)
        self.result = score(self.items, self.ablations)

    def test_risk_detected(self):
        echo = self.result["keyword_echo_check"]
        self.assertTrue(echo["risk_detected"])
        self.assertIsNotNone(echo["generic_responsive_pearson_r"])
        self.assertGreater(echo["generic_responsive_pearson_r"], ECHO_RISK_THRESHOLD)

    def test_high_overlap_generic_responsive_approaches_contextual(self):
        # The literal failure mode from the design brief: generic_
        # responsive items with high keyword overlap should score much
        # closer to contextual-sized deltas than low-overlap ones do.
        echo = self.result["keyword_echo_check"]
        pc = self.result["per_class"]
        low = echo["generic_responsive_low_overlap_mean_delta"]
        high = echo["generic_responsive_high_overlap_mean_delta"]
        self.assertGreater(high, low)
        # high-overlap generic_responsive should close at least half the
        # gap to the contextual mean, relative to the low-overlap group
        contextual_mean = pc["contextual"]["mean_delta"]
        self.assertGreater(high, (low + contextual_mean) / 2 - 1.0)


if __name__ == "__main__":
    unittest.main()
