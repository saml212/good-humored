"""Unit tests for the rejector's labeling instrument (v2 and v3).

No API calls: `complete` is always a fake callable here. Covers:
  - the v3 constrained vocabulary file (loads, normalized-unique, superset
    of the fixture's golds)
  - LABEL_PROMPT_V3 actually presents every vocabulary entry to the model
  - label_topic_v3's validation/retry/UNPARSEABLE layer
  - a regression lock on the v2 constants: EXP-008 is additive-only and a
    live experiment (EXP-004's cascade pilot) depends on v2 not moving.

Run: python3 -m unittest discover -s benchmark/tests -t .
"""

import hashlib
import json
import unittest
from pathlib import Path

from benchmark.metrics import normalize_label
from benchmark.rejector import (LABEL_PROMPT, LABEL_PROMPT_V3,
                                LABEL_PROMPT_VERSION,
                                LABEL_PROMPT_VERSION_V3, OPENING_PROMPT,
                                UNPARSEABLE, VOCABULARY_PATH, label_topic,
                                label_topic_v3, load_vocabulary)

FIXTURES = (Path(__file__).parent.parent / "fixtures"
           / "rejector_validation.jsonl")


def load_fixture_items():
    with open(FIXTURES) as f:
        return [json.loads(line) for line in f if line.strip()]


class FakeComplete:
    """Scripted `complete` stand-in: returns queued replies in order,
    records every prompt it was called with."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._replies:
            raise AssertionError("FakeComplete ran out of scripted replies")
        return self._replies.pop(0)


# ---------------------------------------------------------- vocabulary


class TestVocabularyFile(unittest.TestCase):
    def setUp(self):
        self.vocab = load_vocabulary()

    def test_loads_and_in_size_range(self):
        # Build instructions call for ~100-140 entries.
        self.assertGreaterEqual(len(self.vocab), 100)
        self.assertLessEqual(len(self.vocab), 140)

    def test_every_entry_is_normalize_label_idempotent(self):
        non_idempotent = [e for e in self.vocab if normalize_label(e) != e]
        self.assertEqual(non_idempotent, [],
                         "vocabulary entries must already be in "
                         "normalize_label's canonical form: %r"
                         % non_idempotent)

    def test_no_duplicates_after_normalization(self):
        normed = [normalize_label(e) for e in self.vocab]
        self.assertEqual(len(normed), len(set(normed)),
                         "two distinct lines normalize to the same "
                         "label -- the model would have no way to "
                         "disambiguate between them")

    def test_no_raw_duplicates(self):
        self.assertEqual(len(self.vocab), len(set(self.vocab)))

    def test_has_escape_hatch(self):
        self.assertIn("other", self.vocab)

    def test_path_points_at_fixtures_dir(self):
        self.assertEqual(VOCABULARY_PATH.name, "topic_vocabulary.txt")
        self.assertEqual(VOCABULARY_PATH.parent.name, "fixtures")

    def test_superset_of_fixture_gold_concepts(self):
        """Every fixture group/gold_topic/alt_topic must have a reachable
        entry. NOTE: validate_rejector.score() computes ARI/pair-match
        over the fixture's `group` field via majority-vote label identity
        (see benchmark/validate_rejector.py `score()`), never by string-
        matching `gold_topic` literally -- so this checks that the
        underlying CONCEPT is present, not that this file echoes the
        fixture's exact spelling (topic_vocabulary.txt's header explains
        why e.g. "programming"/"travel" are used instead of the fixture's
        literal "programmers"/"airplanes")."""
        items = load_fixture_items()
        vocab_norm = {normalize_label(e) for e in self.vocab}
        # Concepts every unambiguous group must be able to reach.
        required = {"cat", "dog", "marriage", "work", "doctor", "coffee",
                    "weather", "exercise"}
        missing = required - vocab_norm
        self.assertEqual(missing, set(),
                         "vocabulary is missing entries for fixture "
                         "gold concepts: %r" % missing)
        # ambiguous items' gold_topic / alt_topics must also resolve.
        alt_and_gold = set()
        for i in items:
            if i["group"] == "ambiguous":
                alt_and_gold.add(normalize_label(i["gold_topic"]))
                for a in i.get("alt_topics", []):
                    alt_and_gold.add(normalize_label(a))
        missing_ambig = alt_and_gold - vocab_norm
        self.assertEqual(missing_ambig, set(),
                         "vocabulary is missing entries the ambiguous "
                         "fixture items need: %r" % missing_ambig)

    def test_deliberately_excludes_documented_jitter_synonyms(self):
        """EXP-002's two named misses were cat/pet and health/medicine.
        The fix is to offer only ONE of each pair, not both -- assert
        that design choice actually landed in the file (a future editor
        re-adding "pet" or "medicine" alongside its synonym would
        silently undo the whole point of this vocabulary)."""
        vocab_norm = {normalize_label(e) for e in self.vocab}
        self.assertNotIn("pet", vocab_norm)
        self.assertNotIn("medicine", vocab_norm)
        self.assertNotIn("health", vocab_norm)
        # doctor must still be present as the single answer for the
        # fixture's doctor-visit jokes.
        self.assertIn("doctor", vocab_norm)


# ---------------------------------------------------------- LABEL_PROMPT_V3


class TestLabelPromptV3Contents(unittest.TestCase):
    def test_contains_every_vocab_entry(self):
        vocab = load_vocabulary()
        missing = [e for e in vocab if e not in LABEL_PROMPT_V3]
        self.assertEqual(missing, [],
                         "LABEL_PROMPT_V3 is missing vocabulary entries "
                         "the model was never shown: %r" % missing)

    def test_has_joke_placeholder(self):
        # Must be format()-able with a joke and nothing else missing.
        rendered = LABEL_PROMPT_V3.format(joke="test joke text")
        self.assertIn("test joke text", rendered)

    def test_version_constant_distinct_from_v2(self):
        self.assertEqual(LABEL_PROMPT_VERSION_V3, "v3-constrained")
        self.assertNotEqual(LABEL_PROMPT_VERSION_V3, LABEL_PROMPT_VERSION)


# ---------------------------------------------------------- label_topic_v3


class TestLabelTopicV3(unittest.TestCase):
    TINY_VOCAB = ["cat", "dog", "coffee", "other"]

    def test_valid_reply_first_try(self):
        complete = FakeComplete(["cat"])
        label = label_topic_v3("irrelevant", complete, vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, "cat")
        self.assertEqual(len(complete.calls), 1)

    def test_reply_normalized_before_membership_check(self):
        # Punctuation/case must not cause a false out-of-vocab retry.
        complete = FakeComplete(["Cats."])
        label = label_topic_v3("irrelevant", complete, vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, "cat")
        self.assertEqual(len(complete.calls), 1)

    def test_first_line_only(self):
        complete = FakeComplete(["dog\nI hope that helps!"])
        label = label_topic_v3("irrelevant", complete, vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, "dog")

    def test_out_of_vocab_then_valid_retries_once(self):
        complete = FakeComplete(["parakeet", "cat"])
        label = label_topic_v3("irrelevant", complete, vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, "cat")
        self.assertEqual(len(complete.calls), 2)

    def test_out_of_vocab_twice_returns_unparseable(self):
        complete = FakeComplete(["parakeet", "spaceship pirate"])
        label = label_topic_v3("irrelevant", complete, vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, UNPARSEABLE)
        self.assertEqual(len(complete.calls), 2,
                         "must not retry more than once (bounded, like v2)")

    def test_other_is_a_valid_reply(self):
        complete = FakeComplete(["other"])
        label = label_topic_v3("irrelevant", complete, vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, "other")

    def test_default_vocabulary_is_the_real_file(self):
        # No `vocabulary=` kwarg -> must fall back to the real
        # topic_vocabulary.txt-backed list, not the caller's fixture.
        complete = FakeComplete(["exercise"])
        label = label_topic_v3("irrelevant", complete)
        self.assertEqual(label, "exercise")


# ---------------------------------------------------------- v2 regression lock


class TestV2Untouched(unittest.TestCase):
    """EXP-008 build constraint: v3 is additive only. These hashes lock
    the exact v2 constants as they existed before this file was written
    -- any edit to LABEL_PROMPT/LABEL_PROMPT_VERSION/UNPARSEABLE/
    OPENING_PROMPT (accidental or not) fails this test loudly instead of
    silently invalidating EXP-004's in-flight cascade pilot, which is
    running against v2 right now."""

    def _sha256(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def test_label_prompt_version_hash(self):
        self.assertEqual(
            self._sha256(LABEL_PROMPT_VERSION),
            "fb04dcb6970e4c3d1873de51fd5a50d7bb46b3383113602665c350ec40b5f990")

    def test_label_prompt_hash(self):
        self.assertEqual(
            self._sha256(LABEL_PROMPT),
            "b64efbac84d982e275d6f1303617919bb651d2ddf38c2faa3f38e3506fbe26cc")

    def test_unparseable_sentinel_hash(self):
        self.assertEqual(
            self._sha256(UNPARSEABLE),
            "eca9a27f3d100d1d37ffeb6fc3be2a126382eb8d87f7e8d02192971053ca39a3")

    def test_opening_prompt_hash(self):
        self.assertEqual(
            self._sha256(OPENING_PROMPT),
            "84795a6eed119c4af38273f7736b6ea5e7c1125f641864caea396744f5c8dd99")

    def test_label_topic_v2_behavior_unchanged(self):
        """Not just the constants -- the v2 labeling function's actual
        behavior (first-line-only parsing, normalize_label, the
        word-count shape guard, UNPARSEABLE on repeated failure)."""
        complete = FakeComplete(["Travel!\nhope that's useful"])
        self.assertEqual(label_topic("irrelevant", complete), "travel")

        # >4-word "topic" retried once, then UNPARSEABLE (audit WARN-3).
        rambling = "I am not sure what topic this joke belongs to exactly"
        complete2 = FakeComplete([rambling, rambling])
        self.assertEqual(label_topic("irrelevant", complete2), UNPARSEABLE)
        self.assertEqual(len(complete2.calls), 2)


if __name__ == "__main__":
    unittest.main()
