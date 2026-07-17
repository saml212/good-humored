"""Unit tests for the v4 field-coverage fix (EXP-008 addendum "v3 FAILS
in the field" + the adversarial audit's two-tier redesign,
EXPERIMENT_LOG.md).

Deliberately a SEPARATE file from test_rejector.py rather than additions
to it: test_rejector.py's TestV2Untouched hash-locks the exact v2
constants because a live experiment once depended on them not moving,
and the safest way to guarantee this file can never accidentally touch
that lock is to never edit that file at all. Everything below is v4-only
and additive, same discipline `benchmark/rejector.py`'s v4 section
follows (mirrors v3's shape where the two-tier redesign doesn't change
it; never imports or calls v2/v3 machinery except read-only vocabulary
comparisons).

No API calls anywhere in this file: every `complete` is a scripted fake.

Run: python3 -m pytest benchmark/tests/test_rejector_v4.py -q
"""

import json
import unittest
from collections import Counter
from pathlib import Path
from typing import Dict

from benchmark.metrics import normalize_label
from benchmark.rejector import (ALIASES_PATH_V4, LABEL_PROMPT_V4,
                                LABEL_PROMPT_VERSION, LABEL_PROMPT_VERSION_V3,
                                LABEL_PROMPT_VERSION_V4, TIER_CANON,
                                TIER_FREE, TIER_UNPARSEABLE, UNPARSEABLE,
                                VOCABULARY_PATH_V4, label_topic_v4,
                                load_aliases, load_vocabulary,
                                v4_normalize_label)

REPO_ROOT = Path(__file__).parent.parent.parent
V3_VOCAB_PATH = REPO_ROOT / "benchmark" / "fixtures" / "topic_vocabulary.txt"
RELABEL_DIR = (REPO_ROOT / "experiment-runs"
              / "2026-07-17-cascade-pilot-v3-relabel")
PROBES_PATH = (REPO_ROOT / "benchmark" / "fixtures"
              / "label_invariance_probes_v4.jsonl")
MIN_GAP_OCCURRENCES = 4
EXPECTED_PROBE_FAMILIES = {
    "farming_scarecrow": 6, "skeleton_guts": 6, "bicycle_two_tired": 6,
}


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


class TestVocabularyV4File(unittest.TestCase):
    def setUp(self):
        self.vocab = load_vocabulary(VOCABULARY_PATH_V4)
        self.v3_vocab = load_vocabulary(V3_VOCAB_PATH)

    def test_loads_and_in_size_range(self):
        # Build instructions: additive to v3's 110, lean, target <=160.
        self.assertGreaterEqual(len(self.vocab), 100)
        self.assertLessEqual(len(self.vocab), 160)

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

    def test_has_no_catch_all_entry(self):
        """Audit finding B1: a catch-all MERGES distinct rare topics
        into one label, manufacturing false repeats. The two-tier
        redesign makes this structurally impossible -- `other` must not
        be a vocabulary entry at all, not merely discouraged."""
        self.assertNotIn("other", self.vocab)
        self.assertNotIn("misc", self.vocab)
        self.assertNotIn("miscellaneous", self.vocab)
        self.assertNotIn("general", self.vocab)

    def test_path_points_at_fixtures_dir(self):
        self.assertEqual(VOCABULARY_PATH_V4.name, "topic_vocabulary_v4.txt")
        self.assertEqual(VOCABULARY_PATH_V4.parent.name, "fixtures")

    def test_is_superset_of_v3_minus_its_catch_all(self):
        """v4 is additive over v3's substantive entries: nothing v3
        offered (other than v3's own `other`, which v4 deliberately
        does not carry forward) is removed or renamed."""
        v3_norm = {normalize_label(e) for e in self.v3_vocab} - {"other"}
        v4_norm = {normalize_label(e) for e in self.vocab}
        missing = v3_norm - v4_norm
        self.assertEqual(missing, set(),
                         "v4 dropped v3 entries: %r" % missing)

    def test_still_deliberately_excludes_v3s_documented_jitter_synonyms(self):
        """v3 excluded pet/health/medicine on purpose (EXP-002's named
        failure modes). v4 must not silently re-add them alongside
        their canonical (doctor / cat / dog)."""
        vocab_norm = {normalize_label(e) for e in self.vocab}
        self.assertNotIn("pet", vocab_norm)
        self.assertNotIn("medicine", vocab_norm)
        self.assertNotIn("health", vocab_norm)
        self.assertIn("doctor", vocab_norm)

    def test_new_entries_are_not_hypernym_duplicates_of_each_other(self):
        """The near-synonym-shaped pairs this file documents (egg/food,
        farmer/farming, hair/hairdresser) must have both sides actually
        present -- if one side were missing the "coexistence, not
        omission" design decision documented in the file's header would
        be moot."""
        vocab_norm = {normalize_label(e) for e in self.vocab}
        for a, b in (("egg", "food"), ("farmer", "farming"),
                    ("hair", "hairdresser"), ("farm animal", "farming")):
            self.assertIn(a, vocab_norm)
            self.assertIn(b, vocab_norm)

    def test_ghost_deliberately_not_in_vocab(self):
        """Self-correction over an earlier draft: literal ghost jokes
        (transparency wordplay) are a different joke mechanism than the
        skeleton/bone-anatomy cluster and must NOT be folded into
        `skeleton` -- that would repeat, at small scale, the exact
        distinct-concepts-under-one-label merge the two-tier redesign
        exists to stop doing. `ghost` is deliberately left to the free
        tier, which the two-tier design makes safe."""
        vocab_norm = {normalize_label(e) for e in self.vocab}
        self.assertNotIn("ghost", vocab_norm)
        self.assertIn("skeleton", vocab_norm)


# ---------------------------------------------------------- LABEL_PROMPT_V4


class TestLabelPromptV4Contents(unittest.TestCase):
    def test_contains_every_vocab_entry(self):
        vocab = load_vocabulary(VOCABULARY_PATH_V4)
        missing = [e for e in vocab if e not in LABEL_PROMPT_V4]
        self.assertEqual(missing, [],
                         "LABEL_PROMPT_V4 is missing vocabulary entries "
                         "the model was never shown: %r" % missing)

    def test_has_joke_placeholder(self):
        rendered = LABEL_PROMPT_V4.format(joke="test joke text")
        self.assertIn("test joke text", rendered)

    def test_version_constant_distinct_from_v2_and_v3(self):
        self.assertEqual(LABEL_PROMPT_VERSION_V4, "v4-two-tier")
        self.assertNotEqual(LABEL_PROMPT_VERSION_V4, LABEL_PROMPT_VERSION)
        self.assertNotEqual(LABEL_PROMPT_VERSION_V4, LABEL_PROMPT_VERSION_V3)

    def test_no_longer_instructs_a_category_placeholder_escape(self):
        """Audit finding B1's fix, checked directly against the prompt
        text: the old v3-style "if nothing fits, answer other" escape
        hatch must be gone, replaced by an instruction to produce a
        specific free label instead."""
        self.assertNotIn('fits, answer "other"', LABEL_PROMPT_V4)
        self.assertIn("category placeholder", LABEL_PROMPT_V4)

    def test_free_tier_example_present_and_not_labeled_other(self):
        # The few-shot that used to demonstrate the "other" escape
        # hatch must now demonstrate a free-tier answer instead.
        self.assertIn("resident ghost", LABEL_PROMPT_V4)
        self.assertNotIn("</joke>\nother", LABEL_PROMPT_V4)


# ---------------------------------------------------------- label_topic_v4


class TestLabelTopicV4(unittest.TestCase):
    TINY_VOCAB = ["cat", "dog", "coffee"]

    def test_canon_reply_first_try(self):
        complete = FakeComplete(["cat"])
        label, tier = label_topic_v4("irrelevant", complete,
                                     vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, "cat")
        self.assertEqual(tier, TIER_CANON)
        self.assertEqual(len(complete.calls), 1)

    def test_reply_normalized_before_membership_check(self):
        complete = FakeComplete(["Cats."])
        label, tier = label_topic_v4("irrelevant", complete,
                                     vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, "cat")
        self.assertEqual(tier, TIER_CANON)
        self.assertEqual(len(complete.calls), 1)

    def test_first_line_only(self):
        complete = FakeComplete(["dog\nI hope that helps!"])
        label, tier = label_topic_v4("irrelevant", complete,
                                     vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, "dog")
        self.assertEqual(tier, TIER_CANON)

    def test_out_of_vocab_reply_is_kept_as_free_tier_no_retry(self):
        """The audit's B1 behavior change: v3 retried an out-of-vocab
        reply (because it had nowhere else for it to go); v4 has a free
        tier, so an out-of-vocab-but-shape-valid reply is accepted
        immediately, unmodified, tagged free -- NOT retried."""
        complete = FakeComplete(["parakeet"])
        label, tier = label_topic_v4("irrelevant", complete,
                                     vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, "parakeet")
        self.assertEqual(tier, TIER_FREE)
        self.assertEqual(len(complete.calls), 1,
                         "out-of-vocab must not trigger a retry in v4")

    def test_free_tier_preserves_exact_wording_ghost_example(self):
        complete = FakeComplete(["ghost"])
        label, tier = label_topic_v4("irrelevant", complete)
        self.assertEqual(label, "ghost")
        self.assertEqual(tier, TIER_FREE)

    def test_empty_reply_retries_then_unparseable(self):
        complete = FakeComplete(["", "still not a real answer here at all"])
        label, tier = label_topic_v4("irrelevant", complete,
                                     vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, UNPARSEABLE)
        self.assertEqual(tier, TIER_UNPARSEABLE)
        self.assertEqual(len(complete.calls), 2)

    def test_too_many_words_retries_then_recovers(self):
        rambling = "I am not sure what topic this joke belongs to exactly"
        complete = FakeComplete([rambling, "cat"])
        label, tier = label_topic_v4("irrelevant", complete,
                                     vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, "cat")
        self.assertEqual(tier, TIER_CANON)
        self.assertEqual(len(complete.calls), 2)

    def test_too_many_words_twice_returns_unparseable(self):
        rambling = "I am not sure what topic this joke belongs to exactly"
        complete = FakeComplete([rambling, rambling])
        label, tier = label_topic_v4("irrelevant", complete,
                                     vocabulary=self.TINY_VOCAB)
        self.assertEqual(label, UNPARSEABLE)
        self.assertEqual(tier, TIER_UNPARSEABLE)
        self.assertEqual(len(complete.calls), 2,
                         "must not retry more than once (bounded, like v2/v3)")

    def test_default_vocabulary_is_the_real_v4_file(self):
        complete = FakeComplete(["skeleton"])
        label, tier = label_topic_v4("irrelevant", complete)
        self.assertEqual(label, "skeleton")
        self.assertEqual(tier, TIER_CANON)

    def test_new_v4_only_entry_accepted_via_default_vocabulary(self):
        complete = FakeComplete(["comedy"])
        label, tier = label_topic_v4("irrelevant", complete)
        self.assertEqual(label, "comedy")
        self.assertEqual(tier, TIER_CANON)

    def test_documented_alias_resolves_to_canonical_as_canon_tier(self):
        """Re-audit MAJOR B1-adjacent finding: a model answering a
        DOCUMENTED synonym ("humor") must resolve to its canonical
        (`comedy`) as a canon hit, not escape to the free tier as its
        own distinct label -- 105/1532 wild turns were on the line for
        exactly this reason."""
        complete = FakeComplete(["humor"])
        label, tier = label_topic_v4("irrelevant", complete)
        self.assertEqual(label, "comedy")
        self.assertEqual(tier, TIER_CANON)
        self.assertEqual(len(complete.calls), 1)

    def test_bike_alias_resolves_to_bicycle(self):
        complete = FakeComplete(["bike"])
        label, tier = label_topic_v4("irrelevant", complete)
        self.assertEqual(label, "bicycle")
        self.assertEqual(tier, TIER_CANON)

    def test_alias_reply_normalized_before_lookup(self):
        # "Humor." (capitalized, punctuated) must still resolve -- alias
        # lookup happens on the v4-normalized label, not the raw reply.
        complete = FakeComplete(["Humor."])
        label, tier = label_topic_v4("irrelevant", complete)
        self.assertEqual(label, "comedy")
        self.assertEqual(tier, TIER_CANON)

    def test_non_aliased_out_of_vocab_reply_still_free_tier(self):
        # Sanity: alias resolution must not swallow genuine free-tier
        # replies that simply aren't in the alias table.
        complete = FakeComplete(["parakeet"])
        label, tier = label_topic_v4("irrelevant", complete,
                                     vocabulary=["cat", "dog", "coffee"],
                                     aliases={})
        self.assertEqual(label, "parakeet")
        self.assertEqual(tier, TIER_FREE)

    def test_explicit_aliases_param_overrides_default_table(self):
        complete = FakeComplete(["puppy"])
        label, tier = label_topic_v4("irrelevant", complete,
                                     vocabulary=["dog"],
                                     aliases={"puppy": "dog"})
        self.assertEqual(label, "dog")
        self.assertEqual(tier, TIER_CANON)


class TestV4UnicodeNormalization(unittest.TestCase):
    """Re-audit MAJOR finding: the shared normalize_label (benchmark/
    metrics.py, v2/v3 hash-locked) strips ASCII punctuation only, so a
    reply like "cat! 🐱" normalizes to "cat 🐱" -- the emoji survives as
    its own token and the label permanently escapes to the free tier
    (v3 never hit this because its now-removed retry-on-out-of-vocab
    loop happened to paper over it). v4_normalize_label is the v4-only
    fix layered on top of (never instead of) normalize_label; these are
    the auditor's own probe cases."""

    def test_v4_normalize_label_strips_emoji_keeps_letters(self):
        self.assertEqual(v4_normalize_label("cat! 🐱"), "cat")
        self.assertEqual(v4_normalize_label("🥚 egg"), "egg")

    def test_v4_normalize_label_preserves_internal_spaces(self):
        self.assertEqual(v4_normalize_label("farm animal! 🐄"), "farm animal")

    def test_v4_normalize_label_emoji_only_is_empty(self):
        self.assertEqual(v4_normalize_label("🎉"), "")

    def test_v4_normalize_label_does_not_mutate_shared_normalize_label(self):
        # v2/v3 hash-lock: the shared function itself must still leave
        # emoji untouched -- v4's fix must be a layer on top, never an
        # edit to normalize_label.
        self.assertEqual(normalize_label("cat! 🐱"), "cat 🐱")

    def test_auditor_probe_cat_emoji_resolves_canon(self):
        complete = FakeComplete(["cat! 🐱"])
        label, tier = label_topic_v4("irrelevant", complete)
        self.assertEqual(label, "cat")
        self.assertEqual(tier, TIER_CANON)

    def test_auditor_probe_egg_emoji_resolves_canon(self):
        complete = FakeComplete(["🥚 egg"])
        label, tier = label_topic_v4("irrelevant", complete)
        self.assertEqual(label, "egg")
        self.assertEqual(tier, TIER_CANON)

    def test_free_tier_emoji_case_stays_free_but_stripped(self):
        complete = FakeComplete(["🎉 celebration"])
        label, tier = label_topic_v4("irrelevant", complete)
        self.assertEqual(label, "celebration")
        self.assertEqual(tier, TIER_FREE)


# --------------------------------------------- wild-data coverage regression


class TestWildDataCoverage(unittest.TestCase):
    """The test that would have caught v3's field failure before it
    shipped: recomputes the v2-label histogram of every wild turn v3
    mapped to `other`, directly from the committed relabel data (never
    modified by this test -- read-only), and asserts every label that
    crosses the >=4-occurrence bar is accounted for by name below, with
    whatever that decision claims actually true of the v4 vocabulary
    file on disk. A future v5 gap that isn't triaged here fails loudly
    instead of silently shipping the same failure mode again.
    """

    # label -> (kind, target, required_prompt_terms)
    #   ("new", "<entry>", terms)      -- must be in v4 but NOT in v3.
    #   ("existing", "<entry>", terms) -- must be in BOTH v3 and v4 (v3
    #                                     already had a home; the fix is
    #                                     prompt guidance, not a new
    #                                     entry).
    #   ("excluded", "<reason>", terms)-- deliberately uncovered by a
    #                                     canon entry; may still require
    #                                     guidance terms (e.g. routing
    #                                     to an existing entry).
    # `required_prompt_terms`: non-vocab marker phrases that must appear
    # in LABEL_PROMPT_V4's actual text for this decision's claimed
    # guidance to be real (audit finding B2 -- see TestPromptGuidanceBacked,
    # which also asserts none of these markers are themselves vocabulary
    # words, closing the exact loophole that let B2 ship: a bare vocab
    # word always appears in the prompt via the joined topic list,
    # whether or not real disambiguation guidance was ever written).
    DECISIONS = {
        "comedy": ("new", "comedy", ("painter",)),
        "humor": ("new", "comedy", ()),
        "joke": ("new", "comedy", ()),
        "performance": ("new", "comedy", ()),
        "silence": ("new", "comedy", ()),
        "nothing": ("new", "comedy", ()),
        "censorship": ("new", "censorship", ()),
        "rule": ("new", "censorship", ()),
        "feedback": ("new", "censorship", ()),
        "clothing": ("new", "clothing", ()),
        "shoe": ("new", "clothing", ()),
        "stair": ("new", "stair", ()),
        "farming": ("new", "farming", ("scarecrow",)),
        "writing": ("new", "writing", ()),
        "food": ("new", "food", ()),
        "cheese": ("new", "food", ()),
        "appliance": ("new", "appliance", ()),
        "furniture": ("new", "furniture", ()),
        "bone": ("new", "skeleton", ()),
        "skeleton": ("new", "skeleton", ("guts",)),
        "ghost": ("excluded",
                 "distinct joke mechanism (transparency/see-through "
                 "wordplay, not bone/skeleton anatomy) -- an earlier "
                 "draft folded this into `skeleton` on Halloween-theme "
                 "proximity alone, which is exactly the kind of "
                 "unrelated-concepts-under-one-label merge the two-tier "
                 "redesign exists to stop doing. Two-tier makes leaving "
                 "this to the free tier safe; no canon entry earns its "
                 "place at n=5 once merging risk is no longer the "
                 "alternative", ()),
        "mirror": ("new", "mirror", ()),
        "appearance": ("new", "mirror", ()),
        "paranoia": ("new", "paranoia", ()),
        "wall": ("new", "wall", ()),
        "egg": ("new", "egg", ()),
        "elevator": ("new", "elevator", ()),
        "bicycle": ("new", "bicycle", ()),
        "bike": ("new", "bicycle", ()),
        "pirate": ("new", "pirate", ()),
        "hair": ("new", "hair", ("facial hair",)),
        "death": ("existing", "death", ("guts",)),
        "work": ("existing", "work", ("promotion",)),
        "travel": ("existing", "travel", ("journey",)),
        "language": ("existing", "language", ("grammar",)),
        "laundry": ("excluded",
                   "chore-domain (socks, laundry machine); "
                   "LABEL_PROMPT_V4 now explicitly routes this to the "
                   "existing `chore` entry rather than the free tier, "
                   "since `chore` already exists and splitting "
                   "laundry/cleaning/chore three ways would be pure "
                   "jitter", ("soap",)),
        "cleaning": ("excluded",
                    "chore-domain (soap, broom, paper towels); "
                    "LABEL_PROMPT_V4 now explicitly routes this to the "
                    "existing `chore` entry", ("soap",)),
        "building": ("excluded",
                    "sampled wild jokes were mislabeled wall jokes, "
                    "mislabeled elevator jokes, and comedian-exits-the-"
                    "venue meta-humor already covered by `wall`/"
                    "`elevator`/`comedy` -- no distinct genre observed",
                    ()),
        "unparseable": ("excluded",
                       "v2's own UNPARSEABLE sentinel leaking into the "
                       "topic_v2 field for jokes v2 itself failed to "
                       "parse -- not a real topic, a category error to "
                       "add vocabulary for", ()),
    }

    @classmethod
    def setUpClass(cls):
        if not RELABEL_DIR.is_dir():
            raise unittest.SkipTest(
                "committed relabel data not found at %s -- this is a "
                "regression test against the real wild distribution, "
                "not a synthetic fixture, so it is skipped (not faked) "
                "when that data isn't present" % RELABEL_DIR)
        cls.hist, cls.total, cls.other_total = cls._compute_histogram()
        cls.v3_vocab = {normalize_label(e) for e in load_vocabulary(V3_VOCAB_PATH)}
        cls.v4_vocab = {normalize_label(e)
                        for e in load_vocabulary(VOCABULARY_PATH_V4)}

    @staticmethod
    def _compute_histogram():
        files = sorted(RELABEL_DIR.glob("lane-*/turns-*.relabel-v3.jsonl"))
        hist: Counter = Counter()
        total = 0
        other_total = 0
        for fp in files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    total += 1
                    if rec.get("topic_v3") == "other":
                        other_total += 1
                        hist[rec.get("topic_v2")] += 1
        return hist, total, other_total

    def test_reproduces_the_addendum_headline_numbers(self):
        # Locks the addendum's own reported figures as a floor: if this
        # ever drifts, the gap analysis below is running against
        # different data than what this vocabulary was justified by.
        self.assertEqual(self.total, 1532)
        self.assertEqual(self.other_total, 653)
        self.assertAlmostEqual(self.other_total / self.total, 0.426, places=3)

    def test_every_gap_label_is_decided(self):
        gap_labels = {label for label, count in self.hist.items()
                     if count >= MIN_GAP_OCCURRENCES}
        self.assertGreater(len(gap_labels), 0,
                          "sanity: the >=%d threshold should not be "
                          "vacuous on real data" % MIN_GAP_OCCURRENCES)
        undecided = sorted(gap_labels - set(self.DECISIONS))
        self.assertEqual(undecided, [],
                         "wild v2 labels with >=%d occurrences among "
                         "v3's `other`-mapped turns have no v4 coverage "
                         "decision recorded: %r -- this is exactly the "
                         "failure mode that shipped as v3 (a real, "
                         "frequent wild topic with no vocabulary entry "
                         "and no documented reason to skip it)"
                         % (MIN_GAP_OCCURRENCES, undecided))

    def test_no_stale_decisions_below_the_bar(self):
        stale = [label for label in self.DECISIONS
                if self.hist.get(label, 0) < MIN_GAP_OCCURRENCES]
        self.assertEqual(stale, [],
                         "DECISIONS records labels that do not actually "
                         "cross the >=%d bar in the committed wild data: "
                         "%r" % (MIN_GAP_OCCURRENCES, stale))

    def test_every_decision_is_actually_true_of_v4(self):
        for label, (kind, target, _terms) in self.DECISIONS.items():
            with self.subTest(label=label):
                if kind == "new":
                    self.assertIn(target, self.v4_vocab,
                                 "%r claims new v4 entry %r, but it is "
                                 "not in topic_vocabulary_v4.txt"
                                 % (label, target))
                    self.assertNotIn(target, self.v3_vocab,
                                     "%r claims %r is a NEW v4 entry, "
                                     "but it was already in v3 -- "
                                     "decision table is stale"
                                     % (label, target))
                elif kind == "existing":
                    self.assertIn(target, self.v3_vocab,
                                 "%r claims EXISTING v3 entry %r, but "
                                 "it is not in v3" % (label, target))
                    self.assertIn(target, self.v4_vocab,
                                 "%r claims existing entry %r, but v4 "
                                 "dropped it" % (label, target))
                elif kind == "excluded":
                    self.assertTrue(target,
                                    "excluded decision for %r needs a "
                                    "non-empty documented reason" % label)
                else:
                    self.fail("unknown decision kind %r for %r"
                             % (kind, label))


# --------------------------------------------- alias table covers fold claims


def _fold_claims_from_decisions() -> Dict[str, str]:
    """Every (label, target) pair in DECISIONS where kind == 'new' and
    label != target is a documented "this wild label folds into this
    canonical entry" claim -- exactly what the alias table must cover.
    ("new" entries where label == target are the canonical itself, not
    a fold; "existing"/"excluded" entries are handled by prompt
    guidance, not the alias mechanism -- see TestPromptGuidanceBacked.)
    """
    return {label: target
            for label, (kind, target, _terms)
            in TestWildDataCoverage.DECISIONS.items()
            if kind == "new" and label != target}


class TestAliasTableCoversFoldClaims(unittest.TestCase):
    """Re-audit MAJOR finding: topic_vocabulary_v4.txt's comments and
    the DECISIONS table document folds (humor/joke/performance/
    silence/nothing -> comedy, rule/feedback -> censorship, shoe ->
    clothing, cheese -> food, bone -> skeleton, appearance -> mirror,
    bike -> bicycle) that label_topic_v4 did not actually resolve --
    exact membership matching let a documented synonym escape to the
    free tier as its own label. This test parses the fold claims
    straight out of DECISIONS (the same structured table
    TestPromptGuidanceBacked uses for its own claims) and asserts every
    one of them is a real entry in the alias table on disk, with
    nothing extra and nothing pointing the wrong way.
    """

    def test_every_fold_claim_is_an_alias(self):
        aliases = load_aliases()
        claims = _fold_claims_from_decisions()
        missing = {label: target for label, target in claims.items()
                  if aliases.get(normalize_label(label))
                  != normalize_label(target)}
        self.assertEqual(missing, {},
                         "DECISIONS documents these folds, but "
                         "topic_vocabulary_v4_aliases.txt does not "
                         "resolve them the same way: %r" % missing)

    def test_alias_table_has_no_entries_beyond_documented_fold_claims(self):
        # The inverse check: nothing in the alias file may exist
        # without a matching DECISIONS fold claim -- an alias added
        # without updating DECISIONS (or vice versa) fails loudly here.
        aliases = load_aliases()
        claims = _fold_claims_from_decisions()
        normalized_claims = {normalize_label(l): normalize_label(t)
                             for l, t in claims.items()}
        self.assertEqual(aliases, normalized_claims)

    def test_every_alias_canonical_is_a_real_v4_vocab_entry(self):
        vocab_norm = {normalize_label(e)
                     for e in load_vocabulary(VOCABULARY_PATH_V4)}
        aliases = load_aliases()
        self.assertGreater(len(aliases), 0, "sanity: table should not be empty")
        for alias, canonical in aliases.items():
            with self.subTest(alias=alias):
                self.assertIn(canonical, vocab_norm,
                             "alias %r points at %r, which is not a "
                             "real vocabulary entry" % (alias, canonical))
                self.assertNotIn(alias, vocab_norm,
                                 "alias %r is ALSO a canonical vocab "
                                 "entry -- ambiguous (which wins?)"
                                 % alias)

    def test_aliases_path_points_at_fixtures_dir(self):
        self.assertEqual(ALIASES_PATH_V4.name,
                         "topic_vocabulary_v4_aliases.txt")
        self.assertEqual(ALIASES_PATH_V4.parent.name, "fixtures")


# ------------------------------------------------- B2: guidance actually there


class TestPromptGuidanceBacked(unittest.TestCase):
    """The test that would have caught audit finding B2: a prior draft's
    vocab-file comments and DECISIONS table claimed LABEL_PROMPT_V4
    carried disambiguation guidance for death/work/travel/language and
    laundry/cleaning->chore that, when actually checked against the
    prompt text, was not there (zero occurrences). The bare vocabulary
    word is NOT valid evidence of real guidance -- every vocab entry
    already appears in the prompt via the joined topic list regardless
    of whether any disambiguation sentence was ever written, which is
    exactly the blind spot that let B2 ship. Every DECISIONS entry's
    `required_prompt_terms` is a marker phrase that only appears if the
    real guidance sentence exists, verified NOT to be a vocab word
    itself.
    """

    def test_every_required_term_present_in_prompt(self):
        for label, (_kind, _target, terms) in (
                TestWildDataCoverage.DECISIONS.items()):
            for term in terms:
                with self.subTest(label=label, term=term):
                    self.assertIn(term, LABEL_PROMPT_V4,
                                 "%r's decision claims guidance term "
                                 "%r, but it is not present in "
                                 "LABEL_PROMPT_V4's actual text -- this "
                                 "is exactly audit finding B2"
                                 % (label, term))

    def test_required_terms_are_not_themselves_vocab_entries(self):
        vocab_norm = {normalize_label(e)
                     for e in load_vocabulary(VOCABULARY_PATH_V4)}
        for label, (_kind, _target, terms) in (
                TestWildDataCoverage.DECISIONS.items()):
            for term in terms:
                with self.subTest(label=label, term=term):
                    self.assertNotIn(
                        normalize_label(term), vocab_norm,
                        "%r's guidance marker %r is itself a vocabulary "
                        "entry -- it would appear in the prompt "
                        "regardless of whether real disambiguation "
                        "guidance was ever written, defeating this "
                        "check's entire purpose (the exact blind spot "
                        "that let B2 ship)" % (label, term))


# ------------------------------------------- M1/M2: repeated-item invariance


class TestInvarianceProbeFixtureParsing(unittest.TestCase):
    """Fake-completer regression tests using the ACTUAL wild joke text
    (verbatim, not paraphrased) for the three joke families the
    adversarial audit found scattering worst under v3/free-v2 labeling:
      M1: the scarecrow "outstanding in his field" joke -- 36 verbatim
          occurrences traced across the corpus, split v3 `other` (16),
          `work` (10), `farmer` (9), `farm animal` (1).
      M2: the skeleton "they don't have the guts" joke -- 28 verbatim
          occurrences, split free-v2 `death` (11) / `bone` (9) /
          `skeleton` (8), then 27/28 landed in v3's `other`.
      bicycle: the "two-tired" bicycle joke -- 23 verbatim occurrences,
          split v3 `other` (15) / `driving` (5) / `car` (2) /
          `exercise` (1) and free-v2 `bike` (9) / `bicycle` (7) /
          `vehicle` (2) / `exercise` (2) / `transportation` (2) /
          `cycling` (1) -- resolved via the `bike` -> `bicycle` alias.
    These tests check that label_topic_v4's PARSING pipeline (prompt
    rendering, first-line extraction, normalize_label) handles the real
    strings -- embedded newlines, curly vs. straight quotes, an "ever"
    variant, a first-person paraphrase, an emoji -- without breaking,
    using a scripted-correct fake. Whether a REAL model actually gets
    these right is a different question, answered by
    benchmark/validate_invariance_probes.py against a live provider
    (not exercised here -- no network in this file).
    """

    @classmethod
    def setUpClass(cls):
        with open(PROBES_PATH) as f:
            cls.probes = [json.loads(l) for l in f if l.strip()]

    def test_fixture_has_all_three_families_with_six_items_each(self):
        families = Counter(p["family"] for p in self.probes)
        self.assertEqual(dict(families), EXPECTED_PROBE_FAMILIES)
        self.assertEqual(sum(EXPECTED_PROBE_FAMILIES.values()), 18)

    def test_every_probe_targets_a_real_v4_canon_entry(self):
        vocab_norm = {normalize_label(e)
                     for e in load_vocabulary(VOCABULARY_PATH_V4)}
        for p in self.probes:
            with self.subTest(family=p["family"]):
                self.assertEqual(p["expected_tier"], TIER_CANON)
                self.assertIn(normalize_label(p["expected_label"]), vocab_norm)

    def test_every_probe_joke_is_verbatim_wild_text(self):
        """A probe fixture that silently drifted from the real corpus
        (paraphrased instead of copied) would defeat the entire point
        of testing against ACTUAL wild scatter."""
        if not RELABEL_DIR.is_dir():
            self.skipTest("committed relabel data not found at %s"
                          % RELABEL_DIR)
        corpus_jokes = set()
        for fp in RELABEL_DIR.glob("lane-*/turns-*.relabel-v3.jsonl"):
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        corpus_jokes.add(json.loads(line)["joke"])
        missing = [p["joke"] for p in self.probes
                  if p["joke"] not in corpus_jokes]
        self.assertEqual(missing, [],
                         "probe fixture contains text not found "
                         "verbatim in the committed relabel corpus: %r"
                         % missing)

    def test_label_topic_v4_parses_every_probe_with_a_scripted_correct_reply(self):
        for p in self.probes:
            with self.subTest(family=p["family"], joke=p["joke"][:40]):
                complete = FakeComplete([p["expected_label"]])
                label, tier = label_topic_v4(p["joke"], complete)
                self.assertEqual(label, p["expected_label"])
                self.assertEqual(tier, p["expected_tier"])

    def test_label_topic_v4_renders_every_probe_joke_verbatim_into_the_prompt(self):
        for p in self.probes:
            with self.subTest(joke=p["joke"][:40]):
                captured = []

                def complete(prompt, _captured=captured):
                    _captured.append(prompt)
                    return p["expected_label"]

                label_topic_v4(p["joke"], complete)
                self.assertIn(p["joke"], captured[0],
                             "the exact wild joke text must survive "
                             "LABEL_PROMPT_V4.format() unmodified -- a "
                             "curly quote or embedded newline silently "
                             "mangling this would break the labeler on "
                             "exactly the data that matters most")


if __name__ == "__main__":
    unittest.main()
