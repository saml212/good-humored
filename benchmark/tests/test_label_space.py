"""Tests for LabelSpace (design doc: benchmark/label_space.py).

Fallback-mode tests always run, on every machine, by forcing _load_model
to fail — this is the only way to exercise the degraded path
deterministically on a box that actually HAS sentence_transformers
installed (this one does). Embedding-mode tests need the real model and
are skipped when the dependency isn't importable.

Run: python3 -m unittest discover benchmark/tests -v
"""

import unittest
import warnings
from unittest import mock

from benchmark.label_space import DEFAULT_THRESHOLD, LabelSpace
from benchmark.metrics import normalize_label

try:
    import sentence_transformers  # noqa: F401
    HAS_ST = True
except ImportError:
    HAS_ST = False


def fit_degraded(labels):
    """fit() with _load_model forced to fail -- deterministic fallback
    path regardless of what's actually installed on this machine."""
    with mock.patch.object(LabelSpace, "_load_model",
                           side_effect=ImportError("no sentence_transformers")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ls = LabelSpace().fit(labels)
            assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
                "degraded fit() must warn loudly"
    return ls


class TestFallbackMode(unittest.TestCase):
    def test_degraded_flag_set(self):
        ls = fit_degraded(["fitness", "exercise", "gym"])
        self.assertTrue(ls.degraded)

    def test_identity_over_normalize_label(self):
        ls = fit_degraded(["The Cats!", "puppies", "gym"])
        self.assertEqual(ls.canon("The Cats!"), normalize_label("The Cats!"))
        self.assertEqual(ls.canon("puppies"), "puppy")

    def test_no_cross_label_merging_when_degraded(self):
        # Would cluster together with real embeddings -- must NOT merge
        # in fallback mode (that's the whole point of "identity mapping").
        ls = fit_degraded(["fitness", "exercise", "gym"])
        self.assertNotEqual(ls.canon("fitness"), ls.canon("gym"))
        self.assertNotEqual(ls.canon("exercise"), ls.canon("gym"))

    def test_canonize_paths_degraded(self):
        ls = fit_degraded(["cat", "dog"])
        paths = [["The Cats!", "dog"], ["dog", "puppies"]]
        self.assertEqual(ls.canonize_paths(paths),
                         [["cat", "dog"], ["dog", "puppy"]])

    def test_unseen_label_falls_back_not_keyerror(self):
        ls = fit_degraded(["cat", "dog"])
        self.assertEqual(ls.canon("Gyms"), normalize_label("Gyms"))

    def test_empty_fit_is_safe(self):
        # Empty input returns before even trying to load the model --
        # degraded stays False (never attempted), canon() still works
        # via the unseen-label fallback.
        ls = LabelSpace().fit([])
        self.assertFalse(ls.degraded)
        self.assertEqual(ls.canon("cat"), "cat")

    def test_default_threshold_is_calibrated_value(self):
        self.assertEqual(LabelSpace().threshold, DEFAULT_THRESHOLD)


@unittest.skipIf(not HAS_ST, "sentence_transformers not installed")
class TestEmbeddingMode(unittest.TestCase):
    """Real embedding-mode tests. Model init (~90MB, HF-cached after the
    first call) happens once for the whole class."""

    @classmethod
    def setUpClass(cls):
        cls.mixed_labels = [
            "fitness", "exercise", "gym", "travel", "flying", "airplanes",
            "cat", "dog", "coffee", "tea", "doctors", "lawyers",
        ]
        cls.ls = LabelSpace().fit(cls.mixed_labels)

    def test_not_degraded(self):
        self.assertFalse(self.ls.degraded)

    def test_exp001_fitness_gym_merge(self):
        # EXP-001's named scatter: fitness/gym merge at the calibrated
        # threshold (a disclosed PARTIAL fix -- see label_space.py's
        # DEFAULT_THRESHOLD comment; "exercise" stays a singleton here).
        self.assertEqual(self.ls.canon("fitness"), self.ls.canon("gym"))

    def test_exp001_flying_airplanes_merge(self):
        self.assertEqual(self.ls.canon("flying"), self.ls.canon("airplanes"))

    def test_acid_test_near_neighbors_stay_split(self):
        # The near-neighbor pairs the calibration fixture calls out as
        # must-NOT-merge must actually stay split at the chosen threshold
        # -- this is the false-merge risk the threshold was picked to avoid.
        self.assertNotEqual(self.ls.canon("cat"), self.ls.canon("dog"))
        self.assertNotEqual(self.ls.canon("coffee"), self.ls.canon("tea"))
        self.assertNotEqual(self.ls.canon("doctors"), self.ls.canon("lawyers"))

    def test_representative_is_most_frequent(self):
        labels = ["gym", "gym", "gym", "fitness"]  # sim 0.74 >= threshold
        ls = LabelSpace().fit(labels)
        self.assertEqual(ls.canon("fitness"), "gym")
        self.assertEqual(ls.canon("gym"), "gym")

    def test_representative_tiebreak_shortest(self):
        # Equal frequency (1 each), high-similarity pair (well clear of
        # the threshold boundary) -- shortest string wins the tie.
        ls = LabelSpace().fit(["mother", "mom"])
        self.assertEqual(ls.canon("mother"), "mom")
        self.assertEqual(ls.canon("mom"), "mom")

    def test_canonize_paths_end_to_end(self):
        paths = [["fitness", "gym"], ["cat", "dog"]]
        out = self.ls.canonize_paths(paths)
        self.assertEqual(out[0][0], out[0][1])      # fitness canon == gym canon
        self.assertNotEqual(out[1][0], out[1][1])   # cat canon != dog canon

    def test_unfit_label_falls_back_to_normalized_form(self):
        # A topic never seen at fit() time must not crash or silently
        # join some unrelated class.
        self.assertEqual(self.ls.canon("Basketballs"),
                         normalize_label("Basketballs"))


if __name__ == "__main__":
    unittest.main()
