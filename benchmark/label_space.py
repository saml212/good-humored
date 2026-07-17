"""Semantic label equivalence for trajectory metrics.

EXP-001 found the rejector labels topics correctly but scatters synonyms
across distinct strings (fitness/exercise/gym; travel/flying) — correct
labels, split partitions. metrics.py compares paths by string equality
(jaccard, prefix_agreement_depth, edit_distance all do `==` on labels), so
a synonym scatter silently understates a model's real topic collapse and
overstates its divergence. LabelSpace closes that gap upstream of
metrics.py: fit it on the labels actually observed, then canonicalize
every path through it before calling any metric.

This is the ONLY file in the package allowed to import sentence_transformers
— and it does so lazily, inside fit(), not at module level. metrics.py and
everything else must stay importable with zero ML deps on a fresh box
(validate_rejector.py runs plenty of experiments with no need for a
600MB torch install).
"""

import warnings
from collections import Counter
from typing import Dict, List, Sequence

from .metrics import normalize_label

# Calibrated 2026-07-16 against benchmark/fixtures/label_equivalence_pairs.jsonl
# (32 must-merge / 32 must-not-merge pairs) via calibrate_label_space.py,
# model all-MiniLM-L6-v2. Pairwise sweep (thresh: precision / recall / F0.5):
#   0.50: .721/.969/.760   0.60: .829/.906/.843   0.70: .920/.719/.871
#   0.55: .750/.938/.781   0.65: .893/.781/.868   0.75: 1.00/.594/.880 <- argmax F0.5
# The naive argmax (0.75) is a TRAP: it has zero false merges on the fixture
# but, run through the actual fit()/canon() pipeline on a realistic label
# set, merges NONE of EXP-001's named scatters (fitness/exercise/gym,
# travel/flying) — the pairwise metric misses that union-find is
# transitive (e.g. "gym" bridges "fitness" and "exercise" even when their
# direct cosine sim sits under threshold), so a threshold can look great
# pairwise while doing nothing for the actual problem this file exists to
# solve. Re-ran fit() end-to-end per threshold on a mixed label set
# (EXP-001 scatters + the acid-test near-neighbor distractors) to check
# what really merges:
#   0.60: {fitness,exercise,gym}, {travel,flying,airplanes} fully merge --
#         but so do {cat,dog}, {coffee,tea}, {doctors,lawyers}: the exact
#         near-neighbor pairs the fixture calls out as must-NOT-merge.
#   0.65: {fitness,exercise,gym} still fully merges; {cat,dog} still
#         falsely merges (the single worst offender in the fixture).
#   0.70: {cat,dog} correctly splits, {coffee,tea}/{doctors,lawyers} never
#         merge at or above .65 -- but the EXP-001 clusters only PARTIALLY
#         merge: {fitness,gym} (not exercise), {flying,airplanes} (not
#         travel). Residual false merges: {cars,driving}, {marriage,
#         wedding} -- both flagged as genuine judgment calls in the
#         fixture notes, not clear-cut errors.
#   0.75: EXP-001 clusters do not merge at all.
# Chose 0.70. Per the design mandate (see calibrate_label_space.py
# docstring), a false merge is the dangerous error direction for this
# benchmark's credibility -- it silently inflates the "mode collapse"
# headline number. 0.70 is the highest threshold that still delivers
# real, disclosed EXP-001 value (partial cluster merges) while clearing
# every unambiguous near-neighbor acid test (cat/dog, coffee/tea,
# doctors/lawyers, work/school, math/physics all correctly stay split).
# Known residual limitation, stated plainly: "exercise" and "travel"
# remain their own canonical singletons even though their near-synonyms
# merge -- this is a partial fix, not a complete one. Don't spin it.
DEFAULT_THRESHOLD = 0.70

_MODEL_NAME = "all-MiniLM-L6-v2"


class LabelSpace:
    """Canonicalizes free-text topic labels into equivalence classes.

    fit(labels) embeds each distinct normalized label with
    all-MiniLM-L6-v2, then clusters with COMPLETE-LINKAGE agglomeration:
    two clusters merge only if EVERY cross-pair is >= threshold. The
    canonical representative of a class is its most frequent member
    (ties broken by shortest string, then lexicographic) — so canon()
    output is always a label someone actually produced, never a
    synthetic centroid string, which keeps report.json human-readable.

    Why complete linkage and not union-find (single linkage): hub labels
    bridge distinct topics transitively. Observed live in EXP-003 — 'pet'
    sat >= 0.70 from both 'cat' and 'dog', union-find chained cat<->dog
    into one class, and ARI vs gold fell 0.837 -> 0.659. A false merge
    manufactures collapse evidence, which is the one bias this benchmark
    cannot afford. Complete linkage makes hub-chaining structurally
    impossible: dog joins {cat, pet} only if dog is close to BOTH.

    Degrades to an identity mapping over normalize_label if
    sentence_transformers isn't importable (see fit()) — check `.degraded`
    after fit() if callers need to know whether semantic merging actually
    happened.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.degraded = False
        self._canon_map: Dict[str, str] = {}

    def fit(self, labels: List[str]) -> "LabelSpace":
        """Build the equivalence classes. Safe to call with duplicates —
        frequency counts (for representative selection) are taken over
        the full input list, not the deduped set."""
        norm = [normalize_label(l) for l in labels]
        counts = Counter(norm)
        uniq = list(counts)
        if not uniq:
            return self

        try:
            model = self._load_model()
        except Exception as e:  # ImportError normally; broad on purpose —
            # a half-broken torch install must degrade too, not crash.
            warnings.warn(
                "LabelSpace: sentence_transformers unavailable (%r) -- "
                "degrading to identity mapping over normalize_label. "
                "Synonym scatters (fitness/exercise/gym, travel/flying) "
                "will NOT be merged; trajectory metrics revert to "
                "EXP-001 behavior. Install with: "
                "pip3 install --user sentence-transformers" % (e,),
                RuntimeWarning, stacklevel=2)
            self.degraded = True
            self._canon_map = {l: l for l in uniq}
            return self

        embeddings = model.encode(uniq, normalize_embeddings=True)

        # Complete-linkage agglomeration (see class docstring for why not
        # union-find). Greedy: repeatedly merge the closest cluster pair
        # whose WORST cross-pair similarity clears the threshold. O(n^3)
        # worst case — fine at this scale (distinct topic labels per
        # fixture or pilot sweep: tens, not thousands).
        n = len(uniq)
        sims = embeddings @ embeddings.T  # unit-normalized -> dot = cosine

        clusters: List[List[int]] = [[i] for i in range(n)]

        def linkage(a: List[int], b: List[int]) -> float:
            return min(sims[i][j] for i in a for j in b)

        while True:
            best_l, best_x, best_y = -1.0, -1, -1
            for x in range(len(clusters)):
                for y in range(x + 1, len(clusters)):
                    l = linkage(clusters[x], clusters[y])
                    if l > best_l:
                        best_l, best_x, best_y = l, x, y
            if best_l < self.threshold:
                break
            clusters[best_x].extend(clusters[best_y])
            del clusters[best_y]

        groups: Dict[int, List[str]] = {}
        for ci, members in enumerate(clusters):
            groups[ci] = [uniq[i] for i in members]

        self._canon_map = {}
        for members in groups.values():
            rep = sorted(members, key=lambda m: (-counts[m], len(m), m))[0]
            for m in members:
                self._canon_map[m] = rep
        return self

    def canon(self, label: str) -> str:
        """Map a raw label to its canonical representative. Labels not
        seen at fit() time fall back to their own normalized form (same
        behavior as degraded mode) rather than raising — a cascade run
        can legitimately produce a topic not present in the fit corpus."""
        norm = normalize_label(label)
        return self._canon_map.get(norm, norm)

    def canonize_paths(
        self, paths: Sequence[Sequence[str]]
    ) -> List[List[str]]:
        """Apply canon() elementwise to a list of label paths."""
        return [[self.canon(t) for t in p] for p in paths]

    @staticmethod
    def _load_model():
        from sentence_transformers import SentenceTransformer  # lazy
        return SentenceTransformer(_MODEL_NAME)
