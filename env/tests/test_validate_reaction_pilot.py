"""Unit tests for benchmark/validate_reaction_pilot.py (EXP-023a) --
pure logic only (spearman, onset classifier, stratification); API
calls are never made here."""

import pytest

from benchmark.validate_reaction_pilot import (_stratify, is_laughter_onset,
                                               spearman)


class TestOnset:
    def test_positive_variants(self):
        for s in ("Haha that's great", "  \"LOL ok\"", "lmaooo", "😂😂",
                  "heh, nice one"):
            assert is_laughter_onset(s), s

    def test_negative_variants(self):
        for s in ("That's clever!", "Oh nice.", "I love it",
                  "Solid entry, very droll"):
            assert not is_laughter_onset(s), s

    def test_late_laughter_not_onset(self):
        assert not is_laughter_onset("That's a good one, haha totally")


class TestSpearman:
    def test_perfect(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_inverted(self):
        assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_ties_average_ranks(self):
        r = spearman([1, 1, 2, 3], [1, 2, 3, 4])
        assert 0.5 < r < 1.0

    def test_constant_returns_none(self):
        assert spearman([1, 1, 1], [1, 2, 3]) is None

    def test_short_raises(self):
        with pytest.raises(ValueError):
            spearman([1, 2], [1, 2])


class TestStratify:
    def test_top_mid_bottom(self):
        rows = [{"caption": str(i), "mean": float(i), "votes": 1}
                for i in range(100)]
        out = _stratify(rows)
        assert len(out) == 24
        means = [r["mean"] for r in out]
        assert max(means) == 99.0 and min(means) == 0.0
        mid = sorted(means)[8:16]
        assert 30 < sum(mid) / len(mid) < 70  # middle band present
