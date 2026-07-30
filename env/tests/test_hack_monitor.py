"""Unit tests for env/hack_monitor.py (EXP-022) -- synthetic vectors,
hand-computable, no scoring of the real fixture (real numbers come from
the validator, same discipline as every other validate_* pair)."""

import pytest

from env.hack_monitor import (DecouplingMonitor, Z_CAP, _median,
                              recall_at_fp)

NAMES = ("novelty", "self_repetition", "diversity", "comprehensibility")

# Baseline: tight around (0, 0, 1, 0.3) with small jitter so MAD > 0.
BASELINE = [
    [0.00, 0.00, 1.00, 0.30],
    [0.02, -0.02, 0.95, 0.30],
    [-0.02, 0.02, 1.05, 0.28],
    [0.01, -0.01, 0.98, 0.32],
    [-0.01, 0.01, 1.02, 0.30],
    [0.00, 0.00, 1.01, 0.29],
]


def _fitted():
    m = DecouplingMonitor(NAMES)
    m.fit(BASELINE)
    return m


class TestMedian:
    def test_odd_even(self):
        assert _median([3, 1, 2]) == 2
        assert _median([4, 1, 2, 3]) == 2.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _median([])


class TestMonitor:
    def test_needs_three_components(self):
        with pytest.raises(ValueError):
            DecouplingMonitor(("a", "b"))

    def test_unfitted_raises(self):
        with pytest.raises(ValueError):
            DecouplingMonitor(NAMES).z([0, 0, 0, 0])

    def test_baseline_scores_stay_below_hack_scores(self):
        # Baseline points CAN exceed the z_min gate (a tight
        # distribution's own tails reach ~2.7 robust-z); what matters is
        # SEPARATION -- FP control belongs to the tau sweep. Every
        # baseline score must sit well under a single-dimension hack's.
        m = _fitted()
        hack = m.score([0.00, 0.00, 3.00, 0.30])
        for v in BASELINE:
            assert m.score(v) < hack / 2

    def test_single_dimension_hack_flagged(self):
        m = _fitted()
        hacked = [0.00, 0.00, 3.00, 0.30]  # diversity inflated alone
        assert m.score(hacked) > 5.0
        d = m.decoupling(hacked)
        assert max(d, key=d.get) == "diversity"

    def test_cratered_dimension_also_flagged(self):
        m = _fitted()
        cratered = [0.00, 0.00, 1.00, -1.00]  # comprehensibility craters
        assert m.score(cratered) > 5.0

    def test_genuine_improvement_agreement_not_flagged_but_naive_fires(self):
        m = _fitted()
        # Everything rises together by ~the same robust-z amount.
        better = [0.10, 0.10, 1.35, 0.42]
        zs = m.z(better)
        assert min(zs) > 1.0  # all dimensions moved
        assert m.score(better) < m.naive_score(better)
        # naive sees a big outlier; decoupling sees agreement
        assert m.naive_score(better) >= Z_CAP * 0.3

    def test_z_cap_bounds_constant_component(self):
        m = DecouplingMonitor(NAMES)
        m.fit([[0.0, 0.0, 1.0, 0.3]] * 5)  # MAD == 0 everywhere
        zs = m.z([1.0, 0.0, 1.0, 0.3])
        assert zs[0] == Z_CAP  # floored scale -> capped, not inf
        assert all(abs(z) <= Z_CAP for z in zs)

    def test_group_score_is_max_over_completions(self):
        m = _fitted()
        group = [BASELINE[0], [0.0, 0.0, 3.0, 0.3]]
        assert m.group_score(group) == m.score(group[1])


class TestRecallAtFp:
    def test_perfect_separation(self):
        r = recall_at_fp([0.1] * 10, [5.0] * 5, max_fp=0.10)
        assert r["recall"] == 1.0
        assert r["fp"] <= 0.10

    def test_no_separation(self):
        r = recall_at_fp([1.0] * 10, [1.0] * 5, max_fp=0.10)
        assert r["recall"] == 0.0

    def test_fp_budget_respected(self):
        genuine = [float(i) for i in range(10)]  # 0..9
        # 10% FP budget = the TOP genuine (9.0) is allowed to be flagged,
        # so the threshold sits just above the second-highest (8.0) and
        # both hacks clear it at fp exactly 0.10.
        r = recall_at_fp(genuine, [8.5, 9.5], max_fp=0.10)
        assert r["fp"] == 0.10
        assert r["recall"] == 1.0
        # Zero FP budget: threshold above ALL genuine -> only 9.5 clears.
        r0 = recall_at_fp(genuine, [8.5, 9.5], max_fp=0.0)
        assert r0["fp"] == 0.0
        assert r0["recall"] == 0.5
