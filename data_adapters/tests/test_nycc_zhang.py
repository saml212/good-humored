"""Unit + live-data tests for data_adapters/nycc_zhang.py. The live
tests read the staged SSD data (skip cleanly when unmounted)."""

import pytest

from data_adapters.nycc_zhang import (SUMMARIES_DIR, ZhangLoadStats,
                                      _merge_rows, load_ranked_groups)

live = pytest.mark.skipif(not SUMMARIES_DIR.exists(),
                          reason="SSD data not mounted")


class TestFirewall:
    def test_wrong_license_class_raises(self):
        with pytest.raises(ValueError):
            load_ranked_groups(["commercial_safe"], split="val",
                               max_candidates_per_group=5)

    def test_cap_below_three_raises(self):
        with pytest.raises(ValueError):
            load_ranked_groups(["research_only"], split="val",
                               max_candidates_per_group=2)

    def test_bad_split_raises(self):
        with pytest.raises(ValueError):
            load_ranked_groups(["research_only"], split="test")


@live
class TestLive:
    def test_val_split_loads_with_provenance(self):
        groups, stats = load_ranked_groups(
            ["research_only"], split="val", max_candidates_per_group=30)
        assert stats.contests_loaded >= 70  # 77 val contests, some may miss
        g = groups[0]
        assert g.license_class == "research_only"
        assert g.source_dataset == "nycc-zhang-votes"
        assert 2 <= len(g.candidates) <= 30
        # within-group scores are the crowd means; ordering sane
        scores = [c.score for c in g.candidates]
        assert max(scores) <= 3.0 and min(scores) >= 1.0
        # every candidate has a histogram entry
        for c in g.candidates:
            assert c.candidate_id in g.metadata["histograms"]
            hist = g.metadata["histograms"][c.candidate_id]
            assert set(hist) == {"not_funny", "somewhat_funny", "funny"}
            # histogram total tracks rater_count (merged across files)
            assert sum(hist.values()) == c.rater_count

    def test_stratified_cap_spans_range(self):
        groups, _ = load_ranked_groups(
            ["research_only"], split="val", max_candidates_per_group=30)
        g = groups[0]
        scores = sorted(c.score for c in g.candidates)
        # capped group must span top and bottom, not one tail
        assert scores[-1] - scores[0] > 0.3

    def test_multi_file_contests_merged(self):
        groups, stats = load_ranked_groups(["research_only"], split="val",
                                           max_candidates_per_group=10)
        assert stats.multi_file_contests >= 2  # 510 and 515 at minimum
        by_id = {g.metadata["contest_id"]: g for g in groups}
        assert len(by_id[510].metadata["source_files"]) == 2
