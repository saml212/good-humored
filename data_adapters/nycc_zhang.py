"""NYCC-Zhang crowd-rating adapter -- the 292M-vote corpus (research_only).

Loads the nextml/caption-contest-data per-contest summary CSVs staged at
`~/Experiments/good-humored-data/raw/nycc-full/raw/nextml-summaries/
summaries/` (SSD-backed; see MANIFEST.md) into `RankedGroup`s: one group
per contest, one `Candidate` per caption with the crowd's mean rating as
`score` and the vote count as `rater_count`.

License: the vote data is CC-BY-NC-4.0 (Zhang et al., NeurIPS 2024 D&B)
-- hard-classed `research_only` here; `firewall.assert_license_class`
is the only gate between this data and any trainer, per the package's
standing structural rule. Commercial use is a PENDING Sam decision
(STATE.md decisions list); nothing in this module may be relaxed until
that call is made.

Within-group-only comparability holds naturally: every caption in a
group was rated on the SAME cartoon by the same crowd process, which is
exactly the popularity-bias-free comparison `schema.py`'s design rule
demands. Cross-contest means are still never comparable (different
crowds, different cartoons) -- the one-group-per-contest shape encodes
that structurally.

Multi-methodology contests (a handful were run under two NEXT sampling
algorithms, e.g. LilUCB + RoundRobin, whose per-caption means agree
only at rho~0.2 -- the EXP-023a audit finding): all files for a contest
are merged by VOTES-WEIGHTED mean, identical to the fix shipped in
benchmark/validate_reaction_pilot.py, and the file list is recorded in
group metadata.

Rating DISTRIBUTIONS (the quantile-emulator training signal): the CSVs
carry per-caption `not_funny`/`somewhat_funny`/`funny` counts.
`Candidate` has no metadata field by design, so histograms live in
`RankedGroup.metadata["histograms"]` as a candidate_id-keyed dict --
the loader guarantees candidate_id is set on every candidate for this
purpose.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .firewall import assert_license_class
from .schema import Candidate, RankedGroup

DATA_ROOT = (Path.home() / "Experiments" / "good-humored-data" / "raw"
             / "nycc-full")
SUMMARIES_DIR = DATA_ROOT / "raw" / "nextml-summaries" / "summaries"
SPLITS_PATH = DATA_ROOT / "splits.json"
DESCRIPTIONS_PATH = DATA_ROOT / "contest_descriptions.json"

LICENSE_CLASS = "research_only"  # CC-BY-NC-4.0 -- see module docstring
SOURCE_DATASET = "nycc-zhang-votes"
MIN_VOTES = 5  # captions with fewer independent ratings than this carry
               # more noise than signal; pinned here, overridable


@dataclass
class ZhangLoadStats:
    contests_seen: int = 0
    contests_loaded: int = 0
    captions_loaded: int = 0
    captions_dropped_low_votes: int = 0
    multi_file_contests: int = 0


def _contest_files(contest_id: int) -> List[Path]:
    matches = sorted(SUMMARIES_DIR.glob("%d*.csv" % contest_id))
    return [m for m in matches
            if m.stem == str(contest_id)
            or m.stem.startswith("%d_" % contest_id)]


def _merge_rows(files: List[Path], min_votes: int,
                stats: ZhangLoadStats) -> Dict[str, Dict]:
    """Votes-weighted merge across a contest's summary files, keyed by
    lowercased caption. Histogram counts sum across files."""
    acc: Dict[str, Dict] = {}
    for path in files:
        with open(path) as f:
            for row in csv.DictReader(f):
                caption = row.get("caption", "").strip()
                if not caption:
                    continue
                try:
                    mean = float(row["mean"])
                    votes = int(row["votes"])
                    hist = {k: int(row[k]) for k in
                            ("not_funny", "somewhat_funny", "funny")}
                except (KeyError, ValueError):
                    continue
                key = caption.lower()
                if key in acc:
                    a = acc[key]
                    total = a["votes"] + votes
                    if total:
                        a["mean"] = (a["mean"] * a["votes"]
                                     + mean * votes) / total
                    a["votes"] = total
                    for k in hist:
                        a["hist"][k] += hist[k]
                else:
                    acc[key] = {"caption": caption, "mean": mean,
                                "votes": votes, "hist": hist}
    kept = {}
    for key, rec in acc.items():
        if rec["votes"] < min_votes:
            stats.captions_dropped_low_votes += 1
            continue
        kept[key] = rec
    return kept


def load_ranked_groups(
    allowed_license_classes: List[str],
    split: Optional[str] = None,
    min_votes: int = MIN_VOTES,
    max_candidates_per_group: Optional[int] = None,
) -> tuple:
    """Load NYCC-Zhang contests as `RankedGroup`s.

    `allowed_license_classes` is MANDATORY and positional, same contract
    as every other loader in this package: the built groups are routed
    through `firewall.assert_license_class`, which RAISES ValueError if
    research_only is not in the allowed list -- identical semantics to
    nycc.py/oogiri.py (audit 2026-08-06 fix; the earlier silent-empty
    behavior diverged from the package contract).

    `split`: None (all), "train", or "val" -- the by-contest
    `contest_id % 5` split pinned in splits.json.

    `max_candidates_per_group`: optional cap; when set, candidates are
    kept stratified by mean rating (top/mid/bottom thirds) so a capped
    group still spans the quality range rather than truncating to one
    tail. None loads everything (2.3M captions across 385 contests).

    Returns `(groups, stats)`.
    """
    stats = ZhangLoadStats()

    if max_candidates_per_group is not None and max_candidates_per_group < 3:
        raise ValueError(
            "max_candidates_per_group must be >= 3 (stratified top/mid/"
            "bottom needs one slot per band); got %r"
            % (max_candidates_per_group,))
    if split not in (None, "train", "val"):
        raise ValueError("split must be None, 'train', or 'val'; got %r"
                         % (split,))
    splits = json.loads(SPLITS_PATH.read_text())
    if split is None:
        contest_ids = sorted(splits["contest_ids"]["train"]
                             + splits["contest_ids"]["val"])
    else:
        contest_ids = sorted(splits["contest_ids"][split])
    descriptions = json.loads(DESCRIPTIONS_PATH.read_text())

    groups = []
    for cid in contest_ids:
        stats.contests_seen += 1
        files = _contest_files(cid)
        if not files:
            continue
        if len(files) > 1:
            stats.multi_file_contests += 1
        merged = _merge_rows(files, min_votes, stats)
        if len(merged) < 2:  # a group of one compares nothing
            continue
        records = sorted(merged.values(), key=lambda r: -r["mean"])
        if max_candidates_per_group and len(records) > max_candidates_per_group:
            third = max_candidates_per_group // 3
            mid = (len(records) - third) // 2
            records = (records[:third] + records[mid:mid + third]
                       + records[-(max_candidates_per_group - 2 * third):])
        desc = descriptions.get(str(cid))
        context = ("New Yorker caption contest cartoon: %s" % desc
                   if desc else
                   "New Yorker caption contest #%d (no description "
                   "available)" % cid)
        candidates, histograms = [], {}
        for i, rec in enumerate(records):
            candidate_id = "%d-%d" % (cid, i)
            candidates.append(Candidate(
                text=rec["caption"], score=rec["mean"],
                rater_count=rec["votes"], candidate_id=candidate_id))
            histograms[candidate_id] = rec["hist"]
        groups.append(RankedGroup(
            context=context, candidates=tuple(candidates),
            source_dataset=SOURCE_DATASET, license_class=LICENSE_CLASS,
            source_id="contest-%d" % cid, group_id="nycc-zhang-%d" % cid,
            metadata={"contest_id": cid, "split": split or "all",
                      "source_files": [f.name for f in files],
                      "has_description": bool(desc),
                      "histograms": histograms}))
        stats.contests_loaded += 1
        stats.captions_loaded += len(candidates)
    groups = assert_license_class(groups, allowed=allowed_license_classes)
    return groups, stats
