"""Loader: Oogiri-GO (`zhongshsh/CLoT-Oogiri-GO` on HuggingFace) -> RankedGroup.

license_class is hardcoded to `'research_only'` for every record this
loader produces. Read the flag below before trusting anything else here.

---

FLAGGED DISCREPANCY (verify before ever changing this to commercial_safe):
`references/corpus-sources.md`'s licensing summary table lists Oogiri-GO
as "CC-BY-NC-SA 4.0 / NO [not commercial safe]" -- but its own per-entry
writeup for item 6 qualifies that as "Not explicitly stated (inherited
from original Oogiri dataset; assume CC-BY-NC-SA)", i.e. an ASSUMPTION
about this specific HF repo, not a confirmed read of its actual license
metadata. This session independently queried the HF Hub API for this
exact repo --

    curl https://huggingface.co/api/datasets/zhongshsh/CLoT-Oogiri-GO

-- and found `"cardData": {"license": "mit", ...}` and `"tags":
["license:mit", ...]`. The dataset's own card declares MIT, which (if
taken at face value) would be commercial-safe -- the opposite of
corpus-sources.md's assumption.

This is a real, unresolved conflict between the project's license
source-of-truth document and the primary source. It is NOT resolved here.
Per this build's explicit instructions and the firewall's own
conservative-default philosophy (a contested license claim is treated as
research_only until a human resolves it), this loader stays hardcoded to
`research_only` regardless of the MIT tag. Do not change this without a
human explicitly re-verifying what that MIT tag actually covers -- a
repo-level license tag on a dataset card is sometimes a template default
rather than a verified grant, especially for a dataset built by scraping
two third-party platforms (Bokete, Zhihu -- per `references/datasets.md`)
whose own terms-of-service an uploader's self-applied MIT tag cannot
override on its own.

---

TEXT-ONLY FILTERING (per task spec: "filtering to English text-only
(I2T/T2T text riddles)"): this loader keeps ONLY `T2T` (Text-to-Text)
rows, and drops `I2T` (Image-to-Text) rows. This is a deliberate reading
of "text-only", not a literal keep-both-I2T-and-T2T pass: an I2T row's
`question` field is `null` -- its actual prompt/context IS the (not
downloaded, out of scope for this adapter) image. Without the image, an
I2T row has no usable `RankedGroup.context` at all, so including it would
mean fabricating a context, which this loader refuses to do. From the
full English file (`en.jsonl`, 23,769 rows, checked directly this
session): I2T=17,336 (excluded), T2T=6,433 (candidate pool). See
`OogiriLoadStats` for the exact counts returned alongside every load.

GROUPING: candidates share a `RankedGroup` when they share the exact same
`question` string (the T2T prompt). Of the 6,433 T2T rows: 1,099 distinct
questions, of which 1,009 have >= 2 candidates (6,343 of the 6,433 rows)
and become real `RankedGroup`s; the remaining 90 single-candidate
questions are dropped (a group of 1 carries no comparison signal, and
`to_preference_pairs` would return `[]` for it anyway -- keeping an inert
placeholder group around adds nothing and just costs a caller a null
check).

SCORE: the `star` field -- a like/popularity count on that ONE response
to that ONE prompt. This is a WITHIN-PROMPT signal, comparable only to
its own group's siblings (see `schema.py`'s design rule) -- it is
structurally not the cross-prompt popularity-bias case this project's
hard rule targets, because a `RankedGroup`'s candidates never leave the
group. It is still an engagement-adjacent metric in the weaker sense
`.claude/skills/humor-rl/SKILL.md` "Data" flags generally (a like count
reflects who saw a response and chose to react, not a controlled judge
panel) -- worth remembering when interpreting reward-model quality
trained on this data, and explicitly a WEAKER design than Oogiri-Master's
~100-independent-judges-per-response methodology (`corpus-sources.md`
item 5). This loader is Oogiri-GO, not Oogiri-Master -- Oogiri-Master has
no public pre-built release as of this writing (`corpus-sources.md`:
"Oogiri builds are local").

DOWNLOAD: `en.jsonl` is small enough (~3.7MB total) that "development
sample" and "the entire English-language text split" are the same
download -- no byte-range partial fetch is needed or done. Written to
`~/Experiments/good-humored-data/raw/oogiri-go/en.jsonl`, checked for an
existing copy FIRST (`ensure_sample` is idempotent; this session found it
already present from an earlier step in the same build and did not
re-fetch it). `cn.jsonl`, `jp.jsonl`, and `images.zip` (~1.3GB, the
referenced I2T images) are NEVER downloaded by this module -- see the
documented-not-run full-download commands below.

Full-download commands (NOT executed by this module -- copy-paste only):

    curl -sL -o ~/Experiments/good-humored-data/raw/oogiri-go/cn.jsonl \\
      https://huggingface.co/datasets/zhongshsh/CLoT-Oogiri-GO/resolve/main/cn.jsonl
    curl -sL -o ~/Experiments/good-humored-data/raw/oogiri-go/jp.jsonl \\
      https://huggingface.co/datasets/zhongshsh/CLoT-Oogiri-GO/resolve/main/jp.jsonl
    curl -sL -o ~/Experiments/good-humored-data/raw/oogiri-go/images.zip \\
      https://huggingface.co/datasets/zhongshsh/CLoT-Oogiri-GO/resolve/main/images.zip

    # Equivalent via the `datasets` library (heavy dep, guarded import
    # only -- NOT required for this loader's text-only path):
    #   from datasets import load_dataset
    #   ds = load_dataset("zhongshsh/CLoT-Oogiri-GO")

NETWORK ISOLATION FOR TESTS: the network-touching function (`ensure_sample`)
and the pure parsing function (`parse_en_jsonl`) are fully separate.
`load_ranked_groups(jsonl_path=...)` lets a caller (a test) point directly
at a tiny local fixture file and never touch the network at all.
"""

import json
import os
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from data_adapters.firewall import assert_license_class
from data_adapters.schema import Candidate, RankedGroup

DATASET_URL = (
    "https://huggingface.co/datasets/zhongshsh/CLoT-Oogiri-GO/"
    "resolve/main/en.jsonl")
SOURCE_DATASET = "oogiri-go"
LICENSE_CLASS = "research_only"  # see module docstring -- do not change
                                  # without a human re-verifying the
                                  # flagged MIT-tag discrepancy first.
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # task constraint: sample downloads capped at 50MB
MIN_GROUP_SIZE = 2  # groups of 1 carry no comparison signal; see docstring


@dataclass
class OogiriLoadStats:
    """Counts reported alongside every load, per task spec ("...with
    counts reported"). Every field is computed directly from `en.jsonl`,
    not estimated."""

    total_rows: int
    i2t_excluded: int
    t2t_rows: int
    other_type_excluded: int
    distinct_t2t_prompts: int
    multi_candidate_groups: int
    single_candidate_dropped: int
    ranked_groups_emitted: int
    candidates_emitted: int


def _default_data_dir() -> Path:
    return Path(
        os.environ.get("GOOD_HUMORED_DATA_DIR",
                       "~/Experiments/good-humored-data")).expanduser()


def ensure_sample(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Return the path to `en.jsonl`, downloading it if (and only if) it
    isn't already present. Checks the data dir FIRST, per this build's
    instructions -- never re-downloads an existing file.

    Enforces the 50MB task cap using the response's `Content-Length`
    header (checked before reading the body) AND a hard stop while
    streaming (in case the header is missing or wrong) -- refuses to
    write a partial file if the cap is exceeded either way.
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    out_dir = data_dir / "raw" / "oogiri-go"
    out_path = out_dir / "en.jsonl"
    if out_path.exists():
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "good-humored-data-adapters/1 (research)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(
                "oogiri.ensure_sample: refusing to download en.jsonl -- "
                "reported Content-Length=%s bytes exceeds this loader's "
                "%d byte cap (task constraint: sample downloads capped at "
                "50MB). The upstream file may have grown since this "
                "module was written." % (content_length, MAX_DOWNLOAD_BYTES))
        data = resp.read(MAX_DOWNLOAD_BYTES + 1)
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(
                "oogiri.ensure_sample: en.jsonl exceeded the %d byte cap "
                "mid-download; aborting, nothing written to disk." %
                MAX_DOWNLOAD_BYTES)

    out_path.write_bytes(data)
    return out_path


def parse_en_jsonl(
    jsonl_path: Union[str, Path],
    min_group_size: int = MIN_GROUP_SIZE,
) -> Tuple[List[RankedGroup], OogiriLoadStats]:
    """Pure parsing, NO network: read an already-downloaded (or fixture)
    `en.jsonl`-shaped file and build `RankedGroup`s. Every candidate's
    `license_class` is hardcoded to `research_only` (see module
    docstring).
    """
    jsonl_path = Path(jsonl_path)
    groups_by_question: Dict[str, List[dict]] = defaultdict(list)
    total_rows = i2t = t2t = other = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total_rows += 1
            t = rec.get("type")
            if t == "I2T":
                i2t += 1
                continue
            if t != "T2T":
                other += 1
                continue
            question = rec.get("question")
            if not question or not str(question).strip():
                # A T2T row with no question text has no shared context
                # to group on; per docstring this loader requires a real
                # question string, so this row is dropped rather than
                # merged into a spurious "no prompt" bucket. Counted as
                # "other" (excluded), not as a counted T2T row.
                other += 1
                continue
            t2t += 1
            groups_by_question[question].append(rec)

    groups: List[RankedGroup] = []
    single_dropped = 0
    for idx, (question, rows) in enumerate(sorted(groups_by_question.items())):
        if len(rows) < min_group_size:
            single_dropped += len(rows)
            continue
        group_source_id = "oogiri-go-t2t-%04d" % idx
        candidates = tuple(
            Candidate(
                text=r["text"].strip(),
                score=float(r["star"]),
                rater_count=None,  # Oogiri-GO reports an aggregate like
                                   # count, not a rater N -- see docstring.
                candidate_id="%s-c%d" % (group_source_id, ci),
            )
            for ci, r in enumerate(rows)
            if r.get("text") and str(r["text"]).strip()
            and r.get("star") is not None
        )
        if len(candidates) < min_group_size:
            single_dropped += len(candidates)
            continue
        groups.append(RankedGroup(
            context=question,
            candidates=candidates,
            source_dataset=SOURCE_DATASET,
            license_class=LICENSE_CLASS,
            source_id=group_source_id,
            metadata={"modality": "T2T", "language": "en"},
        ))

    stats = OogiriLoadStats(
        total_rows=total_rows,
        i2t_excluded=i2t,
        t2t_rows=t2t,
        other_type_excluded=other,
        distinct_t2t_prompts=len(groups_by_question),
        multi_candidate_groups=len(groups),
        single_candidate_dropped=single_dropped,
        ranked_groups_emitted=len(groups),
        candidates_emitted=sum(len(g.candidates) for g in groups),
    )
    return groups, stats


def load_ranked_groups(
    allowed_licenses: List[str],
    data_dir: Optional[Union[str, Path]] = None,
    jsonl_path: Optional[Union[str, Path]] = None,
    min_group_size: int = MIN_GROUP_SIZE,
) -> Tuple[List[RankedGroup], OogiriLoadStats]:
    """Public entry point. `allowed_licenses` is REQUIRED and has no
    default (per `firewall.py`'s design) -- since every record this
    loader produces is `research_only`, a caller that does not explicitly
    include `'research_only'` in `allowed_licenses` gets a loud error, not
    silently-empty results and not silently-included data.

    If `jsonl_path` is given, it is read directly with NO network call --
    this is the path tests use, pointed at a tiny local fixture file. If
    omitted, `ensure_sample(data_dir)` is called first (downloads on first
    use, reuses an existing copy on every call after).
    """
    path = Path(jsonl_path) if jsonl_path else ensure_sample(data_dir)
    groups, stats = parse_en_jsonl(path, min_group_size=min_group_size)
    groups = assert_license_class(groups, allowed=allowed_licenses)
    return groups, stats
