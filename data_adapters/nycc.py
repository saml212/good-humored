"""Loader: New Yorker Caption Contest (`jmhessel/newyorker_caption_contest`
on HuggingFace) -> RankedGroup, text-only (`ranking` config).

LICENSE: `license_class='commercial_safe'`. Per `references/corpus-sources.md`
(the firewall's source-of-truth document), NYCC's licensing summary row
reads "CC-BY-4.0 / Yes* / Requires attribution; images may have
restrictions" -- `Yes*` in that table's own legend means "Safe with
attribution/compliance with source ToS". This session independently
confirmed the HF repo's own hub metadata --

    curl https://huggingface.co/api/datasets/jmhessel/newyorker_caption_contest

-- returns `"cardData": {"license": ["cc-by-4.0"]}` and
`"tags": ["license:cc-by-4.0"]`, consistent with both corpus-sources.md and
`references/datasets.md`'s independent entry for this same dataset. No
discrepancy found here (contrast `oogiri.py`, where there is one).

ATTRIBUTION CAVEAT (not encoded as a third `license_class` value -- the
schema only has two -- but recorded on every record's `metadata` so it is
never lost): CC-BY-4.0 requires attribution, and corpus-sources.md
explicitly separately flags that "images may have different licensing"
than the caption text. This loader never touches image bytes (see
"TEXT-ONLY EXTRACTION" below), so the image-licensing caveat does not
apply to what this module emits, but the attribution requirement on the
text itself still does -- `metadata["attribution_required"] = True` and
`metadata["attribution"]` (a citation string) are set on every
`RankedGroup` this loader returns so a downstream trainer/publisher
cannot lose that obligation.

TEXT-ONLY EXTRACTION: the `ranking` config's `image` field is a signed
URL + dimensions (metadata, not raw pixel bytes), and this loader never
follows it. This matches `references/README.md`'s own stated NYCC
text-only approach ("Extract caption text only... images separately if
needed") and corpus-sources.md item 7's "Text-only approach: Extract
caption text only". Discarding the cartoon image does lose real
information -- corpus-sources.md's own "Quality issues" note for this
entry says exactly that ("small relative to text-only corpora; requires
image handling for full dataset") -- this loader accepts that tradeoff
deliberately, not silently.

SHAPE: each row in the `ranking` config IS already one pairwise
comparison -- `caption_choices` is a 2-element list, `label` ("A" or "B")
names the crowd-selected funnier caption. This loader emits ONE
`RankedGroup` per row: `context` is built from the scene-description
fields HF provides without the image (`image_location`,
`image_description`, `image_uncanny_description`, `questions`), and the
two `caption_choices` become two `Candidate`s -- the labeled winner scored
1.0, the other scored 0.0. `to_preference_pairs()` on a 2-candidate group
with a 1.0/0.0 split yields exactly one `PreferencePair` per row, which is
precisely NYCC's own "ranking" task shape (this is not a repurposing of a
different task into a pairwise one; it already is a pairwise task).

`rater_count` is left `None`: this HF config exposes only the FINAL
crowd-selected label, not the underlying vote tally, so no real rater
count is available to attach.

DOWNLOAD, stdlib-only path (default, no `datasets`/`pandas` dependency):
HuggingFace's public "datasets-server" JSON API
(`https://datasets-server.huggingface.co/rows?...`) returns already-parsed
rows as JSON over plain HTTPS -- no parquet/arrow parsing needed, so this
stays pure-stdlib (`urllib.request` + `json`). A capped number of rows
(`limit`, default 300) is fetched and cached to
`~/Experiments/good-humored-data/raw/nycc/<config>_<split>_sample.json`;
an existing cache file is reused, never re-fetched. Row JSON (no image
bytes) runs a few KB each, so even `limit=300` stays well under the 100MB
task cap -- this is checked defensively at fetch time regardless.

DOWNLOAD, `datasets`-library path (heavy dep, guarded import, NOT used by
default, NOT run by this session -- `datasets` is not installed in this
environment): if a full local parquet-backed copy is ever needed,

    HF_HOME=~/Experiments/good-humored-data/hf-cache python3 -c "
    from datasets import load_dataset
    ds = load_dataset('jmhessel/newyorker_caption_contest', 'ranking')
    "

`HF_HOME` is set (via `_ensure_hf_home()`, called before any guarded
`datasets` import in this module) so that IF this path is ever exercised,
its cache lands on the persistent data volume, per this build's
instructions -- never the default `~/.cache/huggingface`.

NETWORK ISOLATION FOR TESTS: `parse_rows(rows)` is pure (a list of
already-fetched row dicts in, `RankedGroup`s out) and never touches the
network; tests call it directly against a tiny fixture list. `fetch_rows`
is the only network-touching function.
"""

import json
import os
import urllib.request
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from data_adapters.firewall import assert_license_class
from data_adapters.schema import Candidate, RankedGroup

API_BASE = "https://datasets-server.huggingface.co/rows"
DATASET_ID = "jmhessel/newyorker_caption_contest"
SOURCE_DATASET = "newyorker_caption_contest"
LICENSE_CLASS = "commercial_safe"  # see module docstring -- CC-BY-4.0, confirmed
ATTRIBUTION = (
    "Jain et al., \"Cartoons and Captions,\" and Hessel et al., \"Do Androids "
    "Laugh at Electric Sheep?\" (NYCC dataset), jmhessel/newyorker_caption_contest "
    "on HuggingFace, CC-BY-4.0.")
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024  # task constraint: 100MB cap
ROWS_PER_PAGE = 100  # HF datasets-server's own per-page cap


@dataclass
class NyccLoadStats:
    """Counts reported alongside every load."""

    rows_fetched: int
    rows_with_two_choices: int
    rows_skipped_malformed: int
    ranked_groups_emitted: int


def _default_data_dir() -> Path:
    return Path(
        os.environ.get("GOOD_HUMORED_DATA_DIR",
                       "~/Experiments/good-humored-data")).expanduser()


def _ensure_hf_home(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Point HF_HOME at the persistent data volume (task instruction: 'Use
    HF_HOME=.../hf-cache for any download'). Only matters for the guarded
    `datasets`-library path documented above -- the default stdlib rows-API
    path below never touches HF_HOME or the `datasets` cache at all -- but
    is set unconditionally so nothing accidentally falls back to
    `~/.cache/huggingface` if that path is ever exercised later."""
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    hf_home = data_dir / "hf-cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    return hf_home


def fetch_rows(
    config: str = "ranking",
    split: str = "train",
    limit: int = 300,
    data_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Network-touching. Fetch up to `limit` rows of `config`/`split` via
    HF's datasets-server JSON rows API, paginating in blocks of
    `ROWS_PER_PAGE`. Caches the raw fetched rows to disk and reuses the
    cache on subsequent calls (checked first, never re-fetched once
    present) -- consistent with every other loader's "look in the data
    dir first" behavior.
    """
    _ensure_hf_home(data_dir)
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    cache_dir = data_dir / "raw" / "nycc"
    cache_path = cache_dir / ("%s_%s_sample.json" % (config, split))
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        return cached[:limit]

    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    total_bytes = 0
    offset = 0
    while len(rows) < limit:
        page_len = min(ROWS_PER_PAGE, limit - len(rows))
        qs = urllib.parse.urlencode({
            "dataset": DATASET_ID, "config": config, "split": split,
            "offset": offset, "length": page_len,
        })
        url = "%s?%s" % (API_BASE, qs)
        req = urllib.request.Request(
            url, headers={"User-Agent": "good-humored-data-adapters/1 (research)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read(MAX_DOWNLOAD_BYTES + 1)
        total_bytes += len(body)
        if total_bytes > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(
                "nycc.fetch_rows: cumulative download exceeded this "
                "loader's %d byte cap (task constraint: sample downloads "
                "capped at 100MB) after %d rows; aborting." %
                (MAX_DOWNLOAD_BYTES, len(rows)))
        payload = json.loads(body)
        page_rows = [r["row"] for r in payload.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += page_len
        if offset >= payload.get("num_rows_total", offset):
            break

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return rows[:limit]


def _build_context(row: Dict[str, Any]) -> str:
    parts = []
    loc = row.get("image_location")
    if loc:
        parts.append("Scene: %s." % loc)
    desc = row.get("image_description")
    if desc:
        parts.append(desc)
    uncanny = row.get("image_uncanny_description")
    if uncanny:
        parts.append("Uncanny element: %s" % uncanny)
    questions = row.get("questions") or []
    if questions:
        parts.append("Question: %s" % " / ".join(questions))
    return " ".join(parts).strip()


def parse_rows(rows: List[Dict[str, Any]]) -> tuple:
    """Pure, NO network: turn already-fetched `ranking`-config row dicts
    into `RankedGroup`s. Returns `(groups, stats)`.

    A row is skipped (counted in `rows_skipped_malformed`) if it lacks
    exactly 2 `caption_choices`, lacks a `label` in `{"A", "B"}`, or has no
    usable context text after `_build_context` -- these are defensive
    checks against a malformed or future-schema-changed row, not expected
    to fire on the real dataset.
    """
    groups: List[RankedGroup] = []
    two_choice = 0
    skipped = 0

    for i, row in enumerate(rows):
        choices = row.get("caption_choices")
        label = row.get("label")
        if not isinstance(choices, list) or len(choices) != 2:
            skipped += 1
            continue
        if label not in ("A", "B"):
            skipped += 1
            continue
        two_choice += 1
        context = _build_context(row)
        if not context:
            skipped += 1
            two_choice -= 1
            continue

        winner_idx = 0 if label == "A" else 1
        source_id = row.get("instance_id") or ("nycc-ranking-row-%d" % i)
        candidates = tuple(
            Candidate(
                text=str(text).strip(),
                score=1.0 if ci == winner_idx else 0.0,
                rater_count=None,
                candidate_id="%s-c%d" % (source_id, ci),
            )
            for ci, text in enumerate(choices)
            if str(text).strip()
        )
        if len(candidates) < 2:
            skipped += 1
            two_choice -= 1
            continue

        contest_number = row.get("contest_number")
        groups.append(RankedGroup(
            context=context,
            candidates=candidates,
            source_dataset=SOURCE_DATASET,
            license_class=LICENSE_CLASS,
            source_id=str(source_id),
            group_id=str(contest_number) if contest_number is not None else None,
            metadata={
                "attribution_required": True,
                "attribution": ATTRIBUTION,
                "winner_source": row.get("winner_source"),
                "note": "Image bytes discarded; context is text-only "
                        "scene description (see module docstring).",
            },
        ))

    stats = NyccLoadStats(
        rows_fetched=len(rows),
        rows_with_two_choices=two_choice,
        rows_skipped_malformed=skipped,
        ranked_groups_emitted=len(groups),
    )
    return groups, stats


def load_ranked_groups(
    allowed_licenses: List[str],
    config: str = "ranking",
    split: str = "train",
    limit: int = 300,
    data_dir: Optional[Union[str, Path]] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
):
    """Public entry point. `allowed_licenses` is REQUIRED (per
    `firewall.py`) -- every record this loader emits is `commercial_safe`,
    so a caller must explicitly include that class to get results back
    (a caller who only wants `research_only` data gets a loud error here,
    not silently-empty results).

    If `rows` is given (a list of already-fetched row dicts), this
    function makes NO network call at all -- this is the path tests use.
    Otherwise `fetch_rows(config, split, limit, data_dir)` is called
    first.
    """
    if rows is None:
        rows = fetch_rows(config=config, split=split, limit=limit,
                          data_dir=data_dir)
    groups, stats = parse_rows(rows)
    groups = assert_license_class(groups, allowed=allowed_licenses)
    return groups, stats
