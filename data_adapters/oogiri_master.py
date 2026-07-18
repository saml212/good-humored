"""Loader: Oogiri-Corpus (the RankedGroup-shaped artifact from Murakami et
al., "Oogiri-Master: Benchmarking Humor Understanding via Oogiri," arXiv:
2512.21494) -> RankedGroup.

**THIS MODULE HAS NO NETWORK-FETCH FUNCTION, UNLIKE `oogiri.py`/`nycc.py`,
BECAUSE NO PUBLIC COPY OF THE DATA EXISTS TO FETCH.** Read the acquisition
section below before assuming `ensure_sample()`-style auto-download is
possible here -- it deliberately raises rather than silently no-op'ing or
guessing at a URL.

---

ACQUISITION STATUS (independently re-verified this session, 2026-07-17 --
not just trusting `references/corpus-sources.md`/`references/datasets.md`,
per this build's task spec):

  - Paper: confirmed real. arXiv:2512.21494, "Oogiri-Master: Benchmarking
    Humor Understanding via Oogiri," Murakami, Kamigaito, Takamura,
    Okumura, 2025 (arXiv YYMM=2512 -> submitted December 2025). Verified
    directly against the arXiv abstract page, the full HTML rendering
    (arxiv.org/html/2512.21494v1), and the PDF -- not secondhand.
  - Dataset claim verified verbatim in the paper's own text: "each prompt
    is paired with approximately 100 diverse candidate responses that are
    rated for funniness by approximately 100 human judges" who vote
    "without access to others' ratings" -- this IS a real quote from the
    paper, not an embellishment introduced by this project's lit review.
  - BUT the mechanism is a blind, simultaneous APPROVAL-VOTE scheme
    ("users vote for the responses that they find funny among all
    submissions"), not a dense judge x candidate score matrix -- and the
    public CSV schema (below) exposes only the AGGREGATE `vote_count` per
    response, never individual judges' raw votes. Structurally this is
    the same shape as Oogiri-GO's `star` field (an aggregate count per
    response, comparable only within its own prompt group -- see
    `oogiri.py`/`schema.py`). The methodological improvement over
    Oogiri-GO is in the COLLECTION protocol (blind/simultaneous voting,
    not a live incrementing like-counter visible to later voters), not in
    what the released CSV format actually contains. State this precisely
    when this adapter's data is ever used -- "popularity-bias-free
    collection, aggregate-count release," not "individual multi-judge
    rating matrix."
  - Exact numbers, verified against the paper's own dataset-construction
    section: Oogiri-Corpus = 908 prompts, 82,536 total (prompt, response)
    rows, ~96 responses/prompt on average, ~172 VOTES PER PROMPT on
    average (i.e. ~172 total votes spread across that prompt's ~96
    responses -- NOT ~172 votes per individual response; per-response
    vote_count is typically much sparser than the "~100 judges" framing
    might suggest on its own).
  - Oogiri-Master.csv (600 rows, MCQA task types binary_diff/binary_same/
    triple/quad/binary_classification, positive label = per-prompt top-3
    by vote_count, negative = bottom-3) is a DIFFERENT, DERIVED artifact
    from Oogiri-Corpus -- an LLM-benchmark eval format, not itself a
    flat ranked-candidates table. This module does NOT parse
    Oogiri-Master.csv (out of scope: it is not a RankedGroup source
    without an extra transformation this build was not asked to build);
    it parses Oogiri-Corpus.csv, the artifact that actually matches
    "prompt + N rated candidates."
  - Oogiri-Corpus.csv schema, verified against the builder repo's README
    (github.com/CyberAgentAILab/oogiri-dataset-builder, fetched directly
    this session): columns `id, prompt_id, prompt, response, vote_count`.
    This module's `EXPECTED_COLUMNS` below matches that exactly.
  - Source platform: Bokete (https://bokete.jp/), a Japanese Oogiri
    website -- confirmed from the paper's own text. Data is
    JAPANESE-LANGUAGE; the builder pipeline does no translation. A model
    or reward function trained on this data will not transparently
    generalize to English humor without separate translation/adaptation
    work (same caveat `references/datasets.md` already flags for this
    dataset).
  - **NO PUBLIC RELEASE OF THE DATA EXISTS ANYWHERE, as of this
    verification pass.** Checked directly, not assumed:
      * GitHub repo contents (`gh api repos/CyberAgentAILab/
        oogiri-dataset-builder/contents/`): only source code
        (`.github/`, `.gitignore`, `.python-version`, `CITATION.cff`,
        `LICENSE`, `README.md`, `pyproject.toml`, `src/`, `uv.lock`) --
        no `data/`, no bundled CSV.
      * GitHub Releases and Tags (`gh api .../releases`, `.../tags`):
        both empty lists. No release artifact exists to download.
      * HuggingFace Hub API search (`GET /api/datasets?search=oogiri`,
        `?search=murakami`, `?author=CyberAgentAILab`): no Oogiri-Master
        or Oogiri-Corpus dataset repo found. (`zhongshsh/CLoT-Oogiri-GO`
        -- the DIFFERENT, already-adapted dataset -- is the only Oogiri
        hit; a couple of unrelated third-party Bokete scrapes by an
        uninvolved HF user "Joctor" also turn up but are NOT this paper's
        dataset, carry no verified license, and are out of scope here --
        not adopted as a substitute.)
      * The builder repo's own README states this explicitly: "The
        generated outputs (Oogiri-Master.csv, Oogiri-Corpus.csv,
        user_preference/*) are produced locally by running the pipeline;
        this repository does not redistribute them."

LICENSE (verified, not guessed): the paper's Ethics Statement states,
verbatim (fetched from arxiv.org/html/2512.21494 this session): "Oogiri-
Corpus and Oogiri-Master will be made available under the CC BY-NC-SA 4.0
license." This is a FUTURE-TENSE promise about the AUTHORS' OWN eventual
release -- it is not automatically the license of data a third party (this
project) would produce by independently running the builder pipeline
against the live Bokete site. The builder repo's own README is explicit
about this gap: "The MIT license on this code does not transfer to the
data. If you publish data derived from the crawled pages, you are
responsible for complying with the source site's copyright notice and any
applicable terms." Bokete's own ToS/robots.txt permissions were NOT
independently verified by this session (out of scope: this module does
not scrape). Net effect: even in the best case (the paper authors' own
eventual release) the license is CC-BY-NC-SA-4.0, i.e. non-commercial --
so `LICENSE_CLASS` below is `research_only` with no ambiguity on that
axis; the additional, unresolved question is whether a THIRD PARTY's own
scrape would even legitimately carry that same license at all (it may
instead be governed solely by Bokete's terms, which is a stricter, murkier
position). Per this project's firewall's own conservative-default
philosophy (`oogiri.py`'s module docstring: "a contested license claim is
treated as research_only until a human resolves it"), this module is
hardcoded to `research_only` and MUST NOT be changed to `commercial_safe`
without Sam explicitly resolving both (a) whether the CC-BY-NC-SA-4.0
promise ever materializes and (b) whether Bokete's ToS permits
redistribution of a self-run scrape at all.

**No download was performed by this session.** Per this build's hard
constraints ("if the license is ambiguous or restrictive, DO NOT
download"), and because there is in fact nothing hosted anywhere to
download (confirmed above, not merely unattempted), this module ships
with NO acquisition path. `ensure_sample()` exists only as a mirror of
`oogiri.py`'s shape and immediately raises, explaining why, rather than
silently no-op'ing or attempting a scrape this build was never authorized
to run. The only way to obtain `Oogiri-Corpus.csv` today is for a human
(Sam) to run the builder pipeline
(`uv run python -m oogiri_dataset_builder pipeline --output-dir ./out`)
themselves, having independently resolved the license question above, and
then point this module's `load_ranked_groups(csv_path=...)` at the
resulting file.

---

CSV PARSING: pure stdlib `csv.DictReader`, no pandas. `EXPECTED_COLUMNS`
is checked against the file's actual header on every parse -- a schema
drift (e.g. the upstream builder renaming `vote_count`) fails loudly with
a clear message rather than silently producing empty/wrong groups.

GROUPING: candidates share a `RankedGroup` when they share the same
`prompt_id` (not raw `prompt` text -- `prompt_id` is the builder's actual
stable key; two rows could in principle carry near-identical prompt text
under different ids, and grouping on the guaranteed-unique id avoids that
edge case). `RankedGroup.context` is the `prompt` text of the group's
first row (rows within a group are expected, not just hoped, to share
identical prompt text -- see `OogiriMasterLoadStats.prompt_text_mismatch`
for a count of groups where that expectation was violated; such groups
still use the first row's prompt text as context, not an error, since a
single inconsistent duplicate should not silently drop an otherwise-valid
group).

SCORE: `vote_count`, a WITHIN-PROMPT aggregate exactly like Oogiri-GO's
`star` (see module docstring above and `schema.py`'s "CRITICAL DESIGN
RULE") -- never compared across groups. `rater_count` is left `None`: the
CSV does not report how many judges were actually shown any given
response (only how many voted for it), so there is no real per-response
denominator to attach (see acquisition note above on why "~100 judges"
is a per-prompt approximation from the paper's prose, not a verified
per-response count in the schema).

NETWORK ISOLATION FOR TESTS: `parse_oogiri_corpus_csv(csv_path=...)` is
pure and never touches the network -- tests point it directly at
`tests/fixtures/oogiri_master_corpus_sample.csv`, a hand-built fixture
matching the verified schema exactly (synthetic data, not scraped).
"""

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from data_adapters.firewall import assert_license_class
from data_adapters.schema import Candidate, RankedGroup

SOURCE_DATASET = "oogiri-corpus"  # the specific CSV this module parses;
                                   # see module docstring for why this
                                   # differs from the module's filename
                                   # (named for the paper's overall title,
                                   # "Oogiri-Master").
LICENSE_CLASS = "research_only"  # see module docstring -- do not change
                                  # without a human resolving BOTH the
                                  # CC-BY-NC-SA-4.0-promise-vs-actual-
                                  # release gap AND the Bokete ToS
                                  # question first.
EXPECTED_COLUMNS = ("id", "prompt_id", "prompt", "response", "vote_count")
MIN_GROUP_SIZE = 2  # groups of 1 carry no comparison signal; matches
                     # oogiri.py's MIN_GROUP_SIZE convention.


@dataclass
class OogiriMasterLoadStats:
    """Counts reported alongside every load, mirroring
    `oogiri.OogiriLoadStats`'s shape."""

    total_rows: int
    malformed_rows_skipped: int
    distinct_prompts: int
    prompt_text_mismatch: int
    multi_candidate_groups: int
    single_candidate_dropped: int
    ranked_groups_emitted: int
    candidates_emitted: int


def _default_data_dir() -> Path:
    return Path(
        os.environ.get("GOOD_HUMORED_DATA_DIR",
                       "~/Experiments/good-humored-data")).expanduser()


def ensure_sample(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Deliberately NOT a working downloader. Unlike `oogiri.ensure_sample`
    / `nycc.fetch_rows`, there is no URL to fetch: no HuggingFace dataset,
    no GitHub release asset, no project-page download link exists for
    Oogiri-Corpus.csv as of this module's verification pass (see module
    docstring's ACQUISITION STATUS section for exactly what was checked).

    Raises immediately and unconditionally, naming the actual acquisition
    path (running the builder pipeline yourself, after resolving the
    license question), rather than silently returning nothing, guessing a
    URL, or attempting to scrape Bokete directly -- this module was never
    authorized to do that and the license/ToS situation is not resolved.
    """
    raise RuntimeError(
        "oogiri_master.ensure_sample: there is no public Oogiri-Corpus.csv "
        "to download -- verified this session (no HF dataset, no GitHub "
        "release/tag/asset on CyberAgentAILab/oogiri-dataset-builder, "
        "builder repo's own README states it does not redistribute "
        "outputs). The only acquisition path is running the builder "
        "pipeline yourself (github.com/CyberAgentAILab/"
        "oogiri-dataset-builder: `uv run python -m oogiri_dataset_builder "
        "pipeline --output-dir ./out`) against the live Bokete site, "
        "which requires independently resolving (a) whether the paper's "
        "promised CC-BY-NC-SA-4.0 license actually covers a self-run "
        "scrape and (b) Bokete's own ToS/robots.txt permissions -- see "
        "this module's docstring. This function will not do that "
        "silently. Once you have a CSV, call "
        "load_ranked_groups(allowed_licenses=..., csv_path=<path>) "
        "directly.")


def parse_oogiri_corpus_csv(
    csv_path: Union[str, Path],
    min_group_size: int = MIN_GROUP_SIZE,
) -> Tuple[List[RankedGroup], OogiriMasterLoadStats]:
    """Pure parsing, NO network: read an Oogiri-Corpus.csv-shaped file
    (columns exactly `id, prompt_id, prompt, response, vote_count`) and
    build `RankedGroup`s, one per distinct `prompt_id`. Every candidate's
    `license_class` is hardcoded to `research_only` (see module
    docstring).

    A row is skipped (counted in `malformed_rows_skipped`) if it is
    missing `prompt_id`, `prompt`, or `response` text, or if `vote_count`
    is not parseable as a float -- defensive checks against a malformed
    row, not expected to fire on real builder output.
    """
    csv_path = Path(csv_path)
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = tuple(reader.fieldnames or ())
        if not set(EXPECTED_COLUMNS).issubset(set(header)):
            raise ValueError(
                "oogiri_master.parse_oogiri_corpus_csv: %r is missing "
                "expected column(s) %r (found columns: %r). This module "
                "targets the exact Oogiri-Corpus.csv schema verified "
                "against the builder repo's README this session -- if "
                "the upstream schema changed, this parser needs updating, "
                "not silently working around a mismatch." %
                (str(csv_path), sorted(set(EXPECTED_COLUMNS) - set(header)),
                 header))

        rows = list(reader)

    total_rows = len(rows)
    malformed = 0
    groups_by_prompt_id: Dict[str, List[dict]] = defaultdict(list)
    prompt_text_by_id: Dict[str, str] = {}
    mismatch_ids = set()

    for rec in rows:
        prompt_id = rec.get("prompt_id")
        prompt = rec.get("prompt")
        response = rec.get("response")
        raw_vote = rec.get("vote_count")
        if not prompt_id or not str(prompt_id).strip():
            malformed += 1
            continue
        if not prompt or not str(prompt).strip():
            malformed += 1
            continue
        if not response or not str(response).strip():
            malformed += 1
            continue
        try:
            vote_count = float(raw_vote)
        except (TypeError, ValueError):
            malformed += 1
            continue

        pid = str(prompt_id)
        prompt_text = str(prompt).strip()
        if pid not in prompt_text_by_id:
            prompt_text_by_id[pid] = prompt_text
        elif prompt_text_by_id[pid] != prompt_text:
            mismatch_ids.add(pid)

        groups_by_prompt_id[pid].append({
            "row_id": rec.get("id"),
            "response": str(response).strip(),
            "vote_count": vote_count,
        })

    groups: List[RankedGroup] = []
    single_dropped = 0
    for pid in sorted(groups_by_prompt_id.keys()):
        candidate_rows = groups_by_prompt_id[pid]
        if len(candidate_rows) < min_group_size:
            single_dropped += len(candidate_rows)
            continue
        candidates = tuple(
            Candidate(
                text=r["response"],
                score=r["vote_count"],
                rater_count=None,  # see module docstring -- no verified
                                   # per-response denominator in this
                                   # schema.
                candidate_id=str(r["row_id"]) if r["row_id"] else
                             "%s-c%d" % (pid, ci),
            )
            for ci, r in enumerate(candidate_rows)
        )
        if len(candidates) < min_group_size:
            single_dropped += len(candidates)
            continue
        groups.append(RankedGroup(
            context=prompt_text_by_id[pid],
            candidates=candidates,
            source_dataset=SOURCE_DATASET,
            license_class=LICENSE_CLASS,
            source_id="oogiri-corpus-prompt-%s" % pid,
            metadata={
                "language": "ja",
                "source_platform": "bokete.jp",
                "vote_semantics": (
                    "aggregate vote_count per response from a blind, "
                    "simultaneous approval-vote round of ~100 judges per "
                    "prompt (paper's own framing) -- NOT individual "
                    "per-judge scores; comparable only within this "
                    "group, per schema.py's design rule."),
            },
        ))

    stats = OogiriMasterLoadStats(
        total_rows=total_rows,
        malformed_rows_skipped=malformed,
        distinct_prompts=len(groups_by_prompt_id),
        prompt_text_mismatch=len(mismatch_ids),
        multi_candidate_groups=len(groups),
        single_candidate_dropped=single_dropped,
        ranked_groups_emitted=len(groups),
        candidates_emitted=sum(len(g.candidates) for g in groups),
    )
    return groups, stats


def load_ranked_groups(
    allowed_licenses: List[str],
    csv_path: Union[str, Path],
    min_group_size: int = MIN_GROUP_SIZE,
) -> Tuple[List[RankedGroup], OogiriMasterLoadStats]:
    """Public entry point. `allowed_licenses` is REQUIRED (per
    `firewall.py`'s design, same as every other loader in this package) --
    since every record this loader produces is `research_only`, a caller
    that does not explicitly include `'research_only'` gets a loud error.

    `csv_path` is ALSO REQUIRED here, unlike `oogiri.load_ranked_groups`'s
    optional `jsonl_path` (which falls back to a real network fetch) --
    there is no fallback fetch for this module (see `ensure_sample`'s
    docstring for why). A caller must supply a path to an
    already-obtained Oogiri-Corpus.csv, produced by independently running
    the builder pipeline after resolving the license question.
    """
    if not csv_path:
        raise ValueError(
            "oogiri_master.load_ranked_groups: csv_path is required -- "
            "there is no automatic download path for this dataset (see "
            "ensure_sample's docstring / this module's ACQUISITION STATUS "
            "section). Pass the path to a locally-obtained "
            "Oogiri-Corpus.csv.")
    groups, stats = parse_oogiri_corpus_csv(csv_path, min_group_size=min_group_size)
    groups = assert_license_class(groups, allowed=allowed_licenses)
    return groups, stats
