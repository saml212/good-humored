"""Loader over the already-downloaded local corpus at
`~/Experiments/good-humored-data/corpus/` (see `DATA.md` and
`~/Experiments/good-humored-data/MANIFEST.md` for the authoritative
layout/counts this module reads).

This corpus is SINGLE JOKES WITH NO RATING GROUP -- each `jokes.jsonl`
line is one joke, optionally carrying an origin `score` (e.g. a Reddit
upvote count) with no sibling candidates rated on the same prompt. That
is exactly `schema.CorpusJoke`'s shape, and this module emits ONLY
`CorpusJoke` records -- never `RankedGroup`, never `PreferencePair`.

NO RANKED GROUPS ARE EVER FAKED FROM UPVOTES HERE. This is the single
most important thing this module does NOT do, stated explicitly because
it would be the easiest possible mistake: two jokes from
`commercial-safe/jokes.jsonl` with scores 500 and 5 are NOT a comparable
pair -- they were posted at different times, seen by different numbers of
people, and a Reddit upvote count measures reach and timing at least as
much as funniness (CLAUDE.md Hard Rules; `references/corpus-sources.md`;
`.claude/skills/humor-rl/SKILL.md` "Data" section: "Engagement metrics
(likes, upvotes) -- measure engagement, not funniness"). There is no
function anywhere in this module that takes two `CorpusJoke`s and
produces a `PreferencePair` or `RankedGroup` from their scores. If you
need reward-model training pairs, use `oogiri.py` or `nycc.py`, whose
source data was actually rated multiple-candidates-per-prompt --
`local_corpus.py`'s data structurally cannot support that without
committing the exact mistake this paragraph is warning against.

LAYOUT ACTUALLY ON DISK (verified this session against
`~/Experiments/good-humored-data/MANIFEST.md`, matches exactly):

    corpus/commercial-safe/jokes.jsonl       887,639 rows, CC-BY-4.0
                                              (single source: SocialGrep/
                                              one-million-reddit-jokes)
    corpus/research-only/jokes.jsonl         310,151 rows, mixed origin:
                                              Fraser/short-jokes (224,955,
                                              unspecified license, treated
                                              research_only per project
                                              policy) + taivop's three
                                              sources (reddit/stupidstuff/
                                              wocka, 85,196 combined,
                                              explicit research-only)
    corpus/chatgpt-25-templates.jsonl        25 rows, CC0 -- the Jentzsch
                                              & Kersting memorized-joke
                                              baseline. Loaded SEPARATELY
                                              by `load_memorized_templates`
                                              below, NEVER bundled into
                                              `load_corpus_jokes`'s output
                                              by default -- see that
                                              function's docstring for why.

Each `jokes.jsonl` line already carries its own `source` and `license`
free-text fields (DATA.md's documented schema) -- this loader propagates
BOTH into `CorpusJoke.metadata` (`original_source`, `original_license_text`)
even though `license_class` (the firewall-facing field) is fixed per
bucket directory. This preserves the finer-grained provenance DATA.md's
schema already captured, without pretending the bucket-level
commercial_safe/research_only split is the whole story -- a caller who
needs to know e.g. "this specific row came from taivop-wocka" still can.

STREAMING: files are read line-by-line, never loaded fully into memory as
a list before filtering -- at 887,639 + 310,151 rows (~390MB combined,
per MANIFEST.md's own disk-usage note) this matters for anything short of
a training box. `limit` is available as an explicit dev/testing
convenience (cap rows read per bucket) -- it is NOT a sampling strategy
for real training data (a truncated head of a JSONL file is not a
representative sample of anything), and its docstring says so.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from data_adapters.firewall import assert_license_class
from data_adapters.schema import CorpusJoke

BUCKET_LICENSE_CLASS = {
    "commercial-safe": "commercial_safe",
    "research-only": "research_only",
}
TEMPLATES_FILENAME = "chatgpt-25-templates.jsonl"
TEMPLATES_SOURCE_DATASET = "chatgpt25-memorized-templates"
TEMPLATES_LICENSE_CLASS = "commercial_safe"  # CC0, per MANIFEST.md


@dataclass
class LocalCorpusLoadStats:
    """Counts reported alongside every load, per bucket read."""

    rows_by_bucket: Dict[str, int]
    rows_skipped_malformed: int
    jokes_emitted: int


def _default_data_dir() -> Path:
    return Path(
        os.environ.get("GOOD_HUMORED_DATA_DIR",
                       "~/Experiments/good-humored-data")).expanduser()


def _corpus_dir(data_dir: Optional[Union[str, Path]]) -> Path:
    base = Path(data_dir) if data_dir else _default_data_dir()
    return base / "corpus"


def _iter_bucket_jokes(bucket_path: Path, source_dataset_default: str,
                       license_class: str, limit: Optional[int]):
    n = 0
    with open(bucket_path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                text = rec["text"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty text")
            except (json.JSONDecodeError, KeyError, ValueError):
                yield None  # signals a skipped/malformed row to the caller
                continue
            n += 1
            yield CorpusJoke(
                text=text,
                source_dataset=rec.get("source", source_dataset_default),
                license_class=license_class,
                source_id=rec.get("id", "%s-%d" % (bucket_path.stem, n)),
                score=rec.get("score"),
                metadata={
                    "original_source": rec.get("source"),
                    "original_license_text": rec.get("license"),
                },
            )


def load_corpus_jokes(
    allowed_licenses: List[str],
    data_dir: Optional[Union[str, Path]] = None,
    limit_per_bucket: Optional[int] = None,
) -> Tuple[List[CorpusJoke], LocalCorpusLoadStats]:
    """Public entry point for the main jokes corpus (commercial-safe +
    research-only buckets). Does NOT include the memorized-template
    baseline -- see `load_memorized_templates` and its docstring for why
    that stays a separate call.

    `allowed_licenses` is REQUIRED (no default, per `firewall.py`). This
    loader is DEFENSE IN DEPTH twice over: it only opens the bucket
    file(s) whose `license_class` is in `allowed_licenses` in the first
    place (a caller who only allows `commercial_safe` never has
    `research-only/jokes.jsonl` opened at all, not just filtered after
    the fact), AND still runs every emitted record through
    `assert_license_class` before returning, so a bug in the
    file-selection logic above would still be caught rather than silently
    leaking a wrong-license row.

    `limit_per_bucket`: dev/testing convenience only (caps rows read per
    bucket file). This is a truncated head-of-file read, NOT a
    representative sample -- do not use it to build an actual training
    subset; use a proper sampling method downstream of the full read if
    you need a subset for real work.
    """
    corpus_dir = _corpus_dir(data_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(
            "local_corpus.load_corpus_jokes: corpus dir %s does not "
            "exist. See DATA.md for where this project's local corpus is "
            "expected to live and how to regenerate it." % corpus_dir)

    rows_by_bucket: Dict[str, int] = {}
    skipped = 0
    jokes: List[CorpusJoke] = []

    for bucket_dirname, license_class in BUCKET_LICENSE_CLASS.items():
        if license_class not in allowed_licenses:
            continue  # never even opens the disallowed bucket's file
        bucket_path = corpus_dir / bucket_dirname / "jokes.jsonl"
        if not bucket_path.exists():
            rows_by_bucket[bucket_dirname] = 0
            continue
        count = 0
        for item in _iter_bucket_jokes(
            bucket_path, source_dataset_default=bucket_dirname,
            license_class=license_class, limit=limit_per_bucket,
        ):
            if item is None:
                skipped += 1
                continue
            jokes.append(item)
            count += 1
        rows_by_bucket[bucket_dirname] = count

    jokes = assert_license_class(jokes, allowed=allowed_licenses)
    stats = LocalCorpusLoadStats(
        rows_by_bucket=rows_by_bucket,
        rows_skipped_malformed=skipped,
        jokes_emitted=len(jokes),
    )
    return jokes, stats


def load_memorized_templates(
    allowed_licenses: List[str],
    data_dir: Optional[Union[str, Path]] = None,
) -> List[CorpusJoke]:
    """Separate, explicit loader for `chatgpt-25-templates.jsonl` -- the
    Jentzsch & Kersting 2023 mode-collapse baseline (the 25 jokes that
    accounted for >90% of a 1,008-joke ChatGPT sample).

    Kept OUT of `load_corpus_jokes`'s default output on purpose: these 25
    rows exist so a novelty penalty / eval pipeline can check "did the
    model just recite one of the famous memorized jokes" (CLAUDE.md Hard
    Rules: "any generation eval MUST include a novelty check against a
    memorized-joke corpus"; see `benchmark/joke_novelty.py`,
    `env/rewards.py`'s `CorpusNoveltyPenalty`). If these same 25 jokes
    were silently mixed into an SFT/training corpus by a caller who just
    asked "give me all the local corpus jokes", the model could be
    trained ON the exact templates the novelty check exists to catch --
    directly defeating the anti-mode-collapse mechanism this whole
    project is built around. Requiring a separate, differently-named call
    to get these rows is a deliberate speed bump against exactly that
    mistake.

    `license_class` is `commercial_safe` (CC0, per MANIFEST.md) --
    `allowed_licenses` is still required (per `firewall.py`'s uniform
    contract), even though this loader only ever produces one class.
    """
    corpus_dir = _corpus_dir(data_dir)
    path = corpus_dir / TEMPLATES_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            "local_corpus.load_memorized_templates: %s does not exist. "
            "See DATA.md for the expected corpus layout." % path)

    jokes = [
        j for j in _iter_bucket_jokes(
            path, source_dataset_default=TEMPLATES_SOURCE_DATASET,
            license_class=TEMPLATES_LICENSE_CLASS, limit=None)
        if j is not None
    ]
    return assert_license_class(jokes, allowed=allowed_licenses)
