# DATA.md — where the data lives

Data and checkpoints for `good-humored` live **outside this repo**, per
project policy (see `CLAUDE.md` → Data). This file is the pointer.

## Location

```
~/Experiments/good-humored-data/
```

Not a git repo. Not gitignored-into this repo — it's simply never staged
here. Nothing under that path should ever be committed to `good-humored`.

## Layout

```
~/Experiments/good-humored-data/
  raw/                                  # untouched downloads
  corpus/
    commercial-safe/jokes.jsonl         # CC-BY and similarly safe sources
    research-only/jokes.jsonl           # unclear/NC-license sources
    chatgpt-25-templates.jsonl          # Jentzsch & Kersting 2023 top-25
                                         # memorized-joke baseline (verbatim,
                                         # arXiv:2306.04563 Appendix B)
    _dedup_stats.json                   # machine-readable dedup stats
  MANIFEST.md                           # per-source counts, licenses,
                                         # dedup stats, exact download commands
```

## What's in it (as of 2026-07-16)

The memorized-joke corpus used for anti-plagiarism / mode-collapse novelty
checking in humor-generation evals (see CLAUDE.md Hard Rules — "any
generation eval MUST include a novelty check against a memorized-joke
corpus"). ~1.2M deduplicated jokes total, split by license into
commercial-safe and research-only buckets, plus the canonical 25-joke
ChatGPT memorization baseline. Full provenance, per-source counts, license
detail, and dedup methodology are in
`~/Experiments/good-humored-data/MANIFEST.md` — read that before using the
corpus for anything, especially before commercial use of the
`commercial-safe` bucket (verify attribution requirements) or any use of
`research-only` (do not use commercially).

## Schema

Each `jokes.jsonl` line:

```json
{"id": "<source>-<n>", "text": "...", "source": "...", "license": "...", "score": <int|null>}
```

## Regenerating

Everything under `corpus/` and `raw/` is reproducible from the exact curl
commands and normalization steps logged in MANIFEST.md — nothing here is
hand-edited or irreplaceable except the MANIFEST itself (which records what
was actually done, including counts and download dates for point-in-time
reproducibility).
