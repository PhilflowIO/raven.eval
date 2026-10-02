# Changelog

Changes to the **measurement protocol** — what a published number means: metric
definitions, normalization, aggregation, uncertainty, datasets and their pins, and
the checks that hold a number to its artifact. New rows measured under an
unchanged protocol are not listed; `BENCHMARKS.md` carries them.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). A
release is a `results-YYYY-MM-DD` tag on a published result round (see README,
"Citing a result"). Everything before the first tag is dated by commit and
untagged, so it has no fixed version to cite.

## `summary.json` schema versions

A Tier-2 run's `summary.json` carries a `schema_version`. `make verify` fails on
a summary whose version is not listed here, rather than re-scoring a format it
would have to guess at (`raven_eval_core/run_manifest.py`).

| Version | Shape |
|---|---|
| 1 | Every summary written before the field existed, identified by `schema_version` being **absent**. Model and dataset revisions, coverage counters, results. Committed artifacts in this format stay as they are: a manifest back-filled later would describe an environment nobody recorded. |
| 2 | Adds `schema_version: 2` and `run_manifest`: `git` (`sha`, `dirty`), `uv_lock_sha256`, `python`, `platform`, `libraries` (name → installed version, `null` if not installed), `gpu` (`null` when inference ran on someone else's hardware — a hosted API or a remote endpoint), `config_sha256` (the resolved config: scoring contract plus the run's pins and settings), `argv`. Never an environment variable's value or an endpoint URL. |

## [Unreleased]

### Added
- Every Tier-2 `summary.json`, WER and DER, embeds a run manifest and a
  `schema_version` (table above). `make verify` and `make promote` reject a
  version they do not know, and a version-2 summary without a well-formed
  manifest.
- `benchmark.config.yaml` is loaded through a strict schema
  (`raven_eval_core/contract.py`): an unknown or missing key fails the load
  instead of being read as "not set".
- `artifacts/SHA256SUMS`: a sha256 manifest over every file under `artifacts/`,
  written by `make manifest`. `make verify` (and CI) fail before re-scoring if any
  file is modified, missing or unlisted.
- The `results-YYYY-MM-DD` release convention: one tag per published result
  round, so a cited number points at fixed bytes.
- `n_dropped` / `dropped_by_reason` on every result row of a run's
  `summary.json`, and carried by promotion into `expected.json`: corpus rows the
  loader could not yield (`empty_reference`, `reference_missing`,
  `clip_missing`, `undecodable`) and that were therefore never attempted. Every
  loader counts them instead of skipping silently; an undecodable clip is a
  counted drop rather than an aborted run. Drops do not block promotion.
  `make verify` holds a committed count to its own breakdown; artifacts promoted
  before the counters existed do not carry the fields and are not failed for it.
  Measured on the full FHNW public half: 2,875 of 2,875 rows yield a clip.
- A conflict-of-interest statement above the result tables in `README.md` and
  `BENCHMARKS.md`, naming the systems Raven's own product runs.

### Changed
- A WER run resumes a finished subset only if it was measured under the same
  commit, lockfile, libraries and config as the resuming run; otherwise the
  subset is re-run. One summary holds one manifest, so it can only describe
  results that share one.

## 2026-09-19 — untagged

### Changed
- WER is aligned **per utterance** and then summed over the corpus. Before, a
  subset was concatenated into one sentence first, so errors cancelled across
  clip boundaries. Unchanged on the flozi subsets and FLEURS (largest move
  +0.001 pp); VoxPopuli and the two Swiss sets had read 0.26–0.85 pp low.
- A failed request is recorded as a prediction line and fails `make verify`;
  `n_samples` in `expected.json` binds each predictions file to its length. A WER
  over only the clips a model finished is no longer publishable.
- DER and WER share one seeded bootstrap (`raven_eval_core.bootstrap`). A paired
  comparison fails when a file exists for only one model instead of dropping it.

### Added
- A 95 % bootstrap interval over utterances on every WER (`wer.uncertainty` in
  `benchmark.config.yaml`), committed to `expected.json` and re-derived by verify.
- WER artifacts record `model_revision` and a per-result `dataset_revision` (or
  archive sha256); `flozi00/asr-german-mixed-evals` is pinned to `9c34cbcc`.
- Sortformer artifacts record the latency preset they were measured at.

## 2026-09-18 — untagged

### Added
- `wer_strict_de_pct` (the benchmark page's strict-de, length-weighted lens) and
  BLEU for every `bleu+wer` subset are written at promote and re-scored by verify.
- Dataset `avemio-german-mixed-test`.

## 2026-09-05 — untagged

### Changed
- The streaming operating point (latency preset) is pinned in
  `benchmark.config.yaml` and asserted against the adapter: the same Sortformer
  v2 checkpoint reads 8.98 or 9.82 % on CALLHOME-de depending on it.
- Training-data overlap is stated per corpus beside the rows it affects
  (DiariZen was trained on AMI and VoxConverse).

## 2026-09-04 — untagged

### Changed
- Each DER collar carries its **own** miss/FA/confusion decomposition, and both
  aggregations (corpus `Σerr/Σtotal` and file-mean) are published;
  `expected.json` holds all ten scalars and verify re-checks every one.
  `der.aggregation` joins the contract file.
- Hosted diarizers are scored with the shared turn folding, local ones without
  it; the page no longer claims an identical protocol across the two lanes.
- Every diarization dataset must declare its spoken language, passed to every
  adapter.

### Added
- Seeded 10 000-resample bootstrap intervals over files, DER by reference speaker
  count, overlap fraction and boundary offsets (`make analyse`).
- The published tables are bound to their artifacts in both directions
  (`tests/test_published_table.py`).
- The `dscore`/md-eval cross-check was executed for the first time; four defects
  fixed, collar conversion (`dscore --collar X` = pyannote collar `2X`) pinned.

## 2026-09-03 — untagged

### Added
- BLEU for translation-shaped corpora (sacrebleu, 13a, case-sensitive, exp
  smoothing, corpus aggregation); a changed BLEU signature fails verify.
- Datasets: FLEURS, MLS-de, VoxPopuli-de, the Swiss SPC and FHNW corpora, the
  Bavarian xSID probe and its Standard-German control.
- Licence and shippability are data on each diarizer spec; no winner mark on a
  non-shippable row.
- `tests/test_dataset_contract.py`: every dataset in the contract has a loader
  and vice versa.

### Changed
- One flag (`eligible_for_aggregate`) keeps every dialect corpus out of
  cross-dataset averages.
- Hosted diarizers are pinned by an explicit vendor model version, never an
  alias; word-level labels are folded into turns by one shared rule.

## 2026-09-02 — untagged

### Changed
- Every dataset and diarizer is pinned to an immutable 40-hex revision, enforced
  by a test. AMI test split added.

## 2026-07-31 — untagged

### Added
- Initial protocol: flozi-strict corpus WER, DER at collar 0.0 and 0.25 with
  overlap scored, and `make verify` re-scoring every committed artifact.
