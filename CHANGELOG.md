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
- Every WER is published with its **runaway count**: utterances whose hypothesis
  has more than 3x the reference's flozi-strict words (`wer.runaway` in
  `benchmark.config.yaml`), as `n_runaway`, `runaway_pct` and
  `runaway_edit_share_pct` (their share of the corpus word edits) in
  `summary.json` and `expected.json`, re-derived by `make verify`, and as the
  `loops` / `loop edits %` columns of the WER table. The WER still counts them.
  Every committed WER artifact gained the three fields (all zero); no existing
  value changed.
- The serving recipe pins **decoding and output length** for every self-hosted
  model: `decoding = {mode = "greedy", max_tokens = 1024}` for vLLM and the
  transformers stand-in, sent with every request together with temperature 0 and
  repetition penalty 1.0; for NeMo the checkpoint's own decoding config, which
  the server now refuses to serve when it differs. Servers report it as
  `x_raven.decoding` and runs record it in the `serving` block. Rows measured
  before were capped by engine defaults (4,096 tokens for Phi-4, the whole context window
  elsewhere) and are not comparable on looping models; `make reparse` cuts their
  recorded output to the cap after the fact, flagged `capped_post_hoc` and
  `"exact": false`.

### Fixed
- MOSS-Transcribe-Diarize's timestamp and speaker markup (`[4.18][S01] … [13.81]`)
  is parsed out of its output in the model layer (`raven_asr.output_formats`)
  instead of being scored as words. Prediction lines of a model with an output
  format carry `raw_prediction` (the server's output) and `segments` beside the
  spoken-text `prediction`; `summary.json` names the `output_format`. Scored raw,
  the strict-de lens put MOSS at ~100 % WER on read speech, and the published
  flozi-strict lens — which drops bracketed text itself — still lost words the
  markup glued together. Runs recorded before are re-derived without inference
  by `make reparse`, which records the step under `post_hoc`. The scoring
  contract (`benchmark.config.yaml`) is unchanged; no committed artifact moves.

### Changed
- The vLLM serving engine moves from 0.28.0 to 0.33.0 (FlashInfer 0.7.2.post1
  with its prebuilt CUDA 13.0 kernels, torch 2.14.0, transformers 5.20.2 now
  pinned too), one engine for every vLLM model. vLLM rows measured on 0.28.0
  are not comparable to rows on 0.33.0; each row's `serving` block names its
  engine, and every vLLM model is re-measured on the new one.
- `granite-speech-4.1-2b-plus` serves IBM's re-trained 2026-10-21 checkpoint
  (`e69efe37`), which vLLM loads from 0.33.0 on, instead of the one the
  September sweep measured. Every other vLLM and transformers recipe was
  already at its repo's newest commit.

### Fixed
- Every vLLM and transformers serving recipe pins the Hugging Face commit it
  loads (`revision`, required). IBM re-uploaded `granite-speech-4.1-2b-plus` on
  2026-10-21 as a re-trained model with renamed weights, which vLLM 0.28.0 cannot
  load; the unpinned recipe had silently followed it. Granite is pinned to the
  checkpoint the September sweep served; every other recipe to its repo's
  current main (2026-11-02), the weights the running round loads anyway.
- The four NeMo recipes pin their Hugging Face commit too. `ASRModel.from_pretrained`
  takes no revision, so each NeMo recipe now names its `.nemo` file
  (`checkpoint_file`, required); the launcher downloads it at the pin and the
  server restores that file and reports the pin in `/health` (`model_revision`).
  Pinned to the commits the November round loads. This changes how parakeet-tdt,
  canary and nemotron are loaded (`restore_from` on the same file instead of
  `from_pretrained`), so a pinned start should reproduce the current rows within
  batching tolerance before the next round relies on it.

### Added
- The serving recipe of the 14 self-hosted models is part of the repo
  (`serving/`): engine pins, one declarative recipe per model
  (`serving/recipes.toml`), the launcher, the NeMo micro-batching server and the
  OpenAI-compatible shim. The servers report `x_raven.engine`, `x_raven.recipe`
  (`id` + SHA-256 over the recipe and the files that execute it) and
  `x_raven.batching`. Each throughput pass and each `summary.json` result row
  records this as `serving` (`null` for servers that report none and for hosted
  APIs); a change of serving setup during a run, or against the passes of a
  subset being resumed, aborts like a served-model mismatch (`reproduce` exit 3).
  `promote` carries it; `make verify` holds a row's block to its passes and
  accepts artifacts without one. Self-hosted rows reproduce to character-level
  batching tolerance, not bit-exactly.
- WER runs record throughput. Every pass over a subset (a resumed subset has
  several) appends its wall time, requests, successes, transcribed audio,
  concurrency, adapter and hardware label (`RAVEN_EVAL_HARDWARE`, never
  detected) to `throughput_<subset>.jsonl` — on SIGTERM too, flagged
  `interrupted`. Each `summary.json` result gets a `throughput` block (`rtfx` =
  `total_audio_s / total_wall_s`, `utterances_per_s`, `comparable` false when
  passes ran at different concurrency or hardware); `promote` carries the passes
  into the artifact. `make verify` does not reproduce throughput — it fails a
  block whose totals or ratios contradict its passes, and accepts artifacts
  without one.
- Self-hosted WER models bind to two endpoint env vars by server type
  (`NEMO_BENCH_URL`, `VLLM_BENCH_URL`, optional `BENCH_STT_API_KEY`); five more
  are registered (Qwen3-ASR-1.7B, Voxtral-Mini-3B-2507, whisper-large-v3,
  moonshine-streaming-small-de, nemotron-3.5-asr-streaming-0.6b). The adapter
  sends 16 kHz audio and aborts the run when the server reports serving another
  model (`x_raven.model`). Replaces `VLLM_PRIMELINE_*`, `NEMO_BENCH_API_KEY` and
  `VLLM_BENCH_API_KEY`.
- WER runs resume per utterance: predictions are appended as they arrive, a
  rerun requests only missing or failed utterances, and the final file and
  scores equal an uninterrupted run's. `reproduce --concurrency`, and an exit
  status that is 0 only for a complete subset.
- WER datasets are read from a local copy at the pinned revision; streaming from
  the Hub is opt-in (`--streaming`). `python -m raven_asr.prefetch` fetches ahead
  of a run; `RAVEN_EVAL_CACHE_DIR` moves every archive-backed corpus (xSID's
  default moves from `~/.cache/raven-eval` to `~/.cache/raven.eval`).
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
