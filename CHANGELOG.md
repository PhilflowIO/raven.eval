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
- **Engine groundwork for Nemotron-3.5-Transcribe-30B-A4B** (#78; its recipes
  follow with the quantized checkpoint). NeMo Speech's `nemo_speechlm` vLLM
  plugin is installed into the one vLLM 0.33.0 environment, pinned to a fork
  commit (`PhilflowIO/Speech@821e0367`) that adapts the plugin to vLLM 0.33's
  moved chat parser (upstream: NVIDIA-NeMo/Speech#16505). A vLLM recipe names
  the plugins it needs (`vllm_plugins`), and the launcher sets `VLLM_PLUGINS`
  to exactly that list for every vLLM model, so no other model's engine loads
  the plugin. The shim gains the card's chat request for the model
  (`nemotron-transcribe` family: verbatim benchmark instruction, audio_url,
  reasoning off). `serving/quantize/nemotron_transcribe_w8a16.py` is the
  committed recipe for the int8 per-channel weight-only build of the language
  model (llm-compressor 0.15.0 model-free, CPU; audio encoder, CTC head, router,
  embeddings and lm_head unchanged bf16), with its own pinned environment. Every
  vLLM recipe hash changes (launcher and vLLM requirements).
- **Voxtral-Small-24B-2507**, quantized and split over two cards (#79):
  RedHat AI's FP8 build (`RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic`,
  per-channel FP8 on the language model's MLP layers; encoder, attention and
  lm_head bf16) served weight-only (W8A16, bf16 activations) through vLLM's
  Marlin kernel at tensor-parallel 2, plus its **bf16 control** (the Mistral
  checkpoint with 14 GB per card offloaded to host RAM, FLEURS-de only). The
  precision block gains an optional `control` key naming the quantized recipe,
  and a fourth checkpoint source, `third-party`, for quantized builds by someone
  other than the model's vendor. A quantized vLLM scheme must name a kernel the
  launcher can pin; `fp8-w8a16-marlin` disables the Humming kernel that vLLM
  0.33.0 ranks first on an RTX 3090.
- Every self-hosted result names its **precision** (#81). Each serving recipe
  declares `precision = {weights, activations, scheme, checkpoint,
  tensor_parallel}` — the format the weights are served in, the format the
  computation runs in, the quantization scheme (`none` for plain floating
  point), where the weights come from (`published` / `vendor` / `self`, with the
  format the checkpoint stores, so a cast is on record) and how many GPUs one
  instance is split over — and validation refuses a recipe without it. vLLM is
  started with `--dtype` / `--tensor-parallel-size` from it, the NeMo server and
  the transformers stand-in refuse to serve weights in another format. Servers
  report it as `x_raven.precision`; it is the fifth key of the `serving` block in
  every throughput pass, `summary.json` row and promoted artifact, so a change of
  precision mid-subset aborts the run (exit 3) like any change of setup, and a
  server that names its recipe but not its precision is refused. The WER table
  has a `precision` column held to each artifact's record. The existing recipes
  were backfilled from their checkpoints at the pinned commits: NeMo models and
  moonshine fp32, Whisper large-v3 / -turbo fp16, every other model bf16, and
  primeline-whisper-large-v3-german fp16 cast from a bf16 checkpoint. Blocks
  recorded before (four or three keys) stay valid.
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
- A second output format: VibeVoice-ASR answers with a JSON list of timed
  speaker turns (`[{"Start","End","Speaker","Content"}, …]`), parsed in the model
  layer like MOSS's markup (`vibevoice-json`): `prediction` is the turns'
  `Content` in order, `raw_prediction` the answer, `segments` the turns. An
  answer the output cap cut mid-turn keeps the words it has. The scoring
  contract is unchanged.

### Fixed
- **Resume no longer keeps rows measured under another setup** (#87). A
  complete subset was reused whenever the directory's `summary.json` matched the
  run — but every invocation rewrites that summary, so after the first dataset
  of a campaign was re-measured under a new commit or serving recipe, the other
  datasets' complete rows from the old one counted as done (one model: 9,799
  new rows for one dataset, 0 for seven). Reuse is now decided per subset
  against the run manifest and the serving block that subset recorded; a
  mismatch moves its files aside (`<results-dir>.superseded-<timestamp>/`) and
  re-measures it. A half-done subset under another serving block is now
  re-measured the same way instead of aborting with exit 3.
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
- `primeline/whisper-large-v3-german` is served in **bf16**, the format its
  `model.safetensors` stores, instead of an fp16 cast (#82). The recipe hash
  changes; earlier rows of this model carry `fp16 (cast from bf16)` and are
  superseded by its run-4 re-measurement.
- The NeMo engine moves from the 3.0.0 release to NeMo Speech commit
  `b19b1a79` (reports 3.1.0), the first that honours `target_lang` in offline
  `transcribe()`. The two prompt-conditioned Nemotron models
  (nemotron-3.5-asr-streaming-0.6b, nemotron-labs-unified-asr-0.6b) are now
  forced to German (`de-DE`) and report `language_forced = true`; before they
  auto-detected the language. Every NeMo recipe hash changes and all five NeMo
  models are re-measured in run 4.
- The vLLM serving engine moves from 0.28.0 to 0.33.0 (FlashInfer 0.7.2.post1
  with its prebuilt CUDA 13.0 kernels, torch 2.14.0, transformers 5.20.2 now
  pinned too), one engine for every vLLM model. vLLM rows measured on 0.28.0
  are not comparable to rows on 0.33.0; each row's `serving` block names its
  engine, and every vLLM model is re-measured on the new one.
- The vLLM environment adds `flashinfer-cubin` (0.7.2.post1, beside
  `flashinfer-python` and `flashinfer-jit-cache` of the same version). vLLM 0.33
  enables FlashInfer only when that package is present or a CUDA toolkit is, so
  on the GPU host every vLLM model so far started with FlashInfer disabled and
  the jit-cache unused. With it, the kernels the pins name are the ones that run;
  vLLM rows served without it are not part of a published round.
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
