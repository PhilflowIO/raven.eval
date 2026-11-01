# Tier-2 reproduce — keys & GPU (the price of a full re-run)

Tier-1 (`make verify`) needs nothing. Tier-2 (`make reproduce`) actually runs a
model over public audio, so it needs that model's credentials or GPU. This is
the honest cost of a from-raw-audio reproduction — documented here, not hidden.

## Install

```bash
make install                      # installs the `asr` extra (datasets, soundfile, …)
# for the Modal-hosted models additionally:
uv sync --extra dev --extra asr --extra modal
```

## Datasets (no key)

### No system FFmpeg needed on this lane

The WER loaders decode audio with **soundfile**, not `torchcodec`. That is a
deliberate constraint, not an accident: `datasets` 5.x decodes an `Audio()`
feature through `torchcodec`, which `dlopen`s the FFmpeg `libav*` shared objects
at runtime — the same trap `TIER2-DER-KEYS.md` documents for the pyannote lane.
Requiring a system FFmpeg build to run `make reproduce METRIC=wer` would be a
much steeper entry price than this lane needs, so the loaders switch decoding off
(`Audio(decode=False)`) and decode the raw bytes themselves. A side benefit: they
observe the file's **true** sample rate instead of the one the dataset card
declares, and dataset cards are wrong about this often enough to matter.

Public HF dataset `flozi00/asr-german-mixed-evals`, subsets `Tuda-De`,
`multilingual_librispeech`, `common_voice_19_0`, plus avemio, FLEURS, MLS and
VoxPopuli German, each pinned to a revision in `raven_asr.config.WER_DATASETS`
(for german-mixed `FLOZI_DATASET_REVISION`, the revision the published rows were
measured at); `--dataset-revision <hash>` overrides it, and the revision a run
actually read is recorded per result in `summary.json`.

### Local copies instead of streaming

A run reads its dataset from a **local copy** of exactly the split's files at the
pinned revision, and fetches them on first use. Hub streaming is opt-in
(`--streaming`): it saves the download for a quick `--limit` smoke run, but it
times out on long jobs, so a full run should never depend on it. Fetch ahead of
an unattended run:

```bash
python -m raven_asr.prefetch --dataset all        # or one id; make prefetch DATASET=all
```

| Env var | Moves | Default |
|---|---|---|
| `HF_HOME` (or `HF_HUB_CACHE`) | HF-hosted corpora: german-mixed, avemio, fleurs, mls-de, voxpopuli-de | `~/.cache/huggingface` |
| `RAVEN_EVAL_CACHE_DIR` | archive-backed corpora (spc-test, fhnw-all-dialects, xsid-*) under `<dir>/corpora` | `~/.cache/raven.eval` |

More specific overrides still win over `RAVEN_EVAL_CACHE_DIR`:
`RAVEN_EVAL_CORPORA_DIR` / `BENCH_CORPORA_DIR` (all archive corpora),
`SWISS_CORPORA_DIR`, `XSID_CORPORA_DIR`.

`summary.json` also carries the run manifest: the commit and dirty flag of this
checkout, the `uv.lock` hash, library versions, a hash of the resolved config and
the command line (`CHANGELOG.md`, "summary.json schema versions"). It holds no
environment variable's value: no key and no endpoint URL is written to it.

## Per-adapter requirements

| Model key (`MODEL=`)                 | Adapter          | Needs |
|--------------------------------------|------------------|-------|
| `modal/parakeet`, `modal/qwen3-asr`, `modal/voxtral-2507`, `modal/voxtral-2602` | `modal_app` | Modal token (`modal token set …`) + the app deployed on your Modal account (GPU runs there). App names in `raven_asr.config.MODAL_APP_NAMES`. |
| `openai-whisper-1`                   | `openai_whisper` | `OPENAI_API_KEY` |
| `deepgram-nova-2`                    | `deepgram`       | `DEEPGRAM_API_KEY` |
| `voxtral-mini-latest`                | `voxtral_mistral`| `MISTRAL_API_KEY` |
| `xai/grok-voice-transcribe-2.0`, `xai/grok-voice-transcribe-1.0` | `xai`            | `XAI_API_KEY` (hosted, us-east-1, **billed per hour of audio**) |
| `assemblyai/universal-3-6-pro-realtime` | `assemblyai_realtime` | `ASSEMBLYAI_API_KEY` (hosted streaming, **billed per hour of audio**, 0.45 $/h; clips are fed at real-time pace, so a run takes audio length ÷ concurrency) |
| Self-hosted, NeMo-style server: `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/canary-1b-v2`, `nvidia/nemotron-3.5-asr-streaming-0.6b`, `primeline/parakeet-primeline` | `vllm_openai` | an OpenAI-compatible ASR server (self-host GPU); base URL including `/v1` in `NEMO_BENCH_URL` |
| Self-hosted, vLLM server (or a shim in front of one): `ibm-granite/granite-speech-4.1-2b-plus`, `CohereLabs/cohere-transcribe-03-2026`, `microsoft/Phi-4-multimodal-instruct`, `OpenMOSS-Team/MOSS-Transcribe-Diarize`, `moonshine-ai/moonshine-streaming-small-de`, `Qwen/Qwen3-ASR-1.7B`, `mistralai/Voxtral-Mini-3B-2507`, `primeline/whisper-large-v3-german`, `primeline/whisper-large-v3-turbo-german`, `openai/whisper-large-v3` | `vllm_openai` | an OpenAI-compatible ASR server (self-host GPU); base URL including `/v1` in `VLLM_BENCH_URL` |

Both self-hosted server types share an optional Bearer token in
`BENCH_STT_API_KEY`. `MODEL=` is the HF repo id; it is what the request posts as
`model` (multipart `POST {base}/audio/transcriptions` with `model`,
`language=de` and a 16 kHz mono WAV — the adapter resamples). If the server's JSON
answer carries `x_raven.model` and it names another model, the run **aborts**
(`reproduce` exit 3) instead of recording that model's transcripts under this
one's name; a server that sends no `x_raven.model` is warned about once and
trusted.

The `*_URL` / `*_API_KEY` variables name **your** endpoints — this repo never
ships URLs or key values, only the env-var names the runner reads.

DER adapters keep their own key table in `docs/TIER2-DER-KEYS.md`; the
key-bearing ones there (`assemblyai-universal-3-5-pro`, `deepgram-nova-3`) are
hosted and **billed per hour of audio**. `ASSEMBLYAI_API_KEY` serves both the
DER lane and the streaming WER lane above. Same rule: names
here, values only in your environment.

Set the key in your environment before `make reproduce`. Missing-key failures
surface per-utterance in the runner log.

## Long and unattended runs

```bash
make reproduce METRIC=wer DATASET=avemio-german-mixed-test MODEL=Qwen/Qwen3-ASR-1.7B CONCURRENCY=16
# = python -m raven_asr.reproduce --metric wer --dataset <id> --model <id> --concurrency 16 --out results/reproduce
```

- `--concurrency N` (`CONCURRENCY=N`) is the number of requests in flight; the
  default is the adapter's own. Samples are pulled from the dataset only as fast
  as requests complete, so memory does not grow with the size of the split.
- **Resume per utterance.** Each transcript is appended to
  `predictions_<subset>.jsonl` as it arrives. Run the same command again after an
  abort and only the utterances without a transcript (failed ones included) are
  requested; at the end the file is rewritten to one line per utterance, and the
  scores equal those of an uninterrupted run. A finished subset is marked by
  `.done_<subset>.json` and skipped. Both only resume under the same commit,
  `uv.lock` and config as the run that wrote them; otherwise the subset starts
  over.
- **Exit status of `reproduce`:** `0` every utterance succeeded and the done
  marker exists · `1` incomplete (some requests failed) — run it again · `3` the
  endpoint serves a different model.
- `MODEL_REV=` is DER-only. A WER model's revision is fixed in the registry: the
  versioned name of a hosted model, and none for a self-hosted endpoint, whose
  loaded weights a client cannot attest.

## Latency / RTFx is NOT a Tier-2 number

`predictions_*.jsonl` records a `latency_s` per utterance, but latency and RTFx
depend on your hardware, region, and network — they are **not** portable. Treat
them as Tier-3 transparency (documented with methodology), never as a published
comparable number. Only WER/CER cross the Tier-2 line as reproducible.

## From run to committed proof

```bash
make reproduce METRIC=wer DATASET=common_voice_19_0 MODEL=modal/parakeet
make promote   RESULTS=results/reproduce/modal-parakeet RUN=$(date +%F)
make verify     # re-scores the promoted predictions with no GPU
```
