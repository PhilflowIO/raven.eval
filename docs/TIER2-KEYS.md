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

## Serving self-hosted models

The 14 self-hosted rows were served with the recipe in `serving/`, and you can
stand up the same servers from it. Every model has one declarative recipe in
`serving/recipes.toml` (engine, dtype, context length, batching defaults,
memory budget, how long audio is chunked, and why); `serving/launch.py` turns a
recipe into the server process. Every recipe pins the Hugging Face commit it
loads (`revision`): a publisher can re-upload under the same repo id, and an
unpinned recipe would then serve other weights than the rows were measured
with. NeMo's `ASRModel.from_pretrained` takes no revision, so a NeMo recipe names
its `.nemo` file instead; the launcher downloads that file at the pin and the
server restores exactly it. Engines are pinned in
`serving/vllm/requirements.txt` (vLLM, FlashInfer with its prebuilt cubins and
JIT kernels for CUDA 13.0, transformers, librosa) and `serving/nemo/requirements.txt` (NeMo). Neither
is part of this repo's own install: each gets its own environment on the GPU host.
Every vLLM model runs on the same engine, the newest vLLM release when it was
pinned; the pin is there so a row can name what served it, and a bump is followed
by re-measuring every vLLM model. vLLM 0.33.0's CUDA 13.0 wheels need an NVIDIA
driver >= 580.

```bash
# once per engine (Python 3.12; see the header of each requirements file)
uv venv --python 3.12 /opt/raven-serving/vllm
uv pip install --python /opt/raven-serving/vllm/bin/python -r serving/vllm/requirements.txt
uv venv --python 3.12 /opt/raven-serving/nemo
uv pip install --python /opt/raven-serving/nemo/bin/python "numba==0.68.1"
uv pip install --python /opt/raven-serving/nemo/bin/python -r serving/nemo/requirements.txt

# a NeMo model: the server speaks the harness's protocol itself
CUDA_VISIBLE_DEVICES=0 RAVEN_SERVE_PORT=8001 \
  /opt/raven-serving/nemo/bin/python serving/launch.py nvidia/canary-1b-v2
export NEMO_BENCH_URL=http://127.0.0.1:8001/v1

# a vLLM (or transformers) model: the engine, plus the shim the harness talks to
CUDA_VISIBLE_DEVICES=0 RAVEN_SERVE_PORT=8000 RAVEN_SERVE_MAX_NUM_SEQS=16 \
  /opt/raven-serving/vllm/bin/python serving/launch.py Qwen/Qwen3-ASR-1.7B &
SHIM_PORT=8002 SHIM_UPSTREAM=http://127.0.0.1:8000 RAVEN_SERVE_MAX_NUM_SEQS=16 \
  /opt/raven-serving/vllm/bin/python serving/vllm/shim.py &
export VLLM_BENCH_URL=http://127.0.0.1:8002/v1
```

`python serving/launch.py <model> --print` shows the exact command without
starting anything. The launcher wants exactly one GPU in `CUDA_VISIBLE_DEVICES`;
`RAVEN_SERVE_HOST`, `RAVEN_SERVE_PORT`, `RAVEN_SERVE_GPU_MEM` and
`RAVEN_SERVE_MAX_NUM_SEQS` override the host-dependent values, and
`RAVEN_SERVE_MIN_FREE_GB` makes it refuse a download that would fill the disk.
The recipes' memory defaults were sized on a 24 GB card shared with other
work; on an empty card give vLLM more.

**What a run records.** Every answer of these servers names its setup in
`x_raven`: `engine` (`vllm 0.33.0`, `nemo 3.0.0 (torch …)`), `recipe` (`id` and
a SHA-256 over that model's recipe and every file that executes it) and
`batching` (NeMo `max_batch` / `wait_ms` / `max_audio_s`, vLLM `max_num_seqs`).
The harness pins the first answer's block, writes it into every throughput pass
and into each `summary.json` result row as `serving`, and **aborts** (exit 3,
nothing recorded) when a later answer — or the earlier passes of a subset being
resumed — names a different one: a score over transcripts from two setups
describes neither. A server that reports no block yields `serving: null` and one
warning. Hosted APIs always record `null`.

**Reproducible to a tolerance, not bit-exact.** The servers batch concurrent
requests, and a batched forward pass can differ from a single-clip pass in the
last bits — enough to flip a character now and then. Measured on FLEURS with
Parakeet-TDT-0.6B-v3: 2 of 862 utterances changed by one character, WER
unchanged, CER +0.0008 pp. A re-run with the same recipe reproduces a
self-hosted row to that tolerance; a re-run with a different recipe hash, engine
or batching is a different measurement.

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
  endpoint serves a different model. On SIGTERM the run first records its
  partial pass (below), then dies by SIGTERM exactly as before.
- **Throughput is recorded per pass.** Each invocation's work on a subset is one
  line in `throughput_<subset>.jsonl`: wall time from its first request,
  requests, successes, seconds of audio transcribed, concurrency, adapter, and
  `interrupted` when it was cut short. Set `RAVEN_EVAL_HARDWARE="1x RTX 3090
  24GB"` (free text) to label the hardware; nothing is auto-detected, so an
  unset label is `null`.
- `MODEL_REV=` is DER-only. A WER model's revision is fixed in the registry: the
  versioned name of a hosted model, and none for a self-hosted endpoint, whose
  loaded weights a client cannot attest.

## Latency and throughput are NOT Tier-2 numbers

`predictions_*.jsonl` records a `latency_s` per utterance, and every
`summary.json` result carries a `throughput` block (`rtfx`, `total_audio_s`,
`total_wall_s`, `utterances_per_s`, `concurrency`, `hardware`, `comparable`, the
`passes` it adds up). Both depend on your hardware, concurrency, region and
network — they are **not** portable. Treat them as Tier-3 transparency (a
measurement of one run, published with its conditions), never as a reproduced
number. Only WER/CER cross the Tier-2 line as reproducible; `make verify` holds
the throughput block to its own arithmetic and nothing more.

## From run to committed proof

```bash
make reproduce METRIC=wer DATASET=common_voice_19_0 MODEL=modal/parakeet
make promote   RESULTS=results/reproduce/modal-parakeet RUN=$(date +%F)
make verify     # re-scores the promoted predictions with no GPU
```
