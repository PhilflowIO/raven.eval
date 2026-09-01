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

Public HF dataset `flozi00/asr-german-mixed-evals`, subsets `Tuda-De`,
`multilingual_librispeech`, `common_voice_19_0`. The loader prefers a local HF
snapshot and falls back to streaming from huggingface.co. Pin a revision with
`--dataset-revision <hash>` (or `FLOZI_DATASET_REVISION` in `raven_asr.config`)
for a byte-reproducible reference set.

## Per-adapter requirements

| Model key (`MODEL=`)                 | Adapter          | Needs |
|--------------------------------------|------------------|-------|
| `modal/parakeet`, `modal/qwen3-asr`, `modal/voxtral-2507`, `modal/voxtral-2602` | `modal_app` | Modal token (`modal token set …`) + the app deployed on your Modal account (GPU runs there). App names in `raven_asr.config.MODAL_APP_NAMES`. |
| `openai-whisper-1`                   | `openai_whisper` | `OPENAI_API_KEY` |
| `deepgram-nova-2`                    | `deepgram`       | `DEEPGRAM_API_KEY` |
| `voxtral-mini-latest`                | `voxtral_mistral`| `MISTRAL_API_KEY` |
| `primeline/*` (whisper/parakeet)     | `vllm_openai`    | a running vLLM OpenAI-compatible endpoint (self-host GPU); base URL in `VLLM_PRIMELINE_URL`, optional key in `VLLM_PRIMELINE_API_KEY` |
| `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/canary-1b-v2` | `vllm_openai` | a running OpenAI-compatible ASR endpoint (self-host GPU, e.g. a NeMo server exposing `/audio/transcriptions`); base URL in `NEMO_BENCH_URL`, optional key in `NEMO_BENCH_API_KEY` |
| `ibm-granite/granite-speech-4.1-2b-plus`, `CohereLabs/cohere-transcribe-03-2026`, `OpenMOSS-Team/MOSS-Transcribe-Diarize`, `microsoft/Phi-4-multimodal-instruct` | `vllm_openai` | a running OpenAI-compatible ASR endpoint (self-host GPU; any server exposing `/audio/transcriptions` works, e.g. vLLM); base URL in `VLLM_BENCH_URL`, optional key in `VLLM_BENCH_API_KEY` |

The `*_URL` / `*_API_KEY` variables name **your** endpoints — this repo never
ships URLs or key values, only the env-var names the runner reads.

Set the key in your environment before `make reproduce`. Missing-key failures
surface per-utterance in the runner log; the run resumes per-subset from
`.done_<subset>.json` markers, so a re-run after fixing a key doesn't repeat
completed subsets.

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
