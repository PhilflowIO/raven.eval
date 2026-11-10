# Live-streaming capacity, 2026-11-09

How many simultaneous live speaker tracks one GPU carries. One RTX 3090 (24 GB),
shared with a running 1.4 GB text-embedding server; the harness ran on the same
host. Measured with `python -m raven_asr.stream` at raven.eval `b69bad5`, one
model at a time; WER re-scored with `--rescore` at `ef8787f` (output parser on
both sides, #111 — no number moved).

Each step opens N sessions at once. Every session plays utterances from one
seeded pool (all of FLEURS German test + a seeded 2,000 of the avemio German
mixed test; 2,862 utterances, 8.6 h) in real time, in the model's own chunk
size, for 180 s. **Final latency** is the time from the end of an utterance's
audio to its final transcript; the client announces the end explicitly, so it
contains no end-of-speech detection. **First partial** counts from the moment
the utterance's audio starts, leading silence included. WER is the strict-de
lens of the offline benchmark over the utterances each step played; *offline*
is the same model's offline prediction of exactly those utterances.

| Model | Streaming setting | Max N, p95 final ≤ 1.0 s, no errors | p95 final at that N | p50 first partial | WER stream / offline at that N | Server VRAM |
|---|---|---:|---:|---:|---:|---:|
| Voxtral-Mini-4B-Realtime-2602 (vLLM 0.33.0) | 480 ms delay, 80 ms chunks | **16** | 0.97 s | 1.8 s | 4.50 / 4.47 % | 20.2 GiB (pre-allocated, 0.85) |
| nemotron-3.5-asr-streaming-0.6b (NeMo, cache-aware) | [56, 6] = 560 ms chunks | **≥ 128** (largest step) | 0.08 s | 1.7 s | 7.02 / 8.10 % | 8.9 GiB |
| nemotron-labs-unified-asr-0.6b (NeMo, buffered) | 5.6 / 0.16 / 0.40 s | **16** | 0.09 s | 1.8 s | 4.79 / 3.97 % | 3.6 GiB |
| moonshine-streaming-small-de | — | skipped | | | | |

Where each one breaks: Voxtral at 32 (p95 1.95 s; GPU at 96 % from 16 on) and
at 64 its KV cache is full and 41 of 64 sessions time out. Unified at 32 (p95
13.8 s, one drop): its buffered pipeline recomputes 5.6 s of left context every
160 ms, as its card says. Nemotron 3.5 used 67 % of the GPU at 128 sessions
with p95 0.08 s; its limit lies above the largest step.

Moonshine is skipped: no live serving path at the pinned engines
(`serving/streaming.toml`, `[skipped]`).

Per model: `curve.json` (identity, pool, parameters, every step with GPU and
engine probes), `utterances_n<N>.jsonl` (every utterance with raw timestamps and
text), `table.md` (the full curve).
