# Live-streaming capacity, Moonshine on the CPU, 2026-11-10

The fourth curve next to `artifacts/2026-11-09-streaming-capacity/`, same
harness, same seeded pool, same steps and rule (p95 final latency ≤ 1.0 s, no
errors), measured at raven.eval `4da7a6e`.

**Not comparable to the GPU rows without this label:** Moonshine streams through
its vendor runtime, `moonshine-voice` 0.1.6, which runs ONNX Runtime on the CPU
only, with the vendor's int8 build of the German Small Streaming model
(`download.moonshine.ai/model/small-streaming-de/quantized_26_08_24`, int8
per-channel weights and int8 MatMul). Hardware: AMD Ryzen 9 7900 (12 cores,
24 threads), idle at start; the harness ran on the same CPU. Runtime settings
are the vendor's defaults (0.5 s update interval, ONNX Runtime's default thread
pool), audio in the docs' 100 ms chunks.

| Max N, p95 final ≤ 1.0 s, no errors | p95 final at that N | p50 first partial | WER stream / offline at that N | Server RSS |
|---:|---:|---:|---:|---:|
| **2** | 0.35 s | 1.7 s | 5.99 / 4.15 % | 0.8 GiB |

At 4 sessions p95 final latency is 1.11 s with the CPU at 88 %; at 8, 2.96 s.
*Offline* is the same checkpoint's offline run through transformers in fp32
(the offline recipe), so the WER gap mixes the streaming mode with int8.
