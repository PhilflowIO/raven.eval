### nvidia/nemotron-3.5-asr-streaming-0.6b — live capacity

att_context_size [56, 6]: chunk of 7 x 80 ms = 0.56 s, cache-aware; chunk 560 ms; 1x RTX 3090 24 GB, shared with a running 1.4 GB text-embedding server (TEI); harness on the same host; threshold p95 final ≤ 1.0 s

| N | p50 final s | p95 final s | p99 final s | p95 first partial s | errors | drops | WER stream % | WER offline % | GPU util mean % | VRAM max MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.02 | 0.04 | 0.04 | 2.49 | 0 | 0 | 2.93 | 5.28 | 2 | 9111 |
| 2 | 0.02 | 0.04 | 0.04 | 3.38 | 0 | 0 | 4.61 | 7.53 | 4 | 9111 |
| 4 | 0.02 | 0.03 | 0.04 | 3.94 | 0 | 0 | 5.52 | 6.84 | 8 | 9119 |
| 8 | 0.02 | 0.05 | 0.06 | 4.70 | 0 | 0 | 5.71 | 6.89 | 15 | 9119 |
| 16 | 0.02 | 0.04 | 0.05 | 5.06 | 0 | 0 | 6.39 | 7.35 | 25 | 9135 |
| 32 | 0.03 | 0.09 | 0.12 | 4.51 | 0 | 0 | 6.77 | 7.95 | 41 | 9135 |
| 64 | 0.04 | 0.06 | 0.10 | 5.06 | 0 | 0 | 6.98 | 8.15 | 58 | 9225 |
| 128 | 0.05 | 0.08 | 0.09 | 4.53 | 0 | 0 | 7.02 | 8.10 | 67 | 10601 |

Max N with p95 final ≤ 1.0 s and no errors: **128** (every step stepped).
