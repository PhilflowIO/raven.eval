### nvidia/nemotron-labs-unified-asr-0.6b — live capacity

left 5.6 s / chunk 0.16 s / right 0.40 s: 0.56 s latency, buffered; chunk 160 ms; 1x RTX 3090 24 GB, shared with a running 1.4 GB text-embedding server (TEI); harness on the same host; threshold p95 final ≤ 1.0 s

| N | p50 final s | p95 final s | p99 final s | p95 first partial s | errors | drops | WER stream % | WER offline % | GPU util mean % | VRAM max MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.02 | 0.04 | 0.04 | 2.74 | 0 | 0 | 4.69 | 2.93 | 10 | 4455 |
| 2 | 0.02 | 0.04 | 0.07 | 3.57 | 0 | 0 | 5.68 | 9.06 | 20 | 4521 |
| 4 | 0.02 | 0.04 | 0.05 | 4.19 | 0 | 0 | 4.66 | 5.83 | 36 | 4607 |
| 8 | 0.03 | 0.05 | 0.05 | 4.66 | 0 | 0 | 4.93 | 4.54 | 66 | 4797 |
| 16 | 0.06 | 0.09 | 0.10 | 4.83 | 0 | 0 | 4.79 | 3.97 | 83 | 5133 |
| 32 | 6.46 | 13.79 | 18.08 | 6.88 | 1 | 1 | 5.43 | 3.69 | 73 | 5371 |
| 64 | 11.79 | 25.43 | 28.19 | 9.36 | 11 | 11 | 5.46 | 4.12 | 95 | 5959 |

Max N with p95 final ≤ 1.0 s and no errors: **16** (one step recorded beyond the first step over the rule).
