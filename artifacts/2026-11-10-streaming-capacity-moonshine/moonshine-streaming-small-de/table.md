### moonshine-ai/moonshine-streaming-small-de — live capacity

vendor default: transcript update every 0.5 s of input, ~80 ms encoder lookahead; chunk 100 ms; CPU only: AMD Ryzen 9 7900 (12 cores, 24 threads), 30 GB RAM, idle at start (load1 0.10); the GPU is not used. Vendor int8 build. Harness on the same host; CPU, weights int8; threshold p95 final ≤ 1.0 s

| N | p50 final s | p95 final s | p99 final s | p95 first partial s | errors | drops | WER stream % | WER offline % | CPU util mean % | server RSS max MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.11 | 0.22 | 0.23 | 2.79 | 0 | 0 | 4.11 | 3.52 | 21 | 508 |
| 2 | 0.06 | 0.35 | 0.55 | 3.47 | 0 | 0 | 5.99 | 4.15 | 48 | 838 |
| 4 | 0.21 | 1.11 | 1.34 | 4.39 | 0 | 0 | 6.77 | 5.36 | 88 | 1468 |
| 8 | 1.39 | 2.96 | 3.26 | 6.01 | 0 | 0 | 6.68 | 5.02 | 96 | 1586 |

Max N with p95 final ≤ 1.0 s and no errors: **2** (one step recorded beyond the first step over the rule).
