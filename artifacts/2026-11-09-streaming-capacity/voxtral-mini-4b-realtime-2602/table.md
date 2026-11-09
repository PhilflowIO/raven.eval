### mistralai/Voxtral-Mini-4B-Realtime-2602 — live capacity

transcription_delay_ms = 480, the checkpoint's own tekken.json value at the pinned revision; chunk 80 ms; 1x RTX 3090 24 GB, shared with a running 1.4 GB text-embedding server (TEI); harness on the same host; threshold p95 final ≤ 1.0 s

| N | p50 final s | p95 final s | p99 final s | p95 first partial s | errors | drops | WER stream % | WER offline % | GPU util mean % | VRAM max MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.34 | 0.39 | 0.52 | 2.68 | 0 | 0 | 3.23 | 2.35 | 21 | 22041 |
| 2 | 0.41 | 0.42 | 0.46 | 4.10 | 0 | 0 | 4.46 | 3.09 | 39 | 22079 |
| 4 | 0.55 | 0.71 | 0.75 | 4.58 | 0 | 0 | 5.17 | 4.84 | 70 | 22131 |
| 8 | 0.74 | 0.88 | 0.95 | 4.71 | 0 | 0 | 5.17 | 5.04 | 91 | 22131 |
| 16 | 0.91 | 0.97 | 1.11 | 5.01 | 0 | 0 | 4.50 | 4.47 | 96 | 22133 |
| 32 | 1.21 | 1.95 | 2.38 | 4.68 | 0 | 0 | 4.91 | 4.78 | 90 | 22329 |
| 64 | 0.98 | 28.13 | 29.76 | 4.65 | 41 | 41 | 5.45 | 5.45 | 82 | 22671 |

Max N with p95 final ≤ 1.0 s and no errors: **16** (one step recorded beyond the first step over the rule).
