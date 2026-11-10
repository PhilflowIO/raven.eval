# Live dictation demo

Dictate into a running live-streaming server and see the text as a subtitle
viewer would, plus the latency they feel:

- **first words after speech onset**: from the first loud 20 ms block the page
  hears to the first text from the server;
- **final text after end of speech**: from the last loud block to the final
  transcript, including the pause the page waits before it tells the server the
  sentence ended (slider, default 0.7 s); the server's own share is shown apart.

No build step, no dependencies: one HTML file. The browser needs a secure
context for the microphone, so serve it from localhost.

```bash
# 1. on the GPU host: start a live server (see serving/streaming.toml for ids)
CUDA_VISIBLE_DEVICES=0 RAVEN_SERVE_PORT=8410 \
  <nemo-venv>/bin/python serving/launch.py nvidia-nemotron-3.5-asr-streaming-0.6b --streaming
#    a second model on another port, e.g. 8411:
#    nvidia-nemotron-labs-unified-asr-0.6b-tf32-chunk032
# 2. on your machine: tunnel the ports (no firewall change on the host)
ssh -N -L 127.0.0.1:8410:127.0.0.1:8410 -L 127.0.0.1:8411:127.0.0.1:8411 <gpu-host>
# 3. serve the page and open http://127.0.0.1:8766/
python3 -m http.server 8766 --bind 127.0.0.1 --directory demo/live
```

The model menu maps to ports 8410 (Nemotron 3.5 Streaming) and 8411 (Nemotron
Unified, TF32 + 0.32 s chunk); the host field defaults to 127.0.0.1.

Not measured by the page: the browser's own capture delay (roughly 20-40 ms)
and anything before the microphone. Speech onset and end are found by level
(12 dB above the running noise floor), so a noisy room shifts both.
