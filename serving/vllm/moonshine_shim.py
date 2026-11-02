# transformers server for moonshine-ai/moonshine-streaming-small-de (recipe
# engine "transformers"). vLLM cannot serve this model — encoder-decoder
# seq2seq outside Whisper was removed with vLLM's V0 deprecation and there is
# no moonshine registry entry — so this file stands in for the vLLM engine on
# the engine bind: shim.py forwards its PASSTHROUGH multipart here unchanged,
# Bearer auth stays in exactly one place, and the harness never learns the
# backend changed. It and a vLLM engine never share a bind at the same time.
#
# Serving contract mirrored from vLLM so shim.py needs zero changes:
#   POST /v1/audio/transcriptions   multipart, returns {"text": ...}
#   GET  /health                    200 when the model is loaded
#   GET  /v1/models                 OpenAI shape — shim.py reads the served
#                                   model id from here
#   GET  /version                   {"version", "engine"} — shim.py reports
#                                   "engine" as x_raven.engine
#
# Model-card contract (https://huggingface.co/moonshine-ai/moonshine-streaming-small-de,
# "Usage", verified 2026-09-01):
#   * MoonshineStreamingForConditionalGeneration.from_pretrained + AutoProcessor
#     (transformers >= 5.2 — models/moonshine_streaming exists at tag v5.2.0,
#     absent at v5.1.0; serving/vllm/requirements.txt satisfies this).
#   * ALWAYS pass attention_mask — the sliding-window encoder only activates
#     with it ("called without a mask it attends over the whole utterance
#     instead, which is a different model from the one that was trained").
#     The processor returns the mask; we forward everything it returns.
#   * Cap max_new_tokens (repetition-loop guard); the card's own formula is
#     max_new_tokens = int((seq_lens * 6.5 / 16000).max()) + 2.
#   * 16 kHz input; the processor pads to whole 80-sample frames itself.
"""Minimal /v1/audio/transcriptions server for moonshine-streaming-small-de."""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import tempfile
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("moonshine-shim")
logging.basicConfig(level=logging.INFO)

MODEL_ID = os.environ.get(
    "MOONSHINE_MODEL", "moonshine-ai/moonshine-streaming-small-de"
)
# The HF commit to load (launch.py passes the recipe's pin); None = the repo's main.
REVISION = os.environ.get("MOONSHINE_REVISION") or None
# Bind where the vLLM engine would bind (launch.py sets both) — see header.
BIND_HOST = os.environ.get("MOONSHINE_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("MOONSHINE_PORT", "8000"))
DEVICE = os.environ.get("MOONSHINE_DEVICE", "cuda")  # "cuda"/"cpu", never an index
# Same house discipline as nemo/server.py: cap this process's VRAM share
# before load. 0.4 mirrors the vLLM default budget; the 0.45 GB model is far
# below it either way.
GPU_MEM_FRACTION = float(os.environ.get("MOONSHINE_GPU_MEM_FRACTION", "0.4"))
MAX_UPLOAD_MB = int(os.environ.get("MOONSHINE_MAX_UPLOAD_MB", "512"))
FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
SAMPLE_RATE = 16000

app = FastAPI(title="raven.eval moonshine server", version="1.1.0")

_model = None
_processor = None
_load_error: str | None = None
_load_lock = threading.Lock()
# One generate() at a time. Requests arrive on the uvicorn threadpool;
# a parallel bench client would otherwise run several generate() calls on the
# one shared module concurrently — the same unguarded-model race the NeMo leg
# had. This model is small and fast; serialising it costs little, and unlike
# the NeMo leg it is not worth a batching queue (one model of fourteen).
_infer_lock = threading.Lock()


def _load() -> None:
    global _model, _processor, _load_error
    with _load_lock:
        if _model is not None or _load_error is not None:
            return
        try:
            import torch
            from transformers import (  # noqa: PLC0415 — heavy, load once
                AutoProcessor,
                MoonshineStreamingForConditionalGeneration,
            )

            if DEVICE.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION)
                device = "cuda"
            else:
                device = "cpu"
            t0 = time.monotonic()
            model = MoonshineStreamingForConditionalGeneration.from_pretrained(
                MODEL_ID, revision=REVISION
            ).eval()
            model.to(device)
            processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
            _model, _processor = model, processor
            logger.info(
                "loaded %s@%s on %s in %.1fs",
                MODEL_ID, REVISION or "main", device, time.monotonic() - t0,
            )
        except Exception as exc:  # noqa: BLE001 — surface via /health, never swallow
            _load_error = f"{type(exc).__name__}: {exc}"
            logger.error("model load failed: %s", _load_error)


@app.on_event("startup")
def _warmup() -> None:
    threading.Thread(target=_load, daemon=True).start()


@app.get("/health")
@app.get("/healthz")
def health() -> JSONResponse:
    loaded = _model is not None
    return JSONResponse(
        status_code=200 if loaded else 503,
        content={
            "status": "ok" if loaded else "loading",
            "model": MODEL_ID,
            "model_loaded": loaded,
            "model_load_error": _load_error,
            "backend": "transformers-moonshine (NOT vLLM)",
        },
    )


@app.get("/version")
def version() -> JSONResponse:
    """Engine identity, in vLLM's /version shape plus the engine string."""
    import torch
    import transformers

    engine = f"transformers {transformers.__version__} (torch {torch.__version__})"
    return JSONResponse({"version": transformers.__version__, "engine": engine})


@app.get("/v1/models")
def models() -> JSONResponse:
    # OpenAI shape — shim.py's health probe reads data[0].id from here to
    # display the effective family (passthrough) for the served model.
    return JSONResponse(
        {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}
    )


def _normalize(path: str) -> str:
    """Decode any container to mono/16 kHz PCM WAV (same rationale as the
    NeMo leg: make the input shape a fact, not a hope)."""
    out = f"{path}.mono16k.wav"
    proc = subprocess.run(
        [
            FFMPEG, "-v", "error", "-nostdin", "-y",
            "-i", path,
            "-map", "a:0",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            "-f", "wav", out,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not os.path.exists(out):
        with contextlib.suppress(OSError):
            os.remove(out)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unprocessable_audio",
                "reason": "undecodable_container",
                "message": proc.stderr.strip()[:300],
            },
        )
    return out


def _transcribe(wav_path: str, max_tokens: int | None = None) -> str:
    import soundfile as sf
    import torch

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    inputs = _processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE)
    # Model-card repetition-loop guard: ~6.5 tokens per second of audio + 2.
    seq_lens = inputs.attention_mask.sum(dim=-1)
    max_new_tokens = int((seq_lens * 6.5 / SAMPLE_RATE).max().item()) + 2
    if max_tokens is not None:
        # The recipe's decoding.max_tokens, sent by the shim with every request.
        max_new_tokens = min(max_new_tokens, max_tokens)
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with _infer_lock, torch.no_grad():
        generated = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    return _processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),  # noqa: ARG001 — OpenAI-compat field; MOONSHINE_MODEL rules
    language: str = Form("de"),  # noqa: ARG001 — German-only model, nothing to force
    response_format: str = Form("json"),
    temperature: float = Form(0.0),  # noqa: ARG001 — greedy decode only
    max_completion_tokens: int | None = Form(None),
):
    if _model is None:
        if _load_error is not None:
            raise HTTPException(
                status_code=500, detail=f"model load failed: {_load_error}"
            )
        raise HTTPException(
            status_code=503,
            detail="model is still loading — poll /health until model_loaded:true",
        )
    # Spool with a hard cap (mkstemp: collision-safe; chunked read: reject
    # oversized bodies before buffering whole — same as nemo/server.py).
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    fd, tmp = tempfile.mkstemp(prefix="moonshine_upload_", suffix=suffix)
    total = 0
    try:
        with os.fdopen(fd, "wb") as fh:
            while chunk := await file.read(1 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds MOONSHINE_MAX_UPLOAD_MB={MAX_UPLOAD_MB} MB",
                    )
                fh.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="empty audio upload")
        wav = await run_in_threadpool(_normalize, tmp)
        try:
            t0 = time.monotonic()
            text = await run_in_threadpool(_transcribe, wav, max_completion_tokens)
            wall = time.monotonic() - t0
        finally:
            if wav != tmp and os.path.exists(wav):
                os.remove(wav)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"transcription failed: {exc}"
        ) from exc
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)
    if response_format == "text":
        return PlainTextResponse(text)
    return JSONResponse(
        {
            "text": text,
            "x_raven": {"model": MODEL_ID, "wall_seconds": round(wall, 3)},
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT)
