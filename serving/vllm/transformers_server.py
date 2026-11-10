# transformers server for the models vLLM cannot load (recipe engine
# "transformers"). It stands in for the vLLM engine on the engine bind: shim.py
# forwards its PASSTHROUGH multipart here unchanged, Bearer auth stays in exactly
# one place, and the harness never learns the backend changed. It and a vLLM
# engine never share a bind at the same time. One file for every such model; the
# recipe's `model_class` names the transformers class that loads it, and
# DECODERS below says how that class turns audio into text. A class without an
# entry is refused at start, never guessed.
#
# Serving contract mirrored from vLLM so shim.py needs zero changes:
#   POST /v1/audio/transcriptions   multipart, returns {"text": ...}
#   GET  /health                    200 when the model is loaded
#   GET  /v1/models                 OpenAI shape — shim.py reads the served
#                                   model id from here
#   GET  /version                   {"version", "engine"} — shim.py reports
#                                   "engine" as x_raven.engine
#
# MoonshineStreamingForConditionalGeneration — moonshine-ai/moonshine-streaming-small-de.
# vLLM has no moonshine class (encoder-decoder seq2seq outside Whisper went with
# vLLM's V0). Model-card contract
# (https://huggingface.co/moonshine-ai/moonshine-streaming-small-de, "Usage",
# verified 2026-09-01):
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
#
# GraniteSpeech5ForCTC — ibm-granite/granite-speech-5.0-470m-turboctc. A CTC
# conformer; vLLM 0.33.0 registers no CTC architecture for it. Model-card
# contract (https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc,
# "Usage with transformers", verified 2026-11-02): transformers >= 5.16.0,
# AutoProcessor at 16 kHz, inputs moved to the model's device AND dtype,
# model.generate(**inputs) (one forward pass, argmax per frame, padded frames
# masked — modeling_granite_speech5.py at v5.20.2), processor.batch_decode(...,
# skip_special_tokens=True), whose ParakeetTokenizer collapses the CTC repeats
# and blanks. Non-autoregressive: one label per 80 ms encoder frame, no loop, so
# decoding.max_tokens has nothing to bound here and is not applied.
#
# WORD TIMESTAMPS (response_format=verbose_json, timestamp_granularities[]=word):
# a CTC model emits one label per encoder frame, so the frame a word's first
# label fires on and the frame its last label ends on ARE the model's own word
# times — no alignment model, the same frames the text is read from
# (ctc_word_times). Words are split where a SentencePiece piece opens a word
# ("▁"). The frame length is the clip length over the number of output frames
# (80 ms for turboctc). A decoder without per-frame output (Moonshine) answers
# verbose_json with 400, as vLLM does for a model class without timestamps.
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

# serving/recipe.py (stdlib only), for the precision check at load.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recipe import FLOAT_FORMATS, weights_mismatch  # noqa: E402

logger = logging.getLogger("transformers-server")
logging.basicConfig(level=logging.INFO)

# launch.py passes these from the recipe: model id, HF commit, class, and the
# torch dtype of precision.weights.
MODEL_ID = os.environ.get("TRANSFORMERS_MODEL", "")
REVISION = os.environ.get("TRANSFORMERS_REVISION") or None  # None = the repo's main
MODEL_CLASS = os.environ.get("TRANSFORMERS_CLASS", "")
# Loaded in exactly this dtype, never transformers' default: the recipe declares
# it (precision.weights), the shim reports it, and _load refuses a model whose
# parameters then sit mostly in another one.
DTYPE = os.environ.get("TRANSFORMERS_DTYPE", "")
# Bind where the vLLM engine would bind (launch.py sets both) — see header.
BIND_HOST = os.environ.get("TRANSFORMERS_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("TRANSFORMERS_PORT", "8000"))
DEVICE = os.environ.get("TRANSFORMERS_DEVICE", "cuda")  # "cuda"/"cpu", never an index
# Same house discipline as nemo/server.py: cap this process's VRAM share
# before load (launch.py passes the recipe's gpu_mem).
GPU_MEM_FRACTION = float(os.environ.get("TRANSFORMERS_GPU_MEM_FRACTION", "0.4"))
MAX_UPLOAD_MB = int(os.environ.get("TRANSFORMERS_MAX_UPLOAD_MB", "512"))
FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
SAMPLE_RATE = 16000

app = FastAPI(title="raven.eval transformers server", version="2.0.0")

_model = None
_processor = None
_load_error: str | None = None
_load_lock = threading.Lock()
# One generate() at a time. Requests arrive on the uvicorn threadpool;
# a parallel bench client would otherwise run several generate() calls on the
# one shared module concurrently — the same unguarded-model race the NeMo leg
# had. These models are small and fast; serialising them costs little, and
# unlike the NeMo leg it is not worth a batching queue.
_infer_lock = threading.Lock()


def _load() -> None:
    global _model, _processor, _load_error
    with _load_lock:
        if _model is not None or _load_error is not None:
            return
        try:
            import torch
            import transformers  # noqa: PLC0415 — heavy, load once
            from transformers import AutoProcessor

            declared = {v: k for k, v in FLOAT_FORMATS.items()}.get(DTYPE)
            if declared is None:
                raise ValueError(
                    f"TRANSFORMERS_DTYPE={DTYPE!r}: launch.py passes the recipe's "
                    f"precision.weights as one of {sorted(FLOAT_FORMATS.values())}"
                )
            if MODEL_CLASS not in DECODERS:
                raise ValueError(
                    f"TRANSFORMERS_CLASS={MODEL_CLASS!r}: this server decodes only "
                    f"{sorted(DECODERS)}"
                )

            if DEVICE.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION)
                device = "cuda"
            else:
                device = "cpu"
            t0 = time.monotonic()
            model = getattr(transformers, MODEL_CLASS).from_pretrained(
                MODEL_ID, revision=REVISION, dtype=getattr(torch, DTYPE)
            ).eval()
            model.to(device)
            counts: dict[str, int] = {}
            for p in model.parameters():
                if p.is_floating_point():
                    name = str(p.dtype).removeprefix("torch.")
                    counts[name] = counts.get(name, 0) + p.numel()
            if problem := weights_mismatch(counts, declared):
                raise RuntimeError(f"loaded precision differs from the recipe: {problem}")
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
            "backend": f"transformers {MODEL_CLASS} (NOT vLLM)",
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


def _moonshine(audio, max_tokens: int | None) -> str:
    import torch

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


WORD_OPENER = "\u2581"  # SentencePiece's word-initial marker


def ctc_word_times(
    frame_ids: list[int],
    blank_id: int,
    opens_word,
    decode,
    frame_s: float,
) -> list[dict]:
    """Word timestamps from one clip's per-frame greedy CTC labels.

    ``frame_ids`` is the argmax label of every output frame (repeats and blanks
    included). A run of one label is one token at the frame it starts on;
    ``opens_word(token_id)`` says whether that token begins a new word (a
    SentencePiece piece starting with "▁"); ``decode(token_ids)`` spells a word.
    A word spans from the first frame of its first token to the end of the last
    frame of its last token's run, in seconds (``frame_s`` per frame).
    """
    tokens: list[list[int]] = []  # [id, first frame, last frame]
    prev = None
    for i, t in enumerate(frame_ids):
        if t != blank_id:
            if t == prev and tokens:
                tokens[-1][2] = i
            else:
                tokens.append([t, i, i])
        prev = t
    words: list[dict] = []
    group: list[list[int]] = []

    def flush() -> None:
        if group:
            text = decode([t for t, _, _ in group]).strip()
            if text:
                words.append({"word": text, "start": round(group[0][1] * frame_s, 3),
                              "end": round((group[-1][2] + 1) * frame_s, 3)})

    for tok in tokens:
        if group and opens_word(tok[0]):
            flush()
            group = []
        group.append(tok)
    flush()
    return words


def _ctc(audio, max_tokens: int | None, want_words: bool = False):
    del max_tokens  # one forward pass, no loop to bound (see header)
    import torch

    inputs = _processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    # BatchFeature.to casts only the floating tensors (the features) to the
    # model's dtype; the attention mask keeps its type.
    inputs = inputs.to(_model.device, dtype=_model.dtype)
    with _infer_lock, torch.no_grad():
        ids = _model.generate(**inputs)
    text = _processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if not want_words:
        return text, None
    tok = _processor.tokenizer
    frames = ids[0].tolist()
    frame_s = (len(audio) / SAMPLE_RATE) / max(1, len(frames))
    ids_seen = sorted(set(frames))
    opens = {i: p.startswith(WORD_OPENER)
             for i, p in zip(ids_seen, tok.convert_ids_to_tokens(ids_seen), strict=True)}
    words = ctc_word_times(
        frames, tok.pad_token_id, lambda t: opens.get(t, False),
        lambda group: tok.decode(group, group_tokens=False, skip_special_tokens=True),
        frame_s,
    )
    return text, words


def _moonshine_text(audio, max_tokens: int | None, want_words: bool = False):
    if want_words:
        raise HTTPException(
            status_code=400,
            detail=f"{MODEL_CLASS} has no per-frame output: word timestamps unavailable",
        )
    return _moonshine(audio, max_tokens), None


#: transformers class -> how a loaded model of that class transcribes one
#: 16 kHz mono float32 clip, as ``(text, words or None)``. The recipe's
#: model_class picks the entry.
DECODERS = {
    "MoonshineStreamingForConditionalGeneration": _moonshine_text,
    "GraniteSpeech5ForCTC": _ctc,
}


def _transcribe(wav_path: str, max_tokens: int | None = None, want_words: bool = False):
    import soundfile as sf

    audio, _sr = sf.read(wav_path, dtype="float32", always_2d=False)
    return DECODERS[MODEL_CLASS](audio, max_tokens, want_words)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),  # noqa: ARG001 — OpenAI-compat field; TRANSFORMERS_MODEL rules
    language: str = Form("de"),  # noqa: ARG001 — neither class takes a language
    response_format: str = Form("json"),
    temperature: float = Form(0.0),  # noqa: ARG001 — greedy decode only
    max_completion_tokens: int | None = Form(None),
    timestamp_granularities: list[str] = Form([], alias="timestamp_granularities[]"),
):
    if response_format not in ("json", "text", "verbose_json"):
        raise HTTPException(
            status_code=400, detail=f"response_format {response_format!r} is not served"
        )
    want_words = response_format == "verbose_json" and "word" in timestamp_granularities
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
    fd, tmp = tempfile.mkstemp(prefix="transformers_upload_", suffix=suffix)
    total = 0
    try:
        with os.fdopen(fd, "wb") as fh:
            while chunk := await file.read(1 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds TRANSFORMERS_MAX_UPLOAD_MB={MAX_UPLOAD_MB} MB",
                    )
                fh.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="empty audio upload")
        wav = await run_in_threadpool(_normalize, tmp)
        try:
            t0 = time.monotonic()
            text, words = await run_in_threadpool(
                _transcribe, wav, max_completion_tokens, want_words
            )
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
    body: dict = {"text": text, "x_raven": {"model": MODEL_ID, "wall_seconds": round(wall, 3)}}
    if words is not None:
        body["words"] = words
    return JSONResponse(body)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT)
