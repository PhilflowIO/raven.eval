# NeMo ASR server for raven.eval's self-hosted models (serving/recipes.toml,
# engine "nemo"). Runs ONE model per process, on the one GPU the launcher
# (serving/launch.py) leaves visible. Not a production service.
#
# CONCURRENCY: every model call goes through ONE inference thread
# (batcher.MicroBatcher). Parallel requests are collected into a single
# batched `model.transcribe(paths)` — up to NEMO_MAX_BATCH clips or whatever
# arrived within NEMO_BATCH_WAIT_MS of the first — and the results are fanned
# back out. Request threads only spool, normalise and wait; they never touch
# the model, so there is no race on its shared decoding state.
#
# Batching is not free for the numbers: a batched forward pass can differ from
# a single-clip pass in the last bits, which can flip a character. Measured on
# FLEURS with parakeet-tdt-0.6b-v3: 2 of 862 transcripts changed by one
# character, WER unchanged, CER +0.0008 pp. Every response therefore names
# the batching parameters (x_raven.batching) next to the recipe and engine.
"""OpenAI-compatible /v1/audio/transcriptions server for arbitrary NeMo ASR models.

Serves whatever `BENCH_MODEL` names — e.g. `nvidia/parakeet-tdt-0.6b-v3` or
`nvidia/canary-1b-v2` — restored from a local .nemo checkpoint via
`ASRModel.restore_from` when `BENCH_MODEL_PATH` is set (launch.py downloads it
at the recipe's pinned revision and names that revision in
`BENCH_MODEL_REVISION`), or, for a server started by hand without one, pulled
from the hub via `ASRModel.from_pretrained` — which takes no revision, so it
loads whatever the repo's main is and /health reports `model_revision: null`.

Model families:
  * Parakeet / FastConformer-TDT (EncDecRNNTBPEModel & friends): plain
    `.transcribe([path], timestamps=True)`, long-form local attention +
    server-side chunking of very long audio.
  * Canary / AED (EncDecMultiTaskModel): the prompt kwargs
    `source_lang="de", target_lang="de", pnc="yes"` are passed straight to
    `.transcribe(...)` — NeMo 2.7.0 forwards `**prompt` as the "legacy
    Canary-1B API" (aed_multitask_models.py, transcribe() docstring). Text
    only: AED timestamp support runs through a different prompt slot and the
    bench scores text, so we keep it minimal on purpose.

  * Prompt-conditioned transducers (EncDecRNNTBPEModelWithPrompt: the
    Nemotron ASR models): `.transcribe(paths, target_lang="de-DE")`. The
    pinned NeMo commit (requirements.txt) carries `target_lang` into the
    transcribe dataloader and conditions every utterance on it; NeMo 3.0.0
    dropped it and could only run language auto-detection.

Language-ID hazard (why `x_raven.language_forced` exists): parakeet-tdt-0.6b-v3
auto-detects the language and offers NO way to force German — a strongly
accented or code-switched clip can silently come back in the wrong language.
Canary and the prompt-conditioned Nemotron models ARE forced to German. The
response flags which regime produced the text so downstream analysis can
split on it.

GPU DISCIPLINE:
  * Device selection happens EXCLUSIVELY via CUDA_VISIBLE_DEVICES in the
    process environment (launch.py requires exactly one index). This process
    must never pick a device index itself — `BENCH_DEVICE` is "cuda"/"cpu",
    never "cuda:1" — so a config typo cannot land it on another card.
  * `torch.cuda.set_per_process_memory_fraction(BENCH_GPU_MEM_FRACTION)` is
    applied BEFORE model load (after `torch.cuda.init()`), so a VRAM-hungry
    candidate OOMs itself already during the load instead of starving
    whatever else shares the card and only being capped afterwards.

Endpoints: /v1/audio/transcriptions (OpenAI-compat) + /health + /healthz.
A failed model load is surfaced in /health (`model_loaded: false` +
`model_load_error`), never swallowed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections.abc import Sequence

logger = logging.getLogger("raven-stt-bench")
logging.basicConfig(level=logging.INFO)

from batcher import MicroBatcher
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

# ─── Env contract (README.md is the human-facing copy of this) ────────────────
BENCH_MODEL = os.environ.get("BENCH_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
# Optional local checkpoint override. When set it MUST exist — we never fall
# back to a hub pull, because a bench that silently measured a different
# artifact than the one named is worse than a bench that failed loudly.
BENCH_MODEL_PATH = os.environ.get("BENCH_MODEL_PATH", "")
# The HF commit BENCH_MODEL_PATH was downloaded at (launch.py passes the recipe's
# pin). Reported only alongside a local checkpoint: a from_pretrained pull
# follows the repo's main, whatever this says.
BENCH_MODEL_REVISION = (
    (os.environ.get("BENCH_MODEL_REVISION") or None) if BENCH_MODEL_PATH else None
)
DEVICE = os.environ.get("BENCH_DEVICE", "cuda")
GPU_MEM_FRACTION = float(os.environ.get("BENCH_GPU_MEM_FRACTION", "0.5"))
# Long-form local attention for the Parakeet family (ignored for Canary, whose
# AED long-form path is NeMo's built-in chunking — see _transcribe_window).
LONGFORM = os.environ.get("BENCH_LONGFORM", "true").lower() in {"1", "true", "yes"}
# Endpoint auth: if STT_API_KEY is set, the transcription route requires a
# matching `Authorization: Bearer <key>` — defence-in-depth on top of the
# bind address.
API_KEY = os.environ.get("STT_API_KEY", "")
# Upload cap (bench clips are minutes, not movies) — same default as the vLLM
# leg's SHIM_MAX_UPLOAD_MB.
MAX_UPLOAD_MB = int(os.environ.get("BENCH_MAX_UPLOAD_MB", "512"))
# Micro-batching. MAX_BATCH bounds GPU memory per call; WAIT_MS is the
# latency a lone request pays waiting for company (20 ms against clips that
# take 50-500 ms to decode). Clips longer than BATCH_MAX_AUDIO_S always run
# alone: Canary's AED long-form chunking only engages for a single input
# (NeMo aed_multitask_models.transcribe: enable_chunking requires one audio or
# batch_size=1), and Canary's own chunks are 30-40 s — batching a longer clip
# would silently change its transcript. 40 s keeps every bench utterance of
# normal length batchable.
MAX_BATCH = int(os.environ.get("NEMO_MAX_BATCH", "16"))
BATCH_WAIT_MS = float(os.environ.get("NEMO_BATCH_WAIT_MS", "20"))
BATCH_MAX_AUDIO_S = float(os.environ.get("NEMO_BATCH_MAX_AUDIO_S", "40"))

# Serving identity (reported in every response as x_raven.engine/.recipe/
# .batching): which recipe of serving/recipes.toml this process runs, hashed
# over the recipe and the files that execute it (serving/recipe.py). launch.py
# names the recipe in RAVEN_SERVING_RECIPE; a server started by hand falls back
# to looking its model up. None when this file runs outside a raven.eval
# checkout — the response then says so instead of guessing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recipe import decoding as _recipe_decoding  # noqa: E402
from recipe import identity as _recipe_identity  # noqa: E402
from recipe import precision as _recipe_precision  # noqa: E402
from recipe import weights_mismatch  # noqa: E402

RECIPE = _recipe_identity(os.environ.get("RAVEN_SERVING_RECIPE") or BENCH_MODEL)
# The recipe's decoding table (x_raven.decoding). This server never changes a
# checkpoint's decoding; it checks the restored model against the table and
# refuses to serve when they differ (decoding_mismatch), so what a response
# reports is what decoded it.
DECODING = _recipe_decoding(os.environ.get("RAVEN_SERVING_RECIPE") or BENCH_MODEL)
# The recipe's precision table (x_raven.precision). This server casts nothing:
# restore_from keeps the checkpoint's stored format and transcribe() runs no
# autocast. It checks the restored parameters against precision.weights and
# refuses to serve when they differ, so what a response reports is what ran.
PRECISION = _recipe_precision(os.environ.get("RAVEN_SERVING_RECIPE") or BENCH_MODEL)


def numel_by_dtype(model) -> dict[str, int]:
    """Floating-point parameter elements of ``model`` per dtype name (``"float32"``)."""
    counts: dict[str, int] = {}
    for p in model.parameters():
        if p.is_floating_point():
            name = str(p.dtype).removeprefix("torch.")
            counts[name] = counts.get(name, 0) + p.numel()
    return counts


def decoding_mismatch(cfg, expected: dict | None) -> str:
    """How a checkpoint's ``cfg.decoding`` differs from the recipe table, or ``""``.

    ``cfg`` is the model's decoding config (OmegaConf or dict). Transducers are
    read at ``greedy.max_symbols``, Canary at ``beam.{beam_size,len_pen,
    max_generation_delta}`` — where NeMo keeps them.
    """
    if expected is None:
        return ""
    get = (lambda node, key: node.get(key) if hasattr(node, "get") else None)
    strategy = get(cfg, "strategy")
    if strategy != expected.get("strategy"):
        return f"strategy {strategy!r} != recipe {expected.get('strategy')!r}"
    where = {"max_symbols": "greedy", "beam_size": "beam", "len_pen": "beam",
             "max_generation_delta": "beam"}
    for key, section in where.items():
        if key not in expected:
            continue
        node = get(cfg, section)
        got = get(node, key) if node is not None else None
        if got is None or float(got) != float(expected[key]):
            return f"{section}.{key} {got!r} != recipe {expected[key]!r}"
    return ""
BATCHING = {"max_batch": MAX_BATCH, "wait_ms": BATCH_WAIT_MS, "max_audio_s": BATCH_MAX_AUDIO_S}
_engine: str | None = None


def _engine_version() -> str | None:
    """``"nemo <version> (torch <version>)"`` once the model stack is imported."""
    global _engine
    if _engine is None and "nemo" in sys.modules and "torch" in sys.modules:
        import nemo
        import torch

        _engine = f"nemo {nemo.__version__} (torch {torch.__version__})"
    return _engine


# What every NeMo ASR checkpoint we bench expects: 16 kHz mono. Both the
# parakeet-v3 and canary-1b-v2 model cards state 16 kHz monochannel input.
# Trailing language tag the Nemotron prompt models emit ("... <de-DE>").
import re as _re

_NEMOTRON_LANG_TAG = _re.compile(r"<[a-z]{2}-[A-Z]{2}>\s*$")

MODEL_SAMPLE_RATE = 16000
MODEL_CHANNELS = 1

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")

# Machine-readable 422 contract: clients key retry decisions off these strings.
ERR_UNPROCESSABLE = "unprocessable_audio"
REASON_UNDECODABLE = "undecodable_container"
REASON_NO_AUDIO_STREAM = "no_audio_stream"
REASON_EMPTY_AUDIO = "empty_audio"


class UnprocessableAudio(Exception):
    """Input the server can never make sense of — a 4xx, not a 5xx."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def _model_is_canary(model=None) -> bool:
    """Canary family ⇒ AED prompt kwargs + forced language.

    Two signals, either suffices: the model id names canary, or the loaded
    class is NeMo's AED multitask model (EncDecMultiTaskModel / a Canary
    subclass). The class check catches a local .nemo checkpoint whose
    BENCH_MODEL label does not say "canary".
    """
    if "canary" in BENCH_MODEL.lower():
        return True
    if model is not None:
        cls = type(model).__name__
        return "EncDecMultiTaskModel" in cls or "Canary" in cls
    return False


def require_key(authorization: str = Header(default="")) -> None:
    if not API_KEY:
        return  # auth disabled — acceptable only on a private bind
    if authorization.removeprefix("Bearer ").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


app = FastAPI(title="Raven STT bench (NeMo)", version="1.0.0")

_model = None
_is_canary: bool = _model_is_canary()  # refined from the class after load
#: The prompt key every utterance of a prompt-conditioned model is forced to.
#: Both Nemotron prompt checkpoints map "de-DE" (and "de") to prompt 9.
PROMPT_LANGUAGE = "de-DE"


def _model_is_prompted(model=None) -> bool:
    """Prompt-conditioned transducer ⇒ transcribe(target_lang=PROMPT_LANGUAGE).

    By name before the load, by class after it (the class decides: a model
    without the prompt mixin would ignore target_lang)."""
    if model is not None:
        return "WithPrompt" in type(model).__name__
    return "nemotron" in BENCH_MODEL.lower()


_is_prompted: bool = _model_is_prompted()  # refined from the class after load
_load_error: str | None = None
_load_seconds: float | None = None
_load_lock = threading.Lock()


def _load_model() -> None:
    global _model, _is_canary, _is_prompted, _load_error, _load_seconds
    with _load_lock:
        if _model is not None or _load_error is not None:
            return
        try:
            import nemo.collections.asr as nemo_asr
            import torch

            # GPU DISCIPLINE: cap this process's share of the card BEFORE the
            # model load — a fraction applied only after the load caps nothing
            # but subsequent allocations, so a VRAM-hungry candidate could
            # already have starved the card while loading. torch.cuda.init()
            # first: the fraction attaches to the current CUDA context. Which
            # physical card that is was decided by CUDA_VISIBLE_DEVICES in the
            # process environment, never here — this process only ever sees
            # logical device 0. See module docstring.
            if DEVICE.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION)

            t0 = time.monotonic()
            if BENCH_MODEL_PATH:
                if not os.path.exists(BENCH_MODEL_PATH):
                    raise FileNotFoundError(
                        f"BENCH_MODEL_PATH={BENCH_MODEL_PATH!r} does not exist — "
                        "refusing to fall back to a hub pull (the bench must "
                        "measure exactly the artifact named)"
                    )
                m = nemo_asr.models.ASRModel.restore_from(
                    BENCH_MODEL_PATH, map_location=DEVICE
                )
            else:
                m = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=BENCH_MODEL, map_location=DEVICE
                )
            if problem := decoding_mismatch(m.cfg.get("decoding"), DECODING):
                raise RuntimeError(
                    f"checkpoint decoding differs from its recipe: {problem}"
                )
            if PRECISION is not None and (
                problem := weights_mismatch(numel_by_dtype(m), PRECISION["weights"])
            ):
                raise RuntimeError(f"checkpoint precision differs from its recipe: {problem}")
            _is_canary = _model_is_canary(m)
            _is_prompted = _model_is_prompted(m)
            if LONGFORM and not _is_canary:
                # Parakeet family only. MUST run before transcribe — this is
                # what makes NeMo emit segment timestamps for long audio (not
                # just avoid OOM).
                m.change_attention_model("rel_pos_local_attn", [256, 256])
                m.change_subsampling_conv_chunking_factor(1)

            _load_seconds = time.monotonic() - t0
            _model = m
            logger.info(
                "loaded %s@%s (%s, canary=%s) in %.1fs",
                BENCH_MODEL, BENCH_MODEL_REVISION or "main", type(m).__name__,
                _is_canary, _load_seconds,
            )
        except Exception as exc:  # noqa: BLE001 — surface GPU/NeMo errors via /health
            _load_error = f"{type(exc).__name__}: {exc}"


@app.on_event("startup")
def _warmup() -> None:
    threading.Thread(target=_load_model, daemon=True).start()
    _batcher.start()


def _get_model():
    if _model is None and _load_error is None:
        _load_model()
    if _load_error is not None:
        raise RuntimeError(f"model load failed: {_load_error}")
    assert _model is not None
    return _model


_in_flight = 0
_in_flight_lock = threading.Lock()
_last_activity = time.monotonic()


@contextlib.contextmanager
def _track_request():
    """Count one in-flight job for /health.

    Nothing reaps this box, but the bench driver polls /health between runs to
    know when the previous file is really done — keep the counter honest.
    """
    global _in_flight, _last_activity
    with _in_flight_lock:
        _in_flight += 1
    try:
        yield
    finally:
        with _in_flight_lock:
            _in_flight = max(_in_flight - 1, 0)
            _last_activity = time.monotonic()


def _load_snapshot() -> tuple[int, float]:
    with _in_flight_lock:
        in_flight = _in_flight
        idle = 0.0 if in_flight else time.monotonic() - _last_activity
    return in_flight, idle


def _wav_seconds(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return 0.0


def _f(value, default: float = 0.0) -> float:
    """Coerce a possibly-missing/None NeMo timestamp to float seconds."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_timestamps(hyp) -> dict:
    """Map a NeMo Hypothesis `.timestamp` dict to {segment: [...], word: [...]}.

    Returns {} when the model produced no
    timestamps (Canary text-only run, older NeMo), so the response degrades to
    text-only rather than erroring.
    """
    raw = getattr(hyp, "timestamp", None) or {}
    if not isinstance(raw, dict):
        return {}

    def _norm(entries, text_key: str) -> list[dict]:
        out: list[dict] = []
        for e in entries or []:
            if not isinstance(e, dict):
                continue
            text = (
                e.get(text_key)
                or e.get("segment")
                or e.get("word")
                or e.get("char")
                or e.get("text")
                or ""
            )
            out.append(
                {
                    text_key: text,
                    "start": _f(e.get("start", e.get("start_time"))),
                    "end": _f(e.get("end", e.get("end_time"))),
                    "start_offset": e.get("start_offset"),
                    "end_offset": e.get("end_offset"),
                }
            )
        return out

    result: dict = {}
    segments = _norm(raw.get("segment"), "segment")
    words = _norm(raw.get("word"), "word")
    if segments:
        result["segment"] = segments
    if words:
        result["word"] = words
    return result


def probe_audio(path: str) -> dict:
    """Return ``{channels, sample_rate, duration}`` for the first audio stream."""
    if shutil.which(FFPROBE) is None:
        raise RuntimeError(f"{FFPROBE} not found — the bench box must have ffmpeg")
    proc = subprocess.run(
        [
            FFPROBE, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels,sample_rate:format=duration",
            "-of", "json", path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise UnprocessableAudio(
            REASON_UNDECODABLE, f"ffprobe could not read the input: {proc.stderr.strip()[:300]}"
        )
    try:
        parsed = json.loads(proc.stdout or "{}")
    except ValueError as exc:
        raise UnprocessableAudio(
            REASON_UNDECODABLE, f"ffprobe returned unparseable output: {exc}"
        ) from exc
    streams = parsed.get("streams") or []
    if not streams:
        raise UnprocessableAudio(
            REASON_NO_AUDIO_STREAM, "the uploaded file carries no audio stream"
        )
    stream = streams[0]
    try:
        duration = float((parsed.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "channels": int(stream.get("channels") or 0),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "duration": duration,
    }


def normalize_audio(path: str) -> tuple[str, dict]:
    """Decode any container to mono/16 kHz PCM WAV for inference.

    The model signature is mono 16 kHz, and
    normalising here makes the input shape a fact instead of a hope — bench
    corpora arrive in whatever container the dataset ships.
    """
    info = probe_audio(path)
    if info["duration"] <= 0.0 and os.path.getsize(path) == 0:
        raise UnprocessableAudio(REASON_EMPTY_AUDIO, "the uploaded file is empty")

    already_ok = (
        info["channels"] == MODEL_CHANNELS
        and info["sample_rate"] == MODEL_SAMPLE_RATE
        and path.lower().endswith(".wav")
    )
    if already_ok:
        return path, {**info, "normalized": False}

    out = f"{path}.mono16k.wav"
    proc = subprocess.run(
        [
            FFMPEG, "-v", "error", "-nostdin", "-y",
            "-i", path,
            "-map", "a:0",
            "-ac", str(MODEL_CHANNELS),
            "-ar", str(MODEL_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            "-f", "wav", out,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not os.path.exists(out):
        if os.path.exists(out):
            os.remove(out)
        raise UnprocessableAudio(
            REASON_UNDECODABLE,
            f"ffmpeg could not decode the input to mono/{MODEL_SAMPLE_RATE} Hz: "
            f"{proc.stderr.strip()[:300]}",
        )
    return out, {**info, "normalized": True}


@app.get("/health")
@app.get("/healthz")
def health() -> JSONResponse:
    in_flight, idle_seconds = _load_snapshot()
    # GPU visibility proof for the operator: with one index in
    # CUDA_VISIBLE_DEVICES this MUST report cuda_device_count=1 — a 2 here
    # means the process can see more than one card. torch is only queried when
    # already imported (the warmup thread imports it), so /health never blocks
    # a poll behind the heavy first torch import.
    cuda_device_count = None
    if "torch" in sys.modules:
        import torch

        cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return JSONResponse(
        {
            "status": "ok",
            # `model_loaded` means ASR readiness and NOTHING else; a load
            # failure shows up here as false + model_load_error, never as a
            # silently-empty server.
            "model_loaded": _model is not None,
            "model_load_error": _load_error,
            "model_load_seconds": round(_load_seconds, 2) if _load_seconds else None,
            "model": BENCH_MODEL,
            "model_path": BENCH_MODEL_PATH or None,
            "model_revision": BENCH_MODEL_REVISION,
            "engine": _engine_version(),
            "recipe": RECIPE,
            "canary": _is_canary,
            "language_forced": _is_canary or _is_prompted,
            "device": DEVICE,
            "gpu_mem_fraction": GPU_MEM_FRACTION,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_count": cuda_device_count,
            "auth_enabled": bool(API_KEY),
            "longform": LONGFORM,
            "batching": {
                "max_batch": MAX_BATCH,
                "wait_ms": BATCH_WAIT_MS,
                "max_audio_s": BATCH_MAX_AUDIO_S,
                **_batcher.stats.snapshot(),
            },
            "in_flight": in_flight,
            "idle_seconds": round(idle_seconds, 1),
        }
    )


# Parakeet long-audio chunking: past the threshold the
# file is windowed, each window transcribed with timestamps, and stitched to
# absolute times. Canary never enters this path — NeMo's AED transcribe() does
# its own long-form chunking (MultiTaskTranscriptionConfig.enable_chunking
# defaults to True, which requires a single input — clips longer than
# NEMO_BATCH_MAX_AUDIO_S therefore always run alone), and without reliable AED
# timestamps we could not de-duplicate our own overlap windows anyway.
CHUNK_THRESHOLD_S = float(os.environ.get("BENCH_CHUNK_THRESHOLD_S", "900"))   # 15 min
CHUNK_LEN_S = float(os.environ.get("BENCH_CHUNK_LEN_S", "600"))               # 10 min
CHUNK_OVERLAP_S = float(os.environ.get("BENCH_CHUNK_OVERLAP_S", "3"))


def _hyp_text(h) -> str:
    return (getattr(h, "text", h if isinstance(h, str) else "") or "").strip()


def _as_hyp_list(hyps, n: int) -> list:
    """Normalise NeMo's transcribe() return to one hypothesis per input.

    RNN-T models of older NeMo lines return `(best, all)`; current ones return
    a list. A length mismatch is a hard error — fanning out a short list would
    hand one request another request's transcript.
    """
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    hyps = list(hyps or [])
    if len(hyps) != n:
        raise RuntimeError(f"transcribe returned {len(hyps)} hypotheses for {n} inputs")
    return hyps


def _transcribe_paths(paths: Sequence[str]) -> list[tuple[str, dict]]:
    """ONE NeMo transcribe over `paths` — called only by the inference thread.

    Same per-family behaviour as the former single-clip path, applied to a
    batch; order of results == order of `paths` (NeMo's documented contract,
    re-checked by _as_hyp_list's length guard).
    """
    model = _get_model()
    batch = list(paths)
    n = len(batch)
    if _is_canary:
        # Canary / AED: force German in + German out, keep punctuation and
        # capitalisation. These kwargs land in `**prompt` of
        # EncDecMultiTaskModel.transcribe(), documented as the "legacy
        # Canary-1B API source_lang=<lang>, target_lang=<lang>, etc."
        # (NeMo v2.7.0 aed_multitask_models.py; canary-1b-v2 model card shows
        # the same source_lang/target_lang call). Text only — no timestamps=
        # kwarg — because AED timestamping is a separate prompt slot and the
        # bench scores text.
        hyps = model.transcribe(
            batch, batch_size=n, source_lang="de", target_lang="de", pnc="yes"
        )
        return [(_hyp_text(h), {}) for h in _as_hyp_list(hyps, n)]
    if _is_prompted:
        # Forced to German: target_lang reaches the transcribe dataloader,
        # which conditions every utterance on that prompt (NeMo
        # rnnt_bpe_models_prompt.py _transcribe_input_manifest_processing and
        # _setup_transcribe_dataloader at the pinned commit).
        hyps = model.transcribe(batch, batch_size=n, target_lang=PROMPT_LANGUAGE)
        # The model may append its locale tag (e.g. "<de-DE>") after the
        # terminal punctuation - transport, not speech.
        return [
            (_NEMOTRON_LANG_TAG.sub("", _hyp_text(h)).strip(), {})
            for h in _as_hyp_list(hyps, n)
        ]
    try:
        hyps = model.transcribe(batch, batch_size=n, timestamps=True)
    except TypeError:
        # NeMo predates the unified `timestamps=` kwarg — text only.
        hyps = model.transcribe(batch, batch_size=n)
    except ValueError as exc:
        # TDT zero-duration-token timestamp bug on very short/edge audio
        # (NeMo #14427). In a batch we cannot tell which clip tripped it, so
        # re-raise: the batcher retries every clip alone, and only the clip
        # that really trips it loses its timestamps (the single-clip branch
        # below) — the others keep theirs, exactly as before batching.
        if n > 1:
            raise
        logger.warning("timestamps=True raised %s; retrying text-only", exc)
        hyps = model.transcribe(batch, batch_size=1)
    return [(_hyp_text(h), _extract_timestamps(h)) for h in _as_hyp_list(hyps, n)]


_batcher = MicroBatcher(
    lambda items: _transcribe_paths([path for path, _audio_s in items]),
    max_batch=MAX_BATCH,
    max_wait_s=BATCH_WAIT_MS / 1000.0,
    is_solo=lambda item: item[1] > BATCH_MAX_AUDIO_S or item[1] <= 0.0,
)


def _transcribe_window(path: str) -> tuple[str, dict, int]:
    """Transcribe one audio window via the inference thread (blocks the caller).

    Returns (text, timestamps, batch_size) — the size of the batched call the
    clip actually rode in, reported in x_raven so a run can prove it batched.
    Unknown duration (0.0) runs alone: never guess a clip into a batch.
    """
    (text, ts), batch_size = _batcher.run((path, _wav_seconds(path)))
    return text, ts, batch_size


def _shift(entries: list[dict], offset_s: float) -> list[dict]:
    out = []
    for e in entries:
        e = dict(e)
        e["start"] = _f(e.get("start")) + offset_s
        e["end"] = _f(e.get("end")) + offset_s
        out.append(e)
    return out


def _transcribe_chunked(path: str, audio_s: float) -> tuple[str, dict]:
    """Window a long file, transcribe each, stitch absolute timestamps (Parakeet)."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    step = max(CHUNK_LEN_S - CHUNK_OVERLAP_S, 1.0)
    starts = []
    s = 0.0
    while s < audio_s:
        starts.append(s)
        s += step
    logger.warning(
        "chunking %.0fs audio into %d window(s) of %.0fs", audio_s, len(starts), CHUNK_LEN_S
    )

    all_seg: list[dict] = []
    all_word: list[dict] = []
    texts: list[str] = []
    for w_start in starts:
        w_end = min(w_start + CHUNK_LEN_S, audio_s)
        seg_data = data[int(w_start * sr): int(w_end * sr)]
        fd, tmp = tempfile.mkstemp(prefix="bench_chunk_", suffix=".wav")
        os.close(fd)  # soundfile reopens by path; mkstemp just owns the name
        try:
            sf.write(tmp, seg_data, sr, subtype="PCM_16")
            text, ts, _batch_size = _transcribe_window(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if text:
            texts.append(text)
        # Drop segments/words that start inside the overlap tail of the
        # PREVIOUS window, so overlapped speech is not duplicated.
        cutoff = 0.0 if w_start == 0 else CHUNK_OVERLAP_S
        for seg in _shift(ts.get("segment", []), w_start):
            if seg["start"] >= w_start + cutoff or w_start == 0:
                all_seg.append(seg)
        for word in _shift(ts.get("word", []), w_start):
            if word["start"] >= w_start + cutoff or w_start == 0:
                all_word.append(word)

    all_seg.sort(key=lambda x: x["start"])
    all_word.sort(key=lambda x: x["start"])
    result: dict = {}
    if all_seg:
        result["segment"] = all_seg
    if all_word:
        result["word"] = all_word
        # Reconstruct the long-form text from the DE-DUPLICATED word stream:
        # joining the raw chunk texts would transcribe every overlap seam
        # twice (up to CHUNK_OVERLAP_S of speech per seam) and inflate WER.
        # Parakeet runs with timestamps=True, so this is the normal path.
        stitched = " ".join(w["word"] for w in all_word if w.get("word")).strip()
    else:
        # Fallback — timestamps absent (pre-unified-kwarg NeMo, or the
        # NeMo#14427 text-only retry): join the chunk texts. RESIDUAL
        # DUPLICATION: each seam repeats up to CHUNK_OVERLAP_S seconds of
        # speech; without word times we cannot de-duplicate, and a bench row
        # produced this way must be read with that caveat.
        stitched = " ".join(texts).strip()
    return stitched, result


def _transcribe(path: str) -> tuple[str, float, dict, int]:
    t0 = time.monotonic()
    audio_s = _wav_seconds(path)
    if audio_s > CHUNK_THRESHOLD_S and not _is_canary:
        text, ts = _transcribe_chunked(path, audio_s)
        batch_size = 1  # every long-form window runs alone (> BATCH_MAX_AUDIO_S)
    else:
        text, ts, batch_size = _transcribe_window(path)
    wall = time.monotonic() - t0
    return text, wall, ts, batch_size


def _respond(
    text: str,
    wall: float,
    audio_s: float,
    response_format: str,
    timestamps: dict | None = None,
    input_info: dict | None = None,
    batch_size: int = 1,
):
    if response_format == "text":
        return PlainTextResponse(text)
    body = {
        "task": "transcribe",
        # Only claim a language when the model was actually FORCED to it
        # (Canary). Parakeet auto-detects and can silently answer in another
        # language — a hard-coded "de" here would launder that into a fact.
        "language": "de" if (_is_canary or _is_prompted) else None,
        "text": text,
        "duration": round(audio_s, 3),
        "x_raven": {
            "model": BENCH_MODEL,
            # False for parakeet-tdt-0.6b-v3: the model auto-detects the
            # language and CANNOT be forced to German, so a wrong-language
            # transcript is possible and downstream analysis must be able to
            # see which regime produced each row. True for Canary
            # (source_lang/target_lang pin it to German) and the prompt-
            # conditioned Nemotron models (target_lang="de-DE").
            "language_forced": _is_canary or _is_prompted,
            # Wall includes the queue wait for a batch slot; under a parallel
            # client it is per-request latency, not GPU time per clip.
            "wall_seconds": round(wall, 3),
            "batch_size": batch_size,
            "audio_seconds": round(audio_s, 3),
            "realtime_factor": round(audio_s / wall, 2) if wall else None,
            "input": input_info or {},
            # Serving identity: the same for every response of this process.
            # The harness pins the first one it sees and aborts on a change.
            "engine": _engine_version(),
            "recipe": RECIPE,
            "batching": BATCHING,
            "decoding": DECODING,
            "precision": PRECISION,
        },
    }
    if timestamps:
        body["timestamps"] = timestamps
    return JSONResponse(body)


def _unprocessable(exc: UnprocessableAudio) -> HTTPException:
    """422, machine-readable — never 5xx for input the server cannot process."""
    return HTTPException(
        status_code=422,
        detail={"error": ERR_UNPROCESSABLE, "reason": exc.reason, "message": exc.message},
    )


def _run(path: str) -> tuple[str, float, float, dict, dict, int]:
    """Normalise → transcribe. Single shared path.

    Runs on a request thread: ffmpeg normalisation is parallel across
    requests; only the model call is funnelled through the inference thread.
    """
    norm, input_info = normalize_audio(path)
    try:
        text, wall, timestamps, batch_size = _transcribe(norm)
    finally:
        if norm != path and os.path.exists(norm):
            os.remove(norm)
    audio_s = float(input_info.get("duration") or 0.0)
    return text, wall, audio_s, timestamps, input_info, batch_size


async def _save_tmp(file: UploadFile) -> str:
    """Spool the upload to a collision-safe temp file, capped at MAX_UPLOAD_MB.

    mkstemp instead of a timestamp-named path (two requests in the same
    millisecond would clobber each other), chunked read instead of one
    read() (the cap must reject an oversized body before it is buffered
    whole), and try/finally so a failed spool never leaves the file behind.
    """
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    fd, tmp = tempfile.mkstemp(prefix="bench_upload_", suffix=suffix)
    total = 0
    try:
        with os.fdopen(fd, "wb") as fh:
            while chunk := await file.read(1 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds BENCH_MAX_UPLOAD_MB={MAX_UPLOAD_MB} MB",
                    )
                fh.write(chunk)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    return tmp


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("bench"),  # noqa: ARG001 — OpenAI-compat field; BENCH_MODEL rules
    language: str = Form("de"),  # noqa: ARG001 — see language_forced in the response
    response_format: str = Form("json"),
    _auth: None = Depends(require_key),
):
    # Readiness gate: while the warmup thread is still loading, answer 503
    # instead of blocking the request on the load lock — the bench driver
    # polls /health and retries, it must not hang a request for minutes.
    if _model is None:
        if _load_error is not None:
            raise HTTPException(status_code=500, detail=f"model load failed: {_load_error}")
        raise HTTPException(
            status_code=503,
            detail="model is still loading — poll /health until model_loaded:true",
        )
    tmp = await _save_tmp(file)
    try:
        with _track_request():
            # Spool/normalise on the threadpool so /health
            # stays answerable; the model call itself is queued to the single
            # inference thread inside _run (see CONCURRENCY in the header).
            text, wall, audio_s, timestamps, input_info, batch_size = (
                await run_in_threadpool(_run, tmp)
            )
    except UnprocessableAudio as exc:
        raise _unprocessable(exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return _respond(
        text, wall, audio_s, response_format, timestamps, input_info, batch_size
    )
