# The surface the harness talks to for every vLLM-served model (and for the
# transformers stand-in, which binds where vLLM would):
#
#   harness ──Bearer──▶ shim  SHIM_HOST:SHIM_PORT  /v1/audio/transcriptions
#                         │
#                         ▼ (loopback only)
#                       engine  SHIM_UPSTREAM  (serving/launch.py, one model at a time)
#
# The engine binds loopback only and has no auth; the shim is the one exposed
# surface, so Bearer auth lives in exactly one place. The harness always speaks
# OpenAI /v1/audio/transcriptions multipart; per model the shim either
#
#   * PASSTHROUGH — forwards the multipart unchanged to vLLM's native
#     /v1/audio/transcriptions (models whose vLLM class implements
#     SupportsTranscription), or
#   * CHAT-TRANSLATE — wraps the audio into a /v1/chat/completions request with
#     an audio content part and returns the completion text as
#     {"text": ...} (audio-capable chat models, for which vLLM's
#     transcriptions endpoint 400s with "does not support Transcriptions API").
#
# The mode is NOT guessed at runtime from behaviour: it follows from the model
# id (FAMILY_BY_PREFIX below), or the operator sets SHIM_FAMILY. Payload shapes
# per family are documented inline with their sources; anything unverified is
# marked  # UNVERIFIED:  loudly.
#
# IDENTITY: every JSON answer carries
#   x_raven.model     the id the engine actually serves (first entry of its
#                     /v1/models; vLLM lists the base model first, LoRA modules
#                     after). The harness aborts when it differs from the model
#                     it measures — a forgotten model swap would otherwise
#                     produce numbers under the wrong name.
#   x_raven.engine    "vllm <version>" from the engine's /version (the
#                     transformers stand-in reports its own engine string)
#   x_raven.recipe    {id, sha256} of the serving recipe of that model
#                     (serving/recipe.py), looked up by the served id
#   x_raven.batching  {"max_num_seqs": n}: RAVEN_SERVE_MAX_NUM_SEQS when set —
#                     give the shim the value the launcher got — else the
#                     recipe's default
#   x_raven.decoding  the recipe's decoding table: mode "greedy" and max_tokens
#   x_raven.precision the recipe's precision table: weights, activations, scheme,
#                     checkpoint, tensor_parallel. launch.py starts the engine
#                     from that same table (--dtype, --tensor-parallel-size), and
#                     the transformers stand-in refuses to serve weights that are
#                     not in it
#
# DECODING is the recipe's, never the engine's default: every request upstream
# carries temperature 0, repetition_penalty 1.0 and the recipe's max_tokens
# (transcriptions: max_completion_tokens; chat: max_tokens). vLLM otherwise
# fills an absent field from the model's generation_config.json and caps output
# at max_model_len — a limit that differed by model and decided how much a
# looping answer cost. A request asking for another temperature is refused.
# The harness records the serving identity next to each result and aborts when
# it changes during a run.
#
# One shared httpx.AsyncClient for the process instead of one per request: a
# parallel client would otherwise pay a fresh connection per clip.
"""OpenAI-transcriptions → vLLM translation shim (identity-reporting)."""

from __future__ import annotations

import base64
import logging
import os
import sys
import time
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

# The recipe registry sits one directory up (serving/recipe.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recipe import RecipeError, load_recipes, resolve  # noqa: E402
from recipe import decoding as recipe_decoding  # noqa: E402
from recipe import identity as recipe_identity  # noqa: E402
from recipe import precision as recipe_precision  # noqa: E402

logger = logging.getLogger("raven-serving-shim")
logging.basicConfig(level=logging.INFO)

BIND_HOST = os.environ.get("SHIM_HOST", "127.0.0.1")  # a private address, never 0.0.0.0
BIND_PORT = int(os.environ.get("SHIM_PORT", "8001"))
UPSTREAM = os.environ.get("SHIM_UPSTREAM", "http://127.0.0.1:8000")
# Empty key == auth disabled, rely on the bind address; set key == every transcription request needs Bearer.
API_KEY = os.environ.get("STT_API_KEY", "")
# Override the family detection (passthrough|<family>); empty = detect by model id.
FAMILY_OVERRIDE = os.environ.get("SHIM_FAMILY", "")
TIMEOUT_S = float(os.environ.get("SHIM_TIMEOUT_S", "600"))
# Upload cap (bench clips are minutes, not movies): reject oversized bodies
# before buffering them whole — same default as the NeMo server's
# BENCH_MAX_UPLOAD_MB.
MAX_UPLOAD_MB = int(os.environ.get("SHIM_MAX_UPLOAD_MB", "512"))

# How long a looked-up served-model id is trusted. A model swap restarts
# vLLM (tens of seconds without a healthy upstream), so 5 s cannot report the
# previous model for a request the new one answered.
SERVED_MODEL_TTL_S = float(os.environ.get("SHIM_SERVED_MODEL_TTL_S", "5"))

app = FastAPI(title="raven.eval serving shim", version="1.1.0")

_http: httpx.AsyncClient | None = None
_served: tuple[float, str | None] = (0.0, None)
_serving_cache: tuple[str | None, dict[str, Any]] = (None, {})


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(
            timeout=TIMEOUT_S,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )
    return _http


@app.on_event("shutdown")
async def _close_client() -> None:
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None


async def _served_model() -> str | None:
    """The id upstream serves right now (cached SERVED_MODEL_TTL_S)."""
    global _served
    fetched_at, model_id = _served
    if model_id is not None and time.monotonic() - fetched_at < SERVED_MODEL_TTL_S:
        return model_id
    try:
        r = await _client().get(f"{UPSTREAM}/v1/models", timeout=5)
        r.raise_for_status()
        data = r.json().get("data") or []
        model_id = str(data[0].get("id")) if data else None
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        logger.warning("served-model lookup failed: %s", exc)
        model_id = None
    _served = (time.monotonic(), model_id)
    return model_id


async def _engine_version() -> str | None:
    """``"vllm <version>"`` (or the stand-in's own engine string), cached per model."""
    try:
        r = await _client().get(f"{UPSTREAM}/version", timeout=5)
        r.raise_for_status()
        body = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("engine version lookup failed: %s", exc)
        return None
    if not isinstance(body, dict):
        return None
    if body.get("engine"):
        return str(body["engine"])
    return f"vllm {body['version']}" if body.get("version") else None


def _batching(served: str) -> dict[str, int] | None:
    """``{"max_num_seqs": n}`` the engine was started with, as far as the shim knows.

    RAVEN_SERVE_MAX_NUM_SEQS is what launch.py passed to vLLM when the operator
    set it; otherwise the recipe default applied. The transformers stand-in
    serialises generate() whatever is set, so its recipe value always holds.
    """
    try:
        recipes = load_recipes()
        recipe = recipes[resolve(served, recipes)]
    except (OSError, RecipeError, ValueError):
        return None
    seqs = recipe.get("max_num_seqs")
    override = os.environ.get("RAVEN_SERVE_MAX_NUM_SEQS", "").strip()
    if recipe["engine"] == "vllm" and override:
        seqs = int(override)
    return {"max_num_seqs": int(seqs)} if seqs is not None else None


async def _serving(served: str | None) -> dict[str, Any]:
    """The serving block for ``served``, looked up once per served model."""
    global _serving_cache
    if served is None:
        return {"engine": None, "recipe": None, "batching": None, "decoding": None,
                "precision": None}
    if _serving_cache[0] != served:
        engine = await _engine_version()
        info = {
            "engine": engine, "recipe": recipe_identity(served),
            "batching": _batching(served), "decoding": recipe_decoding(served),
            "precision": recipe_precision(served),
        }
        if engine is None:
            return info  # do not cache a failed lookup
        _serving_cache = (served, info)
    return dict(_serving_cache[1])


def _with_identity(body: Any, served: str | None, serving: dict[str, Any]) -> Any:
    if isinstance(body, dict):
        x_raven = body.get("x_raven")
        if not isinstance(x_raven, dict):
            x_raven = {}
        x_raven["model"] = served
        x_raven.update(serving)
        body["x_raven"] = x_raven
    return body


def require_key(authorization: str = Header(default="")) -> None:
    if not API_KEY:
        return  # auth disabled — bind to a private address only
    if authorization.removeprefix("Bearer ").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


# --------------------------------------------------------------------------
# Per-model routing table. Verified 2026-09-01 against vLLM sources (per entry;
# flags and their reasons per model are in serving/recipes.toml).
#
# PASSTHROUGH — vLLM class implements SupportsTranscription:
#   * ibm-granite/granite-speech-4.1-2b-plus
#       granite_speech.py:571-576 implements SupportsTranscription; listed in
#       the vLLM docs "Supported Models" Transcription table.
#   * CohereLabs/cohere-transcribe-03-2026
#       cohere_asr.py CohereAsrForConditionalGeneration(SupportsTranscription),
#       merged via vllm-project/vllm#38120. NOTE: vLLM defaults language to
#       "en" for this model (cohere_asr.py validate_language) — the bench
#       client MUST send language=de.
#   * OpenMOSS-Team/MOSS-Transcribe-Diarize
#       moss_transcribe_diarize.py:565-569 (SupportsTranscription); the model
#       card's own serve example curls /v1/audio/transcriptions.
#   * openai/whisper-large-v3, primeline/whisper-large-v3-german,
#     primeline/whisper-large-v3-turbo-german, nyrahealth/CrisperWhisper
#       all four are WhisperForConditionalGeneration (each repo's config.json
#       architectures field); whisper.py:818-820 implements
#       SupportsTranscription. Bench client MUST send language=de — Whisper
#       AUTO-DETECTS when the field is missing (whisper.py validate_language).
#   * Qwen/Qwen3-ASR-1.7B
#       qwen3_asr.py:344-349 (SupportsTranscription), vLLM-native config for
#       model_type qwen3_asr.
#   * mistralai/Voxtral-Mini-3B-2507
#       voxtral.py:324-327 (SupportsTranscription); vLLM docs Transcription
#       table lists it as "Voxtral (Mistral format)" — the launcher carries
#       the mandatory --tokenizer-mode/--config-format/--load-format mistral.
#   * openai/whisper-large-v3-turbo — WhisperForConditionalGeneration, as above.
#   * ibm-granite/granite-speech-5.0-2b
#       config.json architectures = GraniteSpeechForConditionalGeneration, the
#       class of granite_speech.py at v0.33.0 (SupportsTranscription; its
#       get_generation_prompt asks "can you transcribe the speech into a
#       written format?" and names no language).
#   * mistralai/Voxtral-Mini-4B-Realtime-2602
#       voxtral_realtime.py at v0.33.0: VoxtralRealtimeGeneration subclasses
#       VoxtralForConditionalGeneration (SupportsTranscription) and overrides
#       get_generation_prompt with mistral-common's StreamingMode.OFFLINE —
#       the whole file in one request, the transcription delay read from the
#       checkpoint's tekken.json (480 ms at the pinned commit). The /v1/realtime
#       websocket is not needed for file transcription.
#   * moonshine-ai/moonshine-streaming-small-de,
#     ibm-granite/granite-speech-5.0-470m-turboctc
#       NOT vLLM — served by transformers_server.py on the engine bind (vLLM
#       has no class for either). Passthrough works unchanged because the
#       stand-in speaks the same /v1/audio/transcriptions + /v1/models.
#
# CHAT-TRANSLATE:
#   * microsoft/Phi-4-multimodal-instruct — family "phi4mm".
#       phi4mm.py:1031 Phi4MMForCausalLM has NO SupportsTranscription; ASR goes
#       through /v1/chat/completions with model="speech" (the speech-LoRA
#       module name from `--lora-modules speech=...`, per the model card's
#       vLLM-inference section — requesting the base model id would skip the
#       speech adapter). Audio rides as an `input_audio` base64 content part;
#       both `audio_url` and `input_audio` are supported for audio-capable
#       models per vLLM docs "Multimodal Inputs" -> Online Serving -> Audio
#       Inputs. Placeholders like <|audio_1|> must NOT be written into the
#       text — the API server injects them from the content parts (same docs
#       page; the literal placeholder string is only for the offline
#       raw-prompt path).
#   * microsoft/VibeVoice-ASR-HF — family "vibevoice".
#       vLLM 0.33.0 serves VibeVoiceAsrForConditionalGeneration through its
#       Transformers modeling backend (registry.py _TRANSFORMERS_SUPPORTED_MODELS;
#       tests/models/registry.py names this repo), which has no transcription
#       endpoint. The request carries ONLY the audio part: the repo's
#       chat_template.jinja then writes the model's own instruction ("This is a
#       <|AUDIO_DURATION|> seconds audio, please transcribe it with these keys:
#       Start time, End time, Speaker ID, Content"; the processor fills in the
#       duration). A text part would become "extra info" context in that
#       template, so a request prompt is passed as exactly that and nothing is
#       added otherwise — no language: the card says the model "requires no
#       explicit language setting". The answer is a JSON list of speaker turns;
#       the harness parses it (raven_asr.output_formats "vibevoice-json").
#   * nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (and builds of it) — family
#       "nemotron-transcribe". Served by NeMo Speech's vLLM plugin
#       (vllm_plugins = ["nemo_speechlm"]): a chat model with no transcription
#       endpoint, no SupportsTranscription and no language parameter. The request
#       is the card's own: one user message, the instruction text plus the audio
#       as an audio_url data URL, no system message, reasoning off
#       (chat_template_kwargs enable_thinking = false). The instruction is the
#       card's "Exact ASR Benchmark Prompts" verbatim-transcription prompt, which
#       its multilingual Common Voice / MLS numbers use; it says "Language: as
#       spoken", so the model is NOT forced to German (a language= field is not
#       forwarded; the card documents no language-forcing wording).
# --------------------------------------------------------------------------

# UNVERIFIED: "Transcribe the audio clip into text." is the ASR instruction
# used in Microsoft's Phi-4-multimodal examples, but the exact canonical
# wording was not re-confirmed against the model card in this pass. Any
# transcription instruction works; override per run via SHIM_PROMPT or the
# multipart `prompt` field if the card's wording turns out to differ.
PHI4_DEFAULT_PROMPT = os.environ.get(
    "SHIM_PROMPT", "Transcribe the audio clip into text."
)

#: The card's verbatim ASR benchmark instruction (README "Exact ASR Benchmark
#: Prompts", "Verbatim Transcription"), word for word.
NEMOTRON_TRANSCRIBE_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including fillers "
    "and repetitions. Formatting: punctuation and capitalization. Output: transcript only."
)

FAMILY_BY_PREFIX: list[tuple[str, str]] = [
    ("phi-4-multimodal", "phi4mm"),
    ("phi4", "phi4mm"),
    ("vibevoice", "vibevoice"),
    ("nemotron-3.5-transcribe", "nemotron-transcribe"),
    # granite / cohere / moss fall through to "passthrough" (the default).
]


def _known_prefixes(family: str) -> list[str]:
    return [prefix for prefix, fam in FAMILY_BY_PREFIX if fam == family]


def _build_phi4mm(
    *,
    model: str,
    audio_b64: str,
    audio_format: str,
    language: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    del model  # the served LoRA module name is what matters, not the request's
    prompt = user_prompt or PHI4_DEFAULT_PROMPT
    if language and not user_prompt:
        # UNVERIFIED: no documented language-pinning prompt for Phi-4-mm ASR;
        # naming the language in the instruction is a plain-prompt convention,
        # not a card-specified control token.
        prompt = f"{prompt} The audio is in language: {language}."
    return {
        "model": "speech",  # LoRA module name — see routing table above
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": audio_format},
                    },
                ],
            }
        ],
        **_greedy(max_tokens, "max_tokens"),
    }


def _build_vibevoice(
    *,
    model: str,
    audio_b64: str,
    audio_format: str,
    language: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    del language  # the model takes none; see the routing table above
    content: list[dict[str, Any]] = []
    if user_prompt:
        content.append({"type": "text", "text": user_prompt})
    content.append(
        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}}
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        **_greedy(max_tokens, "max_tokens"),
    }


def _build_nemotron_transcribe(
    *,
    model: str,
    audio_b64: str,
    audio_format: str,
    language: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    del language  # "Language: as spoken"; see the routing table above
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt or NEMOTRON_TRANSCRIBE_PROMPT},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/{audio_format};base64,{audio_b64}"},
                    },
                ],
            }
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        **_greedy(max_tokens, "max_tokens"),
    }


CHAT_BUILDERS = {
    "phi4mm": _build_phi4mm,
    "vibevoice": _build_vibevoice,
    "nemotron-transcribe": _build_nemotron_transcribe,
}


def _greedy(max_tokens: int, length_field: str) -> dict[str, Any]:
    """The decoding fields every upstream request carries (see DECODING above)."""
    return {"temperature": 0.0, "repetition_penalty": 1.0, length_field: max_tokens}


def _max_tokens(serving: dict[str, Any]) -> int:
    """The served recipe's max_tokens; no recipe, no request — never a default."""
    dec = serving.get("decoding")
    if not isinstance(dec, dict) or not isinstance(dec.get("max_tokens"), int):
        raise HTTPException(
            status_code=503,
            detail=(
                "the served model has no recipe with decoding.max_tokens in "
                "serving/recipes.toml; refusing to decode under an engine default"
            ),
        )
    return int(dec["max_tokens"])


def _detect_family(model_id: str) -> str:
    if FAMILY_OVERRIDE:
        if FAMILY_OVERRIDE in CHAT_BUILDERS:
            # A chat-family override left over from a previous run would
            # silently chat-translate a PASSTHROUGH model — the measurement
            # would run through the wrong code path and skew the bench.
            # Refuse hard unless the requested model id actually belongs to
            # the overridden family.
            prefixes = _known_prefixes(FAMILY_OVERRIDE)
            if not model_id or not any(p in model_id.lower() for p in prefixes):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"SHIM_FAMILY={FAMILY_OVERRIDE!r} is a chat-translate "
                        f"override for models matching {prefixes}, but this "
                        f"request names model={model_id!r}. Refusing to "
                        "chat-translate a mismatched model: remove SHIM_FAMILY "
                        "from the shim's environment and restart it for "
                        "passthrough models, or send the matching model id."
                    ),
                )
        return FAMILY_OVERRIDE
    lowered = model_id.lower()
    for prefix, family in FAMILY_BY_PREFIX:
        if prefix in lowered:
            return family
    return "passthrough"


def _audio_format(filename: str | None, content_type: str | None) -> str:
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in {"wav", "mp3", "flac", "ogg", "webm", "m4a"}:
            return ext
    if content_type and "/" in content_type:
        return content_type.split("/", 1)[1]
    return "wav"


@app.get("/health")
@app.get("/healthz")
async def health() -> JSONResponse:
    upstream_ok = False
    upstream_models: Any = None
    try:
        client = _client()
        r = await client.get(f"{UPSTREAM}/health", timeout=5)
        upstream_ok = r.status_code == 200
        m = await client.get(f"{UPSTREAM}/v1/models", timeout=5)
        if m.status_code == 200:
            upstream_models = [x.get("id") for x in m.json().get("data", [])]
    except httpx.HTTPError as exc:
        logger.warning("upstream health probe failed: %s", exc)
    # Effective family for the CURRENTLY SERVED model, so the operator can see
    # a stale SHIM_FAMILY override in one curl instead of via a failed bench.
    effective_family: str | None = FAMILY_OVERRIDE or None
    if upstream_models:
        try:
            effective_family = _detect_family(str(upstream_models[0]))
        except HTTPException:
            effective_family = (
                f"MISMATCH: SHIM_FAMILY={FAMILY_OVERRIDE!r} vs served model "
                f"{upstream_models[0]!r} — transcription requests will 400"
            )
    return JSONResponse(
        status_code=200 if upstream_ok else 503,
        content={
            "shim": "ok",
            "upstream": UPSTREAM,
            "upstream_healthy": upstream_ok,
            "upstream_models": upstream_models,
            "family_override": FAMILY_OVERRIDE or None,
            "effective_family": effective_family,
            "auth_enabled": bool(API_KEY),
        },
    )


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),
    language: str = Form(""),
    prompt: str = Form(""),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    authorization: str = Header(default=""),
):
    require_key(authorization)
    # Read chunked with a hard cap — the shim buffers the body (multipart
    # re-forward / base64), so an unbounded upload is an unbounded allocation.
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1 << 20):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds SHIM_MAX_UPLOAD_MB={MAX_UPLOAD_MB} MB",
            )
        chunks.append(chunk)
    audio = b"".join(chunks)
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio upload")

    family = _detect_family(model)
    logger.info(
        "transcription request: model=%s family=%s bytes=%d lang=%s",
        model, family, len(audio), language or "-",
    )

    if temperature != 0.0:
        raise HTTPException(
            status_code=400,
            detail="every recipe decodes greedily; temperature must be 0",
        )
    served = await _served_model()
    serving = await _serving(served)
    max_tokens = _max_tokens(serving)
    client = _client()
    if family == "passthrough":
        # Native SupportsTranscription model — forward multipart unchanged.
        form: dict[str, Any] = {"response_format": response_format}
        if model:
            form["model"] = model
        if language:
            form["language"] = language
        if prompt:
            form["prompt"] = prompt
        form.update({k: str(v) for k, v in _greedy(max_tokens, "max_completion_tokens").items()})
        upstream = await client.post(
            f"{UPSTREAM}/v1/audio/transcriptions",
            data=form,
            files={"file": (file.filename or "audio.wav", audio,
                            file.content_type or "application/octet-stream")},
        )
        if upstream.status_code != 200:
            # Any upstream failure surfaces as ONE status from the shim
            # (502) with the original status preserved in the body + log —
            # the bench client must never mistake a vLLM-side error for
            # its own bad request.
            logger.error(
                "upstream transcriptions returned %d: %s",
                upstream.status_code, upstream.text[:500],
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "upstream_error",
                    "upstream_status": upstream.status_code,
                    "upstream_body": upstream.text[:2000],
                },
            )
        if response_format == "text":
            return PlainTextResponse(upstream.text)
        return JSONResponse(content=_with_identity(upstream.json(), served, serving))

    # Chat-translate family.
    builder = CHAT_BUILDERS.get(family)
    if builder is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown shim family {family!r}; set SHIM_FAMILY",
        )
    payload = builder(
        model=model,
        audio_b64=base64.b64encode(audio).decode("ascii"),
        audio_format=_audio_format(file.filename, file.content_type),
        language=language,
        user_prompt=prompt,
        max_tokens=max_tokens,
    )
    upstream = await client.post(f"{UPSTREAM}/v1/chat/completions", json=payload)
    if upstream.status_code != 200:
        logger.error(
            "upstream chat completion returned %d: %s",
            upstream.status_code, upstream.text[:500],
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "upstream_error",
                "upstream_status": upstream.status_code,
                "upstream_body": upstream.text[:2000],
            },
        )
    body = upstream.json()
    try:
        text = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"unexpected chat completion shape: {exc!r}",
        )
    text = text.strip()
    if response_format == "text":
        return PlainTextResponse(text)
    # OpenAI transcription json shape: {"text": ...}; usage passed through
    # for the bench's cost/latency bookkeeping.
    return JSONResponse(
        content=_with_identity({"text": text, "usage": body.get("usage")}, served, serving)
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=BIND_HOST, port=BIND_PORT)
