"""Live NeMo ASR over vLLM's realtime websocket protocol.

One process, one model, many live sessions. Each session speaks the protocol of
vLLM 0.33.0's ``/v1/realtime`` (vllm/entrypoints/speech_to_text/realtime), so
the capacity harness measures a NeMo model and a vLLM model with one client:

    server: session.created
    client: session.update {model}                    -> the served model id
    client: input_audio_buffer.commit                 -> an utterance starts
    client: input_audio_buffer.append {audio}         -> base64 PCM16, 16 kHz, mono
    client: input_audio_buffer.commit {final: true}   -> its audio has ended
    server: transcription.delta {delta}               -> text the pipeline finalized
    server: transcription.done {text}                 -> the utterance's transcript
    (repeat from the first commit for the next utterance)

One addition, ignored by a vLLM client: ``transcription.partial {text}``, the
pipeline's current partial hypothesis after a step that changed it, so the
harness sees the first words before the pipeline finalizes them.

Inference is NeMo's own streaming pipeline (nemo.collections.asr.inference,
PipelineBuilder) built from the vendored config of the pinned NeMo commit
(conf/) with the stream recipe's keys on top. One thread steps it: each step
takes at most one frame from every session with a frame's worth of audio
(stream_frames.FrameCutter) — up to the recipe's batch_size — and runs
``pipeline.transcribe_step`` on them as one batch, exactly how the pipeline's
file-based ``run()`` steps its streams. Audio a session sent while the GPU was
busy waits in its buffer; under overload that wait is the latency the harness
measures.

Environment (set by launch.py --streaming): RAVEN_STREAM_RECIPE, BENCH_MODEL,
BENCH_MODEL_PATH (the .nemo at the base recipe's revision), BENCH_GPU_MEM_FRACTION.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stream_frames import FrameCutter, FrameSpec

log = logging.getLogger("stream_server")
SAMPLE_RATE = 16000
CONF_DIR = Path(__file__).resolve().parent / "conf"


@dataclass
class StepResult:
    """What one pipeline step produced for one stream."""

    stream_id: int
    final: str = ""      # text finalized in this step
    partial: str = ""    # the current partial hypothesis after this step
    is_last: bool = False


class StreamEngine(Protocol):
    model: str
    frame_samples: int
    max_batch: int

    def step(self, frames: list[FrameSpec]) -> list[StepResult]: ...
    def info(self) -> dict[str, Any]: ...


# ── the NeMo engine ─────────────────────────────────────────────────────────


def _nemo_config(stream: dict[str, Any], checkpoint: str):
    """The vendored pipeline config with exactly the stream recipe's keys on top."""
    from omegaconf import OmegaConf

    pipe = stream["pipeline"]
    cfg = OmegaConf.load(CONF_DIR / f"{pipe}_rnnt.yaml")
    cfg.asr.model_name = checkpoint
    cfg.asr.compute_dtype = "float32"  # the base recipe's fp32 (streaming.toml header)
    cfg.asr.use_amp = False
    cfg.matmul_precision = stream["matmul_precision"]
    cfg.lang = stream["lang"]
    cfg.log_level = 30
    cfg.streaming.batch_size = stream["batch_size"]
    cfg.streaming.request_type = "frame"
    if pipe == "cache_aware":
        cfg.streaming.att_context_size = list(stream["att_context_size"])
        cfg.streaming.num_slots = stream["num_slots"]
    else:
        cfg.streaming.left_padding_size = stream["left_context_s"]
        cfg.streaming.chunk_size = stream["chunk_s"]
        cfg.streaming.right_padding_size = stream["right_context_s"]
    return cfg


class NemoPipelineEngine:
    """NeMo's streaming pipeline for one stream recipe."""

    def __init__(self, sid: str) -> None:
        import torch
        from nemo.collections.asr.inference.factory.pipeline_builder import (
            PipelineBuilder,
        )
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import (
            ASRRequestOptions,
        )
        from recipe import load_recipes, weights_mismatch
        from streaming import load_streams, stream_identity

        recipes = load_recipes()
        streams, _ = load_streams(recipes=recipes)
        self.sid = sid
        self.stream = streams[sid]
        base = recipes[self.stream["base"]]
        self.model = os.environ.get("BENCH_MODEL", base["model"])
        if self.model != base["model"]:
            raise RuntimeError(f"BENCH_MODEL {self.model!r} is not the recipe's {base['model']!r}")
        checkpoint = os.environ.get("BENCH_MODEL_PATH") or ""
        if not checkpoint:
            raise RuntimeError("BENCH_MODEL_PATH is unset: start through launch.py --streaming")
        self._torch, self._Frame, self._Options = torch, Frame, ASRRequestOptions
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(
                float(os.environ.get("BENCH_GPU_MEM_FRACTION", self.stream["gpu_mem"]))
            )
        cfg = _nemo_config(self.stream, checkpoint)
        self.pipeline = PipelineBuilder.build_pipeline(cfg)
        nemo_model = self.pipeline.asr_model.asr_model
        if self.stream["pipeline"] == "buffered":
            # The card's streaming script, att_context_size_as_chunk = true
            # (speech_to_text_streaming_infer_rnnt.py at the NeMo pin): chunked
            # self-attention masks of [left, chunk, right] encoder frames.
            style = nemo_model.cfg.encoder.get("att_context_style")
            if style != "chunked_limited_with_rc":
                raise RuntimeError(
                    f"att_context_style is {style!r}; the card's chunked masks need "
                    "chunked_limited_with_rc"
                )
            per_frame = self.pipeline.window_stride * self.pipeline.subsampling_factor
            ctx = [round(self.stream[k] / per_frame)
                   for k in ("left_context_s", "chunk_s", "right_context_s")]
            nemo_model.encoder.set_default_att_context_size(att_context_size=ctx)
            self.att_context_size = ctx
            frame_s = float(self.pipeline.chunk_size)
        else:
            self.att_context_size = list(self.stream["att_context_size"])
            frame_s = float(self.pipeline.chunk_size_in_secs)
        if abs(frame_s * 1000 - self.stream["chunk_ms"]) > 0.5:
            raise RuntimeError(
                f"the pipeline steps {frame_s * 1000:.1f} ms frames, the recipe's "
                f"chunk_ms is {self.stream['chunk_ms']}"
            )
        self.frame_samples = round(frame_s * SAMPLE_RATE)
        self.max_batch = int(self.stream["batch_size"])
        numel: dict[str, int] = {}
        for p in nemo_model.parameters():
            key = str(p.dtype).removeprefix("torch.")
            numel[key] = numel.get(key, 0) + p.numel()
        if problem := weights_mismatch(numel, base["precision"]["weights"]):
            raise RuntimeError(f"refusing to serve: {problem}")
        self.sep = self.pipeline.get_sep()
        self._text: dict[int, str] = {}
        self._identity = stream_identity(sid)
        self._precision = base["precision"]

    def step(self, frames: list[FrameSpec]) -> list[StepResult]:
        torch = self._torch
        requests = [
            self._Frame(
                samples=torch.from_numpy(f.samples),
                stream_id=f.stream_id,
                is_first=f.is_first,
                is_last=f.is_last,
                length=f.length,
                options=self._Options(language_code=self.stream["lang"]) if f.is_first else None,
            )
            for f in frames
        ]
        outs = self.pipeline.transcribe_step(requests)
        results = []
        for f, out in zip(frames, outs, strict=True):
            final = out.final_transcript
            have = self._text.get(f.stream_id, "")
            if not have:
                final = final.lstrip(self.sep)  # as BasePipeline.run() joins finals
            self._text[f.stream_id] = have + final
            if f.is_last:
                self._text.pop(f.stream_id, None)
            results.append(StepResult(f.stream_id, final, out.partial_transcript, f.is_last))
        return results

    def info(self) -> dict[str, Any]:
        def ver(name: str) -> str | None:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None

        return {
            "stream": self._identity,
            "model": self.model,
            "engine": {"name": "nemo", "nemo_toolkit": ver("nemo_toolkit"), "torch": ver("torch")},
            "pipeline": self.stream["pipeline"],
            "att_context_size": self.att_context_size,
            "frame_ms": self.frame_samples * 1000 / SAMPLE_RATE,
            "lang": self.stream["lang"],
            "precision": self._precision,
            "compute_dtype": "float32",
            "matmul_precision": self.stream["matmul_precision"],
            "batch_size": self.max_batch,
        }


# ── the server ──────────────────────────────────────────────────────────────


@dataclass
class _Utterance:
    stream_id: int
    queue: asyncio.Queue
    text: str = ""
    partial: str = ""


@dataclass
class _Hub:
    """Routes step results from the stepping thread to the session's event loop."""

    loop: asyncio.AbstractEventLoop | None = None
    routes: dict[int, _Utterance] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def deliver(self, result: StepResult) -> None:
        with self.lock:
            utt = self.routes.get(result.stream_id)
            if result.is_last:
                self.routes.pop(result.stream_id, None)
        if utt is not None and self.loop is not None:
            self.loop.call_soon_threadsafe(utt.queue.put_nowait, result)


def create_app(engine: StreamEngine | None = None) -> FastAPI:
    """The app; ``engine`` None builds the NeMo engine from the environment at startup."""
    state: dict[str, Any] = {"engine": engine, "next_id": 0, "error": None}
    hub = _Hub()
    stop = threading.Event()

    def stepper() -> None:
        eng: StreamEngine = state["engine"]
        cutter: FrameCutter = state["cutter"]
        while not stop.is_set():
            if not cutter.ready_event.wait(timeout=0.05):
                continue
            frames, empty = cutter.take(eng.max_batch)
            for sid in empty:
                hub.deliver(StepResult(sid, is_last=True))
            if not frames:
                continue
            try:
                results = eng.step(frames)
            except Exception as exc:  # the pipeline failed: every live session learns it
                log.exception("pipeline step failed")
                state["error"] = f"pipeline step failed: {exc}"
                for f in frames:
                    hub.deliver(StepResult(f.stream_id, is_last=True))
                continue
            for r in results:
                hub.deliver(r)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["engine"] is None:
            state["engine"] = NemoPipelineEngine(os.environ["RAVEN_STREAM_RECIPE"])
        state["cutter"] = FrameCutter(state["engine"].frame_samples)
        hub.loop = asyncio.get_running_loop()
        thread = threading.Thread(target=stepper, name="stepper", daemon=True)
        thread.start()
        yield
        stop.set()
        thread.join(timeout=5)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": state["error"] is None, "error": state["error"]})

    @app.get("/v1/stream/info")
    async def info() -> JSONResponse:
        return JSONResponse(state["engine"].info())

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket) -> None:
        await ws.accept()
        eng: StreamEngine = state["engine"]
        cutter: FrameCutter = state["cutter"]
        await ws.send_text(json.dumps({"type": "session.created"}))
        validated = False
        utt: _Utterance | None = None
        sender: asyncio.Task | None = None

        async def send_events(u: _Utterance) -> None:
            while True:
                r: StepResult = await u.queue.get()
                if r.final:
                    u.text += r.final
                    await ws.send_text(json.dumps({"type": "transcription.delta", "delta": r.final}))
                if r.partial != u.partial:
                    u.partial = r.partial
                    if not r.is_last:
                        await ws.send_text(json.dumps(
                            {"type": "transcription.partial", "text": u.text + r.partial}))
                if r.is_last:
                    await ws.send_text(json.dumps({"type": "transcription.done", "text": u.text}))
                    return

        async def error(msg: str, code: str) -> None:
            await ws.send_text(json.dumps({"type": "error", "error": msg, "code": code}))

        try:
            while True:
                event = json.loads(await ws.receive_text())
                kind = event.get("type")
                if kind == "session.update":
                    if event.get("model") != eng.model:
                        await error(f"The model `{event.get('model')}` does not exist.",
                                    "model_not_found")
                        continue
                    validated = True
                elif kind == "input_audio_buffer.commit":
                    if not validated:
                        await error("Model not validated. Make sure to validate the model "
                                    "by sending a session.update event.", "model_not_validated")
                        continue
                    if event.get("final"):
                        if utt is not None:
                            cutter.end(utt.stream_id)
                        continue
                    if sender is not None and not sender.done():
                        await error("Generation already in progress", "busy")
                        continue
                    sid = state["next_id"]
                    state["next_id"] += 1
                    utt = _Utterance(sid, asyncio.Queue())
                    with hub.lock:
                        hub.routes[sid] = utt
                    cutter.open(sid)
                    sender = asyncio.create_task(send_events(utt))
                elif kind == "input_audio_buffer.append":
                    if utt is None or not cutter.is_open(utt.stream_id):
                        await error("No utterance in progress", "invalid_event")
                        continue
                    pcm = np.frombuffer(base64.b64decode(event["audio"]), dtype=np.int16)
                    cutter.append(utt.stream_id, pcm.astype(np.float32) / 32768.0)
                else:
                    await error(f"Unknown event type: {kind}", "unknown_event")
        except WebSocketDisconnect:
            pass
        finally:
            if utt is not None:
                cutter.end(utt.stream_id)  # the pipeline frees the stream's state
            if sender is not None and not sender.done():
                sender.cancel()

    return app


app = create_app()
