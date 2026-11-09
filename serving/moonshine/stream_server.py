"""Live Moonshine over vLLM's realtime websocket protocol, on the vendor runtime.

The same protocol as serving/nemo/stream_server.py (vLLM 0.33.0's
``/v1/realtime`` plus ``transcription.partial``), so the capacity harness
measures this model like every other. Inference is the vendor's own streaming
library, ``moonshine-voice`` (streaming.toml pins the version, the model files
by sha256 and the precision): one ``Transcriber`` holds the model, every
utterance gets its own stream (``Transcriber.create_stream``, the documented
path for several live sources), audio goes in with ``Stream.add_audio`` —
which runs the transcript update itself whenever the update interval has
passed — and ``Stream.stop`` completes the open line.

Event mapping: ``LineTextChanged`` -> ``transcription.partial`` (completed
lines of this utterance + the open line), ``LineCompleted`` ->
``transcription.delta``, ``stop`` returned -> ``transcription.done`` with the
completed lines joined by a space. The library calls its listeners on the
thread that fed the audio; each session's calls run in a worker thread
(``libmoonshine`` is entered through ctypes, which releases the GIL), in order.

Environment (set by launch.py --streaming): RAVEN_STREAM_RECIPE,
MOONSHINE_VOICE_CACHE (where the model files are kept).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import resource
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE = 16000
#: Worker threads for the sessions' library calls: more than the largest N
#: stepped, so no session waits for a thread — only for the library.
WORKERS = 256


class LiveStream(Protocol):
    def add_audio(self, audio: list[float], sample_rate: int) -> None: ...
    def stop(self) -> Any: ...
    def close(self) -> None: ...


class StreamEngine(Protocol):
    model: str

    def open(self, on_partial, on_line) -> LiveStream: ...
    def info(self) -> dict[str, Any]: ...


def fetch_files(model_url: str, files: dict[str, str], cache: Path) -> Path:
    """The model files at ``model_url``, downloaded once into ``cache`` and
    checked against the recipe's sha256 every start."""
    target = cache / model_url.split("://", 1)[1]
    target.mkdir(parents=True, exist_ok=True)
    for name, want in files.items():
        path = target / name
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(f"{model_url}/{name}", tmp)
            tmp.rename(path)
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise RuntimeError(f"{name}: sha256 {got} is not the recipe's {want}")
    return target


class MoonshineEngine:
    """The vendor runtime for one stream recipe."""

    def __init__(self, sid: str) -> None:
        from moonshine_voice import Transcriber
        from moonshine_voice.moonshine_api import ModelArch
        from moonshine_voice.transcriber import LineCompleted, LineTextChanged
        from recipe import load_recipes
        from streaming import load_streams, stream_identity

        recipes = load_recipes()
        streams, _ = load_streams(recipes=recipes)
        self.stream = s = streams[sid]
        if s["protocol"] != "moonshine-voice":
            raise RuntimeError(f"{sid} is not a moonshine-voice stream recipe")
        want = s["runtime"].split("==", 1)[1]
        have = metadata.version("moonshine-voice")
        if have != want:
            raise RuntimeError(f"moonshine-voice {have} installed, the recipe pins {want}")
        self.model = recipes[s["base"]]["model"]
        cache = Path(os.environ.get("MOONSHINE_VOICE_CACHE", Path.home() / ".cache/moonshine_voice"))
        path = fetch_files(s["model_url"], s["files"], cache)
        self._events = (LineTextChanged, LineCompleted)
        self.transcriber = Transcriber(model_path=str(path), model_arch=ModelArch(s["model_arch"]),
                                       update_interval=s["update_interval_s"])
        self._identity = stream_identity(sid)

    def open(self, on_partial, on_line) -> LiveStream:
        changed, completed = self._events
        stream = self.transcriber.create_stream(update_interval=self.stream["update_interval_s"])

        def listener(event) -> None:
            if isinstance(event, completed):
                on_line(event.line.text)
            elif isinstance(event, changed):
                on_partial(event.line.text)

        stream.add_listener(listener)
        stream.start()
        return stream

    def info(self) -> dict[str, Any]:
        def ver(name: str) -> str | None:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None

        s = self.stream
        return {
            "stream": self._identity,
            "model": self.model,
            "engine": {"name": "moonshine-voice", "moonshine-voice": ver("moonshine-voice"),
                       "library_version": self.transcriber.get_version()},
            "device": s["device"],
            "cpu_count": os.cpu_count(),
            "model_url": s["model_url"],
            "model_arch": s["model_arch"],
            "update_interval_s": s["update_interval_s"],
            "precision": s["precision"],
        }


def _rss_mib() -> float:
    """This process's peak resident memory (MiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def create_app(engine: StreamEngine | None = None) -> FastAPI:
    state: dict[str, Any] = {"engine": engine, "active": 0}
    pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="moonshine")

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["engine"] is None:
            state["engine"] = MoonshineEngine(os.environ["RAVEN_STREAM_RECIPE"])
        yield
        pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/v1/stream/info")
    async def info() -> JSONResponse:
        return JSONResponse(state["engine"].info())

    @app.get("/v1/stream/stats")
    async def stats() -> JSONResponse:
        return JSONResponse({"rss_max_mib": round(_rss_mib(), 1), "active_streams": state["active"]})

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket) -> None:
        await ws.accept()
        eng: StreamEngine = state["engine"]
        loop = asyncio.get_running_loop()
        await ws.send_text(json.dumps({"type": "session.created"}))
        validated = False
        stream: LiveStream | None = None
        lines: list[str] = []
        outbox: asyncio.Queue = asyncio.Queue()

        async def sender() -> None:
            while True:
                event = await outbox.get()
                if event is None:
                    return
                await ws.send_text(json.dumps(event))

        send_task = asyncio.create_task(sender())

        def emit(event: dict) -> None:  # from a worker thread
            loop.call_soon_threadsafe(outbox.put_nowait, event)

        def on_partial(text: str) -> None:
            emit({"type": "transcription.partial", "text": " ".join([*lines, text]).strip()})

        def on_line(text: str) -> None:
            text = text.strip()
            if text:
                lines.append(text)
                emit({"type": "transcription.delta", "delta": text + " "})

        async def error(msg: str, code: str) -> None:
            await outbox.put({"type": "error", "error": msg, "code": code})

        async def finish(s: LiveStream) -> None:
            await loop.run_in_executor(pool, s.stop)
            await loop.run_in_executor(pool, s.close)
            state["active"] -= 1

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
                        if stream is not None:
                            s, stream = stream, None
                            await finish(s)
                            await outbox.put({"type": "transcription.done", "text": " ".join(lines)})
                        continue
                    if stream is not None:
                        await error("Generation already in progress", "busy")
                        continue
                    lines = []
                    stream = await loop.run_in_executor(pool, eng.open, on_partial, on_line)
                    state["active"] += 1
                elif kind == "input_audio_buffer.append":
                    if stream is None:
                        await error("No utterance in progress", "invalid_event")
                        continue
                    pcm = np.frombuffer(base64.b64decode(event["audio"]), dtype=np.int16)
                    audio = (pcm.astype(np.float32) / 32768.0).tolist()
                    await loop.run_in_executor(pool, stream.add_audio, audio, SAMPLE_RATE)
                else:
                    await error(f"Unknown event type: {kind}", "unknown_event")
        except WebSocketDisconnect:
            pass
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    await finish(stream)
            await outbox.put(None)
            with contextlib.suppress(Exception):
                await send_task

    return app


app = create_app()
