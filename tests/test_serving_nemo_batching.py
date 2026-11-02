"""Micro-batching contract of the NeMo server (serving/nemo/server.py).

Moved from the operator's private deployment repo together with the server.
The server funnels every model call through ONE inference thread that batches
concurrent clips — calling `model.transcribe` straight from the uvicorn
threadpool would race on one shared model under a parallel client. Pinned here,
against the served file (loaded by path), with a mock model so no GPU/NeMo is
needed:

1. A batched transcript equals the single-request transcript for every clip,
   and each caller gets its own clip's result (ordering / fan-out).
2. Concurrent requests really are batched, and the model is never entered by
   two threads at once.
3. One failing clip does not poison the batch: it fails alone, the others
   succeed.
4. The NeMo#14427 timestamp retry keeps its old per-clip semantics.
5. Long clips (> NEMO_BATCH_MAX_AUDIO_S) never share a call — Canary's AED
   long-form chunking only engages for a single input.
6. Every response names its serving setup (x_raven.engine/.recipe/.batching).
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

REPO_ROOT = Path(__file__).resolve().parents[1]
NEMO_DIR = REPO_ROOT / "serving" / "nemo"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setenv("NEMO_MAX_BATCH", "8")
    monkeypatch.setenv("NEMO_BATCH_WAIT_MS", "50")
    monkeypatch.setenv("NEMO_BATCH_MAX_AUDIO_S", "40")
    monkeypatch.setenv("BENCH_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    monkeypatch.delenv("RAVEN_SERVING_RECIPE", raising=False)
    monkeypatch.syspath_prepend(str(NEMO_DIR))  # server.py: `from batcher import ...`
    sys.modules.pop("batcher", None)
    mod = _load("raven_serving_nemo_server", NEMO_DIR / "server.py")
    yield mod
    mod._batcher.stop()


class _Hyp:
    def __init__(self, text: str, ts: dict | None = None) -> None:
        self.text = text
        self.timestamp = ts or {}


class FakeModel:
    """Deterministic per-clip output; records every call and any overlap."""

    def __init__(self, delay_s: float = 0.03) -> None:
        self.calls: list[list[str]] = []
        self.delay_s = delay_s
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    @staticmethod
    def text_for(path: str) -> str:
        return f"transkript {Path(path).stem}"

    def transcribe(self, paths, batch_size=1, timestamps=None, **_kw):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append([Path(p).stem for p in paths])
        try:
            time.sleep(self.delay_s)
            assert batch_size == len(paths), "batch_size must equal the batch"
            if any("poison" in p for p in paths):
                raise RuntimeError("CUDA error: device-side assert (poisoned clip)")
            if timestamps and any("edge" in p for p in paths):
                raise ValueError("TDT zero-duration token (NeMo #14427)")
            out = []
            for p in paths:
                ts = {"word": [{"word": Path(p).stem, "start": 0.0, "end": 0.5}]} if timestamps else {}
                out.append(_Hyp(self.text_for(p), ts))
            return out
        finally:
            with self._lock:
                self.active -= 1


def _wav(path: Path, seconds: float) -> str:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return str(path)


def _clips(tmp_path: Path, n: int, seconds: float = 1.0, prefix: str = "clip") -> list[str]:
    return [_wav(tmp_path / f"{prefix}{i:02d}.wav", seconds) for i in range(n)]


def _parallel(fn, items, workers: int):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, it) for it in items]
        return [f.result() if f.exception() is None else f.exception() for f in futs]


def test_batched_output_equals_per_clip_output(server, tmp_path):
    clips = _clips(tmp_path, 24)
    model = FakeModel()
    server._model = model

    sequential = [server._transcribe_window(p)[:2] for p in clips]
    assert all(len(call) == 1 for call in model.calls), "sequential client must not batch"

    model.calls.clear()
    concurrent = [r[:2] for r in _parallel(server._transcribe_window, clips, workers=16)]

    assert concurrent == sequential, "batching changed a transcript or mixed clips up"
    for path, (text, ts) in zip(clips, concurrent):
        assert text == FakeModel.text_for(path)
        assert ts["word"][0]["word"] == Path(path).stem
    assert max(len(c) for c in model.calls) > 1, "concurrent requests were never batched"
    assert max(len(c) for c in model.calls) <= server.MAX_BATCH
    assert model.max_active == 1, "two threads were inside the model at once"


def test_reported_batch_size_matches_the_call(server, tmp_path):
    model = FakeModel(delay_s=0.1)
    server._model = model
    results = _parallel(server._transcribe_window, _clips(tmp_path, 8), workers=8)
    sizes = sorted({r[2] for r in results})
    call_sizes = sorted({len(c) for c in model.calls})
    assert sizes == call_sizes


def test_failing_clip_does_not_poison_the_batch(server, tmp_path):
    good = _clips(tmp_path, 7)
    bad = _wav(tmp_path / "poison.wav", 1.0)
    model = FakeModel(delay_s=0.05)
    server._model = model

    items = good[:3] + [bad] + good[3:]
    results = _parallel(server._transcribe_window, items, workers=8)

    for path, res in zip(items, results):
        if path == bad:
            assert isinstance(res, RuntimeError)
        else:
            assert not isinstance(res, BaseException), res
            assert res[0] == FakeModel.text_for(path)
    assert server._batcher.stats.fallbacks >= 1


def test_timestamp_bug_retry_keeps_per_clip_semantics(server, tmp_path):
    clips = _clips(tmp_path, 5)
    edge = _wav(tmp_path / "edge.wav", 1.0)
    server._model = FakeModel(delay_s=0.05)
    items = clips + [edge]
    results = _parallel(server._transcribe_window, items, workers=6)

    by_path = dict(zip(items, results))
    assert by_path[edge][0] == FakeModel.text_for(edge)
    assert by_path[edge][1] == {}, "the tripping clip falls back to text-only"
    for p in clips:
        assert by_path[p][1]["word"], f"{p} lost its timestamps because a neighbour tripped"


def test_long_clips_never_share_a_call(server, tmp_path):
    long_clips = _clips(tmp_path, 3, seconds=45.0, prefix="long")
    short = _clips(tmp_path, 5)
    model = FakeModel(delay_s=0.05)
    server._model = model
    _parallel(server._transcribe_window, long_clips + short, workers=8)

    for call in model.calls:
        if any(name.startswith("long") for name in call):
            assert len(call) == 1, f"a long clip shared a call: {call}"


def test_canary_batch_keeps_forced_german(server, tmp_path, monkeypatch):
    seen: list[dict] = []

    class Canary(FakeModel):
        def transcribe(self, paths, batch_size=1, timestamps=None, **kw):
            seen.append(kw)
            return super().transcribe(paths, batch_size=batch_size)

    monkeypatch.setattr(server, "_is_canary", True)
    server._model = Canary()
    _parallel(server._transcribe_window, _clips(tmp_path, 6), workers=6)
    assert seen and all(
        kw == {"source_lang": "de", "target_lang": "de", "pnc": "yes"} for kw in seen
    )


def test_request_threads_never_touch_the_model(server):
    """Static guard: only _transcribe_paths (the inference thread) calls .transcribe()."""
    import ast

    tree = ast.parse((NEMO_DIR / "server.py").read_text(encoding="utf-8"))
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "transcribe"
                and fn.name != "_transcribe_paths"
            ):
                offenders.append(f"{fn.name}:{node.lineno}")
    assert not offenders, (
        f"model.transcribe outside the inference thread reintroduces the race: {offenders}"
    )


def test_health_reports_batching(server):
    from fastapi.testclient import TestClient

    body = TestClient(server.app).get("/health").json()
    assert body["batching"]["max_batch"] == 8
    assert body["batching"]["wait_ms"] == 50.0
    assert body["batching"]["max_audio_s"] == 40.0


def test_responses_name_the_serving_setup(server):
    import json

    sys.path.insert(0, str(REPO_ROOT / "serving"))
    from recipe import recipe_sha256

    body = json.loads(server._respond("hallo", 0.5, 1.0, "json").body)
    x = body["x_raven"]
    assert x["model"] == "nvidia/parakeet-tdt-0.6b-v3"
    assert x["recipe"] == {
        "id": "nvidia-parakeet-tdt-0.6b-v3",
        "sha256": recipe_sha256("nvidia-parakeet-tdt-0.6b-v3"),
    }
    assert x["batching"] == {"max_batch": 8, "wait_ms": 50.0, "max_audio_s": 40.0}
    assert x["decoding"] == {"mode": "greedy", "strategy": "greedy_batch", "max_symbols": 10}
    assert "engine" in x  # None until NeMo is imported; "nemo <v> (torch <v>)" after


# The decoding configs the four pinned checkpoints carry (read from each .nemo's
# model_config.yaml at its recipe revision).
_TDT = {"strategy": "greedy_batch", "greedy": {"max_symbols": 10},
        "beam": {"beam_size": 2}}
_CANARY = {"strategy": "beam", "beam": {"beam_size": 1, "len_pen": 0.0,
                                        "max_generation_delta": 50}}


def test_a_checkpoint_matching_its_recipe_decoding_passes(server):
    sys.path.insert(0, str(REPO_ROOT / "serving"))
    from recipe import load_recipes

    recipes = load_recipes()
    for rid, cfg in (("nvidia-parakeet-tdt-0.6b-v3", _TDT), ("primeline-parakeet", _TDT),
                     ("nvidia-nemotron-3.5-asr-streaming-0.6b", _TDT),
                     ("nvidia-canary-1b-v2", _CANARY)):
        assert server.decoding_mismatch(cfg, recipes[rid]["decoding"]) == "", rid


@pytest.mark.parametrize(
    ("rid", "cfg"),
    [
        ("nvidia-parakeet-tdt-0.6b-v3", {**_TDT, "strategy": "beam"}),
        ("nvidia-parakeet-tdt-0.6b-v3", {**_TDT, "greedy": {"max_symbols": 5}}),
        ("nvidia-parakeet-tdt-0.6b-v3", {"strategy": "greedy_batch"}),  # no max_symbols
        ("nvidia-canary-1b-v2",
         {**_CANARY, "beam": {"beam_size": 4, "len_pen": 0.0, "max_generation_delta": 50}}),
    ],
)
def test_a_checkpoint_that_decodes_otherwise_is_refused(server, rid, cfg):
    sys.path.insert(0, str(REPO_ROOT / "serving"))
    from recipe import load_recipes

    expected = load_recipes()[rid]["decoding"]
    assert server.decoding_mismatch(cfg, expected) != ""


def test_launcher_names_the_recipe_the_server_reports(monkeypatch):
    monkeypatch.setenv("NEMO_MAX_BATCH", "8")
    monkeypatch.setenv("RAVEN_SERVING_RECIPE", "primeline-parakeet")
    monkeypatch.setenv("BENCH_MODEL", "primeline/parakeet-primeline")
    monkeypatch.syspath_prepend(str(NEMO_DIR))
    sys.modules.pop("batcher", None)
    mod = _load("raven_serving_nemo_server_2", NEMO_DIR / "server.py")
    try:
        assert mod.RECIPE["id"] == "primeline-parakeet"
    finally:
        mod._batcher.stop()


@pytest.mark.parametrize("path", ["", "/ckpt/m.nemo"])
def test_health_names_the_revision_only_for_a_downloaded_checkpoint(monkeypatch, path):
    """The pin holds only through the .nemo file launch.py downloaded at it; a
    from_pretrained pull follows main, so the server must not claim the pin."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BENCH_MODEL", "nvidia/canary-1b-v2")
    monkeypatch.setenv("BENCH_MODEL_PATH", path)
    monkeypatch.setenv("BENCH_MODEL_REVISION", "d455706339a6b32e1aa40f82c713a482a0c938e2")
    monkeypatch.syspath_prepend(str(NEMO_DIR))
    sys.modules.pop("batcher", None)
    mod = _load("raven_serving_nemo_server_3", NEMO_DIR / "server.py")
    try:
        body = TestClient(mod.app).get("/health").json()
        expected = "d455706339a6b32e1aa40f82c713a482a0c938e2" if path else None
        assert body["model_revision"] == expected
        assert body["model_path"] == (path or None)
    finally:
        mod._batcher.stop()
