"""Stream recipes, the live frame cutter and the NeMo stream server's protocol.

No GPU, no NeMo: the server runs with a fake engine that "transcribes" a frame
as the count of frames seen, so the protocol, the frame rules and the event
order are tested exactly as the real engine drives them.
"""

from __future__ import annotations

import base64
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

SERVING = Path(__file__).resolve().parents[1] / "serving"
sys.path.insert(0, str(SERVING))
sys.path.insert(0, str(SERVING / "nemo"))

import launch
import recipe
import streaming
from stream_frames import FrameCutter

# ── recipes ─────────────────────────────────────────────────────────────────


def test_streaming_toml_loads_and_names_its_skips():
    streams, skipped = streaming.load_streams()
    assert set(streams) == {
        "voxtral-mini-4b-realtime-2602",
        "nvidia-nemotron-3.5-asr-streaming-0.6b",
        "nvidia-nemotron-labs-unified-asr-0.6b",
    }
    assert "moonshine-streaming-small-de" in skipped
    for sid in streams:
        assert len(streaming.stream_sha256(sid)) == 64


def _load(tmp_path: Path, text: str):
    p = tmp_path / "s.toml"
    p.write_text(text)
    return streaming.load_streams(p)


BASE = """
[streams.x]
base = "nvidia-nemotron-3.5-asr-streaming-0.6b"
protocol = "nemo-pipeline"
pipeline = "cache_aware"
att_context_size = [56, 6]
chunk_ms = 560
lang = "de-DE"
batch_size = 8
num_slots = 16
matmul_precision = "highest"
gpu_mem = 0.5
latency = "l"
notes = "n"
"""


def test_valid_minimal_recipe(tmp_path):
    streams, _ = _load(tmp_path, BASE)
    assert streams["x"]["pipeline"] == "cache_aware"


@pytest.mark.parametrize("edit, msg", [
    (("protocol = \"nemo-pipeline\"", "protocol = \"vllm-realtime\""), "do not apply"),
    (("base = \"nvidia-nemotron-3.5-asr-streaming-0.6b\"", "base = \"nope\""), "no recipe"),
    (("num_slots = 16", "num_slots = 4"), "num_slots"),
    (("chunk_ms = 560", "chunk_ms = 565"), "chunk_ms"),
    (("att_context_size = [56, 6]", "att_context_size = [56, 6]\nchunk_s = 0.5"), "do not apply"),
    (("base = \"nvidia-nemotron-3.5-asr-streaming-0.6b\"",
      "base = \"voxtral-mini-4b-realtime-2602\""), "serves nemo"),
])
def test_recipe_rejects(tmp_path, edit, msg):
    with pytest.raises(recipe.RecipeError, match=msg):
        _load(tmp_path, BASE.replace(*edit))


def test_skip_needs_reason(tmp_path):
    with pytest.raises(recipe.RecipeError, match="reason"):
        _load(tmp_path, BASE + '\n[skipped.y]\nreason = ""\n')


def test_hash_moves_with_the_table_and_the_base(monkeypatch):
    recipes = recipe.load_recipes()
    streams, _ = streaming.load_streams(recipes=recipes)
    sid = "nvidia-nemotron-3.5-asr-streaming-0.6b"
    h = streaming.stream_sha256(sid, streams, recipes)
    changed = {**streams, sid: {**streams[sid], "att_context_size": [56, 13]}}
    assert streaming.stream_sha256(sid, changed, recipes) != h
    base = dict(recipes)
    base[streams[sid]["base"]] = {**recipes[streams[sid]["base"]], "max_batch": 1}
    assert streaming.stream_sha256(sid, streams, base) != h


def test_launch_vllm_realtime_is_the_base_serve_with_the_session_cap():
    recipes = recipe.load_recipes()
    streams, _ = streaming.load_streams(recipes=recipes)
    argv, env = launch.build_stream("voxtral-mini-4b-realtime-2602", streams, recipes, {},
                                    python="/v/bin/python")
    assert argv[:3] == ["/v/bin/vllm", "serve", "mistralai/Voxtral-Mini-4B-Realtime-2602"]
    assert argv[argv.index("--max-num-seqs") + 1] == "128"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.85"
    assert "--compilation-config" in argv  # the base recipe's own flags ride along
    assert env["RAVEN_STREAM_RECIPE"] == "voxtral-mini-4b-realtime-2602"


def test_launch_nemo_pipeline_runs_the_stream_server():
    recipes = recipe.load_recipes()
    streams, _ = streaming.load_streams(recipes=recipes)
    sid = "nvidia-nemotron-labs-unified-asr-0.6b"
    argv, env = launch.build_stream(sid, streams, recipes, {}, python="/p",
                                    checkpoint="/c.nemo")
    assert argv[:4] == ["/p", "-m", "uvicorn", "stream_server:app"]
    assert env["RAVEN_STREAM_RECIPE"] == sid
    assert env["BENCH_MODEL_PATH"] == "/c.nemo"
    assert env["BENCH_GPU_MEM_FRACTION"] == "0.85"


def test_vendored_configs_carry_the_keys_the_server_overrides():
    for name in ("cache_aware_rnnt", "buffered_rnnt"):
        text = (SERVING / "nemo" / "conf" / f"{name}.yaml").read_text()
        assert "b19b1a79d86ec30c1de078682b4b3ed0f59098bb" in text
    req = (SERVING / "nemo" / "requirements.txt").read_text()
    assert "b19b1a79d86ec30c1de078682b4b3ed0f59098bb" in req


# ── frame cutter ────────────────────────────────────────────────────────────


def test_frames_full_then_padded_last():
    c = FrameCutter(4)
    c.open(1)
    c.append(1, np.ones(6, dtype=np.float32))
    frames, _ = c.take(8)
    assert len(frames) == 1 and frames[0].is_first and not frames[0].is_last
    assert frames[0].length == 4
    assert c.take(8) == ([], [])  # 2 samples wait for more audio or the end
    c.end(1)
    frames, _ = c.take(8)
    f = frames[0]
    assert (f.is_first, f.is_last, f.length) == (False, True, 2)
    assert f.samples.tolist() == [1, 1, 0, 0]
    assert not c.is_open(1)


def test_end_on_a_frame_boundary_closes_with_one_zero_sample():
    c = FrameCutter(4)
    c.open(1)
    c.append(1, np.ones(4, dtype=np.float32))
    assert not c.take(8)[0][0].is_last
    c.end(1)
    f = c.take(8)[0][0]
    assert f.is_last and f.length == 1 and not f.samples.any()


def test_one_frame_per_stream_per_step_oldest_first_and_capped():
    t = [0.0]
    c = FrameCutter(2, clock=lambda: t[0])
    for sid in (1, 2, 3):
        c.open(sid)
    for sid in (3, 1, 2):
        t[0] += 1
        c.append(sid, np.ones(4, dtype=np.float32))
    frames, _ = c.take(2)
    assert [f.stream_id for f in frames] == [3, 1]
    frames, _ = c.take(8)
    assert sorted(f.stream_id for f in frames) == [1, 2, 3]


def test_stream_without_audio_closes_outside_the_pipeline():
    c = FrameCutter(4)
    c.open(7)
    c.end(7)
    assert c.take(8) == ([], [7])


# ── server protocol ─────────────────────────────────────────────────────────


class FakeEngine:
    model = "org/fake"
    frame_samples = 1600  # 100 ms
    max_batch = 4

    def __init__(self):
        self.seen: dict[int, int] = {}
        self.batches: list[int] = []

    def step(self, frames):
        from stream_server import StepResult

        self.batches.append(len(frames))
        out = []
        for f in frames:
            k = self.seen.get(f.stream_id, 0) + 1
            self.seen[f.stream_id] = k
            out.append(StepResult(f.stream_id, final=f"w{k} " if k % 2 == 0 or f.is_last else "",
                                  partial=f"p{k}", is_last=f.is_last))
        return out

    def info(self):
        return {"stream": {"id": "x", "sha256": "0" * 64}, "frame_ms": 100.0}


def _pcm(seconds: float) -> str:
    n = int(16000 * seconds)
    return base64.b64encode((np.ones(n) * 1000).astype("<i2").tobytes()).decode()


def test_server_speaks_the_vllm_realtime_protocol():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from stream_server import create_app

    eng = FakeEngine()
    with TestClient(create_app(eng)) as client, client.websocket_connect("/v1/realtime") as ws:
        assert json.loads(ws.receive_text())["type"] == "session.created"
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        assert json.loads(ws.receive_text())["code"] == "model_not_validated"
        ws.send_text(json.dumps({"type": "session.update", "model": "org/fake"}))
        for _ in range(2):  # two utterances on one connection
            ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
            for _ in range(3):
                ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(0.1)}))
            ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(0.05)}))
            ws.send_text(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            events = []
            while True:
                ev = json.loads(ws.receive_text())
                events.append(ev)
                if ev["type"] == "transcription.done":
                    break
            kinds = [e["type"] for e in events]
            assert "transcription.partial" in kinds
            deltas = "".join(e["delta"] for e in events if e["type"] == "transcription.delta")
            assert events[-1]["text"] == deltas == "w2 w4 "
        assert client.get("/v1/stream/info").json()["frame_ms"] == 100.0
    assert max(eng.batches) <= FakeEngine.max_batch


def test_server_rejects_another_model():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from stream_server import create_app

    with TestClient(create_app(FakeEngine())) as client, \
            client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "session.update", "model": "other"}))
        assert json.loads(ws.receive_text())["code"] == "model_not_found"


def test_streaming_toml_is_valid_toml_with_card_urls():
    data = tomllib.loads((SERVING / "streaming.toml").read_text())
    for sid, s in data["streams"].items():
        assert "https://huggingface.co/" in s["notes"], sid
