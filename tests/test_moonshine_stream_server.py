"""How the Moonshine live server shares the vendor library between sessions.

No moonshine-voice install: the pool, the stream wrapper and the thread switch
are driven with fake Transcribers, the recipe keys through streaming.toml.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

SERVING = Path(__file__).resolve().parents[1] / "serving"
sys.path.insert(0, str(SERVING))

import recipe
import streaming


def _server():
    spec = importlib.util.spec_from_file_location(
        "moonshine_stream_server_pool", SERVING / "moonshine" / "stream_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


srv = _server()


def _load(tmp_path: Path, text: str):
    p = tmp_path / "s.toml"
    p.write_text(text)
    return streaming.load_streams(p)


# ── recipe keys ─────────────────────────────────────────────────────────────


def test_published_recipe_states_the_vendor_default_and_the_variant_differs_only_in_sharing():
    streams, _ = streaming.load_streams()
    vendor = streams["moonshine-streaming-small-de"]
    tuned = streams["moonshine-streaming-small-de-per-stream"]
    assert (vendor["transcribers"], vendor["ort_threads"]) == ("shared", "vendor")
    assert (tuned["transcribers"], tuned["ort_threads"]) == ("per_stream", "single")
    same = set(vendor) - {"transcribers", "ort_threads", "notes"}
    assert {k: vendor[k] for k in same} == {k: tuned[k] for k in same}
    assert streaming.stream_sha256("moonshine-streaming-small-de") != \
        streaming.stream_sha256("moonshine-streaming-small-de-per-stream")


@pytest.mark.parametrize("edit, msg", [
    (('transcribers = "shared"', 'transcribers = "pool"'), "transcribers must be one of"),
    (('ort_threads = "vendor"', 'ort_threads = 4'), "ort_threads must be one of"),
    (('transcribers = "shared"\n', ''), "needs"),
    (('ort_threads = "vendor"\n', ''), "needs"),
])
def test_moonshine_sharing_keys_are_validated(tmp_path, edit, msg):
    text = (SERVING / "streaming.toml").read_text()
    assert edit[0] in text
    with pytest.raises(recipe.RecipeError, match=msg):
        _load(tmp_path, text.replace(*edit, 1))


def test_sharing_keys_are_part_of_the_stream_identity(tmp_path):
    text = (SERVING / "streaming.toml").read_text()
    sid = "moonshine-streaming-small-de"
    a, _ = _load(tmp_path, text)
    b, _ = _load(tmp_path, text.replace('ort_threads = "vendor"', 'ort_threads = "single"', 1))
    assert streaming.stream_sha256(sid, streams=a) != streaming.stream_sha256(sid, streams=b)


# ── thread switch ───────────────────────────────────────────────────────────


def test_ort_threads_sets_and_clears_the_library_switch(monkeypatch):
    monkeypatch.setenv(srv.SINGLE_THREAD_ENV, "1")
    srv.apply_ort_threads("vendor")
    import os
    assert srv.SINGLE_THREAD_ENV not in os.environ  # an inherited value cannot leak in
    srv.apply_ort_threads("single")
    assert os.environ[srv.SINGLE_THREAD_ENV] == "1"
    with pytest.raises(RuntimeError):
        srv.apply_ort_threads("4")


# ── transcriber pool ────────────────────────────────────────────────────────


class FakeTranscriber:
    loaded = 0

    def __init__(self):
        FakeTranscriber.loaded += 1
        self.id = FakeTranscriber.loaded


def _pool(shared: bool):
    FakeTranscriber.loaded = 0
    return srv.TranscriberPool(FakeTranscriber, shared=shared)


def test_shared_pool_hands_every_session_the_same_transcriber():
    pool = _pool(shared=True)
    a, b = pool.take(), pool.take()
    assert a is b is pool.first
    pool.give(a)
    assert pool.loaded == FakeTranscriber.loaded == 1


def test_per_stream_pool_grows_to_the_live_sessions_and_reuses_them():
    pool = _pool(shared=False)
    live = [pool.take() for _ in range(3)]
    assert len({t.id for t in live}) == 3
    assert pool.loaded == FakeTranscriber.loaded == 3
    for t in live:
        pool.give(t)
    again = [pool.take() for _ in range(3)]
    assert {t.id for t in again} == {t.id for t in live}  # no new load
    assert FakeTranscriber.loaded == 3


def test_per_stream_pool_never_hands_one_transcriber_to_two_live_sessions():
    pool = _pool(shared=False)
    held, lock, errors = set(), threading.Lock(), []

    def session():
        for _ in range(200):
            t = pool.take()
            with lock:
                if t.id in held:
                    errors.append(t.id)
                held.add(t.id)
            with lock:
                held.discard(t.id)
            pool.give(t)

    ths = [threading.Thread(target=session) for _ in range(8)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    assert errors == []
    assert pool.loaded <= 8


class FakeLibStream:
    def __init__(self):
        self.calls = []

    def add_audio(self, audio, sample_rate):
        self.calls.append(("add", len(audio), sample_rate))

    def stop(self):
        self.calls.append(("stop",))
        return "t"

    def close(self):
        self.calls.append(("close",))


def test_pooled_stream_gives_its_transcriber_back_once_on_close():
    released = []
    lib = FakeLibStream()
    s = srv.PooledStream(lib, lambda: released.append(1))
    s.add_audio([0.0] * 1600, 16000)
    assert s.stop() == "t"
    s.close()
    s.close()
    assert released == [1]
    assert lib.calls == [("add", 1600, 16000), ("stop",), ("close",), ("close",)]


def test_pooled_stream_releases_even_when_the_library_close_fails():
    released = []

    class Failing(FakeLibStream):
        def close(self):
            raise RuntimeError("boom")

    s = srv.PooledStream(Failing(), lambda: released.append(1))
    with pytest.raises(RuntimeError):
        s.close()
    assert released == [1]


# ── model download ──────────────────────────────────────────────────────────


def test_fetch_files_names_its_client(tmp_path, monkeypatch):
    import hashlib
    import io

    body = b"weights"
    seen = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req):
        seen.append((req.full_url, req.get_header("User-agent")))
        return Resp(body)

    monkeypatch.setattr(srv.urllib.request, "urlopen", urlopen)
    files = {"a.ort": hashlib.sha256(body).hexdigest()}
    target = srv.fetch_files("https://example.invalid/m/v1", files, tmp_path)
    assert (target / "a.ort").read_bytes() == body
    assert seen == [("https://example.invalid/m/v1/a.ort", srv.USER_AGENT)]
    assert not srv.USER_AGENT.startswith("Python-urllib")
