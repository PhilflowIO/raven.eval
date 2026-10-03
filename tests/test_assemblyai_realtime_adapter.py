"""AssemblyAI streaming adapter — session protocol, transcript assembly, failures.

No network: each test runs a local WebSocket server on 127.0.0.1 that speaks the
v3 message shapes, and points the adapter at it with real-time pacing off.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from websockets.asyncio.server import ServerConnection, serve
from websockets.http11 import Request, Response

from raven_asr import runner
from raven_asr.adapters.assemblyai_realtime import (
    AssemblyAIRealtimeAdapter,
    SessionRate,
    SessionRefused,
)
from raven_asr.config import KNOWN_MODELS
from raven_asr.retry import TransientStreamError, with_retry

Handler = Callable[[ServerConnection], Awaitable[None]]


def _clip() -> tuple[np.ndarray, int]:
    # 0.35 s at 16 kHz: three full 100 ms chunks and one partial one.
    return np.full(5600, 0.25, dtype=np.float32), 16000


def _begin(model: str) -> dict[str, Any]:
    """The Begin message as the v3 API sends it (observed 2026-10-02)."""
    return {"type": "Begin", "id": "s1", "configuration": {"model": model}}


async def _drain(ws: ServerConnection, seen: dict[str, Any]) -> None:
    """Read audio until Terminate, recording what the client sent."""
    seen["audio_bytes"] = 0
    seen["chunks"] = 0
    async for message in ws:
        if isinstance(message, bytes):
            seen["audio_bytes"] += len(message)
            seen["chunks"] += 1
        elif json.loads(message).get("type") == "Terminate":
            seen["terminated"] = True
            return


def _run(
    handler: Handler,
    *,
    process_request: Callable[[ServerConnection, Request], Response | None] | None = None,
    retried: bool = False,
) -> Any:
    async def main() -> Any:
        async with serve(handler, "127.0.0.1", 0, process_request=process_request) as srv:
            port = srv.sockets[0].getsockname()[1]
            adapter = AssemblyAIRealtimeAdapter(
                base_url=f"ws://127.0.0.1:{port}/v3/ws", realtime=False, timeout_s=5
            )
            call = adapter.atranscribe if retried else _unretried(adapter)
            return await call(*_clip())

    return asyncio.run(main())


def _unretried(adapter: AssemblyAIRealtimeAdapter) -> Callable[..., Awaitable[Any]]:
    """One bare session: no transport retry, no waiting out refusals."""
    inner = AssemblyAIRealtimeAdapter._session.__wrapped__  # type: ignore[attr-defined]
    return lambda audio, sr: inner(adapter, audio, sr)


_LIMIT_ERROR = {
    "type": "Error",
    "error": "Unauthorized Connection: Too many concurrent sessions",
}


class _FakeClock:
    """Virtual time for SessionRate: sleeping advances the clock, nothing waits."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aai-test")


def test_session_protocol_and_transcript() -> None:
    seen: dict[str, Any] = {}

    async def handler(ws: ServerConnection) -> None:
        seen["path"] = ws.request.path
        seen["auth"] = ws.request.headers.get("Authorization")
        await ws.send(json.dumps(_begin("universal-3-6-pro")))
        await _drain(ws, seen)
        for turn in (
            {"turn_order": 0, "transcript": "Guten", "end_of_turn": False},
            {"turn_order": 0, "transcript": "Guten Tag.", "end_of_turn": True},
            # re-sent final for the same turn replaces, never duplicates
            {"turn_order": 0, "transcript": "Guten Tag!", "end_of_turn": True},
            {"turn_order": 1, "transcript": "Wie geht", "end_of_turn": False},
        ):
            await ws.send(json.dumps({"type": "Turn", **turn}))
        await ws.send(json.dumps({"type": "Termination", "audio_duration_seconds": 0.35}))

    result = _run(handler)

    query = parse_qs(urlparse(seen["path"]).query)
    assert query["speech_model"] == ["universal-3-6-pro"]
    assert query["sample_rate"] == ["16000"]
    assert query["encoding"] == ["pcm_s16le"]
    assert json.loads(query["language_codes"][0]) == ["de"]
    # no Bearer prefix — AssemblyAI takes the raw key
    assert seen["auth"] == "aai-test"
    assert seen["audio_bytes"] == 5600 * 2
    assert seen["chunks"] == 4
    assert seen["terminated"]
    # the partial-only turn keeps its words and is counted
    assert result.text == "Guten Tag! Wie geht"
    assert result.raw["n_unfinalised_turns"] == 1
    assert result.raw["n_turns"] == 2
    assert result.raw["latency_kind"] == "finalisation"
    assert result.latency_s >= 0.0


def test_pin_mismatch_is_fatal() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.send(json.dumps(_begin("universal-3-5-pro")))
        await _drain(ws, {})
        await ws.send(json.dumps({"type": "Termination"}))

    with pytest.raises(RuntimeError, match="universal-3-5-pro"):
        _run(handler)


def test_unattested_session_is_fatal() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.send(json.dumps({"type": "Begin", "id": "s1"}))
        await _drain(ws, {})
        await ws.send(json.dumps({"type": "Termination"}))

    with pytest.raises(RuntimeError, match="served model=None"):
        _run(handler)


def test_rate_limited_handshake_is_transient() -> None:
    def reject(conn: ServerConnection, request: Request) -> Response:
        return conn.respond(429, "slow down\n")

    async def handler(ws: ServerConnection) -> None:  # never reached
        raise AssertionError

    with pytest.raises(TransientStreamError, match="429"):
        _run(handler, process_request=reject)


def test_auth_failure_is_not_retried() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.close(1008, "Invalid API key")

    with pytest.raises(RuntimeError, match="1008") as info:
        _run(handler)
    assert not isinstance(info.value, TransientStreamError)


def test_vendor_error_message_fails_the_utterance() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.send(json.dumps({"type": "Error", "error": "bad sample rate"}))
        await _drain(ws, {})

    with pytest.raises(RuntimeError, match="bad sample rate"):
        _run(handler)


def test_session_limit_error_is_a_refusal() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.send(json.dumps(_LIMIT_ERROR))
        await _drain(ws, {})

    with pytest.raises(SessionRefused, match="session limit"):
        _run(handler)


def test_session_limit_close_code_is_a_refusal() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.close(3009, "Unauthorized Connection: Too many concurrent sessions")

    with pytest.raises(SessionRefused, match="3009"):
        _run(handler)


def test_refusals_wait_without_spending_the_retry_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """More refusals than the transport retry allows, and the utterance still lands."""
    caplog.set_level("INFO", logger="raven_asr.assemblyai")
    state = {"sessions": 0}

    async def handler(ws: ServerConnection) -> None:
        state["sessions"] += 1
        if state["sessions"] <= 7:  # with_retry gives up after 6 attempts
            await ws.send(json.dumps(_LIMIT_ERROR))
            await _drain(ws, {})
            return
        await ws.send(json.dumps(_begin("universal-3-6-pro")))
        await _drain(ws, {})
        await ws.send(json.dumps({"type": "Turn", "turn_order": 0,
                                  "transcript": "Grüezi", "end_of_turn": True}))
        await ws.send(json.dumps({"type": "Termination"}))

    async def main() -> Any:
        async with serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            rate = SessionRate(8, window_s=0.01)
            adapter = AssemblyAIRealtimeAdapter(
                base_url=f"ws://127.0.0.1:{port}/v3/ws", realtime=False,
                timeout_s=5, session_rate=rate,
            )
            return await adapter.atranscribe(*_clip()), rate

    result, rate = asyncio.run(main())
    assert result.text == "Grüezi"
    assert result.raw["n_refused_sessions"] == 7
    assert state["sessions"] == 8
    assert rate.per_min < 8  # the refusals taught it a lower limit
    # a run log must show every refusal and every rate change, or a slow run
    # cannot be told apart from a refused one
    refusals = [r for r in caplog.records if "session refused" in r.message]
    assert len(refusals) == 7 and all(r.levelname == "WARNING" for r in refusals)
    assert "(7 for this utterance" in refusals[-1].message
    assert any("rate halved" in r.message for r in caplog.records)


def test_more_utterances_than_the_limit_all_land() -> None:
    """A run larger than one minute's budget finishes with zero failures."""
    served = {"n": 0}

    async def handler(ws: ServerConnection) -> None:
        served["n"] += 1
        await ws.send(json.dumps(_begin("universal-3-6-pro")))
        await _drain(ws, {})
        await ws.send(json.dumps({"type": "Turn", "turn_order": 0,
                                  "transcript": "ok", "end_of_turn": True}))
        await ws.send(json.dumps({"type": "Termination"}))

    async def main() -> list[Any]:
        async with serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            adapter = AssemblyAIRealtimeAdapter(
                base_url=f"ws://127.0.0.1:{port}/v3/ws", realtime=False,
                timeout_s=5, session_rate=SessionRate(3, window_s=0.05),
            )
            return list(await asyncio.gather(
                *(adapter.atranscribe(*_clip()) for _ in range(10))
            ))

    results = asyncio.run(main())
    assert [r.text for r in results] == ["ok"] * 10
    assert served["n"] == 10


def test_endless_refusal_fails_loudly() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.send(json.dumps(_LIMIT_ERROR))
        await _drain(ws, {})

    async def main() -> Any:
        async with serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            adapter = AssemblyAIRealtimeAdapter(
                base_url=f"ws://127.0.0.1:{port}/v3/ws", realtime=False,
                timeout_s=5, session_rate=SessionRate(5, window_s=0.01),
                refusal_patience_s=0.05,
            )
            return await adapter.atranscribe(*_clip())

    with pytest.raises(RuntimeError, match="kept refusing") as info:
        asyncio.run(main())
    assert not isinstance(info.value, TransientStreamError)


def test_rate_holds_openings_to_the_window() -> None:
    clock = _FakeClock()
    rate = SessionRate(3, clock=clock, sleep=clock.sleep)

    async def open_five() -> list[float]:
        opened = []
        for _ in range(5):
            await rate.acquire()
            opened.append(clock.now)
        return opened

    # three at once, the fourth only when the first leaves the 60 s window
    assert asyncio.run(open_five()) == [0.0, 0.0, 0.0, 60.0, 60.0]


def test_rate_halves_once_per_burst_and_grows_like_the_vendor() -> None:
    clock = _FakeClock()
    rate = SessionRate(10, clock=clock, sleep=clock.sleep)
    for _ in range(4):  # one burst refuses four utterances in flight
        rate.refused()
    assert rate.per_min == 5

    async def fill_windows(n: int) -> None:
        for _ in range(n):
            for _ in range(int(rate.per_min)):
                await rate.acquire()
            clock.now += 60.0

    asyncio.run(fill_windows(3))
    # the burst's window does not count as headroom; full windows after it do
    assert 5 * 1.1 <= rate.per_min <= 5 * 1.1**2 + 1e-9


def test_rate_rejects_a_limit_below_one() -> None:
    with pytest.raises(ValueError):
        SessionRate(0.5)


def test_rate_reads_the_paid_limit_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASSEMBLYAI_SESSIONS_PER_MIN", "100")
    assert AssemblyAIRealtimeAdapter().session_rate.per_min == 100


def test_retry_reruns_a_failed_session() -> None:
    calls = {"n": 0}

    @with_retry(base_backoff=0.0)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientStreamError("closed 1011")
        return "ok"

    assert asyncio.run(flaky()) == "ok"
    assert calls["n"] == 3


def test_missing_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ASSEMBLYAI_API_KEY"):
        AssemblyAIRealtimeAdapter()


def test_registry_builds_the_adapter() -> None:
    spec = KNOWN_MODELS["assemblyai/universal-3-6-pro-realtime"]
    adapter = runner._make_adapter(spec)
    assert isinstance(adapter, AssemblyAIRealtimeAdapter)
    assert adapter.model_id == "universal-3-6-pro"
    assert adapter.provider_id == spec.label
    assert spec.revision == spec.model_id

