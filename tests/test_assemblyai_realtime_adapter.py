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
from raven_asr.adapters.assemblyai_realtime import AssemblyAIRealtimeAdapter
from raven_asr.config import KNOWN_MODELS
from raven_asr.retry import TransientStreamError, with_retry

Handler = Callable[[ServerConnection], Awaitable[None]]


def _clip() -> tuple[np.ndarray, int]:
    # 0.35 s at 16 kHz: three full 100 ms chunks and one partial one.
    return np.full(5600, 0.25, dtype=np.float32), 16000


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
    inner = AssemblyAIRealtimeAdapter.atranscribe.__wrapped__  # type: ignore[attr-defined]
    return lambda audio, sr: inner(adapter, audio, sr)


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aai-test")


def test_session_protocol_and_transcript() -> None:
    seen: dict[str, Any] = {}

    async def handler(ws: ServerConnection) -> None:
        seen["path"] = ws.request.path
        seen["auth"] = ws.request.headers.get("Authorization")
        await ws.send(json.dumps({"type": "Begin", "id": "s1"}))
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
        await ws.send(json.dumps({"type": "Begin", "speech_model": "universal-3-5-pro"}))
        await _drain(ws, {})
        await ws.send(json.dumps({"type": "Termination"}))

    with pytest.raises(RuntimeError, match="universal-3-5-pro"):
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

