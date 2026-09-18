"""xAI Grok Voice Transcribe adapter — request shape, parsing, registry wiring.

No network: httpx.AsyncClient is swapped for one bound to a MockTransport.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import numpy as np
import pytest

from raven_asr import runner
from raven_asr.adapters.xai import XaiSttAdapter
from raven_asr.config import KNOWN_MODELS


def _silence() -> tuple[np.ndarray, int]:
    return np.zeros(1600, dtype=np.float32), 16000


def _route(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real_client = httpx.AsyncClient

    def _client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client)


def _field_order(request: httpx.Request) -> list[str]:
    body = request.read().decode("latin-1")
    names: list[str] = []
    for part in body.split("Content-Disposition: form-data; name=\"")[1:]:
        names.append(part.split('"', 1)[0])
    return names


def test_request_shape_and_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["fields"] = _field_order(request)
        seen["body"] = request.read().decode("latin-1")
        return httpx.Response(200, json={"text": " Guten Tag. ", "duration": 0.1})

    _route(monkeypatch, handler)
    result = asyncio.run(XaiSttAdapter().atranscribe(*_silence()))

    assert seen["url"] == "https://api.x.ai/v1/stt"
    assert seen["auth"] == "Bearer xai-test"
    # file must be the LAST field — the vendor ignores options sent after it
    assert seen["fields"] == ["model", "language", "format", "file"]
    assert "grok-voice-transcribe-2.0" in seen["body"]
    assert "\r\n\r\nde\r\n" in seen["body"]
    assert result.text == "Guten Tag."
    assert result.raw["body"] == {"text": " Guten Tag. ", "duration": 0.1}


def test_missing_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="XAI_API_KEY"):
        XaiSttAdapter()


def test_registry_builds_the_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    spec = KNOWN_MODELS["xai/grok-voice-transcribe-2.0"]
    adapter = runner._make_adapter(spec)
    assert isinstance(adapter, XaiSttAdapter)
    assert adapter.model_id == "grok-voice-transcribe-2.0"
    assert adapter.provider_id == spec.label
