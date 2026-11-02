"""The self-hosted endpoint contract, against a mock server (no network).

What a long unattended run against a NeMo or vLLM server relies on: the request
shape, 16 kHz audio on the wire, one HTTP client for the whole run, and an abort
when the server answers for a different model than the run is labelled with.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import Callable

import pytest

pytest.importorskip("httpx")
pytest.importorskip("soxr")

import httpx
import numpy as np

from raven_asr.adapters.base import ServedModelMismatchError
from raven_asr.adapters.vllm_openai import VllmOpenAIAdapter

MODEL = "nvidia/canary-1b-v2"


def _adapter(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str | None = None,
) -> VllmOpenAIAdapter:
    monkeypatch.setenv("NEMO_BENCH_URL", "http://bench.invalid/v1/")
    if api_key is None:
        monkeypatch.delenv("BENCH_STT_API_KEY", raising=False)
    else:
        monkeypatch.setenv("BENCH_STT_API_KEY", api_key)
    return VllmOpenAIAdapter(
        provider_id="nvidia-canary-1b-v2",
        model_id=MODEL,
        base_url_env="NEMO_BENCH_URL",
        api_key_env="BENCH_STT_API_KEY",
        transport=httpx.MockTransport(handler),
    )


def _wav_rate(body: bytes) -> int:
    """Sample rate from the RIFF header of the WAV inside a multipart body."""
    riff = body.index(b"RIFF")
    return int(struct.unpack_from("<I", body, riff + 24)[0])


def _run(adapter: VllmOpenAIAdapter, audio: np.ndarray, sr: int, n: int = 1) -> list[str]:
    async def go() -> list[str]:
        try:
            out = await asyncio.gather(*(adapter.atranscribe(audio, sr) for _ in range(n)))
            return [r.text for r in out]
        finally:
            await adapter.aclose()

    return asyncio.run(go())


def test_request_is_a_16khz_multipart_post_with_model_and_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": " hallo ", "x_raven": {"model": MODEL}})

    adapter = _adapter(monkeypatch, handler, api_key="s3cret")
    # 48 kHz, like the SPC corpus: the server must still receive 16 kHz.
    texts = _run(adapter, np.zeros(48_000, dtype=np.float32), 48_000)

    assert texts == ["hallo"]
    (req,) = seen
    assert req.method == "POST"
    assert str(req.url) == "http://bench.invalid/v1/audio/transcriptions"
    assert req.headers["authorization"] == "Bearer s3cret"
    body = req.read()
    assert b'name="model"\r\n\r\nnvidia/canary-1b-v2' in body
    assert b'name="language"\r\n\r\nde' in body
    assert _wav_rate(body) == 16_000


def test_no_authorization_header_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "x", "x_raven": {"model": MODEL}})

    _run(_adapter(monkeypatch, handler), np.zeros(1600, dtype=np.float32), 16_000)
    assert "authorization" not in seen[0].headers


def test_one_http_client_serves_every_request(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[httpx.AsyncClient] = []
    real_init = httpx.AsyncClient.__init__

    def counting_init(self: httpx.AsyncClient, *a: object, **kw: object) -> None:
        created.append(self)
        real_init(self, *a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.AsyncClient, "__init__", counting_init)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "x", "x_raven": {"model": MODEL}})

    adapter = _adapter(monkeypatch, handler)
    _run(adapter, np.zeros(1600, dtype=np.float32), 16_000, n=25)
    assert len(created) == 1
    assert created[0].is_closed
    assert adapter._client is None  # closed and released by aclose()


def test_a_server_serving_another_model_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"text": "x", "x_raven": {"model": "nvidia/parakeet-tdt-0.6b-v3"}}
        )

    adapter = _adapter(monkeypatch, handler)
    with pytest.raises(ServedModelMismatchError, match="parakeet-tdt"):
        _run(adapter, np.zeros(1600, dtype=np.float32), 16_000)


def test_a_server_without_an_echo_is_warned_about_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "x"})

    adapter = _adapter(monkeypatch, handler)
    with caplog.at_level(logging.WARNING, logger="raven_asr.adapters.vllm_openai"):
        texts = _run(adapter, np.zeros(1600, dtype=np.float32), 16_000, n=5)
    assert texts == ["x"] * 5
    warnings = [r for r in caplog.records if "does not report" in r.getMessage()]
    assert len(warnings) == 1


def test_missing_endpoint_env_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BENCH_URL", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_BENCH_URL"):
        VllmOpenAIAdapter(provider_id="p", model_id="m", base_url_env="VLLM_BENCH_URL")


def test_a_model_with_an_output_format_is_parsed_and_its_raw_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MOSS markup leaves the adapter as spoken text, the server's string beside it."""
    served = "[0.48][S01]guten tag[3.12][3.12][S02]hallo[4.0]"
    monkeypatch.setenv("VLLM_BENCH_URL", "http://bench.invalid/v1/")
    moss = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    adapter = VllmOpenAIAdapter(
        provider_id="moss-transcribe-diarize", model_id=moss,
        base_url_env="VLLM_BENCH_URL", api_key_env=None,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"text": served, "x_raven": {"model": moss}})
        ),
        output_format="moss-timed-speakers",
    )

    async def go():
        try:
            return await adapter.atranscribe(np.zeros(16_000, dtype=np.float32), 16_000)
        finally:
            await adapter.aclose()

    result = asyncio.run(go())
    assert result.text == "guten tag hallo"
    assert result.raw_text == served
    assert [s["speaker"] for s in result.segments] == ["S01", "S02"]


def test_a_plain_model_records_no_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(
        monkeypatch,
        lambda request: httpx.Response(200, json={"text": " [0.5] hallo ", "x_raven": {"model": MODEL}}),
    )

    async def go():
        try:
            return await adapter.atranscribe(np.zeros(16_000, dtype=np.float32), 16_000)
        finally:
            await adapter.aclose()

    result = asyncio.run(go())
    # No output format: the server's text is the transcript, untouched.
    assert (result.text, result.raw_text, result.segments) == ("[0.5] hallo", None, None)
