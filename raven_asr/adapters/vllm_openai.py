"""OpenAI-compatible ``/v1/audio/transcriptions`` adapter (async)."""

from __future__ import annotations

import asyncio
import os
import time

import httpx
import numpy as np

from ..retry import with_retry
from .base import TranscribeResult, encode_wav_pcm16

DEFAULT_LANGUAGE = "de"
DEFAULT_TIMEOUT_S = 300.0


class VllmOpenAIAdapter:
    """Posts WAV bytes to ``{base_url}/audio/transcriptions``."""

    def __init__(
        self,
        *,
        provider_id: str,
        model_id: str,
        base_url_env: str = "VLLM_PRIMELINE_URL",
        api_key_env: str | None = "VLLM_PRIMELINE_API_KEY",
        language: str = DEFAULT_LANGUAGE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        base_url = os.environ.get(base_url_env)
        if not base_url:
            raise RuntimeError(
                f"{base_url_env} is not set — expected an OpenAI-compatible "
                f"endpoint base URL"
            )
        self.provider_id = provider_id
        self.model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._api_key = os.environ.get(api_key_env) if api_key_env else None
        self._language = language
        self._timeout_s = timeout_s

    @with_retry()
    async def atranscribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> TranscribeResult:
        wav_bytes = encode_wav_pcm16(audio, sample_rate)
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        files = {"file": ("clip.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model_id, "language": self._language}
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(
                f"{self._base_url}/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
            )
        r.raise_for_status()
        latency = time.monotonic() - started
        body: dict[str, object] = r.json()
        text = str(body.get("text", "")).strip()
        return TranscribeResult(
            text=text,
            latency_s=latency,
            raw={"body": body, "endpoint": self._base_url},
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        return asyncio.run(self.atranscribe(audio, sample_rate))
