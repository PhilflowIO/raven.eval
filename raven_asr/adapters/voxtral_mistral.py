"""Voxtral via Mistral API (async)."""

from __future__ import annotations

import asyncio
import os
import time

import httpx
import numpy as np

from ..retry import with_retry
from .base import TranscribeResult, encode_wav_pcm16

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MODEL = "voxtral-mini-latest"
DEFAULT_TIMEOUT_S = 180.0


class VoxtralMistralAdapter:
    """Mistral Voxtral transcription adapter."""

    def __init__(
        self,
        *,
        provider_id: str = "voxtral-mistral",
        model_id: str = DEFAULT_MODEL,
        api_key_env: str = "MISTRAL_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        language: str = "de",
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set — required for Mistral Voxtral API"
            )
        self.provider_id = provider_id
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._timeout_s = timeout_s

    @with_retry()
    async def atranscribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> TranscribeResult:
        wav_bytes = encode_wav_pcm16(audio, sample_rate)
        headers = {"Authorization": f"Bearer {self._api_key}"}
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
