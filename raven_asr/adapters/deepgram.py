"""Deepgram Nova-2/3 — closed-source reference for the German bench (async)."""

from __future__ import annotations

import asyncio
import os
import time

import httpx
import numpy as np

from ..retry import with_retry
from .base import TranscribeResult, encode_wav_pcm16

DEFAULT_BASE_URL = "https://api.deepgram.com/v1"
DEFAULT_MODEL = "nova-2"
DEFAULT_TIMEOUT_S = 180.0


class DeepgramAdapter:
    """Adapter for Deepgram's prerecorded transcription endpoint."""

    def __init__(
        self,
        *,
        provider_id: str = "deepgram",
        model_id: str = DEFAULT_MODEL,
        api_key_env: str = "DEEPGRAM_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        language: str = "de",
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set — required for Deepgram API"
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
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/wav",
        }
        params = {
            "model": self.model_id,
            "language": self._language,
            "smart_format": "false",
            "punctuate": "true",
        }
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(
                f"{self._base_url}/listen",
                content=wav_bytes,
                params=params,
                headers=headers,
            )
        r.raise_for_status()
        latency = time.monotonic() - started
        body: dict[str, object] = r.json()
        text = _extract_transcript(body)
        return TranscribeResult(
            text=text,
            latency_s=latency,
            raw={"body": body, "endpoint": self._base_url},
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        return asyncio.run(self.atranscribe(audio, sample_rate))


def _extract_transcript(body: dict[str, object]) -> str:
    """Pull the top alternative transcript from Deepgram's nested response."""
    results = body.get("results")
    if not isinstance(results, dict):
        return ""
    channels = results.get("channels")
    if not isinstance(channels, list) or not channels:
        return ""
    first = channels[0]
    if not isinstance(first, dict):
        return ""
    alts = first.get("alternatives")
    if not isinstance(alts, list) or not alts:
        return ""
    top = alts[0]
    if not isinstance(top, dict):
        return ""
    transcript = top.get("transcript", "")
    return str(transcript).strip()
