"""xAI Grok Voice Transcribe — hosted speech-to-text (async).

REST ``POST https://api.x.ai/v1/stt``, multipart form. Two request rules from the
vendor documentation (docs.x.ai/developers/model-capabilities/audio/speech-to-text,
read 2026-09-18) that silently change the result rather than erroring:

* ``file`` must be the LAST multipart field — option fields sent after it may be
  ignored for streamed uploads. httpx encodes ``data`` before ``files``, which is
  the order required; the adapter test pins it.
* ``language`` only enables text formatting; the model transcribes any supported
  language regardless. ``format=true`` (inverse text normalization, spoken numbers
  to digits) requires ``language``. We send ``language=de`` + ``format=true`` so the
  output is the written form a user would read — the strict-de lens re-expands
  digits to words, so ITN does not move the WER in either direction.

Filler words are removed by default (``filler_words=false``); left at default, as
every other hosted adapter here is measured at its vendor default.
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx
import numpy as np

from ..retry import with_retry
from .base import TranscribeResult, encode_wav_pcm16

DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-voice-transcribe-2.0"
DEFAULT_TIMEOUT_S = 180.0


class XaiSttAdapter:
    """Adapter for xAI's batch speech-to-text endpoint."""

    def __init__(
        self,
        *,
        provider_id: str = "xai",
        model_id: str = DEFAULT_MODEL,
        api_key_env: str = "XAI_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        language: str = "de",
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set — required for xAI STT API")
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
        data = {"model": self.model_id, "language": self._language, "format": "true"}
        files = {"file": ("clip.wav", wav_bytes, "audio/wav")}
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(
                f"{self._base_url}/stt", data=data, files=files, headers=headers
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
