"""Common types and helpers for ASR provider adapters."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class TranscribeResult:
    """Output of one transcription call.

    Attributes:
        text: Raw model output, pre-normalize.
        latency_s: Wall-clock seconds from request send to response decode.
        raw: Provider-specific debug payload (kept for traceability).
    """

    text: str
    latency_s: float
    raw: dict[str, object]


class ASRAdapter(Protocol):
    """Protocol every provider adapter implements.

    Adapters are async-first: the runner exclusively calls ``atranscribe``
    inside an ``asyncio.gather`` to maximise concurrency. The sync
    ``transcribe`` method exists only as a compatibility shim for legacy
    callers and tests — implementations may simply delegate via
    ``asyncio.run(self.atranscribe(...))``.
    """

    provider_id: str
    model_id: str

    async def atranscribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> TranscribeResult:
        """Async: submit audio, return transcript + latency."""
        ...

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        """Sync wrapper around ``atranscribe`` for legacy callers."""
        ...


def sync_from_async(
    adapter: ASRAdapter, audio: np.ndarray, sample_rate: int
) -> TranscribeResult:
    """Helper: run ``adapter.atranscribe(...)`` from synchronous code."""
    return asyncio.run(adapter.atranscribe(audio, sample_rate))


def encode_wav_pcm16(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono PCM into a 16-bit WAV in-memory.

    Matches the codec the Qwen3-ASR-Server expects and what every
    OpenAI-compatible ``/v1/audio/transcriptions`` endpoint accepts.
    """
    if audio.ndim != 1:
        audio = np.asarray(audio).mean(axis=tuple(range(1, audio.ndim)))
    buffer = io.BytesIO()
    sf.write(
        buffer,
        audio.astype(np.float32, copy=False),
        sample_rate,
        subtype="PCM_16",
        format="WAV",
    )
    return buffer.getvalue()
