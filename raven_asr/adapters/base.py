"""Common types and helpers for ASR provider adapters."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import soundfile as sf

# Defined beside the serving block (stdlib-only, so promote/verify can import
# it); re-exported here, where every adapter has always imported it from.
from ..serving import ServedModelMismatchError, ServingChangedError

__all__ = [
    "ASRAdapter", "ServedModelMismatchError", "ServingChangedError",
    "TranscribeResult", "encode_wav_pcm16", "parse_word_timestamps", "resample_mono",
    "sync_from_async",
]


@dataclass(frozen=True)
class TranscribeResult:
    """Output of one transcription call.

    Attributes:
        text: The spoken text, pre-normalize — the model output with any markup
            of the model's own output format removed (raven_asr.output_formats).
        latency_s: Wall-clock seconds from request send to response decode.
        raw: Provider-specific debug payload (kept for traceability).
        raw_text: The server's transcript exactly as returned, set only when the
            model has an output format and ``text`` was parsed out of it.
        segments: The speaker turns that format carried (JSON-ready dicts), or
            None when the model has no output format.
        words: The word timestamps the server returned
            (``[{"word", "start", "end"}]``, seconds from the start of the clip),
            or None when it returned none (:func:`parse_word_timestamps`).
    """

    text: str
    latency_s: float
    raw: dict[str, object]
    raw_text: str | None = None
    segments: list[dict[str, object]] | None = None
    words: list[dict[str, object]] | None = None


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


def parse_word_timestamps(body: dict[str, object]) -> list[dict[str, object]] | None:
    """The word timestamps in a transcription response, or None when it has none.

    Two shapes reach the harness. The NeMo server (serving/nemo/server.py,
    ``_extract_timestamps``) answers every request with
    ``timestamps: {"word": [{"word", "start", "end"}, ...], "segment": [...]}``;
    an OpenAI ``verbose_json`` answer (vLLM, the transformers stand-in) carries
    ``words: [{"word", "start", "end"}, ...]``. Both become one list of
    ``{"word", "start", "end"}`` with float seconds. An entry without a usable
    start/end is dropped rather than guessed; an empty list means the server
    sent the field but no word in it.
    """
    nested = body.get("timestamps")
    raw: object = nested.get("word") if isinstance(nested, dict) else None
    if raw is None:
        raw = body.get("words")
    if not isinstance(raw, list):
        return None
    words: list[dict[str, object]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("word") or entry.get("text") or "").strip()
        try:
            start = float(entry["start"])
            end = float(entry["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if text:
            words.append({"word": text, "start": start, "end": max(start, end)})
    return words


def sync_from_async(
    adapter: ASRAdapter, audio: np.ndarray, sample_rate: int
) -> TranscribeResult:
    """Helper: run ``adapter.atranscribe(...)`` from synchronous code."""
    return asyncio.run(adapter.atranscribe(audio, sample_rate))


def resample_mono(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono float audio to ``target_rate`` with soxr "VHQ" (no-op if equal).

    For endpoints whose input contract is one fixed rate. Loaders yield the
    corpus's native rate (the Swiss corpora are 44.1/48 kHz), so the conversion
    happens at the adapter boundary — the one place that knows the contract.
    """
    if sample_rate == target_rate:
        return audio
    import soxr

    return np.asarray(
        soxr.resample(
            np.asarray(audio, dtype=np.float32),
            sample_rate,
            target_rate,
            quality="VHQ",
        ),
        dtype=np.float32,
    )


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
