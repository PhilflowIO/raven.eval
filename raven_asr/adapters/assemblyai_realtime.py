"""AssemblyAI Universal-3.6 Pro Realtime — hosted streaming speech-to-text (async).

The first *streaming* adapter in this harness. Every other adapter posts a whole
clip and reads one answer; a realtime model is measured the way it is used: the
clip is fed over AssemblyAI's v3 WebSocket (``wss://streaming.assemblyai.com/v3/ws``)
in 100 ms PCM16 chunks **at real-time pace**, and the transcript is what the
session finalises. Feeding faster than real time would measure a mode no live
meeting ever runs in. Vendor documentation read 2026-10-02
(assemblyai.com/docs/api-reference/streaming-api, migration guide to 3.6 Pro).

Request rules that change the result rather than erroring:

* ``speech_model`` must be sent: a connection without it is served by the
  default model (``universal-3-5-pro`` on the date above), so an omitted pin
  silently measures the predecessor.
* ``language_codes=["de"]`` — 3.6 Pro code-switches across 32 languages and
  biases per token toward the listed ones. A single-element list is the
  vendor's documented monolingual setting, the counterpart of ``language=de`` on
  every other adapter here. ``mode`` and the turn-silence knobs stay at vendor
  default, as every hosted adapter here is measured at its vendor default.
* Formatting is always on in 3.6 Pro (no ``format_turns``); the strict-de lens
  normalises it away.

The transcript is the concatenation of the session's *final* turns
(``end_of_turn: true``), keyed by ``turn_order`` so a re-sent turn replaces
rather than duplicates. ``Terminate`` makes the server flush the audio it holds
and close with a ``Termination`` message; a turn that never received its final
form by then contributes its last partial and is counted in ``raw`` — dropping
it would delete words the model did hear.

``latency_s`` here is **finalisation latency**: seconds from the last audio chunk
leaving the client to the last final turn arriving. At real-time pace, request
wall-clock is just the clip length, so the harness's usual "send → response"
reading would say nothing about the model.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

import numpy as np

from ..retry import TransientStreamError, with_retry
from .base import TranscribeResult

DEFAULT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
DEFAULT_MODEL = "universal-3-6-pro"
DEFAULT_CHUNK_S = 0.1
DEFAULT_TIMEOUT_S = 60.0

# Handshake statuses and close codes a fresh session can recover from. Anything
# else (auth 1008, bad params, 4xx handshake) is a configuration error and must
# fail the run loudly, not be retried five times.
_RETRYABLE_HANDSHAKE = frozenset({408, 425, 429, 500, 502, 503, 504})
_RETRYABLE_CLOSE = frozenset({1006, 1011, 1012, 1013, 1014})


@dataclass
class _Session:
    """What one streaming session produced, collected by the receive loop."""

    finals: dict[int, str] = field(default_factory=dict)
    partials: dict[int, str] = field(default_factory=dict)
    last_final_at: float | None = None
    begin: dict[str, object] = field(default_factory=dict)
    termination: dict[str, object] = field(default_factory=dict)
    n_messages: int = 0

    def transcript(self) -> tuple[str, int]:
        """Final turns in order; a turn left without a final contributes its partial."""
        orders = sorted(set(self.finals) | set(self.partials))
        unfinalised = [o for o in orders if o not in self.finals]
        parts = [self.finals.get(o, self.partials.get(o, "")).strip() for o in orders]
        return " ".join(p for p in parts if p), len(unfinalised)


def _pcm16(audio: np.ndarray) -> bytes:
    """Float mono in [-1, 1] → little-endian signed 16-bit PCM (``pcm_s16le``)."""
    if audio.ndim != 1:
        audio = np.asarray(audio).mean(axis=tuple(range(1, audio.ndim)))
    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class AssemblyAIRealtimeAdapter:
    """Adapter for AssemblyAI's v3 streaming WebSocket, paced at real time."""

    def __init__(
        self,
        *,
        provider_id: str = "assemblyai-universal-3-6-pro-realtime",
        model_id: str = DEFAULT_MODEL,
        api_key_env: str = "ASSEMBLYAI_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        language: str = "de",
        chunk_s: float = DEFAULT_CHUNK_S,
        realtime: bool = True,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set — required for AssemblyAI streaming API"
            )
        self.provider_id = provider_id
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._language = language
        self._chunk_s = chunk_s
        # Off only in tests: a unit test must not sleep through a clip.
        self._realtime = realtime
        self._timeout_s = timeout_s

    def _url(self, sample_rate: int) -> str:
        params = {
            "speech_model": self.model_id,
            "sample_rate": str(sample_rate),
            "encoding": "pcm_s16le",
            "language_codes": json.dumps([self._language]),
        }
        return f"{self._base_url}?{urlencode(params)}"

    @with_retry()
    async def atranscribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> TranscribeResult:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        pcm = _pcm16(audio)
        step = round(self._chunk_s * sample_rate) * 2  # bytes per chunk
        session = _Session()
        try:
            async with connect(
                self._url(sample_rate),
                additional_headers={"Authorization": self._api_key},
                open_timeout=self._timeout_s,
                max_size=None,
            ) as ws:
                audio_done = await self._exchange(ws, pcm, step, session)
        except InvalidStatus as exc:
            status = exc.response.status_code
            if status in _RETRYABLE_HANDSHAKE:
                raise TransientStreamError(f"handshake HTTP {status}") from exc
            raise RuntimeError(
                f"AssemblyAI streaming handshake rejected: HTTP {status}"
            ) from exc
        except ConnectionClosed as exc:
            code = exc.rcvd.code if exc.rcvd is not None else 1006
            reason = exc.rcvd.reason if exc.rcvd is not None else ""
            if code in _RETRYABLE_CLOSE:
                raise TransientStreamError(f"closed {code} {reason!r}") from exc
            raise RuntimeError(
                f"AssemblyAI streaming session closed {code}: {reason!r}"
            ) from exc
        except (OSError, TimeoutError) as exc:
            raise TransientStreamError(f"{type(exc).__name__}: {exc}") from exc

        if not session.termination:
            raise TransientStreamError("session closed without a Termination message")
        _check_pin(self.model_id, session.begin)

        text, n_unfinalised = session.transcript()
        last = session.last_final_at
        latency = max(0.0, last - audio_done) if last is not None else 0.0
        return TranscribeResult(
            text=text,
            latency_s=latency,
            raw={
                "begin": session.begin,
                "termination": session.termination,
                "n_turns": len(set(session.finals) | set(session.partials)),
                "n_unfinalised_turns": n_unfinalised,
                "n_messages": session.n_messages,
                "endpoint": self._base_url,
                "latency_kind": "finalisation",
            },
        )

    async def _exchange(
        self, ws: object, pcm: bytes, step: int, session: _Session
    ) -> float:
        """Send and receive concurrently; return when the audio was fully sent.

        Both directions run as tasks under one supervisor: whichever fails first
        decides the outcome and the other is cancelled and awaited, so a vendor
        ``Error`` stops the sender mid-clip and a dead socket never leaves an
        orphaned reader behind.
        """
        receiver = asyncio.create_task(_receive(ws, session))
        sender = asyncio.create_task(self._send(ws, pcm, step))
        try:
            await asyncio.wait(
                {sender, receiver}, return_when=asyncio.FIRST_EXCEPTION
            )
            # The receiver's failure carries the vendor's reason; prefer it.
            if receiver.done() and receiver.exception() is not None:
                raise receiver.exception()  # type: ignore[misc]
            audio_done = await sender
            await asyncio.wait_for(receiver, timeout=self._timeout_s)
            return audio_done
        finally:
            for task in (sender, receiver):
                task.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)

    async def _send(self, ws: object, pcm: bytes, step: int) -> float:
        """Feed the clip chunk by chunk, then ``Terminate``; return send-end time."""
        started = time.monotonic()
        for i, offset in enumerate(range(0, len(pcm), step)):
            if self._realtime:
                delay = started + i * self._chunk_s - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            await ws.send(pcm[offset : offset + step])  # type: ignore[attr-defined]
        audio_done = time.monotonic()
        await ws.send(json.dumps({"type": "Terminate"}))  # type: ignore[attr-defined]
        return audio_done

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        return asyncio.run(self.atranscribe(audio, sample_rate))


async def _receive(ws: object, session: _Session) -> None:
    """Read messages until ``Termination``; the server closes right after it."""
    async for message in ws:  # type: ignore[attr-defined]
        if isinstance(message, bytes):
            continue
        body = json.loads(message)
        session.n_messages += 1
        kind = body.get("type")
        if kind == "Begin":
            session.begin = body
        elif kind == "Turn":
            order = int(body.get("turn_order", 0))
            transcript = str(body.get("transcript", ""))
            if body.get("end_of_turn"):
                session.finals[order] = transcript
                session.last_final_at = time.monotonic()
            else:
                session.partials[order] = transcript
        elif kind == "Termination":
            session.termination = body
            return
        elif kind == "Error":
            raise RuntimeError(f"AssemblyAI streaming error: {body.get('error')!r}")


def _check_pin(model_id: str, begin: dict[str, object]) -> None:
    """Fail if the session says it is serving a model other than the pinned one.

    The v3 ``Begin`` message names the served model in ``configuration.model``
    (undocumented; observed 2026-10-02). A mismatch is fatal — the number would
    be published under the wrong name — and so is a ``Begin`` that stops naming
    it: the session would then be unattested, which is how a silent fallback to
    the default model would look.
    """
    configuration = begin.get("configuration")
    served = configuration.get("model") if isinstance(configuration, dict) else None
    if served != model_id:
        raise RuntimeError(
            f"AssemblyAI served model={served!r} but the pin is {model_id!r}"
        )
