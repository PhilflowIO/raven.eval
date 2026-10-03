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

Every utterance is one session, and AssemblyAI limits how many **new** sessions
an account may open per minute — not how many are open at once (free 5, paid
100+, growing 10 % per minute while at least 70 % used; account-wide, not per
key; docs/streaming/rate-limits, read 2026-10-03). An excess is refused with
"Too many concurrent sessions", despite the name. Sessions are therefore opened
through a shared :class:`SessionRate`, and a refusal waits for the next slot
instead of spending the retry budget meant for broken connections.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from urllib.parse import urlencode

import numpy as np

from ..retry import TransientStreamError, with_retry
from .base import TranscribeResult

DEFAULT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
DEFAULT_MODEL = "universal-3-6-pro"
DEFAULT_CHUNK_S = 0.1
DEFAULT_TIMEOUT_S = 60.0
# The documented starting limit of a free account: safe on any account. A paid
# account starts at 100+; set ASSEMBLYAI_SESSIONS_PER_MIN to its dashboard value
# rather than waiting for the rate to grow there.
DEFAULT_SESSIONS_PER_MIN = 5.0
SESSIONS_PER_MIN_ENV = "ASSEMBLYAI_SESSIONS_PER_MIN"
# How long one utterance may keep being refused before it is recorded as failed:
# long enough to ride out a burst, short enough that a dead account is noticed.
DEFAULT_REFUSAL_PATIENCE_S = 600.0

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


class SessionRefused(Exception):
    """The account's new-sessions-per-minute limit refused a session.

    Deliberately not a :class:`TransientStreamError`: the generic retry would
    spend its short exponential budget on it. Capacity is waited for in
    :meth:`AssemblyAIRealtimeAdapter.atranscribe`, paced by :class:`SessionRate`.
    """


class SessionRate:
    """Opens sessions no faster than a per-minute rate, shared by all utterances.

    A sliding window of the last ``window_s`` seconds of openings, mirroring the
    vendor's own accounting. The rate follows the account the way the vendor
    documents its limit moving: a refusal halves it (the account allows less
    than assumed), and a full window used to at least 70 % without a refusal
    grows it by 10 %, as the vendor's auto-scaling does.

    No asyncio primitives: check-and-record has no ``await`` between them, so
    concurrent utterances on one event loop cannot both take the last slot.
    """

    def __init__(
        self,
        per_min: float,
        *,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if per_min < 1:
            raise ValueError(f"per_min must be at least 1, got {per_min}")
        self.per_min = per_min
        self._window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._opened: deque[float] = deque()
        self._window_start = clock()
        self._opened_in_window = 0
        self._refused_in_window = False

    async def acquire(self) -> None:
        """Wait for a free slot, then count one session as opened now."""
        while True:
            now = self._clock()
            while self._opened and now - self._opened[0] >= self._window_s:
                self._opened.popleft()
            self._adapt(now)
            if len(self._opened) < int(self.per_min):
                self._opened.append(now)
                self._opened_in_window += 1
                return
            await self._sleep(self._opened[0] + self._window_s - now)

    def refused(self) -> None:
        """The vendor refused a session: assume the limit is lower than the rate.

        Halved once per window: a burst refuses every utterance in flight at
        the same moment, and that is one piece of evidence, not one per refusal.
        """
        if not self._refused_in_window:
            self.per_min = max(1.0, self.per_min / 2)
        self._refused_in_window = True

    def _adapt(self, now: float) -> None:
        if now - self._window_start < self._window_s:
            return
        busy = self._opened_in_window >= 0.7 * int(self.per_min)
        if busy and not self._refused_in_window:
            self.per_min *= 1.1
        self._window_start = now
        self._opened_in_window = 0
        self._refused_in_window = False


def _initial_rate() -> float:
    value = os.environ.get(SESSIONS_PER_MIN_ENV)
    return float(value) if value else DEFAULT_SESSIONS_PER_MIN


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
        session_rate: SessionRate | None = None,
        refusal_patience_s: float = DEFAULT_REFUSAL_PATIENCE_S,
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
        # One per adapter, and the runner builds one adapter per model, so every
        # in-flight utterance of a run draws from the same per-minute budget.
        self.session_rate = session_rate or SessionRate(_initial_rate())
        self._refusal_patience_s = refusal_patience_s

    def _url(self, sample_rate: int) -> str:
        params = {
            "speech_model": self.model_id,
            "sample_rate": str(sample_rate),
            "encoding": "pcm_s16le",
            "language_codes": json.dumps([self._language]),
        }
        return f"{self._base_url}?{urlencode(params)}"

    async def atranscribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> TranscribeResult:
        """Transcribe one utterance, waiting out refusals by the session limit."""
        started = time.monotonic()
        n_refused = 0
        while True:
            try:
                result = await self._session(audio, sample_rate)
            except SessionRefused as exc:
                self.session_rate.refused()
                n_refused += 1
                if time.monotonic() - started > self._refusal_patience_s:
                    raise RuntimeError(
                        f"AssemblyAI kept refusing new sessions for "
                        f"{self._refusal_patience_s:.0f}s ({n_refused} refusals); "
                        f"the account's session limit is not available: {exc}"
                    ) from exc
                continue
            result.raw["n_refused_sessions"] = n_refused
            return result

    @with_retry()
    async def _session(
        self, audio: np.ndarray, sample_rate: int
    ) -> TranscribeResult:
        """One streaming session; transport failures are retried, refusals raised."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        pcm = _pcm16(audio)
        step = round(self._chunk_s * sample_rate) * 2  # bytes per chunk
        session = _Session()
        # Inside the retried function: a retry opens a new session too.
        await self.session_rate.acquire()
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
            if code == _CLOSE_SESSION_LIMIT or _SESSION_LIMIT in reason:
                raise SessionRefused(f"closed {code} {reason!r}") from exc
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


# How AssemblyAI refuses a session over the new-sessions-per-minute limit: this
# text in an ``Error`` frame (observed 2026-10-02) and close code 3009 (documented;
# the rate-limits page says 1008, so the text is matched on any close as well).
_SESSION_LIMIT = "Too many concurrent sessions"
_CLOSE_SESSION_LIMIT = 3009


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
            error = str(body.get("error", ""))
            if _SESSION_LIMIT in error:
                raise SessionRefused(f"session limit: {error!r}")
            raise RuntimeError(f"AssemblyAI streaming error: {error!r}")


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
