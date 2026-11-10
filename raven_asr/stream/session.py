"""One live session: utterances played in real time over the realtime websocket.

The protocol is vLLM 0.33.0's ``/v1/realtime`` (serving/nemo/stream_server.py
speaks it too). Per utterance the session commits, sends the audio in chunks of
``chunk_ms`` on an absolute real-time schedule — chunk ``k`` covers
``[k*c, (k+1)*c)`` of the utterance and leaves when that much audio has been
"spoken", at ``t0 + (k+1)*c`` — then commits ``final`` and waits for
``transcription.done`` before starting the next utterance (vLLM runs one
utterance per connection at a time). The session ends with the first
utterance that starts after ``duration_s``.

Times are ``time.monotonic()`` seconds, recorded raw per utterance so any other
latency definition can be computed later:

  t_start       the utterance's audio began (t0; the first chunk leaves at t0 + c)
  t_first_send  the first chunk actually left
  t_audio_end   the last chunk actually left (the end of the audio, live)
  t_first_text  the first event that carried text (delta or partial)
  t_done        transcription.done arrived

  ttfp_s        t_first_text - t_start: time to first partial, counted from the
                start of the utterance's audio — leading silence included
  first_text_after_speech_s
                t_first_text - (t_start + speech_onset_s): time until the first
                words appear, counted from when speech starts in the audio
                (pool.speech_onset_s); what a live-subtitle viewer waits for
  final_s       t_done - t_audio_end: final latency
  send_lag_s    the worst delay of a chunk behind its schedule; a large one means
                the CLIENT fell behind, and the step's latencies are suspect
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from .pool import SAMPLE_RATE, Utterance

DONE_TIMEOUT_S = 30.0
CONNECT_TIMEOUT_S = 30.0


@dataclass
class UtteranceRecord:
    session: int
    index: int
    sample_id: str
    subset: str
    reference: str
    duration_s: float
    t_start: float
    t_first_send: float | None = None
    t_audio_end: float | None = None
    t_first_text: float | None = None
    t_done: float | None = None
    text: str | None = None
    send_lag_s: float = 0.0
    error: str | None = None
    speech_onset_s: float | None = None

    @property
    def ttfp_s(self) -> float | None:
        return None if self.t_first_text is None else self.t_first_text - self.t_start

    @property
    def first_text_after_speech_s(self) -> float | None:
        if self.t_first_text is None or self.speech_onset_s is None:
            return None
        return self.t_first_text - (self.t_start + self.speech_onset_s)

    @property
    def final_s(self) -> float | None:
        if self.t_done is None or self.t_audio_end is None:
            return None
        return self.t_done - self.t_audio_end

    def as_json(self) -> dict:
        d = asdict(self)
        d["ttfp_s"] = self.ttfp_s
        d["first_text_after_speech_s"] = self.first_text_after_speech_s
        d["final_s"] = self.final_s
        return d


async def run_session(
    url: str,
    model: str,
    playlist: Iterator[Utterance],
    *,
    session: int,
    chunk_ms: int,
    duration_s: float,
    start_delay_s: float = 0.0,
    done_timeout_s: float = DONE_TIMEOUT_S,
    clock=time.monotonic,
) -> tuple[list[UtteranceRecord], str | None]:
    """Play utterances until ``duration_s`` of session time has passed.

    Returns ``(records, session_error)``. A session error (the connection could
    not be opened or broke) ends the session: that is a drop.
    """
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    records: list[UtteranceRecord] = []
    await asyncio.sleep(start_delay_s)
    chunk_bytes = int(SAMPLE_RATE * chunk_ms / 1000) * 2
    chunk_s = chunk_ms / 1000
    try:
        async with asyncio.timeout(CONNECT_TIMEOUT_S):
            ws = await connect(url, max_size=None, open_timeout=CONNECT_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - any failure to connect is a drop
        return records, f"connect: {type(exc).__name__}: {exc}"
    try:
        created = json.loads(await asyncio.wait_for(ws.recv(), CONNECT_TIMEOUT_S))
        if created.get("type") != "session.created":
            return records, f"expected session.created, got {created.get('type')}"
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        session_end = clock() + duration_s  # session time starts once the server is ready
        index = 0
        while clock() < session_end:
            utt = next(playlist)
            rec = UtteranceRecord(session, index, utt.sample_id, utt.subset, utt.reference,
                                  utt.duration_s, t_start=0.0, speech_onset_s=utt.onset_s)
            index += 1
            records.append(rec)
            await _play(ws, utt, rec, chunk_bytes, chunk_s, done_timeout_s, clock)
            if rec.error is not None:
                return records, rec.error
    except ConnectionClosed as exc:
        if records and records[-1].error is None and records[-1].t_done is None:
            records[-1].error = f"connection closed: {exc}"
        return records, f"connection closed: {exc}"
    finally:
        await ws.close()
    return records, None


async def _play(ws, utt: Utterance, rec: UtteranceRecord, chunk_bytes: int, chunk_s: float,
                done_timeout_s: float, clock) -> None:
    done = asyncio.Event()

    async def reader() -> None:
        async for raw in ws:
            ev = json.loads(raw)
            kind = ev.get("type")
            now = clock()
            if kind in ("transcription.delta", "transcription.partial"):
                if rec.t_first_text is None and (ev.get("delta") or ev.get("text")):
                    rec.t_first_text = now
            elif kind == "transcription.done":
                rec.t_done = now
                rec.text = ev.get("text") or ""
                if rec.t_first_text is None and rec.text:
                    rec.t_first_text = now
                done.set()
                return
            elif kind == "error":
                rec.error = f"server error: {ev.get('code')}: {ev.get('error')}"
                done.set()
                return

    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    task = asyncio.create_task(reader())
    try:
        pcm = utt.pcm16
        t0 = rec.t_start = clock()
        n_chunks = max(1, -(-len(pcm) // chunk_bytes))
        for k in range(n_chunks):
            due = t0 + (k + 1) * chunk_s if k < n_chunks - 1 else t0 + utt.duration_s
            wait = due - clock()
            if wait > 0:
                await asyncio.sleep(wait)
            now = clock()
            rec.send_lag_s = max(rec.send_lag_s, now - due)
            piece = pcm[k * chunk_bytes:(k + 1) * chunk_bytes]
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(piece).decode()}))
            if k == 0:
                rec.t_first_send = clock()
            if rec.error is not None:
                return
        rec.t_audio_end = clock()
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        try:
            await asyncio.wait_for(done.wait(), done_timeout_s)
        except TimeoutError:
            rec.error = f"no transcription.done within {done_timeout_s:.0f} s"
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110 - reader is done either way
                pass
        elif task.exception() is not None and rec.error is None:
            rec.error = f"reader: {task.exception()}"
