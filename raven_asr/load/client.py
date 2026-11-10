"""One production-shaped request: the upload, its read budget, its outcome.

What production does per block (flow.raven ``api/app/providers/_parakeet.py``
and ``base.stt_answer_seconds``, master 2026-11-09), and what this client does:

* the read timeout is the block's answer budget, ``min(300, 60 + 1.0 × audio_s)``
  — a request still unanswered then is a **timeout**;
* every request asks for word timestamps — here as OpenAI ``verbose_json`` with
  ``timestamp_granularities[]=word`` where the server can answer it, else plain
  ``json`` (:func:`probe_timestamps` decides once per server and records why);
* a failed block is retried once (#7205) in the accuracy run; the capacity
  sweep never retries, so a step's errors are first-attempt errors.

The audio is the block's 16 kHz mono PCM16 WAV (raven_asr.load.workload). Word
timestamps are read with the adapter's parser
(:func:`raven_asr.adapters.base.parse_word_timestamps`), the same code the
leaderboard harness keeps them with.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import httpx

from ..adapters.base import parse_word_timestamps

#: flow.raven base.stt_answer_seconds: 60 s floor plus real time, capped at 300 s.
BUDGET_FLOOR_S = 60.0
BUDGET_PER_AUDIO_S = 1.0
BUDGET_CAP_S = 300.0

ASK_WORDS = "verbose_json+word"
ASK_JSON = "json"


def answer_budget_s(audio_s: float) -> float:
    """The read timeout production gives a request carrying ``audio_s`` of audio."""
    return min(BUDGET_CAP_S, BUDGET_FLOOR_S + BUDGET_PER_AUDIO_S * audio_s)


def form_fields(model: str, ask: str) -> dict[str, str | list[str]]:
    data: dict[str, str | list[str]] = {"model": model, "language": "de"}
    if ask == ASK_WORDS:
        data["response_format"] = "verbose_json"
        data["timestamp_granularities[]"] = ["word"]
    else:
        data["response_format"] = "json"
    return data


@dataclass
class Outcome:
    """What one request did; ``status`` is ok | timeout | http_error | conn_error | aborted."""

    key: str
    audio_s: float
    budget_s: float
    t_start: float
    t_end: float | None = None
    status: str = "aborted"
    http_status: int | None = None
    error: str | None = None
    text: str | None = None
    words: list[dict] | None = None
    attempts: int = 0
    first_error: str | None = None
    server_wall_s: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def latency_s(self) -> float | None:
        return None if self.t_end is None else self.t_end - self.t_start

    def as_json(self) -> dict:
        d = asdict(self)
        d["latency_s"] = self.latency_s
        return d


async def post_once(
    client: httpx.AsyncClient, url: str, model: str, wav: bytes, audio_s: float, ask: str,
) -> tuple[str, int | None, str | None, dict | None]:
    """``(status, http_status, error, body)`` of one upload under the answer budget."""
    budget = answer_budget_s(audio_s)
    timeout = httpx.Timeout(connect=10.0, read=budget, write=60.0, pool=None)
    try:
        r = await client.post(url, files={"file": ("block.wav", wav, "audio/wav")},
                              data=form_fields(model, ask), timeout=timeout)
    except httpx.TimeoutException as exc:
        return "timeout", None, f"{type(exc).__name__}", None
    except httpx.TransportError as exc:
        return "conn_error", None, f"{type(exc).__name__}: {exc}"[:300], None
    if r.status_code != 200:
        return "http_error", r.status_code, r.text[:300], None
    try:
        return "ok", 200, None, r.json()
    except ValueError as exc:
        return "http_error", 200, f"non-JSON answer: {exc}"[:300], None


async def transcribe(
    client: httpx.AsyncClient, url: str, model: str, key: str, wav: bytes,
    audio_s: float, ask: str, *, retries: int = 0,
) -> Outcome:
    """Send one block; with ``retries`` = 1 a failed block is sent once more, as in production."""
    out = Outcome(key=key, audio_s=audio_s, budget_s=answer_budget_s(audio_s),
                  t_start=time.monotonic())
    for attempt in range(retries + 1):
        out.attempts = attempt + 1
        if attempt:
            out.t_start = time.monotonic()  # latency is the answering attempt's
        status, code, err, body = await post_once(client, url, model, wav, audio_s, ask)
        out.t_end = time.monotonic()
        out.status, out.http_status, out.error = status, code, err
        if status == "ok" and body is not None:
            out.text = str(body.get("text", "")).strip()
            out.words = parse_word_timestamps(body)
            xr = body.get("x_raven") if isinstance(body.get("x_raven"), dict) else {}
            out.server_wall_s = xr.get("wall_seconds") if xr else None
            if xr and xr.get("batch_size") is not None:
                out.extra["batch_size"] = xr["batch_size"]
            return out
        if attempt == 0:
            out.first_error = f"{status}: {err}"[:300]
    return out


def serving_identity(body: dict) -> dict:
    """The per-process part of an answer's ``x_raven``: which model, engine, recipe, setup."""
    xr = body.get("x_raven") if isinstance(body.get("x_raven"), dict) else {}
    return {k: xr.get(k) for k in ("model", "engine", "recipe", "batching", "decoding",
                                   "precision") if k in xr}


async def probe_timestamps(client: httpx.AsyncClient, url: str, model: str, wav: bytes,
                           audio_s: float) -> dict:
    """Ask once for word timestamps; record whether the server can answer them.

    Returns ``{"ask", "word_timestamps", "refusal"}``: ``ask`` is what every
    request of the run will send. A server that refuses ``verbose_json`` (4xx,
    or the shim's 502 wrapping the engine's 400) is asked plain ``json``; one
    that answers without words is recorded as answering without them.
    """
    status, code, err, body = await post_once(client, url, model, wav, audio_s, ASK_WORDS)
    if status == "ok" and body is not None:
        words = parse_word_timestamps(body)
        return {"ask": ASK_WORDS, "word_timestamps": bool(words), "refusal": None,
                "fields": sorted(body.keys()), "x_raven": serving_identity(body)}
    if status == "http_error":
        status2, code2, err2, body2 = await post_once(client, url, model, wav, audio_s, ASK_JSON)
        if status2 != "ok" or body2 is None:
            raise RuntimeError(f"the server answers neither verbose_json ({code}: {err}) "
                               f"nor json ({code2}: {err2})")
        words = parse_word_timestamps(body2)
        return {"ask": ASK_JSON, "word_timestamps": bool(words),
                "refusal": f"HTTP {code}: {err}", "fields": sorted(body2.keys()),
                "x_raven": serving_identity(body2)}
    raise RuntimeError(f"timestamp probe failed: {status} {err}")
