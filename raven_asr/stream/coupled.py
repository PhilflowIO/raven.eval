"""Coupled live design: a streaming model shows the live words, a batch model the final text.

    python -m raven_asr.stream.coupled --stream <stream-id> --final-url http://H:P/transcribe \\
        --hardware-label "<card, tenants>" --case "<what else the batch server is doing>" \\
        --out DIR [--steps 1,4,8,16] [--duration-s 300] [--endpoint-pause-s 0.7]

The question it answers: can the production batch STT take over the end of every
utterance while the streaming model carries the words as they are spoken? For N
concurrent speakers (one live session each, raven_asr.stream.session) every
utterance of the seeded pool is played in real time into the stream server,
ending at the **endpoint**: the speech end (:func:`pool.speech_offset_s`) plus
``--endpoint-pause-s`` of silence — the pause a live endpointer waits before it
calls the utterance over. The utterance's audio is cut there (padded with
silence where the recording ends earlier), so both models see exactly the same
audio, and at that moment — on the utterance's own real-time clock — the same
audio goes as one WAV upload to the batch endpoint (``--final-url``, the
``/transcribe`` of the production image, asked for timestamps like production).

Recorded per utterance (``utterances_n<N>.jsonl``), raw ``time.monotonic()``:
the stream record (raven_asr.stream.session) plus ``t_speech_end``, ``t_due``
(the endpoint), ``t_sent``, ``t_final_done``, the batch answer and its status.

Per step (``curve.json``, ``table.md``):
  final_after_speech_s     t_final_done - t_speech_end, the endpoint pause
                           included: when the final text stands, counted from
                           the moment the speaker stopped. The decision metric.
  final_request_s          t_final_done - t_sent: the batch server alone
  stream_after_speech_s    the stream model's transcription.done, same origin
  WER                      stream vs reference, batch vs reference, and stream
                           vs batch (the batch final as reference) on the
                           utterances both answered
  errors                   an empty answer counts as an error for both models:
                           a NeMo server that ran out of memory answers empty
                           (raven.eval#124), and the latency of an empty answer
                           is no latency at all

Without ``--final-url`` only the stream model runs (the same utterances, the
same endpoint), so its latency under other load can be compared with the
coupled runs.

``--pool-cache DIR``: the decoded pool as one PCM file plus an index, written
once by ``--build-pool-cache`` and read by every later run. Decoding the corpus
through ``datasets`` peaks at several GB of host RAM (4.4 GB for FLEURS German
test, measured on dave 2026-11-14) while the pool itself is ~0.3 GB; on a host
whose RAM is shared with production services that peak must not coincide with
the load the run measures.

``--hold-file F``: no step starts while F exists (an outside guard pauses the
load by creating it and appends ``<unix time> <reason>`` to ``F.log``), and a
step during which F existed is flagged ``held_during_step`` and repeated once.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from raven_eval_core.run_manifest import build_run_manifest
from raven_eval_core.wer import corpus_wer_strict_de_pct

from ..capacity import pcts
from ..load.workload import wav_header
from ..probes import Sampler, gpu_identity, threads
from .__main__ import (
    _stream_recipe,
    apply_health,
    server_engine,
    server_health,
    spoken_text,
)
from .pool import (
    DEFAULT_SEED,
    SAMPLE_RATE,
    Utterance,
    build_pool,
    session_playlist,
    speech_offset_s,
)
from .session import DONE_TIMEOUT_S, UtteranceRecord, run_session

SCHEMA = "raven.stream-coupled/1"
DEFAULT_STEPS = (1, 4, 8, 16)
DEFAULT_PAUSE_S = 0.7
FINAL_TIMEOUT_S = 60.0


def endpointed(utt: Utterance, pause_s: float) -> tuple[Utterance, float]:
    """``(utterance cut at speech end + pause, speech end s)``; padded with silence if short."""
    end_speech = speech_offset_s(utt.pcm16)
    n = round((end_speech + pause_s) * SAMPLE_RATE) * 2
    pcm = utt.pcm16[:n]
    if len(pcm) < n:
        pcm = pcm + b"\x00" * (n - len(pcm))
    return Utterance(utt.sample_id, utt.subset, utt.reference, pcm, utt.onset_s), end_speech


def endpointed_playlist(pool: list[Utterance], session: int, n: int, pause_s: float,
                        ends: dict[str, float]) -> Iterator[Utterance]:
    for utt in session_playlist(pool, session, n):
        cut, end = endpointed(utt, pause_s)
        ends[cut.sample_id] = end
        yield cut


POOL_CACHE_SCHEMA = "raven.stream-pool-cache/1"


def save_pool_cache(pool: list[Utterance], spec: dict, cache: Path) -> None:
    """``cache/pool.pcm`` (every utterance's PCM16, concatenated) + ``cache/pool.json``."""
    cache.mkdir(parents=True, exist_ok=True)
    rows, off = [], 0
    with (cache / "pool.pcm").open("wb") as fh:
        for u in pool:
            fh.write(u.pcm16)
            rows.append({"sample_id": u.sample_id, "subset": u.subset, "reference": u.reference,
                         "onset_s": u.onset_s, "offset": off, "bytes": len(u.pcm16)})
            off += len(u.pcm16)
    (cache / "pool.json").write_text(json.dumps(
        {"schema": POOL_CACHE_SCHEMA, "spec": spec, "rows": rows}, ensure_ascii=False))


def load_pool_cache(cache: Path) -> tuple[list[Utterance], dict]:
    meta = json.loads((cache / "pool.json").read_text())
    if meta.get("schema") != POOL_CACHE_SCHEMA:
        raise SystemExit(f"{cache}: not a pool cache ({meta.get('schema')})")
    pool = []
    with (cache / "pool.pcm").open("rb") as fh:
        for r in meta["rows"]:
            fh.seek(r["offset"])
            pool.append(Utterance(r["sample_id"], r["subset"], r["reference"],
                                  fh.read(r["bytes"]), r["onset_s"]))
    return pool, meta["spec"]


def wav_of(pcm16: bytes) -> bytes:
    return wav_header(len(pcm16) // 2) + pcm16


@dataclass
class FinalRecord:
    """The batch model's answer to one utterance."""

    session: int
    index: int
    sample_id: str
    audio_s: float
    t_speech_end: float
    t_due: float
    t_sent: float | None = None
    t_final_done: float | None = None
    status: str = "pending"  # ok | empty | http_error | timeout | conn_error | pending
    http_status: int | None = None
    text: str | None = None
    error: str | None = None

    @property
    def final_after_speech_s(self) -> float | None:
        return None if self.t_final_done is None else self.t_final_done - self.t_speech_end

    @property
    def final_request_s(self) -> float | None:
        if self.t_final_done is None or self.t_sent is None:
            return None
        return self.t_final_done - self.t_sent


async def post_final(client: httpx.AsyncClient, url: str, rec: FinalRecord, pcm16: bytes,
                     timeout_s: float, clock=time.monotonic) -> None:
    """Wait for the endpoint on the utterance's clock, then upload it like production does."""
    wait = rec.t_due - clock()
    if wait > 0:
        await asyncio.sleep(wait)
    rec.t_sent = clock()
    try:
        r = await client.post(url, files={"file": ("utterance.wav", wav_of(pcm16), "audio/wav")},
                              data={"include_timestamps": "true", "should_chunk": "true"},
                              timeout=httpx.Timeout(timeout_s, connect=10.0))
    except httpx.TimeoutException as exc:
        rec.t_final_done, rec.status, rec.error = clock(), "timeout", type(exc).__name__
        return
    except httpx.TransportError as exc:
        rec.t_final_done, rec.status = clock(), "conn_error"
        rec.error = f"{type(exc).__name__}: {exc}"[:300]
        return
    rec.t_final_done = clock()
    rec.http_status = r.status_code
    if r.status_code != 200:
        rec.status, rec.error = "http_error", r.text[:300]
        return
    try:
        rec.text = str(r.json().get("text", "")).strip()
    except ValueError as exc:
        rec.status, rec.error = "http_error", f"non-JSON answer: {exc}"[:300]
        return
    rec.status = "ok" if rec.text else "empty"


async def coupled_session(url: str, model: str, pool: list[Utterance], i: int, n: int, *,
                          chunk_ms: int, duration_s: float, start_delay_s: float,
                          done_timeout_s: float, pause_s: float, final_url: str | None,
                          client: httpx.AsyncClient | None, final_timeout_s: float):
    """One live speaker: the stream session, plus one batch upload per utterance at its endpoint."""
    ends: dict[str, float] = {}
    finals: list[FinalRecord] = []
    tasks: list[asyncio.Task] = []

    def on_utterance(utt: Utterance, rec: UtteranceRecord) -> None:
        end = ends[utt.sample_id]
        fr = FinalRecord(i, rec.index, utt.sample_id, utt.duration_s,
                         t_speech_end=rec.t_start + end, t_due=rec.t_start + utt.duration_s)
        finals.append(fr)
        if final_url is not None and client is not None:
            tasks.append(asyncio.create_task(
                post_final(client, final_url, fr, utt.pcm16, final_timeout_s)))

    recs, err = await run_session(
        url, model, endpointed_playlist(pool, i, n, pause_s, ends), session=i,
        chunk_ms=chunk_ms, duration_s=duration_s, start_delay_s=start_delay_s,
        done_timeout_s=done_timeout_s, on_utterance=on_utterance)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return recs, finals, err


async def run_step(url: str, model: str, pool: list[Utterance], n: int, *, chunk_ms: int,
                   duration_s: float, done_timeout_s: float, seed: int, pause_s: float,
                   final_url: str | None, final_timeout_s: float):
    from .__main__ import start_delays

    delays = start_delays(n, chunk_ms, seed)
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(limits=limits) as client:
        results = await asyncio.gather(*(
            coupled_session(url, model, pool, i, n, chunk_ms=chunk_ms, duration_s=duration_s,
                            start_delay_s=delays[i], done_timeout_s=done_timeout_s,
                            pause_s=pause_s, final_url=final_url, client=client,
                            final_timeout_s=final_timeout_s)
            for i in range(n)))
    recs = [r for rs, _, _ in results for r in rs]
    finals = [f for _, fs, _ in results for f in fs]
    errs = [e for _, _, e in results]
    return recs, finals, errs


def _wer(refs: list[str], hyps: list[str]) -> float | None:
    if not refs:
        return None
    try:
        return round(corpus_wer_strict_de_pct(refs, hyps), 3)
    except ValueError:
        return None


def coupled_summary(n: int, recs: list[UtteranceRecord], finals: list[FinalRecord],
                    session_errors: list[str | None], probes: dict, threshold_s: float,
                    spoken, coupled: bool) -> dict:
    """One step: the stream model's and the batch model's latencies, errors and WER."""
    by_key = {(f.session, f.index): f for f in finals}
    done = [r for r in recs if r.error is None and r.t_done is not None]
    stream_empty = [r for r in done if not spoken(r.text or "").strip()]
    stream_ok = [r for r in done if spoken(r.text or "").strip()]
    stream_errors = [r for r in recs if r.error is not None]

    def after_speech(r: UtteranceRecord) -> float | None:
        f = by_key.get((r.session, r.index))
        return None if f is None or r.t_done is None else r.t_done - f.t_speech_end

    out: dict = {
        "n": n,
        "utterances": len(recs),
        "drops": sum(1 for e in session_errors if e is not None),
        "stream": {
            "completed": len(stream_ok),
            "errors": len(stream_errors),
            "empty": len(stream_empty),
            "error_kinds": sorted({(r.error or "")[:120] for r in stream_errors}),
            "final_s": pcts([r.final_s for r in stream_ok if r.final_s is not None]),
            "after_speech_s": pcts([v for r in stream_ok if (v := after_speech(r)) is not None]),
            "first_text_after_speech_s": pcts([r.first_text_after_speech_s for r in stream_ok
                                               if r.first_text_after_speech_s is not None]),
            "send_lag_max_s": round(max((r.send_lag_s for r in recs), default=0.0), 4),
            "wer_strict_pct": _wer([r.reference for r in stream_ok],
                                   [spoken(r.text or "") for r in stream_ok]),
        },
        "probes": probes,
    }
    ok_stream = {(r.session, r.index): r for r in stream_ok}
    if coupled:
        sent = [f for f in finals if f.status != "pending"]
        ok = [f for f in sent if f.status == "ok"]
        bad = [f for f in sent if f.status != "ok"]
        refs = {(r.session, r.index): r.reference for r in recs}
        both = [(ok_stream[(f.session, f.index)], f) for f in ok
                if (f.session, f.index) in ok_stream]
        lag = [f.t_sent - f.t_due for f in sent if f.t_sent is not None]
        out["final"] = {
            "sent": len(sent),
            "completed": len(ok),
            "errors": len(bad),
            "error_kinds": sorted({f"{f.status}: {(f.error or '')[:100]}" for f in bad}),
            "after_speech_s": pcts([f.final_after_speech_s for f in ok]),
            "request_s": pcts([f.final_request_s for f in ok]),
            "after_speech_max_s": (round(max(f.final_after_speech_s for f in ok), 4)
                                   if ok else None),
            "send_lag_max_s": round(max(lag, default=0.0), 4),
            "wer_strict_pct": _wer([refs[(f.session, f.index)] for f in ok],
                                   [f.text or "" for f in ok]),
        }
        out["paired"] = {
            "utterances": len(both),
            "stream_vs_reference_wer_pct": _wer([r.reference for r, _ in both],
                                                [spoken(r.text or "") for r, _ in both]),
            "final_vs_reference_wer_pct": _wer([r.reference for r, _ in both],
                                               [f.text or "" for _, f in both]),
            "stream_vs_final_wer_pct": _wer([f.text or "" for _, f in both],
                                            [spoken(r.text or "") for r, _ in both]),
        }
        p95 = out["final"]["after_speech_s"]["p95"]
        out["within_threshold"] = bool(ok) and not bad and out["drops"] == 0 \
            and p95 is not None and p95 <= threshold_s
    else:
        p95 = out["stream"]["final_s"]["p95"]
        out["within_threshold"] = bool(stream_ok) and not stream_errors and not stream_empty \
            and out["drops"] == 0 and p95 is not None and p95 <= threshold_s
    return out


def _f(v, fmt: str = "{:.2f}") -> str:
    return "–" if v is None else fmt.format(v)


def render_table(curve: dict) -> str:
    coupled = curve["final_url"] is not None
    lines = [(f"### {curve['model']} live + {'batch final' if coupled else 'no batch final'} "
              f"— {curve['case']}"), "",
             (f"{curve['latency']}; endpoint = speech end + "
              f"{curve['params']['endpoint_pause_s']} s; {curve['hardware_label']}; "
              f"threshold p95 ≤ {curve['params']['threshold_s']} s "
              f"({'batch final' if coupled else 'stream final'})"), ""]
    if coupled:
        lines += [("| N | utt | final after speech p50 / p95 / p99 s | final request p50 / p95 s "
                   "| stream after speech p50 / p95 s | final err | stream err+empty | drops "
                   "| WER stream % | WER final % | WER stream vs final % | GPU util % "
                   "| VRAM max MiB |"),
                  "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s in curve["steps"]:
            fa, fr, sa = s["final"]["after_speech_s"], s["final"]["request_s"], \
                s["stream"]["after_speech_s"]
            lines.append(
                f"| {s['n']} | {s['utterances']} | {_f(fa['p50'])} / {_f(fa['p95'])} / "
                f"{_f(fa['p99'])} | {_f(fr['p50'])} / {_f(fr['p95'])} | {_f(sa['p50'])} / "
                f"{_f(sa['p95'])} | {s['final']['errors']} | "
                f"{s['stream']['errors'] + s['stream']['empty']} | {s['drops']} | "
                f"{_f(s['paired']['stream_vs_reference_wer_pct'])} | "
                f"{_f(s['paired']['final_vs_reference_wer_pct'])} | "
                f"{_f(s['paired']['stream_vs_final_wer_pct'])} | "
                f"{_f(s['probes'].get('util_mean_pct'), '{:.0f}')} | "
                f"{_f(s['probes'].get('memory_used_max_mib'), '{:.0f}')} |")
    else:
        lines += [("| N | utt | stream final p50 / p95 / p99 s | stream after speech p50 / p95 s "
                   "| errors+empty | drops | WER stream % | GPU util % | VRAM max MiB |"),
                  "|---:|---:|---|---|---:|---:|---:|---:|---:|"]
        for s in curve["steps"]:
            sf, sa = s["stream"]["final_s"], s["stream"]["after_speech_s"]
            lines.append(
                f"| {s['n']} | {s['utterances']} | {_f(sf['p50'])} / {_f(sf['p95'])} / "
                f"{_f(sf['p99'])} | {_f(sa['p50'])} / {_f(sa['p95'])} | "
                f"{s['stream']['errors'] + s['stream']['empty']} | {s['drops']} | "
                f"{_f(s['stream']['wer_strict_pct'])} | "
                f"{_f(s['probes'].get('util_mean_pct'), '{:.0f}')} | "
                f"{_f(s['probes'].get('memory_used_max_mib'), '{:.0f}')} |")
    lines += [f"\nN = {s['n']}: the stream server reported `{s['server_error']}`."
              for s in curve["steps"] if "server_error" in s]
    lines += [f"\nN = {s['n']}: the hold file appeared during the step (repeat {s['attempt']})."
              for s in curve["steps"] if s.get("held_during_step")]
    return "\n".join(lines) + "\n"


def wait_while_held(hold: Path | None) -> float:
    """Seconds waited until ``hold`` was gone."""
    if hold is None:
        return 0.0
    t = time.monotonic()
    while hold.exists():
        time.sleep(2.0)
    return round(time.monotonic() - t, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--stream", help="stream recipe id (serving/streaming.toml)")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    ap.add_argument("--http", default="http://127.0.0.1:8000")
    ap.add_argument("--final-url", help="the batch server's /transcribe; omit for stream only")
    ap.add_argument("--hardware-label")
    ap.add_argument("--case", help="what else runs, e.g. 'batch server idle'")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--steps", default=",".join(map(str, DEFAULT_STEPS)))
    ap.add_argument("--duration-s", type=float, default=300.0)
    ap.add_argument("--endpoint-pause-s", type=float, default=DEFAULT_PAUSE_S)
    ap.add_argument("--threshold-s", type=float, default=1.0)
    ap.add_argument("--done-timeout-s", type=float, default=DONE_TIMEOUT_S)
    ap.add_argument("--final-timeout-s", type=float, default=FINAL_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pause-s", type=float, default=10.0, help="idle time between steps")
    ap.add_argument("--pool", default="fleurs",
                    help="pool sources, 'dataset[:take],…' (default: FLEURS German test only)")
    ap.add_argument("--server-pid", type=int, help="the stream server's pid (CPU per step)")
    ap.add_argument("--hold-file", type=Path, help="no step runs while this file exists")
    ap.add_argument("--pool-cache", type=Path, help="read the decoded pool from this dir")
    ap.add_argument("--build-pool-cache", action="store_true",
                    help="decode --pool into --pool-cache and exit")
    args = ap.parse_args(argv)

    sources = tuple((d, int(t) if t else None) for d, _, t in
                    (x.partition(":") for x in args.pool.split(",") if x))
    if args.build_pool_cache:
        if args.pool_cache is None:
            ap.error("--build-pool-cache needs --pool-cache")
        pool, pool_spec = build_pool(sources=sources, seed=args.seed)
        save_pool_cache(pool, pool_spec, args.pool_cache)
        print(json.dumps(pool_spec))
        return 0
    if not (args.stream and args.hardware_label and args.case and args.out):
        ap.error("--stream, --hardware-label, --case and --out are required for a run")
    stream, base, identity, base_identity = _stream_recipe(args.stream)
    model = base["model"]
    engine = server_engine(args.http, stream["protocol"], model, identity)
    if args.pool_cache is not None:
        pool, pool_spec = load_pool_cache(args.pool_cache)
        if pool_spec["seed"] != args.seed:
            raise SystemExit(f"{args.pool_cache} holds seed {pool_spec['seed']}, not {args.seed}")
    else:
        pool, pool_spec = build_pool(sources=sources, seed=args.seed)
    spoken = spoken_text(model)
    steps_n = [int(x) for x in args.steps.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    params = {"steps": steps_n, "duration_s": args.duration_s,
              "endpoint_pause_s": args.endpoint_pause_s, "threshold_s": args.threshold_s,
              "done_timeout_s": args.done_timeout_s, "final_timeout_s": args.final_timeout_s,
              "pause_s": args.pause_s, "seed": args.seed, "pool": args.pool,
              "final_form": {"include_timestamps": "true", "should_chunk": "true"}}
    curve = {
        "schema": SCHEMA,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stream": identity, "base_recipe": base_identity, "model": model,
        "chunk_ms": stream["chunk_ms"], "latency": stream["latency"], "engine": engine,
        "final_url": args.final_url, "case": args.case, "hardware_label": args.hardware_label,
        "gpu": gpu_identity(), "pool": pool_spec, "params": params,
        "run_manifest": build_run_manifest(resolved_config=params,
                                           libraries=("websockets", "httpx"), gpu=None),
        "steps": [],
    }

    def write() -> None:
        (args.out / "curve.json").write_text(json.dumps(curve, indent=2, ensure_ascii=False) + "\n")

    kw = {"chunk_ms": stream["chunk_ms"], "done_timeout_s": args.done_timeout_s,
          "seed": args.seed, "pause_s": args.endpoint_pause_s, "final_url": args.final_url,
          "final_timeout_s": args.final_timeout_s}
    # One utterance before the curve, not counted: CUDA init and first-call costs.
    warm_recs, warm_finals, warm_err = asyncio.run(run_step(
        args.url, model, pool, 1, duration_s=0.001, **{**kw, "done_timeout_s": 120.0,
                                                        "final_timeout_s": 120.0}))
    if warm_err[0] is not None or not warm_recs or warm_recs[0].error is not None:
        raise SystemExit(f"warm-up failed: {warm_err[0] or warm_recs[0].error}")
    if args.final_url and (not warm_finals or warm_finals[0].status not in ("ok", "empty")):
        raise SystemExit(f"warm-up of the batch final failed: {warm_finals and warm_finals[0]}")
    curve["warmup"] = {"stream_final_s": warm_recs[0].final_s,
                       "final_request_s": warm_finals[0].final_request_s if warm_finals else None}

    processes = {"client": lambda: threads(os.getpid())}
    if args.server_pid:
        processes["server"] = lambda: threads(args.server_pid)
    for n in steps_n:
        for attempt in (1, 2):
            waited = wait_while_held(args.hold_file)
            print(f"[coupled] {args.case}: N = {n}, {args.duration_s:.0f} s (attempt {attempt})",
                  flush=True)
            t = time.monotonic()
            started = datetime.now(UTC).isoformat(timespec="seconds")
            with Sampler(None, processes=processes) as sampler:
                recs, finals, errs = asyncio.run(run_step(
                    args.url, model, pool, n, duration_s=args.duration_s, **kw))
            held = args.hold_file is not None and _held_since(args.hold_file, t)
            step = coupled_summary(n, recs, finals, errs, sampler.summary(), args.threshold_s,
                                   spoken, args.final_url is not None)
            step.update({"wall_s": round(time.monotonic() - t, 1), "started_utc": started,
                         "attempt": attempt, "held_wait_before_s": waited,
                         "held_during_step": held})
            apply_health(step, server_health(args.http, stream["protocol"]))
            by_key = {(f.session, f.index): f for f in finals}
            with (args.out / f"utterances_n{n}{'' if attempt == 1 else '_r2'}.jsonl").open(
                    "w", encoding="utf-8") as fh:
                for r in recs:
                    row = r.as_json()
                    f = by_key.get((r.session, r.index))
                    if f is not None:
                        row["final"] = {**asdict(f),
                                        "final_after_speech_s": f.final_after_speech_s,
                                        "final_request_s": f.final_request_s}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            curve["steps"].append(step)
            write()
            fin = step.get("final", {}).get("after_speech_s", {})
            print(f"[coupled]   final after speech p50/p95 {fin.get('p50')}/{fin.get('p95')} s; "
                  f"stream final p95 {step['stream']['final_s']['p95']} s; "
                  f"errors stream {step['stream']['errors']}+{step['stream']['empty']} empty, "
                  f"final {step.get('final', {}).get('errors')}; held {held}", flush=True)
            if not held:
                break
        time.sleep(args.pause_s)
    curve["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    write()
    (args.out / "table.md").write_text(render_table(curve), encoding="utf-8")
    print(render_table(curve))
    return 0


def _held_since(hold: Path, t_monotonic: float) -> bool:
    """Whether ``hold`` exists now or was created after ``t_monotonic`` (and removed again)."""
    if hold.exists():
        return True
    log = hold.with_suffix(hold.suffix + ".log")
    if not log.exists():
        return False
    since = time.time() - (time.monotonic() - t_monotonic)
    return any(float(line.split()[0]) >= since for line in log.read_text().splitlines()
               if line.strip() and line.split()[0].replace(".", "", 1).isdigit())


if __name__ == "__main__":
    sys.exit(main())

