"""Live-streaming capacity: how many simultaneous live sessions one server sustains.

    python -m raven_asr.stream --stream <stream-id> --hardware-label "<card, tenants>" \\
        --out results/streaming/<date>-<stream-id> [--offline-predictions <dir>]

Start the server first (``python serving/launch.py <stream-id> --streaming``) on
this host. For N = 1, 2, 4, …, 128 the harness opens N sessions at once, each
playing utterances of the seeded pool in real time for ``--duration-s``
(raven_asr.stream.session), and records one step of the curve
(raven_asr.stream.curve). It stops one step after the first step whose p95
final latency exceeds ``--threshold-s`` or that saw an error or a drop.

Written to ``--out`` after every step (an interrupted run keeps its steps):
  curve.json             the curve: identity, pool, parameters, every step
  utterances_n<N>.jsonl  every utterance of step N with its raw timestamps and text
  table.md               the short table (at the end)

``--rescore --out <dir> [--offline-predictions <dir>]`` recomputes only the WER
of a finished curve from its utterance files, latencies untouched.

The client is asyncio: one process runs every session on one event loop, one
core. At hundreds of sessions that loop can fall behind (it base64-encodes and
JSON-frames every chunk of every session), and a late chunk or a late read
shows up as server latency. ``--client-procs K`` splits a step's sessions
round-robin over K worker processes, forked once at startup, each with its own
loop; the schedule (start offsets, playlists) is the same for any K. Every step
records the CPU of the client processes (and of the server with
``--server-pid``) in % of one core, per thread: a busiest client thread near
100 % means the step measured the harness, not the server.

After every step the harness asks the server's ``/health``. A NeMo stream
server whose pipeline step failed (out of memory, say) ends the affected
utterances with an ordinary ``transcription.done`` — the transcript is empty,
the latency looks excellent — and reports the failure only there; such a step
breaks the rule, with the server's error on record, however good it looks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from raven_eval_core.run_manifest import build_run_manifest

from ..capacity import max_n_within, next_action
from ..config import OUTPUT_FORMATS
from ..output_formats import parser_for
from ..probes import Sampler, gpu_identity, threads
from .curve import (
    DEFAULT_STEPS,
    DEFAULT_THRESHOLD_S,
    load_offline,
    render_table,
    step_summary,
)
from .pool import DEFAULT_SEED, build_pool, session_playlist
from .session import DONE_TIMEOUT_S, UtteranceRecord, run_session

SCHEMA = "raven.stream-capacity/1"
_SERVING = Path(__file__).resolve().parents[2] / "serving"


def _stream_recipe(sid: str) -> tuple[dict, dict, dict, dict]:
    """``(stream, base recipe, stream identity, base identity)`` from this checkout."""
    sys.path.insert(0, str(_SERVING))
    import recipe
    import streaming

    recipes = recipe.load_recipes()
    streams, skipped = streaming.load_streams(recipes=recipes)
    if sid in skipped:
        raise SystemExit(f"{sid} is skipped: {skipped[sid]['reason']}")
    sid = streaming.resolve_stream(sid, streams, recipes)
    s = streams[sid]
    return (s, recipes[s["base"]], streaming.stream_identity(sid),
            {"id": s["base"], "sha256": recipe.recipe_sha256(s["base"], recipes)})


def server_engine(http: str, protocol: str, model: str, identity: dict) -> dict:
    """What the server reports about itself; aborts when it serves something else."""
    with httpx.Client(timeout=10.0) as c:
        if protocol in ("nemo-pipeline", "moonshine-voice"):
            info = c.get(f"{http}/v1/stream/info").json()
            if info.get("stream") != identity:
                raise SystemExit(f"the server serves {info.get('stream')}, this checkout "
                                 f"hashes the recipe as {identity}")
            return {**{k: v for k, v in info.items() if k != "stream"},
                    "name": info.get("engine", {}).get("name", "nemo")}
        models = [m["id"] for m in c.get(f"{http}/v1/models").json().get("data", [])]
        if model not in models:
            raise SystemExit(f"the server serves {models}, not {model}")
        return {"name": "vllm", "version": c.get(f"{http}/version").json().get("version")}


def server_health(http: str, protocol: str) -> dict | None:
    """The server's ``/health`` after a step (our own servers; vLLM has no error field)."""
    if protocol not in ("nemo-pipeline", "moonshine-voice"):
        return None
    try:
        with httpx.Client(timeout=10.0) as c:
            return c.get(f"{http}/health").json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"ok": False, "error": f"health check failed: {type(exc).__name__}: {exc}"}


def apply_health(step: dict, health: dict | None) -> dict:
    """A step during which the server reported a failure breaks the rule."""
    if health is not None and not health.get("ok", True):
        step["server_error"] = str(health.get("error") or "server unhealthy")[:300]
        step["within_threshold"] = False
    return step


def spoken_text(model: str):
    """``raw -> spoken text`` for ``model``, the offline scorer's output parser."""
    parser = parser_for(OUTPUT_FORMATS.get(model))
    return (lambda raw: raw) if parser is None else (lambda raw: parser(raw).text)


def rescore(out: Path, offline_dir: Path | None) -> int:
    """Recompute every step's WER of a finished curve from its utterance files.

    Latencies, errors and probes stay as measured; only the scoring is redone,
    with the model's output parser and, if given, the offline predictions.
    """
    curve = json.loads((out / "curve.json").read_text())
    spoken = spoken_text(curve["model"])
    offline = load_offline(offline_dir)
    fields = set(UtteranceRecord.__dataclass_fields__)
    for i, old in enumerate(curve["steps"]):
        rows = [json.loads(line) for line in
                (out / f"utterances_n{old['n']}.jsonl").read_text().splitlines() if line]
        recs = [UtteranceRecord(**{k: v for k, v in r.items() if k in fields}) for r in rows]
        new = step_summary(old["n"], recs, [], old["probes"],
                           offline, curve["params"]["threshold_s"], spoken)
        old["wer_strict_pct"] = new["wer_strict_pct"]
        old["offline_wer_strict_pct"] = new["offline_wer_strict_pct"]
        curve["steps"][i] = old
    curve["output_format"] = OUTPUT_FORMATS.get(curve["model"])
    curve["offline_predictions"] = bool(offline)
    (out / "curve.json").write_text(json.dumps(curve, indent=2, ensure_ascii=False) + "\n")
    (out / "table.md").write_text(render_table(curve), encoding="utf-8")
    print(render_table(curve))
    return 0


def start_delays(n: int, chunk_ms: int, seed: int) -> list[float]:
    """Each session's start offset, uniform within one chunk, seeded per step."""
    rng = random.Random(seed * 1000 + n)
    return [rng.uniform(0, chunk_ms / 1000) for _ in range(n)]


async def _sessions(url: str, model: str, pool: list, n: int, sessions: list[int],
                    delays: list[float], *, chunk_ms: int, duration_s: float,
                    done_timeout_s: float) -> list:
    return await asyncio.gather(*(
        run_session(url, model, session_playlist(pool, i, n), session=i, chunk_ms=chunk_ms,
                    duration_s=duration_s, start_delay_s=delays[i],
                    done_timeout_s=done_timeout_s)
        for i in sessions
    ))


def _flatten(results: list) -> tuple[list, list]:
    return [r for recs, _ in results for r in recs], [err for _, err in results]


async def run_step(url: str, model: str, pool: list, n: int, *, chunk_ms: int,
                   duration_s: float, done_timeout_s: float, seed: int):
    """One step on this process's event loop."""
    return _flatten(await _sessions(
        url, model, pool, n, list(range(n)), start_delays(n, chunk_ms, seed),
        chunk_ms=chunk_ms, duration_s=duration_s, done_timeout_s=done_timeout_s))


#: The pool, set before the client workers are forked: they inherit it instead
#: of receiving ~1 GB of PCM through a pipe.
_POOL: list = []


def _worker(job: tuple) -> list:
    url, model, n, sessions, delays, kw = job
    return asyncio.run(_sessions(url, model, _POOL, n, sessions, delays, **kw))


class ClientPool:
    """``procs`` forked client processes, each running its share of a step's sessions."""

    def __init__(self, pool: list, procs: int) -> None:
        global _POOL
        _POOL = pool
        self.procs = procs
        # fork: the workers inherit the decoded pool. Created before the
        # sampler thread exists, and the pool never replaces a worker.
        self._mp = multiprocessing.get_context("fork").Pool(procs)

    def pids(self) -> list[int]:
        return [os.getpid(), *(p.pid for p in self._mp._pool)]

    def tids(self) -> list[str]:
        return [t for pid in self.pids() for t in threads(pid)]

    def run_step(self, url: str, model: str, n: int, *, chunk_ms: int, duration_s: float,
                 done_timeout_s: float, seed: int):
        delays = start_delays(n, chunk_ms, seed)
        kw = {"chunk_ms": chunk_ms, "duration_s": duration_s, "done_timeout_s": done_timeout_s}
        jobs = [(url, model, n, list(range(k, n, self.procs)), delays, kw)
                for k in range(min(self.procs, n))]
        by_session: dict[int, tuple] = {}
        for job, part in zip(jobs, self._mp.map(_worker, jobs, chunksize=1), strict=True):
            by_session.update(zip(job[3], part, strict=True))
        return _flatten([by_session[i] for i in range(n)])

    def close(self) -> None:
        self._mp.close()
        self._mp.join()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--stream", help="stream recipe id (serving/streaming.toml)")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    ap.add_argument("--http", default="http://127.0.0.1:8000")
    ap.add_argument("--hardware-label",
                    help="the card and what else runs on it, e.g. '1x RTX 3090 24 GB, "
                         "shared with a 1.4 GB embedding server'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", default=",".join(map(str, DEFAULT_STEPS)))
    ap.add_argument("--duration-s", type=float, default=180.0)
    ap.add_argument("--threshold-s", type=float, default=DEFAULT_THRESHOLD_S)
    ap.add_argument("--done-timeout-s", type=float, default=DONE_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pause-s", type=float, default=10.0, help="idle time between steps")
    ap.add_argument("--offline-predictions", type=Path,
                    help="the model's offline run dir (predictions_*.jsonl) for the WER pair")
    ap.add_argument("--client-procs", type=int, default=1,
                    help="client processes the sessions of a step are split over (default 1)")
    ap.add_argument("--server-pid", type=int,
                    help="the server's pid, to record its CPU per step (same host)")
    ap.add_argument("--rescore", action="store_true",
                    help="only recompute the WER of the finished curve in --out")
    args = ap.parse_args(argv)
    if args.rescore:
        return rescore(args.out, args.offline_predictions)
    if not args.stream or not args.hardware_label:
        ap.error("--stream and --hardware-label are required for a run")
    if args.client_procs < 1:
        ap.error("--client-procs must be >= 1")

    stream, base, identity, base_identity = _stream_recipe(args.stream)
    model = base["model"]
    engine = server_engine(args.http, stream["protocol"], model, identity)
    steps_n = [int(x) for x in args.steps.split(",")]
    offline = load_offline(args.offline_predictions)
    pool, pool_spec = build_pool(seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    params = {"steps": steps_n, "duration_s": args.duration_s, "threshold_s": args.threshold_s,
              "done_timeout_s": args.done_timeout_s, "pause_s": args.pause_s,
              "start_stagger": "uniform within one chunk, seeded", "seed": args.seed,
              "client_procs": args.client_procs}
    curve = {
        "schema": SCHEMA,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stream": identity,
        "base_recipe": base_identity,
        "model": model,
        "revision": base["revision"],
        "protocol": stream["protocol"],
        "chunk_ms": stream["chunk_ms"],
        "latency": stream["latency"],
        # The stream recipe's own precision where it has one (a vendor build),
        # else the base recipe's; and the device it runs on.
        "precision": stream.get("precision", base["precision"]),
        "device": stream.get("device", "cuda"),
        "engine": engine,
        "hardware_label": args.hardware_label,
        "gpu": gpu_identity(),
        "pool": pool_spec,
        "output_format": OUTPUT_FORMATS.get(model),
        "offline_predictions": bool(offline),
        "params": params,
        "run_manifest": build_run_manifest(resolved_config=params, libraries=("websockets",),
                                           gpu=None),
        "steps": [],
        "max_n_within_threshold": None,
        "stopped_because": "running",
    }
    metrics = f"{args.http}/metrics" if stream["protocol"] == "vllm-realtime" else None
    stats = f"{args.http}/v1/stream/stats" if stream["protocol"] == "moonshine-voice" else None

    def write() -> None:
        (args.out / "curve.json").write_text(json.dumps(curve, indent=2, ensure_ascii=False) + "\n")

    # One utterance before the curve, not counted: the first request pays for
    # CUDA initialization and kernel compilation, which no live session would.
    warm, warm_err = asyncio.run(run_step(
        args.url, model, pool, 1, chunk_ms=stream["chunk_ms"], duration_s=0.001,
        done_timeout_s=max(args.done_timeout_s, 120.0), seed=args.seed))
    if warm_err[0] is not None or not warm or warm[0].error is not None:
        raise SystemExit(f"warm-up failed: {warm_err[0] or warm[0].error}")
    curve["warmup"] = {"sample_id": warm[0].sample_id, "final_s": warm[0].final_s}

    clients = ClientPool(pool, args.client_procs) if args.client_procs > 1 else None
    # per thread: the busiest one is the event loop (or the stepper) that capped the step
    processes = {"client": clients.tids if clients else (lambda: threads(os.getpid()))}
    if args.server_pid:
        processes["server"] = lambda: threads(args.server_pid)
    for n in steps_n:
        print(f"[stream] {identity['id']}: N = {n}, {args.duration_s:.0f} s", flush=True)
        t = time.monotonic()
        with Sampler(metrics, stats_url=stats, processes=processes) as sampler:
            if clients is None:
                records, session_errors = asyncio.run(run_step(
                    args.url, model, pool, n, chunk_ms=stream["chunk_ms"],
                    duration_s=args.duration_s, done_timeout_s=args.done_timeout_s,
                    seed=args.seed))
            else:
                records, session_errors = clients.run_step(
                    args.url, model, n, chunk_ms=stream["chunk_ms"],
                    duration_s=args.duration_s, done_timeout_s=args.done_timeout_s,
                    seed=args.seed)
        step = step_summary(n, records, session_errors, sampler.summary(), offline,
                            args.threshold_s, spoken_text(model))
        step["wall_s"] = round(time.monotonic() - t, 1)
        apply_health(step, server_health(args.http, stream["protocol"]))
        curve["steps"].append(step)
        with (args.out / f"utterances_n{n}.jsonl").open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.as_json(), ensure_ascii=False) + "\n")
        curve["max_n_within_threshold"] = max_n_within(curve["steps"])
        write()
        print(f"[stream]   p50/p95 final {step['final_s']['p50']}/{step['final_s']['p95']} s, "
              f"errors {step['errors']}, drops {step['drops']}, WER {step['wer_strict_pct']}"
              + (f", SERVER ERROR {step['server_error']}" if "server_error" in step else ""),
              flush=True)
        action = next_action(curve["steps"])
        if action == "stop":
            curve["stopped_because"] = "one step recorded beyond the first step over the rule"
            break
        time.sleep(args.pause_s)
    else:
        curve["stopped_because"] = (
            "every step stepped" if next_action(curve["steps"]) == "continue"
            else "the last step broke the rule; no larger step configured")
    if clients is not None:
        clients.close()
    curve["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    write()
    (args.out / "table.md").write_text(render_table(curve), encoding="utf-8")
    print(render_table(curve))
    return 0


if __name__ == "__main__":
    sys.exit(main())
