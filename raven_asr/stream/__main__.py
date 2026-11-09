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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from raven_eval_core.run_manifest import build_run_manifest

from .curve import (
    DEFAULT_STEPS,
    DEFAULT_THRESHOLD_S,
    load_offline,
    max_n_within,
    next_action,
    render_table,
    step_summary,
)
from .pool import DEFAULT_SEED, build_pool, session_playlist
from .probes import Sampler, gpu_identity
from .session import DONE_TIMEOUT_S, run_session

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
        if protocol == "nemo-pipeline":
            info = c.get(f"{http}/v1/stream/info").json()
            if info.get("stream") != identity:
                raise SystemExit(f"the server serves {info.get('stream')}, this checkout "
                                 f"hashes the recipe as {identity}")
            return {"name": "nemo", **{k: v for k, v in info.items() if k != "stream"}}
        models = [m["id"] for m in c.get(f"{http}/v1/models").json().get("data", [])]
        if model not in models:
            raise SystemExit(f"the server serves {models}, not {model}")
        return {"name": "vllm", "version": c.get(f"{http}/version").json().get("version")}


async def run_step(url: str, model: str, pool: list, n: int, *, chunk_ms: int,
                   duration_s: float, done_timeout_s: float, seed: int):
    rng = random.Random(seed * 1000 + n)
    tasks = [
        run_session(url, model, session_playlist(pool, i, n), session=i, chunk_ms=chunk_ms,
                    duration_s=duration_s, start_delay_s=rng.uniform(0, chunk_ms / 1000),
                    done_timeout_s=done_timeout_s)
        for i in range(n)
    ]
    results = await asyncio.gather(*tasks)
    records = [r for recs, _ in results for r in recs]
    return records, [err for _, err in results]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--stream", required=True, help="stream recipe id (serving/streaming.toml)")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    ap.add_argument("--http", default="http://127.0.0.1:8000")
    ap.add_argument("--hardware-label", required=True,
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
    args = ap.parse_args(argv)

    stream, base, identity, base_identity = _stream_recipe(args.stream)
    model = base["model"]
    engine = server_engine(args.http, stream["protocol"], model, identity)
    steps_n = [int(x) for x in args.steps.split(",")]
    offline = load_offline(args.offline_predictions)
    pool, pool_spec = build_pool(seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    params = {"steps": steps_n, "duration_s": args.duration_s, "threshold_s": args.threshold_s,
              "done_timeout_s": args.done_timeout_s, "pause_s": args.pause_s,
              "start_stagger": "uniform within one chunk, seeded", "seed": args.seed}
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
        "precision": base["precision"],
        "engine": engine,
        "hardware_label": args.hardware_label,
        "gpu": gpu_identity(),
        "pool": pool_spec,
        "offline_predictions": bool(offline),
        "params": params,
        "run_manifest": build_run_manifest(resolved_config=params, libraries=("websockets",),
                                           gpu=None),
        "steps": [],
        "max_n_within_threshold": None,
        "stopped_because": "running",
    }
    metrics = f"{args.http}/metrics" if stream["protocol"] == "vllm-realtime" else None

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

    for n in steps_n:
        print(f"[stream] {identity['id']}: N = {n}, {args.duration_s:.0f} s", flush=True)
        t = time.monotonic()
        with Sampler(metrics) as sampler:
            records, session_errors = asyncio.run(run_step(
                args.url, model, pool, n, chunk_ms=stream["chunk_ms"],
                duration_s=args.duration_s, done_timeout_s=args.done_timeout_s, seed=args.seed))
        step = step_summary(n, records, session_errors, sampler.summary(), offline,
                            args.threshold_s)
        step["wall_s"] = round(time.monotonic() - t, 1)
        curve["steps"].append(step)
        with (args.out / f"utterances_n{n}.jsonl").open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.as_json(), ensure_ascii=False) + "\n")
        curve["max_n_within_threshold"] = max_n_within(curve["steps"])
        write()
        print(f"[stream]   p50/p95 final {step['final_s']['p50']}/{step['final_s']['p95']} s, "
              f"errors {step['errors']}, drops {step['drops']}, WER {step['wer_strict_pct']}",
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
    curve["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    write()
    (args.out / "table.md").write_text(render_table(curve), encoding="utf-8")
    print(render_table(curve))
    return 0


if __name__ == "__main__":
    sys.exit(main())
