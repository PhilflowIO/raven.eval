"""Offline load test: production-shaped requests from long recordings, against one server.

    python -m raven_asr.load prepare --corpus DIR --ids a,b --cache DIR
    python -m raven_asr.load sweep   --url U --model M --kind vad|window --out DIR ...
    python -m raven_asr.load run     --url U --model M --kind vad|window --n 4 --out DIR ...
    python -m raven_asr.load score   --out DIR --kind vad|window ...
    python -m raven_asr.load probe-long --url U --model M --recording ID --out DIR ...
    python -m raven_asr.load compare --baseline DIR --runs DIR,DIR --kind vad

Common: ``--corpus`` (``<id>.opus``/``.gt.txt``/``.manifest.json``[/``.vtt``]),
``--ids`` (comma-separated recording ids), ``--cache`` (decoded audio + plans,
raven_asr.load.workload). ``--url`` is the server's
``…/v1/audio/transcriptions``; start the server first
(``python serving/launch.py <recipe>``, plus ``serving/vllm/shim.py`` for vLLM and
transformers recipes).

prepare     decode + silencedetect once, plan both workloads (VAD blocks, ~600 s
            windows), print what production would do with each recording.
sweep       the capacity curve, N = 1, 2, 4 … 64 requests in flight, one fixed
            slice per step (raven_asr.load.sweep). Writes curve_<kind>.json,
            outcomes_<kind>_n<N>.jsonl, table_<kind>.md.
run         every block of every recording at N in flight, retried once like
            production; resumable. Writes answers_<kind>.jsonl and run_<kind>.json
            (the run is also one capacity point, over the whole corpus).
score       stitched per-recording WER, windows, loops, runaway, insertion rate,
            word-timestamp check against VTT cues (raven_asr.load.score, .timing).
            Writes score_<kind>.json and score_<kind>.md.
probe-long  one request per growing prefix of a recording (600 s … whole), under
            a generous timeout, to see what the server does with long input.
compare     paired bootstrap of WER between runs, per recording and corpus.
variant-env the host knobs of a named load variant (serving/load_variants.toml).

``--variant <base>@<name>`` records which named server setting the run measured;
the server must have been started with that variant's knobs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from raven_eval_core.run_manifest import build_run_manifest

from ..adapters.base import parse_word_timestamps
from ..capacity import max_n_within, next_action
from ..probes import Sampler, gpu_identity
from . import score as sc
from . import sweep as sw
from .client import Outcome, form_fields, probe_timestamps
from .timing import parse_vtt, summarize, vtt_check
from .vad import Block, remap_word
from .variants import load_variants
from .variants import record as variant_record
from .workload import KINDS, Item, Prepared, all_items, find_recordings, prepare, sweep_slice

SCHEMA = "raven.offline-load/1"


def _prepared(args) -> list[Prepared]:
    recs = find_recordings(Path(args.corpus), [x for x in args.ids.split(",") if x])
    return [prepare(r, Path(args.cache)) for r in recs]


def _wav_of(prepared: list[Prepared]):
    by_id = {p.recording.rid: p for p in prepared}
    return lambda it: by_id[it.rid].wav_bytes(it.block)


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _header(args, prepared: list[Prepared], kind: str, timestamps: dict) -> dict:
    return {
        "schema": SCHEMA,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "recipe": args.recipe,
        "variant": variant_record(args.variant, args.recipe),
        "kind": kind,
        "hardware_label": args.hardware_label,
        "gpu": gpu_identity(),
        "timestamps": timestamps,
        "recordings": [{"id": p.recording.rid, "duration_s": round(p.duration_s, 1),
                        "speech_ratio": round(p.plan["speech_ratio"], 3),
                        "production_path": p.plan["production_path"],
                        "blocks": len(p.plan["workloads"][kind]),
                        "source_sha256": p.plan["source_sha256"]} for p in prepared],
    }


async def _probe(args, prepared: list[Prepared], kind: str) -> dict:
    it = all_items(prepared, kind)[0]
    async with httpx.AsyncClient() as c:
        return await probe_timestamps(c, args.url, args.model, _wav_of(prepared)(it),
                                      it.block.audio_s)


def cmd_prepare(args) -> int:
    for p in _prepared(args):
        pl = p.plan
        vb = pl["workloads"]["vad"]
        print(f"{p.recording.rid}: {p.duration_s / 60:.1f} min, speech ratio "
              f"{pl['speech_ratio']:.2f}, production: {pl['production_path']}; "
              f"{len(vb)} VAD blocks, {len(pl['workloads']['window'])} windows")
    return 0


def cmd_sweep(args) -> int:
    prepared = _prepared(args)
    kind = args.kind
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps_n = [int(x) for x in args.steps.split(",")]
    timestamps = asyncio.run(_probe(args, prepared, kind))
    available = len(all_items(prepared, kind))
    curve = {**_header(args, prepared, kind, timestamps),
             "params": {"steps": steps_n, "seed": args.seed, "pause_s": args.pause_s,
                        "slice": {str(n): sw.slice_size(kind, n, available) for n in steps_n},
                        "retries": 0, "step_cap_s": sw.STEP_CAP_S},
             "steps": [], "max_n_within_threshold": None, "stopped_because": "running"}
    curve["run_manifest"] = build_run_manifest(resolved_config=curve["params"],
                                               libraries=("httpx",), gpu=None)
    wav_of = _wav_of(prepared)
    # One block before the curve, not counted: CUDA init and kernel compilation.
    warm = all_items(prepared, kind)[0]
    asyncio.run(sw.run_items(args.url, args.model, [warm], wav_of, 1, timestamps["ask"]))
    for n in steps_n:
        items = sweep_slice(prepared, kind, sw.slice_size(kind, n, available), args.seed)
        print(f"[load] {args.model} {kind}: N = {n}, {len(items)} requests", flush=True)
        step, outs = asyncio.run(sw.run_step(args.url, args.model, items, wav_of, n,
                                             timestamps["ask"], args.metrics_url))
        curve["steps"].append(step)
        with (out / f"outcomes_{kind}_n{n}.jsonl").open("w", encoding="utf-8") as fh:
            for o in outs:
                d = o.as_json()
                d.pop("words", None)
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        curve["max_n_within_threshold"] = max_n_within(curve["steps"])
        _write(out / f"curve_{kind}.json", curve)
        print(f"[load]   RTFx {step['rtfx']} (steady {step['rtfx_steady']}), p95 "
              f"{step['latency_s']['p95']} s, p95/budget {step['latency_over_budget']['p95']}, "
              f"errors {step['errors']}, timeouts {step['timeouts']}, aborts {step['aborts']}",
              flush=True)
        action = next_action(curve["steps"])
        if action == "stop":
            curve["stopped_because"] = "one step recorded beyond the first step over the rule"
            break
        time.sleep(args.pause_s)
    else:
        curve["stopped_because"] = ("every step passed" if next_action(curve["steps"]) == "continue"
                                    else "the last step broke the rule; no larger step configured")
    curve["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    _write(out / f"curve_{kind}.json", curve)
    (out / f"table_{kind}.md").write_text(sw.render_table(curve), encoding="utf-8")
    print(sw.render_table(curve))
    return 0


def cmd_run(args) -> int:
    prepared = _prepared(args)
    kind = args.kind
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answers = out / f"answers_{kind}.jsonl"
    done: set[str] = set()
    if answers.exists():
        for line in answers.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] == "ok":
                done.add(row["key"])
    items = [it for it in all_items(prepared, kind) if it.key not in done]
    if args.limit:
        items = items[:args.limit]
    timestamps = asyncio.run(_probe(args, prepared, kind))
    head = {**_header(args, prepared, kind, timestamps),
            "params": {"n": args.n, "retries": 1, "resumed_ok": len(done),
                       "requests": len(items), "limit": args.limit}}
    head["run_manifest"] = build_run_manifest(resolved_config=head["params"],
                                              libraries=("httpx",), gpu=None)
    print(f"[load] {args.model} {kind}: {len(items)} requests at N = {args.n} "
          f"({len(done)} already answered)", flush=True)
    fh = answers.open("a", encoding="utf-8")
    count = [0]

    def on_done(it: Item, o: Outcome) -> None:
        d = o.as_json()
        d["rid"], d["kind"], d["index"] = it.rid, it.kind, it.block.index
        fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        fh.flush()
        count[0] += 1
        if count[0] % 500 == 0:
            print(f"[load]   {count[0]}/{len(items)}", flush=True)

    c0 = sw.cpu_seconds()
    with Sampler(args.metrics_url) as sampler:
        outs, t0, t_last = asyncio.run(sw.run_items(
            args.url, args.model, items, _wav_of(prepared), args.n, timestamps["ask"],
            retries=1, cap_s=args.cap_s, on_done=on_done))
    fh.close()
    head["capacity_point"] = sw.step_summary(args.n, outs, t0, t_last, sampler.summary(),
                                             sw.cpu_seconds() - c0)
    head["retried"] = sum(1 for o in outs if o.attempts > 1)
    head["retried_recovered"] = sum(1 for o in outs if o.attempts > 1 and o.status == "ok")
    head["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    _write(out / f"run_{kind}.json", head)
    cp = head["capacity_point"]
    print(f"[load] done: RTFx {cp['rtfx']}, errors {cp['errors']}, timeouts {cp['timeouts']}, "
          f"retried {head['retried']} (recovered {head['retried_recovered']})", flush=True)
    return 0


def _answers(out: Path, kind: str) -> dict[str, dict[int, dict]]:
    """Latest row per block: ``{rid: {index: row}}`` (an ok row beats a failed one)."""
    rows: dict[str, dict[int, dict]] = {}
    for line in (out / f"answers_{kind}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        cur = rows.setdefault(r["rid"], {}).get(r["index"])
        if cur is None or r["status"] == "ok" or cur["status"] != "ok":
            rows[r["rid"]][r["index"]] = r
    return rows


def cmd_score(args) -> int:
    prepared = _prepared(args)
    kind = args.kind
    out = Path(args.out)
    rows = _answers(out, kind)
    scores: dict[str, dict] = {}
    timing: dict[str, dict] = {}
    missing = {}
    for p in prepared:
        rid = p.recording.rid
        got = rows.get(rid, {})
        answers = []
        words_src: list[dict] = []
        n_words_blocks = 0
        for b in p.blocks(kind):
            r = got.get(b.index)
            ok = r is not None and r["status"] == "ok"
            answers.append(sc.Answer(b.index, b.source_start, b.source_end,
                                     r["text"] if ok else None))
            if ok and r.get("words"):
                n_words_blocks += 1
                for w in r["words"]:
                    s, e = remap_word(b.pieces, float(w["start"]), float(w["end"]))
                    words_src.append({"word": w["word"], "start": s, "end": e})
        missing[rid] = sum(1 for b in p.blocks(kind) if b.index not in got)
        s = sc.score_recording(p.recording.reference.read_text(encoding="utf-8"), answers)
        s["dialect"] = p.recording.dialect
        s["production_path"] = p.plan["production_path"]
        s["duration_s"] = round(p.duration_s, 1)
        s["blocks_with_words"] = n_words_blocks
        s["never_sent"] = missing[rid]
        scores[rid] = s
        if p.recording.vtt is not None:
            words_src.sort(key=lambda w: w["start"])
            timing[rid] = (vtt_check(p.recording.vtt, words_src) if words_src else
                           {**summarize([], len(parse_vtt(p.recording.vtt.read_text()))),
                            "note": "the server returned no word timestamps"})
    std = {k: v for k, v in scores.items() if not v["dialect"]}
    dia = {k: v for k, v in scores.items() if v["dialect"]}
    result = {
        "schema": SCHEMA + "/score",
        "kind": kind,
        "run": (json.loads((out / f"run_{kind}.json").read_text())
                if (out / f"run_{kind}.json").exists() else None),
        "lens": ("flozi-strict (normalize_flozi + jiwer wer_standardize), "
                 "one alignment per recording"),
        "recordings": scores,
        "corpus": sc.corpus_interval(scores),
        "corpus_standard_german": sc.corpus_interval(std) if std else None,
        "corpus_dialect": sc.corpus_interval(dia) if dia else None,
        "timing_vs_vtt": timing,
    }
    _write(out / f"score_{kind}.json", result)
    (out / f"score_{kind}.md").write_text(render_score(result), encoding="utf-8")
    print(render_score(result))
    return 0


def render_score(res: dict) -> str:
    run = res.get("run") or {}
    lines = [f"### {run.get('model', '?')} — long-form accuracy, {res['kind']} blocks", "",
             f"Lens: {res['lens']}. Windows of {sc.WINDOW_REF_WORDS} reference words.", "",
             ("| recording | min | WER % [95 %] | ins % | del % | hyp/ref | worst window % "
              "| runaway win | dropout win | loops (ref) | runaway answers | unanswered |"),
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rid, s in res["recordings"].items():
        ra = s["runaway_answers"] or {}
        lines.append(
            f"| {rid}{' (dialect)' if s['dialect'] else ''} | {s['duration_s'] / 60:.0f} "
            f"| {s['wer_pct']:.2f} [{s['wer_ci95'][0]:.2f}, {s['wer_ci95'][1]:.2f}] "
            f"| {s['insertion_rate_pct']:.2f} | {s['deletion_rate_pct']:.2f} "
            f"| {s['hyp_over_ref']:.3f} | {s['worst_window_wer_pct']} | {s['runaway_windows']} "
            f"| {s['dropout_windows']} | {s['loop_count']} ({s['ref_loop_count']}) "
            f"| {ra.get('n', '–')}/{ra.get('of', '–')} | {s['failed_answers']} |")
    for name in ("corpus", "corpus_standard_german", "corpus_dialect"):
        c = res.get(name)
        if c:
            lines.append(f"\n{name}: WER {c['wer_pct']:.2f} % [{c['ci95'][0]:.2f}, "
                         f"{c['ci95'][1]:.2f}] over {c['windows']} windows")
    for rid, t in res["timing_vs_vtt"].items():
        lines.append(f"\ntiming vs VTT ({rid}): {json.dumps(t)}")
    return "\n".join(lines) + "\n"


def cmd_compare(args) -> int:
    base_dir = Path(args.baseline)
    base = json.loads((base_dir / f"score_{args.kind}.json").read_text())
    lines = [f"### Paired WER deltas vs {base['run']['model']} ({args.kind} blocks)", "",
             "| model | recording | WER % | Δ pp vs baseline [95 %] |", "|---|---|---:|---:|"]
    out = {"baseline": base["run"]["model"], "kind": args.kind, "models": {}}
    for d in [x for x in args.runs.split(",") if x]:
        other = json.loads((Path(d) / f"score_{args.kind}.json").read_text())
        m = other["run"]["model"]
        res = {}
        groups = {"corpus": None, "standard_german": False, "dialect": True}
        for gname, flag in groups.items():
            a = {k: v for k, v in other["recordings"].items()
                 if flag is None or v["dialect"] == flag}
            b = {k: v for k, v in base["recordings"].items()
                 if flag is None or v["dialect"] == flag}
            if a:
                res[gname] = {"wer_pct": sc.corpus_interval(a)["wer_pct"], **sc.paired_delta(a, b)}
        for rid in other["recordings"]:
            a = {rid: other["recordings"][rid]}
            b = {rid: base["recordings"][rid]}
            res[rid] = {"wer_pct": other["recordings"][rid]["wer_pct"], **sc.paired_delta(a, b)}
        out["models"][m] = res
        for k, v in res.items():
            lines.append(f"| {m} | {k} | {v['wer_pct']:.2f} | {v['delta_pp']:+.2f} "
                         f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] |")
    if args.out:
        _write(Path(args.out) / f"compare_{args.kind}.json", out)
        (Path(args.out) / f"compare_{args.kind}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def cmd_probe_long(args) -> int:
    """Growing prefixes of one recording, one request each, generous timeout."""
    prepared = _prepared(args)
    p = next(x for x in prepared if x.recording.rid == args.recording)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cues = parse_vtt(p.recording.vtt.read_text()) if p.recording.vtt else None
    rows = []
    timestamps = asyncio.run(_probe(args, prepared, "window"))
    durations = [float(x) for x in args.durations.split(",")]
    for dur in durations:
        dur = min(dur, p.duration_s)
        frames = int(dur * 16000)
        blk = Block(0, ((0, frames),), ((0.0, dur, 0.0),))
        wav = p.wav_bytes(blk)

        async def once() -> tuple:
            timeout = httpx.Timeout(connect=10.0, read=args.timeout_s, write=120.0, pool=None)
            async with httpx.AsyncClient(timeout=timeout) as c:
                t = time.monotonic()
                res = await post_once_long(c, args.url, args.model, wav, timestamps["ask"],
                                           args.timeout_s)
                return res, time.monotonic() - t

        with Sampler(args.metrics_url) as sampler:
            (status, code, err, body), wall = asyncio.run(once())
        row = {"duration_s": round(dur, 1), "status": status, "http_status": code,
               "error": err, "wall_s": round(wall, 2), "probes": sampler.summary()}
        if body is not None:
            text = str(body.get("text", ""))
            hyp = sc.tokens(text)
            row["hyp_words"] = len(hyp)
            row["loops"] = [{"n": n, "reps": r, "gram": " ".join(hyp[s:s + n])}
                            for s, n, r in sc.loop_spans(hyp)]
            if cues:
                ref_text = " ".join(c.text for c in cues if c.start < dur)
                ref = sc.tokens(ref_text)
                c = sc.edits(sc.align(ref, hyp))
                row.update({"ref_words": len(ref),
                            "wer_vs_vtt_pct": round(100.0 * (c["substitute"] + c["delete"]
                                                             + c["insert"]) / len(ref), 2),
                            "deletion_rate_pct": round(100.0 * c["delete"] / len(ref), 2),
                            "insertion_rate_pct": round(100.0 * c["insert"] / len(ref), 2)})
            # The tail: does the answer still cover the last minutes of the clip?
            words = parse_word_timestamps(body)
            if words:
                row["last_word_end_s"] = round(max(float(w["end"]) for w in words), 1)
        rows.append(row)
        print(f"[probe-long] {dur:.0f} s: {status} {code or ''} in {row['wall_s']} s; "
              f"VRAM max {row['probes'].get('memory_used_max_mib')} MiB; "
              f"WER vs VTT {row.get('wer_vs_vtt_pct')}; loops {len(row.get('loops', []))}",
              flush=True)
        _write(out / "probe_long.json", {"schema": SCHEMA + "/probe-long", "model": args.model,
                                         "recording": args.recording, "timestamps": timestamps,
                                         "hardware_label": args.hardware_label, "rows": rows})
        if status != "ok" and args.stop_on_failure:
            break
    return 0


async def post_once_long(client, url, model, wav, ask, timeout_s):
    """One upload with a fixed (not production) timeout; same shape as client.post_once."""
    try:
        r = await client.post(url, files={"file": ("long.wav", wav, "audio/wav")},
                              data=form_fields(model, ask),
                              timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=120.0,
                                                    pool=None))
    except httpx.TimeoutException as exc:
        return "timeout", None, type(exc).__name__, None
    except httpx.TransportError as exc:
        return "conn_error", None, f"{type(exc).__name__}: {exc}"[:300], None
    if r.status_code != 200:
        return "http_error", r.status_code, r.text[:500], None
    return "ok", 200, None, r.json()


def cmd_variant_env(args) -> int:
    for k, v in load_variants()[args.variant]["env"].items():
        print(f"{k}={v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, server: bool = True) -> None:
        p.add_argument("--corpus", required=True)
        p.add_argument("--ids", required=True)
        p.add_argument("--cache", required=True)
        if server:
            p.add_argument("--url", required=True, help="…/v1/audio/transcriptions")
            p.add_argument("--model", required=True, help="the served model id")
            p.add_argument("--recipe", help="serving recipe id (recorded)")
            p.add_argument("--variant", help="named load variant, if any (recorded)")
            p.add_argument("--hardware-label", required=True)
            p.add_argument("--metrics-url", help="vLLM engine /metrics (engine gauges)")

    p = sub.add_parser("prepare")
    common(p, server=False)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("sweep")
    common(p)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", default=",".join(map(str, sw.DEFAULT_STEPS)))
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--pause-s", type=float, default=10.0)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="only the first K pending blocks")
    p.add_argument("--cap-s", type=float, default=6 * 3600.0)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("score")
    common(p, server=False)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("probe-long")
    common(p)
    p.add_argument("--recording", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--durations", default="600,1200,1800,2700,3600,6000")
    p.add_argument("--timeout-s", type=float, default=1800.0)
    p.add_argument("--stop-on-failure", action="store_true")
    p.set_defaults(fn=cmd_probe_long)

    p = sub.add_parser("compare")
    p.add_argument("--baseline", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("variant-env", help="print a load variant's host knobs as KEY=VALUE")
    p.add_argument("variant")
    p.set_defaults(fn=cmd_variant_env)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
