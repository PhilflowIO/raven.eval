"""CLI: build the test sets, run the arms, compare.

    python -m raven_asr.vad_ab fleurs-corpus --parquet P --out CORPUS
    (forced alignment: mfa align CORPUS german_mfa german_mfa ALIGN)
    python -m raven_asr.vad_ab build-a --corpus CORPUS --align ALIGN --demand D --musan M --out A
    python -m raven_asr.vad_ab manifest-b --callhome data/diar/callhome-de --out B/manifest.json
    python -m raven_asr.vad_ab run --manifest A/manifest.json --arms 0-silencedetect,2-silero-0.5 --out R
    python -m raven_asr.vad_ab compare --manifest A/manifest.json --results R --out R/compare-A.json
    python -m raven_asr.vad_ab crosstalk --manifest A/manifest.json --results R --arm 3-silero-0.3-norm

Engines run single-threaded per worker process; ``--workers`` sets how many
tracks run at once. A finished (arm, track) is skipped on a re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def _run_one(arm_name: str, track: dict, out_dir: str) -> str:
    import numpy as np

    from .arms import ARMS, run_arm

    res = run_arm(ARMS[arm_name], track["wav"])
    score = res.pop("score10")
    d = Path(out_dir, arm_name)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{track['id']}.npy", score)
    tmp = d / f"{track['id']}.json.tmp"
    tmp.write_text(json.dumps(res))
    tmp.replace(d / f"{track['id']}.json")
    return track["id"]


def cmd_run(a: argparse.Namespace) -> int:
    manifest = json.loads(Path(a.manifest).read_text())
    tracks = manifest["tracks"][: a.limit] if a.limit else manifest["tracks"]
    todo = [(arm, t) for arm in a.arms.split(",") for t in tracks
            if not Path(a.out, arm, f"{t['id']}.json").exists()]
    print(f"{len(todo)} (arm, track) pairs to run", flush=True)
    if a.workers <= 1:
        for arm, t in todo:
            _run_one(arm, t, a.out)
            print(f"{arm} {t['id']}", flush=True)
        return 0
    with ProcessPoolExecutor(a.workers) as pool:
        futs = {pool.submit(_run_one, arm, t, a.out): (arm, t["id"]) for arm, t in todo}
        for f in as_completed(futs):
            arm, tid = futs[f]
            f.result()
            print(f"{arm} {tid}", flush=True)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    from .report import compare, markdown

    manifest = json.loads(Path(a.manifest).read_text())
    arms = a.arms.split(",") if a.arms else sorted(p.name for p in Path(a.results).iterdir() if p.is_dir())
    cmp = compare(manifest["tracks"], a.results, arms)
    Path(a.out).write_text(json.dumps(cmp, indent=1))
    md = markdown(a.title or Path(a.manifest).parent.name, cmp)
    Path(a.out).with_suffix(".md").write_text(md)
    print(md)
    return 0


def cmd_crosstalk(a: argparse.Namespace) -> int:
    from .report import crosstalk_ab

    manifest = json.loads(Path(a.manifest).read_text())
    res = crosstalk_ab(manifest["tracks"], a.results, a.arm, a.threshold)
    out = Path(a.out or Path(a.results, f"crosstalk-{a.arm}.json"))
    out.write_text(json.dumps(res, indent=1))
    for k in ("flag_off", "flag_on"):
        r = res[k]
        print(f"{k}: missed {r['missed_pct']} %  FA {r['false_alarm_s_per_h']} s/h  "
              f"crosstalk sent {r['crosstalk_sent_pct']} %")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m raven_asr.vad_ab")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fleurs-corpus", help="FLEURS de rows → wav + lab for forced alignment")
    s.add_argument("--parquet", required=True)
    s.add_argument("--out", required=True)

    s = sub.add_parser("build-a", help="render Testset A")
    for k in ("corpus", "align", "demand", "musan", "out"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--seed", type=int, default=20261111)
    s.add_argument("--no-process", action="store_true", help="skip WebRTC NS + Opus (debug only)")

    s = sub.add_parser("manifest-b", help="Testset B manifest from CALLHOME-de + gold RTTM")
    s.add_argument("--callhome", required=True)
    s.add_argument("--out", required=True)

    s = sub.add_parser("run", help="run arms over a manifest")
    s.add_argument("--manifest", required=True)
    s.add_argument("--arms", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--limit", type=int, default=0)

    s = sub.add_parser("compare", help="compare the arms that have results")
    s.add_argument("--manifest", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--arms", default="")
    s.add_argument("--out", required=True)
    s.add_argument("--title", default="")

    s = sub.add_parser("crosstalk", help="winner with the cross-track flag off / on")
    s.add_argument("--manifest", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--threshold", type=float, default=None)
    s.add_argument("--out", default="")

    a = p.parse_args(argv)
    if a.cmd == "fleurs-corpus":
        from .testset_a import write_fleurs_corpus

        print(write_fleurs_corpus(a.parquet, a.out), "utterances")
        return 0
    if a.cmd == "build-a":
        from .testset_a import build

        m = build(a.corpus, a.align, a.demand, a.musan, a.out, a.seed, process=not a.no_process)
        print(len(m["tracks"]), "tracks,", m["aligned_utterances"], "aligned utterances")
        return 0
    if a.cmd == "manifest-b":
        from .testset_a import callhome_manifest

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        print(len(callhome_manifest(a.callhome, a.out)["tracks"]), "tracks")
        return 0
    return {"run": cmd_run, "compare": cmd_compare, "crosstalk": cmd_crosstalk}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
