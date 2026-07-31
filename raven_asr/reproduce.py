"""Tier-2 entrypoint: download a public HF subset → infer → score → print a row.

Backs ``make reproduce METRIC=wer DATASET=<subset> MODEL=<model>``. Runs the
ported harness against one flozi public subset with the chosen adapter, writes
``predictions_<subset>.jsonl`` + ``summary.json`` (via the runner), and prints
the WER/CER in ``BENCHMARKS.md`` form. This is the "your own keys / GPU" lane —
the metric core is shared with Tier-1 (both call
``raven_eval_core.flozi_wer.evaluate``), so the number you get here is the same
number ``make verify`` re-scores later.

After a run, turn its output into a committable Tier-1 artifact with:
    python -m raven_asr.promote --results-dir <out>/<label> --run-name <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import FLOZI_SUBSETS, KNOWN_MODELS
from .runner import run

_SUPPORTED_METRICS = ("wer",)


def _print_benchmarks_row(summary_path: Path, model: str) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print()
    print("| model | dataset | WER strict % | CER % | n | reproduce |")
    print("|-------|---------|------------:|------:|--:|-----------|")
    for r in summary.get("results", []):
        cmd = (
            f"`make reproduce METRIC=wer DATASET={r['subset']} MODEL={model}`"
        )
        print(
            f"| {summary.get('model_id', model)} | {r['subset']} | "
            f"{r['wer_pct']:.4f} | {r['cer_pct']:.4f} | {r['n_samples']} | {cmd} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raven-asr-reproduce", description=__doc__
    )
    parser.add_argument("--metric", required=True, help="metric to reproduce (wer)")
    parser.add_argument(
        "--dataset", required=True,
        help=f"public flozi subset: {', '.join(FLOZI_SUBSETS)}",
    )
    parser.add_argument(
        "--model", required=True,
        help="model key from raven_asr.config.KNOWN_MODELS",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="cap samples per subset (smoke runs)")
    parser.add_argument("--out", type=Path, default=Path("results/reproduce"),
                        help="output root (a per-model subdir is created)")
    parser.add_argument("--streaming", action="store_true",
                        help="stream from HF instead of a local snapshot")
    parser.add_argument("--dataset-revision", default=None,
                        help="pin the HF dataset revision (reproducibility)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.metric not in _SUPPORTED_METRICS:
        parser.error(
            f"unsupported METRIC={args.metric!r}; supported: "
            f"{', '.join(_SUPPORTED_METRICS)} (DER lands in Etappe 5)"
        )
    if args.dataset not in FLOZI_SUBSETS:
        parser.error(
            f"unknown DATASET={args.dataset!r}; public subsets are "
            f"{', '.join(FLOZI_SUBSETS)}"
        )
    spec = KNOWN_MODELS.get(args.model)
    if spec is None:
        parser.error(
            f"unknown MODEL={args.model!r}; see raven_asr.config.KNOWN_MODELS"
        )

    out_dir = args.out / spec.label
    yaml_path = run(
        model_key=args.model,
        subsets=[args.dataset],
        limit=args.limit,
        out_dir=out_dir,
        streaming=args.streaming,
        revision=args.dataset_revision,
    )
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        _print_benchmarks_row(summary_path, args.model)
    print(f"\nwrote {yaml_path}")
    print(
        f"Tier-1 artifact:  python -m raven_asr.promote "
        f"--results-dir {out_dir} --run-name <date-or-name>"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
