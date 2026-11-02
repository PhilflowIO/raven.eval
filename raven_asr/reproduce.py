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

A run is resumable per utterance: run the same command again and it requests
only the clips that have no transcription yet (see ``raven_asr.runner``).

Exit status, for an unattended caller that only reads the code:
    0  every utterance of the dataset succeeded and ``.done_<subset>.json`` exists
    1  the run finished but is incomplete (failed requests) — run it again
    3  the endpoint serves a different model than ``--model``, or its serving
       setup (raven_asr.serving) changed under the run; nothing recorded
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters.base import ServedModelMismatchError
from .config import FLOZI_SUBSETS, KNOWN_MODELS, WER_DATASETS, resolve_wer_dataset
from .runner import _safe, run

_SUPPORTED_METRICS = ("wer",)

EXIT_INCOMPLETE = 1
EXIT_SERVED_MODEL_MISMATCH = 3


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _subset_complete(out_dir: Path, subset: str) -> bool:
    """Whether ``subset`` finished with every utterance transcribed.

    The ``.done_`` marker is only ever written for a pass without a failed
    request, and removed by any later pass that was not one.
    """
    return (out_dir / f".done_{_safe(subset)}.json").is_file()


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
        help=(
            "public WER dataset id: " + ", ".join(sorted(WER_DATASETS))
            + " — or a german-mixed subset: " + ", ".join(FLOZI_SUBSETS)
        ),
    )
    parser.add_argument(
        "--model", required=True,
        help="model key from raven_asr.config.KNOWN_MODELS",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="cap samples per subset (smoke runs)")
    parser.add_argument("--out", type=Path, default=Path("results/reproduce"),
                        help="output root (a per-model subdir is created)")
    parser.add_argument("--concurrency", type=_positive_int, default=None,
                        help="requests in flight (default: the adapter's own)")
    parser.add_argument("--streaming", action="store_true",
                        help="stream from the HF Hub instead of the pinned local "
                        "snapshot (saves the download for a --limit smoke run)")
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
    try:
        resolve_wer_dataset(args.dataset)
    except KeyError as exc:
        parser.error(str(exc))
    spec = KNOWN_MODELS.get(args.model)
    if spec is None:
        parser.error(
            f"unknown MODEL={args.model!r}; see raven_asr.config.KNOWN_MODELS"
        )

    out_dir = args.out / spec.label
    try:
        yaml_path = run(
            model_key=args.model,
            subsets=[args.dataset],
            limit=args.limit,
            out_dir=out_dir,
            streaming=args.streaming,
            concurrency=args.concurrency,
            revision=args.dataset_revision,
        )
    except ServedModelMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_SERVED_MODEL_MISMATCH
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        _print_benchmarks_row(summary_path, args.model)
    print(f"\nwrote {yaml_path}")
    if not _subset_complete(out_dir, args.dataset):
        print(
            f"incomplete: {args.dataset} has failed or missing utterances; run "
            f"the same command again to retry only those",
            file=sys.stderr,
        )
        return EXIT_INCOMPLETE
    print(
        f"Tier-1 artifact:  python -m raven_asr.promote "
        f"--results-dir {out_dir} --run-name <date-or-name>"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
