"""Tier-2 DER entrypoint: prepare public data → diarize → score → print a row.

Backs ``make reproduce METRIC=der DATASET=<id> MODEL=pyannote-community-1``. For
one public diarization dataset it: (1) prepares gold RTTMs (+ documents audio),
(2) runs the pinned diarizer over each recording to a hypothesis RTTM, (3) scores
DER at both collars with the shared core, (4) writes ``summary.json`` + the
``gold/`` and ``hyp/`` RTTM trees, and prints the ``BENCHMARKS.md`` row.

This is the "your own HF token + gated license + GPU" lane (docs/TIER2-DER-KEYS.md).
The metric core is shared with Tier-1 (both call ``raven_eval_core.der``), so the
DER you get here is the DER ``make verify`` re-scores from the committed RTTMs.

After a run, turn its output into a committable Tier-1 artifact with:
    python -m raven_diar.promote --results-dir <out>/<label> --run-name <name>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from raven_eval_core.der import to_rttm

from .config import DER_DATASETS, KNOWN_DIARIZERS
from .datasets.base import DiarDatasetLoader
from .score import score_rttm_pairs

logger = logging.getLogger("raven_diar.reproduce")

_SUPPORTED_METRICS = ("der",)


def _make_loader(loader_name: str, split: str | None = None) -> DiarDatasetLoader:
    if loader_name == "voxconverse":
        from .datasets.voxconverse import VoxConverseLoader
        return VoxConverseLoader(**({"split": split} if split else {}))
    if loader_name == "callhome_de":
        from .datasets.callhome_de import CallhomeDeLoader
        return CallhomeDeLoader()
    if loader_name == "ami":
        from .datasets.ami import AMILoader
        return AMILoader(**({"split": split} if split else {}))
    raise ValueError(f"unknown loader: {loader_name}")


def _make_diarizer(model_key: str, revision: str | None):
    spec = KNOWN_DIARIZERS[model_key]
    if spec.adapter == "pyannote_community1":
        from .adapters.pyannote_community1 import PyannoteCommunity1Diarizer
        return PyannoteCommunity1Diarizer(
            provider_id=spec.label,
            model_id=spec.model_id,
            revision=revision or spec.revision,
        )
    raise ValueError(f"unknown diarizer adapter: {spec.adapter}")


def _print_benchmarks_row(dataset: str, model: str, score) -> None:
    print()
    print("| model | dataset | DER (collar 0.0) | DER (collar 0.25) | "
          "miss | FA | conf | n | reproduce |")
    print("|-------|---------|-----------------:|------------------:|"
          "-----:|---:|-----:|--:|-----------|")
    cmd = f"`make reproduce METRIC=der DATASET={dataset} MODEL={model}`"
    print(
        f"| {model} | {dataset} | {score.der_full:.2f} | {score.der_classic:.2f} | "
        f"{score.miss:.2f} | {score.fa:.2f} | {score.conf:.2f} | "
        f"{score.n_files} | {cmd} |"
    )


def run(
    *,
    dataset: str,
    model_key: str,
    root: Path,
    out_dir: Path,
    limit: int | None,
    dataset_revision: str | None,
    model_revision: str | None,
    skip_prepare: bool,
) -> Path:
    """Prepare → diarize → score one dataset. Returns the summary.json path."""
    ds_spec = DER_DATASETS[dataset]
    loader = _make_loader(ds_spec.loader, ds_spec.split)
    if not skip_prepare:
        loader.prepare(root, revision=dataset_revision or ds_spec.revision)

    diarizer = _make_diarizer(model_key, model_revision)
    gold_dir = out_dir / "gold" / dataset
    hyp_dir = out_dir / "hyp" / dataset
    gold_dir.mkdir(parents=True, exist_ok=True)
    hyp_dir.mkdir(parents=True, exist_ok=True)

    rttm_pairs: list[tuple[Path, Path]] = []
    n_skipped = 0
    for df in loader.iter_files(root, limit=limit):
        if df.audio_path is None:
            n_skipped += 1
            logger.warning(
                "no audio for %s/%s — download it (see prepare() output); skipping",
                dataset, df.file_id,
            )
            continue
        result = diarizer.diarize(df.audio_path)
        hyp_path = hyp_dir / f"{df.file_id}.rttm"
        hyp_path.write_text(
            to_rttm(result.segments, file_id=df.file_id), encoding="utf-8"
        )
        gold_path = gold_dir / f"{df.file_id}.rttm"
        gold_path.write_bytes(df.gold_rttm_path.read_bytes())
        rttm_pairs.append((gold_path, hyp_path))
        logger.info("diarized %s (%.1fs, %d turns)",
                    df.file_id, result.latency_s, len(result.segments))

    if not rttm_pairs:
        raise RuntimeError(
            f"no files scored for {dataset!r} (skipped {n_skipped} for missing "
            f"audio) — download the audio and re-run (docs/TIER2-DER-KEYS.md)."
        )

    score = score_rttm_pairs(dataset, rttm_pairs)
    spec = KNOWN_DIARIZERS[model_key]
    summary = {
        "model_id": spec.model_id,
        "label": spec.label,
        "adapter": spec.adapter,
        "model_revision": model_revision or spec.revision,
        "dataset": dataset,
        "dataset_revision": dataset_revision or ds_spec.revision,
        "limit_per_dataset": limit,
        "n_skipped_no_audio": n_skipped,
        "results": {dataset: score.as_dict()},
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _print_benchmarks_row(dataset, model_key, score)
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-diar-reproduce", description=__doc__)
    parser.add_argument("--metric", required=True, help="metric to reproduce (der)")
    parser.add_argument(
        "--dataset", required=True,
        help=f"public DER dataset: {', '.join(DER_DATASETS)}",
    )
    parser.add_argument(
        "--model", required=True,
        help=f"diarizer key: {', '.join(KNOWN_DIARIZERS)}",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="cap files per dataset (smoke runs)")
    parser.add_argument("--root", type=Path, default=Path("data/diar"),
                        help="dataset preparation root (audio + gold live here)")
    parser.add_argument("--out", type=Path, default=Path("results/reproduce-der"),
                        help="output root (a per-model subdir is created)")
    parser.add_argument("--dataset-revision", default=None,
                        help="pin the dataset revision (tag/commit/HF hash)")
    parser.add_argument("--model-revision", default=None,
                        help="pin the diarizer HF revision hash")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="assume data already prepared under --root")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.metric not in _SUPPORTED_METRICS:
        parser.error(
            f"unsupported METRIC={args.metric!r}; supported: "
            f"{', '.join(_SUPPORTED_METRICS)} (WER lives in raven_asr.reproduce)"
        )
    if args.dataset not in DER_DATASETS:
        parser.error(
            f"unknown DATASET={args.dataset!r}; public DER datasets are "
            f"{', '.join(DER_DATASETS)}"
        )
    if args.model not in KNOWN_DIARIZERS:
        parser.error(
            f"unknown MODEL={args.model!r}; see raven_diar.config.KNOWN_DIARIZERS"
        )

    spec = KNOWN_DIARIZERS[args.model]
    out_dir = args.out / spec.label
    summary_path = run(
        dataset=args.dataset,
        model_key=args.model,
        root=args.root,
        out_dir=out_dir,
        limit=args.limit,
        dataset_revision=args.dataset_revision,
        model_revision=args.model_revision,
        skip_prepare=args.skip_prepare,
    )
    print(f"\nwrote {summary_path}")
    print(
        f"Tier-1 artifact:  python -m raven_diar.promote "
        f"--results-dir {out_dir} --run-name <date-or-name>"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
