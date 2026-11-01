"""Fetch WER datasets at their pinned revision into the local cache, ahead of a run.

    python -m raven_asr.prefetch --dataset fleurs
    python -m raven_asr.prefetch --dataset all

A full run should read its corpus from local disk, not from the network: Hub
streaming times out on long jobs, and an unattended overnight run has nobody to
restart it. This downloads (and, for the archive-backed corpora, sha256-verifies
and unpacks) exactly what ``python -m raven_asr.reproduce`` reads, so a later run
finds it locally and never streams. Already-present files are not fetched again.

Where it lands:

* HF-hosted corpora (german-mixed, avemio, fleurs, mls-de, voxpopuli-de) go to
  the HF Hub cache, which follows ``HF_HOME`` / ``HF_HUB_CACHE``.
* Archive-backed corpora (spc-test, fhnw-all-dialects, xsid-*) go to
  ``$RAVEN_EVAL_CACHE_DIR/corpora`` (default ``~/.cache/raven.eval/corpora``),
  unless ``RAVEN_EVAL_CORPORA_DIR`` / ``BENCH_CORPORA_DIR`` or a corpus-specific
  variable (``SWISS_CORPORA_DIR``, ``XSID_CORPORA_DIR``) points elsewhere.

Exit code 0 only if every requested dataset is complete locally.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import WER_DATASETS, resolve_wer_dataset
from .datasets import load_loader_class

logger = logging.getLogger("raven_asr.prefetch")

ALL = "all"


def prefetch(dataset_id: str) -> list[str]:
    """Make ``dataset_id`` locally complete at its registry pin; return its paths."""
    spec = WER_DATASETS[dataset_id]
    loader = load_loader_class(dataset_id)(streaming=False, revision=spec.revision)
    return [str(p) for p in loader.prefetch()]


def _dataset_ids(selector: str) -> list[str]:
    if selector == ALL:
        return list(WER_DATASETS)
    dataset_id, _ = resolve_wer_dataset(selector)
    return [dataset_id]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-asr-prefetch", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", required=True,
        help=f"a WER dataset id ({', '.join(sorted(WER_DATASETS))}) or '{ALL}'",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        ids = _dataset_ids(args.dataset)
    except KeyError as exc:
        parser.error(str(exc))

    failed: list[str] = []
    for dataset_id in ids:
        try:
            paths = prefetch(dataset_id)
        except Exception as exc:  # noqa: BLE001 - reported per dataset, exit code says it
            logger.error("[%s] prefetch failed: %s: %s", dataset_id, type(exc).__name__, exc)
            failed.append(dataset_id)
            continue
        print(f"[{dataset_id}] ready: {len(paths)} path(s)")
        for p in paths:
            print(f"  {p}")
    if failed:
        print(f"prefetch incomplete: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
