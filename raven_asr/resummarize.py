"""Rebuild a results directory's ``summary.json`` from the subsets it finished.

Until the runner kept every finished subset in the summary (#100), a model
measured one subset per invocation ended with a summary holding only the last
subset, and ``raven_asr.promote`` refuses such a directory. This rewrites the
summary (and ``model-index.yaml``) through the runner's own writer, from what the
directory already recorded: no inference, no re-scoring. Idempotent.

    uv run --extra asr python -m raven_asr.resummarize results/<run>/<label> [...]

Subsets that finished before per-subset run manifests were recorded (no
``.conditions_<subset>.json``) cannot be held to the summary's manifest. They are
left out unless ``--adopt-unrecorded`` is given, which takes them on the
operator's word and records that in the summary's ``post_hoc`` list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runner import resummarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-asr-resummarize", description=__doc__)
    parser.add_argument("results_dirs", nargs="+", type=Path,
                        help="runner output dir(s), each holding summary.json + .done_ markers")
    parser.add_argument("--adopt-unrecorded", action="store_true",
                        help="also take finished subsets that record no run manifest "
                             "(written before per-subset sidecars); recorded as post_hoc")
    args = parser.parse_args(argv)
    for d in args.results_dirs:
        subsets = resummarize(d, adopt_unrecorded=args.adopt_unrecorded)
        print(f"{d}: summary holds {len(subsets)} subset(s): {', '.join(subsets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
