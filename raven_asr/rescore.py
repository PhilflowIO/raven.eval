"""Backfill *newly required* fields into a committed WER artifact's expected.json.

The WER counterpart of ``raven_diar.rescore``. When ``make verify`` starts
requiring a field it did not require before — the numeric entity hit rate —
every WER artifact committed before that change fails for a missing field
rather than a wrong number. Those artifacts came from hosted APIs billed per
hour of audio or from a GPU run; re-running them for a number that is a
deterministic function of ``predictions_*.jsonl`` already in the repository
would be theatre.

So this tool derives the required fields **from the artifact's own committed
predictions**, through ``raven_asr.promote.add_required_fields`` — the function
promotion uses for a new run — and writes only the keys that were absent.

The guard is the same as for DER: it **never changes a value that already
exists**. A recomputed field that disagrees with a committed one is drift for
``make verify`` to report, and this tool refuses to touch that file.

Only fields ``make verify`` requires are backfilled. Opt-in keys (``bleu``,
``wer_strict_de_pct``) are a decision taken at promotion, not a gap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import artifact_subsets
from .promote import add_required_fields

#: A recomputed value may differ from a committed one only by float jitter. Same
#: constant as the Tier-1 re-score tolerance, and for the same reason.
TOLERANCE_PCT = 0.05


def find_wer_model_dirs(artifacts_dir: Path) -> list[Path]:
    """Every dir that holds a WER artifact (``predictions_*.jsonl``)."""
    return sorted({p.parent for p in artifacts_dir.rglob("predictions_*.jsonl")})


def _agrees(committed: object, recomputed: object) -> bool:
    if isinstance(committed, bool) or isinstance(recomputed, bool):
        return committed == recomputed
    if isinstance(committed, int) and isinstance(recomputed, int):
        return committed == recomputed      # counts are exact
    if isinstance(committed, (int, float)) and isinstance(recomputed, (int, float)):
        return abs(float(committed) - float(recomputed)) <= TOLERANCE_PCT
    return committed == recomputed


def backfill(model_dir: Path, *, dry_run: bool = False) -> tuple[list[str], list[str]]:
    """Add missing required fields to one WER artifact's expected.json.

    Returns ``(added, conflicts)``: the ``subset.field`` keys written, and the
    ones whose committed value disagrees with the recomputation. A non-empty
    ``conflicts`` means nothing is written.
    """
    expected_path = model_dir / "expected.json"
    if not expected_path.exists():
        return [], ["no expected.json — promote the run, do not backfill one"]
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    subsets = artifact_subsets(model_dir)

    conflicts = [f"{s}: no expected entry" for s in subsets if s not in expected]
    derived: dict[str, dict] = {s: {} for s in subsets if s in expected}
    add_required_fields(derived, [subsets[s] for s in derived])

    added: list[str] = []
    updated = {k: dict(v) for k, v in expected.items()}
    for subset, fields in derived.items():
        for name, got in fields.items():
            if name in updated[subset]:
                if not _agrees(updated[subset][name], got):
                    conflicts.append(
                        f"{subset}.{name}: committed {updated[subset][name]}, "
                        f"recomputed {got}"
                    )
                continue
            updated[subset][name] = got
            added.append(f"{subset}.{name}")

    if conflicts or not added or dry_run:
        return added, conflicts
    expected_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return added, conflicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-asr-rescore", description=__doc__)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="root to scan for committed WER artifacts (default: artifacts/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would be added without writing",
    )
    args = parser.parse_args(argv)

    model_dirs = find_wer_model_dirs(args.artifacts_dir)
    if not model_dirs:
        print(f"no WER artifacts under {args.artifacts_dir}", file=sys.stderr)
        return 2

    n_added = 0
    failed = False
    for model_dir in model_dirs:
        added, conflicts = backfill(model_dir, dry_run=args.dry_run)
        rel = model_dir.relative_to(args.artifacts_dir)
        if conflicts:
            failed = True
            print(f"FAIL {rel}: committed values disagree with the predictions — "
                  f"nothing written:", file=sys.stderr)
            for c in conflicts:
                print(f"      {c}", file=sys.stderr)
            continue
        if added:
            n_added += len(added)
            verb = "would add" if args.dry_run else "added"
            print(f"{verb} {len(added)} field(s) to {rel}/expected.json")
    if failed:
        return 1
    print(f"{'would add' if args.dry_run else 'added'} {n_added} field(s) total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
