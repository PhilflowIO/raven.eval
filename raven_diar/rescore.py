"""Backfill *newly published* scalars into a committed artifact's expected.json.

When the scorer starts reporting a quantity it did not report before — a second
collar's error decomposition, the file-mean aggregation — every artifact
committed before that change has an ``expected.json`` without the new keys, and
``make verify`` fails them for a missing field rather than a wrong number. The
run that produced those artifacts needed a GPU, a gated model licence and, for
the hosted rows, paid API calls; re-running it to obtain a number that is a
deterministic function of RTTMs already in the repository would be theatre.

So this tool recomputes the artifact's scores **from its own committed
gold/hyp RTTMs**, with the same module the runner and ``scripts/verify.py`` use,
and writes only the keys that were absent.

The guard that keeps this honest — and keeps it from becoming a way to make a
red build green — is that it **never changes a value that already exists**. If a
recomputed field disagrees with a committed one, that is exactly the drift
``make verify`` exists to catch, and this tool refuses to touch the file and
says so. Silencing that would delete the only mechanism standing between a
published table and a wrong number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .score import DerScore, score_rttm_pairs

#: A recomputed value may differ from a committed one only by float jitter. Same
#: constant as the Tier-1 re-score tolerance, and for the same reason.
TOLERANCE_PCT = 0.05


def find_der_model_dirs(artifacts_dir: Path) -> list[Path]:
    """Every dir that holds a DER artifact (a ``gold/`` subtree + expected.json)."""
    return sorted({
        p.parent
        for p in artifacts_dir.rglob("expected.json")
        if (p.parent / "gold").is_dir() and any((p.parent / "gold").rglob("*.rttm"))
    })


def _rttm_pairs(model_dir: Path, dataset: str) -> list[tuple[Path, Path]]:
    gold_dir = model_dir / "gold" / dataset
    hyp_dir = model_dir / "hyp" / dataset
    pairs: list[tuple[Path, Path]] = []
    for gold in sorted(gold_dir.glob("*.rttm")):
        hyp = hyp_dir / gold.name
        if not hyp.exists():
            raise FileNotFoundError(f"missing hypothesis RTTM {hyp}")
        pairs.append((gold, hyp))
    return pairs


def backfill(model_dir: Path, *, dry_run: bool = False) -> tuple[list[str], list[str]]:
    """Add missing published scalars to one artifact's expected.json.

    Returns ``(added, conflicts)``: the ``dataset.field`` keys written, and the
    ones whose committed value disagrees with the recomputation beyond
    :data:`TOLERANCE_PCT`. A non-empty ``conflicts`` means nothing is written.
    """
    expected_path = model_dir / "expected.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    gold_root = model_dir / "gold"

    added: list[str] = []
    conflicts: list[str] = []
    updated = {k: dict(v) for k, v in expected.items()}
    for ds_dir in sorted(p for p in gold_root.iterdir() if p.is_dir()):
        dataset = ds_dir.name
        if dataset not in updated:
            conflicts.append(f"{dataset}: no expected entry")
            continue
        score = score_rttm_pairs(dataset, _rttm_pairs(model_dir, dataset))
        for field_name in DerScore.EXPECTED_FIELDS:
            got = round(float(getattr(score, field_name)), 4)
            if field_name in updated[dataset]:
                delta = abs(got - float(updated[dataset][field_name]))
                if delta > TOLERANCE_PCT:
                    conflicts.append(
                        f"{dataset}.{field_name}: committed "
                        f"{updated[dataset][field_name]}, recomputed {got} "
                        f"(Δ{delta:.4f} > {TOLERANCE_PCT})"
                    )
                continue
            updated[dataset][field_name] = got
            added.append(f"{dataset}.{field_name}")

    if conflicts or not added or dry_run:
        return added, conflicts
    expected_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return added, conflicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-diar-rescore", description=__doc__)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="root to scan for committed DER artifacts (default: artifacts/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would be added without writing",
    )
    args = parser.parse_args(argv)

    model_dirs = find_der_model_dirs(args.artifacts_dir)
    if not model_dirs:
        print(f"no DER artifacts under {args.artifacts_dir}", file=sys.stderr)
        return 2

    n_added = 0
    failed = False
    for model_dir in model_dirs:
        added, conflicts = backfill(model_dir, dry_run=args.dry_run)
        rel = model_dir.relative_to(args.artifacts_dir)
        if conflicts:
            failed = True
            print(f"FAIL {rel}: committed values disagree with the RTTMs — "
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
