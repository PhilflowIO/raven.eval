"""Promote a Tier-2 DER run into a committable Tier-1 artifact.

DER analogue of ``raven_asr.promote``. Takes a runner output dir
(``results/.../<label>/`` holding ``gold/<dataset>/*.rttm``,
``hyp/<dataset>/*.rttm`` and ``summary.json``) and materialises the layout
``scripts/verify.py`` re-scores:

    artifacts/<run-name>/<label>/
        gold/<dataset>/<file>.rttm    (copied verbatim — the reference diarization)
        hyp/<dataset>/<file>.rttm     (copied verbatim — the diarizer's hypothesis)
        expected.json                 ({dataset: {der_full, der_classic, miss, fa, conf}})
        summary.json                  (provenance: model, revisions, limit)

``expected.json`` is derived from ``summary.json`` (never hand-typed), so a
committed number can only ever equal what the scorer produced. Once committed,
``make verify`` recomputes DER from the copied RTTMs with the SAME core scorer —
no GPU, no gated model — and asserts a match within tolerance.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import sys
from pathlib import Path


def _safe_run_name(name: str) -> str:
    """Sanitize a run-name into a single path segment (mirrors raven_asr.promote)."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "run"
    if safe in {".", ".."}:
        safe = "run"
    return safe


def _expected_from_summary(summary: dict) -> dict[str, dict[str, float]]:
    """Build {dataset: {der_full, der_classic, miss, fa, conf}} from summary.json."""
    expected: dict[str, dict[str, float]] = {}
    for dataset, res in summary.get("results", {}).items():
        expected[dataset] = {
            "der_full": round(float(res["der_full"]), 4),
            "der_classic": round(float(res["der_classic"]), 4),
            "miss": round(float(res["miss"]), 4),
            "fa": round(float(res["fa"]), 4),
            "conf": round(float(res["conf"]), 4),
        }
    return expected


def promote(results_dir: Path, artifacts_dir: Path, run_name: str) -> Path:
    """Copy gold+hyp RTTM trees + emit expected.json under artifacts/<run>/<label>/."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"no summary.json in {results_dir} — run the diarizer first "
            f"(python -m raven_diar.reproduce ...)"
        )
    gold_root = results_dir / "gold"
    hyp_root = results_dir / "hyp"
    if not gold_root.is_dir() or not any(gold_root.rglob("*.rttm")):
        raise FileNotFoundError(
            f"no gold/*.rttm in {results_dir} — nothing to promote"
        )
    if not hyp_root.is_dir() or not any(hyp_root.rglob("*.rttm")):
        raise FileNotFoundError(
            f"no hyp/*.rttm in {results_dir} — nothing to promote"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = _expected_from_summary(summary)
    if not expected:
        raise ValueError(f"summary.json in {results_dir} has no results to promote")

    dest = artifacts_dir / _safe_run_name(run_name) / results_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for sub in ("gold", "hyp"):
        src = results_dir / sub
        dst = dest / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    (dest / "expected.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.copy2(summary_path, dest / "summary.json")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-diar-promote", description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="runner output dir (…/<label>/) with gold/ hyp/ + summary.json",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="destination root (default: artifacts/)",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="run subdir name (default: today's date, YYYY-MM-DD)",
    )
    args = parser.parse_args(argv)

    run_name = args.run_name or datetime.date.today().isoformat()
    dest = promote(args.results_dir, args.artifacts_dir, run_name)
    n_gold = len(list((dest / "gold").rglob("*.rttm")))
    n_hyp = len(list((dest / "hyp").rglob("*.rttm")))
    print(f"promoted {n_gold} gold + {n_hyp} hyp RTTM(s) -> {dest}")
    print(f"  wrote {dest / 'expected.json'}")
    print("Now: make verify   (re-scores the copied RTTMs, no GPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
