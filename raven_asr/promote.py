"""Promote a Tier-2 run into a committable Tier-1 artifact (Handoff Etappe-4 task 4).

Takes a runner output dir (``results/.../<label>/`` holding
``predictions_<subset>.jsonl`` + ``summary.json``) and materialises the exact
layout ``scripts/verify.py`` re-scores:

    artifacts/<run-name>/<label>/
        predictions_<subset>.jsonl   (copied verbatim — the per-utterance model output)
        expected.json                ({subset: {wer_pct, cer_pct}} from summary.json,
                                      plus wer_strict_de_pct scored from the predictions)

Once committed, ``make verify`` re-computes the corpus WER/CER from the copied
predictions with the SAME core scorer that produced ``summary.json`` and asserts
they match ``expected.json`` within tolerance — no GPU, no keys. That is the
trust payoff: anyone re-checks the published number in seconds.

``expected.json`` is derived from ``summary.json`` (never hand-typed) so it can
only ever equal what the scorer actually produced.
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
    """Sanitize a run-name into a single path segment.

    Prevents ``--run-name`` from escaping ``artifacts/`` via ``../`` or an
    absolute path. Mirrors the filename sanitizer in ``raven_asr.runner``.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "run"
    if safe in {".", ".."}:
        safe = "run"
    return safe


def _expected_from_summary(summary: dict) -> dict[str, dict[str, float]]:
    """Build the {subset: {wer_pct, cer_pct}} contract from a runner summary.json."""
    expected: dict[str, dict[str, float]] = {}
    for r in summary.get("results", []):
        subset = r["subset"]
        expected[subset] = {
            "wer_pct": round(float(r["wer_pct"]), 4),
            "cer_pct": round(float(r["cer_pct"]), 4),
        }
    return expected


def _read_pairs(path: Path) -> tuple[list[str], list[str]]:
    refs: list[str] = []
    preds: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            row = json.loads(raw)
            refs.append(row["reference"])
            preds.append(row["prediction"])
    return refs, preds


def _add_bleu(expected: dict[str, dict], preds: list[Path]) -> None:
    """Attach corpus BLEU + its signature to every translation-shaped subset.

    ``benchmark.config.yaml`` makes BLEU the headline for a ``bleu+wer`` corpus
    (spoken dialect, Standard German reference). The verifier already re-scores
    a committed ``bleu`` key; this is where a real run gets one, from the
    dataset's declared metric rather than from a flag the caller could forget.
    Scored on raw text, exactly as ``scripts/verify.py`` re-scores it.
    """
    from raven_asr.config import WER_DATASETS, resolve_wer_dataset
    from raven_eval_core import bleu_signature, corpus_bleu_score

    for path in preds:
        subset = _subset_for(path, expected)
        if subset is None:
            continue
        try:
            dataset_id, _ = resolve_wer_dataset(subset)
        except KeyError:
            continue
        if WER_DATASETS[dataset_id].metric != "bleu+wer":
            continue
        refs, hyps = _read_pairs(path)
        expected[subset]["bleu"] = round(corpus_bleu_score(refs, hyps), 4)
        expected[subset]["bleu_signature"] = bleu_signature()


def _subset_for(path: Path, expected: dict) -> str | None:
    # Same filename rule as raven_asr.runner._safe (not imported: the runner
    # pulls numpy/tqdm, and promote must stay light).
    for s in expected:
        if path.name == f"predictions_{re.sub(r'[^A-Za-z0-9._-]+', '_', s)}.jsonl":
            return s
    return None


def _add_strict_de(expected: dict[str, dict[str, float]], preds: list[Path]) -> None:
    """Attach the strict-de corpus WER to every subset that has predictions.

    The published ``wer_pct`` is flozi-strict; the Raven benchmark page reads
    strict-de, length-weighted (``raven_eval_core.corpus_wer_strict_de_pct``).
    Committing both lets ``make verify`` defend the page number too, instead of
    the page computing it from these predictions on its own.
    """
    from raven_eval_core import corpus_wer_strict_de_pct

    for path in preds:
        subset = _subset_for(path, expected)
        if subset is None:
            continue
        refs, hyps = _read_pairs(path)
        expected[subset]["wer_strict_de_pct"] = round(
            corpus_wer_strict_de_pct(refs, hyps), 4
        )


def promote(
    results_dir: Path,
    artifacts_dir: Path,
    run_name: str,
) -> Path:
    """Copy predictions + emit expected.json under ``artifacts/<run-name>/<label>/``.

    Returns the destination model dir. Raises on a results dir that has no
    predictions or no summary.json (so we never emit an empty/undefended
    artifact).
    """
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"no summary.json in {results_dir} — run the model first "
            f"(python -m raven_asr.reproduce ...)"
        )
    preds = sorted(results_dir.glob("predictions_*.jsonl"))
    if not preds:
        raise FileNotFoundError(
            f"no predictions_*.jsonl in {results_dir} — nothing to promote"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = _expected_from_summary(summary)
    _add_strict_de(expected, preds)
    _add_bleu(expected, preds)
    if not expected:
        raise ValueError(f"summary.json in {results_dir} has no results to promote")

    dest = artifacts_dir / _safe_run_name(run_name) / results_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for p in preds:
        shutil.copy2(p, dest / p.name)
    (dest / "expected.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Carry the summary alongside for provenance (model_id, adapter, limit).
    shutil.copy2(summary_path, dest / "summary.json")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raven-asr-promote", description=__doc__
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="runner output dir (…/<label>/) with predictions + summary.json",
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
    n_preds = len(list(dest.glob("predictions_*.jsonl")))
    print(f"promoted {n_preds} prediction file(s) -> {dest}")
    print(f"  wrote {dest / 'expected.json'}")
    print("Now: make verify   (re-scores the copied predictions, no GPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
