#!/usr/bin/env python3
"""Tier-1 zero-setup verification: re-score committed model outputs → published WER.

Scans ``artifacts/**/<run>/<model>/predictions_<subset>.jsonl`` (the exact
per-utterance ``{"reference","prediction","latency_s"}`` lines Raven's eval
runner writes), re-computes the **corpus** WER + CER per subset with our own
scorer, and asserts each matches the expected value committed alongside in
``artifacts/<run>/<model>/expected.json``. Runs with no GPU and no API keys.

Exit code is nonzero on any mismatch OR on an empty artifacts dir (so CI can't
silently go green on a run that produced nothing).

--------------------------------------------------------------------------------
Why this file re-implements the flozi WER pipeline instead of calling
``raven_eval_core.normalize_strict_de``
--------------------------------------------------------------------------------
The published Raven ASR numbers come from the german-asr runner, whose
``evaluate()`` (flow.raven/evaluation/german-asr/german_asr/wer.py) uses the
*flozi-strict* method. That method is materially DIFFERENT from
``raven_eval_core.normalize_strict_de`` — using the latter would NOT reproduce
the published number. The concrete divergences (each would move WER):

  1. Number handling. flozi runs ``alpha2digit(text,"de")`` — number *words*
     -> *digits* ("drei" -> "3"). raven_eval_core runs ``num2words`` in the
     OPPOSITE direction (digits -> words). Opposite canonical forms.
  2. Fillers. flozi-strict does NOT strip fillers (that is a separate
     ``compute_wer_filler_tolerant_pct`` variant). ``normalize_strict_de``
     strips äh/ähm/… unconditionally, deleting reference words -> different WER.
  3. Transliteration. flozi runs ``unidecode`` (umlauts preserved via an
     escape/restore dance). ``normalize_strict_de`` does not.
  4. Case + dedup. flozi keeps case in ``normalize_text`` and defers
     lowercasing + contiguous-dup collapse to jiwer's
     ``wer_standardize_contiguous`` transforms at WER time.
     ``normalize_strict_de`` lowercases inline and calls plain ``jiwer.wer``.

So the flozi pipeline (same ``unidecode``/``alpha2digit`` normalization, same
``wer_standardize_contiguous`` transforms, same corpus aggregation, same
raw-text CER) is what scores here. Goal, per the task contract: same
``predictions_*.jsonl`` in -> same published ``wer_pct`` out.

Canonical implementation (imported, NOT copied): ``raven_eval_core.flozi_wer``.
Etappe 3 mirrored the flozi pipeline inline here; Etappe 4 collapsed that copy
into the core so the runner (writer) and this re-scorer (checker) share one
implementation and cannot drift. ``normalize_flozi`` / ``corpus_wer_pct`` /
``corpus_cer_pct`` are re-exported below under their historic names so callers
and tests keep working.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from raven_eval_core.der import compute_der_corpus, load_rttm
from raven_eval_core.flozi_wer import corpus_cer_pct, corpus_wer_pct
from raven_eval_core.flozi_wer import normalize_flozi as normalize_text

# Abs-diff tolerance on wer_pct / cer_pct. 0.05 pp absorbs float/lib jitter
# (jiwer/text_to_num patch releases) without hiding a real regression: a single
# extra word error on a subset of a few hundred ref words moves WER by >0.1 pp.
TOLERANCE_PCT = 0.05

# DER re-score tolerance (percentage points). pyannote.metrics DER is
# deterministic given the same RTTMs, so 0.05 pp only absorbs float/round jitter.
DER_TOLERANCE_PCT = 0.05

__all__ = [
    "DER_TOLERANCE_PCT",
    "TOLERANCE_PCT",
    "corpus_cer_pct",
    "corpus_wer_pct",
    "find_der_model_dirs",
    "main",
    "normalize_text",
    "score_der_dir",
    "score_jsonl",
    "verify",
    "verify_der",
]

# --- artifact scanning + comparison -------------------------------------------

_PRED_RE = re.compile(r"^predictions_(?P<subset>.+)\.jsonl$")


def score_jsonl(path: Path) -> tuple[float, float, int]:
    """Read one predictions_*.jsonl -> (wer_pct, cer_pct, n_samples)."""
    refs: list[str] = []
    preds: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
                refs.append(row["reference"])
                preds.append(row["prediction"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{path}:{lineno}: bad record: {exc}") from exc
    return corpus_wer_pct(refs, preds), corpus_cer_pct(refs, preds), len(refs)


def find_model_dirs(artifacts_dir: Path) -> list[Path]:
    """Every dir containing at least one predictions_<subset>.jsonl."""
    dirs = {
        p.parent
        for p in artifacts_dir.rglob("predictions_*.jsonl")
        if _PRED_RE.match(p.name)
    }
    return sorted(dirs)


def verify(artifacts_dir: Path) -> tuple[bool, list[dict]]:
    """Re-score every model dir and compare to its expected.json.

    Returns ``(all_ok, rows)``. ``all_ok`` is False if any subset mismatches,
    an expected.json is missing/malformed, or NO model dirs were found at all.
    """
    model_dirs = find_model_dirs(artifacts_dir)
    rows: list[dict] = []
    if not model_dirs:
        return False, rows

    all_ok = True
    for model_dir in model_dirs:
        rel = model_dir.relative_to(artifacts_dir)
        expected_path = model_dir / "expected.json"
        expected: dict = {}
        if not expected_path.exists():
            rows.append(
                {"model": str(rel), "subset": "*", "status": "FAIL",
                 "detail": "missing expected.json"}
            )
            all_ok = False
        else:
            try:
                expected = json.loads(expected_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                rows.append(
                    {"model": str(rel), "subset": "*", "status": "FAIL",
                     "detail": f"malformed expected.json: {exc}"}
                )
                all_ok = False
                continue

        for pred_path in sorted(model_dir.glob("predictions_*.jsonl")):
            m = _PRED_RE.match(pred_path.name)
            assert m  # glob guarantees the pattern
            subset = m.group("subset")
            wer_pct, cer_pct, n = score_jsonl(pred_path)
            exp = expected.get(subset)
            if exp is None:
                rows.append(
                    {"model": str(rel), "subset": subset, "n": n,
                     "wer": wer_pct, "cer": cer_pct, "status": "FAIL",
                     "detail": "no expected entry for subset"}
                )
                all_ok = False
                continue
            wer_ok = abs(wer_pct - float(exp["wer_pct"])) <= TOLERANCE_PCT
            cer_ok = abs(cer_pct - float(exp["cer_pct"])) <= TOLERANCE_PCT
            ok = wer_ok and cer_ok
            all_ok = all_ok and ok
            rows.append(
                {"model": str(rel), "subset": subset, "n": n,
                 "wer": wer_pct, "cer": cer_pct,
                 "exp_wer": float(exp["wer_pct"]), "exp_cer": float(exp["cer_pct"]),
                 "status": "PASS" if ok else "FAIL",
                 "detail": "" if ok else
                 f"Δwer={wer_pct - float(exp['wer_pct']):+.3f} "
                 f"Δcer={cer_pct - float(exp['cer_pct']):+.3f} (tol {TOLERANCE_PCT})"}
            )
    return all_ok, rows


# --- Tier-1 DER re-score (recompute DER from committed gold+hyp RTTMs) --------
#
# Layout: artifacts/<run>/<model>/{gold,hyp}/<dataset>/<file>.rttm + expected.json
#   expected.json = {"<dataset>": {der_full, der_classic, miss, fa, conf}}  (percent)
# We re-load gold+hyp, recompute corpus DER at both collars with the SAME core
# (raven_eval_core.der — pyannote.metrics, no torch), and assert a match. This is
# the DER analogue of the WER re-score above; it needs no GPU and no gated model.
_DER_COLLARS = {"full": 0.0, "classic": 0.25}


def find_der_model_dirs(artifacts_dir: Path) -> list[Path]:
    """Every dir that holds a DER artifact (a ``gold/`` subtree + expected.json)."""
    dirs = {
        p.parent
        for p in artifacts_dir.rglob("expected.json")
        if (p.parent / "gold").is_dir() and any((p.parent / "gold").rglob("*.rttm"))
    }
    return sorted(dirs)


def score_der_dir(model_dir: Path) -> dict[str, dict[str, float]]:
    """Recompute per-dataset DER (percent) from the committed gold/hyp RTTMs.

    Returns ``{dataset: {der_full, der_classic, miss, fa, conf, n_files}}``. Each
    dataset aggregates every ``gold/<dataset>/<file>.rttm`` against its
    ``hyp/<dataset>/<file>.rttm`` via NIST-correct corpus accumulation.
    """
    gold_root = model_dir / "gold"
    hyp_root = model_dir / "hyp"
    out: dict[str, dict[str, float]] = {}
    for ds_dir in sorted(p for p in gold_root.iterdir() if p.is_dir()):
        dataset = ds_dir.name
        pairs = []
        for gold_rttm in sorted(ds_dir.glob("*.rttm")):
            hyp_rttm = hyp_root / dataset / gold_rttm.name
            if not hyp_rttm.exists():
                raise FileNotFoundError(
                    f"missing hypothesis RTTM for {dataset}/{gold_rttm.name} "
                    f"(expected {hyp_rttm})"
                )
            pairs.append((load_rttm(gold_rttm), load_rttm(hyp_rttm)))
        full = compute_der_corpus(pairs, collar=_DER_COLLARS["full"], skip_overlap=False)
        classic = compute_der_corpus(
            pairs, collar=_DER_COLLARS["classic"], skip_overlap=False
        )
        out[dataset] = {
            "der_full": full.der * 100.0,
            "der_classic": classic.der * 100.0,
            "miss": full.miss * 100.0,
            "fa": full.false_alarm * 100.0,
            "conf": full.confusion * 100.0,
            "n_files": float(len(pairs)),
        }
    return out


_DER_FIELDS = ("der_full", "der_classic", "miss", "fa", "conf")


def verify_der(artifacts_dir: Path) -> tuple[bool, list[dict]]:
    """Re-score every committed DER artifact and compare to its expected.json.

    Returns ``(all_ok, rows)``. ``all_ok`` is False on any field mismatch or a
    malformed/absent expected entry. An empty result (no DER artifacts) yields
    ``(True, [])`` — the *combined* emptiness guard lives in :func:`main`, so a
    repo with only WER artifacts is not failed for lacking DER ones.
    """
    model_dirs = find_der_model_dirs(artifacts_dir)
    rows: list[dict] = []
    all_ok = True
    for model_dir in model_dirs:
        rel = model_dir.relative_to(artifacts_dir)
        try:
            expected = json.loads(
                (model_dir / "expected.json").read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            rows.append({"model": str(rel), "dataset": "*", "status": "FAIL",
                         "detail": f"malformed expected.json: {exc}"})
            all_ok = False
            continue
        recomputed = score_der_dir(model_dir)
        for dataset, got in recomputed.items():
            exp = expected.get(dataset)
            if exp is None:
                rows.append({"model": str(rel), "dataset": dataset,
                             "n": int(got["n_files"]), "status": "FAIL",
                             "detail": "no expected entry for dataset"})
                all_ok = False
                continue
            deltas = {
                f: got[f] - float(exp[f]) for f in _DER_FIELDS if f in exp
            }
            missing = [f for f in _DER_FIELDS if f not in exp]
            ok = not missing and all(
                abs(d) <= DER_TOLERANCE_PCT for d in deltas.values()
            )
            all_ok = all_ok and ok
            worst = max(deltas.items(), key=lambda kv: abs(kv[1]), default=("", 0.0))
            detail = ""
            if missing:
                detail = f"missing expected fields: {missing}"
            elif not ok:
                detail = (f"Δ{worst[0]}={worst[1]:+.3f} (tol {DER_TOLERANCE_PCT})")
            rows.append({
                "model": str(rel), "dataset": dataset, "n": int(got["n_files"]),
                "der_full": got["der_full"], "exp_full": float(exp["der_full"]),
                "der_classic": got["der_classic"],
                "miss": got["miss"], "fa": got["fa"], "conf": got["conf"],
                "status": "PASS" if ok else "FAIL", "detail": detail,
            })
    return all_ok, rows


def _print_der_table(rows: list[dict]) -> None:
    hdr = (f"{'model':<28} {'dataset':<14} {'n':>4} {'der0%':>8} {'exp0%':>8} "
           f"{'der25%':>8} {'miss%':>7} {'fa%':>6} {'conf%':>7} {'status':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.get('model',''):<28} {r.get('dataset',''):<14} "
            f"{r.get('n',''):>4} "
            f"{r.get('der_full',float('nan')):>8.3f} "
            f"{r.get('exp_full',float('nan')):>8.3f} "
            f"{r.get('der_classic',float('nan')):>8.3f} "
            f"{r.get('miss',float('nan')):>7.3f} "
            f"{r.get('fa',float('nan')):>6.3f} "
            f"{r.get('conf',float('nan')):>7.3f} "
            f"{r['status']:>6}"
            + (f"  {r['detail']}" if r.get("detail") else "")
        )


def _print_table(rows: list[dict]) -> None:
    hdr = f"{'model':<28} {'subset':<14} {'n':>4} {'wer%':>8} {'exp%':>8} {'cer%':>8} {'status':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.get('model',''):<28} {r.get('subset',''):<14} "
            f"{r.get('n',''):>4} "
            f"{r.get('wer',float('nan')):>8.3f} "
            f"{r.get('exp_wer',float('nan')):>8.3f} "
            f"{r.get('cer',float('nan')):>8.3f} "
            f"{r['status']:>6}"
            + (f"  {r['detail']}" if r.get("detail") else "")
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="root to scan for predictions_*.jsonl (default: artifacts/)",
    )
    args = parser.parse_args(argv)

    if not args.artifacts_dir.exists():
        print(f"FAIL: artifacts dir does not exist: {args.artifacts_dir}",
              file=sys.stderr)
        return 2

    wer_ok, wer_rows = verify(args.artifacts_dir)
    der_ok, der_rows = verify_der(args.artifacts_dir)

    if not wer_rows and not der_rows:
        print(
            f"FAIL: no WER (predictions_*.jsonl) or DER (gold/hyp RTTMs) artifacts "
            f"found under {args.artifacts_dir} — nothing to verify (Tier-1 must not "
            "go green on an empty run).",
            file=sys.stderr,
        )
        return 2

    all_ok = True
    total = 0
    n_fail = 0
    if wer_rows:
        print("== WER ==")
        _print_table(wer_rows)
        all_ok = all_ok and wer_ok
        total += len(wer_rows)
        n_fail += sum(1 for r in wer_rows if r["status"] == "FAIL")
        print()
    if der_rows:
        print("== DER ==")
        _print_der_table(der_rows)
        all_ok = all_ok and der_ok
        total += len(der_rows)
        n_fail += sum(1 for r in der_rows if r["status"] == "FAIL")
        print()

    if all_ok:
        print(f"OK: {total} row(s) reproduced within ±{TOLERANCE_PCT} pp.")
        return 0
    print(f"FAIL: {n_fail}/{total} row(s) did not reproduce.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
