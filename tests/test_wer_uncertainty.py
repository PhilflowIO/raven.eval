"""WER counting and its interval: per-utterance alignment, exact decomposition,
and a paired comparison that refuses to compare different data.

Each test guards a sentence BENCHMARKS.md now leans on — that a WER is counted
clip by clip, that its printed interval is around the printed number and not a
cousin of it, and that "2.0 is ahead of 1.0" is only said where the paired
interval excludes zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raven_asr.analysis import paired_wer_delta, wer_interval
from raven_eval_core.bootstrap import (
    UnpairedUnitsError,
    bootstrap_ratio_ci,
    pair_by_id,
    ratio_pct,
)
from raven_eval_core.flozi_wer import corpus_wer_pct, utterance_word_errors
from raven_eval_core.wer import corpus_wer_strict_de_pct, utterance_strict_de_units

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"


def test_an_error_does_not_cancel_across_a_clip_boundary():
    """The case that changed the published count on 2026-09-19.

    "heute" is missing from the end of clip 1 and surplus at the start of clip 2.
    Concatenated, the two errors cancel to zero; the clips are separate
    recordings, so they are two errors.
    """
    refs = ["wir gehen heute", "morgen regnet es"]
    hyps = ["wir gehen", "heute morgen regnet es"]
    assert corpus_wer_pct(refs, hyps) == pytest.approx(2 / 6 * 100)
    assert utterance_word_errors(refs, hyps) == [(1, 3), (1, 3)]


def _committed_pairs() -> list[tuple[list[str], list[str]]]:
    out = []
    for p in sorted(ARTIFACTS.rglob("predictions_*.jsonl")):
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
        out.append(([r["reference"] for r in rows], [r["prediction"] for r in rows]))
    return out


def test_published_wer_decomposes_exactly_per_utterance():
    """Σedits/Σref_words over the units IS the published number, on every artifact."""
    pairs = _committed_pairs()
    assert pairs
    for refs, hyps in pairs:
        assert ratio_pct(utterance_word_errors(refs, hyps)) == corpus_wer_pct(refs, hyps)


def test_strict_de_decomposes_exactly_per_utterance():
    for refs, hyps in _committed_pairs():
        assert ratio_pct(utterance_strict_de_units(refs, hyps)) == pytest.approx(
            corpus_wer_strict_de_pct(refs, hyps), abs=1e-9
        )


def test_interval_brackets_the_point_and_is_seeded():
    units = [(float(i % 4), 10.0) for i in range(40)]
    a = bootstrap_ratio_ci(units, resamples=300, seed=1, confidence=0.95)
    b = bootstrap_ratio_ci(units, resamples=300, seed=1, confidence=0.95)
    assert a.lo <= a.point <= a.hi
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_pairing_refuses_a_unit_only_one_side_has():
    with pytest.raises(UnpairedUnitsError, match="only in B"):
        pair_by_id({"x": 1}, {"x": 1, "y": 2})


def _pred(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_paired_delta_of_a_run_against_itself_is_zero(tmp_path: Path):
    rows = [{"sample_id": f"s{i}", "reference": "a b c d", "prediction": "a b x d"
             if i % 2 else "a b c d"} for i in range(10)]
    p = _pred(tmp_path, "a", rows)
    d = paired_wer_delta(p, p, resamples=200)
    assert d.point == 0.0 and d.lo <= 0.0 <= d.hi


def test_paired_delta_refuses_the_same_id_on_different_references(tmp_path: Path):
    """Two runs that read different data under one id are not a model comparison."""
    a = _pred(tmp_path, "a", [{"sample_id": "s0", "reference": "a b", "prediction": "a b"},
                              {"sample_id": "s1", "reference": "c d", "prediction": "c d"}])
    b = _pred(tmp_path, "b", [{"sample_id": "s0", "reference": "a b", "prediction": "a b"},
                              {"sample_id": "s1", "reference": "c e", "prediction": "c e"}])
    with pytest.raises(ValueError, match="different references"):
        paired_wer_delta(a, b, resamples=50)


def test_paired_delta_needs_sample_ids(tmp_path: Path):
    a = _pred(tmp_path, "a", [{"reference": "a b", "prediction": "a b"}])
    with pytest.raises(ValueError, match="sample_id"):
        paired_wer_delta(a, a, resamples=50)


def test_no_interval_around_a_run_with_failures(tmp_path: Path):
    p = _pred(tmp_path, "a", [{"sample_id": "s0", "reference": "a b", "prediction": None,
                               "error": "ReadTimeout"}])
    with pytest.raises(ValueError, match="failed request"):
        wer_interval(p, resamples=50)
