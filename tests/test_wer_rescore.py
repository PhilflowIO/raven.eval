"""``make rescore`` for WER artifacts (raven_asr.rescore).

It may add a field an older artifact lacks; it may never change one it has.
"""

from __future__ import annotations

import json
from pathlib import Path

from raven_asr import rescore

_ROWS = [
    {"sample_id": "s0", "reference": "es kamen 23 Leute", "prediction": "es kamen 22 Leute"},
    {"sample_id": "s1", "reference": "um 10 Uhr", "prediction": "um 10 Uhr"},
]


def _artifact(tmp_path: Path, expected: dict) -> Path:
    model = tmp_path / "run" / "model"
    model.mkdir(parents=True)
    (model / "predictions_S.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in _ROWS), encoding="utf-8"
    )
    (model / "expected.json").write_text(json.dumps({"S": expected}), encoding="utf-8")
    return model


def _committed(model: Path) -> dict:
    return json.loads((model / "expected.json").read_text())["S"]


def test_backfills_the_entity_score_and_keeps_everything_else(tmp_path: Path):
    model = _artifact(tmp_path, {"wer_pct": 14.2857, "cer_pct": 1.0, "n_samples": 2})
    added, conflicts = rescore.backfill(model)
    assert not conflicts
    exp = _committed(model)
    assert exp["entity_hit_rate_pct"] == 50.0
    assert (exp["n_entities"], exp["n_utterances_with_entities"]) == (2, 2)
    assert "S.entity_ci_lo" in added and "S.wer_ci_lo" in added
    # untouched: what was committed stays committed, byte for byte
    assert (exp["wer_pct"], exp["cer_pct"], exp["n_samples"]) == (14.2857, 1.0, 2)


def test_a_second_run_adds_nothing(tmp_path: Path):
    model = _artifact(tmp_path, {"wer_pct": 14.2857, "cer_pct": 1.0, "n_samples": 2})
    rescore.backfill(model)
    before = (model / "expected.json").read_bytes()
    assert rescore.backfill(model) == ([], [])
    assert (model / "expected.json").read_bytes() == before


def test_refuses_to_overwrite_a_disagreeing_value(tmp_path: Path):
    """A wrong committed number is verify's to report, not rescore's to fix."""
    model = _artifact(tmp_path, {"wer_pct": 14.2857, "cer_pct": 1.0, "n_samples": 2,
                                 "entity_hit_rate_pct": 100.0})
    before = (model / "expected.json").read_bytes()
    _, conflicts = rescore.backfill(model)
    assert any("entity_hit_rate_pct" in c for c in conflicts)
    assert (model / "expected.json").read_bytes() == before


def test_counts_must_agree_exactly(tmp_path: Path):
    model = _artifact(tmp_path, {"wer_pct": 14.2857, "cer_pct": 1.0, "n_samples": 2,
                                 "n_entities": 3})
    _, conflicts = rescore.backfill(model)
    assert any("n_entities" in c for c in conflicts)


def test_dry_run_writes_nothing(tmp_path: Path):
    model = _artifact(tmp_path, {"wer_pct": 14.2857, "cer_pct": 1.0, "n_samples": 2})
    before = (model / "expected.json").read_bytes()
    added, conflicts = rescore.backfill(model, dry_run=True)
    assert added and not conflicts
    assert (model / "expected.json").read_bytes() == before
