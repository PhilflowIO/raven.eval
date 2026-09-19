"""Tests for the Tier-1 re-scorer (scripts/verify.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_ARTIFACTS = REPO_ROOT / "artifacts"


def _load_verify():
    """Import scripts/verify.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "verify", REPO_ROOT / "scripts" / "verify.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


verify = _load_verify()


def test_demo_fixture_passes():
    """The committed _demo fixture must re-score to its own expected.json."""
    all_ok, rows = verify.verify(DEMO_ARTIFACTS)
    assert rows, "demo fixture must produce at least one row"
    assert all_ok, f"demo fixture did not reproduce: {rows}"
    assert all(r["status"] == "PASS" for r in rows)
    # exit-success through the CLI entrypoint too
    assert verify.main(["--artifacts-dir", str(DEMO_ARTIFACTS)]) == 0


def test_demo_expected_is_truthful():
    """expected.json must equal what the scorer actually computes (no guessing)."""
    pred = DEMO_ARTIFACTS / "_demo" / "demo-model" / "predictions_Demo-DE.jsonl"
    exp = json.loads(
        (DEMO_ARTIFACTS / "_demo" / "demo-model" / "expected.json").read_text()
    )
    wer_pct, cer_pct, _ = verify.score_jsonl(pred)
    assert abs(wer_pct - exp["Demo-DE"]["wer_pct"]) <= verify.TOLERANCE_PCT
    assert abs(cer_pct - exp["Demo-DE"]["cer_pct"]) <= verify.TOLERANCE_PCT


def test_wrong_expected_fails(tmp_path: Path):
    """A deliberately wrong expected.json must make verify FAIL (nonzero exit)."""
    model = tmp_path / "run" / "model"
    model.mkdir(parents=True)
    (model / "predictions_S.jsonl").write_text(
        json.dumps({"reference": "hallo welt", "prediction": "hallo welt",
                    "latency_s": 0.1}) + "\n",
        encoding="utf-8",
    )
    # Perfect transcript -> real WER 0.0; assert something absurd instead.
    (model / "expected.json").write_text(
        json.dumps({"S": {"wer_pct": 42.0, "cer_pct": 42.0, "n_samples": 1}}), encoding="utf-8"
    )
    all_ok, rows = verify.verify(tmp_path)
    assert rows
    assert not all_ok
    assert any(r["status"] == "FAIL" for r in rows)
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 1


# ── Coverage: the number must stand for the whole subset ─────────────────────


def _artifact(tmp_path: Path, lines: list[dict], expected: dict) -> Path:
    model = tmp_path / "run" / "model"
    model.mkdir(parents=True)
    (model / "predictions_S.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8"
    )
    (model / "expected.json").write_text(json.dumps({"S": expected}), encoding="utf-8")
    return model


_OK = {"reference": "hallo welt", "prediction": "hallo welt", "latency_s": 0.1}
_FAILED = {"reference": "guten tag", "prediction": None, "latency_s": None,
           "error": "ReadTimeout: timed out"}


def test_a_failed_request_line_fails_verify(tmp_path: Path):
    """The published WER over the clips that worked is still 0.0 — and still wrong.

    Scored around the failure, this artifact re-scores perfectly. That is exactly
    why coverage is checked before the number.
    """
    _artifact(tmp_path, [_OK, _FAILED], {"wer_pct": 0.0, "cer_pct": 0.0, "n_samples": 2})
    all_ok, rows = verify.verify(tmp_path)
    assert not all_ok
    assert any("failed request" in r["detail"] for r in rows)
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 1


def test_a_lost_prediction_line_fails_verify(tmp_path: Path):
    """n_samples binds the file to its length; a dropped line cannot hide."""
    _artifact(tmp_path, [_OK], {"wer_pct": 0.0, "cer_pct": 0.0, "n_samples": 2})
    all_ok, rows = verify.verify(tmp_path)
    assert not all_ok
    assert any("n_samples=2" in r["detail"] for r in rows)


def test_expected_without_n_samples_fails(tmp_path: Path):
    _artifact(tmp_path, [_OK], {"wer_pct": 0.0, "cer_pct": 0.0})
    all_ok, rows = verify.verify(tmp_path)
    assert not all_ok
    assert any("does not commit n_samples" in r["detail"] for r in rows)


def test_a_hand_edited_interval_fails_verify(tmp_path: Path):
    """The committed interval is re-derived like the point, not trusted."""
    good = verify.verify(DEMO_ARTIFACTS)[1]
    assert all("wer_ci" in r for r in good), "every demo row carries its interval"
    other = {"reference": "guten tag", "prediction": "guten abend", "latency_s": 0.1}
    _artifact(tmp_path, [_OK, other, _OK, other],
              {"wer_pct": 25.0, "cer_pct": 0.0, "n_samples": 4,
               "wer_ci_lo": 0.0, "wer_ci_hi": 1.0})
    all_ok, rows = verify.verify(tmp_path)
    assert not all_ok
    assert any("WER interval" in r["detail"] for r in rows)


def test_read_pairs_refuses_to_score_around_a_failure(tmp_path: Path):
    model = _artifact(tmp_path, [_OK, _FAILED], {})
    with pytest.raises(ValueError, match="failed request"):
        verify.read_pairs(model / "predictions_S.jsonl")


def test_empty_artifacts_fails(tmp_path: Path):
    """An artifacts dir with no predictions must FAIL, not silently pass."""
    all_ok, rows = verify.verify(tmp_path)
    assert not all_ok
    assert rows == []
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 2


def test_missing_expected_fails(tmp_path: Path):
    """predictions present but no expected.json -> FAIL."""
    model = tmp_path / "run" / "model"
    model.mkdir(parents=True)
    (model / "predictions_S.jsonl").write_text(
        json.dumps({"reference": "a b c", "prediction": "a b c",
                    "latency_s": 0.1}) + "\n",
        encoding="utf-8",
    )
    all_ok, rows = verify.verify(tmp_path)
    assert not all_ok
    assert any("missing expected.json" in r.get("detail", "") for r in rows)


@pytest.mark.parametrize(
    ("ref", "hyp", "expect_zero"),
    [
        ("dreiundzwanzig kunden", "23 kunden", True),   # alpha2digit collapses
        ("die strasse", "die straße", True),            # ß -> ss both sides
        ("hallo welt", "hallo mars", False),            # real substitution
    ],
)
def test_flozi_normalization_semantics(ref, hyp, expect_zero):
    """Spot-check that the mirrored flozi pipeline behaves as documented."""
    w = verify.corpus_wer_pct([ref], [hyp])
    assert (w == 0.0) is expect_zero


# ── BLEU: the same artifact, one optional key ────────────────────────────────


def test_demo_bleu_fixture_passes():
    """The committed BLEU fixture must re-score to its own expected.json."""
    all_ok, rows = verify.verify(DEMO_ARTIFACTS)
    bleu_rows = [r for r in rows if "bleu" in r]
    assert bleu_rows, "artifacts/_demo_bleu must produce at least one BLEU row"
    assert all(r["status"] == "PASS" for r in bleu_rows)
    assert all_ok


def test_demo_bleu_expected_is_truthful():
    """expected.json's bleu must equal what the scorer computes (no guessing)."""
    model = DEMO_ARTIFACTS / "_demo_bleu" / "demo-model"
    exp = json.loads((model / "expected.json").read_text())["Demo-CH-DE"]
    got = verify.score_jsonl_bleu(model / "predictions_Demo-CH-DE.jsonl")
    assert abs(got - exp["bleu"]) <= verify.BLEU_TOLERANCE
    # The committed signature must be the one this environment actually produces —
    # a sacrebleu upgrade that changes tokenization must be visible, not silent.
    assert exp["bleu_signature"] == verify.bleu_signature()


def test_bleu_fixture_is_the_case_that_motivates_the_metric():
    """WER reads the dialect fixture as broken; BLEU reads it as largely correct."""
    model = DEMO_ARTIFACTS / "_demo_bleu" / "demo-model"
    pred = model / "predictions_Demo-CH-DE.jsonl"
    wer_pct, _, _ = verify.score_jsonl(pred)
    bleu = verify.score_jsonl_bleu(pred)
    assert wer_pct > 10.0
    assert bleu > 60.0


def test_wrong_expected_bleu_fails(tmp_path: Path):
    """A deliberately wrong bleu in expected.json must make verify FAIL."""
    model = tmp_path / "run" / "model"
    model.mkdir(parents=True)
    (model / "predictions_S.jsonl").write_text(
        json.dumps({"reference": "hallo welt", "prediction": "hallo welt",
                    "latency_s": 0.1}) + "\n",
        encoding="utf-8",
    )
    # WER/CER are correct; only the BLEU is absurd -> the run must still fail.
    (model / "expected.json").write_text(
        json.dumps({"S": {"wer_pct": 0.0, "cer_pct": 0.0, "n_samples": 1, "bleu": 99.0}}),
        encoding="utf-8",
    )
    all_ok, rows = verify.verify(tmp_path)
    assert not all_ok
    assert any("Δbleu=" in r.get("detail", "") for r in rows)
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 1


def test_subset_without_bleu_key_is_not_scored_for_bleu(tmp_path: Path):
    """BLEU is opt-in: a plain transcription subset must not grow a BLEU row."""
    model = tmp_path / "run" / "model"
    model.mkdir(parents=True)
    (model / "predictions_S.jsonl").write_text(
        json.dumps({"reference": "hallo welt", "prediction": "hallo welt",
                    "latency_s": 0.1}) + "\n",
        encoding="utf-8",
    )
    (model / "expected.json").write_text(
        json.dumps({"S": {"wer_pct": 0.0, "cer_pct": 0.0, "n_samples": 1,
                          "wer_ci_lo": 0.0, "wer_ci_hi": 0.0}}),
        encoding="utf-8",
    )
    all_ok, rows = verify.verify(tmp_path)
    assert all_ok
    assert rows and all("bleu" not in r for r in rows)


def test_bleu_scoring_is_offline():
    """Tier-1 promise: the BLEU path opens no socket.

    sacrebleu can download WMT test sets on demand; our scoring path must never
    reach that code. Any socket creation during a re-score fails this test.
    """
    import socket

    model = DEMO_ARTIFACTS / "_demo_bleu" / "demo-model"
    real = socket.socket

    def _forbidden(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError("BLEU re-score attempted network access")

    socket.socket = _forbidden  # type: ignore[assignment]
    try:
        verify.score_jsonl_bleu(model / "predictions_Demo-CH-DE.jsonl")
    finally:
        socket.socket = real  # type: ignore[assignment]
