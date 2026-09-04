"""Tests for the Tier-1 DER re-scorer + the raven_diar harness (no GPU/network).

Everything here runs on tiny synthetic RTTMs whose DER is hand-computable, so the
DER machinery is exercised end-to-end without a diarizer, a GPU, or the gated
pyannote model. Mirrors tests/test_verify.py for the WER path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from raven_diar.score import DerScore
from raven_eval_core.der import to_rttm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_ARTIFACTS = REPO_ROOT / "artifacts"
DEMO_DER_DIR = DEMO_ARTIFACTS / "_demo_der" / "demo-diarizer"


def _load_verify():
    spec = importlib.util.spec_from_file_location(
        "verify", REPO_ROOT / "scripts" / "verify.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


verify = _load_verify()


def _write_pair(model_dir: Path, dataset: str, fid: str, gold, hyp) -> None:
    g = model_dir / "gold" / dataset
    h = model_dir / "hyp" / dataset
    g.mkdir(parents=True, exist_ok=True)
    h.mkdir(parents=True, exist_ok=True)
    (g / f"{fid}.rttm").write_text(to_rttm(gold, file_id=fid))
    (h / f"{fid}.rttm").write_text(to_rttm(hyp, file_id=fid))


# ── committed demo fixture ───────────────────────────────────────────────────


def test_demo_der_fixture_passes():
    """The committed _demo_der fixture must re-score to its own expected.json."""
    all_ok, rows = verify.verify_der(DEMO_ARTIFACTS)
    assert rows, "demo DER fixture must produce at least one row"
    assert all_ok, f"demo DER fixture did not reproduce: {rows}"
    assert all(r["status"] == "PASS" for r in rows)


def test_demo_der_expected_is_truthful():
    """expected.json must equal what the scorer computes (never hand-typed)."""
    expected = json.loads((DEMO_DER_DIR / "expected.json").read_text())
    recomputed = verify.score_der_dir(DEMO_DER_DIR)
    for dataset, exp in expected.items():
        got = recomputed[dataset]
        for field in DerScore.EXPECTED_FIELDS:
            assert abs(got[field] - exp[field]) <= verify.DER_TOLERANCE_PCT


def test_demo_der_components_sum_to_full():
    """miss + fa + conf == der_full by construction (collar-0 decomposition)."""
    got = verify.score_der_dir(DEMO_DER_DIR)["demo-set"]
    assert got["miss"] + got["fa"] + got["conf"] == pytest.approx(
        got["der_full"], abs=1e-6
    )


def test_main_runs_both_wer_and_der_green():
    """`verify.py` over the real artifacts dir re-scores WER + DER, exits 0."""
    assert verify.main(["--artifacts-dir", str(DEMO_ARTIFACTS)]) == 0


# ── failure / emptiness guards ───────────────────────────────────────────────


def test_wrong_der_expected_fails(tmp_path: Path):
    """A deliberately wrong expected.json must make verify_der FAIL."""
    model = tmp_path / "run" / "diarizer"
    _write_pair(model, "d", "f1", [(0, 10, "A")], [(0, 8, "A")])  # miss 2/10=20%
    (model / "expected.json").write_text(
        json.dumps({"d": {f: 1.0 for f in DerScore.EXPECTED_FIELDS}})
    )
    all_ok, rows = verify.verify_der(tmp_path)
    assert rows
    assert not all_ok
    assert any(r["status"] == "FAIL" for r in rows)
    # combined CLI must return nonzero (1) — there is a real mismatch.
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 1


def test_correct_der_expected_passes(tmp_path: Path):
    """A truthful expected.json (from the scorer) re-scores green through main."""
    model = tmp_path / "run" / "diarizer"
    _write_pair(model, "d", "f1", [(0, 10, "A"), (10, 20, "B")], [(0, 20, "Z")])
    got = verify.score_der_dir(model)["d"]
    (model / "expected.json").write_text(
        json.dumps({"d": {k: round(got[k], 4)
                          for k in DerScore.EXPECTED_FIELDS}})
    )
    all_ok, rows = verify.verify_der(tmp_path)
    assert all_ok and rows
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 0


def test_missing_hyp_rttm_raises(tmp_path: Path):
    """A gold RTTM with no matching hyp must raise, not silently score."""
    model = tmp_path / "run" / "diarizer"
    (model / "gold" / "d").mkdir(parents=True)
    (model / "gold" / "d" / "f1.rttm").write_text(to_rttm([(0, 5, "A")], file_id="f1"))
    (model / "hyp" / "d").mkdir(parents=True)
    (model / "expected.json").write_text(json.dumps({"d": {}}))
    with pytest.raises(FileNotFoundError):
        verify.score_der_dir(model)


def test_empty_artifacts_still_fails(tmp_path: Path):
    """No WER and no DER artifacts → exit 2 (unchanged emptiness guard)."""
    assert verify.verify_der(tmp_path) == (True, [])
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 2
