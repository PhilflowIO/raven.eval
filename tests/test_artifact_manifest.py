"""Tests for the artifact seal (scripts/manifest.py) and its gate in verify.py.

The committed manifest must describe the committed tree exactly, and each of the
three ways a tree can drift from it — a changed byte, a deleted file, an added
file — must stop ``verify.py`` before it re-scores anything.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


manifest = _load("manifest_under_test", "manifest.py")
verify = _load("verify", "verify.py")


def _artifact(root: Path) -> Path:
    """A minimal, truthful WER artifact: verify passes on it once sealed."""
    model = root / "run" / "model"
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
    return model


# ── the committed seal ───────────────────────────────────────────────────────


def test_committed_manifest_matches_the_committed_artifacts():
    """If this fails after an intended artifact change: `make manifest`, commit."""
    drift = manifest.check_manifest(ARTIFACTS)
    assert drift.ok, drift.report(ARTIFACTS)


def test_committed_manifest_is_canonical():
    """Sorted, sha256sum-formatted, self-excluding — byte-identical to a rewrite."""
    committed = (ARTIFACTS / manifest.MANIFEST_NAME).read_text(encoding="utf-8")
    assert committed == manifest.render(manifest.build_manifest(ARTIFACTS))
    assert f"  {manifest.MANIFEST_NAME}\n" not in committed


def test_digest_is_plain_sha256(tmp_path: Path):
    """The format promise: `sha256sum -c` reads it, no repo code needed."""
    (tmp_path / "a.txt").write_bytes(b"raven")
    manifest.write_manifest(tmp_path)
    expected = hashlib.sha256(b"raven").hexdigest()
    assert (tmp_path / manifest.MANIFEST_NAME).read_text() == f"{expected}  a.txt\n"


# ── each kind of drift fails, and fails verify ───────────────────────────────


def test_sealed_tree_passes_verify(tmp_path: Path):
    _artifact(tmp_path)
    manifest.write_manifest(tmp_path)
    assert manifest.check_manifest(tmp_path).ok
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 0


def test_modified_file_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A changed byte fails even where the number it feeds would re-score green."""
    model = _artifact(tmp_path)
    manifest.write_manifest(tmp_path)
    # Same scores, different bytes: verify's re-score alone would not notice.
    (model / "expected.json").write_text(
        json.dumps({"S": {"wer_pct": 0.0, "cer_pct": 0.0, "n_samples": 1,
                          "wer_ci_lo": 0.0, "wer_ci_hi": 0.0}}, indent=2),
        encoding="utf-8",
    )
    drift = manifest.check_manifest(tmp_path)
    assert drift.modified == ["run/model/expected.json"]
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "modified (sha256 differs): 1" in err
    assert "make manifest" in err


def test_missing_file_fails(tmp_path: Path):
    model = _artifact(tmp_path)
    manifest.write_manifest(tmp_path)
    (model / "predictions_S.jsonl").unlink()
    drift = manifest.check_manifest(tmp_path)
    assert drift.missing == ["run/model/predictions_S.jsonl"]
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 1


def test_unlisted_file_fails(tmp_path: Path):
    """Nothing is ignored by pattern: an added file is an unsealed file."""
    model = _artifact(tmp_path)
    manifest.write_manifest(tmp_path)
    (model / "notes.txt").write_text("added after the release\n", encoding="utf-8")
    drift = manifest.check_manifest(tmp_path)
    assert drift.unlisted == ["run/model/notes.txt"]
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 1


def test_missing_manifest_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _artifact(tmp_path)
    assert manifest.check_manifest(tmp_path).manifest_missing
    assert verify.main(["--artifacts-dir", str(tmp_path)]) == 1
    assert "is not sealed" in capsys.readouterr().err


def test_malformed_manifest_line_raises(tmp_path: Path):
    (tmp_path / manifest.MANIFEST_NAME).write_text("not-a-digest a.txt\n")
    with pytest.raises(ValueError, match="not a sha256sum line"):
        manifest.check_manifest(tmp_path)


def test_check_cli_reports_drift(tmp_path: Path):
    model = _artifact(tmp_path)
    assert manifest.main(["--artifacts-dir", str(tmp_path)]) == 0
    assert manifest.main(["--artifacts-dir", str(tmp_path), "--check"]) == 0
    (model / "expected.json").write_text("{}", encoding="utf-8")
    assert manifest.main(["--artifacts-dir", str(tmp_path), "--check"]) == 1
