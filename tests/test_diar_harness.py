"""raven_diar harness tests — score, config, converters, promote (no GPU/network).

The diarizer adapter itself needs a GPU + the gated model, so it is not exercised
here; everything downstream of "we have a hypothesis RTTM" is, on synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raven_diar.config import COLLARS, DER_DATASETS, KNOWN_DIARIZERS
from raven_diar.datasets.callhome_de import segments_from_row
from raven_diar.promote import _safe_run_name, promote
from raven_diar.score import score_segment_pairs
from raven_eval_core.der import to_rttm

# ── score.py ─────────────────────────────────────────────────────────────────


def test_score_matches_hand_computed_der():
    # m1: conf 10 / total 20; m2: miss 2 / total 10; m3: fa 5 / total 10.
    pairs = [
        ([(0, 10, "A"), (10, 20, "B")], [(0, 20, "Z")]),
        ([(0, 10, "A")], [(0, 8, "A")]),
        ([(0, 10, "A")], [(0, 10, "A"), (10, 15, "B")]),
    ]
    s = score_segment_pairs("demo", pairs)
    assert s.n_files == 3
    assert s.der_full == pytest.approx(42.5, abs=1e-3)   # 17/40
    assert s.miss == pytest.approx(5.0, abs=1e-3)        # 2/40
    assert s.fa == pytest.approx(12.5, abs=1e-3)         # 5/40
    assert s.conf == pytest.approx(25.0, abs=1e-3)       # 10/40
    assert s.miss + s.fa + s.conf == pytest.approx(s.der_full, abs=1e-6)


def test_score_two_collars_differ():
    pairs = [([(0, 10, "A")], [(0, 8, "A")])]
    s = score_segment_pairs("demo", pairs)
    assert s.der_full != pytest.approx(s.der_classic)  # collar removes a window


def test_expected_entry_shape():
    s = score_segment_pairs("demo", [([(0, 10, "A")], [(0, 20, "B")])])
    entry = s.expected_entry()
    assert set(entry) == {"der_full", "der_classic", "miss", "fa", "conf"}


# ── config contract ──────────────────────────────────────────────────────────


def test_collars_match_public_contract():
    assert COLLARS["full"] == {"collar": 0.0, "skip_overlap": False}
    assert COLLARS["classic"] == {"collar": 0.25, "skip_overlap": False}


def test_datasets_and_diarizers_registered():
    assert set(DER_DATASETS) == {
        "voxconverse", "voxconverse-test", "callhome-de", "ami",
    }
    # VoxConverse ships two gold splits; each id must select exactly one.
    assert DER_DATASETS["voxconverse"].split == "dev"
    assert DER_DATASETS["voxconverse-test"].split == "test"
    assert "pyannote-community-1" in KNOWN_DIARIZERS
    assert (
        KNOWN_DIARIZERS["pyannote-community-1"].model_id
        == "pyannote/speaker-diarization-community-1"
    )


# ── CALLHOME-de converter (the German anchor) ────────────────────────────────


def test_callhome_converter_zips_parallel_lists():
    row = {
        "timestamps_start": [0.0, 3.0, 6.5],
        "timestamps_end": [2.5, 6.0, 9.0],
        "speakers": ["spk_A", "spk_B", "spk_A"],
    }
    segs = segments_from_row(row)
    assert segs == [(0.0, 2.5, "spk_A"), (3.0, 6.0, "spk_B"), (6.5, 9.0, "spk_A")]


def test_callhome_converter_accepts_aliases_and_drops_empty():
    row = {"starts": [0.0, 5.0], "ends": [5.0, 5.0], "speaker": ["A", "B"]}
    # second turn is zero-length -> dropped
    assert segments_from_row(row) == [(0.0, 5.0, "A")]


def test_callhome_converter_rejects_mismatched_lists():
    with pytest.raises(ValueError):
        segments_from_row({"timestamps_start": [0.0, 1.0],
                           "timestamps_end": [1.0], "speakers": ["A"]})


def test_callhome_converter_rejects_missing_fields():
    with pytest.raises(ValueError):
        segments_from_row({"audio": {}})


# ── promote.py ───────────────────────────────────────────────────────────────


def test_safe_run_name_blocks_traversal():
    assert _safe_run_name("../../etc") == "etc"
    assert _safe_run_name("/abs/path") == "abs_path"
    assert _safe_run_name("..") == "run"
    assert _safe_run_name("2026-07-30") == "2026-07-30"


def test_promote_builds_tier1_artifact(tmp_path: Path):
    # Fake a completed run dir with gold/, hyp/ and a summary.json.
    run = tmp_path / "results" / "pyannote-community-1"
    (run / "gold" / "voxconverse").mkdir(parents=True)
    (run / "hyp" / "voxconverse").mkdir(parents=True)
    (run / "gold" / "voxconverse" / "f1.rttm").write_text(
        to_rttm([(0, 10, "A"), (10, 20, "B")], file_id="f1")
    )
    (run / "hyp" / "voxconverse" / "f1.rttm").write_text(
        to_rttm([(0, 20, "Z")], file_id="f1")
    )
    s = score_segment_pairs("voxconverse", [([(0, 10, "A"), (10, 20, "B")], [(0, 20, "Z")])])
    (run / "summary.json").write_text(
        json.dumps({"model_id": "m", "results": {"voxconverse": s.as_dict()}})
    )

    dest = promote(run, tmp_path / "artifacts", "2026-07-30")
    expected = json.loads((dest / "expected.json").read_text())
    assert set(expected["voxconverse"]) == {"der_full", "der_classic", "miss", "fa", "conf"}
    assert (dest / "gold" / "voxconverse" / "f1.rttm").exists()
    assert (dest / "hyp" / "voxconverse" / "f1.rttm").exists()

    # And the promoted artifact must re-score green through the Tier-1 verifier.
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "verify", Path(__file__).resolve().parent.parent / "scripts" / "verify.py"
    )
    verify = importlib.util.module_from_spec(spec)
    sys.modules["verify"] = verify
    spec.loader.exec_module(verify)
    all_ok, rows = verify.verify_der(tmp_path / "artifacts")
    assert all_ok and rows


def test_promote_rejects_empty_run(tmp_path: Path):
    run = tmp_path / "results" / "x"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"results": {}}))
    with pytest.raises(FileNotFoundError):
        promote(run, tmp_path / "artifacts", "r")
