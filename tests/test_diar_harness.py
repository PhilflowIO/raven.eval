"""raven_diar harness tests — score, config, converters, promote (no GPU/network).

The diarizer adapter itself needs a GPU + the gated model, so it is not exercised
here; everything downstream of "we have a hypothesis RTTM" is, on synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raven_diar.config import (
    AGGREGATION_ALSO_REPORTED,
    AGGREGATION_PRIMARY,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COLLARS,
    DER_DATASETS,
    KNOWN_DIARIZERS,
)
from raven_diar.datasets.ami import AMILoader, audio_url
from raven_diar.datasets.callhome_de import segments_from_row
from raven_diar.promote import _safe_run_name, promote
from raven_diar.score import DerScore, score_segment_pairs
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
    """expected.json carries EVERY published scalar, both collars included.

    The field list is read from the scorer, not repeated here: a quantity that
    reaches BENCHMARKS.md must reach expected.json too, or it is published
    without a Tier-1 re-score behind it. Pinned literally is only the invariant
    that matters — both collars carry their own decomposition, and both
    aggregation conventions are present.
    """
    s = score_segment_pairs("demo", [([(0, 10, "A")], [(0, 20, "B")])])
    entry = s.expected_entry()
    assert set(entry) == set(DerScore.EXPECTED_FIELDS)
    assert {"der_full", "miss", "fa", "conf"} <= set(entry)
    assert {"der_classic", "miss_classic", "fa_classic", "conf_classic"} <= set(entry)
    assert {"der_full_filemean", "der_classic_filemean"} <= set(entry)


# ── config contract ──────────────────────────────────────────────────────────


def _public_contract() -> dict:
    """The committed public scoring contract, read from disk."""
    import yaml

    return yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "benchmark.config.yaml").read_text()
    )


def test_collars_match_public_contract():
    """The collars in code must equal the ones the contract FILE publishes.

    Read, not restated. A test that asserts 0.25 against a literal 0.25 written
    beside it passes just as happily when benchmark.config.yaml says 0.3 — and
    then every published DER is produced under a rule the committed contract
    misstates, which is the one failure this repo exists to make impossible. The
    turn-folding threshold below has always been checked this way; the collars,
    the more score-moving of the two, were not.
    """
    variants = _public_contract()["der"]["variants"]
    published = {
        v["name"]: {"collar": float(v["collar"]),
                    "skip_overlap": bool(v["skip_overlap"])}
        for v in variants
    }
    assert published == COLLARS, (
        "benchmark.config.yaml and raven_diar.config.COLLARS disagree about the "
        "collars every published DER is computed under."
    )


def test_aggregation_and_uncertainty_match_public_contract():
    """The aggregation convention and the bootstrap settings are contract too.

    Both move a published number: the two aggregations are 0.334 pp apart on the
    German CALLHOME row, and an unseeded interval is not reproducible at all.
    """
    der = _public_contract()["der"]
    assert der["aggregation"]["primary"] == AGGREGATION_PRIMARY
    assert tuple(der["aggregation"]["also_reported"]) == AGGREGATION_ALSO_REPORTED
    assert der["uncertainty"]["resamples"] == BOOTSTRAP_RESAMPLES
    assert der["uncertainty"]["seed"] == BOOTSTRAP_SEED
    assert der["uncertainty"]["confidence"] == BOOTSTRAP_CONFIDENCE


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


#: Vendor aliases that mean "whatever is newest" — the exact thing a published
#: number may not hang on, because it moves without us doing anything.
FLOATING_ALIASES = frozenset({"", "latest", "main", "master", "head", "stable", "default"})


def test_dataset_revisions_are_pinned_commits():
    """A published number must hang on an immutable revision, never a branch."""
    import re

    for spec in DER_DATASETS.values():
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), (
            f"{spec}: revision {spec.revision!r} is not a full commit hash"
        )


def test_diarizer_revisions_are_commits_unless_the_vendor_offers_none():
    """Same invariant, one declared exception — hosted APIs publish no commit.

    A hosted diarizer (Deepgram, AssemblyAI, …) exposes only a model *alias*,
    which the vendor can re-train behind. That weakness must be declared per
    spec (`revision_kind="vendor-alias"`), never smuggled in by loosening the
    regex for everyone: a local-weights spec that silently stopped pinning a
    commit would then pass unnoticed.
    """
    import re

    for key, spec in KNOWN_DIARIZERS.items():
        if spec.revision_kind == "commit":
            assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), (
                f"{key}: revision {spec.revision!r} is not a full commit hash"
            )
        elif spec.revision_kind == "vendor-alias":
            assert spec.revision, f"{key}: a vendor alias must still be explicit"
            assert not re.fullmatch(r"[0-9a-f]{40}", spec.revision), (
                f"{key}: revision looks like a commit — declare revision_kind"
                f"='commit' so the strict rule applies"
            )
            # "whatever is newest" is the one alias a published number may not
            # hang on: it moves without us doing anything.
            assert spec.revision.strip().lower() not in FLOATING_ALIASES, (
                f"{key}: revision {spec.revision!r} is a floating vendor alias"
            )
        else:
            raise AssertionError(
                f"{key}: unknown revision_kind {spec.revision_kind!r} — expected "
                f"'commit' or 'vendor-alias'"
            )


def test_where_it_runs_and_how_it_is_pinned_stay_consistent():
    """`hosted` and `revision_kind` are different questions with one sane pairing.

    A model whose weights we fetch ourselves always has a commit to pin, so a
    local spec declaring "vendor-alias" is a spec that quietly stopped pinning.
    The reverse is deliberately NOT asserted: a hosted API that ever offers a
    real immutable version should be allowed to claim it.
    """
    for key, spec in KNOWN_DIARIZERS.items():
        if not spec.hosted:
            assert spec.revision_kind == "commit", (
                f"{key}: runs from weights we fetch, so it has a commit to pin — "
                f"'vendor-alias' here would be an unpinned local model"
            )


# ── AMI loader (gold from the pinned setup clone, audio from the mirror) ──────


def _fake_setup_clone(base: Path, split: str, file_ids: list[str]) -> str:
    """Materialise a real (tiny) git repo shaped like AMI-diarization-setup and
    return its HEAD hash, so prepare()'s pin verification runs for real."""
    import subprocess

    clone = base / "_setup"
    src = clone / "only_words" / "rttms" / split
    src.mkdir(parents=True, exist_ok=True)
    for fid in file_ids:
        (src / f"{fid}.rttm").write_text(
            to_rttm([(0.0, 1.0, "A")], file_id=fid), encoding="utf-8"
        )
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"]}
    if not (clone / ".git").is_dir():
        subprocess.run(["git", "init", "-q", str(clone)], check=True, env=env)
    subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(clone), "commit", "-q", "-m", "gold", "--allow-empty"],
                   check=True, env=env)
    return subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True, env=env).stdout.strip()


def test_ami_audio_url_matches_download_script():
    assert audio_url("EN2002a") == (
        "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
        "EN2002a/audio/EN2002a.Mix-Headset.wav"
    )


def test_ami_prepare_fetches_only_the_split_and_only_missing(tmp_path: Path):
    root = tmp_path
    base = root / "ami"
    _fake_setup_clone(base, "test", ["EN2002a", "ES2004a"])
    head = _fake_setup_clone(base, "dev", ["IS1008a"])  # other split — must NOT be fetched
    (base / "audio").mkdir()
    (base / "audio" / "ES2004a.Mix-Headset.wav").write_bytes(b"already-here")
    fetched: list[str] = []

    def fetcher(url: str, dest: Path) -> None:
        fetched.append(url)
        dest.write_bytes(b"wav")

    loader = AMILoader(split="test", fetcher=fetcher)
    loader.prepare(root, revision=head)  # clone exists at the pin → no fetch
    assert fetched == [audio_url("EN2002a")]
    files = list(loader.iter_files(root))
    assert [f.file_id for f in files] == ["EN2002a", "ES2004a"]
    assert all(f.audio_path is not None for f in files)
    # Second prepare is a no-op: gold present, audio present.
    loader.prepare(root, revision=head)
    assert fetched == [audio_url("EN2002a")]


def test_ami_prepare_rejects_stale_clone_it_cannot_move(tmp_path: Path):
    """A pre-existing clone at the wrong commit is never silently used."""
    import subprocess

    root = tmp_path
    _fake_setup_clone(root / "ami", "test", ["EN2002a"])
    loader = AMILoader(split="test", fetcher=lambda u, d: d.write_bytes(b"wav"))
    # No remote 'origin' in the fake repo → the fetch to move HEAD fails loudly.
    with pytest.raises(subprocess.CalledProcessError):
        loader.prepare(root, revision="0" * 40)


def test_ami_failed_download_skips_loudly(tmp_path: Path, capsys):
    root = tmp_path
    head = _fake_setup_clone(root / "ami", "test", ["EN2002a"])

    def failing(url: str, dest: Path) -> None:
        raise OSError("mirror down")

    loader = AMILoader(split="test", fetcher=failing)
    loader.prepare(root, revision=head)
    assert "download FAILED" in capsys.readouterr().out
    (only,) = loader.iter_files(root)
    assert only.audio_path is None  # runner treats this as skip-with-warning


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

    # A stale sibling dataset from an earlier run in the same results dir must
    # NOT be promoted — it has no expected entry and would fail `make verify`.
    (run / "gold" / "stale-set").mkdir()
    (run / "hyp" / "stale-set").mkdir()
    (run / "gold" / "stale-set" / "old.rttm").write_text(to_rttm([(0, 1, "A")], file_id="old"))
    (run / "hyp" / "stale-set" / "old.rttm").write_text(to_rttm([(0, 1, "A")], file_id="old"))

    dest = promote(run, tmp_path / "artifacts", "2026-07-30")
    expected = json.loads((dest / "expected.json").read_text())
    assert set(expected["voxconverse"]) == set(DerScore.EXPECTED_FIELDS)
    assert (dest / "gold" / "voxconverse" / "f1.rttm").exists()
    assert (dest / "hyp" / "voxconverse" / "f1.rttm").exists()
    assert not (dest / "gold" / "stale-set").exists()
    assert not (dest / "hyp" / "stale-set").exists()

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


def test_the_published_gap_threshold_matches_the_contract():
    """The folding threshold is a published quantity, not an implementation detail.

    A ~1 pp DER swing across a 4x sweep of it is twenty times the reproduction
    tolerance, so a reader who does not know the value cannot reproduce the
    number. Config and code must therefore not be able to drift apart.
    """
    import yaml

    from raven_diar.adapters.aggregate import DEFAULT_GAP_MERGE_S

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "benchmark.config.yaml").read_text()
    )
    assert config["der"]["turn_gap_merge_s"] == DEFAULT_GAP_MERGE_S, (
        "benchmark.config.yaml publishes a turn-folding threshold that the "
        "shared aggregator does not use — every hosted DER row would be "
        "measured under rules the committed contract misstates."
    )


# ── reproduce.run: the summary records what was measured, not just the pins ──


def _fake_run(tmp_path: Path, monkeypatch, diarizer) -> dict:
    """Drive ``raven_diar.reproduce.run`` with a one-file loader and ``diarizer``."""
    from raven_diar import reproduce
    from raven_diar.datasets.base import DiarFile

    gold = tmp_path / "f1.rttm"
    gold.write_text(to_rttm([(0, 10, "A"), (10, 20, "B")], file_id="f1"))
    audio = tmp_path / "f1.wav"
    audio.write_bytes(b"")

    class _Loader:
        def prepare(self, root, revision=None):  # pragma: no cover - not called
            raise AssertionError("skip_prepare=True must skip the download")

        def iter_files(self, root, limit=None):
            yield DiarFile("f1", audio, gold, "voxconverse")

    monkeypatch.setattr(reproduce, "_make_loader", lambda *a, **k: _Loader())
    monkeypatch.setattr(reproduce, "_make_diarizer", lambda *a, **k: diarizer)
    out = tmp_path / "out"
    summary_path = reproduce.run(
        dataset="voxconverse",
        model_key="sortformer-streaming-4spk-v2",
        root=tmp_path,
        out_dir=out,
        limit=None,
        dataset_revision=None,
        model_revision=None,
        skip_prepare=True,
    )
    return json.loads(summary_path.read_text())


class _StubDiarizer:
    """Returns one fixed hypothesis; optionally declares a ``run_config``."""

    def __init__(self, run_config=None):
        if run_config is not None:
            self.run_config = run_config

    def diarize(self, audio_path):
        from raven_diar.adapters.base import DiarizeResult

        return DiarizeResult(segments=[(0.0, 20.0, "Z")], latency_s=0.1)


def test_summary_records_the_latency_preset_a_run_was_measured_under(
    tmp_path: Path, monkeypatch
):
    """A run at a non-default streaming preset is a different number.

    ``promote`` copies this summary into the artifact, so recording the preset is
    what stops a diagnostic run from being published as if it were the shipped
    configuration.
    """
    config = {"latency_preset": "low-latency", "streaming_config": {"chunk_len": 6}}
    summary = _fake_run(tmp_path, monkeypatch, _StubDiarizer(run_config=config))
    assert summary["diarizer_config"] == config


def test_an_adapter_without_a_run_config_leaves_the_summary_shape_alone(
    tmp_path: Path, monkeypatch
):
    """Committed summaries have no such key; hosted adapters must not grow a null."""
    summary = _fake_run(tmp_path, monkeypatch, _StubDiarizer())
    assert "diarizer_config" not in summary
