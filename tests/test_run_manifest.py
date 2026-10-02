"""The run manifest and ``schema_version`` of a Tier-2 ``summary.json``.

Two promises are held here. A run — WER or DER — writes down what produced it.
And a summary in a format this code does not know stops ``verify`` instead of
being re-scored on a guess.

The runs are driven over the committed ``_demo`` fixtures with stub inference,
so the whole chain (run → promote → verify) executes without a GPU or a key and
the promoted numbers can be held against the fixtures' own ``expected.json``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from raven_eval_core import run_manifest
from raven_eval_core.der import load_rttm
from raven_eval_core.run_manifest import (
    LEGACY_SCHEMA_VERSION,
    MANIFEST_FIELDS,
    SCHEMA_VERSION,
    UnknownSchemaVersionError,
    summary_problem,
    summary_schema_version,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
DEMO_WER = ARTIFACTS / "_demo" / "demo-model"
DEMO_DER = ARTIFACTS / "_demo_der" / "demo-diarizer"


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


def _assert_is_a_manifest(summary: dict) -> dict:
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary_problem(summary) == ""
    manifest = summary["run_manifest"]
    assert set(manifest) == set(MANIFEST_FIELDS)
    # This suite runs from a git checkout, so both git facts are known.
    assert len(manifest["git"]["sha"]) == 40
    assert isinstance(manifest["git"]["dirty"], bool)
    assert manifest["uv_lock_sha256"] == run_manifest.uv_lock_sha256()
    assert manifest["libraries"]["jiwer"]
    assert manifest["libraries"]["pyannote.metrics"]
    assert manifest["argv"] == run_manifest.cli_invocation()
    return manifest


# ── WER: a run on _demo ──────────────────────────────────────────────────────


@pytest.fixture
def demo_wer_run(monkeypatch: pytest.MonkeyPatch):
    """``raven_asr.runner`` wired to replay the committed ``_demo`` predictions."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("tqdm")
    from raven_asr import runner
    from raven_asr.adapters.base import TranscribeResult
    from raven_asr.config import WerDatasetSpec
    from raven_asr.datasets.base import Sample

    rows = [
        json.loads(line)
        for line in (DEMO_WER / "predictions_Demo-DE.jsonl").read_text().splitlines()
    ]
    samples = [
        Sample(
            audio=np.array([float(i), 0.0], dtype=np.float32), sample_rate=16000,
            reference=row["reference"], sample_id=f"Demo-DE-{i}", subset="Demo-DE",
        )
        for i, row in enumerate(rows)
    ]
    calls: list[int] = []

    class _Loader:
        def iter_samples(self, subset, limit=None):
            return samples if limit is None else samples[:limit]

    class _ReplayAdapter:
        provider_id = "demo"
        model_id = "demo"

        async def atranscribe(self, audio, sample_rate):
            calls.append(int(audio[0]))
            return TranscribeResult(
                text=rows[int(audio[0])]["prediction"], latency_s=0.001,
                raw={"replay": True},
            )

    demo = WerDatasetSpec(
        id="demo", loader="demo", license="fixture", source="artifacts/_demo",
        revision="fixture", subsets=("Demo-DE",), durability="hf",
    )
    monkeypatch.setattr(runner, "WER_DATASETS", {"demo": demo})
    monkeypatch.setattr(runner, "resolve_wer_dataset", lambda _s: ("demo", "Demo-DE"))
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset", lambda _s, **_kw: (_Loader(), "Demo-DE")
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _ReplayAdapter())

    def _run(out_dir: Path) -> dict:
        runner.run(
            model_key="primeline/whisper-large-v3-german", subsets=["Demo-DE"],
            limit=None, out_dir=out_dir,
        )
        return json.loads((out_dir / "summary.json").read_text())

    _run.calls = calls  # type: ignore[attr-defined]
    _run.n_samples = len(samples)  # type: ignore[attr-defined]
    return _run


def test_a_wer_run_on_demo_writes_a_manifest(tmp_path: Path, demo_wer_run):
    summary = demo_wer_run(tmp_path / "results" / "demo-model")
    assert [r["n_failed"] for r in summary["results"]] == [0]
    manifest = _assert_is_a_manifest(summary)
    # Every ASR adapter calls an endpoint: there is no local accelerator to name.
    assert manifest["gpu"] is None


def test_a_tampered_schema_version_fails_verify(tmp_path: Path, demo_wer_run):
    """run → promote → verify is green; an unknown version turns it red."""
    from raven_asr import promote as promote_mod

    results = tmp_path / "results" / "demo-model"
    demo_wer_run(results)
    artifacts = tmp_path / "artifacts"
    dest = promote_mod.promote(results, artifacts, run_name="demo")

    # The promoted run reproduces the committed fixture's own published number.
    committed = json.loads((DEMO_WER / "expected.json").read_text())["Demo-DE"]
    promoted = json.loads((dest / "expected.json").read_text())["Demo-DE"]
    assert promoted["wer_pct"] == committed["wer_pct"]
    assert _assert_is_a_manifest(json.loads((dest / "summary.json").read_text()))

    all_ok, rows = verify.verify(artifacts)
    assert all_ok and rows, rows
    verify.manifest.write_manifest(artifacts)
    assert verify.main(["--artifacts-dir", str(artifacts)]) == 0

    summary = json.loads((dest / "summary.json").read_text())
    summary["schema_version"] = SCHEMA_VERSION + 1
    (dest / "summary.json").write_text(json.dumps(summary))
    all_ok, rows = verify.verify(artifacts)
    assert not all_ok
    assert [r["status"] for r in rows] == ["FAIL"]
    assert "unknown schema_version" in rows[0]["detail"]
    # Sealed again, so the exit code is the format check's and not the seal's.
    verify.manifest.write_manifest(artifacts)
    assert verify.main(["--artifacts-dir", str(artifacts)]) == 1


def test_a_current_summary_without_its_manifest_fails_verify(
    tmp_path: Path, demo_wer_run
):
    """The version is a claim about the shape; the shape has to be there."""
    from raven_asr import promote as promote_mod

    results = tmp_path / "results" / "demo-model"
    demo_wer_run(results)
    dest = promote_mod.promote(results, tmp_path / "artifacts", run_name="demo")
    summary = json.loads((dest / "summary.json").read_text())
    del summary["run_manifest"]
    (dest / "summary.json").write_text(json.dumps(summary))
    all_ok, rows = verify.verify(tmp_path / "artifacts")
    assert not all_ok
    assert "requires a run_manifest" in rows[0]["detail"]


def test_promote_refuses_a_summary_in_an_unknown_format(tmp_path: Path, demo_wer_run):
    from raven_asr import promote as promote_mod

    results = tmp_path / "results" / "demo-model"
    summary = demo_wer_run(results)
    summary["schema_version"] = 99
    (results / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="unknown schema_version 99"):
        promote_mod.promote(results, tmp_path / "artifacts", run_name="demo")
    assert not (tmp_path / "artifacts").exists()


def test_the_manifest_carries_no_environment_value(
    tmp_path: Path, demo_wer_run, monkeypatch: pytest.MonkeyPatch
):
    """Env-var NAMES are config; their values — URLs and keys — never ship."""
    monkeypatch.setenv("VLLM_PRIMELINE_URL", "https://endpoint.invalid/secret-host")
    monkeypatch.setenv("VLLM_PRIMELINE_API_KEY", "sk-not-a-real-key-0123456789")
    text = json.dumps(demo_wer_run(tmp_path / "results" / "demo-model"))
    assert "endpoint.invalid" not in text
    assert "sk-not-a-real-key" not in text


def test_a_subset_is_resumed_only_under_the_conditions_it_was_measured_in(
    tmp_path: Path, demo_wer_run, monkeypatch: pytest.MonkeyPatch
):
    """One summary holds one manifest, so it may only describe one environment."""
    from raven_asr import runner

    out = tmp_path / "results" / "demo-model"
    demo_wer_run(out)
    assert len(demo_wer_run.calls) == demo_wer_run.n_samples

    demo_wer_run(out)  # same commit, lockfile and config: resumed, no request
    assert len(demo_wer_run.calls) == demo_wer_run.n_samples

    real = runner._run_manifest

    def _other_commit(*args, **kwargs):
        manifest = real(*args, **kwargs)
        manifest["git"] = {"sha": "0" * 40, "dirty": False}
        return manifest

    monkeypatch.setattr(runner, "_run_manifest", _other_commit)
    summary = demo_wer_run(out)
    assert len(demo_wer_run.calls) == 2 * demo_wer_run.n_samples
    assert summary["run_manifest"]["git"]["sha"] == "0" * 40


# ── DER: a run on _demo_der ──────────────────────────────────────────────────


def _demo_der_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_key: str) -> Path:
    """Drive ``raven_diar.reproduce.run`` over the committed ``_demo_der`` RTTMs."""
    from raven_diar import reproduce
    from raven_diar.adapters.base import DiarizeResult
    from raven_diar.config import DiarDatasetSpec
    from raven_diar.datasets.base import DiarFile

    audio = tmp_path / "audio"
    audio.mkdir()

    class _Loader:
        def iter_files(self, root, limit=None):
            for gold in sorted((DEMO_DER / "gold" / "demo-set").glob("*.rttm")):
                wav = audio / f"{gold.stem}.wav"
                wav.write_bytes(b"")
                yield DiarFile(gold.stem, wav, gold, "demo-set")

    class _ReplayDiarizer:
        def diarize(self, audio_path):
            hyp = DEMO_DER / "hyp" / "demo-set" / f"{Path(audio_path).stem}.rttm"
            return DiarizeResult(segments=load_rttm(hyp), latency_s=0.1)

    demo = DiarDatasetSpec(
        id="demo-set", loader="demo", license="fixture",
        source="artifacts/_demo_der", revision="fixture", language="de",
    )
    monkeypatch.setattr(reproduce, "DER_DATASETS", {"demo-set": demo})
    monkeypatch.setattr(reproduce, "_make_loader", lambda *a, **k: _Loader())
    monkeypatch.setattr(reproduce, "_make_diarizer", lambda *a, **k: _ReplayDiarizer())
    out = tmp_path / "results" / "demo-diarizer"
    reproduce.run(
        dataset="demo-set", model_key=model_key, root=tmp_path, out_dir=out,
        limit=None, dataset_revision=None, model_revision=None, skip_prepare=True,
    )
    return out


def test_a_der_run_on_demo_writes_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from raven_diar.promote import promote

    results = _demo_der_run(tmp_path, monkeypatch, "pyannote-community-1")
    manifest = _assert_is_a_manifest(json.loads((results / "summary.json").read_text()))
    # A local diarizer ran on this machine: the accelerator is stated, GPU or not.
    assert set(manifest["gpu"]) == {"cuda_available", "cuda", "devices"}

    artifacts = tmp_path / "artifacts"
    dest = promote(results, artifacts, "demo")
    assert json.loads((dest / "expected.json").read_text()) == json.loads(
        (DEMO_DER / "expected.json").read_text()
    )
    all_ok, rows = verify.verify_der(artifacts)
    assert all_ok and rows, rows

    summary = json.loads((dest / "summary.json").read_text())
    summary["schema_version"] = "2"
    (dest / "summary.json").write_text(json.dumps(summary))
    all_ok, rows = verify.verify_der(artifacts)
    assert not all_ok
    assert "unknown schema_version" in rows[0]["detail"]


def test_a_hosted_der_run_names_no_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    results = _demo_der_run(tmp_path, monkeypatch, "deepgram-nova-3")
    summary = json.loads((results / "summary.json").read_text())
    assert _assert_is_a_manifest(summary)["gpu"] is None


def test_wer_and_der_manifests_have_one_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, demo_wer_run
):
    der = json.loads(
        (_demo_der_run(tmp_path, monkeypatch, "deepgram-nova-3") / "summary.json")
        .read_text()
    )["run_manifest"]
    wer = demo_wer_run(tmp_path / "results" / "demo-model")["run_manifest"]
    assert set(wer) == set(der)
    # Different harness, different resolved config — the hash has to tell.
    assert wer["config_sha256"] != der["config_sha256"]


# ── schema_version ───────────────────────────────────────────────────────────


def test_an_absent_schema_version_is_the_legacy_format():
    assert summary_schema_version({"model_id": "m"}) == LEGACY_SCHEMA_VERSION
    assert summary_problem({"model_id": "m"}) == ""


@pytest.mark.parametrize("version", [0, 3, 99, -1, "2", 2.0, True, None])
def test_an_unknown_schema_version_is_rejected(version):
    with pytest.raises(UnknownSchemaVersionError):
        summary_schema_version({"schema_version": version})
    assert "unknown schema_version" in summary_problem({"schema_version": version})


def test_a_manifest_with_an_unknown_field_is_rejected():
    manifest = run_manifest.build_run_manifest(
        resolved_config={"a": 1}, libraries=(), gpu=None
    )
    good = {"schema_version": SCHEMA_VERSION, "run_manifest": manifest}
    assert summary_problem(good) == ""
    extra = {**good, "run_manifest": {**manifest, "hostname": "box"}}
    assert "unknown ['hostname']" in summary_problem(extra)
    # A manifest nobody would read: present, but under the legacy version.
    assert "without schema_version" in summary_problem({"run_manifest": manifest})


def test_every_committed_summary_is_in_a_format_verify_reads():
    summaries = sorted(ARTIFACTS.rglob("summary.json"))
    assert summaries, "a guard that matches nothing guards nothing"
    problems = {
        str(p.relative_to(REPO_ROOT)): problem
        for p in summaries
        if (problem := summary_problem(json.loads(p.read_text(encoding="utf-8"))))
    }
    assert not problems


def test_the_config_hash_ignores_key_order_and_sees_every_value():
    a = run_manifest.config_sha256({"x": 1, "y": {"b": 2, "a": 3}})
    assert a == run_manifest.config_sha256({"y": {"a": 3, "b": 2}, "x": 1})
    assert a != run_manifest.config_sha256({"x": 1, "y": {"b": 2, "a": 4}})


# ── git state ────────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t.invalid",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True,
    )


def test_git_state_names_the_commit_and_flags_uncommitted_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = tmp_path / "checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-q", "-m", "init")
    monkeypatch.setattr(run_manifest, "_REPO_ROOT", repo.resolve())

    state = run_manifest.git_state()
    assert len(state["sha"]) == 40 and state["dirty"] is False

    (repo / "new_module.py").write_text("y = 2\n")  # untracked code is code that ran
    assert run_manifest.git_state() == {"sha": state["sha"], "dirty": True}


def test_git_state_claims_nothing_outside_its_own_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An install nested in someone else's repository must not borrow its SHA."""
    outer = tmp_path / "outer"
    nested = outer / "site-packages"
    nested.mkdir(parents=True)
    _git(outer, "init", "-q")
    monkeypatch.setattr(run_manifest, "_REPO_ROOT", nested.resolve())
    assert run_manifest.git_state() == {"sha": None, "dirty": None}
    assert run_manifest.uv_lock_sha256() is None
