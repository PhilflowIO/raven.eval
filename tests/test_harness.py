"""Tier-2 harness tests: runner (fake adapter) + promote → Tier-1 verify round-trip.

The runner pulls in the ``asr`` extra (numpy/tqdm/pyyaml/datasets). When only the
Tier-1 ``dev`` env is installed these are skipped, so the metric-core test suite
still runs standalone. CI installs ``--extra asr`` so they execute.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("numpy")
pytest.importorskip("tqdm")
pytest.importorskip("yaml")

import numpy as np
import yaml

from raven_asr import promote as promote_mod
from raven_asr import runner
from raven_asr.adapters.base import TranscribeResult
from raven_asr.config import KNOWN_MODELS, ModelSpec
from raven_asr.datasets.base import RowTally, Sample


class _PerfectAdapter:
    provider_id = "perfect"
    model_id = "perfect-mock"

    def __init__(self, registry: dict[float, str]) -> None:
        self._registry = registry

    async def atranscribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        return TranscribeResult(
            text=self._registry[float(audio[0])], latency_s=0.001, raw={"mock": True}
        )


class _FakeLoader:
    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples

    def iter_samples(self, subset: str, limit: int | None = None) -> list[Sample]:
        out = [s for s in self._samples if s.subset == subset]
        return out if limit is None else out[:limit]


def _make_samples(subset: str, refs: list[str]) -> tuple[list[Sample], dict[float, str]]:
    samples: list[Sample] = []
    registry: dict[float, str] = {}
    for i, ref in enumerate(refs):
        key = float(i + 1)
        samples.append(
            Sample(
                audio=np.array([key, 0.0, 0.0], dtype=np.float32),
                sample_rate=16000,
                reference=ref,
                sample_id=f"{subset}-{i}",
                subset=subset,
            )
        )
        registry[key] = ref
    return samples, registry


def test_runner_end_to_end_writes_predictions_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples, registry = _make_samples(
        "Tuda-De", ["hallo welt", "guten tag", "wie geht es dir"]
    )
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_FakeLoader(samples), "Tuda-De"),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _PerfectAdapter(registry))

    out_dir = tmp_path / "perfect"
    yaml_path = runner.run(
        model_key="primeline/whisper-large-v3-german",
        subsets=["Tuda-De"],
        limit=None,
        out_dir=out_dir,
        concurrency=2,
    )
    assert yaml_path.is_file()
    loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    mi = cast(list[dict[str, Any]], loaded["model-index"])
    metrics = cast(list[dict[str, Any]], mi[0]["results"][0]["metrics"])
    assert next(m for m in metrics if m["type"] == "wer")["value"] == 0.0
    # predictions_*.jsonl (the Tier-1 artifact source) was written
    preds = list(out_dir.glob("predictions_Tuda-De.jsonl"))
    assert preds and preds[0].read_text().strip()


def _load_verify():
    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("verify", repo / "scripts" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_promote_then_verify_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run → promote → verify chain must go green (the trust payoff)."""
    samples, registry = _make_samples("Tuda-De", ["hallo welt", "guten tag"])
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_FakeLoader(samples), "Tuda-De"),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _PerfectAdapter(registry))

    results_dir = tmp_path / "results" / "primeline-whisper-large-v3-german"
    runner.run(
        model_key="primeline/whisper-large-v3-german",
        subsets=["Tuda-De"],
        limit=None,
        out_dir=results_dir,
    )

    artifacts = tmp_path / "artifacts"
    dest = promote_mod.promote(results_dir, artifacts, run_name="testrun")
    assert (dest / "expected.json").is_file()
    assert (dest / "predictions_Tuda-De.jsonl").is_file()

    # expected.json is derived from summary.json, never hand-typed
    expected = json.loads((dest / "expected.json").read_text())
    assert "Tuda-De" in expected
    assert expected["Tuda-De"]["wer_pct"] == 0.0
    # the page lens is committed next to the published one, scored from the predictions
    assert expected["Tuda-De"]["wer_strict_de_pct"] == 0.0
    # and so is the numeric entity score — here: no number, so no hit rate
    assert expected["Tuda-De"]["n_entities"] == 0
    assert expected["Tuda-De"]["entity_hit_rate_pct"] is None

    # every prediction line names its utterance and carries its audio length
    lines = [json.loads(x) for x in (dest / "predictions_Tuda-De.jsonl").read_text().splitlines()]
    assert all("sample_id" in x and x["duration_s"] > 0 for x in lines)
    # and expected.json commits to how many there are
    assert expected["Tuda-De"]["n_samples"] == len(lines) == 2
    # a run whose loader kept no drop count commits none — absent, not zero
    assert "n_dropped" not in expected["Tuda-De"]
    assert "dropped_by_reason" not in expected["Tuda-De"]

    verify = _load_verify()
    all_ok, rows = verify.verify(artifacts)
    assert rows
    assert all_ok, f"promoted artifact failed re-score: {rows}"

    # a hand-edited page number is caught like a hand-edited published one
    expected["Tuda-De"]["wer_strict_de_pct"] = 5.0
    (dest / "expected.json").write_text(json.dumps(expected))
    tampered_ok, tampered_rows = verify.verify(artifacts)
    assert not tampered_ok
    assert any("wer_strict_de" in r.get("detail", "") for r in tampered_rows)


class _FlakyAdapter(_PerfectAdapter):
    """Perfect, except on the clips it was told to fail on."""

    def __init__(self, registry: dict[float, str], fail_on: set[float]) -> None:
        super().__init__(registry)
        self._fail_on = fail_on

    async def atranscribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        if float(audio[0]) in self._fail_on:
            raise TimeoutError("upstream timed out")
        return await super().atranscribe(audio, sample_rate)


def test_a_failed_request_is_recorded_and_blocks_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure is counted, written down, retried next time, and never published.

    Before this, the failed clip vanished: WER over the two that worked read 0.0
    and nothing downstream could tell a subset of the split from all of it.
    """
    samples, registry = _make_samples("Tuda-De", ["hallo welt", "guten tag", "wie geht es"])
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_FakeLoader(samples), "Tuda-De"),
    )
    monkeypatch.setattr(
        runner, "_make_adapter", lambda _spec: _FlakyAdapter(registry, fail_on={2.0})
    )
    results_dir = tmp_path / "results" / "primeline-whisper-large-v3-german"
    runner.run(
        model_key="primeline/whisper-large-v3-german", subsets=["Tuda-De"],
        limit=None, out_dir=results_dir,
    )

    summary = json.loads((results_dir / "summary.json").read_text())
    (row,) = summary["results"]
    assert (row["n_attempted"], row["n_ok"], row["n_failed"]) == (3, 2, 1)
    assert row["n_samples"] == 2

    lines = [json.loads(x) for x in (results_dir / "predictions_Tuda-De.jsonl")
             .read_text().splitlines()]
    assert [x["sample_id"] for x in lines] == ["Tuda-De-0", "Tuda-De-1", "Tuda-De-2"]
    failed = lines[1]
    assert failed["prediction"] is None
    assert failed["error"].startswith("TimeoutError")

    # No resume marker: the next invocation re-runs the subset.
    assert not (results_dir / ".done_Tuda-De.json").exists()

    with pytest.raises(promote_mod.IncompleteRunError, match="1 of 3 requests failed"):
        promote_mod.promote(results_dir, tmp_path / "artifacts", run_name="r")
    assert not (tmp_path / "artifacts").exists()


def test_multi_model_path_records_subsets_like_the_single_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both paths go through one recorder: predictions, coverage, revisions.

    The N-model path used to score without writing predictions at all, so its
    output could never become a Tier-1 artifact.
    """
    samples, registry = _make_samples("Tuda-De", ["hallo welt", "guten tag"])
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_FakeLoader(samples), "Tuda-De"),
    )
    monkeypatch.setattr(
        runner, "_make_adapter",
        lambda spec: _FlakyAdapter(
            registry, fail_on={1.0} if spec.label.endswith("turbo-german") else set()
        ),
    )
    import asyncio

    asyncio.run(runner.run_multi_async(
        model_keys=["primeline/whisper-large-v3-german",
                    "primeline/whisper-large-v3-turbo-german"],
        subsets=["Tuda-De"], limit=None, out_root=tmp_path,
    ))
    for label, n_failed in (("primeline-whisper-large-v3-german", 0),
                            ("primeline-whisper-large-v3-turbo-german", 1)):
        out = tmp_path / label
        (row,) = json.loads((out / "summary.json").read_text())["results"]
        assert row["n_failed"] == n_failed
        assert row["dataset_revision"]
        lines = (out / "predictions_Tuda-De.jsonl").read_text().splitlines()
        assert len(lines) == 2


def test_summary_names_the_revisions_it_was_measured_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config comment promised this; now the summary actually carries it."""
    from raven_asr.config import FLOZI_DATASET_REVISION

    samples, registry = _make_samples("Tuda-De", ["hallo welt"])
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_FakeLoader(samples), "Tuda-De"),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _PerfectAdapter(registry))
    out = tmp_path / "xai-grok-voice-transcribe-2.0"
    runner.run(model_key="xai/grok-voice-transcribe-2.0", subsets=["Tuda-De"],
               limit=None, out_dir=out)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["model_revision"] == "grok-voice-transcribe-2.0"
    assert summary["results"][0]["dataset_revision"] == FLOZI_DATASET_REVISION

    # an explicit --dataset-revision is what gets recorded, not the registry pin
    out2 = tmp_path / "pinned"
    runner.run(model_key="xai/grok-voice-transcribe-2.0", subsets=["Tuda-De"],
               limit=None, out_dir=out2, revision="abc123")
    assert json.loads((out2 / "summary.json").read_text())["results"][0][
        "dataset_revision"] == "abc123"


def _capture_vllm_adapter(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace VllmOpenAIAdapter with a kwargs-recorder (no env/network needed)."""
    vllm_mod = pytest.importorskip("raven_asr.adapters.vllm_openai")
    captured: dict[str, Any] = {}

    class _FakeAdapter:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(vllm_mod, "VllmOpenAIAdapter", _FakeAdapter)
    return captured


def test_make_adapter_threads_endpoint_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelSpec.base_url_env/api_key_env must reach the adapter constructor."""
    captured = _capture_vllm_adapter(monkeypatch)
    runner._make_adapter(KNOWN_MODELS["nvidia/canary-1b-v2"])
    assert captured == {
        "provider_id": "nvidia-canary-1b-v2",
        "model_id": "nvidia/canary-1b-v2",
        "base_url_env": "NEMO_BENCH_URL",
        "api_key_env": "BENCH_STT_API_KEY",
        "output_format": None,
    }


def test_make_adapter_refuses_a_self_hosted_spec_without_an_endpoint() -> None:
    """No silent fallback endpoint: a spec that names none is a registry bug."""
    spec = ModelSpec(model_id="x/y", adapter="vllm_openai", label="x-y")
    with pytest.raises(ValueError, match="base_url_env"):
        runner._make_adapter(spec)


# Which HF id runs on which server type is the interface to whoever serves the
# models; a change here is a change for them, so it is pinned.
_NEMO_SERVED = {
    "nvidia/parakeet-tdt-0.6b-v3",
    "nvidia/canary-1b-v2",
    "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "primeline/parakeet-primeline",
    "nvidia/nemotron-labs-unified-asr-0.6b",
}
_VLLM_SERVED = {
    "ibm-granite/granite-speech-4.1-2b-plus",
    "CohereLabs/cohere-transcribe-03-2026",
    "microsoft/Phi-4-multimodal-instruct",
    "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    "moonshine-ai/moonshine-streaming-small-de",
    "Qwen/Qwen3-ASR-1.7B",
    "mistralai/Voxtral-Mini-3B-2507",
    "primeline/whisper-large-v3-german",
    "primeline/whisper-large-v3-turbo-german",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "ibm-granite/granite-speech-5.0-2b",
    "ibm-granite/granite-speech-5.0-470m-turboctc",
    "mistralai/Voxtral-Mini-4B-Realtime-2602",
    "microsoft/VibeVoice-ASR-HF",
    "RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic",
    "mistralai/Voxtral-Small-24B-2507",
    "PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16",
    "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
}


def test_all_self_hosted_models_bind_to_their_server_type() -> None:
    from raven_asr.config import SELF_HOSTED_MODELS

    assert set(SELF_HOSTED_MODELS) == _NEMO_SERVED | _VLLM_SERVED
    assert len(SELF_HOSTED_MODELS) == 24
    for key in _NEMO_SERVED | _VLLM_SERVED:
        spec = KNOWN_MODELS[key]
        assert spec.adapter == "vllm_openai"
        # The model posted (and echoed back) is the HF id the run is keyed by.
        assert spec.model_id == key
        expected_url = "NEMO_BENCH_URL" if key in _NEMO_SERVED else "VLLM_BENCH_URL"
        assert spec.base_url_env == expected_url
        assert spec.api_key_env == "BENCH_STT_API_KEY"
        # The client cannot see which weights a server loaded.
        assert spec.revision is None
    labels = [s.label for s in KNOWN_MODELS.values()]
    assert len(labels) == len(set(labels)), "result-directory labels must be unique"
    # A published artifact's directory name is its label; it must not move.
    assert KNOWN_MODELS["primeline/parakeet-primeline"].label == "primeline-parakeet"


def test_promote_refuses_empty_results(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        promote_mod.promote(empty, tmp_path / "artifacts", run_name="x")


def test_promote_scores_bleu_on_a_translation_shaped_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bleu+wer corpus gets its headline BLEU at promote time, and verify holds it."""
    samples, registry = _make_samples(
        "spc-test", ["das ist heute ein guter tag", "wir gehen morgen früh nach hause"]
    )
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_FakeLoader(samples), "spc-test"),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _PerfectAdapter(registry))

    results_dir = tmp_path / "results" / "primeline-whisper-large-v3-german"
    runner.run(
        model_key="primeline/whisper-large-v3-german",
        subsets=["spc-test"],
        limit=None,
        out_dir=results_dir,
    )
    artifacts = tmp_path / "artifacts"
    dest = promote_mod.promote(results_dir, artifacts, run_name="bleurun")

    exp = json.loads((dest / "expected.json").read_text())["spc-test"]
    assert exp["bleu"] == 100.0
    assert exp["bleu_signature"].startswith("nrefs:1|")

    all_ok, rows = _load_verify().verify(artifacts)
    assert all_ok, rows


def test_promote_leaves_bleu_off_a_dictation_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples, registry = _make_samples("fleurs", ["hallo welt"])
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_FakeLoader(samples), "fleurs"),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _PerfectAdapter(registry))
    results_dir = tmp_path / "results" / "m"
    runner.run(model_key="primeline/whisper-large-v3-german", subsets=["fleurs"],
               limit=None, out_dir=results_dir)
    dest = promote_mod.promote(results_dir, tmp_path / "artifacts", run_name="r")
    assert "bleu" not in json.loads((dest / "expected.json").read_text())["fleurs"]


def test_summary_records_the_entity_score_beside_the_wer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run summary carries the hit rate and the count it rests on."""
    samples, registry = _make_samples(
        "Tuda-De", ["es kamen dreiundzwanzig Leute", "hallo welt", "um 10 Uhr"]
    )
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_FakeLoader(samples), "Tuda-De"),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _PerfectAdapter(registry))
    results_dir = tmp_path / "results" / "primeline-whisper-large-v3-german"
    runner.run(
        model_key="primeline/whisper-large-v3-german", subsets=["Tuda-De"],
        limit=None, out_dir=results_dir,
    )
    (row,) = json.loads((results_dir / "summary.json").read_text())["results"]
    assert row["entity_hit_rate_pct"] == 100.0
    assert (row["n_entities"], row["n_utterances_with_entities"]) == (2, 2)


class _DroppingLoader(_FakeLoader):
    """A loader that, like the archive-backed ones, could not yield some rows."""

    def __init__(self, samples: list[Sample], dropped: dict[str, int]) -> None:
        super().__init__(samples)
        self._dropped = dropped
        self.tally = RowTally()

    def iter_samples(self, subset: str, limit: int | None = None) -> list[Sample]:
        out = super().iter_samples(subset, limit)
        n = sum(self._dropped.values())
        self.tally = RowTally(
            rows_read=len(out) + n, yielded=len(out), dropped=dict(self._dropped))
        return out


def test_summary_shows_rows_the_loader_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run over fewer rows than the corpus must say so in its own output."""
    samples, registry = _make_samples("Tuda-De", ["hallo welt", "guten tag"])
    dropped = {"clip_missing": 2, "empty_reference": 1}
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_DroppingLoader(samples, dropped), "Tuda-De"),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _PerfectAdapter(registry))
    out = tmp_path / "primeline-whisper-large-v3-german"
    with caplog.at_level("WARNING", logger="raven_asr.runner"):
        runner.run(model_key="primeline/whisper-large-v3-german",
                   subsets=["Tuda-De"], limit=None, out_dir=out)

    (row,) = json.loads((out / "summary.json").read_text())["results"]
    assert row["n_dropped"] == 3
    assert row["dropped_by_reason"] == dropped
    # Dropped rows were never attempted; they are not hidden inside n_failed.
    assert (row["n_attempted"], row["n_ok"], row["n_failed"]) == (2, 2, 0)
    assert "dropped 3 row(s)" in caplog.text

    # A resumed run still shows them: the marker carries the counts.
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: pytest.fail("subset should resume from its marker"),
    )
    # The summary stays in place: its manifest is what makes the run resumable.
    # Blank the counts in it, so the assertion below can only pass if the
    # rewritten summary took them from the marker.
    summary = json.loads((out / "summary.json").read_text())
    summary["results"][0].update(n_dropped=None, dropped_by_reason=None)
    (out / "summary.json").write_text(json.dumps(summary))
    runner.run(model_key="primeline/whisper-large-v3-german",
               subsets=["Tuda-De"], limit=None, out_dir=out)
    (row,) = json.loads((out / "summary.json").read_text())["results"]
    assert (row["n_dropped"], row["dropped_by_reason"]) == (3, dropped)


def test_summary_distinguishes_no_drops_from_a_loader_without_a_tally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``0`` is a claim the loader made; ``null`` means nobody counted."""
    samples, registry = _make_samples("Tuda-De", ["hallo welt"])
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _PerfectAdapter(registry))
    for name, loader, expected in (
        ("counted", _DroppingLoader(samples, {}), (0, {})),
        ("uncounted", _FakeLoader(samples), (None, None)),
    ):
        monkeypatch.setattr(
            runner, "_iter_loader_for_subset",
            lambda _s, _l=loader, **_kw: (_l, "Tuda-De"),
        )
        out = tmp_path / name
        runner.run(model_key="primeline/whisper-large-v3-german",
                   subsets=["Tuda-De"], limit=None, out_dir=out)
        (row,) = json.loads((out / "summary.json").read_text())["results"]
        assert (row["n_dropped"], row["dropped_by_reason"]) == expected


@pytest.mark.parametrize("dropped", [{}, {"empty_reference": 1, "clip_missing": 2}])
def test_promote_publishes_dropped_rows_and_verify_holds_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dropped: dict[str, int]
) -> None:
    """Drops do not block promotion, but the published artifact has to show them."""
    samples, registry = _make_samples("Tuda-De", ["hallo welt", "guten tag"])
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_DroppingLoader(samples, dropped), "Tuda-De"),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _PerfectAdapter(registry))
    results_dir = tmp_path / "results" / "primeline-whisper-large-v3-german"
    runner.run(model_key="primeline/whisper-large-v3-german",
               subsets=["Tuda-De"], limit=None, out_dir=results_dir)

    artifacts = tmp_path / "artifacts"
    dest = promote_mod.promote(results_dir, artifacts, run_name="r")
    expected = json.loads((dest / "expected.json").read_text())
    assert expected["Tuda-De"]["n_dropped"] == sum(dropped.values())
    assert expected["Tuda-De"]["dropped_by_reason"] == dropped
    assert expected["Tuda-De"]["n_samples"] == 2

    verify = _load_verify()
    all_ok, rows = verify.verify(artifacts)
    assert all_ok, rows

    # A count that no longer matches its own breakdown is a hand edit.
    expected["Tuda-De"]["n_dropped"] += 1
    (dest / "expected.json").write_text(json.dumps(expected))
    tampered_ok, tampered_rows = verify.verify(artifacts)
    assert not tampered_ok
    assert "n_dropped" in tampered_rows[0]["detail"]


def test_make_adapter_threads_the_output_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model whose server wraps its transcript gets its parser at the adapter."""
    captured = _capture_vllm_adapter(monkeypatch)
    runner._make_adapter(KNOWN_MODELS["OpenMOSS-Team/MOSS-Transcribe-Diarize"])
    assert captured["output_format"] == "moss-timed-speakers"


_LEAK_REFS = [
    f"satz nummer {n} erzählt von einem ganz anderen ort und einer anderen zeit {w}"
    for n, w in zip(
        ["eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht"],
        ["berlin", "hamburg", "köln", "bremen", "essen", "leipzig", "dresden", "bonn"],
        strict=True,
    )
]


class _CrosstalkAdapter(_PerfectAdapter):
    """Each transcript runs on into the next clip's words (raven.eval#89)."""

    async def atranscribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        key = float(audio[0])
        nxt = self._registry.get(key + 1.0, self._registry[1.0])
        return TranscribeResult(
            text=f"{self._registry[key]} {nxt}", latency_s=0.001, raw={"mock": True}
        )


def _run_leaking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapter_cls) -> Path:
    samples, registry = _make_samples("Tuda-De", _LEAK_REFS)
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_FakeLoader(samples), "Tuda-De"),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: adapter_cls(registry))
    out_dir = tmp_path / "primeline-whisper-large-v3-german"
    runner.run(
        model_key="primeline/whisper-large-v3-german", subsets=["Tuda-De"],
        limit=None, out_dir=out_dir, concurrency=4,
    )
    return out_dir


def test_transcripts_with_text_of_other_clips_fail_the_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = _run_leaking(tmp_path, monkeypatch, _CrosstalkAdapter)

    assert not (out_dir / ".done_Tuda-De.json").exists()
    report = json.loads((out_dir / ".leaked_Tuda-De.json").read_text())
    assert (report["n_leaked"], report["n_utterances"]) == (8, 8)
    first = report["examples"][0]
    assert (first["sample_id"], first["text_of"]) == ("Tuda-De-0", "Tuda-De-1")
    # Not scored: no WER row was written for the subset.
    summary = out_dir / "summary.json"
    assert not summary.exists() or not json.loads(summary.read_text()).get("results")
    # The rows stay for inspection.
    assert len((out_dir / "predictions_Tuda-De.jsonl").read_text().splitlines()) == 8


def test_a_clean_pass_clears_an_old_leak_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "primeline-whisper-large-v3-german"
    out_dir.mkdir()
    (out_dir / ".leaked_Tuda-De.json").write_text("{}")
    _run_leaking(tmp_path, monkeypatch, _PerfectAdapter)
    assert not (out_dir / ".leaked_Tuda-De.json").exists()
    assert (out_dir / ".done_Tuda-De.json").exists()


def test_reproduce_exits_4_on_a_leaked_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from raven_asr import reproduce

    out_root = tmp_path / "out"
    label = KNOWN_MODELS["primeline/whisper-large-v3-german"].label

    def fake_run(**kw: Any) -> Path:
        out_dir = kw["out_dir"]
        out_dir.mkdir(parents=True)
        (out_dir / ".leaked_fleurs.json").write_text(json.dumps(
            {"n_leaked": 9, "n_utterances": 20, "leaked_pct": 45.0}))
        return out_dir / "model-index.yaml"

    monkeypatch.setattr(reproduce, "run", fake_run)
    rc = reproduce.main([
        "--metric", "wer", "--dataset", "fleurs",
        "--model", "primeline/whisper-large-v3-german", "--out", str(out_root),
    ])
    assert rc == reproduce.EXIT_LEAKED == 4
    assert "9 of 20 transcripts" in capsys.readouterr().err
    assert (out_root / label / ".leaked_fleurs.json").exists()


# ── every finished subset stays in the summary (#100) ──────────────────────────


def _one_subset_per_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, between=None
) -> Path:
    """Two invocations into one dir, one subset each — how a campaign drives it.

    ``between(results_dir)`` runs after the first invocation.
    """
    refs = ["hallo welt", "guten tag"]
    tuda, registry = _make_samples("Tuda-De", refs)
    cv, _ = _make_samples("common_voice_19_0", refs)  # same refs: one registry
    by_subset = {"Tuda-De": tuda, "common_voice_19_0": cv}
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda s, **_kw: (_FakeLoader(by_subset[s]), s),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: _PerfectAdapter(registry))
    results_dir = tmp_path / "results" / "primeline-whisper-large-v3-german"
    for i, subset in enumerate(("Tuda-De", "common_voice_19_0")):
        if i == 1 and between is not None:
            between(results_dir)
        runner.run(model_key="primeline/whisper-large-v3-german", subsets=[subset],
                   limit=None, out_dir=results_dir)
    return results_dir


def test_a_second_invocation_keeps_the_first_subset_in_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = _one_subset_per_invocation(tmp_path, monkeypatch)
    summary = json.loads((results_dir / "summary.json").read_text())
    assert sorted(r["subset"] for r in summary["results"]) == ["Tuda-De", "common_voice_19_0"]

    artifacts = tmp_path / "artifacts"
    dest = promote_mod.promote(results_dir, artifacts, run_name="two-invocations")
    assert sorted(json.loads((dest / "expected.json").read_text())) == [
        "Tuda-De", "common_voice_19_0"]
    all_ok, rows = _load_verify().verify(artifacts)
    assert all_ok, rows


def test_a_subset_finished_under_other_conditions_is_not_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def other_commit(results_dir: Path) -> None:
        sidecar = results_dir / ".conditions_Tuda-De.json"
        data = json.loads(sidecar.read_text())
        data["run_manifest"]["git"] = {"sha": "0" * 40, "dirty": False}
        sidecar.write_text(json.dumps(data))

    results_dir = _one_subset_per_invocation(tmp_path, monkeypatch, between=other_commit)
    summary = json.loads((results_dir / "summary.json").read_text())
    assert [r["subset"] for r in summary["results"]] == ["common_voice_19_0"]


def test_an_old_one_row_summary_is_rebuilt_and_promote_refuses_it_until_then(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = _one_subset_per_invocation(tmp_path, monkeypatch)
    # What a runner before #100 left behind: the last invocation's row only.
    path = results_dir / "summary.json"
    summary = json.loads(path.read_text())
    summary["results"] = [r for r in summary["results"] if r["subset"] == "common_voice_19_0"]
    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="resummarize"):
        promote_mod.promote(results_dir, tmp_path / "artifacts", run_name="old")

    assert sorted(runner.resummarize(results_dir)) == ["Tuda-De", "common_voice_19_0"]
    assert runner.resummarize(results_dir) == runner.resummarize(results_dir)  # idempotent
    dest = promote_mod.promote(results_dir, tmp_path / "artifacts", run_name="old")
    assert sorted(json.loads((dest / "expected.json").read_text())) == [
        "Tuda-De", "common_voice_19_0"]


def test_subsets_without_a_recorded_manifest_are_adopted_only_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = _one_subset_per_invocation(tmp_path, monkeypatch)
    # A dir from before per-subset sidecars: no .conditions_, one summary row.
    (results_dir / ".conditions_Tuda-De.json").unlink()
    path = results_dir / "summary.json"
    summary = json.loads(path.read_text())
    summary["results"] = [r for r in summary["results"] if r["subset"] == "common_voice_19_0"]
    path.write_text(json.dumps(summary))

    assert runner.resummarize(results_dir) == ["common_voice_19_0"]
    assert sorted(runner.resummarize(results_dir, adopt_unrecorded=True)) == [
        "Tuda-De", "common_voice_19_0"]
    steps = json.loads(path.read_text())["post_hoc"]
    assert [s["adopted_without_manifest"] for s in steps if s["step"] == "resummarize"] == [
        ["Tuda-De"]]
    dest = promote_mod.promote(results_dir, tmp_path / "artifacts", run_name="adopted")
    assert "post_hoc" in json.loads((dest / "summary.json").read_text())
