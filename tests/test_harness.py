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
from raven_asr.datasets.base import Sample


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

    verify = _load_verify()
    all_ok, rows = verify.verify(artifacts)
    assert rows
    assert all_ok, f"promoted artifact failed re-score: {rows}"


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
    spec = ModelSpec(
        model_id="nvidia/canary-1b-v2",
        adapter="vllm_openai",
        label="nvidia-canary-1b-v2",
        base_url_env="NEMO_BENCH_URL",
        api_key_env="NEMO_BENCH_API_KEY",
    )
    runner._make_adapter(spec)
    assert captured == {
        "provider_id": "nvidia-canary-1b-v2",
        "model_id": "nvidia/canary-1b-v2",
        "base_url_env": "NEMO_BENCH_URL",
        "api_key_env": "NEMO_BENCH_API_KEY",
    }


def test_make_adapter_default_none_keeps_legacy_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Specs without endpoint envs must resolve to the adapter's legacy
    defaults (VLLM_PRIMELINE_*)."""
    vllm_mod = pytest.importorskip("raven_asr.adapters.vllm_openai")
    captured = _capture_vllm_adapter(monkeypatch)
    spec = KNOWN_MODELS["primeline/whisper-large-v3-german"]
    assert spec.base_url_env is None and spec.api_key_env is None
    runner._make_adapter(spec)
    assert captured == {
        "provider_id": "primeline-whisper-large-v3-german",
        "model_id": "primeline/whisper-large-v3-german",
        "base_url_env": vllm_mod.DEFAULT_BASE_URL_ENV,
        "api_key_env": vllm_mod.DEFAULT_API_KEY_ENV,
    }
    assert captured["base_url_env"] == "VLLM_PRIMELINE_URL"
    assert captured["api_key_env"] == "VLLM_PRIMELINE_API_KEY"


def test_wave2_registry_entries_carry_endpoint_envs() -> None:
    """The flow.raven#5137 wave-2 entries bind to their bench env pairs."""
    nemo = {"nvidia/parakeet-tdt-0.6b-v3", "nvidia/canary-1b-v2"}
    vllm = {
        "ibm-granite/granite-speech-4.1-2b-plus",
        "CohereLabs/cohere-transcribe-03-2026",
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "microsoft/Phi-4-multimodal-instruct",
    }
    for key in nemo | vllm:
        spec = KNOWN_MODELS[key]
        assert spec.adapter == "vllm_openai"
        expected_url = "NEMO_BENCH_URL" if key in nemo else "VLLM_BENCH_URL"
        expected_key = "NEMO_BENCH_API_KEY" if key in nemo else "VLLM_BENCH_API_KEY"
        assert spec.base_url_env == expected_url
        assert spec.api_key_env == expected_key


def test_promote_refuses_empty_results(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        promote_mod.promote(empty, tmp_path / "artifacts", run_name="x")
