"""Unattended full runs: per-utterance resume, a bounded lazy pool, hard aborts.

A full corpus is hours of requests. These tests pin what makes such a run
survivable: every transcript is on disk the moment it exists, a rerun requests
only what is missing or failed, a run killed with SIGKILL resumes to the exact
numbers an uninterrupted run gets, the sample stream is never materialised, and
an endpoint serving the wrong model stops the run instead of filling it.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import signal
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("tqdm")

import numpy as np

from raven_asr import runner
from raven_asr.adapters.base import ServedModelMismatchError, TranscribeResult
from raven_asr.datasets.base import RowTally, Sample

MODEL = "primeline/whisper-large-v3-german"
SUBSET = "Tuda-De"
WORDS = ["hallo", "welt", "guten", "tag", "wie", "geht", "es", "dir", "heute", "abend"]


def _samples(n: int) -> list[Sample]:
    return [
        Sample(
            audio=np.array([float(i), 0.0, 0.0], dtype=np.float32),
            sample_rate=16000,
            reference=" ".join(WORDS[(i + j) % len(WORDS)] for j in range(3 + i % 4)),
            sample_id=f"{SUBSET}-{i}",
            subset=SUBSET,
        )
        for i in range(n)
    ]


class _LazyLoader:
    """Yields samples one at a time and counts how many it has handed out."""

    def __init__(self, samples: list[Sample], dropped: dict[str, int] | None = None):
        self._samples = samples
        self._dropped = dropped or {}
        self.pulled = 0
        self.tally = RowTally()

    def iter_samples(self, subset: str, limit: int | None = None) -> Iterator[Sample]:
        tally = self.tally = RowTally()
        for s in self._samples[:limit]:
            tally.rows_read += 1
            tally.yielded += 1
            self.pulled += 1
            yield s
        for reason, n in self._dropped.items():
            tally.rows_read += n
            tally.dropped[reason] = n


class _NoisyAdapter:
    """Deterministic, imperfect transcripts, so the WER under test is not 0."""

    provider_id = "noisy"
    model_id = "noisy"

    def __init__(self, samples: list[Sample], fail_on: set[int] | None = None):
        self._refs = {int(s.audio[0]): s.reference for s in samples}
        self._fail_on = fail_on or set()
        self.calls: list[int] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.closed = False

    async def atranscribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        i = int(audio[0])
        self.calls.append(i)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            # Finish out of order, so completion order differs from split order.
            await asyncio.sleep(0.001 * ((i * 7) % 5))
            if i in self._fail_on:
                raise TimeoutError("upstream timed out")
            words = self._refs[i].split()
            if i % 3 == 0:
                words = words[:-1]  # a deletion
            if i % 4 == 0:
                words = [*words, "äh"]  # an insertion
            return TranscribeResult(text=" ".join(words), latency_s=0.001, raw={})
        finally:
            self.in_flight -= 1

    async def aclose(self) -> None:
        self.closed = True


def _wire(
    monkeypatch: pytest.MonkeyPatch, loader: _LazyLoader, adapter: Any
) -> None:
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset", lambda _s, **_kw: (loader, SUBSET)
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: adapter)


def _run(out: Path, concurrency: int = 4) -> None:
    runner.run(
        model_key=MODEL, subsets=[SUBSET], limit=None, out_dir=out,
        concurrency=concurrency,
    )


def _lines(out: Path) -> list[dict[str, Any]]:
    text = (out / f"predictions_{SUBSET}.jsonl").read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines()]


def _scores(out: Path) -> list[dict[str, Any]]:
    return json.loads((out / "summary.json").read_text())["results"]


def test_a_rerun_requests_only_failed_utterances_and_writes_each_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(12)
    out = tmp_path / "run"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples, fail_on={2, 7}))
    _run(out)
    first = _lines(out)
    assert [x["sample_id"] for x in first] == [s.sample_id for s in samples]
    assert [x["sample_id"] for x in first if x.get("error")] == ["Tuda-De-2", "Tuda-De-7"]
    assert not (out / f".done_{SUBSET}.json").exists()

    retry = _NoisyAdapter(samples)
    _wire(monkeypatch, _LazyLoader(samples), retry)
    _run(out)
    assert sorted(retry.calls) == [2, 7]
    final = _lines(out)
    assert [x["sample_id"] for x in final] == [s.sample_id for s in samples]
    assert not any(x.get("error") for x in final)
    assert (out / f".done_{SUBSET}.json").exists()
    assert not (out / f".progress_{SUBSET}.json").exists()

    # The resumed run scores exactly what one clean run scores.
    clean = tmp_path / "clean"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples))
    _run(clean)
    assert _scores(out) == _scores(clean)
    assert _lines(out) == _lines(clean)


def _killed_run(out: Path, n: int, kill_at: int) -> None:
    """Child process body: run until the ``kill_at``-th request, then SIGKILL."""
    samples = _samples(n)

    class _Dying(_NoisyAdapter):
        async def atranscribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
            if len(self.calls) + 1 == kill_at:
                os.kill(os.getpid(), signal.SIGKILL)
            return await super().atranscribe(audio, sample_rate)

    loader = _LazyLoader(samples)
    adapter = _Dying(samples)
    runner._iter_loader_for_subset = lambda _s, **_kw: (loader, SUBSET)  # type: ignore[assignment]
    runner._make_adapter = lambda _spec: adapter  # type: ignore[assignment]
    _run(out)
    os._exit(0)  # pragma: no cover - the kill above must happen first


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_killed_run_resumes_to_the_scores_of_an_uninterrupted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n = 40
    samples = _samples(n)

    clean = tmp_path / "clean"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples))
    _run(clean)

    killed = tmp_path / "killed"
    child = multiprocessing.get_context("fork").Process(
        target=_killed_run, args=(killed, n, 23)
    )
    child.start()
    child.join(60)
    assert child.exitcode == -signal.SIGKILL
    assert not (killed / "summary.json").exists(), "killed inside the first subset"
    survived = {x["sample_id"] for x in _lines(killed)}
    assert 0 < len(survived) < n

    resume = _NoisyAdapter(samples)
    _wire(monkeypatch, _LazyLoader(samples), resume)
    _run(killed)
    # Only what the killed run had not written down is requested again ...
    assert {f"{SUBSET}-{i}" for i in resume.calls} == {
        s.sample_id for s in samples
    } - survived
    # ... and the result is indistinguishable from a run that was never killed.
    assert _lines(killed) == _lines(clean)
    assert _scores(killed) == _scores(clean)
    assert (killed / f".done_{SUBSET}.json").read_text() == (
        clean / f".done_{SUBSET}.json"
    ).read_text()


def test_a_torn_last_line_is_ignored_and_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(6)
    out = tmp_path / "run"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples, fail_on={5}))
    _run(out)
    path = out / f"predictions_{SUBSET}.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:3]) + "\n" + lines[3][: len(lines[3]) // 2])

    again = _NoisyAdapter(samples)
    _wire(monkeypatch, _LazyLoader(samples), again)
    _run(out)
    assert sorted(again.calls) == [3, 4, 5]
    assert len(_lines(out)) == 6


def test_a_line_appended_after_a_torn_one_survives_a_second_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "run"
    out.mkdir()
    manifest = runner._run_manifest(runner.KNOWN_MODELS[MODEL], None, None)
    log = runner._PredictionLog(out, SUBSET, manifest)
    log.append(runner._Outcome(sample_id="Tuda-De-0", reference="a", duration_s=1.0,
                               prediction="a", latency_s=0.1))
    log.close()
    with log.path.open("a") as fh:
        fh.write('{"reference": "b", "predic')  # killed mid-write
    log = runner._PredictionLog(out, SUBSET, manifest)
    log.append(runner._Outcome(sample_id="Tuda-De-2", reference="c", duration_s=1.0,
                               prediction="c", latency_s=0.1))
    log.close()
    assert set(runner._PredictionLog(out, SUBSET, manifest).prior) == {
        "Tuda-De-0", "Tuda-De-2"
    }


def test_a_log_from_other_run_conditions_is_not_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(5)
    out = tmp_path / "run"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples, fail_on={4}))
    _run(out)

    real = runner._run_manifest

    def _other_commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        manifest = real(*args, **kwargs)
        manifest["git"] = {"sha": "0" * 40, "dirty": False}
        return manifest

    monkeypatch.setattr(runner, "_run_manifest", _other_commit)
    again = _NoisyAdapter(samples)
    _wire(monkeypatch, _LazyLoader(samples), again)
    _run(out)
    assert sorted(again.calls) == [0, 1, 2, 3, 4]


def test_a_changed_reference_is_not_answered_from_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(4)
    out = tmp_path / "run"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples, fail_on={3}))
    _run(out)

    edited = [
        Sample(audio=s.audio, sample_rate=s.sample_rate, sample_id=s.sample_id,
               subset=s.subset, reference=s.reference + (" neu" if i == 1 else ""))
        for i, s in enumerate(samples)
    ]
    again = _NoisyAdapter(edited)
    _wire(monkeypatch, _LazyLoader(edited), again)
    _run(out)
    assert sorted(again.calls) == [1, 3]


def test_the_pool_never_holds_more_samples_than_requests_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(60)
    held: list[int] = []

    class _Watched(_NoisyAdapter):
        def __init__(self, loader: _LazyLoader, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            self._loader = loader
            self.done = 0

        async def atranscribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
            held.append(self._loader.pulled - self.done)
            try:
                return await super().atranscribe(audio, sample_rate)
            finally:
                self.done += 1

    loader = _LazyLoader(samples, dropped={"empty_reference": 2})
    adapter = _Watched(loader, samples)
    _wire(monkeypatch, loader, adapter)
    _run(tmp_path / "run", concurrency=3)
    assert adapter.max_in_flight == 3
    assert max(held) <= 3
    assert loader.pulled == 60
    # The drop tally is read after the lazy pass is exhausted, so it is complete.
    (row,) = _scores(tmp_path / "run")
    assert (row["n_dropped"], row["dropped_by_reason"]) == (2, {"empty_reference": 2})
    assert adapter.closed


def test_a_served_model_mismatch_aborts_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(30)

    class _Swapped(_NoisyAdapter):
        async def atranscribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
            if int(audio[0]) >= 5:
                raise ServedModelMismatchError("serves another model")
            return await super().atranscribe(audio, sample_rate)

    adapter = _Swapped(samples)
    out = tmp_path / "run"
    _wire(monkeypatch, _LazyLoader(samples), adapter)
    with pytest.raises(ServedModelMismatchError):
        _run(out, concurrency=2)
    assert not (out / f".done_{SUBSET}.json").exists()
    assert adapter.closed
    # The mismatch is not recorded as a failed utterance of this model.
    assert not any(x.get("error") for x in _lines(out))


def test_the_models_path_bounds_each_model_by_its_own_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(30)
    adapters = {
        "primeline/whisper-large-v3-german": _NoisyAdapter(samples),
        "primeline/whisper-large-v3-turbo-german": _NoisyAdapter(samples),
    }
    monkeypatch.setattr(
        runner, "_iter_loader_for_subset",
        lambda _s, **_kw: (_LazyLoader(samples), SUBSET),
    )
    monkeypatch.setattr(runner, "_make_adapter", lambda spec: adapters[spec.model_id])
    asyncio.run(runner.run_multi_async(
        model_keys=list(adapters), subsets=[SUBSET], limit=None, out_root=tmp_path,
        concurrency_overrides={
            "primeline/whisper-large-v3-german": 2,
            "primeline/whisper-large-v3-turbo-german": 5,
        },
    ))
    assert adapters["primeline/whisper-large-v3-german"].max_in_flight == 2
    assert adapters["primeline/whisper-large-v3-turbo-german"].max_in_flight == 5
    assert all(a.closed for a in adapters.values())


def test_the_runner_cli_threads_per_model_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def _capture(**kwargs: Any) -> dict[str, Path]:
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(runner, "run_multi_async", _capture)
    runner.main([
        "--models", "primeline/whisper-large-v3-german,openai/whisper-large-v3",
        "--subsets", SUBSET, "--out", str(tmp_path), "--concurrency", "8",
        "--model-concurrency", "openai/whisper-large-v3=3",
    ])
    assert seen["concurrency_overrides"] == {
        "primeline/whisper-large-v3-german": 8,
        "openai/whisper-large-v3": 3,
    }
