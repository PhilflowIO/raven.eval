"""Throughput: every pass over a subset is timed, kept, and added up honestly.

Per-utterance latency under a request pool measures queueing as much as the
model. These tests pin the number that replaces it as a speed figure: each
pass's audio over its wall time, appended durably (also when SIGTERM cuts the
pass short), summed across a resumed subset, carried into the artifact, and held
to its own arithmetic by ``make verify`` — without making older artifacts fail.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("tqdm")

import numpy as np
from test_resume import (
    MODEL,
    SUBSET,
    _LazyLoader,
    _NoisyAdapter,
    _run,
    _samples,
    _wire,
)
from test_verify import verify as verify_mod

from raven_asr import promote as promote_mod
from raven_asr import runner, throughput
from raven_asr.adapters.base import TranscribeResult

SIDECAR = f"throughput_{SUBSET}.jsonl"


def _passes(out: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (out / SIDECAR).read_text().splitlines()]


def _block(out: Path) -> dict[str, Any]:
    (row,) = json.loads((out / "summary.json").read_text())["results"]
    return row["throughput"]


def _audio_s(samples: list, ids: set[int] | None = None) -> float:
    return sum(
        len(s.audio) / s.sample_rate for i, s in enumerate(samples)
        if ids is None or i in ids
    )


# ── Recording one pass ───────────────────────────────────────────────────────


def test_a_pass_records_its_requests_audio_and_wall_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(throughput.HARDWARE_ENV, "1x RTX 3090 24GB")
    samples = _samples(12)
    out = tmp_path / "run"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples, fail_on={3}))
    _run(out, concurrency=5)

    (p,) = _passes(out)
    assert set(p) == set(throughput.PASS_FIELDS)
    assert (p["n_requests"], p["n_ok"]) == (12, 11)
    # A failed request transcribed nothing, so it adds no audio.
    assert p["audio_s"] == pytest.approx(_audio_s(samples, set(range(12)) - {3}), abs=1e-3)
    assert p["concurrency"] == 5
    assert p["adapter"] == runner.KNOWN_MODELS[MODEL].adapter
    assert p["hardware"] == "1x RTX 3090 24GB"
    assert p["interrupted"] is False
    assert p["wall_s"] > 0
    assert p["started_at"].endswith("Z") and p["finished_at"] >= p["started_at"]


def test_hardware_is_null_unless_the_operator_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(throughput.HARDWARE_ENV, raising=False)
    samples = _samples(3)
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples))
    _run(tmp_path)
    assert _passes(tmp_path)[0]["hardware"] is None
    assert _block(tmp_path)["hardware"] is None


def test_a_pass_that_requests_nothing_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(5)
    out = tmp_path / "run"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples, fail_on={4}))
    _run(out)
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples))
    _run(out)  # retries clip 4: a second pass
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples))
    _run(out)  # subset done: resumed from the marker, no pass at all
    assert [p["n_requests"] for p in _passes(out)] == [5, 1]


# ── A resumed subset: several passes, one block ──────────────────────────────


def test_a_resumed_subset_adds_up_its_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(12)
    out = tmp_path / "run"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples, fail_on={2, 7}))
    _run(out, concurrency=4)
    retry = _NoisyAdapter(samples)
    _wire(monkeypatch, _LazyLoader(samples), retry)
    _run(out, concurrency=4)
    assert sorted(retry.calls) == [2, 7]

    first, second = _passes(out)
    assert (first["n_requests"], first["n_ok"]) == (12, 10)
    assert (second["n_requests"], second["n_ok"]) == (2, 2)

    block = _block(out)
    assert block["passes"] == [first, second]
    assert block["total_wall_s"] == pytest.approx(first["wall_s"] + second["wall_s"], abs=1e-3)
    # Every utterance's audio, counted once: the failures only in the pass that
    # transcribed them.
    assert block["total_audio_s"] == pytest.approx(_audio_s(samples), abs=1e-3)
    assert block["n_ok"] == 12
    assert block["rtfx"] == pytest.approx(
        block["total_audio_s"] / block["total_wall_s"], rel=1e-3
    )
    assert block["utterances_per_s"] == pytest.approx(12 / block["total_wall_s"], rel=1e-3)
    assert (block["concurrency"], block["comparable"], block["comparable_reason"]) == (
        4, True, None
    )
    assert throughput.throughput_problem(block) == ""


def test_passes_at_different_concurrency_are_not_comparable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _samples(8)
    out = tmp_path / "run"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples, fail_on={1}))
    _run(out, concurrency=2)
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples))
    _run(out, concurrency=6)
    block = _block(out)
    assert block["concurrency"] == [2, 6]
    assert block["comparable"] is False
    assert "concurrency" in block["comparable_reason"]


def test_passes_of_a_discarded_log_are_discarded_with_it(
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
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples))
    _run(out)
    # The subset started over, so only the new pass describes its transcripts.
    assert [p["n_requests"] for p in _passes(out)] == [5]


# ── SIGTERM: the partial pass is flushed, the exit is unchanged ──────────────


def _terminated_run(out: Path, n: int, term_at: int) -> None:
    """Child process body: SIGTERM itself on the ``term_at``-th request."""
    samples = _samples(n)

    class _Terminated(_NoisyAdapter):
        async def atranscribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
            if len(self.calls) + 1 == term_at:
                os.kill(os.getpid(), signal.SIGTERM)
            return await super().atranscribe(audio, sample_rate)

    loader = _LazyLoader(samples)
    adapter = _Terminated(samples)
    runner._iter_loader_for_subset = lambda _s, **_kw: (loader, SUBSET)  # type: ignore[assignment]
    runner._make_adapter = lambda _spec: adapter  # type: ignore[assignment]
    _run(out, concurrency=3)
    os._exit(0)  # pragma: no cover - SIGTERM must end the process first


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_sigterm_flushes_the_partial_pass_and_still_dies_by_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n = 30
    out = tmp_path / "run"
    child = multiprocessing.get_context("fork").Process(
        target=_terminated_run, args=(out, n, 12)
    )
    child.start()
    child.join(60)
    # Exactly the exit an unhandled SIGTERM gives: a supervisor sees no change.
    assert child.exitcode == -signal.SIGTERM
    assert not (out / f".done_{SUBSET}.json").exists()

    (partial,) = _passes(out)
    assert partial["interrupted"] is True
    recorded = sum(1 for _ in (out / f"predictions_{SUBSET}.jsonl").open())
    assert 0 < partial["n_requests"] == recorded < n

    # The resumed run adds its pass to the interrupted one.
    samples = _samples(n)
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples))
    _run(out, concurrency=3)
    first, second = _passes(out)
    assert (first["interrupted"], second["interrupted"]) == (True, False)
    assert first["n_requests"] + second["n_requests"] == n
    block = _block(out)
    assert block["n_ok"] == n
    assert block["total_audio_s"] == pytest.approx(_audio_s(samples), abs=1e-3)
    assert throughput.throughput_problem(block) == ""


# ── The block's arithmetic ───────────────────────────────────────────────────


def _pass(**kw: Any) -> dict[str, Any]:
    base = {
        "started_at": "2026-11-02T10:00:00.000Z", "finished_at": "2026-11-02T10:01:40.000Z",
        "wall_s": 100.0, "n_requests": 50, "n_ok": 50, "audio_s": 2500.0,
        "concurrency": 8, "adapter": "vllm_openai", "hardware": "1x RTX 3090 24GB",
        "interrupted": False,
    }
    return {**base, **kw}


def test_summarize_math() -> None:
    block = throughput.summarize([
        _pass(), _pass(wall_s=60.0, audio_s=900.0, n_requests=20, n_ok=19, interrupted=True),
    ])
    assert block is not None
    assert (block["total_wall_s"], block["total_audio_s"], block["n_ok"]) == (160.0, 3400.0, 69)
    assert block["rtfx"] == round(3400 / 160, 4) == 21.25
    assert block["utterances_per_s"] == round(69 / 160, 4)
    assert (block["concurrency"], block["hardware"], block["comparable"]) == (
        8, "1x RTX 3090 24GB", True
    )
    assert throughput.summarize([]) is None


def test_different_hardware_is_not_comparable() -> None:
    block = throughput.summarize([_pass(), _pass(hardware="1x A100 80GB")])
    assert block is not None
    assert block["hardware"] == ["1x RTX 3090 24GB", "1x A100 80GB"]
    assert block["comparable"] is False
    assert "hardware" in block["comparable_reason"]


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("rtfx", 30.0, "rtfx="),
        ("total_wall_s", 120.0, "total_wall_s="),
        ("total_audio_s", 3000.0, "total_audio_s="),
        ("utterances_per_s", 9.0, "utterances_per_s="),
        ("n_ok", 1, "n_ok="),
    ],
)
def test_an_edited_block_contradicts_itself(field: str, value: Any, needle: str) -> None:
    block = throughput.summarize([_pass(), _pass(wall_s=60.0, audio_s=900.0)])
    assert throughput.throughput_problem(block) == ""
    assert needle in throughput.throughput_problem({**block, field: value})


def test_a_torn_pass_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / SIDECAR
    throughput.append_pass(path, _pass())
    with path.open("a") as fh:
        fh.write('{"started_at": "2026-')  # killed mid-write
    throughput.append_pass(path, _pass(wall_s=5.0))
    assert [p["wall_s"] for p in throughput.read_passes(path)] == [100.0, 5.0]


# ── Into the artifact, and through make verify ───────────────────────────────


def _promoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    samples = _samples(10)
    results = tmp_path / "results" / "model"
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples, fail_on={6}))
    _run(results)
    _wire(monkeypatch, _LazyLoader(samples), _NoisyAdapter(samples))
    _run(results)
    return promote_mod.promote(results, tmp_path / "artifacts", run_name="r")


def _edit_summary(model_dir: Path, edit: Any) -> None:
    path = model_dir / "summary.json"
    summary = json.loads(path.read_text())
    edit(summary["results"][0])
    path.write_text(json.dumps(summary))


def test_promote_carries_the_passes_and_verify_accepts_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = _promoted(tmp_path, monkeypatch)
    assert [p["n_requests"] for p in _passes(dest)] == [10, 1]
    assert len(_block(dest)["passes"]) == 2
    # Throughput is a measurement, never a committed expectation.
    assert not any("rtfx" in v for v in json.loads((dest / "expected.json").read_text()).values())
    all_ok, rows = verify_mod.verify(tmp_path / "artifacts")
    assert all_ok, rows


def test_verify_fails_an_inconsistent_throughput_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = _promoted(tmp_path, monkeypatch)
    _edit_summary(dest, lambda row: row["throughput"].update(rtfx=row["throughput"]["rtfx"] * 2))
    all_ok, rows = verify_mod.verify(tmp_path / "artifacts")
    assert not all_ok
    assert any("throughput: rtfx=" in r["detail"] for r in rows)


def test_an_artifact_without_throughput_still_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = _promoted(tmp_path, monkeypatch)
    _edit_summary(dest, lambda row: row.pop("throughput"))
    (dest / SIDECAR).unlink()
    all_ok, rows = verify_mod.verify(tmp_path / "artifacts")
    assert all_ok, rows
