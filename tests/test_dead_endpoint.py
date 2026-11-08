"""A dead endpoint stops the subset instead of failing the rest of the corpus.

raven.eval#93: a vLLM engine that had died answered 503 to every request while
its API server stayed up, and the harness spent two hours recording 1,007 failed
clips. Everything here runs against mocked endpoints.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from raven_asr import reproduce, runner
from raven_asr.adapters.base import TranscribeResult
from raven_asr.adapters.vllm_openai import VllmOpenAIAdapter
from raven_asr.datasets.base import Sample
from raven_asr.retry import (
    EndpointDeadError,
    TransientStreamError,
    is_endpoint_failure,
)

MODEL = "primeline/whisper-large-v3-german"
SUBSET = "Tuda-De"


def _samples(n: int) -> list[Sample]:
    return [
        Sample(
            audio=np.array([float(i)], dtype=np.float32), sample_rate=16000,
            reference=f"wort {i}", sample_id=f"{SUBSET}-{i}", subset=SUBSET,
        )
        for i in range(n)
    ]


class _Loader:
    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples
        self.tally = None

    def iter_samples(self, subset: str, limit: int | None = None) -> Any:
        yield from self._samples


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://bench.invalid/v1/audio/transcriptions")
    return httpx.HTTPStatusError("boom", request=request, response=httpx.Response(status, request=request))


class _Dying:
    """Answers the first ``alive`` clips, then fails every request with ``error``."""

    provider_id = "dying"
    model_id = "dying"

    def __init__(self, alive: int, error: Exception, health: bool | None = None, probes: bool = False):
        self._alive, self._error, self._health = alive, error, health
        self.calls = 0
        self.probed = 0
        if probes:
            self.aprobe_health = self._probe

    async def _probe(self) -> bool | None:
        self.probed += 1
        return self._health

    async def atranscribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        await asyncio.sleep(0)
        i = self.calls
        self.calls += 1
        if i >= self._alive:
            raise self._error
        return TranscribeResult(text=f"wort {int(audio[0])}", latency_s=0.001, raw={})

    async def aclose(self) -> None:
        pass


def _go(monkeypatch: pytest.MonkeyPatch, adapter: Any, out: Path, n: int = 500) -> None:
    monkeypatch.setattr(runner, "_iter_loader_for_subset", lambda _s, **_k: (_Loader(_samples(n)), SUBSET))
    monkeypatch.setattr(runner, "_make_adapter", lambda _spec: adapter)
    runner.run(model_key=MODEL, subsets=[SUBSET], limit=None, out_dir=out, concurrency=2)


@pytest.mark.parametrize(
    "error",
    [_status_error(503), _status_error(500), httpx.ConnectError("refused"), httpx.ReadTimeout("slow")],
)
def test_a_streak_of_endpoint_failures_stops_the_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    adapter = _Dying(alive=5, error=error)
    with pytest.raises(EndpointDeadError, match="in a row"):
        _go(monkeypatch, adapter, tmp_path)
    assert adapter.calls < 5 + runner.DEAD_ENDPOINT_STREAK + 4
    assert not (tmp_path / f".done_{SUBSET}.json").exists()


def test_a_failed_health_probe_stops_it_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _Dying(alive=0, error=_status_error(503), health=False, probes=True)
    with pytest.raises(EndpointDeadError, match="health probe"):
        _go(monkeypatch, adapter, tmp_path)
    assert adapter.probed == 1
    assert adapter.calls < runner.DEAD_ENDPOINT_STREAK


def test_a_healthy_probe_does_not_stop_it_before_the_streak_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _Dying(alive=0, error=_status_error(503), health=True, probes=True)
    with pytest.raises(EndpointDeadError, match="in a row"):
        _go(monkeypatch, adapter, tmp_path)
    assert adapter.calls >= runner.DEAD_ENDPOINT_STREAK


def test_clip_failures_never_add_up_to_a_dead_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every clip rejected with a 400: the server answers, the run finishes."""
    adapter = _Dying(alive=0, error=_status_error(400))
    _go(monkeypatch, adapter, tmp_path, n=40)
    assert adapter.calls == 40
    assert not (tmp_path / f".done_{SUBSET}.json").exists()


def test_a_success_ends_the_streak() -> None:
    watch = runner._EndpointWatch(_Dying(0, RuntimeError()), concurrency=1)
    bad = runner._Outcome("a", "r", 1.0, error="503", endpoint_failure=True)
    good = runner._Outcome("b", "r", 1.0, prediction="ok")

    async def go() -> None:
        for _ in range(3):
            for _ in range(runner.DEAD_ENDPOINT_STREAK - 1):
                await watch.observe(bad)
            await watch.observe(good)

    asyncio.run(go())
    assert watch.streak == 0


def test_the_limit_covers_everything_in_flight() -> None:
    watch = runner._EndpointWatch(_Dying(0, RuntimeError()), concurrency=32)
    bad = runner._Outcome("a", "r", 1.0, error="503", endpoint_failure=True)

    async def go() -> None:
        for _ in range(63):
            await watch.observe(bad)
        with pytest.raises(EndpointDeadError):
            await watch.observe(bad)

    asyncio.run(go())


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_status_error(503), True),
        (_status_error(500), True),
        (_status_error(429), False),
        (_status_error(400), False),
        (httpx.ConnectError("x"), True),
        (httpx.ReadTimeout("x"), True),
        (TransientStreamError("x"), True),
        (ValueError("x"), False),
    ],
)
def test_what_counts_as_the_endpoints_fault(exc: Exception, expected: bool) -> None:
    assert is_endpoint_failure(exc) is expected


def _adapter(monkeypatch: pytest.MonkeyPatch, handler: Any) -> VllmOpenAIAdapter:
    monkeypatch.setenv("VLLM_BENCH_URL", "http://bench.invalid/v1/")
    monkeypatch.delenv("BENCH_STT_API_KEY", raising=False)
    return VllmOpenAIAdapter(
        provider_id="p", model_id="m", base_url_env="VLLM_BENCH_URL",
        api_key_env="BENCH_STT_API_KEY", transport=httpx.MockTransport(handler),
    )


def _probe(adapter: VllmOpenAIAdapter) -> bool | None:
    async def go() -> bool | None:
        try:
            return await adapter.aprobe_health()
        finally:
            await adapter.aclose()

    return asyncio.run(go())


@pytest.mark.parametrize(("status", "expected"), [(200, True), (503, False), (500, False), (404, None)])
def test_health_probe_reads_the_status(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: bool | None
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(status)

    assert _probe(_adapter(monkeypatch, handler)) is expected
    assert seen == ["http://bench.invalid/health"]


@pytest.mark.parametrize(
    ("error", "expected"), [(httpx.ConnectError("refused"), False), (httpx.ReadTimeout("slow"), None)]
)
def test_health_probe_tells_refused_from_slow(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: bool | None
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    assert _probe(_adapter(monkeypatch, handler)) is expected


def test_reproduce_exits_5_on_a_dead_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run(**_kw: Any) -> Path:
        raise EndpointDeadError("health probe failed")

    monkeypatch.setattr(reproduce, "run", fake_run)
    rc = reproduce.main(["--metric", "wer", "--dataset", "fleurs", "--model", MODEL, "--out", str(tmp_path)])
    assert rc == reproduce.EXIT_ENDPOINT_DEAD == 5
    assert "health probe failed" in capsys.readouterr().err
    assert not list(tmp_path.rglob(".done_*"))
