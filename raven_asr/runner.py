"""CLI runner — async, per-subset resumable, flozi-comparable YAML output."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import sys
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, cast

from raven_eval_core.entities import entity_hit_rate
from tqdm import tqdm

from raven_eval_core.contract import resolved_contract
from raven_eval_core.flozi_wer import evaluate, runaway_stats
from raven_eval_core.run_manifest import (
    MANIFEST_FIELDS,
    SCHEMA_VERSION,
    build_run_manifest,
    same_run_conditions,
)

from . import serving, throughput
from .adapters.base import ASRAdapter, ServedModelMismatchError
from .config import (
    FLOZI_REFERENCE_WER,
    RUNAWAY_LENGTH_RATIO,
    KNOWN_MODELS,
    MODAL_APP_NAMES,
    WER_DATASETS,
    ModelSpec,
    resolve_wer_dataset,
)
from .datasets import load_loader_class
from .datasets.base import DatasetLoader, RowTally, Sample
from .model_index import ResultEntry, build_model_index, write_yaml

logger = logging.getLogger("raven_asr.runner")


# Per-adapter default concurrency. Override via ``--concurrency N`` on the CLI.
DEFAULT_CONCURRENCY: dict[str, int] = {
    "modal_app": 20,
    "vllm_openai": 10,
    "voxtral_mistral": 8,
    "deepgram": 20,
    "openai_whisper": 3,
    # xAI documents 10 requests/s for REST STT; 8 in flight stays under it.
    "xai": 8,
    # Streaming sessions run at real-time pace, so throughput is concurrency x
    # audio speed. What the account limits is new sessions per minute, and the
    # adapter paces those itself; this only has to cover the sessions open at
    # once at a paid account's 100/min with utterances of a few seconds.
    "assemblyai_realtime": 16,
}


# The client-side inference stack, recorded in the run manifest beside the
# scorer libraries every harness shares (raven_eval_core.run_manifest).
MANIFEST_LIBRARIES: tuple[str, ...] = ("datasets", "soundfile", "httpx", "modal")


@dataclass(frozen=True)
class _Outcome:
    """One attempted utterance — a transcription or the reason there is none.

    ``prediction`` is None exactly when the request failed, and ``error`` then
    says why. A failure is recorded, never dropped: dropping it would score the
    model only on the clips it managed, and a model that times out on the
    hardest clips would read *better* than one that transcribes them badly.
    """

    sample_id: str
    reference: str
    # Audio length is a property of the input, not of the model: it is what
    # turns latency into RTFx and a per-minute price into a cost.
    duration_s: float
    prediction: str | None = None
    latency_s: float | None = None
    error: str | None = None
    # Set only for a model with an output format (raven_asr.output_formats):
    # the server's transcript as returned, and the speaker turns it carried.
    # ``prediction`` is then the spoken text parsed out of it.
    raw_prediction: str | None = None
    segments: list[dict[str, object]] | None = None


@dataclass(frozen=True)
class _RunResult:
    subset: str
    outcomes: list[_Outcome] = field(default_factory=list)
    # Rows the loader dropped before any request was made, by reason; None when
    # the loader keeps no tally (see datasets.base.RowTally).
    dropped: dict[str, int] | None = None

    @property
    def succeeded(self) -> list[_Outcome]:
        return [o for o in self.outcomes if o.error is None]

    @property
    def n_failed(self) -> int:
        return sum(1 for o in self.outcomes if o.error is not None)


def _make_adapter(spec: ModelSpec) -> ASRAdapter:
    if spec.adapter == "vllm_openai":
        from .adapters.vllm_openai import VllmOpenAIAdapter
        if spec.base_url_env is None:
            raise ValueError(
                f"{spec.label}: a self-hosted model needs base_url_env "
                f"(NEMO_BENCH_URL or VLLM_BENCH_URL) in raven_asr.config"
            )
        return VllmOpenAIAdapter(
            provider_id=spec.label,
            model_id=spec.model_id,
            base_url_env=spec.base_url_env,
            api_key_env=spec.api_key_env,
            output_format=spec.output_format,
        )
    if spec.adapter == "voxtral_mistral":
        from .adapters.voxtral_mistral import VoxtralMistralAdapter
        return VoxtralMistralAdapter(provider_id=spec.label, model_id=spec.model_id)
    if spec.adapter == "deepgram":
        from .adapters.deepgram import DeepgramAdapter
        return DeepgramAdapter(provider_id=spec.label, model_id=spec.model_id)
    if spec.adapter == "openai_whisper":
        from .adapters.openai_whisper import OpenAIWhisperAdapter
        return OpenAIWhisperAdapter(provider_id=spec.label, model_id=spec.model_id)
    if spec.adapter == "xai":
        from .adapters.xai import XaiSttAdapter
        return XaiSttAdapter(
            provider_id=spec.label,
            model_id=spec.model_id,
            api_key_env=spec.api_key_env or "XAI_API_KEY",
        )
    if spec.adapter == "assemblyai_realtime":
        from .adapters.assemblyai_realtime import AssemblyAIRealtimeAdapter
        return AssemblyAIRealtimeAdapter(
            provider_id=spec.label,
            model_id=spec.model_id,
            api_key_env=spec.api_key_env or "ASSEMBLYAI_API_KEY",
        )
    if spec.adapter == "modal_app":
        from .adapters.modal_app import ModalAppAdapter
        app_name = MODAL_APP_NAMES.get(spec.label)
        if app_name is None:
            raise ValueError(
                f"no MODAL_APP_NAMES entry for {spec.label!r}; "
                f"add it in raven_asr.config"
            )
        return ModalAppAdapter(
            provider_id=spec.label, model_id=spec.model_id, app_name=app_name
        )
    raise ValueError(f"unknown adapter: {spec.adapter}")


def _dataset_pin(subset: str, revision: str | None = None) -> tuple[str | None, str | None]:
    """``(revision, sha256)`` the run actually reads ``subset`` at.

    An explicit ``revision`` (``--dataset-revision``) wins over the registry pin.
    Recorded beside every score, because a WER is a statement about one fixed
    set of references and is meaningless without naming which.
    """
    dataset_id, _ = resolve_wer_dataset(subset)
    spec = WER_DATASETS[dataset_id]
    return (revision if revision is not None else spec.revision), spec.sha256


def _iter_loader_for_subset(
    subset: str,
    *,
    streaming: bool = False,
    revision: str | None = None,
) -> tuple[DatasetLoader, str]:
    """Build the loader that owns ``subset`` and the subset name it expects.

    ``subset`` is either a ``WER_DATASETS`` id or a ``german-mixed`` subset name;
    ``resolve_wer_dataset`` maps both onto the owning dataset. Public repo: only
    public datasets ship here — Raven's private meeting corpus (Tier-3) is
    deliberately not portable and stays internal.
    """
    dataset_id, internal_subset = resolve_wer_dataset(subset)
    rev, _ = _dataset_pin(subset, revision)
    loader_cls = load_loader_class(dataset_id)
    # Local snapshot unless the caller asks to stream: a full run streamed from
    # the Hub dies on Hub timeouts (raven_asr.datasets.hf_snapshot).
    loader = loader_cls(streaming=streaming, revision=rev)
    return loader, internal_subset


def _dropped_rows(loader: DatasetLoader) -> dict[str, int] | None:
    """The loader's per-reason drop counts after a pass, or None if it has none.

    Read after ``iter_samples`` is exhausted — the tally is only final then.
    """
    tally = getattr(loader, "tally", None)
    return dict(tally.dropped) if isinstance(tally, RowTally) else None


def _safe(name: str) -> str:
    """Filesystem-safe subset-marker filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


async def _attempt(adapter: ASRAdapter, sample: Sample) -> _Outcome:
    """Transcribe one sample; a failure becomes an outcome, not an absence."""
    duration_s = len(sample.audio) / float(sample.sample_rate)
    try:
        r = await adapter.atranscribe(sample.audio, sample.sample_rate)
    except ServedModelMismatchError:
        # A fact about the endpoint, not this clip: every later transcript would
        # be another model's under this model's name. Abort, never record.
        raise
    except Exception as exc:
        logger.exception(
            "[%s] transcribe failed on %s", adapter.provider_id, sample.sample_id
        )
        return _Outcome(
            sample_id=sample.sample_id,
            reference=sample.reference,
            duration_s=duration_s,
            error=f"{type(exc).__name__}: {exc}",
        )
    return _Outcome(
        sample_id=sample.sample_id,
        reference=sample.reference,
        duration_s=duration_s,
        prediction=r.text,
        latency_s=r.latency_s,
        raw_prediction=r.raw_text,
        segments=r.segments,
    )


_END: Any = object()


def _prediction_line(o: _Outcome) -> dict[str, object]:
    """The ``predictions_*.jsonl`` row of one attempted utterance."""
    line: dict[str, object] = {
        "reference": o.reference,
        "prediction": o.prediction,
        "latency_s": o.latency_s,
        "sample_id": o.sample_id,
        "duration_s": round(o.duration_s, 6),
    }
    if o.raw_prediction is not None:
        # Only a model with an output format writes these: the server's string
        # the prediction was parsed from, so the parse can be audited and redone.
        line["raw_prediction"] = o.raw_prediction
        line["segments"] = o.segments
    if o.error is not None:
        line["error"] = o.error
    return line


class _PredictionLog:
    """``predictions_<subset>.jsonl`` while a subset is in progress.

    Every outcome is appended and flushed the moment it exists, so a run killed
    20 hours into a corpus keeps 20 hours of transcripts. The log is bound to
    the run manifest it was written under through a ``.conditions_<subset>.json``
    sidecar, which outlives the pass: it is how a later invocation tells whether
    the subset's rows — in progress or complete — were measured as it measures.

    The log trusts its caller to have made that check: :func:`_gate_existing_rows`
    runs before a log is opened and moves rows from other conditions aside, so
    whatever is still on disk here may be resumed.

    On resume, an utterance with a recorded transcription for the *same
    reference text* is not requested again; a failed one is. The newest line
    per ``sample_id`` wins, and a torn last line (the process died mid-write) is
    ignored. When the subset finishes, :func:`_write_predictions` rewrites the
    file atomically to exactly one line per utterance, in split order.
    """

    def __init__(
        self, out_dir: Path, subset: str, manifest: dict[str, Any]
    ) -> None:
        self.path = out_dir / f"predictions_{_safe(subset)}.jsonl"
        self.conditions_path = _conditions_path(out_dir, subset)
        # The passes that produced this log. They describe these transcripts, so
        # they are kept and discarded with them, never apart.
        self.throughput_path = throughput.throughput_path(out_dir, _safe(subset))
        self._legacy_progress_path = _legacy_progress_path(out_dir, subset)
        self._manifest = manifest
        self._fh: Any = None
        self.prior: dict[str, dict[str, Any]] = self._read_prior()
        if self.prior:
            logger.info(
                "resuming %s: %d utterance(s) already recorded",
                self.path.name, len(self.prior),
            )
        else:
            # No transcript of this subset will be resumed, so no earlier pass
            # describes the work this run's summary will report.
            self.throughput_path.unlink(missing_ok=True)

    def _read_prior(self) -> dict[str, dict[str, Any]]:
        prior: dict[str, dict[str, Any]] = {}
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return prior
        for raw in text.splitlines():
            try:
                line = json.loads(raw)
            except json.JSONDecodeError:
                continue  # torn write from a killed process
            if isinstance(line, dict) and isinstance(line.get("sample_id"), str):
                prior[line["sample_id"]] = line
        return prior

    def cached(self, sample: Sample) -> _Outcome | None:
        """The recorded transcription of ``sample``, or None if it must be run."""
        line = self.prior.get(sample.sample_id)
        if (
            line is None
            or line.get("error") is not None
            or not isinstance(line.get("prediction"), str)
            or line.get("reference") != sample.reference
        ):
            return None
        return _Outcome(
            sample_id=sample.sample_id,
            reference=sample.reference,
            duration_s=float(line["duration_s"]),
            prediction=line["prediction"],
            latency_s=line.get("latency_s"),
            raw_prediction=line.get("raw_prediction"),
            segments=line.get("segments"),
        )

    def append(self, outcome: _Outcome) -> None:
        if self._fh is None:
            self.conditions_path.write_text(
                json.dumps({"run_manifest": self._manifest}), encoding="utf-8"
            )
            # Superseded by the sidecar just written (same content, lasting name).
            self._legacy_progress_path.unlink(missing_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
            if self._torn_tail():
                self._fh.write("\n")  # never glue a new line onto a torn one
        self._fh.write(json.dumps(_prediction_line(outcome), ensure_ascii=False) + "\n")
        self._fh.flush()

    def _torn_tail(self) -> bool:
        try:
            with self.path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                if fh.tell() == 0:
                    return False
                fh.seek(-1, os.SEEK_END)
                return fh.read(1) != b"\n"
        except OSError:
            return False

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


#: ``_PassMeter.record``: the adapter has not learnt its serving block (yet).
_NO_SERVING: Any = object()


def _serving_of(adapter: ASRAdapter) -> Any:
    """The adapter's pinned serving block, or :data:`_NO_SERVING` when there is
    none to record: a hosted API (no serving concept; its summary row says null)
    or a self-hosted adapter that has not seen a response yet.
    """
    if not getattr(adapter, "serving_known", False):
        return _NO_SERVING
    return getattr(adapter, "serving", None)


@dataclass
class _PassMeter:
    """What one lane transcribed in this pass, and over how much wall time.

    The clock starts at the lane's first request, not at the pass: a resumed
    pass may first skip through hours of clips the log already answers.
    """

    n_requests: int = 0
    n_ok: int = 0
    audio_s: float = 0.0
    started_mono: float | None = None
    started_at: datetime | None = None

    def start(self) -> None:
        if self.started_mono is None:
            self.started_mono = time.monotonic()
            self.started_at = datetime.now(timezone.utc)

    def count(self, outcome: _Outcome) -> None:
        self.n_requests += 1
        if outcome.error is None:
            self.n_ok += 1
            # A failed request transcribed nothing; a fast failure is not speed.
            self.audio_s += outcome.duration_s

    def record(
        self,
        *,
        concurrency: int,
        adapter: str,
        interrupted: bool,
        serving_block: Any = _NO_SERVING,
    ) -> dict[str, Any] | None:
        """The pass line, or None if this lane requested nothing in the pass."""
        if self.started_mono is None or self.started_at is None or not self.n_requests:
            return None
        line = {
            "started_at": _iso(self.started_at),
            "finished_at": _iso(datetime.now(timezone.utc)),
            "wall_s": round(time.monotonic() - self.started_mono, 3),
            "n_requests": self.n_requests,
            "n_ok": self.n_ok,
            "audio_s": round(self.audio_s, 3),
            "concurrency": concurrency,
            "adapter": adapter,
            "hardware": throughput.hardware_label(),
            "interrupted": interrupted,
        }
        if serving_block is not _NO_SERVING:
            # Which setup answered (raven_asr.serving); null for an endpoint
            # that does not report one. Left out when no response arrived to
            # say, so a pass of failed requests cannot contradict the next one.
            line["serving"] = serving_block
        return line


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class _Lane:
    """One model's share of a subset pass: its adapter, pool, log and meter."""

    key: str
    adapter: ASRAdapter
    concurrency: int
    log: _PredictionLog
    #: The adapter kind (``ModelSpec.adapter``), recorded with the pass.
    adapter_name: str = ""
    outcomes: dict[str, _Outcome] = field(default_factory=dict)
    meter: _PassMeter = field(default_factory=_PassMeter)
    semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.semaphore = asyncio.Semaphore(self.concurrency)
        # A resumed subset was partly transcribed already: pin the setup its
        # earlier passes recorded, so the first answer from a different one
        # aborts the run instead of being mixed into the score.
        prior = serving.of_passes(throughput.read_passes(self.log.throughput_path))
        expect = getattr(self.adapter, "expect_serving", None)
        if prior is not None and expect is not None:
            expect(prior)

    def record_pass(self, *, interrupted: bool) -> None:
        """Append this lane's pass to its subset's throughput log, if it did work."""
        line = self.meter.record(
            concurrency=self.concurrency, adapter=self.adapter_name,
            interrupted=interrupted,
            serving_block=_serving_of(self.adapter),
        )
        if line is not None:
            throughput.append_pass(self.log.throughput_path, line)


async def _run_pool(samples: Iterable[Sample], lanes: list[_Lane], desc: str) -> list[str]:
    """Feed ``samples`` lazily through every lane; return sample ids in split order.

    The loader is pulled one sample at a time (off the event loop, so decoding
    never stalls responses in flight), and no more samples are held than the
    lanes have requests in flight: memory is bounded by the concurrency, not by
    the size of the subset. A sample whose transcription a lane already has is
    not sent to that lane again.

    A :class:`ServedModelMismatchError` from any request stops the pass: the
    remaining requests are cancelled and the error propagates.
    """
    window = asyncio.Semaphore(sum(lane.concurrency for lane in lanes))
    order: list[str] = []
    seen: set[str] = set()
    pending: set[asyncio.Task[None]] = set()
    fatal: list[BaseException] = []
    pbars = {
        lane.key: tqdm(desc=f"{lane.adapter.provider_id}/{desc}", unit="clip", position=i)
        for i, lane in enumerate(lanes)
    }

    async def _one(lane: _Lane, sample: Sample) -> None:
        lane.meter.start()
        async with lane.semaphore:
            outcome = await _attempt(lane.adapter, sample)
        lane.log.append(outcome)
        lane.meter.count(outcome)
        lane.outcomes[outcome.sample_id] = outcome
        pbars[lane.key].update(1)

    async def _fan_out(sample: Sample, todo: list[_Lane]) -> None:
        try:
            await asyncio.gather(*(_one(lane, sample) for lane in todo))
        except Exception as exc:  # noqa: BLE001 - re-raised by the producer loop
            # _attempt turns every per-clip failure into an outcome, so what
            # arrives here ends the run (a served-model mismatch, a full disk).
            fatal.append(exc)
        finally:
            window.release()

    iterator: Iterator[Sample] = iter(samples)
    try:
        while not fatal:
            await window.acquire()
            sample = await asyncio.to_thread(next, iterator, _END)
            if sample is _END:
                window.release()
                break
            if sample.sample_id in seen:
                raise ValueError(
                    f"duplicate sample_id {sample.sample_id!r} in {desc}: "
                    f"per-utterance resume needs ids unique within a subset"
                )
            seen.add(sample.sample_id)
            order.append(sample.sample_id)
            todo: list[_Lane] = []
            for lane in lanes:
                cached = lane.log.cached(sample)
                if cached is None:
                    todo.append(lane)
                else:
                    lane.outcomes[sample.sample_id] = cached
                    pbars[lane.key].update(1)
            if not todo:
                window.release()
                continue
            task = asyncio.create_task(_fan_out(sample, todo))
            pending.add(task)
            task.add_done_callback(pending.discard)
            del sample
        if pending:
            await asyncio.gather(*pending)
        if fatal:
            raise fatal[0]
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for lane in lanes:
            lane.log.close()
        for pb in pbars.values():
            pb.close()
    return order


async def _run_subset_lanes(
    lanes: list[_Lane],
    subset: str,
    limit: int | None,
    *,
    streaming: bool = False,
    revision: str | None = None,
) -> dict[str, _RunResult]:
    """One pass over ``subset`` for every lane; a result per lane key.

    Every lane's pass is appended to its throughput log however the pass ends —
    a pass cut short (SIGTERM, a served-model mismatch) is flagged
    ``interrupted`` and still counts what it transcribed.
    """
    loader, internal_subset = _iter_loader_for_subset(
        subset, streaming=streaming, revision=revision
    )
    try:
        order = await _run_pool(
            loader.iter_samples(internal_subset, limit=limit), lanes, subset
        )
    except BaseException:
        for lane in lanes:
            lane.record_pass(interrupted=True)
        raise
    for lane in lanes:
        lane.record_pass(interrupted=False)
    # The tally is final only now that the iterator is exhausted.
    dropped = _dropped_rows(loader)
    return {
        lane.key: _RunResult(
            subset=subset,
            outcomes=[lane.outcomes[sid] for sid in order],
            dropped=dropped,
        )
        for lane in lanes
    }


class RunTerminated(Exception):
    """The run received SIGTERM and stopped after recording its partial pass."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"terminated by signal {signum}")
        self.signum = signum


@contextlib.asynccontextmanager
async def _stop_on_sigterm() -> AsyncIterator[None]:
    """Turn SIGTERM into a cancellation of the run, so its pass is recorded.

    By default SIGTERM ends the process where it stands, and the interrupted
    pass's throughput is lost with it. Here it cancels the running task: the
    pool cancels its requests, every lane appends its pass flagged
    ``interrupted``, adapters close, and :class:`RunTerminated` leaves the event
    loop. :func:`_exit_like_signal` then ends the process by the same signal,
    so a supervisor sees exactly the exit it saw before.

    Outside the main thread no handler can be installed and nothing changes.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    received: list[int] = []

    def _on_sigterm() -> None:
        if not received and task is not None:
            received.append(signal.SIGTERM)
            logger.warning("SIGTERM: recording the interrupted pass, then exiting")
            task.cancel()

    try:
        loop.add_signal_handler(signal.SIGTERM, _on_sigterm)
        installed = True
    except (NotImplementedError, RuntimeError, ValueError):
        installed = False
    try:
        yield
    except asyncio.CancelledError:
        if not received:
            raise
        if task is not None:
            task.uncancel()
        raise RunTerminated(received[0]) from None
    finally:
        if installed:
            loop.remove_signal_handler(signal.SIGTERM)


def _exit_like_signal(signum: int) -> NoReturn:
    """End this process by ``signum`` with its default action, as if never caught."""
    sys.stdout.flush()
    sys.stderr.flush()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
    raise SystemExit(128 + signum)  # pragma: no cover - only if the kill is ignored


async def _close_adapter(adapter: ASRAdapter) -> None:
    """Release an adapter's connections, if it holds any."""
    aclose = getattr(adapter, "aclose", None)
    if aclose is not None:
        await aclose()


def _r4(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _compare_against_flozi(
    model_id: str, entries: Iterable[ResultEntry]
) -> list[dict[str, object]]:
    ref_table = FLOZI_REFERENCE_WER.get(model_id, {})
    out: list[dict[str, object]] = []
    for entry in entries:
        ref_wer = ref_table.get(entry.subset_arg)
        drift_pct: float | None = None
        if ref_wer is not None and ref_wer > 0:
            drift_pct = round((entry.wer_pct - ref_wer) / ref_wer * 100, 2)
        out.append(
            {
                "subset": entry.subset_arg,
                "n_samples": entry.n_samples,
                # Coverage: the WER above is over n_samples = n_ok utterances.
                # n_failed > 0 means the number describes a subset of the split
                # the model happened to finish, and promote refuses to publish it.
                "n_attempted": entry.n_samples + entry.n_failed,
                "n_ok": entry.n_samples,
                "n_failed": entry.n_failed,
                # Rows the loader could not yield, so never attempted: the
                # score does not cover them. null = the loader keeps no tally.
                "n_dropped": entry.n_dropped,
                "dropped_by_reason": entry.dropped_by_reason,
                "dataset_revision": entry.dataset_revision,
                "dataset_sha256": entry.dataset_sha256,
                "wer_pct": round(entry.wer_pct, 4),
                "wer_filler_tolerant_pct": round(
                    entry.wer_filler_tolerant_pct, 4
                ),
                "cer_pct": round(entry.cer_pct, 4),
                # Did the transcript get the numbers right — the error WER
                # hides. Rests on n_entities numbers, not on n_samples.
                "entity_hit_rate_pct": (
                    None if entry.entity_hit_rate_pct is None
                    else round(entry.entity_hit_rate_pct, 4)
                ),
                "n_entities": entry.n_entities,
                "n_utterances_with_entities": entry.n_utterances_with_entities,
                # Hypotheses that ran away from the audio (loops), and how much
                # of the WER above they are. Counted in it, named beside it.
                "n_runaway": entry.n_runaway,
                "runaway_pct": _r4(entry.runaway_pct),
                "runaway_edit_share_pct": _r4(entry.runaway_edit_share_pct),
                "flozi_reference_wer": ref_wer,
                "drift_pct_vs_flozi": drift_pct,
            }
        )
    return out


def _run_manifest(
    spec: ModelSpec, limit: int | None, revision: str | None
) -> dict[str, Any]:
    """The manifest of one invocation for one model, built before any request.

    The hashed config is everything that decides what a number means and is not
    already named per result: the scoring contract, the model binding (env-var
    *names* only — ``ModelSpec`` holds no value) and the two run settings that
    change which clips are scored. Each result carries its own dataset pin.

    ``gpu`` is null for every ASR adapter: all of them call an endpoint, and a
    client cannot attest the hardware behind one.
    """
    return build_run_manifest(
        resolved_config={
            "contract": resolved_contract(),
            "model": asdict(spec),
            "limit_per_subset": limit,
            "dataset_revision_override": revision,
        },
        libraries=MANIFEST_LIBRARIES,
        gpu=None,
    )


# ── Which existing rows of a subset may be reused ────────────────────────────
#
# A subset's rows on disk (complete or in progress) are reused only when they
# were measured exactly as this run measures: under the same run manifest
# (commit, lockfile, platform, libraries, config) and, for a self-hosted model,
# by the same serving setup (engine, recipe id + hash, batching, decoding,
# precision). The check is per subset, against what that subset recorded — not
# against the directory's summary.json, which every invocation rewrites and
# which therefore names whichever run wrote last, not the run that measured a
# given subset. Rows that fail the check are moved aside, never deleted, and
# the subset is measured again.


def _conditions_path(out_dir: Path, subset: str) -> Path:
    """The sidecar naming the run manifest a subset's rows were measured under."""
    return out_dir / f".conditions_{_safe(subset)}.json"


def _legacy_progress_path(out_dir: Path, subset: str) -> Path:
    """The sidecar's name before it outlived the pass (in-progress subsets only)."""
    return out_dir / f".progress_{_safe(subset)}.json"


def _subset_files(out_dir: Path, subset: str) -> list[Path]:
    """Every file that records one subset's measurement, existing or not."""
    safe = _safe(subset)
    return [
        out_dir / f"predictions_{safe}.jsonl",
        out_dir / f".done_{safe}.json",
        _conditions_path(out_dir, subset),
        _legacy_progress_path(out_dir, subset),
        throughput.throughput_path(out_dir, safe),
    ]


def _recorded_manifest(out_dir: Path, subset: str) -> Any:
    """The run manifest a subset's rows recorded, or None if they recorded none."""
    for path in (_conditions_path(out_dir, subset), _legacy_progress_path(out_dir, subset)):
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(sidecar, dict):
            return sidecar.get("run_manifest")
    return None


def _manifest_mismatch(recorded: Any, manifest: dict[str, Any]) -> str:
    """Why ``recorded`` does not describe this run, or ``""`` if it does."""
    if not isinstance(recorded, dict):
        return "the rows record no run manifest (written before per-subset manifests)"
    if same_run_conditions(recorded, manifest):
        return ""
    fields = [f for f in MANIFEST_FIELDS if f != "argv" and recorded.get(f) != manifest.get(f)]
    detail = ", ".join(fields)
    if "git" in fields:
        old = (recorded.get("git") or {}).get("sha") if isinstance(recorded.get("git"), dict) else None
        new = (manifest.get("git") or {}).get("sha")
        detail += f" (commit {str(old)[:12]} -> {str(new)[:12]})"
    return f"the run manifest differs: {detail}"


async def _current_serving(adapter: ASRAdapter) -> Any:
    """The serving block the endpoint answers with now, or :data:`_NO_SERVING`
    for an adapter without one (a hosted API).

    A complete subset sends no request, so without asking there is nothing to
    compare its recorded setup with; :meth:`aprobe_serving` sends one throwaway
    clip and pins the block, so the run that follows is held to it as well.
    """
    probe = getattr(adapter, "aprobe_serving", None)
    if probe is None:
        return _NO_SERVING
    return await probe()


def _serving_mismatch(out_dir: Path, subset: str, current: Any) -> str:
    """Why the subset's recorded serving block is not ``current``, or ``""``."""
    if current is _NO_SERVING:
        return ""
    passes = throughput.read_passes(throughput.throughput_path(out_dir, _safe(subset)))
    if not any("serving" in p for p in passes):
        return "the rows record no serving setup, so they cannot be held to this one"
    try:
        recorded = serving.of_passes(passes)
    except serving.ServingChangedError as exc:
        return str(exc)
    if recorded == current:
        return ""
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return (
            f"the serving setup differs: recorded {serving.describe(recorded)}, "
            f"the endpoint now answers with {serving.describe(current)}"
        )
    changed = [
        f"{key} {json.dumps(recorded.get(key), sort_keys=True)} -> "
        f"{json.dumps(current.get(key), sort_keys=True)}"
        for key in serving.SERVING_KEYS if recorded.get(key) != current.get(key)
    ]
    return "the serving setup differs: " + "; ".join(changed)


@dataclass
class _Supersede:
    """Where one invocation moves rows from other conditions: a timestamped
    sibling of the results directory, outside every ``predictions_*`` glob.
    """

    stamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    def dest(self, out_dir: Path) -> Path:
        return out_dir.parent / f"{out_dir.name}.superseded-{self.stamp}"

    def move(self, out_dir: Path, subset: str, reason: str) -> Path:
        dest = self.dest(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for path in _subset_files(out_dir, subset):
            if path.exists():
                os.replace(path, dest / path.name)
        with (dest / "superseded.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"subset": subset, "reason": reason, "at": _iso(datetime.now(timezone.utc))},
                ensure_ascii=False,
            ) + "\n")
        _drop_post_hoc_for(out_dir, subset, dest)
        return dest


def _drop_post_hoc_for(out_dir: Path, subset: str, dest: Path) -> None:
    """Forget the post-hoc steps the summary records for a superseded subset.

    ``post_hoc`` describes what was done to recorded predictions; the moved-aside
    ones took that history with them (a copy of the summary goes along), and
    the re-measured rows have none yet.
    """
    path = out_dir / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    steps = summary.get("post_hoc") if isinstance(summary, dict) else None
    if not isinstance(steps, list):
        return
    if not (dest / "summary.json").exists():
        (dest / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    kept = []
    for step in steps:
        if isinstance(step, dict) and isinstance(step.get("subsets"), list):
            step = {**step, "subsets": [s for s in step["subsets"] if s != subset]}
            if not step["subsets"]:
                continue
        kept.append(step)
    if kept:
        summary["post_hoc"] = kept
    else:
        summary.pop("post_hoc", None)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


async def _gate_existing_rows(
    *,
    out_dir: Path,
    subset: str,
    manifest: dict[str, Any],
    adapter: ASRAdapter,
    supersede: _Supersede,
    label: str,
) -> ResultEntry | None:
    """Hold a subset's rows on disk to this run; return its entry if complete.

    Rows measured under another run manifest or serving setup — complete or
    half done alike — are moved aside with a log line naming why, and the
    subset is measured again. Rows that pass are resumed: a complete subset
    (``.done_`` marker) is returned and not requested at all, a partial one is
    continued by :class:`_PredictionLog`.
    """
    if not any(p.exists() for p in _subset_files(out_dir, subset)):
        return None
    reason = _manifest_mismatch(_recorded_manifest(out_dir, subset), manifest)
    if not reason:
        reason = _serving_mismatch(out_dir, subset, await _current_serving(adapter))
    if reason:
        dest = supersede.move(out_dir, subset, reason)
        logger.warning(
            "[%s] subset %s: not reusing its recorded rows — %s; moved them to "
            "%s and measuring the subset again", label, subset, reason, dest,
        )
        return None
    marker = out_dir / f".done_{_safe(subset)}.json"
    if not marker.exists():
        return None
    try:
        entry = ResultEntry(**json.loads(marker.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("[%s] ignoring corrupt marker %s: %s; re-running subset",
                       label, marker, exc)
        return None
    logger.info("[%s] subset %s already done under this run's manifest and "
                "serving setup (resumed from %s)", label, subset, marker)
    return entry


def _write_outputs(
    out_dir: Path,
    spec: ModelSpec,
    entries: list[ResultEntry],
    limit: int | None,
    manifest: dict[str, Any],
    post_hoc: list[dict[str, Any]] | None = None,
) -> Path:
    yaml_path = out_dir / "model-index.yaml"
    write_yaml(yaml_path, build_model_index(model_id=spec.model_id, results=entries))
    results = _compare_against_flozi(spec.model_id, entries)
    for row in results:
        # Measured on this machine, not re-derivable from the predictions; null
        # for a subset whose passes predate throughput recording.
        passes = throughput.read_passes(
            throughput.throughput_path(out_dir, _safe(str(row["subset"])))
        )
        row["throughput"] = throughput.summarize(passes)
        # The serving setup every pass of this subset ran on (raven_asr.serving);
        # null without passes, for servers that report none, and for hosted APIs.
        row["serving"] = serving.of_passes(passes)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "model_id": spec.model_id,
        "label": spec.label,
        "adapter": spec.adapter,
        "model_revision": spec.revision,
        "limit_per_subset": limit,
        # How the server's output became the scored prediction
        # (raven_asr.output_formats); null for a plain-text model.
        "output_format": spec.output_format,
        "results": results,
        "run_manifest": manifest,
    }
    if post_hoc is None:
        post_hoc = _recorded_post_hoc(out_dir)
    if post_hoc:
        # Steps applied to the recorded predictions after inference
        # (raven_asr.reparse). The manifest above still names the run that
        # produced the server output; these name what was done to it since.
        summary["post_hoc"] = post_hoc
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return yaml_path


def _recorded_post_hoc(out_dir: Path) -> list[dict[str, Any]] | None:
    """The ``post_hoc`` steps an earlier summary in ``out_dir`` recorded, if any."""
    try:
        previous = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    steps = previous.get("post_hoc") if isinstance(previous, dict) else None
    return steps if isinstance(steps, list) else None


def _entry_from_row(row: dict[str, Any]) -> ResultEntry:
    """The ResultEntry a summary row was written from (``_compare_against_flozi``)."""
    subset = str(row["subset"])
    return ResultEntry(
        dataset_id=_dataset_id_for_subset(subset),
        dataset_name=subset,
        subset_arg=subset,
        wer_pct=float(row["wer_pct"]),
        cer_pct=float(row["cer_pct"]),
        n_samples=int(row["n_samples"]),
        wer_filler_tolerant_pct=float(row.get("wer_filler_tolerant_pct") or 0.0),
        n_failed=int(row.get("n_failed") or 0),
        n_dropped=row.get("n_dropped"),
        dropped_by_reason=row.get("dropped_by_reason"),
        dataset_revision=row.get("dataset_revision"),
        dataset_sha256=row.get("dataset_sha256"),
        entity_hit_rate_pct=row.get("entity_hit_rate_pct"),
        n_entities=int(row.get("n_entities") or 0),
        n_utterances_with_entities=int(row.get("n_utterances_with_entities") or 0),
        n_runaway=row.get("n_runaway"),
        runaway_pct=row.get("runaway_pct"),
        runaway_edit_share_pct=row.get("runaway_edit_share_pct"),
    )


def _scored(entry: ResultEntry, rows: list[dict[str, Any]]) -> ResultEntry:
    """``entry`` with every score re-derived from its predictions lines."""
    ok = [r for r in rows if r.get("error") is None and isinstance(r.get("prediction"), str)]
    refs = [r["reference"] for r in ok]
    preds = [r["prediction"] for r in ok]
    metrics = evaluate(refs, preds)
    entities = entity_hit_rate(refs, preds)
    runaway = runaway_stats(refs, preds, RUNAWAY_LENGTH_RATIO)
    return replace(
        entry,
        wer_pct=metrics.wer_pct,
        cer_pct=metrics.cer_pct,
        n_samples=metrics.n_samples,
        wer_filler_tolerant_pct=metrics.wer_filler_tolerant_pct,
        n_failed=len(rows) - len(ok),
        entity_hit_rate_pct=entities.hit_rate_pct,
        n_entities=entities.n_entities,
        n_utterances_with_entities=entities.n_utterances_with_entities,
        n_runaway=runaway.n_runaway,
        runaway_pct=runaway.runaway_pct,
        runaway_edit_share_pct=runaway.edit_share_pct,
    )


def rescore_recorded(
    out_dir: Path,
    spec: ModelSpec,
    subsets: Iterable[str],
    post_hoc_steps: list[dict[str, Any]],
) -> None:
    """Re-score ``subsets`` of a finished run from their (rewritten) predictions.

    For :mod:`raven_asr.reparse`: the predictions changed after inference, so
    every score derived from them is derived again, by the same functions a run
    uses — the resume markers, ``summary.json`` and ``model-index.yaml`` alike.
    What the run measured and cannot be re-derived (throughput, serving, the
    run manifest, the dataset pin) is kept as recorded. ``post_hoc_steps`` are
    appended to the summary's ``post_hoc`` list.
    """
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    targets = set(subsets)
    entries: list[ResultEntry] = []
    for row in summary.get("results", []):
        subset = str(row["subset"])
        marker = out_dir / f".done_{_safe(subset)}.json"
        entry = (
            ResultEntry(**json.loads(marker.read_text(encoding="utf-8")))
            if marker.exists() else _entry_from_row(row)
        )
        if subset in targets:
            path = out_dir / f"predictions_{_safe(subset)}.jsonl"
            lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
                     if x.strip()]
            entry = _scored(entry, lines)
            if marker.exists():
                marker.write_text(json.dumps(asdict(entry)), encoding="utf-8")
        entries.append(entry)
    # Subsets finished by an earlier invocation into the same dir carry only a
    # marker, not a summary row; their marker is re-scored all the same.
    for subset in targets - {e.subset_arg for e in entries}:
        marker = out_dir / f".done_{_safe(subset)}.json"
        path = out_dir / f"predictions_{_safe(subset)}.jsonl"
        if marker.exists() and path.exists():
            lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
                     if x.strip()]
            entry = _scored(ResultEntry(**json.loads(marker.read_text(encoding="utf-8"))), lines)
            marker.write_text(json.dumps(asdict(entry)), encoding="utf-8")
    steps = list(summary.get("post_hoc") or []) + list(post_hoc_steps)
    _write_outputs(
        out_dir, spec, entries, summary.get("limit_per_subset"),
        summary.get("run_manifest"), post_hoc=steps,
    )


def _write_predictions(path: Path, outcomes: list[_Outcome]) -> None:
    """One line per ATTEMPTED utterance, failures included, in split order.

    Raw refs/preds make a methodology change re-scorable without re-running the
    bench (ADR-KV-024 §3 "persistence gap"). A failed line carries
    ``"prediction": null`` plus ``"error"``, so the artifact itself shows which
    clips have no transcription — a reader of the file cannot mistake a failure
    for a model that answered with silence.

    Written to a temp file and renamed over the per-utterance log, so a crash
    leaves either the full log or the compacted file, never half of one.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(json.dumps(_prediction_line(o), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _record_subset(
    *,
    out_dir: Path,
    spec: ModelSpec,
    result: _RunResult,
    entries: list[ResultEntry],
    limit: int | None,
    manifest: dict[str, Any],
    revision: str | None = None,
) -> None:
    """Persist one finished subset: predictions, score, resume marker, outputs.

    The single place a subset is recorded, for the one-model and the N-model
    path alike, so the two cannot disagree about what an artifact contains.
    """
    subset = result.subset
    marker = out_dir / f".done_{_safe(subset)}.json"
    # The marker is rewritten below only if this pass is complete. An older one
    # must not outlive a pass that was not: it would be resumed as this run's.
    marker.unlink(missing_ok=True)
    if not result.outcomes:
        logger.warning(
            "subset %s produced no samples; skipping (rows dropped by the "
            "loader: %s)", subset, result.dropped or "none reported",
        )
        return
    _write_predictions(out_dir / f"predictions_{_safe(subset)}.jsonl", result.outcomes)
    ok = result.succeeded
    n_failed = result.n_failed
    if not ok:
        logger.error(
            "[%s] subset %s: all %d requests failed; nothing to score",
            spec.label, subset, n_failed,
        )
        return
    refs = [o.reference for o in ok]
    preds = [cast(str, o.prediction) for o in ok]
    metrics = evaluate(refs, preds)
    entities = entity_hit_rate(refs, preds)
    runaway = runaway_stats(refs, preds, RUNAWAY_LENGTH_RATIO)
    dataset_revision, dataset_sha256 = _dataset_pin(subset, revision)
    entry = ResultEntry(
        dataset_id=_dataset_id_for_subset(subset),
        dataset_name=subset,
        subset_arg=subset,
        wer_pct=metrics.wer_pct,
        cer_pct=metrics.cer_pct,
        n_samples=metrics.n_samples,
        wer_filler_tolerant_pct=metrics.wer_filler_tolerant_pct,
        n_failed=n_failed,
        n_dropped=None if result.dropped is None else sum(result.dropped.values()),
        dropped_by_reason=result.dropped,
        dataset_revision=dataset_revision,
        dataset_sha256=dataset_sha256,
        entity_hit_rate_pct=entities.hit_rate_pct,
        n_entities=entities.n_entities,
        n_utterances_with_entities=entities.n_utterances_with_entities,
        n_runaway=runaway.n_runaway,
        runaway_pct=runaway.runaway_pct,
        runaway_edit_share_pct=runaway.edit_share_pct,
    )
    entries.append(entry)
    if entry.n_dropped:
        logger.warning(
            "[%s] subset %s: the loader dropped %d row(s) before any request "
            "(%s) — the score covers %d utterances, not the whole corpus",
            spec.label, subset, entry.n_dropped, entry.dropped_by_reason,
            metrics.n_samples,
        )
    if n_failed:
        # No resume marker: the next invocation runs the subset again instead
        # of resuming into a number that silently leaves clips out — and, from
        # the per-utterance log, requests only the clips that failed.
        logger.error(
            "[%s] subset %s: %d of %d requests failed — WER %.4f%% covers only "
            "the %d that succeeded and is NOT publishable; re-run the subset",
            spec.label, subset, n_failed, len(result.outcomes),
            metrics.wer_pct, metrics.n_samples,
        )
    else:
        marker.write_text(json.dumps(asdict(entry)), encoding="utf-8")
        # Finished. The conditions sidecar stays: it is what lets the next
        # invocation hold this complete subset to its own manifest.
        _legacy_progress_path(out_dir, subset).unlink(missing_ok=True)
        logger.info(
            "[%s] subset %s done: %d samples, WER %.4f%%",
            spec.label, subset, metrics.n_samples, metrics.wer_pct,
        )
    # Roll outputs after every subset so a crash leaves usable artefacts.
    _write_outputs(out_dir, spec, entries, limit, manifest)


def _dataset_id_for_subset(subset: str) -> str:
    """HF slug recorded in model-index.yaml for the dataset owning ``subset``."""
    dataset_id, _ = resolve_wer_dataset(subset)
    return WER_DATASETS[dataset_id].source


async def run_async(
    *,
    model_key: str,
    subsets: list[str],
    limit: int | None,
    out_dir: Path,
    streaming: bool = False,
    concurrency: int | None = None,
    revision: str | None = None,
) -> Path:
    """Execute one model across requested subsets; resumable per-subset."""
    spec = KNOWN_MODELS.get(model_key)
    if spec is None:
        raise ValueError(
            f"unknown model {model_key!r}; add it to "
            f"raven_asr.config.KNOWN_MODELS"
        )
    effective_concurrency = (
        concurrency if concurrency is not None
        else DEFAULT_CONCURRENCY.get(spec.adapter, 4)
    )
    if effective_concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {effective_concurrency}")
    adapter = _make_adapter(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ResultEntry] = []
    manifest = _run_manifest(spec, limit, revision)
    supersede = _Supersede()

    async with _stop_on_sigterm():
        try:
            for subset in subsets:
                done = await _gate_existing_rows(
                    out_dir=out_dir, subset=subset, manifest=manifest,
                    adapter=adapter, supersede=supersede, label=spec.label,
                )
                if done is not None:
                    entries.append(done)
                    continue

                lane = _Lane(
                    key=model_key, adapter=adapter, concurrency=effective_concurrency,
                    log=_PredictionLog(out_dir, subset, manifest),
                    adapter_name=spec.adapter,
                )
                results = await _run_subset_lanes(
                    [lane], subset, limit, streaming=streaming, revision=revision,
                )
                _record_subset(
                    out_dir=out_dir, spec=spec, result=results[model_key],
                    entries=entries, limit=limit, manifest=manifest, revision=revision,
                )
        finally:
            await _close_adapter(adapter)

    return _write_outputs(out_dir, spec, entries, limit, manifest)


def run(
    *,
    model_key: str,
    subsets: list[str],
    limit: int | None,
    out_dir: Path,
    streaming: bool = False,
    concurrency: int | None = None,
    revision: str | None = None,
) -> Path:
    """Sync wrapper around :func:`run_async` — the path ``reproduce`` drives.

    On SIGTERM the interrupted pass is recorded, then the process ends by
    SIGTERM as it would have without the handler.
    """
    try:
        return asyncio.run(
            run_async(
                model_key=model_key,
                subsets=subsets,
                limit=limit,
                out_dir=out_dir,
                streaming=streaming,
                concurrency=concurrency,
                revision=revision,
            )
        )
    except RunTerminated as exc:
        _exit_like_signal(exc.signum)


async def run_multi_async(
    *,
    model_keys: list[str],
    subsets: list[str],
    limit: int | None,
    out_root: Path,
    streaming: bool = False,
    concurrency_overrides: dict[str, int] | None = None,
    revision: str | None = None,
) -> dict[str, Path]:
    """Benchmark N models against the same subsets with a single dataset load.

    Architecture (per [[feedback-shared-dataset-loader-for-parallel-benches]]):
    one process owns the loader; for each subset, the sample stream is iterated
    once and every sample is fanned out to all N adapters, each behind its own
    bounded pool. This is the structurally-correct parallel-bench pattern —
    spawning N runner subprocesses (each loading the dataset independently) was
    rejected as RAM-quadratic. Resume works per utterance and per model, exactly
    as on the one-model path.

    ``concurrency_overrides`` maps a model key to its in-flight request bound;
    models without one use their adapter's default.

    Returns a mapping of model_key → path to that model's ``model-index.yaml``.
    """
    specs: dict[str, ModelSpec] = {}
    adapters: dict[str, ASRAdapter] = {}
    concurrency: dict[str, int] = {}
    out_dirs: dict[str, Path] = {}
    entries: dict[str, list[ResultEntry]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    concurrency_overrides = concurrency_overrides or {}
    unknown = set(concurrency_overrides) - set(model_keys)
    if unknown:
        raise ValueError(
            f"concurrency override for model(s) not in this run: {sorted(unknown)}"
        )

    for key in model_keys:
        spec = KNOWN_MODELS.get(key)
        if spec is None:
            raise ValueError(
                f"unknown model {key!r}; add it to raven_asr.config.KNOWN_MODELS"
            )
        specs[key] = spec
        concurrency[key] = concurrency_overrides.get(
            key, DEFAULT_CONCURRENCY.get(spec.adapter, 4)
        )
        if concurrency[key] < 1:
            raise ValueError(f"[{key}] concurrency must be >= 1, got {concurrency[key]}")
        out_dirs[key] = out_root / spec.label
        out_dirs[key].mkdir(parents=True, exist_ok=True)
        manifests[key] = _run_manifest(spec, limit, revision)
        entries[key] = []
    supersede = _Supersede()

    async with _stop_on_sigterm():
        try:
            for key in model_keys:
                adapters[key] = _make_adapter(specs[key])
            # Resume what each model's subsets recorded under this run's
            # conditions; move the rest aside before any lane opens a log.
            done_pairs: set[tuple[str, str]] = set()
            for key in model_keys:
                for subset in subsets:
                    done = await _gate_existing_rows(
                        out_dir=out_dirs[key], subset=subset, manifest=manifests[key],
                        adapter=adapters[key], supersede=supersede, label=key,
                    )
                    if done is not None:
                        entries[key].append(done)
                        done_pairs.add((key, subset))
            for subset in subsets:
                pending = [k for k in model_keys if (k, subset) not in done_pairs]
                if not pending:
                    logger.info(
                        "subset %s already done for all requested models; skipping",
                        subset,
                    )
                    continue
                lanes = [
                    _Lane(
                        key=k, adapter=adapters[k], concurrency=concurrency[k],
                        log=_PredictionLog(out_dirs[k], subset, manifests[k]),
                        adapter_name=specs[k].adapter,
                    )
                    for k in pending
                ]
                results = await _run_subset_lanes(
                    lanes, subset, limit, streaming=streaming, revision=revision,
                )
                for k in pending:
                    _record_subset(
                        out_dir=out_dirs[k], spec=specs[k], result=results[k],
                        entries=entries[k], limit=limit, manifest=manifests[k],
                        revision=revision,
                    )
        finally:
            for adapter in adapters.values():
                await _close_adapter(adapter)

    yaml_paths: dict[str, Path] = {}
    for k in model_keys:
        yaml_paths[k] = _write_outputs(
            out_dirs[k], specs[k], entries[k], limit, manifests[k]
        )
    return yaml_paths


def _parse_model_concurrency(values: list[str], model_keys: list[str]) -> dict[str, int]:
    """``KEY=N`` pairs from ``--model-concurrency`` into a per-model bound."""
    out: dict[str, int] = {}
    for item in values:
        key, sep, n = item.rpartition("=")
        if not sep or not key:
            raise ValueError(f"--model-concurrency expects KEY=N, got {item!r}")
        if key not in model_keys:
            raise ValueError(f"--model-concurrency names {key!r}, which is not in --models")
        out[key] = int(n)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="german-asr-eval")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--model",
        help="Single model key (legacy path; for multi-model use --models)",
    )
    grp.add_argument(
        "--models",
        help="Comma-separated model keys — runs them in one process with a "
        "shared dataset loader (recommended for N>1)",
    )
    parser.add_argument("--subsets", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--streaming", action="store_true",
        help="stream from the HF Hub instead of the pinned local snapshot",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Requests in flight per model (overrides the adapter default); "
        "with --models it applies to every model",
    )
    parser.add_argument(
        "--model-concurrency",
        action="append",
        default=[],
        metavar="KEY=N",
        help="Per-model bound on the --models path; repeatable, wins over "
        "--concurrency for that model",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    if not subsets:
        parser.error("--subsets must list at least one subset")
    if args.model_concurrency and not args.models:
        parser.error("--model-concurrency needs --models")

    if args.models:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        for k in model_keys:
            if k not in KNOWN_MODELS:
                parser.error(
                    f"unknown model {k!r}; see raven_asr.config.KNOWN_MODELS"
                )
        try:
            overrides = _parse_model_concurrency(args.model_concurrency, model_keys)
        except ValueError as exc:
            parser.error(str(exc))
        if args.concurrency is not None:
            overrides = {k: overrides.get(k, args.concurrency) for k in model_keys}
        try:
            paths = asyncio.run(
                run_multi_async(
                    model_keys=model_keys,
                    subsets=subsets,
                    limit=args.limit,
                    out_root=args.out,
                    streaming=args.streaming,
                    concurrency_overrides=overrides,
                )
            )
        except ServedModelMismatchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        except RunTerminated as exc:
            _exit_like_signal(exc.signum)
        for k, p in paths.items():
            print(f"[{k}] wrote {p}")
        return 0

    spec = KNOWN_MODELS.get(args.model)
    if spec is None:
        parser.error(
            f"unknown --model {args.model!r}; see raven_asr.config.KNOWN_MODELS"
        )
    out_dir = args.out if args.out.name == spec.label else args.out / spec.label
    try:
        yaml_path = asyncio.run(
            run_async(
                model_key=args.model,
                subsets=subsets,
                limit=args.limit,
                out_dir=out_dir,
                streaming=args.streaming,
                concurrency=args.concurrency,
            )
        )
    except ServedModelMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except RunTerminated as exc:
        _exit_like_signal(exc.signum)
    print(f"wrote {yaml_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
