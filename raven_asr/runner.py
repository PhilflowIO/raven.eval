"""CLI runner — async, per-subset resumable, flozi-comparable YAML output."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from raven_eval_core.entities import entity_hit_rate
from tqdm import tqdm

from raven_eval_core.contract import resolved_contract
from raven_eval_core.flozi_wer import evaluate
from raven_eval_core.run_manifest import (
    SCHEMA_VERSION,
    build_run_manifest,
    same_run_conditions,
)

from .adapters.base import ASRAdapter, ServedModelMismatchError
from .config import (
    FLOZI_REFERENCE_WER,
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
    if o.error is not None:
        line["error"] = o.error
    return line


class _PredictionLog:
    """``predictions_<subset>.jsonl`` while a subset is in progress.

    Every outcome is appended and flushed the moment it exists, so a run killed
    20 hours into a corpus keeps 20 hours of transcripts. The log is bound to
    the run manifest it was written under through a ``.progress_<subset>.json``
    sidecar — the same guard that decides whether a finished subset may be
    resumed — and is only resumed from when that manifest matches: a transcript
    from another commit, lockfile or config is not this run's.

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
        self.progress_path = out_dir / f".progress_{_safe(subset)}.json"
        self._manifest = manifest
        self._fh: Any = None
        self.prior: dict[str, dict[str, Any]] = {}
        if self._same_conditions():
            self.prior = self._read_prior()
            if self.prior:
                logger.info(
                    "resuming %s: %d utterance(s) already recorded",
                    self.path.name, len(self.prior),
                )
        elif self.path.exists() or self.progress_path.exists():
            logger.warning(
                "%s was written under a different commit, lockfile or config "
                "than this run (or by a run without per-utterance resume); "
                "starting the subset over", self.path.name,
            )
            self.path.unlink(missing_ok=True)
            self.progress_path.unlink(missing_ok=True)

    def _same_conditions(self) -> bool:
        try:
            progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(progress, dict) and same_run_conditions(
            progress.get("run_manifest"), self._manifest
        )

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
        )

    def append(self, outcome: _Outcome) -> None:
        if self._fh is None:
            self.progress_path.write_text(
                json.dumps({"run_manifest": self._manifest}), encoding="utf-8"
            )
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


@dataclass
class _Lane:
    """One model's share of a subset pass: its adapter, pool and log."""

    key: str
    adapter: ASRAdapter
    concurrency: int
    log: _PredictionLog
    outcomes: dict[str, _Outcome] = field(default_factory=dict)
    semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.semaphore = asyncio.Semaphore(self.concurrency)


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
        async with lane.semaphore:
            outcome = await _attempt(lane.adapter, sample)
        lane.log.append(outcome)
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
    """One pass over ``subset`` for every lane; a result per lane key."""
    loader, internal_subset = _iter_loader_for_subset(
        subset, streaming=streaming, revision=revision
    )
    order = await _run_pool(
        loader.iter_samples(internal_subset, limit=limit), lanes, subset
    )
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


async def _close_adapter(adapter: ASRAdapter) -> None:
    """Release an adapter's connections, if it holds any."""
    aclose = getattr(adapter, "aclose", None)
    if aclose is not None:
        await aclose()


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


def _resumable(out_dir: Path, manifest: dict[str, Any]) -> bool:
    """Whether ``out_dir``'s finished subsets were measured as this run measures.

    A summary holds one manifest, so it can only describe subsets that share
    one. Markers left by a different commit, lockfile or config — or by a run
    from before the manifest existed — are not resumed into it.
    """
    try:
        previous = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(previous, dict) and same_run_conditions(
        previous.get("run_manifest"), manifest
    )


def _write_outputs(
    out_dir: Path,
    spec: ModelSpec,
    entries: list[ResultEntry],
    limit: int | None,
    manifest: dict[str, Any],
) -> Path:
    yaml_path = out_dir / "model-index.yaml"
    write_yaml(yaml_path, build_model_index(model_id=spec.model_id, results=entries))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "model_id": spec.model_id,
        "label": spec.label,
        "adapter": spec.adapter,
        "model_revision": spec.revision,
        "limit_per_subset": limit,
        "results": _compare_against_flozi(spec.model_id, entries),
        "run_manifest": manifest,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return yaml_path


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
        # Finished: nothing left to resume from the per-utterance log.
        (out_dir / f".progress_{_safe(subset)}.json").unlink(missing_ok=True)
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
    resumable = _resumable(out_dir, manifest)

    try:
        for subset in subsets:
            marker = out_dir / f".done_{_safe(subset)}.json"
            if marker.exists() and not resumable:
                logger.warning(
                    "subset %s was finished under a different commit, lockfile "
                    "or config than this run; re-running it instead of "
                    "resuming", subset
                )
            elif marker.exists():
                try:
                    cached = json.loads(marker.read_text(encoding="utf-8"))
                    entries.append(ResultEntry(**cached))
                    logger.info(
                        "subset %s already done (resumed from %s)", subset, marker
                    )
                    continue
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning(
                        "ignoring corrupt marker %s: %s; re-running subset",
                        marker, exc,
                    )

            lane = _Lane(
                key=model_key, adapter=adapter, concurrency=effective_concurrency,
                log=_PredictionLog(out_dir, subset, manifest),
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
    """Sync wrapper around :func:`run_async` — kept for test/ad-hoc callers."""
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
        resumable = _resumable(out_dirs[key], manifests[key])
        # Resume from any already-present markers.
        loaded: list[ResultEntry] = []
        for subset in subsets:
            mp = out_dirs[key] / f".done_{_safe(subset)}.json"
            if mp.exists() and not resumable:
                logger.warning(
                    "[%s] subset %s was finished under a different commit, "
                    "lockfile or config than this run; re-running it", key, subset
                )
            elif mp.exists():
                try:
                    loaded.append(
                        ResultEntry(**json.loads(mp.read_text(encoding="utf-8")))
                    )
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning(
                        "[%s] ignoring corrupt marker %s: %s", key, mp, exc
                    )
        entries[key] = loaded

    done_pairs: set[tuple[str, str]] = {
        (k, e.subset_arg) for k, es in entries.items() for e in es
    }

    try:
        for key in model_keys:
            adapters[key] = _make_adapter(specs[key])
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
    print(f"wrote {yaml_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
