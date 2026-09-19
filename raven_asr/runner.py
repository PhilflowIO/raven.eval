"""CLI runner — async, per-subset resumable, flozi-comparable YAML output."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from tqdm import tqdm

from raven_eval_core.flozi_wer import evaluate

from .adapters.base import ASRAdapter
from .config import (
    FLOZI_REFERENCE_WER,
    KNOWN_MODELS,
    MODAL_APP_NAMES,
    WER_DATASETS,
    ModelSpec,
    resolve_wer_dataset,
)
from .datasets import load_loader_class
from .datasets.base import DatasetLoader, Sample
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
}


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

    @property
    def succeeded(self) -> list[_Outcome]:
        return [o for o in self.outcomes if o.error is None]

    @property
    def n_failed(self) -> int:
        return sum(1 for o in self.outcomes if o.error is not None)


def _make_adapter(spec: ModelSpec) -> ASRAdapter:
    if spec.adapter == "vllm_openai":
        from .adapters.vllm_openai import (
            DEFAULT_API_KEY_ENV,
            DEFAULT_BASE_URL_ENV,
            VllmOpenAIAdapter,
        )
        # Per-model endpoint env-var names (flow.raven#5137); a spec that
        # leaves them None keeps the legacy VLLM_PRIMELINE_* defaults.
        return VllmOpenAIAdapter(
            provider_id=spec.label,
            model_id=spec.model_id,
            base_url_env=spec.base_url_env or DEFAULT_BASE_URL_ENV,
            api_key_env=spec.api_key_env or DEFAULT_API_KEY_ENV,
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
    spec = WER_DATASETS[dataset_id]
    rev, _ = _dataset_pin(subset, revision)
    loader_cls = load_loader_class(dataset_id)
    loader = loader_cls(
        streaming=streaming or spec.stream_by_default, revision=rev
    )
    return loader, internal_subset


def _safe(name: str) -> str:
    """Filesystem-safe subset-marker filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


async def _attempt(adapter: ASRAdapter, sample: Sample) -> _Outcome:
    """Transcribe one sample; a failure becomes an outcome, not an absence."""
    duration_s = len(sample.audio) / float(sample.sample_rate)
    try:
        r = await adapter.atranscribe(sample.audio, sample.sample_rate)
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


async def run_subset_async(
    adapter: ASRAdapter,
    subset: str,
    limit: int | None,
    concurrency: int,
    *,
    streaming: bool = False,
    revision: str | None = None,
) -> _RunResult:
    """Transcribe every sample in ``subset`` concurrently, in split order."""
    loader, internal_subset = _iter_loader_for_subset(
        subset, streaming=streaming, revision=revision
    )
    samples: list[Sample] = list(loader.iter_samples(internal_subset, limit=limit))
    if not samples:
        return _RunResult(subset=subset)

    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(samples), desc=f"{adapter.provider_id}/{subset}", unit="clip")

    async def _one(sample: Sample) -> _Outcome:
        async with semaphore:
            try:
                return await _attempt(adapter, sample)
            finally:
                pbar.update(1)

    outcomes = await asyncio.gather(*(_one(s) for s in samples))
    pbar.close()
    return _RunResult(subset=subset, outcomes=list(outcomes))


# Back-compat sync wrapper (some tests + ad-hoc callers still expect it).
def run_subset(
    adapter: ASRAdapter,
    subset: str,
    limit: int | None,
    *,
    streaming: bool = False,
    concurrency: int = 1,
) -> _RunResult:
    return asyncio.run(
        run_subset_async(
            adapter, subset, limit, concurrency, streaming=streaming
        )
    )


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
                "dataset_revision": entry.dataset_revision,
                "dataset_sha256": entry.dataset_sha256,
                "wer_pct": round(entry.wer_pct, 4),
                "wer_filler_tolerant_pct": round(
                    entry.wer_filler_tolerant_pct, 4
                ),
                "cer_pct": round(entry.cer_pct, 4),
                "flozi_reference_wer": ref_wer,
                "drift_pct_vs_flozi": drift_pct,
            }
        )
    return out


def _write_outputs(
    out_dir: Path, spec: ModelSpec, entries: list[ResultEntry], limit: int | None
) -> Path:
    yaml_path = out_dir / "model-index.yaml"
    write_yaml(yaml_path, build_model_index(model_id=spec.model_id, results=entries))
    summary = {
        "model_id": spec.model_id,
        "label": spec.label,
        "adapter": spec.adapter,
        "model_revision": spec.revision,
        "limit_per_subset": limit,
        "results": _compare_against_flozi(spec.model_id, entries),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return yaml_path


def _write_predictions(path: Path, outcomes: list[_Outcome]) -> None:
    """One line per ATTEMPTED utterance, failures included.

    Raw refs/preds make a methodology change re-scorable without re-running the
    bench (ADR-KV-024 §3 "persistence gap"). A failed line carries
    ``"prediction": null`` plus ``"error"``, so the artifact itself shows which
    clips have no transcription — a reader of the file cannot mistake a failure
    for a model that answered with silence.
    """
    with path.open("w", encoding="utf-8") as fh:
        for o in outcomes:
            line: dict[str, object] = {
                "reference": o.reference,
                "prediction": o.prediction,
                "latency_s": o.latency_s,
                "sample_id": o.sample_id,
                "duration_s": round(o.duration_s, 6),
            }
            if o.error is not None:
                line["error"] = o.error
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def _record_subset(
    *,
    out_dir: Path,
    spec: ModelSpec,
    result: _RunResult,
    entries: list[ResultEntry],
    limit: int | None,
    revision: str | None = None,
) -> None:
    """Persist one finished subset: predictions, score, resume marker, outputs.

    The single place a subset is recorded, for the one-model and the N-model
    path alike, so the two cannot disagree about what an artifact contains.
    """
    subset = result.subset
    if not result.outcomes:
        logger.warning("subset %s produced no samples; skipping", subset)
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
    metrics = evaluate([o.reference for o in ok], [cast(str, o.prediction) for o in ok])
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
        dataset_revision=dataset_revision,
        dataset_sha256=dataset_sha256,
    )
    entries.append(entry)
    if n_failed:
        # No resume marker: the next invocation re-runs the subset instead of
        # resuming into a number that silently leaves clips out.
        logger.error(
            "[%s] subset %s: %d of %d requests failed — WER %.4f%% covers only "
            "the %d that succeeded and is NOT publishable; re-run the subset",
            spec.label, subset, n_failed, len(result.outcomes),
            metrics.wer_pct, metrics.n_samples,
        )
    else:
        (out_dir / f".done_{_safe(subset)}.json").write_text(
            json.dumps(asdict(entry)), encoding="utf-8"
        )
        logger.info(
            "[%s] subset %s done: %d samples, WER %.4f%%",
            spec.label, subset, metrics.n_samples, metrics.wer_pct,
        )
    # Roll outputs after every subset so a crash leaves usable artefacts.
    _write_outputs(out_dir, spec, entries, limit)


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
    adapter = _make_adapter(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ResultEntry] = []

    for subset in subsets:
        marker = out_dir / f".done_{_safe(subset)}.json"
        if marker.exists():
            try:
                cached = json.loads(marker.read_text(encoding="utf-8"))
                entries.append(ResultEntry(**cached))
                logger.info(
                    "subset %s already done (resumed from %s)", subset, marker
                )
                continue
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "ignoring corrupt marker %s: %s; re-running subset", marker, exc
                )

        run_result = await run_subset_async(
            adapter, subset, limit, effective_concurrency,
            streaming=streaming, revision=revision,
        )
        _record_subset(
            out_dir=out_dir, spec=spec, result=run_result, entries=entries,
            limit=limit, revision=revision,
        )

    return _write_outputs(out_dir, spec, entries, limit)


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
) -> dict[str, Path]:
    """Benchmark N models against the same subsets with a single dataset load.

    Architecture (per [[feedback-shared-dataset-loader-for-parallel-benches]]):
    one process owns the loader + arrow mmap; for each subset, the sample
    stream is iterated once and every sample is fanned-out to all N adapters
    via per-adapter ``asyncio.Semaphore``s. This is the structurally-correct
    parallel-bench pattern — spawning N runner subprocesses (each loading
    the dataset independently) was rejected as RAM-quadratic.

    Returns a mapping of model_key → path to that model's ``model-index.yaml``.
    """
    specs: dict[str, ModelSpec] = {}
    adapters: dict[str, ASRAdapter] = {}
    semaphores: dict[str, asyncio.Semaphore] = {}
    out_dirs: dict[str, Path] = {}
    entries: dict[str, list[ResultEntry]] = {}
    concurrency_overrides = concurrency_overrides or {}

    for key in model_keys:
        spec = KNOWN_MODELS.get(key)
        if spec is None:
            raise ValueError(
                f"unknown model {key!r}; add it to raven_asr.config.KNOWN_MODELS"
            )
        specs[key] = spec
        adapters[key] = _make_adapter(spec)
        conc = concurrency_overrides.get(
            key, DEFAULT_CONCURRENCY.get(spec.adapter, 4)
        )
        semaphores[key] = asyncio.Semaphore(conc)
        out_dirs[key] = out_root / spec.label
        out_dirs[key].mkdir(parents=True, exist_ok=True)
        # Resume from any already-present markers.
        loaded: list[ResultEntry] = []
        for subset in subsets:
            mp = out_dirs[key] / f".done_{_safe(subset)}.json"
            if mp.exists():
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

    for subset in subsets:
        pending = [k for k in model_keys if (k, subset) not in done_pairs]
        if not pending:
            logger.info(
                "subset %s already done for all requested models; skipping", subset
            )
            continue

        loader, internal_subset = _iter_loader_for_subset(subset, streaming=streaming)
        samples: list[Sample] = list(
            loader.iter_samples(internal_subset, limit=limit)
        )
        if not samples:
            logger.warning("subset %s produced no samples; skipping", subset)
            continue

        pbars = {
            k: tqdm(
                total=len(samples),
                desc=f"{adapters[k].provider_id}/{subset}",
                unit="clip",
                position=i,
            )
            for i, k in enumerate(pending)
        }

        async def _one(
            model_key: str,
            sample: Sample,
            _pbars: dict[str, tqdm] = pbars,
        ) -> _Outcome:
            async with semaphores[model_key]:
                try:
                    return await _attempt(adapters[model_key], sample)
                finally:
                    _pbars[model_key].update(1)

        # gather preserves argument order, so each model's outcomes come back in
        # split order regardless of which request finished first.
        per_model = await asyncio.gather(
            *(asyncio.gather(*(_one(k, s) for s in samples)) for k in pending)
        )
        for pb in pbars.values():
            pb.close()
        # Free the materialised sample list before moving on to the next subset.
        del samples

        for k, outcomes in zip(pending, per_model, strict=True):
            _record_subset(
                out_dir=out_dirs[k], spec=specs[k],
                result=_RunResult(subset=subset, outcomes=list(outcomes)),
                entries=entries[k], limit=limit,
            )

    yaml_paths: dict[str, Path] = {}
    for k in model_keys:
        yaml_paths[k] = _write_outputs(out_dirs[k], specs[k], entries[k], limit)
    return yaml_paths


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
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override per-adapter default concurrency (single-model mode only)",
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

    if args.models:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        for k in model_keys:
            if k not in KNOWN_MODELS:
                parser.error(
                    f"unknown model {k!r}; see raven_asr.config.KNOWN_MODELS"
                )
        paths = asyncio.run(
            run_multi_async(
                model_keys=model_keys,
                subsets=subsets,
                limit=args.limit,
                out_root=args.out,
                streaming=args.streaming,
            )
        )
        for k, p in paths.items():
            print(f"[{k}] wrote {p}")
        return 0

    spec = KNOWN_MODELS.get(args.model)
    if spec is None:
        parser.error(
            f"unknown --model {args.model!r}; see raven_asr.config.KNOWN_MODELS"
        )
    out_dir = args.out if args.out.name == spec.label else args.out / spec.label
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
    print(f"wrote {yaml_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
