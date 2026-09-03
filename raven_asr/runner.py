"""CLI runner — async, per-subset resumable, flozi-comparable YAML output."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from raven_eval_core.flozi_wer import evaluate

from .adapters.base import ASRAdapter
from .config import (
    FLOZI_DATASET_REVISION,
    FLOZI_REFERENCE_WER,
    KNOWN_MODELS,
    MODAL_APP_NAMES,
    ModelSpec,
)
from .datasets.base import DatasetLoader, Sample
from .datasets.flozi_mixed_evals import FloziMixedEvalsLoader
from .model_index import ResultEntry, build_model_index, write_yaml

logger = logging.getLogger("raven_asr.runner")


# Per-adapter default concurrency. Override via ``--concurrency N`` on the CLI.
DEFAULT_CONCURRENCY: dict[str, int] = {
    "modal_app": 20,
    "vllm_openai": 10,
    "voxtral_mistral": 8,
    "deepgram": 20,
    "openai_whisper": 3,
}


@dataclass(frozen=True)
class _RunResult:
    subset: str
    references: list[str]
    predictions: list[str]
    latencies: list[float]


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


def _iter_loader_for_subset(
    subset: str,
    *,
    streaming: bool = False,
    revision: str | None = None,
) -> tuple[DatasetLoader, str]:
    # Public repo: only the flozi public datasets ship here. Raven's private
    # meeting corpus (Tier-3) is deliberately not portable and stays internal.
    rev = revision if revision is not None else FLOZI_DATASET_REVISION
    return FloziMixedEvalsLoader(streaming=streaming, revision=rev), subset


def _safe(name: str) -> str:
    """Filesystem-safe subset-marker filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


async def run_subset_async(
    adapter: ASRAdapter,
    subset: str,
    limit: int | None,
    concurrency: int,
    *,
    streaming: bool = False,
    revision: str | None = None,
) -> _RunResult:
    """Transcribe every sample in ``subset`` concurrently."""
    loader, internal_subset = _iter_loader_for_subset(
        subset, streaming=streaming, revision=revision
    )
    samples: list[Sample] = list(loader.iter_samples(internal_subset, limit=limit))
    if not samples:
        return _RunResult(subset=subset, references=[], predictions=[], latencies=[])

    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(samples), desc=f"{adapter.provider_id}/{subset}", unit="clip")

    async def _one(sample: Sample) -> tuple[str, str, float] | None:
        async with semaphore:
            try:
                r = await adapter.atranscribe(sample.audio, sample.sample_rate)
                return (sample.reference, r.text, r.latency_s)
            except Exception:
                logger.exception("transcribe failed on sample %s", sample.sample_id)
                return None
            finally:
                pbar.update(1)

    tasks = [asyncio.create_task(_one(s)) for s in samples]
    results = await asyncio.gather(*tasks)
    pbar.close()

    pairs = [r for r in results if r is not None]
    if not pairs:
        return _RunResult(subset=subset, references=[], predictions=[], latencies=[])
    refs_t, preds_t, lats_t = zip(*pairs, strict=False)
    return _RunResult(
        subset=subset,
        references=list(refs_t),
        predictions=list(preds_t),
        latencies=list(lats_t),
    )


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
        "limit_per_subset": limit,
        "results": _compare_against_flozi(spec.model_id, entries),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return yaml_path


def _dataset_id_for_subset(subset: str) -> str:
    return "flozi00/asr-german-mixed-evals"


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
        if not run_result.references:
            logger.warning("subset %s produced no samples; skipping", subset)
            continue
        metrics = evaluate(run_result.references, run_result.predictions)
        entry = ResultEntry(
            dataset_id=_dataset_id_for_subset(subset),
            dataset_name=subset,
            subset_arg=subset,
            wer_pct=metrics.wer_pct,
            cer_pct=metrics.cer_pct,
            n_samples=metrics.n_samples,
            wer_filler_tolerant_pct=metrics.wer_filler_tolerant_pct,
        )
        entries.append(entry)
        # Persist raw refs/preds so future methodology changes are re-scorable
        # without re-running the bench (ADR-KV-024 §3 "persistence gap").
        predictions_path = out_dir / f"predictions_{_safe(subset)}.jsonl"
        with predictions_path.open("w", encoding="utf-8") as fh:
            for ref, pred, lat in zip(
                run_result.references,
                run_result.predictions,
                run_result.latencies,
                strict=True,
            ):
                fh.write(
                    json.dumps(
                        {"reference": ref, "prediction": pred, "latency_s": lat},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        marker.write_text(json.dumps(asdict(entry)), encoding="utf-8")
        # Roll outputs after every subset so a crash leaves usable artefacts.
        _write_outputs(out_dir, spec, entries, limit)

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

        per_refs: dict[str, list[str]] = {k: [] for k in pending}
        per_preds: dict[str, list[str]] = {k: [] for k in pending}
        per_lats: dict[str, list[float]] = {k: [] for k in pending}
        per_fails: dict[str, int] = {k: 0 for k in pending}
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
            _refs: dict[str, list[str]] = per_refs,
            _preds: dict[str, list[str]] = per_preds,
            _lats: dict[str, list[float]] = per_lats,
            _fails: dict[str, int] = per_fails,
            _pbars: dict[str, tqdm] = pbars,
        ) -> None:
            async with semaphores[model_key]:
                try:
                    r = await adapters[model_key].atranscribe(
                        sample.audio, sample.sample_rate
                    )
                    _refs[model_key].append(sample.reference)
                    _preds[model_key].append(r.text)
                    _lats[model_key].append(r.latency_s)
                except Exception:
                    _fails[model_key] = _fails[model_key] + 1
                    logger.exception(
                        "[%s] transcribe failed on %s", model_key, sample.sample_id
                    )
                finally:
                    _pbars[model_key].update(1)

        tasks = [
            asyncio.create_task(_one(k, s))
            for k in pending
            for s in samples
        ]
        await asyncio.gather(*tasks)
        for pb in pbars.values():
            pb.close()
        # Free the materialised sample list before moving on to the next subset.
        del samples

        for k in pending:
            if not per_refs[k]:
                logger.warning(
                    "[%s] subset %s produced no successful samples (failures: %d)",
                    k,
                    subset,
                    per_fails[k],
                )
                continue
            m = evaluate(per_refs[k], per_preds[k])
            entry = ResultEntry(
                dataset_id=_dataset_id_for_subset(subset),
                dataset_name=subset,
                subset_arg=subset,
                wer_pct=m.wer_pct,
                cer_pct=m.cer_pct,
                n_samples=m.n_samples,
            )
            entries[k].append(entry)
            (out_dirs[k] / f".done_{_safe(subset)}.json").write_text(
                json.dumps(asdict(entry)), encoding="utf-8"
            )
            _write_outputs(out_dirs[k], specs[k], entries[k], limit)
            logger.info(
                "[%s] subset %s done: %d samples, WER %.4f%% (%d failures)",
                k,
                subset,
                m.n_samples,
                m.wer_pct,
                per_fails[k],
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
