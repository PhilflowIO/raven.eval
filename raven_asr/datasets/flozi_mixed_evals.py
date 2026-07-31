"""Loader for ``flozi00/asr-german-mixed-evals`` — flozi-aligned reference dataset.

The dataset has a single ``train`` split containing rows from three sources
identified by the ``from`` column: ``Tuda-De``, ``multilingual librispeech``,
and ``common_voice_19_0``. Reference text lives in the ``references`` column,
audio in ``audio`` (bytes form, decoded with soundfile to dodge the
torchcodec dependency that ``Audio()`` casting requires).

The loader prefers a local HF cache snapshot — populated up-front by
``huggingface_hub.snapshot_download`` — so long bench runs don't depend on
``huggingface.co`` parquet streaming, which has been observed to time out
intermittently on 5h+ jobs. Falls back to ``load_dataset(..., streaming=True)``
when no local cache is available.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf

from .base import Sample

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset, IterableDataset

DATASET_ID = "flozi00/asr-german-mixed-evals"
DEFAULT_SPLIT = "train"
TARGET_SAMPLE_RATE = 16000

SUBSETS = ("Tuda-De", "multilingual librispeech", "common_voice_19_0")

SUBSET_ALIASES = {
    "multilingual_librispeech": "multilingual librispeech",
    "Multilingual LibriSpeech": "multilingual librispeech",
    "MLS": "multilingual librispeech",
}


def _hf_cache_root() -> Path:
    """Resolve the HF Hub cache root, honoring ``HF_HOME`` / ``HF_HUB_CACHE``."""
    env_hub = os.environ.get("HF_HUB_CACHE")
    if env_hub:
        return Path(env_hub)
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(env_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _find_local_parquets(dataset_id: str) -> list[str] | None:
    """Look for a populated local snapshot, return parquet paths if found."""
    repo_dir = _hf_cache_root() / f"datasets--{dataset_id.replace('/', '--')}"
    snap_root = repo_dir / "snapshots"
    if not snap_root.is_dir():
        return None
    # Pick the snapshot with the most parquets (in case multiple revisions sit there).
    best: tuple[int, list[str]] | None = None
    for snap in snap_root.iterdir():
        if not snap.is_dir():
            continue
        parquets = sorted(str(p) for p in (snap / "data").glob("*.parquet"))
        if not parquets:
            continue
        if best is None or len(parquets) > best[0]:
            best = (len(parquets), parquets)
    if best is None:
        return None
    return best[1]


class FloziMixedEvalsLoader:
    """Streams (audio, reference, id, subset) tuples from flozi-mixed-evals."""

    name = DATASET_ID

    def __init__(
        self,
        dataset_id: str = DATASET_ID,
        split: str = DEFAULT_SPLIT,
        streaming: bool = False,
        revision: str | None = None,
    ) -> None:
        self._dataset_id = dataset_id
        self._split = split
        self._streaming = streaming
        # Pin a HF dataset revision for byte-reproducible references. ``None``
        # resolves to ``main`` HEAD at run time (only applies to the network
        # download/stream path — a local parquet snapshot is already frozen).
        self._revision = revision
        self._cache: Dataset | IterableDataset | None = None
        self._source: str | None = None  # "local-parquet" | "hf-streaming" | "hf-download"

    def _load(self) -> Dataset | IterableDataset:
        from datasets import load_dataset

        if self._cache is not None:
            return self._cache

        local_parquets = _find_local_parquets(self._dataset_id)
        if local_parquets:
            # Direct parquet load bypasses the dataset-script + network round-trip
            # entirely. Honors ``streaming`` flag — IterableDataset for memory-safe
            # iteration, Dataset for repeat access.
            ds = load_dataset(
                "parquet",
                data_files={self._split: local_parquets},
                split=self._split,
                streaming=self._streaming,
            )
            self._source = "local-parquet"
        else:
            ds = load_dataset(
                self._dataset_id,
                split=self._split,
                streaming=self._streaming,
                revision=self._revision,
            )
            self._source = "hf-streaming" if self._streaming else "hf-download"

        self._cache = ds
        return ds

    @staticmethod
    def _decode_audio(audio_field: object) -> tuple[np.ndarray, int]:
        if not isinstance(audio_field, dict):
            raise TypeError(f"unexpected audio field type: {type(audio_field)}")
        if "array" in audio_field and audio_field["array"] is not None:
            arr = np.asarray(audio_field["array"], dtype=np.float32)
            sr = int(audio_field.get("sampling_rate", TARGET_SAMPLE_RATE))
            return arr, sr
        raw = audio_field.get("bytes")
        if raw is None:
            raise ValueError("audio field has neither 'array' nor 'bytes'")
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(sr)

    def iter_samples(self, subset: str, limit: int | None = None) -> Iterator[Sample]:
        canonical_subset = SUBSET_ALIASES.get(subset, subset)
        if canonical_subset != "All" and canonical_subset not in SUBSETS:
            raise ValueError(
                f"unknown subset {subset!r}; expected one of "
                f"{(*SUBSETS, *SUBSET_ALIASES.keys(), 'All')}"
            )
        ds: Any = self._load()
        emitted = 0
        for idx, row in enumerate(ds):
            row_from = row.get("from")
            if canonical_subset != "All" and row_from != canonical_subset:
                continue
            arr, sr = self._decode_audio(row["audio"])
            yield Sample(
                audio=arr,
                sample_rate=sr,
                reference=row["references"],
                sample_id=f"{row_from or 'unknown'}-{idx}",
                subset=row_from or canonical_subset,
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return
