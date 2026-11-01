"""Loader for ``flozi00/asr-german-mixed-evals`` — flozi-aligned reference dataset.

The dataset has a single ``train`` split containing rows from three sources
identified by the ``from`` column: ``Tuda-De``, ``multilingual librispeech``,
and ``common_voice_19_0``. Reference text lives in the ``references`` column,
audio in ``audio`` (bytes form, decoded with soundfile to dodge the
torchcodec dependency that ``Audio()`` casting requires).

The loader reads a local snapshot of the pinned revision's 25 parquet shards
(fetched once into the HF cache if absent; ``python -m raven_asr.prefetch`` does
it ahead of a run), so long bench runs don't depend on ``huggingface.co``
parquet streaming, which has been observed to time out intermittently on 5h+
jobs. ``streaming=True`` streams from the Hub instead. The snapshot is looked up
at the pinned revision only: a cached snapshot of another revision is not the
corpus the recorded pin names.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf

from .base import DECODE_ERRORS, DROP_UNDECODABLE, RowTally, Sample
from .hf_snapshot import local_split, snapshot_paths

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


#: The 25 shards of the single ``train`` split at the pinned revision.
DATA_FILES: tuple[str, ...] = tuple(
    f"data/train-{i:05d}-of-00025.parquet" for i in range(25)
)


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
        # Pin a HF dataset revision for byte-reproducible references. The runner
        # threads the registry pin; ``None`` only works with streaming, where it
        # resolves to ``main`` HEAD at run time.
        self._revision = revision
        self._cache: Dataset | IterableDataset | None = None
        self._source: str | None = None  # "local-snapshot" | "hf-streaming"
        self.tally = RowTally()

    def _load(self) -> Dataset | IterableDataset:
        if self._cache is not None:
            return self._cache

        if self._streaming:
            from datasets import load_dataset

            ds = load_dataset(
                self._dataset_id,
                split=self._split,
                streaming=True,
                revision=self._revision,
            )
            self._source = "hf-streaming"
        else:
            ds = local_split(self._dataset_id, self._revision, DATA_FILES, self._split)
            self._source = "local-snapshot"

        self._cache = ds
        return ds

    def prefetch(self) -> list[Path]:
        """Fetch the pinned shards into the local cache; return their paths."""
        paths = snapshot_paths(self._dataset_id, self._revision, DATA_FILES, download=True)
        assert paths is not None
        return paths

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
        # Counts the rows of the requested subset only: a row belonging to
        # another subset is filtered, not dropped. See RowTally.
        tally = self.tally = RowTally()
        try:
            for idx, row in enumerate(ds):
                row_from = row.get("from")
                if canonical_subset != "All" and row_from != canonical_subset:
                    continue
                tally.rows_read += 1
                try:
                    arr, sr = self._decode_audio(row["audio"])
                except DECODE_ERRORS:
                    tally.drop(DROP_UNDECODABLE)
                    continue
                tally.yielded += 1
                yield Sample(
                    audio=arr,
                    sample_rate=sr,
                    reference=row["references"],
                    sample_id=f"{row_from or 'unknown'}-{idx}",
                    subset=row_from or canonical_subset,
                )
                if limit is not None and tally.yielded >= limit:
                    return
        finally:
            tally.log(f"{self.name}/{canonical_subset}")
