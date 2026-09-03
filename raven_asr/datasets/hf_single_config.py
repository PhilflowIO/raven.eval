"""Shared base for the single-config HF audio corpora (FLEURS / MLS-de / VoxPopuli-de).

``flozi_mixed_evals`` is a one-repo, one-split dataset whose three subsets live in
a ``from`` column, so it can load flat ``data/*.parquet`` files out of a local HF
snapshot. The corpora in this module are shaped differently: one *config* per
language inside a multi-hundred-GB repo, with a conventional
``{train,validation,test}`` split layout. There is no flat parquet directory to
short-circuit to, so the fetch goes through ``datasets.load_dataset`` with the
config name and a **pinned revision** — the harness fetches the corpus itself, it
never assumes it is already on disk.

Streaming is the default here for the same reason: a full ``test`` split of
VoxPopuli-de or MLS-de is far larger than the handful of records a Tier-2 run
consumes, and ``--limit`` must not pay for a full download first. Set
``streaming=False`` to materialize into the local HF cache instead (useful when
the same split is re-run repeatedly).

Every loader in this module yields the same :class:`~raven_asr.datasets.base.Sample`
contract as ``flozi_mixed_evals`` — one dataset contract in this repo, not two.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Sample

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset, IterableDataset

TARGET_SAMPLE_RATE = 16000

# Subset selector accepted by every loader in addition to its own id.
ALL_SUBSETS = "All"


class HFSingleConfigLoader:
    """Streams :class:`Sample` records from one config/split of an HF corpus.

    Subclasses set the class attributes below. The ``subset`` attribute is the
    dataset id as it appears in ``benchmark.config.yaml`` — the loader accepts
    exactly that id (or ``"All"``) so a caller's ``--dataset fleurs`` reaches the
    right loader without a second naming scheme.
    """

    # --- subclass contract ---------------------------------------------------
    name: str = ""
    subset: str = ""                # == benchmark.config.yaml datasets.wer[].id
    hf_dataset_id: str = ""
    hf_config: str | None = None
    hf_split: str = "test"
    audio_column: str = "audio"
    text_column: str = "text"
    license: str = "unknown"        # SPDX-ish slug
    source_url: str = ""            # human-readable provenance link

    def __init__(self, streaming: bool = True, revision: str | None = None) -> None:
        self._streaming = streaming
        # Pin a HF dataset revision for byte-reproducible references. ``None``
        # resolves to ``main`` HEAD at run time — the pins live in
        # ``raven_asr.config.WER_DATASETS`` and the runner threads them here.
        self._revision = revision
        self._cache: Dataset | IterableDataset | None = None
        self._source: str | None = None  # "hf-streaming" | "hf-download"

    # ----- loading -----------------------------------------------------------

    def _load(self) -> Dataset | IterableDataset:
        from datasets import load_dataset

        if self._cache is not None:
            return self._cache

        kwargs: dict[str, Any] = {
            "split": self.hf_split,
            "streaming": self._streaming,
            "revision": self._revision,
        }
        if self.hf_config is not None:
            ds = load_dataset(self.hf_dataset_id, self.hf_config, **kwargs)
        else:
            ds = load_dataset(self.hf_dataset_id, **kwargs)
        self._source = "hf-streaming" if self._streaming else "hf-download"
        self._cache = ds
        return ds

    # ----- per-row extraction ------------------------------------------------

    @staticmethod
    def _decode_audio(audio_field: object) -> tuple[np.ndarray, int]:
        """Decode a ``datasets`` Audio() field to (float32 mono, sample_rate)."""
        if not isinstance(audio_field, dict):
            raise TypeError(f"unexpected audio field type: {type(audio_field)}")
        arr_obj = audio_field.get("array")
        if arr_obj is not None:
            arr = np.asarray(arr_obj, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1).astype(np.float32)
            sr = int(audio_field.get("sampling_rate", TARGET_SAMPLE_RATE))
            return arr, sr
        # Bytes form (rare for these configs) — decoded with soundfile, which
        # dodges the torchcodec dependency an Audio() cast would pull in.
        raw = audio_field.get("bytes")
        if raw is None:
            raise ValueError("audio field has neither 'array' nor 'bytes'")
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(sr)

    def _reference_from_row(self, row: dict[str, Any]) -> str:
        ref = row.get(self.text_column)
        if ref is None:
            raise KeyError(
                f"{self.name}: text column {self.text_column!r} not in row "
                f"(have {sorted(row.keys())})"
            )
        return str(ref)

    def _sample_id(self, idx: int, row: dict[str, Any]) -> str:
        return f"{self.subset}-{idx}"

    # ----- public contract ---------------------------------------------------

    def iter_samples(self, subset: str, limit: int | None = None) -> Iterator[Sample]:
        if subset not in (self.subset, ALL_SUBSETS):
            raise ValueError(
                f"unknown subset {subset!r} for loader {self.name!r}; "
                f"expected one of {(self.subset, ALL_SUBSETS)}"
            )
        ds: Any = self._load()
        emitted = 0
        for idx, row in enumerate(ds):
            arr, sr = self._decode_audio(row[self.audio_column])
            reference = self._reference_from_row(row)
            if not reference:
                # An empty reference makes WER undefined (division by zero on
                # that utterance's word count) — drop it loudly rather than
                # letting it silently deflate the corpus denominator.
                continue
            yield Sample(
                audio=arr,
                sample_rate=sr,
                reference=reference,
                sample_id=self._sample_id(idx, row),
                subset=self.subset,
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return
