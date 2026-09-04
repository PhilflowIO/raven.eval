"""SPC test loader — Swiss Parliaments Corpus, Bernese cantonal parliament.

Swiss German **speech** paired with Standard German **text**: the speaker talks
Bernese dialect, the parliamentary record writes standard German. That makes the
corpus translation-shaped rather than transcription-shaped, which is why it
declares ``metric="bleu+wer"`` in ``raven_asr.config.WER_DATASETS`` — WER alone
punishes a correct translation for not being a transliteration.

Provenance
    source        https://huggingface.co/datasets/i4ds/SPC_test
    revision      48c389fb6b88c8e80f03273a677a422729183e06 (2025-05-21), pinned in
                  ``raven_asr.config.WER_DATASETS['spc-test'].revision``
    acquisition   the two test parquet shards, fetched by URL from the Hub's
                  ``resolve/<revision>`` endpoint and verified against the
                  sha256 pins in :data:`SHARDS` (which match HF paths-info,
                  checked 2026-09-01)
    durability    rank 2 ("hf") — a versioned Hub repo at a pinned revision.
    license       MIT, **inferred, not stated**. See the licence note below.

Licence note (do not quietly upgrade this to a fact)
    The ``i4ds/SPC_test`` dataset card carries no licence tag at all — the Hub
    API returns ``cardData`` without a ``license`` key (checked 2026-09-03). MIT
    is inherited from the upstream FHNW publication this repo re-releases:
    https://www.cs.technik.fhnw.ch/i4ds-datasets lists "Swiss Parliaments Corpus
    … License: MIT". MIT permits commercial evaluation, so the inference is
    load-bearing; ``/NOTICE`` records it as an inference rather than as a stated
    licence.

Schema (verified against the local shards, columns re-checked on load)
    ``audio``         struct{bytes, path} — FLAC payload
    ``sentence``      Standard German transcript (the scoring reference)
    ``client_id``     int64 speaker id
    ``path``          clip filename
    ``iou_estimate``  float64 forced-alignment quality, surfaced as metadata

Sample-rate trap: the card declares a 16 kHz ``Audio()`` feature, but the raw
FLAC bytes decode at 48 kHz. We decode the bytes ourselves and yield the true
native rate — the ``datasets`` library would have resampled silently, and an
adapter that receives a WAV header claiming the wrong rate transcribes noise.

Only ~2 MB of the 810 MB is touched by a ``--limit 3`` smoke run: the shards are
read in small batches and the iterator returns as soon as the limit is hit.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

from .base import Sample
from .local_archive import RemoteArtifact, ensure_artifact, swiss_corpora_dir
from .local_archive import decode_audio_bytes as _decode

DATASET_ID: Final[str] = "spc-test"
HF_DATASET_ID: Final[str] = "i4ds/SPC_test"
HF_REVISION: Final[str] = "48c389fb6b88c8e80f03273a677a422729183e06"
SUBDIR: Final[str] = "spc"

_RESOLVE = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/{HF_REVISION}/data"

#: The two test shards, pinned by digest. Sizes and digests from
#: ``raven-bench-corpora/swiss/SOURCES.md`` (2026-09-01), each cross-checked
#: against the Hub's paths-info sha256 for the same file.
SHARDS: Final[tuple[RemoteArtifact, ...]] = (
    RemoteArtifact(
        url=f"{_RESOLVE}/test-00000-of-00002.parquet",
        filename="test-00000-of-00002.parquet",
        sha256="5bbf264be0d0faf2e63b94e4df2fa2f36bfa1ca9b4d3c2f8767057d2d9c0019f",
        size_bytes=406_025_446,
        durability="hf",
    ),
    RemoteArtifact(
        url=f"{_RESOLVE}/test-00001-of-00002.parquet",
        filename="test-00001-of-00002.parquet",
        sha256="1daf40810dbd2d19d9b0aaf30a7f0530c7cd6348d2210d2c9024be923c2cc128",
        size_bytes=404_266_081,
        durability="hf",
    ),
)

TEXT_COLUMN: Final[str] = "sentence"
AUDIO_COLUMN: Final[str] = "audio"

ALL_SUBSETS: Final[str] = "All"


class SpcTestLoader:
    """Yields SPC test utterances from the pinned, checksum-verified shards."""

    name = HF_DATASET_ID
    subset = DATASET_ID
    hf_dataset_id = HF_DATASET_ID
    hf_config = None
    hf_split = "test"
    audio_column = AUDIO_COLUMN
    text_column = TEXT_COLUMN
    license = "MIT (inferred from upstream FHNW SPC; the HF card carries no tag)"
    source_url = f"https://huggingface.co/datasets/{HF_DATASET_ID}"

    def __init__(self, streaming: bool = False, revision: str | None = None) -> None:
        # ``streaming`` is part of the loader constructor contract the runner
        # calls with; it has no meaning for a checksum-pinned local archive and
        # is accepted-and-ignored rather than silently changing behaviour.
        del streaming
        self._revision = revision or HF_REVISION
        if self._revision != HF_REVISION:
            raise ValueError(
                f"{DATASET_ID}: revision {self._revision!r} requested, but the "
                f"shard digests in this loader pin {HF_REVISION!r}. Re-pin "
                "SHARDS (URL + sha256 + size) together with the revision — a "
                "revision without matching digests verifies nothing."
            )

    def _shard_paths(self) -> list[Path]:
        directory = swiss_corpora_dir() / SUBDIR
        return [ensure_artifact(shard, directory) for shard in SHARDS]

    def iter_samples(self, subset: str, limit: int | None = None) -> Iterator[Sample]:
        if subset not in (self.subset, ALL_SUBSETS):
            raise ValueError(
                f"unknown subset {subset!r} for loader {self.name!r}; "
                f"expected one of {(self.subset, ALL_SUBSETS)}"
            )
        import pyarrow.parquet as pq

        emitted = 0
        for path in self._shard_paths():
            parquet = pq.ParquetFile(str(path))
            # Small batches keep memory flat: each row carries ~250 kB of FLAC.
            for batch in parquet.iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    reference = str(row.get(TEXT_COLUMN) or "")
                    if not reference:
                        # An empty reference makes WER undefined on that
                        # utterance; drop it rather than deflate the denominator.
                        continue
                    audio, sample_rate = _decode(row[AUDIO_COLUMN]["bytes"])
                    yield Sample(
                        audio=audio,
                        sample_rate=sample_rate,
                        reference=reference,
                        sample_id=self._sample_id(emitted, row),
                        subset=self.subset,
                    )
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return

    @staticmethod
    def _sample_id(index: int, row: dict[str, Any]) -> str:
        clip = row.get("path")
        return f"{DATASET_ID}-{clip}" if clip else f"{DATASET_ID}-{index}"
