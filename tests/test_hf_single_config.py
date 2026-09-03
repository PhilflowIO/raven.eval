"""Tests for the single-config HF loaders (FLEURS / MLS-de / VoxPopuli-de).

The regression these pin: ``datasets`` decodes an ``Audio()`` feature while it
builds the row, and since ``datasets`` 5.x that decode goes through
``torchcodec``. ``torchcodec`` ``dlopen``s the FFmpeg ``libav*`` shared objects,
so letting it into the WER lane would make a plain
``make reproduce METRIC=wer`` require a system FFmpeg build — the exact trap
``docs/TIER2-DER-KEYS.md`` documents for the DER lane. All three loaders failed
identically with ``ImportError: To support decoding audio data, please install
'torchcodec'`` before the ``decode=False`` cast, which is why they had never
been run against real data.

No network here: the fixtures stand in for a streamed HF dataset.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from raven_asr.datasets.fleurs_de import FleursDeLoader
from raven_asr.datasets.mls_german import MlsGermanLoader
from raven_asr.datasets.voxpopuli_de import VoxPopuliDeLoader

ALL_LOADERS = [FleursDeLoader, MlsGermanLoader, VoxPopuliDeLoader]


def _flac_bytes(seconds: float = 0.1, sample_rate: int = 48000) -> bytes:
    """A tiny real audio payload — decoded by soundfile, never by torchcodec."""
    buf = io.BytesIO()
    samples = np.zeros(int(seconds * sample_rate), dtype="float32")
    sf.write(buf, samples, sample_rate, format="FLAC")
    return buf.getvalue()


class _FakeDataset:
    """Stands in for a streamed HF dataset and records the cast it was given."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.casts: list[tuple[str, Any]] = []

    def cast_column(self, column: str, feature: Any) -> "_FakeDataset":
        self.casts.append((column, feature))
        return self

    def __iter__(self) -> Any:
        return iter(self._rows)


@pytest.mark.parametrize("loader_cls", ALL_LOADERS)
def test_audio_decoding_is_switched_off_at_load(loader_cls: type) -> None:
    """The Audio() feature must be cast to ``decode=False`` before iteration."""
    from datasets import Audio

    loader = loader_cls()
    ds = _FakeDataset([])
    loader._disable_audio_decoding(ds)

    assert len(ds.casts) == 1, "expected exactly one cast_column call"
    column, feature = ds.casts[0]
    assert column == loader.audio_column
    assert isinstance(feature, Audio)
    assert feature.decode is False, (
        "decoding must stay off — datasets would otherwise decode via torchcodec "
        "while building the row, before _decode_audio ever sees the field"
    )


@pytest.mark.parametrize("loader_cls", ALL_LOADERS)
def test_bytes_form_decodes_without_torchcodec(loader_cls: type) -> None:
    """The bytes branch is the normal path and must not import torchcodec."""
    loader = loader_cls()
    arr, sr = loader._decode_audio({"bytes": _flac_bytes(), "path": "x.flac"})

    assert arr.dtype == np.float32
    assert arr.ndim == 1
    assert sr == 48000, "the file's TRUE rate, not the rate the card declares"


def test_decode_reports_the_files_true_sample_rate() -> None:
    """A published WER must rest on the real rate, not the declared one.

    Dataset cards routinely declare 16 kHz for audio whose bytes are 48 kHz.
    Self-decoding is what makes that visible instead of silently resampled.
    """
    loader = FleursDeLoader()
    _, sr = loader._decode_audio({"bytes": _flac_bytes(sample_rate=44100)})
    assert sr == 44100


def test_missing_audio_payload_names_the_keys_it_saw() -> None:
    loader = FleursDeLoader()
    with pytest.raises(ValueError, match="neither 'array' nor 'bytes'"):
        loader._decode_audio({"path": "x.flac"})


@pytest.mark.parametrize("loader_cls", ALL_LOADERS)
def test_empty_reference_is_dropped_not_counted(loader_cls: type) -> None:
    """An empty reference makes WER undefined — it must not reach the scorer."""
    loader = loader_cls()
    audio = {"bytes": _flac_bytes(), "path": "x.flac"}
    loader._cache = _FakeDataset(
        [
            {loader.audio_column: audio, **{c: "" for c in _text_cols(loader)}},
            {loader.audio_column: audio, **{c: "echter text" for c in _text_cols(loader)}},
        ]
    )

    samples = list(loader.iter_samples(loader.subset))
    assert [s.reference for s in samples] == ["echter text"]


def _text_cols(loader: Any) -> tuple[str, ...]:
    """Every text column a loader might read, so the fixture satisfies all three."""
    return (
        loader.text_column,
        "raw_transcription",
        "transcription",
        "transcript",
        "raw_text",
        "normalized_text",
        "text",
    )


@pytest.mark.parametrize("loader_cls", ALL_LOADERS)
def test_unknown_subset_is_rejected(loader_cls: type) -> None:
    loader = loader_cls()
    with pytest.raises(ValueError, match="unknown subset"):
        next(loader.iter_samples("not-a-subset"))
