"""Multilingual LibriSpeech German loader — ``facebook/multilingual_librispeech``.

MLS is read audiobook speech derived from LibriVox. The German config is spelled
out (``"german"``, not ``"de"``).

Provenance
    source        https://huggingface.co/datasets/facebook/multilingual_librispeech
    revision      pinned in ``raven_asr.config.WER_DATASETS['mls-de'].revision``
    license       CC-BY-4.0 (card: "Public Domain, Creative Commons Attribution
                  4.0 International Public License (CC-BY-4.0)") — attribution in
                  ``/NOTICE``.
    durability    rank 2 — versioned HF dataset with a pinned revision. Upstream
                  is the OpenSLR SLR94 archive (institutional, rank 1), but the
                  path this harness runs is the HF repo, so that is what is
                  pinned; no sha256, the Hub revision is the content address.

Schema note that cost a sweep once
    The HF card *prose* lists a ``text`` field, but the streamed schema has no
    such column — the reference lives in ``transcript`` (row keys: audio,
    audio_duration, begin_time, chapter_id, end_time, file, id, original_path,
    speaker_id, transcript). Trusting the card over the schema made every record
    raise ``KeyError`` and produced zero MLS rows in the 2026-06-08 sweep of the
    internal bench. Keep ``text_column = "transcript"``.
"""

from __future__ import annotations

from .hf_single_config import HFSingleConfigLoader

DATASET_ID = "facebook/multilingual_librispeech"


class MlsGermanLoader(HFSingleConfigLoader):
    """Yields MLS German test samples."""

    name = DATASET_ID
    subset = "mls-de"
    hf_dataset_id = DATASET_ID
    hf_config = "german"
    hf_split = "test"
    audio_column = "audio"
    text_column = "transcript"
    license = "CC-BY-4.0"
    source_url = "https://huggingface.co/datasets/facebook/multilingual_librispeech"
