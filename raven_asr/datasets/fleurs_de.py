"""FLEURS German loader — ``google/fleurs``, config ``de_de``.

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is
the speech version of the FLoRes MT benchmark: read parallel sentences, 102
languages, one config per ``<lang>_<region>``. German is ``de_de``.

Provenance
    source        https://huggingface.co/datasets/google/fleurs
    revision      pinned in ``raven_asr.config.WER_DATASETS['fleurs'].revision``
    license       CC-BY-4.0 (the card states "Creative Commons license (CC-BY)";
                  the repo is tagged ``cc-by-4.0``)
    durability    rank 2 — versioned HF dataset with a pinned revision. The
                  corpus also has a paper (arXiv 2205.12446), but the acquisition
                  path we actually run is the HF repo, so that is what is pinned.
                  No sha256 is recorded: a pinned HF revision is content-addressed
                  by the Hub, this is not a fetched loose archive.

Schema notes (HF dataset card, verified 2026-06-07)
    * splits: train / validation / test; ``test`` is ~350 sentences.
    * ``audio`` is a ``datasets`` Audio() field at 16 kHz.
    * two text columns: ``transcription`` (normalized) and ``raw_transcription``.
      We prefer ``raw_transcription`` — un-normalized, closer to the spoken form,
      and the flozi-strict normalizer downstream expects to do its own work.
"""

from __future__ import annotations

from typing import Any

from .hf_single_config import HFSingleConfigLoader

DATASET_ID = "google/fleurs"


class FleursDeLoader(HFSingleConfigLoader):
    """Yields FLEURS German test samples."""

    name = DATASET_ID
    subset = "fleurs"
    hf_dataset_id = DATASET_ID
    hf_config = "de_de"
    hf_split = "test"
    audio_column = "audio"
    text_column = "raw_transcription"
    license = "CC-BY-4.0"
    source_url = "https://huggingface.co/datasets/google/fleurs"

    def _reference_from_row(self, row: dict[str, Any]) -> str:
        raw = row.get("raw_transcription")
        if raw:
            return str(raw)
        norm = row.get("transcription")
        if norm:
            return str(norm)
        return ""
