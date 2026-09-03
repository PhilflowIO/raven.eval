"""VoxPopuli German loader — ``facebook/voxpopuli``, config ``de``.

VoxPopuli is European-Parliament plenary speech with force-aligned transcripts —
spontaneous, accented, far from the read-speech register of FLEURS and MLS, which
is exactly why it earns a slot next to them.

Provenance
    source        https://huggingface.co/datasets/facebook/voxpopuli
    revision      pinned in ``raven_asr.config.WER_DATASETS['voxpopuli-de'].revision``
    license       CC0-1.0 (card: "distributed under CC0 license"; the repo also
                  carries an ``other`` tag for the unlabelled portions we do not
                  touch).
    durability    rank 2 — versioned HF dataset with a pinned revision. No
                  sha256: the Hub revision is the content address, this is not a
                  loose fetched archive.

Schema notes (HF dataset card, verified 2026-06-07)
    * configs are bare language codes — German is ``"de"``.
    * splits: train / validation / test; we score ``test``.
    * ``audio`` is a ``datasets`` Audio() field at 16 kHz.
    * two text columns: ``raw_text`` (orthographic) and ``normalized_text``.
      ``raw_text`` is empty on some rows, so we prefer it and fall back to
      ``normalized_text``.
"""

from __future__ import annotations

from typing import Any

from .hf_single_config import HFSingleConfigLoader

DATASET_ID = "facebook/voxpopuli"


class VoxPopuliDeLoader(HFSingleConfigLoader):
    """Yields VoxPopuli German test samples."""

    name = DATASET_ID
    subset = "voxpopuli-de"
    hf_dataset_id = DATASET_ID
    hf_config = "de"
    hf_split = "test"
    audio_column = "audio"
    text_column = "raw_text"
    license = "CC0-1.0"
    source_url = "https://huggingface.co/datasets/facebook/voxpopuli"

    def _reference_from_row(self, row: dict[str, Any]) -> str:
        raw = row.get("raw_text")
        if raw:
            return str(raw)
        norm = row.get("normalized_text")
        if norm:
            return str(norm)
        return ""
