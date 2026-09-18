"""avemio German mixed test loader — ``avemio/ASR-GERMAN-MIXED-TEST``.

The test split of ``flozi00/asr-german-mixed``, re-uploaded test-split-only by
avemio so an eval does not pull the 136 GB full corpus. Re-transcribed Common
Voice + Multilingual LibriSpeech German, interleaved in one ``test`` split.

This is NOT the same corpus as ``german-mixed`` (``flozi00/asr-german-mixed-evals``,
three subsets selected by the ``from`` column): different repo, different row
order, different reference transcriptions. It is registered separately because
the Raven benchmark page publishes its "avemio" column on exactly these rows —
the first N of this split — and ADR-app-0054 requires every published number to
be reproducible from this repo.

Provenance
    source        https://huggingface.co/datasets/avemio/ASR-GERMAN-MIXED-TEST
    revision      pinned in ``raven_asr.config.WER_DATASETS['avemio-german-mixed-test'].revision``
                  (the repo's last commit, 2025-01-07 — the numbers the page
                  published were measured at an unpinned HEAD that was already
                  this commit)
    license       CC-BY-4.0 — the card declares no SPDX tag; it asks users to
                  respect the component licenses (Common Voice CC0, MLS
                  CC-BY-4.0), and CC-BY-4.0 is the strictest of the two.
                  Attribution: primeline + flozi00 (see /NOTICE).
    durability    rank 2 — versioned HF dataset at a pinned revision.

Schema notes (HF dataset card, verified 2026-06-07; re-checked 2026-09-18)
    * one ``test`` split, no config.
    * the card's own eval script reads ``transkription`` for the reference and
      casts ``audio`` to Audio(sampling_rate=16000).
"""

from __future__ import annotations

from .hf_single_config import HFSingleConfigLoader

DATASET_ID = "avemio/ASR-GERMAN-MIXED-TEST"


class AvemioGermanMixedTestLoader(HFSingleConfigLoader):
    """Yields avemio German mixed test samples in split order."""

    name = DATASET_ID
    subset = "avemio-german-mixed-test"
    hf_dataset_id = DATASET_ID
    hf_config = None
    hf_split = "test"
    audio_column = "audio"
    text_column = "transkription"
    license = "CC-BY-4.0"
    source_url = "https://huggingface.co/datasets/avemio/ASR-GERMAN-MIXED-TEST"
