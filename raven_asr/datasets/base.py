"""Common types for dataset loaders."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Sample:
    """A single ASR evaluation sample.

    Attributes:
        audio: PCM float32 mono samples in [-1.0, 1.0].
        sample_rate: Hz, always 16000 for flozi-aligned evaluation.
        reference: Ground-truth transcript (raw, pre-normalize).
        sample_id: Stable identifier for traceability across runs.
        subset: Subset label (e.g. "Tuda-De", "common_voice_19_0").
        metadata: Optional per-sample facts the four scoring fields cannot carry
            — dialect region, a secondary (dialectal) reference transcription,
            an intent label. Added for the dialect corpora, whose rows are only
            interpretable with their provenance attached; empty for every
            HF-backed loader. Trailing and defaulted, so existing positional
            construction is unaffected. Stable keys:

              * ``dialect_region``    — e.g. "oberbayern-laendlich"
              * ``reference_dialect`` — secondary dialectal reference text
              * ``split`` / ``clip``  — upstream split and clip path
    """

    audio: np.ndarray
    sample_rate: int
    reference: str
    sample_id: str
    subset: str
    # compare=False: provenance is not part of a sample's identity, and keeping
    # a dict out of the generated __eq__/__hash__ avoids surprises on a frozen
    # dataclass.
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


class DatasetLoader(Protocol):
    """Protocol implemented by every dataset loader.

    Loaders must yield samples lazily so the runner can apply --limit
    without materializing the full dataset.
    """

    name: str

    def iter_samples(self, subset: str, limit: int | None = None) -> Iterator[Sample]:
        """Yield up to ``limit`` samples for the requested subset."""
        ...
