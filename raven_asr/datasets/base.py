"""Common types for dataset loaders."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

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
    """

    audio: np.ndarray
    sample_rate: int
    reference: str
    sample_id: str
    subset: str


class DatasetLoader(Protocol):
    """Protocol implemented by every dataset loader.

    Loaders must yield samples lazily so the runner can apply --limit
    without materializing the full dataset.
    """

    name: str

    def iter_samples(self, subset: str, limit: int | None = None) -> Iterator[Sample]:
        """Yield up to ``limit`` samples for the requested subset."""
        ...
