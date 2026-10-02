"""Common types for dataset loaders."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import numpy as np


@dataclass(frozen=True)
class Sample:
    """A single ASR evaluation sample.

    Attributes:
        audio: PCM float32 mono samples in [-1.0, 1.0].
        sample_rate: Hz, the corpus's native rate (16000 for the flozi-aligned
            sets, 44100 for the Swiss corpora). Adapters whose endpoint needs
            a fixed rate convert at their own boundary.
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


#: Why a corpus row produced no sample. The keys of ``RowTally.dropped`` and of
#: ``dropped_by_reason`` in a run's ``summary.json``.
DROP_EMPTY_REFERENCE: Final[str] = "empty_reference"
#: The row points at a reference that does not exist (xSID: no parallel
#: Standard German sentence), as opposed to one that exists and is empty.
DROP_REFERENCE_MISSING: Final[str] = "reference_missing"
DROP_CLIP_MISSING: Final[str] = "clip_missing"
DROP_UNDECODABLE: Final[str] = "undecodable"

#: What decoding a damaged clip raises: ``soundfile.LibsndfileError`` is a
#: ``RuntimeError``; a malformed row shape surfaces as one of the others.
#: Deliberately not ``Exception`` — a missing ``soundfile`` install must abort
#: the run, not be counted as one undecodable clip per row.
DECODE_ERRORS: Final[tuple[type[Exception], ...]] = (
    RuntimeError, ValueError, TypeError,
)

_tally_logger = logging.getLogger("raven_asr.datasets")


@dataclass
class RowTally:
    """What one ``iter_samples`` pass did with the corpus rows it read.

    A loader that cannot yield a row — no reference text, no clip in the
    archive, bytes that do not decode — skips it so one bad row does not abort a
    run over thousands. Skipping is only honest if it is counted: a score over
    2,870 of 2,875 rows that reads as a score over the corpus is the failure
    this type exists to prevent. The loader keeps the tally on ``self.tally``;
    the runner copies it into the run summary.

    ``rows_read`` counts rows the pass looked at, so under ``--limit`` it covers
    the rows up to the limit, not the corpus.
    """

    rows_read: int = 0
    yielded: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    @property
    def n_dropped(self) -> int:
        return sum(self.dropped.values())

    def log(self, dataset: str) -> None:
        """Report the pass; a warning as soon as a single row was dropped."""
        if self.n_dropped:
            reasons = ", ".join(f"{k}={v}" for k, v in sorted(self.dropped.items()))
            _tally_logger.warning(
                "[%s] %d of %d rows read were DROPPED and are not scored (%s); "
                "%d yielded",
                dataset, self.n_dropped, self.rows_read, reasons, self.yielded,
            )
        else:
            _tally_logger.info(
                "[%s] %d rows read, %d yielded, none dropped",
                dataset, self.rows_read, self.yielded,
            )


class DatasetLoader(Protocol):
    """Protocol implemented by every dataset loader.

    Loaders must yield samples lazily so the runner can apply --limit
    without materializing the full dataset.

    Every loader in this package keeps a :class:`RowTally` on ``self.tally``,
    reset at the start of every ``iter_samples`` pass. It is not part of the
    protocol, so a test double may omit it: a loader without one is reported as
    "drops unknown" (``null``), which is different from "no drops".
    """

    name: str

    def iter_samples(self, subset: str, limit: int | None = None) -> Iterator[Sample]:
        """Yield up to ``limit`` samples for the requested subset."""
        ...
