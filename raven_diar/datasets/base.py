"""Common types for diarization dataset loaders.

A diarization loader is deliberately *file-oriented*, not sample-oriented (unlike
``raven_asr``): a diarizer consumes a whole recording (minutes long) and emits a
hypothesis RTTM for it, which is scored against a gold RTTM for the same file. So
loaders yield :class:`DiarFile` — a ``(file_id, audio_path, gold_rttm_path)``
triple — and never materialise audio arrays in memory.

``prepare()`` is the download step the USER runs (large audio is never fetched by
CI or committed here). ``iter_files()`` then walks the prepared tree lazily so
``--limit`` works without reading every file.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DiarFile:
    """One recording + its gold diarization.

    Attributes:
        file_id: Stable identifier (RTTM ``file-id``), used for gold/hyp filenames.
        audio_path: Path to the recording the diarizer runs on. ``None`` when the
            gold RTTM is present but the (large, caller-fetched) audio is not — the
            runner treats that as a skip-with-warning, never a silent pass.
        gold_rttm_path: Path to the gold (reference) RTTM.
        dataset: Dataset id (e.g. ``voxconverse``).
    """

    file_id: str
    audio_path: Path | None
    gold_rttm_path: Path
    dataset: str


class DiarDatasetLoader(Protocol):
    """Protocol implemented by every diarization dataset loader."""

    name: str
    dataset_id: str

    def prepare(self, root: Path, revision: str | None = None) -> None:
        """Download / materialise gold RTTMs (and document audio) under ``root``.

        Idempotent: a no-op when the expected files are already present. Large
        audio that a dataset's license forbids redistributing is NOT fetched
        here — the loader documents the exact caller-run download instead.
        """
        ...

    def iter_files(self, root: Path, limit: int | None = None) -> Iterator[DiarFile]:
        """Yield up to ``limit`` prepared :class:`DiarFile` records lazily."""
        ...
