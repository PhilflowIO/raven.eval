"""Common types for diarizer adapters.

A diarizer adapter is the DER analogue of ``raven_asr``'s ASR adapter: it takes a
recording and returns a *hypothesis diarization* — a list of
``(start, end, speaker)`` segments — which the harness serialises to a hypothesis
RTTM and scores against gold. Adapters keep their heavy backends (torch,
pyannote.audio) behind lazy imports so this module stays importable GPU-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from raven_eval_core.der import DiarSegment


@dataclass(frozen=True)
class DiarizeResult:
    """Output of one diarization call.

    Attributes:
        segments: Hypothesis (start, end, speaker) turns.
        latency_s: Wall-clock seconds for the diarization (Tier-3 transparency —
            hardware-dependent, never a published comparable number).
        raw: Backend-specific debug payload (kept for traceability).
    """

    segments: list[DiarSegment]
    latency_s: float
    raw: dict[str, object] = field(default_factory=dict)


class Diarizer(Protocol):
    """Protocol every diarizer adapter implements."""

    provider_id: str
    model_id: str

    def diarize(self, audio_path: Path) -> DiarizeResult:
        """Run diarization on the recording at ``audio_path``."""
        ...
