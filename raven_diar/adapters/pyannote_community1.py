"""pyannote/speaker-diarization-community-1 adapter (behind the ``diar`` extra).

Runs the pinned ``pyannote/speaker-diarization-community-1`` pipeline on a
recording and returns its hypothesis turns. This is the ONE diarizer wired for
Etappe 5; the module is structured so a sortformer / diarizen adapter can be
added later by implementing the same :class:`~raven_diar.adapters.base.Diarizer`
protocol.

Three hard requirements — all documented in ``docs/TIER2-DER-KEYS.md``:
  1. An HF token (``HF_TOKEN`` / ``HUGGINGFACE_TOKEN``, or passed explicitly).
  2. Acceptance of the GATED model license on huggingface.co for the slug below.
  3. A GPU (CPU works but is impractically slow for meeting-length audio).

``torch`` + ``pyannote.audio`` are imported lazily inside :meth:`diarize` so
importing this module (e.g. for the registry) never pulls torch — the Tier-1
re-score path stays GPU-free.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from raven_eval_core.der import DiarSegment

from .base import DiarizeResult

MODEL_ID = "pyannote/speaker-diarization-community-1"


def _resolve_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for env in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env)
        if val:
            return val
    return None


class PyannoteCommunity1Diarizer:
    """Diarizer adapter wrapping the community-1 pretrained pipeline."""

    def __init__(
        self,
        provider_id: str = "pyannote-community-1",
        model_id: str = MODEL_ID,
        revision: str | None = None,
        # Accepted and ignored: this model is language-agnostic (it scores
        # speaker turns, not words). It is in the signature because the runner
        # builds EVERY adapter with the same kwargs — branching on
        # ``spec.hosted`` in the dispatcher is precisely what the registry
        # exists to prevent.
        language: str | None = None,
        hf_token: str | None = None,
        device: str | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        # Revision precedence: explicit arg > env override > floating "main".
        self.revision = (
            revision
            or os.environ.get("PYANNOTE_COMMUNITY1_REVISION")
            or "main"
        )
        self._hf_token = _resolve_token(hf_token)
        self._device = device
        self._pipeline = None  # lazily built on first diarize()

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        import torch  # diar extra
        from pyannote.audio import Pipeline  # diar extra

        pipeline = Pipeline.from_pretrained(
            self.model_id,
            token=self._hf_token,
            revision=self.revision,
        )
        if pipeline is None:
            raise RuntimeError(
                f"Pipeline.from_pretrained({self.model_id!r}) returned None — "
                "accept the gated model license on huggingface.co and provide a "
                "valid HF token (see docs/TIER2-DER-KEYS.md)."
            )
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(torch.device(device))
        self._pipeline = pipeline
        return pipeline

    def diarize(self, audio_path: Path) -> DiarizeResult:
        pipeline = self._ensure_pipeline()
        started = time.perf_counter()
        output = pipeline(str(audio_path))
        latency_s = time.perf_counter() - started
        # pyannote.audio >= 4 returns a DiarizeOutput dataclass; 3.x returned the
        # Annotation directly. Score the overlap-aware speaker_diarization in both
        # cases (never exclusive_speaker_diarization, which drops overlapped speech
        # and would silently understate confusion against overlap-aware gold).
        annotation = getattr(output, "speaker_diarization", output)
        segments: list[DiarSegment] = [
            (float(turn.start), float(turn.end), str(label))
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
        return DiarizeResult(
            segments=segments,
            latency_s=latency_s,
            raw={"model_id": self.model_id, "revision": self.revision},
        )


#: Registry entry point (see raven_diar.registry) — the runner resolves
#: ``DiarizerSpec.adapter == "pyannote_community1"`` to this attribute.
ADAPTER = PyannoteCommunity1Diarizer
