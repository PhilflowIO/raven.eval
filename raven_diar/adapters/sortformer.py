"""nvidia/diar_sortformer_4spk-v1 adapter (behind the ``sortformer`` extra).

The second diarizer, and the proof that the registry seam is real: this module
plus one :class:`~raven_diar.config.DiarizerSpec` entry is the whole change — the
runner's dispatch was not touched.

Sortformer is an end-to-end neural diarizer (FastConformer + Transformer, 123 M
params) that emits arrival-time-ordered speaker turns directly, so no
word/utterance folding is involved (see ``adapters/aggregate.py`` for why that
matters for the hosted adapters, and why it must NOT be applied here).

Compared with pyannote community-1 this one is cheaper to run:
  * **No HF token, no gated licence.** The weights are public (CC-BY-NC-4.0 —
    non-commercial; fine for a published benchmark, read it before shipping it).
  * **No API key of any kind** — local GPU inference.
  * A GPU is still strongly recommended; CPU works but is impractically slow.
  * Hard cap of **4 speakers** — the checkpoint is ``4spk-v1``. On audio with
    more speakers the extra ones are folded into the four tracks, which shows up
    as speaker confusion. That is a property of the model, not a bug here, and it
    is why AMI (4 speakers) and CALLHOME (2) are in scope and larger meetings are
    not.

``nemo`` and ``torch`` are imported lazily inside :meth:`diarize`, so importing
this module for the registry never pulls either — the Tier-1 re-score path stays
GPU-free and NeMo-free, and the repo stays installable and testable without the
``sortformer`` extra.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from raven_eval_core.der import DiarSegment

from .base import DiarizeResult

MODEL_ID = "nvidia/diar_sortformer_4spk-v1"
#: The ``.nemo`` checkpoint inside the HF repo; fetched at the pinned revision.
CHECKPOINT_FILENAME = "diar_sortformer_4spk-v1.nemo"


def _parse_segment(item: object) -> DiarSegment | None:
    """Turn one NeMo segment into ``(start, end, speaker)``.

    ``SortformerEncLabelModel.diarize`` documents its output as strings shaped
    ``"<begin_seconds> <end_seconds> <speaker_index>"``; released NeMo versions
    have also returned plain tuples. Accept both rather than pin the harness to
    one NeMo release, and reject anything else loudly.
    """
    if isinstance(item, str):
        parts = item.split()
        if len(parts) < 3:
            raise ValueError(f"unparseable sortformer segment: {item!r}")
        start, end, speaker = parts[0], parts[1], " ".join(parts[2:])
    elif isinstance(item, (tuple, list)) and len(item) >= 3:
        start, end, speaker = item[0], item[1], item[2]
    else:
        raise ValueError(f"unexpected sortformer segment type: {item!r}")
    start_s, end_s = float(start), float(end)
    if end_s <= start_s:
        return None  # zero-length turns are not scoreable and break RTTM
    return (start_s, end_s, str(speaker))


class SortformerDiarizer:
    """Diarizer adapter wrapping NVIDIA's Sortformer 4-speaker checkpoint."""

    def __init__(
        self,
        provider_id: str = "sortformer-4spk-v1",
        model_id: str = MODEL_ID,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 1,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        # Revision precedence: explicit arg > env override > floating "main".
        # A published number must never hang on "main" — the spec in config.py
        # pins the commit, so the floating fallback only bites ad-hoc use.
        self.revision = revision or os.environ.get("SORTFORMER_REVISION") or "main"
        self._device = device
        self._batch_size = batch_size
        self._model = None  # lazily built on first diarize()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        import torch  # sortformer extra
        from huggingface_hub import hf_hub_download  # sortformer extra
        from nemo.collections.asr.models import (  # sortformer extra
            SortformerEncLabelModel,
        )

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Download the pinned checkpoint ourselves instead of
        # ``from_pretrained(model_id)``: NeMo's loader takes no ``revision``, so
        # it would silently track the HF branch and a published DER could drift
        # under a re-upload. hf_hub_download does take one.
        checkpoint = hf_hub_download(
            repo_id=self.model_id,
            filename=CHECKPOINT_FILENAME,
            revision=self.revision,
        )
        model = SortformerEncLabelModel.restore_from(
            restore_path=checkpoint,
            map_location=device,
            strict=False,
        )
        model.eval()
        self._model = model
        return model

    def diarize(self, audio_path: Path) -> DiarizeResult:
        model = self._ensure_model()
        started = time.perf_counter()
        predicted = model.diarize(audio=str(audio_path), batch_size=self._batch_size)
        latency_s = time.perf_counter() - started
        # One entry per input recording; we always pass exactly one.
        per_file = predicted[0] if predicted and isinstance(predicted[0], list) else predicted
        segments: list[DiarSegment] = []
        for item in per_file or []:
            parsed = _parse_segment(item)
            if parsed is not None:
                segments.append(parsed)
        segments.sort(key=lambda t: (t[0], t[1], t[2]))
        return DiarizeResult(
            segments=segments,
            latency_s=latency_s,
            raw={
                "model_id": self.model_id,
                "revision": self.revision,
                "checkpoint": CHECKPOINT_FILENAME,
                "max_speakers": 4,
            },
        )


#: Registry entry point (see raven_diar.registry) — the runner resolves
#: ``DiarizerSpec.adapter == "sortformer"`` to this attribute.
ADAPTER = SortformerDiarizer
