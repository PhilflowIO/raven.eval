"""NVIDIA Sortformer adapter (behind the ``sortformer`` extra).

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
  * Hard cap of **4 speakers** — every released checkpoint is ``4spk``. On audio
    with more speakers the extra ones are folded into the four tracks, which
    shows up as speaker confusion. That is a property of the model, not a bug
    here, and it is why AMI (4 speakers) and CALLHOME (2) are in scope and
    larger meetings are not.

**One adapter, two checkpoints.** ``diar_sortformer_4spk-v1`` (offline,
CC-BY-NC-4.0) and ``diar_streaming_sortformer_4spk-v2`` (streaming, CC-BY-4.0)
are the same ``SortformerEncLabelModel`` class with the same ``diarize()``
signature and the same segment output shape, so a second adapter module would be
a copy with one constant changed. The two things that genuinely differ are data,
not code: the checkpoint filename (derived from the model id — both HF repos ship
``<repo-name>.nemo``) and the streaming configuration, which only the streaming
checkpoint has. ``LATENCY_PRESETS`` holds NVIDIA's four recommended streaming
configurations by name and ``DEFAULT_PRESETS`` says which one a checkpoint runs
under; v1 is deliberately absent from the latter, because applying a streaming
config to the offline checkpoint would change a number we have already
published.

The offline v1 checkpoint cannot process meeting-length audio at all — its
attention memory grows with the square of the duration (measured on an RTX 3090:
1.02 GB at 2 min, 12.65 GB at 10 min, OOM at 12 min), while the shortest AMI test
meeting is 14 minutes. The streaming checkpoint processes audio in fixed-size
chunks with a bounded speaker cache, which is what makes AMI reachable.

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

#: NVIDIA's four recommended streaming configurations, verbatim from the model
#: card's "Setting up Streaming Configuration" table (fetched 2026-09-04), all in
#: 80 ms frames. Latency is INPUT-BUFFER latency (chunk + right context), which
#: is why it is in the key: the name is the thing a product decision is made on,
#: the five integers are how it is spelled to NeMo.
#:
#:   name                latency   RTF     chunk  right  fifo  update  spkcache
#:   very-high-latency    30.4 s   0.002    340     40     40    300      188
#:   high-latency         10.0 s   0.005    124      1    124    124      188
#:   low-latency           1.04 s  0.093      6      7    188    144      188
#:   ultra-low-latency     0.32 s  0.180      3      1    188    144      188
#:
#: ONE config per run and no per-dataset tuning: the scoring contract forbids it,
#: and NVIDIA's post-processing YAMLs are exactly that (one optimised on
#: CALLHOME-part1, another on DIHARD-III-dev), so they are not applied here.
LATENCY_PRESETS: dict[str, dict[str, int]] = {
    "very-high-latency": {
        "chunk_len": 340,
        "chunk_right_context": 40,
        "fifo_len": 40,
        "spkcache_update_period": 300,
        "spkcache_len": 188,
    },
    "high-latency": {
        "chunk_len": 124,
        "chunk_right_context": 1,
        "fifo_len": 124,
        "spkcache_update_period": 124,
        "spkcache_len": 188,
    },
    "low-latency": {
        "chunk_len": 6,
        "chunk_right_context": 7,
        "fifo_len": 188,
        "spkcache_update_period": 144,
        "spkcache_len": 188,
    },
    "ultra-low-latency": {
        "chunk_len": 3,
        "chunk_right_context": 1,
        "fifo_len": 188,
        "spkcache_update_period": 144,
        "spkcache_len": 188,
    },
}

#: Which preset a checkpoint runs under when nobody says otherwise. This is the
#: PUBLISHED configuration: every committed v2 row was measured at
#: "very-high-latency", the lowest-DER and lowest-RTF row NVIDIA offers.
#: A model id that is ABSENT here gets no streaming config at all — that is the
#: case for the offline v1 checkpoint, whose published rows must not shift
#: underneath us, and asking for a preset on it is an error rather than a no-op.
DEFAULT_PRESETS: dict[str, str] = {
    "nvidia/diar_streaming_sortformer_4spk-v2": "very-high-latency",
}

#: Env override for the preset, mirroring ``SORTFORMER_REVISION``. Selecting a
#: non-default preset is a DIAGNOSTIC move (it answers "does a lower latency
#: setting explain a gap against someone else's number"), never a publishing one
#: — which is why the effective preset travels into ``summary.json`` via
#: :attr:`SortformerDiarizer.run_config`, so a run measured under a non-default
#: setting cannot be promoted as if it were the published one.
PRESET_ENV_VAR = "SORTFORMER_LATENCY_PRESET"


def _checkpoint_filename(model_id: str) -> str:
    """The ``.nemo`` file inside the HF repo; fetched at the pinned revision.

    Both Sortformer repos name the checkpoint after the repo itself
    (``nvidia/diar_sortformer_4spk-v1`` → ``diar_sortformer_4spk-v1.nemo``,
    verified against the HF file listings on 2026-09-03), so this is a
    convention rather than a table that has to be maintained per checkpoint.
    """
    return f"{model_id.rsplit('/', 1)[-1]}.nemo"


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
        latency_preset: str | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        # Revision precedence: explicit arg > env override > floating "main".
        # A published number must never hang on "main" — the spec in config.py
        # pins the commit, so the floating fallback only bites ad-hoc use.
        self.revision = revision or os.environ.get("SORTFORMER_REVISION") or "main"
        self._device = device
        self._batch_size = batch_size
        # Same precedence for the streaming preset, and the same reason: the
        # runner builds every adapter with the same three kwargs, so an ad-hoc
        # diagnostic gets at this through the environment rather than through a
        # per-adapter kwarg the dispatcher would have to know about.
        self.latency_preset = self._resolve_preset(
            latency_preset or os.environ.get(PRESET_ENV_VAR) or None
        )
        self._streaming_config = (
            dict(LATENCY_PRESETS[self.latency_preset]) if self.latency_preset else {}
        )
        self._model = None  # lazily built on first diarize()

    def _resolve_preset(self, requested: str | None) -> str | None:
        """Name of the preset this instance runs under, or None for offline v1.

        Fails loudly on an unknown name and on a preset asked of a checkpoint
        that has no streaming mode: both are configuration mistakes that would
        otherwise be discovered as a silently different DER.
        """
        default = DEFAULT_PRESETS.get(self.model_id)
        if requested is None:
            return default
        if requested not in LATENCY_PRESETS:
            raise ValueError(
                f"unknown latency preset {requested!r}; known: "
                f"{', '.join(LATENCY_PRESETS)}"
            )
        if default is None:
            raise ValueError(
                f"{self.model_id} is not a streaming checkpoint — it takes no "
                f"latency preset (asked for {requested!r})"
            )
        return requested

    @property
    def run_config(self) -> dict[str, object]:
        """What the runner records in ``summary.json`` beside the pins.

        The preset is part of WHAT WAS MEASURED, exactly like the revision: two
        runs of the same checkpoint at different latency settings are different
        numbers. Naming it in the summary is what keeps a diagnostic run from
        being promoted as if it were the published configuration.
        """
        return {
            "latency_preset": self.latency_preset,
            "streaming_config": self._streaming_config or None,
        }

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
            filename=_checkpoint_filename(self.model_id),
            revision=self.revision,
        )
        model = SortformerEncLabelModel.restore_from(
            restore_path=checkpoint,
            map_location=device,
            strict=False,
        )
        model.eval()
        self._apply_streaming_config(model)
        self._model = model
        return model

    def _apply_streaming_config(self, model) -> None:
        """Set the streaming knobs for a streaming checkpoint; no-op otherwise.

        The model card sets these as plain attributes on ``sortformer_modules``
        and then calls ``_check_streaming_parameters()``, which raises if the
        combination is invalid — so a typo here fails loudly at load time rather
        than silently producing a differently-configured number.
        """
        config = self._streaming_config
        if not config:
            return
        modules = model.sortformer_modules
        for key, value in config.items():
            if not hasattr(modules, key):
                raise ValueError(
                    f"{self.model_id}: sortformer_modules has no {key!r} — this "
                    f"NeMo build does not take the documented streaming config"
                )
            setattr(modules, key, value)
        modules._check_streaming_parameters()

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
                "checkpoint": _checkpoint_filename(self.model_id),
                "max_speakers": 4,
                "latency_preset": self.latency_preset,
                "streaming_config": self._streaming_config or None,
            },
        )


#: Registry entry point (see raven_diar.registry) — the runner resolves
#: ``DiarizerSpec.adapter == "sortformer"`` to this attribute.
ADAPTER = SortformerDiarizer
