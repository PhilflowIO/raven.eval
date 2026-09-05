"""DiariZen adapter (behind the ``diarizen`` extra).

The fifth diarizer, and the first one this page cites *against* itself: the ETH
diarization benchmark names DiariZen the strongest open-source candidate on
German telephone speech, and every other row here was measured while that claim
stayed unmeasured. It is also the only candidate believed not to collapse in the
many-speaker regime — its clustering resolves up to 20 speakers globally, where
Sortformer is hard-capped at four — which is exactly the regime a meeting product
lives in.

**The weights are CC-BY-NC-4.0.** Like Sortformer v1, this is a reference row
under ADR-app-0036: measured and shown, never awarded, never shipped. The
registry test enforces that, and ``shippable=False`` in the spec is the
declaration it reads.

**Why this lane is heavier than the others.** DiariZen does not ship a usable
pyannote.audio dependency — it vendors a *modified* pyannote.audio 3.1.1 inside
its own repository, and the two import each other (the fork's
``pyannote/audio/core/model.py`` imports ``diarizen.utils``, and its
``pipelines/clustering.py`` imports ``diarizen.clustering.VBx``). The fork also
changes signatures the pipeline calls: ``SpeakerDiarization.__init__`` takes
``config``/``seg_duration``/``device``, ``get_segmentations`` takes ``soft``, and
``VBxClustering`` — the clustering method the published checkpoints configure —
does not exist upstream at all. So upstream ``pyannote.audio==3.1.1`` is NOT a
substitute, and this extra installs both halves from the same pinned commit.
That is also why it is its own extra and conflicts with ``diar``: pyannote.audio
3.1.1 and >=4.0 cannot share an environment.

**Both weight sets are pinned here, not by the library.**
``DiariZenPipeline.from_pretrained`` calls ``snapshot_download`` for the
checkpoint and ``hf_hub_download`` for the speaker-embedding model without a
``revision`` on either, so using it would leave a published number hanging on two
HuggingFace branches. This adapter downloads both itself at pinned revisions and
constructs the pipeline directly — the constructor is public and the library's
own CLI uses it the same way.

``torch``, ``huggingface_hub`` and ``diarizen`` are imported lazily inside
:meth:`diarize`, so importing this module for the registry pulls none of them and
the Tier-1 re-score path stays GPU-free.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from raven_eval_core.der import DiarSegment

from .base import DiarizeResult

MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"

#: The speaker-embedding model DiariZen clusters with. It is a SECOND weights
#: artifact that decides the number, fetched by the library from a different HF
#: repo — so it is pinned here beside the checkpoint rather than left to track a
#: branch. Public (not gated): no token, no licence acceptance.
EMBEDDING_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
EMBEDDING_FILENAME = "pytorch_model.bin"
#: HF `main` of the embedding repo as of 2026-09-04.
EMBEDDING_REVISION = "837717ddb9ff5507820346191109dc79c958d614"


class DiariZenDiarizer:
    """Diarizer adapter wrapping the DiariZen WavLM+Conformer pipeline."""

    def __init__(
        self,
        provider_id: str = "diarizen-wavlm-large-s80-md-v2",
        model_id: str = MODEL_ID,
        revision: str | None = None,
        # Accepted and ignored: this model is language-agnostic (it scores
        # speaker turns, not words). It is in the signature because the runner
        # builds EVERY adapter with the same kwargs — branching on
        # ``spec.hosted`` in the dispatcher is precisely what the registry
        # exists to prevent.
        language: str | None = None,
        embedding_revision: str | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        # Same precedence as the other local-weights adapters: explicit arg >
        # env override > floating "main". The spec pins the commit, so the
        # floating fallback only bites ad-hoc use.
        self.revision = revision or os.environ.get("DIARIZEN_REVISION") or "main"
        self.embedding_revision = (
            embedding_revision
            or os.environ.get("DIARIZEN_EMBEDDING_REVISION")
            or EMBEDDING_REVISION
        )
        self._pipeline = None  # lazily built on first diarize()

    @property
    def run_config(self) -> dict[str, object]:
        """What the runner records in ``summary.json`` beside the model pin.

        The embedding model is a second set of weights that changes the answer,
        so a summary naming only the checkpoint would under-describe the run.
        """
        return {
            "embedding_model_id": EMBEDDING_MODEL_ID,
            "embedding_revision": self.embedding_revision,
        }

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from diarizen.pipelines.inference import DiariZenPipeline  # diarizen extra
        from huggingface_hub import (  # diarizen extra
            hf_hub_download,
            snapshot_download,
        )

        # Deliberately NOT DiariZenPipeline.from_pretrained: it passes no
        # revision to either download, so both weight sets would follow a branch.
        hub = snapshot_download(repo_id=self.model_id, revision=self.revision)
        embedding = hf_hub_download(
            repo_id=EMBEDDING_MODEL_ID,
            filename=EMBEDDING_FILENAME,
            revision=self.embedding_revision,
        )
        # The snapshot IS the config: seg_duration, clustering method and the
        # speaker bounds come from its config.toml, so nothing here overrides
        # them — per-corpus tuning is what the scoring contract forbids.
        self._pipeline = DiariZenPipeline(
            diarizen_hub=Path(hub).expanduser().absolute(),
            embedding_model=embedding,
            rttm_out_dir=None,  # the harness writes the RTTMs, in its own layout
        )
        return self._pipeline

    def diarize(self, audio_path: Path) -> DiarizeResult:
        pipeline = self._ensure_pipeline()
        started = time.perf_counter()
        annotation = pipeline(str(audio_path))
        latency_s = time.perf_counter() - started
        segments: list[DiarSegment] = []
        for turn, _track, label in annotation.itertracks(yield_label=True):
            start_s, end_s = float(turn.start), float(turn.end)
            if end_s <= start_s:
                continue  # zero-length turns are not scoreable and break RTTM
            segments.append((start_s, end_s, str(label)))
        segments.sort(key=lambda t: (t[0], t[1], t[2]))
        return DiarizeResult(
            segments=segments,
            latency_s=latency_s,
            raw={
                "model_id": self.model_id,
                "revision": self.revision,
                "embedding_model_id": EMBEDDING_MODEL_ID,
                "embedding_revision": self.embedding_revision,
                "n_speakers": len({s[2] for s in segments}),
            },
        )


#: Registry entry point (see raven_diar.registry) — the runner resolves
#: ``DiarizerSpec.adapter == "diarizen"`` to this attribute.
ADAPTER = DiariZenDiarizer
