"""Registry of public DER datasets + supported diarizers, and the scoring contract.

Mirrors ``raven_asr.config``. The two-collar contract is the SSOT that
``benchmark.config.yaml`` documents publicly; ``score.py`` reads ``COLLARS`` so
the reported numbers and the committed contract cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ── Scoring contract (must match benchmark.config.yaml `der.variants`) ────────
# DER is reported TWICE — the conventions are not comparable, so we publish both.
COLLARS: Final[dict[str, dict[str, float | bool]]] = {
    "full": {"collar": 0.0, "skip_overlap": False},     # pyannote / DIHARD
    "classic": {"collar": 0.25, "skip_overlap": False},  # NIST / CALLHOME
}


@dataclass(frozen=True)
class DiarDatasetSpec:
    """A public diarization dataset + how its gold RTTMs are obtained."""

    id: str
    loader: str          # loader module under raven_diar.datasets
    license: str
    # HF slug or upstream repo; pinned by ``revision`` for reproducibility.
    source: str
    revision: str        # git tag/commit or HF revision hash — pin, never floating
    notes: str = ""
    # Optional split selector for loaders that ship more than one gold split
    # (VoxConverse ships dev/ + test/). None → the loader default.
    split: str | None = None


# The PUBLIC datasets the DER Tier-2 harness scores against. Audio is
# pulled by the caller (never redistributed here); only gold/hyp *RTTM* labels
# are ever committed under artifacts/ (VoxConverse/AMI labels are CC-BY-4.0,
# CALLHOME is cite-to-use). See /NOTICE.
DER_DATASETS: Final[dict[str, DiarDatasetSpec]] = {
    # EASIEST — VoxConverse ships gold RTTMs directly (no conversion). Fastest
    # path to a real end-to-end DER; recommended first smoke.
    "voxconverse": DiarDatasetSpec(
        id="voxconverse",
        loader="voxconverse",
        license="CC-BY-4.0 (labels); audio = YouTube owners",
        source="https://github.com/joonson/voxconverse",
        # Pinned to an upstream COMMIT: joonson/voxconverse publishes no tags at
        # all (only branches master + ver0.2), so the previous "v0.3" pin made
        # prepare() fail with "Remote branch v0.3 not found".
        revision="24bf60be297701cd7e4ef18550c6d390c1b87365",
        notes="dev split; dev/ + test/ gold RTTMs shipped verbatim in the repo.",
        split="dev",
    ),
    # Same corpus, TEST split — the split pyannote reports its published
    # VoxConverse DER on, so this is the directly comparable number.
    "voxconverse-test": DiarDatasetSpec(
        id="voxconverse-test",
        loader="voxconverse",
        license="CC-BY-4.0 (labels); audio = YouTube owners",
        source="https://github.com/joonson/voxconverse",
        revision="24bf60be297701cd7e4ef18550c6d390c1b87365",
        notes="test split (232 files); gold RTTMs shipped verbatim in the repo.",
        split="test",
    ),
    # STRATEGIC — the German anchor. Gold labels via the diarizers-community
    # processing of talkbank/callhome (deu config). Converter in the loader.
    "callhome-de": DiarDatasetSpec(
        id="callhome-de",
        loader="callhome_de",
        license="TalkBank (cite-to-use)",
        source="talkbank/callhome",
        # Pinned to the HF dataset commit the published 16.08 % DER was measured
        # on (artifacts/2026-07-31-callhome-de/…/summary.json: dataset_revision).
        # "main" would let a published number drift with the upstream branch.
        revision="17c8a153215aa7c50b805078fd6284ba81c2fc47",
        notes="config=deu; 2-speaker German telephone; gold from speaker segments.",
    ),
    # AMI — 4-speaker meetings. Prepared gold RTTMs via the canonical
    # pyannote AMI-diarization-setup (only_words) at a pinned commit.
    "ami": DiarDatasetSpec(
        id="ami",
        loader="ami",
        license="CC-BY-4.0",
        source="https://github.com/pyannote/AMI-diarization-setup",
        # Pinned to the setup repo's HEAD as of 2022-10-24 (verified 2026-09-02;
        # the repo has 22 commits and no tags). The only_words/rttms/test gold
        # files were last touched in 126863e1 (2020-12-23) and are byte-identical
        # at this commit. The earlier placeholder "0c1b4b6" was not a commit of
        # that repository at all.
        revision="67c2d539286e89f68952d5dcf83912bd9f01dfae",
        notes=(
            "test split (16 meetings, Mix-Headset); gold RTTMs from the only_words "
            "annotations. Audio is fetched by prepare() from the Edinburgh AMI mirror."
        ),
        split="test",
    ),
}


@dataclass(frozen=True)
class DiarizerSpec:
    """Identifies a diarizer + its adapter binding for the runner."""

    model_id: str        # HF slug (gated model — needs an accepted license + token)
    adapter: str         # adapter module under raven_diar.adapters
    label: str           # short id used in result/artifact directory names
    revision: str        # HF revision hash/tag — pin, never floating


KNOWN_DIARIZERS: Final[dict[str, DiarizerSpec]] = {
    # pyannote speaker-diarization-community-1. GATED: accept the model license on
    # HF + provide an HF token + a GPU. Pin the revision to the commit you accept.
    "pyannote-community-1": DiarizerSpec(
        model_id="pyannote/speaker-diarization-community-1",
        adapter="pyannote_community1",
        label="pyannote-community-1",
        # Pinned to the HF model commit every published DER row was measured on
        # (artifacts/*/pyannote-community-1/summary.json: model_revision).
        revision="3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
    ),
    # NVIDIA Sortformer 4spk-v1. NOT gated, NO API key: public CC-BY-NC-4.0
    # weights run locally through NeMo (`--extra sortformer`). Hard 4-speaker
    # cap — see raven_diar/adapters/sortformer.py.
    "sortformer-4spk-v1": DiarizerSpec(
        model_id="nvidia/diar_sortformer_4spk-v1",
        adapter="sortformer",
        label="sortformer-4spk-v1",
        # HF `main` as of 2026-09-03 (repo has no tags; lastModified 2025-12-15).
        revision="9f17b10df44c0a4c8f3c86fbddc9ee2d6ab9ac08",
    ),
    # Adding the next diarizer (diarizen, a hosted API, …) is a module under
    # raven_diar/adapters/ exposing an ``ADAPTER`` factory plus a spec entry
    # here. The runner resolves ``adapter`` through raven_diar.registry and is
    # not touched — see that module's docstring.
}
