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

#: How per-file scores become one dataset number. ``corpus`` is Σerr/Σtotal over
#: files (CALLHOME/DIHARD, the pyannote model cards); ``file_mean`` is the
#: unweighted mean of per-file DERs (the ETH diarization benchmark). The two are
#: 0.334 pp apart on our German CALLHOME row, so both are computed and both are
#: committed — ``primary`` is only which one the headline table prints.
AGGREGATION_PRIMARY: Final[str] = "corpus"
AGGREGATION_ALSO_REPORTED: Final[tuple[str, ...]] = ("file_mean",)

#: Every published interval is a percentile bootstrap over FILES — the unit the
#: corpora were drawn in. Seeded, because an interval that moves between two runs
#: of the same command is not a published number. Mirrored by
#: ``benchmark.config.yaml`` → ``der.uncertainty`` and asserted equal by
#: ``tests/test_diar_harness.py``.
BOOTSTRAP_RESAMPLES: Final[int] = 10_000
BOOTSTRAP_SEED: Final[int] = 20260903
BOOTSTRAP_CONFIDENCE: Final[float] = 0.95


@dataclass(frozen=True)
class DiarDatasetSpec:
    """A public diarization dataset + how its gold RTTMs are obtained."""

    id: str
    loader: str          # loader module under raven_diar.datasets
    license: str
    # HF slug or upstream repo; pinned by ``revision`` for reproducibility.
    source: str
    revision: str        # git tag/commit or HF revision hash — pin, never floating
    # WHAT LANGUAGE IS SPOKEN in the corpus. Required, and required for a
    # reason: a hosted diarizer has no diarization-only endpoint — it folds
    # turns out of ASR word timings — so an ASR request sent in the wrong
    # language returns no words, and a DER run silently reports ~100 % miss
    # instead of failing. That is what a German-only default did to AMI. The
    # local models (pyannote, Sortformer) are language-agnostic and ignore it;
    # they still receive it, because the runner builds every adapter with the
    # same kwargs and a conditional in the dispatcher is exactly what the
    # registry exists to avoid.
    language: str
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
        language="en",
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
        language="en",
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
        language="de",
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
        language="en",
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

    model_id: str        # HF slug, or `<vendor>/<model>` for a hosted API
    adapter: str         # adapter module under raven_diar.adapters
    label: str           # short id used in result/artifact directory names
    revision: str        # the pin — never floating; strictness per `revision_kind`
    # WHERE it runs. True for a diarizer behind a vendor API rather than from
    # weights we fetch: it selects the dependency extra and means an API key
    # instead of a GPU.
    hosted: bool = False
    # HOW IMMUTABLE the pin is — a different question from `hosted`, and the one
    # a published number actually depends on. "commit" = a 40-hex HF/git commit;
    # the bytes cannot change underneath us. "vendor-alias" = the most specific
    # selector a hosted API offers, which is a MOVING target: the vendor can
    # re-train behind it and nothing in the API says so. Declaring that weakness
    # per diarizer keeps it testable, instead of loosening the rule for everyone
    # — which would let a local spec silently stop pinning a commit. A hosted
    # adapter must additionally assert at run time that the response names the
    # alias it asked for.
    revision_kind: str = "commit"
    # SPDX-ish slug for the WEIGHTS (for a hosted API: the terms the vendor
    # grants). ``DiarDatasetSpec`` has carried a licence from the start; the
    # diarizer side did not, which left the one fact that decides how a row may
    # be published living only in a code comment.
    license: str = "unknown"
    # Whether a row from this diarizer may compete for a winner mark. Stated
    # rather than derived from ``license``, because "may we ship this" is a
    # decision about our product, not a string match: getting it wrong by
    # regex-ing a licence slug is exactly the failure worth avoiding. Under
    # ADR-app-0036 a non-commercial model is measured and shown, never awarded.
    shippable: bool = True


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
        license="MIT (gated: accept the model conditions on HF)",
        shippable=True,
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
        license="CC-BY-NC-4.0",
        # Non-commercial weights. Under ADR-app-0036 this is a reference row:
        # it may be measured and displayed — and it currently beats the shipped
        # default on German telephone speech — but it must never carry a winner
        # mark, because we could not ship the thing that won.
        shippable=False,
    ),
    # NVIDIA Streaming Sortformer 4spk-v2 — the SHIPPABLE Sortformer. Same
    # adapter module as v1 (same NeMo model class, same output shape); what
    # differs is data: a streaming config and a different checkpoint file.
    "sortformer-streaming-4spk-v2": DiarizerSpec(
        model_id="nvidia/diar_streaming_sortformer_4spk-v2",
        adapter="sortformer",
        label="sortformer-streaming-4spk-v2",
        # HF `main` as of 2026-09-03 (repo has no tags; lastModified
        # 2026-08-12). Pinned via hf_hub_download in the adapter — NeMo's
        # from_pretrained takes no revision and would track the branch.
        revision="5240a64075176943f677d30fa2171c780229f341",
        # Verbatim from the model card, section "Licence" (2026-09-03):
        # "License to use this model is covered by the CC-BY-4.0." This is the
        # whole difference from v1: CC-BY-4.0 permits commercial use, so a row
        # from this checkpoint may compete for a winner mark.
        license="CC-BY-4.0",
        shippable=True,
    ),
    # Deepgram, the first HOSTED diarizer: no GPU, no weights — an API key
    # (DEEPGRAM_API_KEY) and per-second billing. Deepgram has no diarization-only
    # endpoint, so `model_id` is the pinned ASR model whose word timings the turns
    # are folded from, and `revision` is the pinned `diarize_model` version (the
    # analogue of an HF revision hash) — never "latest".
    "deepgram-nova-3": DiarizerSpec(
        model_id="nova-3-general",
        adapter="deepgram",
        label="deepgram-nova-3",
        revision="v2",
        hosted=True,
        revision_kind="vendor-alias",
        # A hosted API grants terms of service, not a weights licence. Commercial
        # use is what the paid tier is for, so a Deepgram row may compete.
        license="Deepgram commercial terms of service (paid API)",
        shippable=True,
    ),
    # AssemblyAI Universal-3.5 Pro. HOSTED: needs ASSEMBLYAI_API_KEY, no GPU.
    # AssemblyAI has NO diarization-only endpoint — diarization is the
    # `speaker_labels` flag on a transcription request, so a DER run also pays
    # for a transcript (0.21 $/h model + 0.02 $/h diarization add-on, verified on
    # assemblyai.com/pricing 2026-09-03). `revision` IS the model alias: the
    # vendor publishes no immutable version, and its DEFAULT is a two-model
    # fallback chain, so the adapter sends this alias alone and asserts the
    # response's `speech_model_used` matches it.
    "assemblyai-universal-3-5-pro": DiarizerSpec(
        model_id="assemblyai/universal-3-5-pro",
        adapter="assemblyai",
        label="assemblyai-universal-3-5-pro",
        revision="universal-3-5-pro",
        revision_kind="vendor-alias",
        hosted=True,
        # A hosted API grants terms of service, not a weights licence.
        # Commercial use is what the paid tier is for, so this row may compete.
        license="AssemblyAI commercial terms of service (paid API)",
        shippable=True,
    ),
    # Adding the next diarizer (diarizen, a hosted API, …) is a module under
    # raven_diar/adapters/ exposing an ``ADAPTER`` factory plus a spec entry
    # here. The runner resolves ``adapter`` through raven_diar.registry and is
    # not touched — see that module's docstring.
}
