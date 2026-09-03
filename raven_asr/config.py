"""Registry of supported models, adapters, and the flozi-published reference numbers.

The reference numbers below come from `flozi00/asr-german-mixed-evals` datacard
(snapshot 2026-05-18). They serve as sanity-check anchors — a run that drifts
>15% from these on the same model+subset indicates a methodological bug, not a
new finding. See README.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Per-subset reference WER (%), from flozi-published table on
# https://huggingface.co/datasets/flozi00/asr-german-mixed-evals
# Subset keys match the `from` column values in the dataset.
FLOZI_REFERENCE_WER: Final[dict[str, dict[str, float]]] = {
    "primeline/whisper-large-v3-turbo-german": {
        "All": 2.62, "Tuda-De": 6.37, "multilingual_librispeech": 2.06, "common_voice_19_0": 3.22,
    },
    "nyrahealth/CrisperWhisper": {
        "All": 2.68, "Tuda-De": 5.17, "multilingual_librispeech": 2.85, "common_voice_19_0": 1.90,
    },
    "primeline/whisper-large-v3-german": {
        "All": 2.76, "Tuda-De": 7.78, "multilingual_librispeech": 2.12, "common_voice_19_0": 3.31,
    },
    "primeline/parakeet-primeline": {
        "All": 2.95, "Tuda-De": 4.11, "multilingual_librispeech": 2.60, "common_voice_19_0": 3.03,
    },
    "openai/whisper-large-v3": {
        "All": 3.28, "Tuda-De": 7.86, "multilingual_librispeech": 2.85, "common_voice_19_0": 3.46,
    },
    "nvidia/parakeet-tdt-0.6b-v3": {
        "All": 3.64, "Tuda-De": 7.05, "multilingual_librispeech": 2.95, "common_voice_19_0": 3.70,
    },
}

# Subset identifiers (match `from` column in flozi00/asr-german-mixed-evals).
# Per-subset licenses (committing the reference transcripts):
#   Tuda-De                  CC-BY-4.0  — ok to commit *with attribution* (see NOTICE)
#   common_voice_19_0        CC0        — ok to commit
#   multilingual_librispeech CC-BY-4.0  — ok to commit *with attribution* (see NOTICE)
FLOZI_SUBSETS: Final[tuple[str, ...]] = (
    "Tuda-De",
    "multilingual_librispeech",
    "common_voice_19_0",
)

# Public repo: only the flozi public subsets. Raven's private meeting corpus is
# Tier-3 (not portable) and lives only in the internal harness.
SUBSETS: Final[tuple[str, ...]] = FLOZI_SUBSETS

# Pin the dataset revision for byte-reproducible references. ``None`` = the HF
# ``main`` HEAD at run time (documented caveat — set this to the commit hash of
# your local snapshot for a fully reproducible Tier-2 run). Threaded through to
# the loader + recorded in summary.json / model-index.yaml.
FLOZI_DATASET_REVISION: Final[str | None] = None

DRIFT_TOLERANCE_PCT: Final[float] = 15.0


@dataclass(frozen=True)
class WerDatasetSpec:
    """A public WER dataset + how its audio and references are obtained.

    Mirrors ``raven_diar.config.DiarDatasetSpec``. ``id`` is the key published in
    ``benchmark.config.yaml`` under ``datasets.wer[].id``; the contract test
    asserts the two lists match in both directions.
    """

    id: str
    loader: str              # loader module under raven_asr.datasets
    license: str
    source: str              # HF slug or upstream URL — the acquisition path
    revision: str | None     # HF revision hash — pin, never floating
    subsets: tuple[str, ...]  # subset selectors the harness accepts for this id
    # Durability of the acquisition path, per ADR-app-0054: "doi" (DOI or
    # institutional archive) > "hf" (versioned HF dataset at a pinned revision) >
    # "vendor" (vendor-hosted link — flagged, never silently relied on).
    durability: str
    # These corpora are hundreds of GB; a --limit run must not pay for a full
    # download first, so the loader streams unless the caller says otherwise.
    stream_by_default: bool = False
    # sha256 of the fetched archive, where the source IS a loose archive rather
    # than a content-addressed HF revision. None for every HF-backed entry.
    sha256: str | None = None
    notes: str = ""


# The PUBLIC datasets the WER Tier-2 harness scores against. Audio is fetched by
# the harness from the source below; we never redistribute restricted audio, only
# per-utterance predictions/references land under artifacts/. See /NOTICE.
WER_DATASETS: Final[dict[str, WerDatasetSpec]] = {
    # The flozi-aligned reference corpus — three subsets in one repo, selected by
    # the `from` column. This is the dataset every published WER row in
    # BENCHMARKS.md was measured on.
    "german-mixed": WerDatasetSpec(
        id="german-mixed",
        loader="flozi_mixed_evals",
        license="Tuda-De CC-BY-4.0; MLS CC-BY-4.0; Common Voice CC0-1.0",
        source="flozi00/asr-german-mixed-evals",
        # Deliberately unpinned: the committed 2026-07-30 numbers were measured
        # against `main` HEAD, so naming a hash here would claim a pin the
        # published artifacts do not actually carry. Pass DATASET_REV=<sha> for a
        # fully pinned re-run. Tracked as the one floating pin in this table.
        revision=FLOZI_DATASET_REVISION,
        subsets=FLOZI_SUBSETS,
        durability="hf",
        notes="single train split; subsets selected by the `from` column.",
    ),
    # FLEURS — read parallel sentences, the speech side of FLoRes.
    "fleurs": WerDatasetSpec(
        id="fleurs",
        loader="fleurs_de",
        license="CC-BY-4.0",
        source="google/fleurs",
        revision="70bb2e84b976b7e960aa89f1c648e09c59f894dd",
        subsets=("fleurs",),
        durability="hf",
        stream_by_default=True,
        notes="config=de_de, split=test (~350 sentences); raw_transcription.",
    ),
    # MLS German — read audiobook speech (LibriVox), the register whisper-class
    # models are strongest on; the low-WER end of the German spread.
    "mls-de": WerDatasetSpec(
        id="mls-de",
        loader="mls_german",
        license="CC-BY-4.0",
        source="facebook/multilingual_librispeech",
        revision="2e83e61823b4c47dcbcb1980bb88601274127609",
        subsets=("mls-de",),
        durability="hf",
        stream_by_default=True,
        notes="config=german (spelled out, not 'de'), split=test; `transcript`.",
    ),
    # VoxPopuli German — European-Parliament plenary speech. Spontaneous and
    # accented, so it anchors the hard end that read-speech corpora cannot.
    "voxpopuli-de": WerDatasetSpec(
        id="voxpopuli-de",
        loader="voxpopuli_de",
        license="CC0-1.0",
        source="facebook/voxpopuli",
        revision="42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
        subsets=("voxpopuli-de",),
        durability="hf",
        stream_by_default=True,
        notes="config=de, split=test; raw_text with normalized_text fallback.",
    ),
}


def resolve_wer_dataset(selector: str) -> tuple[str, str]:
    """Map a ``--dataset`` argument onto ``(dataset_id, loader subset)``.

    Accepts either a dataset id from :data:`WER_DATASETS` or one of the
    ``german-mixed`` subset names (``Tuda-De``, …), which the CLI has taken since
    Etappe 4 and which must keep working. A bare dataset id selects that
    dataset's whole eval set (``"All"`` for the multi-subset corpus).
    """
    if selector in FLOZI_SUBSETS:
        return "german-mixed", selector
    spec = WER_DATASETS.get(selector)
    if spec is None:
        known = sorted({*WER_DATASETS, *FLOZI_SUBSETS})
        raise KeyError(f"unknown WER dataset {selector!r}; known: {', '.join(known)}")
    if len(spec.subsets) == 1:
        return spec.id, spec.subsets[0]
    return spec.id, "All"


@dataclass(frozen=True)
class ModelSpec:
    """Identifies a model + its adapter binding for the runner."""

    model_id: str
    adapter: str  # adapter module name under raven_asr.adapters
    label: str  # short identifier used in result directory names


KNOWN_MODELS: Final[dict[str, ModelSpec]] = {
    "primeline/whisper-large-v3-german": ModelSpec(
        model_id="primeline/whisper-large-v3-german",
        adapter="vllm_openai",
        label="primeline-whisper-large-v3-german",
    ),
    "primeline/whisper-large-v3-turbo-german": ModelSpec(
        model_id="primeline/whisper-large-v3-turbo-german",
        adapter="vllm_openai",
        label="primeline-whisper-large-v3-turbo-german",
    ),
    "primeline/parakeet-primeline": ModelSpec(
        model_id="primeline/parakeet-primeline",
        adapter="vllm_openai",
        label="primeline-parakeet",
    ),
    "voxtral-mini-latest": ModelSpec(
        model_id="voxtral-mini-latest",
        adapter="voxtral_mistral",
        label="voxtral-mini",
    ),
    "deepgram-nova-2": ModelSpec(
        model_id="nova-2",
        adapter="deepgram",
        label="deepgram-nova-2",
    ),
    "openai-whisper-1": ModelSpec(
        model_id="whisper-1",
        adapter="openai_whisper",
        label="openai-whisper-1",
    ),
    # Modal-hosted STT apps. Each must expose a parameterized
    # `transcribe(audio_bytes, sr)` function.
    "modal/parakeet": ModelSpec(
        model_id="primeline/parakeet-primeline",
        adapter="modal_app",
        label="modal-parakeet",
    ),
    "modal/qwen3-asr": ModelSpec(
        model_id="Qwen/Qwen3-ASR-1.7B",
        adapter="modal_app",
        label="modal-qwen3-asr",
    ),
    "modal/voxtral-2507": ModelSpec(
        model_id="mistralai/Voxtral-Mini-3B-2507",
        adapter="modal_app",
        label="modal-voxtral-2507",
    ),
    "modal/voxtral-2602": ModelSpec(
        model_id="mistralai/Voxtral-Mini-3B-2602",
        adapter="modal_app",
        label="modal-voxtral-2602",
    ),
}

# Modal App name lookup keyed by ModelSpec.label.
# Mirrors the `modal.App("…")` names in the internal STT-bench Modal deployment.
MODAL_APP_NAMES: Final[dict[str, str]] = {
    "modal-parakeet": "raven-stt-parakeet",
    "modal-qwen3-asr": "raven-stt-qwen3-asr",
    "modal-voxtral-2507": "raven-stt-voxtral",
    "modal-voxtral-2602": "raven-stt-voxtral-2602",
}
