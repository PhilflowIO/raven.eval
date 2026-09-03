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
class ModelSpec:
    """Identifies a model + its adapter binding for the runner."""

    model_id: str
    adapter: str  # adapter module name under raven_asr.adapters
    label: str  # short identifier used in result directory names
    # Optional per-model endpoint env-var *names* (never values). When None the
    # adapter's own defaults apply (legacy behaviour). Only the env-var names
    # live in this public repo; the URLs/keys stay in the operator's env.
    base_url_env: str | None = None
    api_key_env: str | None = None


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
    # Wave-2 re-bench models (flow.raven#5137). Served on self-hosted
    # OpenAI-compatible endpoints; per-model endpoint env *names* below.
    # Of these, only nvidia/parakeet-tdt-0.6b-v3 has a flozi-published anchor
    # in FLOZI_REFERENCE_WER — the others deliberately carry no anchor (no
    # published number on this dataset to sanity-check against).
    "nvidia/parakeet-tdt-0.6b-v3": ModelSpec(
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        adapter="vllm_openai",
        label="nvidia-parakeet-tdt-0.6b-v3",
        base_url_env="NEMO_BENCH_URL",
        api_key_env="NEMO_BENCH_API_KEY",
    ),
    "nvidia/canary-1b-v2": ModelSpec(
        model_id="nvidia/canary-1b-v2",
        adapter="vllm_openai",
        label="nvidia-canary-1b-v2",
        base_url_env="NEMO_BENCH_URL",
        api_key_env="NEMO_BENCH_API_KEY",
    ),
    "ibm-granite/granite-speech-4.1-2b-plus": ModelSpec(
        model_id="ibm-granite/granite-speech-4.1-2b-plus",
        adapter="vllm_openai",
        label="granite-speech-4.1-2b-plus",
        base_url_env="VLLM_BENCH_URL",
        api_key_env="VLLM_BENCH_API_KEY",
    ),
    "CohereLabs/cohere-transcribe-03-2026": ModelSpec(
        model_id="CohereLabs/cohere-transcribe-03-2026",
        adapter="vllm_openai",
        label="cohere-transcribe-03-2026",
        base_url_env="VLLM_BENCH_URL",
        api_key_env="VLLM_BENCH_API_KEY",
    ),
    "OpenMOSS-Team/MOSS-Transcribe-Diarize": ModelSpec(
        model_id="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        adapter="vllm_openai",
        label="moss-transcribe-diarize",
        base_url_env="VLLM_BENCH_URL",
        api_key_env="VLLM_BENCH_API_KEY",
    ),
    "microsoft/Phi-4-multimodal-instruct": ModelSpec(
        model_id="microsoft/Phi-4-multimodal-instruct",
        adapter="vllm_openai",
        label="phi-4-multimodal",
        base_url_env="VLLM_BENCH_URL",
        api_key_env="VLLM_BENCH_API_KEY",
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
