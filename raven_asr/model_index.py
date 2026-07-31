"""HuggingFace ``model-index`` YAML emitter.

Schema follows the style of ``flozi00/whisper-large-german-lora-cv13``
and similar flozi model cards so our YAML can be pasted into a model
card frontmatter without further massage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ResultEntry:
    """One (subset, metric) row inside a model-index entry.

    ``wer_pct`` is the flozi-strict variant; ``wer_filler_tolerant_pct``
    is the filler-tolerant variant reported alongside it.
    """

    dataset_id: str  # HF dataset path, e.g. "flozi00/asr-german-mixed-evals"
    dataset_name: str  # human-readable subset label, e.g. "Tuda-De"
    subset_arg: str  # value of the `from` column / subset selector
    wer_pct: float
    cer_pct: float
    n_samples: int
    wer_filler_tolerant_pct: float = 0.0


def build_model_index(
    *,
    model_id: str,
    results: list[ResultEntry],
    name: str | None = None,
) -> dict[str, object]:
    """Assemble the ``model-index`` dict matching flozi-style model cards."""
    entries: list[dict[str, object]] = []
    for r in results:
        entry: dict[str, object] = {
            "task": {
                "type": "automatic-speech-recognition",
                "name": "Automatic Speech Recognition",
            },
            "dataset": {
                "type": r.dataset_id,
                "name": r.dataset_name,
                "args": r.subset_arg,
            },
            "metrics": [
                {
                    "type": "wer",
                    "value": round(r.wer_pct, 4),
                    "name": "Test WER",
                    "verified": False,
                },
                {
                    "type": "cer",
                    "value": round(r.cer_pct, 4),
                    "name": "Test CER",
                    "verified": False,
                },
                {
                    "type": "wer",
                    "value": round(r.wer_filler_tolerant_pct, 4),
                    "name": "Test WER (filler-tolerant)",
                    "verified": False,
                },
            ],
        }
        entries.append(entry)
    return {
        "model-index": [
            {
                "name": name or model_id,
                "results": entries,
            }
        ],
    }


def write_yaml(path: Path, doc: dict[str, object]) -> None:
    """Write the model-index dict as YAML with stable ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
