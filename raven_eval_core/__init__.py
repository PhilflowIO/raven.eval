"""raven-eval-core — standalone DER + WER scoring.

The secret-free metric core of the raven.eval public benchmark. Everything here
is a pure ``(reference, hypothesis) -> number`` function built on the standard
libs (``pyannote.metrics`` for DER, ``jiwer`` for WER/CER).
"""

from __future__ import annotations

from .der import (
    DerComponents,
    DiarSegment,
    DiarSegmentText,
    TimedWord,
    assign_word_speakers,
    compute_der,
    compute_der_components,
    compute_der_corpus,
    load_rttm,
    parse_rttm,
    to_rttm,
)
from .flozi_wer import (
    FloziWerResult,
    corpus_cer_pct,
    corpus_wer_filler_tolerant_pct,
    corpus_wer_pct,
    evaluate,
    normalize_flozi,
    strip_fillers,
)
from .wer import (
    WerResult,
    compute_wer,
    normalize_permissive,
    normalize_strict_de,
    normalize_verbatim,
)

__all__ = [
    # flozi-strict ASR-WER (the published-number SSOT)
    "normalize_flozi",
    "strip_fillers",
    "corpus_wer_pct",
    "corpus_cer_pct",
    "corpus_wer_filler_tolerant_pct",
    "evaluate",
    "FloziWerResult",
    # DER
    "compute_der",
    "compute_der_components",
    "compute_der_corpus",
    "DerComponents",
    "parse_rttm",
    "load_rttm",
    "to_rttm",
    "assign_word_speakers",
    "DiarSegment",
    "DiarSegmentText",
    "TimedWord",
    # WER
    "compute_wer",
    "normalize_strict_de",
    "normalize_permissive",
    "normalize_verbatim",
    "WerResult",
]

__version__ = "0.1.0"
