"""raven-eval-core — standalone DER + WER + BLEU scoring.

The secret-free metric core of the raven.eval public benchmark. Everything here
is a pure ``(reference, hypothesis) -> number`` function built on the standard
libs (``pyannote.metrics`` for DER, ``jiwer`` for WER/CER, ``sacrebleu`` for
BLEU).

``SCORED_METRICS`` names the metrics this core implements. It is the
implementation side of the scoring contract: ``benchmark.config.yaml`` declares
exactly these metric blocks, and ``tests/test_metric_contract.py`` fails the
build in both directions if the two ever disagree.
"""

from __future__ import annotations

from .bleu import (
    BLEU_EFFECTIVE_ORDER,
    BLEU_LOWERCASE,
    BLEU_MAX_NGRAM_ORDER,
    BLEU_SMOOTH_METHOD,
    BLEU_TOKENIZE,
    BleuResult,
    bleu_signature,
    corpus_bleu_score,
    evaluate_bleu,
    sentence_bleu_diagnostic,
)
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

#: Metrics implemented by this core. Mirrored by the top-level metric blocks in
#: benchmark.config.yaml; the pairing is enforced by tests/test_metric_contract.py.
SCORED_METRICS: frozenset[str] = frozenset({"der", "wer", "bleu"})

__all__ = [
    "SCORED_METRICS",
    # BLEU (translation-shaped corpora, e.g. Swiss-German dialect -> standard German)
    "corpus_bleu_score",
    "evaluate_bleu",
    "sentence_bleu_diagnostic",
    "bleu_signature",
    "BleuResult",
    "BLEU_TOKENIZE",
    "BLEU_LOWERCASE",
    "BLEU_SMOOTH_METHOD",
    "BLEU_EFFECTIVE_ORDER",
    "BLEU_MAX_NGRAM_ORDER",
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
