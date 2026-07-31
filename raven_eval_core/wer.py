"""WER / CER / IER computation with three German normalization variants.

Pure ``(reference, hypothesis) -> number`` scoring — no fixtures, no DB, no
schema coupling. Three normalization lenses:

  - strict-de: lowercase, ß→ss, num→word (``num2words(de)``), Umlaute kept,
    punctuation stripped, fillers (äh/ähm/mhm) stripped, whitespace collapsed
  - permissive: lowercase, only whitespace + punctuation stripped (no num
    expansion, no filler strip)
  - verbatim: only lowercase + whitespace collapse, punctuation kept

All three are computed per (reference, hypothesis) pair so consumers can pick
their lens without re-running the scorer.

Speaker-label stripping
-----------------------
Transcripts sometimes carry bracketed speaker tags (``[sprecher0]``, ``[S1]``).
These are stripped before scoring. In the original Raven source this regex
carried literal person names (``phil|chris|helena``) — PRIVATE data that must
NOT ship in a public package. That has been generalized: stripping is driven by
(a) generic patterns (``[sprecher…]`` / ``[speaker N]`` / ``[SPEAKER_N]`` /
short bracketed initials) plus (b) a caller-supplied ``speaker_labels`` list
(default empty). No human name is hardcoded — a caller who wants a specific
label scrubbed passes it explicitly. A spoken name (e.g. the word "Helena" in
the audio) is therefore never stripped by default, which is the correct
behavior: it is content, not a diarization tag.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

import jiwer
from num2words import num2words

# Generic (non-name) bracketed speaker-tag patterns. NO hardcoded human names —
# see module docstring. ``[a-z]{1,3}`` covers short bracketed initials/aliases
# like ``[s1]`` / ``[ab]``; it is length-bounded so it cannot match a real name.
_GENERIC_LABEL_PATTERNS: tuple[str, ...] = (
    r"sprecher[^\]]*",
    r"speaker[\s_]*\d+",
    r"[a-z]{1,3}",
)

_FILLERS = re.compile(
    r"\b(mhm+|hm+|äh+|ähm+|öh+|ähem|em|ää+|ehm|hmm+)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def _build_speaker_label_re(speaker_labels: Iterable[str] | None) -> re.Pattern[str]:
    """Compile the bracketed speaker-label regex.

    Combines the generic (name-free) patterns with any caller-supplied literal
    labels. Only bracketed occurrences (``[label]``) are matched — a bare word
    in the transcript body is never treated as a speaker tag.
    """
    parts = list(_GENERIC_LABEL_PATTERNS)
    for lab in speaker_labels or ():
        lab = lab.strip()
        if lab:
            parts.append(re.escape(lab))
    return re.compile(r"\[(" + "|".join(parts) + r")\]", re.IGNORECASE)


#: Default label regex (generic patterns only, no caller labels, no names).
_DEFAULT_SPEAKER_LABEL = _build_speaker_label_re(None)


def _strip_speaker_labels(
    text: str, speaker_labels: Iterable[str] | None = None
) -> str:
    label_re = (
        _DEFAULT_SPEAKER_LABEL
        if not speaker_labels
        else _build_speaker_label_re(speaker_labels)
    )
    return label_re.sub(" ", text)


def _expand_numbers_de(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0).replace(",", ".")
        try:
            if "." in raw:
                return num2words(float(raw), lang="de")
            return num2words(int(raw), lang="de")
        except (ValueError, NotImplementedError):
            return match.group(0)

    return _NUMBER.sub(_replace, text)


def normalize_strict_de(
    text: str, speaker_labels: Iterable[str] | None = None
) -> str:
    """Industry-standard German normalization for ranking-grade WER."""
    text = _strip_speaker_labels(text, speaker_labels)
    text = text.lower()
    text = text.replace("ß", "ss")
    text = _expand_numbers_de(text)
    text = _FILLERS.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def normalize_permissive(
    text: str, speaker_labels: Iterable[str] | None = None
) -> str:
    """Light normalization for diagnostic WER (no number/filler munging)."""
    text = _strip_speaker_labels(text, speaker_labels)
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def normalize_verbatim(
    text: str, speaker_labels: Iterable[str] | None = None
) -> str:
    """Verbatim WER — keeps punctuation, only lowercases + collapses spaces."""
    text = _strip_speaker_labels(text, speaker_labels)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


@dataclass(frozen=True)
class WerResult:
    wer_strict: float
    wer_permissive: float
    wer_verbatim: float
    cer: float
    sub: int
    ins: int
    delete: int
    ref_word_count: int
    ier: float  # ins / ref_word_count — hallucination proxy on speech audio


def _safe_wer(ref: str, hyp: str) -> float:
    if not ref or not hyp:
        return 1.0 if ref or hyp else 0.0
    return jiwer.wer(ref, hyp)


def _safe_cer(ref: str, hyp: str) -> float:
    if not ref or not hyp:
        return 1.0 if ref or hyp else 0.0
    return jiwer.cer(ref, hyp)


def compute_wer(
    reference: str,
    hypothesis: str,
    speaker_labels: Iterable[str] | None = None,
) -> WerResult:
    """Compute the full WER metric stack for one (ref, hyp) pair.

    ``speaker_labels`` (optional) is a list of literal bracketed speaker tags to
    strip in addition to the generic patterns — e.g. ``["Helena"]`` to scrub
    ``[Helena]`` diarization tags. Empty/None → generic patterns only (no name
    is ever hardcoded; see module docstring).
    """
    ref_strict = normalize_strict_de(reference, speaker_labels)
    hyp_strict = normalize_strict_de(hypothesis, speaker_labels)
    ref_perm = normalize_permissive(reference, speaker_labels)
    hyp_perm = normalize_permissive(hypothesis, speaker_labels)
    ref_verb = normalize_verbatim(reference, speaker_labels)
    hyp_verb = normalize_verbatim(hypothesis, speaker_labels)

    ref_words = ref_strict.split()
    ref_word_count = len(ref_words)

    if ref_strict and hyp_strict:
        details = jiwer.process_words(ref_strict, hyp_strict)
        sub = details.substitutions
        ins = details.insertions
        delete = details.deletions
    else:
        sub = ins = delete = 0

    ier = (ins / ref_word_count) if ref_word_count else 0.0

    return WerResult(
        wer_strict=_safe_wer(ref_strict, hyp_strict),
        wer_permissive=_safe_wer(ref_perm, hyp_perm),
        wer_verbatim=_safe_wer(ref_verb, hyp_verb),
        cer=_safe_cer(ref_strict, hyp_strict),
        sub=sub,
        ins=ins,
        delete=delete,
        ref_word_count=ref_word_count,
        ier=ier,
    )
