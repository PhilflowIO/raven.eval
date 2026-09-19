"""flozi-strict German ASR WER/CER — the single source of truth for published numbers.

This module is the canonical scoring path behind every WER number Raven
publishes for German ASR. It is imported by **both** sides of the pipeline:

  - the Tier-2 runner (``raven_asr.runner``) that produces ``predictions_*.jsonl``
  - the Tier-1 re-scorer (``scripts/verify.py``) that re-checks them with no GPU

Because both import the *same* functions here, a runner result and its later
re-score are byte-identical by construction — there is no second copy to drift
against. (Etappe 3 shipped a hand-mirrored copy inside ``verify.py``; Etappe 4
collapses that duplication into this one module — the SSOT decision.)

Why this is NOT ``raven_eval_core.normalize_strict_de``
-------------------------------------------------------
``normalize_strict_de`` (in ``raven_eval_core.wer``) is a *different*, more
general diagnostic lens — it lowercases inline, expands digits→words via
``num2words``, and strips fillers unconditionally. The published flozi numbers
use the *opposite* canonical forms and would not reproduce under it. The
material divergences (each moves WER):

  1. Numbers: flozi ``alpha2digit(de)`` (words→digits, "drei"→"3");
     ``normalize_strict_de`` runs ``num2words`` (digits→words). Opposite.
  2. Fillers: flozi-strict does NOT strip fillers (that is the separate
     ``corpus_wer_filler_tolerant_pct`` variant); ``normalize_strict_de`` always does.
  3. Transliteration: flozi runs ``unidecode`` (umlauts escaped/restored);
     ``normalize_strict_de`` does not.
  4. Case: flozi keeps case in ``normalize_flozi`` and defers lowercasing to
     jiwer's ``wer_standardize`` transforms at WER time; ``normalize_strict_de``
     lowercases inline and calls plain jiwer.

Both lenses are legitimate; only *this* one produces the published ASR-WER
table, so it lives in its own named module rather than overloading the general
scorer.

The one deliberate divergence from flozi: alignment is per utterance
---------------------------------------------------------------------
flozi's pipeline scores with jiwer's ``wer_standardize_contiguous``, whose
``ReduceToSingleSentence`` step concatenates the whole subset into ONE sentence
before aligning. That lets an edit at the end of one clip cancel against an edit
at the start of the next — "wir gehen heute" / "wir gehen" followed by "morgen
regnet es" / "heute morgen regnet es" is two errors, and zero once concatenated.
The clips are separate recordings, so that match is an artifact of the scorer,
and it flatters every model on exactly the corpora where clip edges carry
errors: measured 2026-09-19, the concatenated WER read 0.26-0.85 pp low on
VoxPopuli and both Swiss-dialect sets, and identical (≤0.001 pp) on flozi's own
Tuda-De / MLS / Common Voice subsets and on FLEURS.

So every utterance is aligned on its own (``wer_standardize``, the same
transforms minus the sentence reduction) and the corpus WER is
Σedits / Σreference words over those alignments. That also makes the published
number exactly decomposable per utterance, which is what an interval needs
(:func:`utterance_word_errors`). Normalization stays flozi's, verbatim.

flozi-canonical normalization is ported verbatim from the ``normalize_text``
function on the ``flozi00/asr-german-mixed-evals`` HuggingFace dataset card
(commit ``9c34cbcc0e75b841f6abc4d4e452eeb61dcab156``). Pipeline order is
load-bearing — do not reorder. See /NOTICE for attribution and the licensing
note on this ported normalization.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass

from jiwer import cer, process_words, wer, wer_standardize
from text_to_num import alpha2digit
from unidecode import unidecode

_UMLAUT_COUPLES: tuple[tuple[str, str], ...] = (
    ("ä", "ae"),
    ("ö", "oe"),
    ("ü", "ue"),
    ("Ä", "Ae"),
    ("Ö", "Oe"),
    ("Ü", "Ue"),
)

_BRACKET_RE = re.compile(r"\[.*?\]")
_QUOTE_RE = re.compile(r"['\"]")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_flozi(text: str, lang: str = "de") -> str:
    """flozi-canonical German ASR normalization. Pipeline order is load-bearing.

    Case is preserved here; lowercasing happens in jiwer's ``wer_standardize``
    transforms at WER-computation time, matching flozi's upstream pipeline. Umlauts survive ``unidecode`` via an escape/restore
    dance so "Müller" stays "Müller" (not "Muller"), keeping the umlaut signal.
    """
    for umlaut, ascii_form in _UMLAUT_COUPLES:
        text = text.replace(umlaut, f"__{ascii_form}__")
    text = text.replace("ß", "ss")
    text = text.replace(",,", "")
    text = text.replace('"', "")
    text = unidecode(text)
    for umlaut, ascii_form in _UMLAUT_COUPLES:
        text = text.replace(f"__{ascii_form}__", umlaut)
    # text2num raises on edge cases like all-empty input; fall through with the
    # unchanged text. flozi's reference does the same.
    with contextlib.suppress(Exception):
        text = alpha2digit(text, lang)
    text = _BRACKET_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)
    text = _NON_WORD_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text.strip())
    return text.strip()


# Default filler stop-list for the filler-tolerant WER variant. Matched
# case-insensitively against whole tokens after flozi-canonical normalization.
DEFAULT_FILLER_STOPLIST: frozenset[str] = frozenset(
    {
        "äh",
        "ähm",
        "ähhh",
        "hmm",
        "hm",
        "öh",
        "öhm",
        "also",
        "halt",
        "ne",
        "nö",
    }
)


def strip_fillers(
    text: str, *, stoplist: frozenset[str] = DEFAULT_FILLER_STOPLIST
) -> str:
    """Drop filler-tokens from a whitespace-tokenised string (symmetric on ref+hyp)."""
    return " ".join(t for t in text.split() if t.lower() not in stoplist)


@dataclass(frozen=True)
class FloziWerResult:
    """Per-subset flozi-strict evaluation output.

    ``wer_pct`` is the flozi-strict variant (the published number).
    """

    wer_pct: float
    cer_pct: float
    n_samples: int
    wer_filler_tolerant_pct: float = 0.0


def _check_aligned(references: list[str], predictions: list[str]) -> None:
    if len(references) != len(predictions):
        raise ValueError(
            f"refs ({len(references)}) and preds ({len(predictions)}) must align"
        )


def _wer_pct(refs_n: list[str], preds_n: list[str]) -> float:
    if not refs_n:
        return 0.0
    return (
        wer(
            refs_n,
            preds_n,
            reference_transform=wer_standardize,
            hypothesis_transform=wer_standardize,
        )
        * 100
    )


def corpus_wer_pct(references: list[str], predictions: list[str]) -> float:
    """flozi-strict **corpus** WER (%). Normalize each pair, align each utterance
    on its own, then total word edits / total reference words (a corpus
    aggregate, NOT a mean of per-utterance WERs, and NOT one alignment over the
    concatenated subset — see the module docstring)."""
    _check_aligned(references, predictions)
    if not references:
        return 0.0
    refs_n = [normalize_flozi(r) for r in references]
    preds_n = [normalize_flozi(p) for p in predictions]
    return _wer_pct(refs_n, preds_n)


def utterance_word_errors(
    references: list[str], predictions: list[str]
) -> list[tuple[int, int]]:
    """Per-utterance ``(word edits, reference words)`` under the published method.

    The decomposition of :func:`corpus_wer_pct`: the same normalization, the same
    per-utterance jiwer alignment, one pair per utterance, so that
    ``Σedits / Σref_words * 100`` IS the published corpus WER rather than an
    approximation of it. This is what a bootstrap resamples — the utterance is
    the unit the test sets were drawn in.
    """
    _check_aligned(references, predictions)
    if not references:
        return []
    out = process_words(
        [normalize_flozi(r) for r in references],
        [normalize_flozi(p) for p in predictions],
        reference_transform=wer_standardize,
        hypothesis_transform=wer_standardize,
    )
    units: list[tuple[int, int]] = []
    for ref_words, chunks in zip(out.references, out.alignments, strict=True):
        edits = 0
        for c in chunks:
            if c.type == "substitute" or c.type == "delete":
                edits += c.ref_end_idx - c.ref_start_idx
            elif c.type == "insert":
                edits += c.hyp_end_idx - c.hyp_start_idx
        units.append((edits, len(ref_words)))
    return units


def corpus_wer_filler_tolerant_pct(
    references: list[str], predictions: list[str]
) -> float:
    """Filler-tolerant corpus WER (%). flozi-strict + filler stop-list applied
    symmetrically to references and hypotheses."""
    _check_aligned(references, predictions)
    if not references:
        return 0.0
    refs_f = [strip_fillers(normalize_flozi(r)) for r in references]
    preds_f = [strip_fillers(normalize_flozi(p)) for p in predictions]
    return _wer_pct(refs_f, preds_f)


def corpus_cer_pct(references: list[str], predictions: list[str]) -> float:
    """CER (%) on RAW (un-normalized) text via plain jiwer.cer.

    CER is deliberately computed on un-normalized text to keep character-level
    sensitivity (case, punctuation), and without the word-level ``wer_standardize``
    transforms (they reduce text to word tokens and yield a meaningless 100% CER
    on a single-char substitution)."""
    _check_aligned(references, predictions)
    if not references:
        return 0.0
    return cer(references, predictions) * 100


def evaluate(references: list[str], predictions: list[str]) -> FloziWerResult:
    """Compute flozi-strict WER, filler-tolerant WER, and CER on one paired list."""
    return FloziWerResult(
        wer_pct=corpus_wer_pct(references, predictions),
        cer_pct=corpus_cer_pct(references, predictions),
        n_samples=len(references),
        wer_filler_tolerant_pct=corpus_wer_filler_tolerant_pct(
            references, predictions
        ),
    )
