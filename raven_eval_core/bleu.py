"""BLEU — the translation-shaped metric, for corpora where WER alone mismeasures.

Why this module exists
----------------------
WER assumes reference and hypothesis are the *same* utterance in the *same*
language, so every token difference is an error. Some of the corpora raven.eval
scores are not shaped like that. Swiss-German dialect speech is *spoken* in
dialect and *transcribed* in standard German: the reference is a translation of
what was said, not a transcript of it. Legitimate lexical and word-order choices
("gäll" → "nicht wahr", "lueg" → "schau") are counted as substitutions by WER
even when the output is perfect. BLEU, an n-gram precision metric with a brevity
penalty, is the field-standard answer to exactly that shape.

BLEU here is **corpus-level** — the published number is always the single score
over the whole subset (Σ n-gram matches / Σ n-gram candidates), never a mean of
per-sentence scores. :func:`sentence_bleu_diagnostic` exists for inspecting
individual rows and is explicitly *not* a publishable number; see its docstring.

Which BLEU (the decision, written down)
---------------------------------------
"BLEU" alone is not a number. Tokenizer, case handling, smoothing, max n-gram
order and effective-order all move it by whole points, which is why cross-paper
BLEU comparisons were unreliable for years (Post 2018, "A Call for Clarity in
Reporting BLEU Scores", WMT). We therefore score with **sacrebleu**, the
implementation that fixed that problem by pinning tokenization and emitting a
signature string that names every convention it used.

The pinned conventions (mirrored in ``benchmark.config.yaml`` → ``bleu.variants``,
and asserted equal by ``tests/test_metric_contract.py``):

  * ``tokenize="13a"`` — the mteval-v13a tokenizer, the WMT default. Suitable for
    German/Swiss-German: it is punctuation-splitting and script-agnostic, and it
    does *not* do the aggressive Unicode class-folding of ``intl``.
  * ``lowercase=False`` — case-sensitive (signature ``case:mixed``). German nouns
    are capitalized; folding case would hide a real class of output errors.
  * ``smooth_method="exp"`` — sacrebleu's default exponential smoothing. Matters
    only when a higher-order n-gram count is zero, which corpus-level scoring on a
    real subset practically never hits, but leaving it to a library default that
    could shift between releases is exactly what this module refuses to do.
  * ``effective_order=False`` — always all 4 orders at corpus level. (sacrebleu
    forces effective order on for *sentence*-level scoring; that is one more
    reason the sentence value is a diagnostic, not a published number.)

Rejected alternative: ``nltk.translate.bleu_score``. It has no pinned tokenizer —
it scores whatever token lists the caller hands it, which puts the single most
score-moving convention outside the published contract. sacrebleu also carries
its own version inside the signature, so a number stays attributable to the exact
implementation that produced it.

The library version is not pinned by hand in a comment; :func:`bleu_signature`
reads it out of the installed sacrebleu at score time, and the harness records it
next to the number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sacrebleu.metrics import BLEU

# --- the pinned scoring conventions -------------------------------------------
# These five constants ARE the contract. benchmark.config.yaml restates them for
# third-party readers and tests/test_metric_contract.py fails the build if the two
# ever disagree, in either direction.
BLEU_TOKENIZE = "13a"
BLEU_LOWERCASE = False
BLEU_SMOOTH_METHOD = "exp"
BLEU_EFFECTIVE_ORDER = False
BLEU_MAX_NGRAM_ORDER = 4

__all__ = [
    "BLEU_EFFECTIVE_ORDER",
    "BLEU_LOWERCASE",
    "BLEU_MAX_NGRAM_ORDER",
    "BLEU_SMOOTH_METHOD",
    "BLEU_TOKENIZE",
    "BleuResult",
    "bleu_signature",
    "corpus_bleu_score",
    "evaluate_bleu",
    "sentence_bleu_diagnostic",
]


@dataclass(frozen=True)
class BleuResult:
    """Corpus BLEU plus the components that explain it.

    ``bleu`` is on sacrebleu's 0–100 scale (higher is better), the same scale the
    machine-translation literature reports. ``precisions`` are the per-order
    n-gram precisions (1..4, percent). ``signature`` records the exact conventions
    and sacrebleu version the score was produced under.
    """

    bleu: float
    precisions: tuple[float, ...]
    brevity_penalty: float
    length_ratio: float
    hyp_len: int
    ref_len: int
    n_samples: int
    signature: str


def _scorer(*, effective_order: bool = BLEU_EFFECTIVE_ORDER) -> BLEU:
    return BLEU(
        tokenize=BLEU_TOKENIZE,
        lowercase=BLEU_LOWERCASE,
        smooth_method=BLEU_SMOOTH_METHOD,
        max_ngram_order=BLEU_MAX_NGRAM_ORDER,
        effective_order=effective_order,
    )


def bleu_signature(n_refs: int = 1) -> str:
    """The sacrebleu signature for the pinned conventions, e.g.
    ``nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0``.

    Publish this string next to any BLEU number. It is what makes the number
    comparable to somebody else's, and it carries the sacrebleu version, so a
    future library change is visible rather than silent.

    ``n_refs`` only labels the signature (sacrebleu refuses to emit one before it
    knows how many references it scored against); it does not affect any score.
    """
    # sacrebleu refuses to emit a signature until it knows the reference count, so
    # seed a throwaway scorer with n_refs empty reference streams. This object is
    # never used to score anything.
    return str(BLEU(
        tokenize=BLEU_TOKENIZE,
        lowercase=BLEU_LOWERCASE,
        smooth_method=BLEU_SMOOTH_METHOD,
        max_ngram_order=BLEU_MAX_NGRAM_ORDER,
        effective_order=BLEU_EFFECTIVE_ORDER,
        references=[[""]] * n_refs,
    ).get_signature())


def _as_reference_streams(
    references: Sequence[str] | Sequence[Sequence[str]],
) -> list[list[str]]:
    """Normalize our (per-utterance) reference shape into sacrebleu's (per-stream).

    We accept the shape the rest of this repo uses — one reference per utterance
    (``["ref1", "ref2", ...]``) or several per utterance
    (``[["a1","a2"], ["b1","b2"], ...]``) — and transpose it into sacrebleu's
    ``corpus_score`` shape, which is a list of *reference streams* (stream *k*
    holds the k-th reference of every utterance).

    Multi-reference rows must all carry the same number of references; sacrebleu
    has no notion of a ragged reference set, and silently padding one would
    quietly change the number.
    """
    if not references:
        return [[]]
    if isinstance(references[0], str):
        return [[str(r) for r in references]]

    rows = [list(r) for r in references]  # type: ignore[arg-type]
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(
            f"ragged multi-reference input: rows carry {sorted(widths)} references. "
            f"BLEU needs the same number of references for every utterance."
        )
    n_refs = widths.pop()
    return [[str(row[k]) for row in rows] for k in range(n_refs)]


def _check_aligned(
    references: Sequence[str] | Sequence[Sequence[str]],
    predictions: Sequence[str],
) -> None:
    if len(references) != len(predictions):
        raise ValueError(
            f"refs ({len(references)}) and preds ({len(predictions)}) must align"
        )


def corpus_bleu_score(
    references: Sequence[str] | Sequence[Sequence[str]],
    predictions: Sequence[str],
) -> float:
    """Corpus BLEU (0–100) under the pinned conventions. The published number.

    ``references`` is one reference per utterance, or a list of alternatives per
    utterance. Scoring is corpus-level: n-gram statistics are summed over the whole
    list before the ratio is taken — this is NOT the mean of per-sentence BLEUs and
    the two differ materially on short utterances.
    """
    _check_aligned(references, predictions)
    if not references:
        return 0.0
    streams = _as_reference_streams(references)
    return float(_scorer().corpus_score(list(predictions), streams).score)


def evaluate_bleu(
    references: Sequence[str] | Sequence[Sequence[str]],
    predictions: Sequence[str],
) -> BleuResult:
    """Corpus BLEU with its components and the signature it was computed under."""
    _check_aligned(references, predictions)
    if not references:
        return BleuResult(
            bleu=0.0,
            precisions=(0.0, 0.0, 0.0, 0.0),
            brevity_penalty=0.0,
            length_ratio=0.0,
            hyp_len=0,
            ref_len=0,
            n_samples=0,
            signature=bleu_signature(),
        )
    scorer = _scorer()
    score = scorer.corpus_score(list(predictions), _as_reference_streams(references))
    return BleuResult(
        bleu=float(score.score),
        precisions=tuple(float(p) for p in score.precisions),
        brevity_penalty=float(score.bp),
        length_ratio=float(score.sys_len / score.ref_len) if score.ref_len else 0.0,
        hyp_len=int(score.sys_len),
        ref_len=int(score.ref_len),
        n_samples=len(predictions),
        signature=str(scorer.get_signature()),
    )


def sentence_bleu_diagnostic(
    reference: str | Sequence[str],
    prediction: str,
) -> float:
    """Per-utterance BLEU — a DIAGNOSTIC, never a published number.

    Sentence-level BLEU is not the corpus number restricted to one row: sacrebleu
    turns ``effective_order`` on for single sentences (short segments would
    otherwise score 0 whenever a 4-gram never matches), so this value is computed
    under a *different* convention than :func:`corpus_bleu_score`. Use it to find
    the worst rows in a run; never average it, never publish it, never compare it
    to somebody else's corpus BLEU.
    """
    refs = [reference] if isinstance(reference, str) else list(reference)
    scorer = _scorer(effective_order=True)
    return float(scorer.sentence_score(prediction, refs).score)
