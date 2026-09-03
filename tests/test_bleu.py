"""Tests for raven_eval_core.bleu — the translation-shaped metric.

The first test is the important one: it anchors our implementation to a BLEU
value published *outside* this repo, so "our BLEU" cannot quietly become a
self-consistent number that agrees only with itself.
"""

from __future__ import annotations

import pytest

from raven_eval_core.bleu import (
    BLEU_LOWERCASE,
    BLEU_MAX_NGRAM_ORDER,
    BLEU_SMOOTH_METHOD,
    BLEU_TOKENIZE,
    bleu_signature,
    corpus_bleu_score,
    evaluate_bleu,
    sentence_bleu_diagnostic,
)

# ── the external anchor ──────────────────────────────────────────────────────
#
# Hypothesis + three references from Papineni et al. 2002 ("BLEU: a Method for
# Automatic Evaluation of Machine Translation", ACL) — the canonical worked
# example of the metric. NLTK ships this exact quadruple in the documented
# example of ``nltk.translate.bleu_score.sentence_bleu`` and publishes the
# expected result as ``0.5045666840058485`` (unsmoothed, 4-gram, uniform
# weights). On our 0-100 scale that is 50.456668400584846.
#
# Reproducing it pins two things at once: our n-gram/brevity-penalty arithmetic,
# and the fact that the 13a tokenizer agrees with NLTK's whitespace tokens on
# plain ASCII words. Scored with smoothing OFF because the published value is the
# unsmoothed BLEU — that is the only convention deviation, and it is what makes
# the comparison legitimate rather than approximate.
PAPINENI_HYP = (
    "It is a guide to action which ensures that the military "
    "always obeys the commands of the party"
)
PAPINENI_REFS = [
    ("It is a guide to action that ensures that the military will forever "
     "heed Party commands"),
    ("It is the guiding principle which guarantees the military forces always "
     "being under the command of the Party"),
    ("It is the practical guide for the army always to heed the directions of "
     "the party"),
]
PAPINENI_BLEU = 50.456668400584846  # = NLTK's published 0.5045666840058485 * 100


def test_matches_the_published_papineni_nltk_value() -> None:
    """Our BLEU reproduces an externally published value to 10 decimal places."""
    sacrebleu = pytest.importorskip("sacrebleu")
    scorer = sacrebleu.BLEU(
        tokenize=BLEU_TOKENIZE,
        lowercase=BLEU_LOWERCASE,
        smooth_method="none",  # the published value is unsmoothed
        max_ngram_order=BLEU_MAX_NGRAM_ORDER,
        effective_order=False,
    )
    got = scorer.corpus_score([PAPINENI_HYP], [[r] for r in PAPINENI_REFS]).score
    assert got == pytest.approx(PAPINENI_BLEU, abs=1e-10)


def test_published_conventions_reproduce_the_anchor_within_smoothing_noise() -> None:
    """The SHIPPED path (exp smoothing) lands on the same anchor.

    ``smooth_method="exp"`` only kicks in when an n-gram order has zero matches.
    All four orders match here, so the pinned published conventions must return
    the identical unsmoothed value — proving the smoothing choice is inert on a
    normal corpus and only guards the degenerate case.
    """
    got = corpus_bleu_score([PAPINENI_REFS[0]], [PAPINENI_HYP])
    multi = corpus_bleu_score([PAPINENI_REFS], [PAPINENI_HYP])
    assert BLEU_SMOOTH_METHOD == "exp"
    assert multi == pytest.approx(PAPINENI_BLEU, abs=1e-10)
    # single-reference scoring is a different (harder) task, so a lower score
    assert 0.0 < got < multi


def test_identical_output_scores_100() -> None:
    refs = ["Wir treffen uns am Montag.", "Das Protokoll ist fertig."]
    assert corpus_bleu_score(refs, list(refs)) == pytest.approx(100.0)


def test_unrelated_output_scores_near_zero_not_exactly_zero() -> None:
    """Documents what ``smooth_method="exp"`` actually does to the floor.

    The hypothesis shares only the final "." with the reference, so orders 2-4
    have zero matches. Unsmoothed BLEU would be exactly 0; exponential smoothing
    replaces those zeros with a decaying value and leaves a small positive floor.
    This is the pinned convention, so the floor is a property of the published
    number and is asserted rather than glossed over.
    """
    refs = ["Wir treffen uns am Montag um zehn Uhr im grossen Raum."]
    hyps = ["Katzen schlafen gerne lange."]
    got = corpus_bleu_score(refs, hyps)
    assert 0.0 < got < 5.0

    # Nothing at all in common -> exactly 0, even with smoothing on.
    assert corpus_bleu_score(["aaa bbb ccc ddd"], ["eee fff ggg hhh"]) == 0.0


def test_bleu_is_case_sensitive() -> None:
    """German capitalizes nouns; folding case would hide a real error class."""
    refs = ["Der Bericht liegt dem Vorstand vor."]
    assert BLEU_LOWERCASE is False
    assert corpus_bleu_score(refs, ["der bericht liegt dem vorstand vor."]) < 100.0


def test_corpus_bleu_is_not_the_mean_of_sentence_bleus() -> None:
    """The published number aggregates n-gram counts, it does not average rows."""
    refs = [
        "Wir treffen uns am Montag um zehn Uhr im grossen Besprechungsraum.",
        "Ja.",
    ]
    hyps = [
        "Wir treffen uns am Montag um zehn Uhr im grossen Besprechungsraum.",
        "Nein.",
    ]
    corpus = corpus_bleu_score(refs, hyps)
    mean = sum(
        sentence_bleu_diagnostic(r, h) for r, h in zip(refs, hyps)
    ) / len(refs)
    assert corpus != pytest.approx(mean)


def test_brevity_penalty_punishes_truncation() -> None:
    ref = "Der Vorstand hat den Bericht am Montag ohne Gegenstimme angenommen."
    full = evaluate_bleu([ref], [ref])
    short = evaluate_bleu([ref], ["Der Vorstand hat den Bericht"])
    assert full.brevity_penalty == pytest.approx(1.0)
    assert short.brevity_penalty < 1.0
    assert short.bleu < full.bleu
    assert short.hyp_len < short.ref_len


def test_evaluate_reports_components_and_signature() -> None:
    ref = "Wir verschieben den Termin auf naechste Woche."
    res = evaluate_bleu([ref], [ref])
    assert res.n_samples == 1
    assert len(res.precisions) == BLEU_MAX_NGRAM_ORDER
    assert res.length_ratio == pytest.approx(1.0)
    assert "tok:13a" in res.signature
    assert res.signature == bleu_signature()


def test_signature_names_every_pinned_convention() -> None:
    sig = bleu_signature()
    for token in ("nrefs:1", "case:mixed", "eff:no", "tok:13a", "smooth:exp", "version:"):
        assert token in sig, f"{token!r} missing from signature {sig!r}"
    assert bleu_signature(3).startswith("nrefs:3")


def test_multi_reference_scores_at_least_as_high_as_any_single_reference() -> None:
    hyp = "Das Team trifft sich morgen."
    alts = ["Das Team trifft sich morgen.", "Wir sehen uns morgen im Team."]
    multi = corpus_bleu_score([alts], [hyp])
    singles = [corpus_bleu_score([a], [hyp]) for a in alts]
    assert multi >= max(singles) - 1e-9


def test_ragged_multi_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="ragged multi-reference"):
        corpus_bleu_score([["a", "b"], ["c"]], ["a", "c"])


def test_misaligned_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="must align"):
        corpus_bleu_score(["a", "b"], ["a"])


def test_empty_input_is_zero_not_a_crash() -> None:
    assert corpus_bleu_score([], []) == 0.0
    assert evaluate_bleu([], []).n_samples == 0


def test_sentence_diagnostic_uses_effective_order() -> None:
    """A 3-word row can still score >0 — impossible under the corpus convention."""
    assert sentence_bleu_diagnostic("Guten Morgen zusammen", "Guten Morgen zusammen") > 0
    assert corpus_bleu_score(["Guten Morgen zusammen"], ["Guten Morgen zusammen"]) == 0.0


def test_dialect_shape_bleu_survives_what_wer_punishes() -> None:
    """The reason this metric exists, in one assertion.

    Swiss-German dialect -> standard German is a translation task: a correct
    rendering can differ from the reference in word choice and order. WER charges
    every such difference as an error; BLEU still credits the matching n-grams.
    """
    jiwer = pytest.importorskip("jiwer")
    ref = "Wir haben gestern lange über das neue Projekt gesprochen"
    hyp = "Wir haben gestern lang über das neue Projekt geredet"
    wer_pct = jiwer.wer(ref, hyp) * 100
    bleu = corpus_bleu_score([ref], [hyp])
    assert wer_pct > 20.0  # WER reads this as a badly broken transcript
    assert bleu > 40.0  # BLEU reads it as a largely correct translation
