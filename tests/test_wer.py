"""WER regression tests: hand-counted error rates + German normalization.

WER = (substitutions + deletions + insertions) / reference_word_count.
"""

from __future__ import annotations

import pytest

from raven_eval_core import (
    compute_wer,
    normalize_permissive,
    normalize_strict_de,
    normalize_verbatim,
)


def test_perfect_match_is_zero():
    r = compute_wer("das ist ein test heute", "das ist ein test heute")
    assert r.wer_strict == pytest.approx(0.0)
    assert r.sub == 0 and r.ins == 0 and r.delete == 0
    assert r.ref_word_count == 5


def test_single_substitution():
    # 5 ref words, one wrong → 1/5 = 0.2.
    r = compute_wer("das ist ein test heute", "das ist ein toast heute")
    assert r.wer_strict == pytest.approx(0.2)
    assert r.sub == 1 and r.ins == 0 and r.delete == 0


def test_single_deletion():
    r = compute_wer("das ist ein test heute", "das ist ein test")
    assert r.wer_strict == pytest.approx(0.2)
    assert r.delete == 1 and r.sub == 0 and r.ins == 0


def test_single_insertion():
    r = compute_wer("das ist ein test heute", "das ist ein test heute morgen")
    assert r.wer_strict == pytest.approx(0.2)
    assert r.ins == 1 and r.sub == 0 and r.delete == 0
    # IER = insertions / ref_word_count.
    assert r.ier == pytest.approx(1 / 5)


def test_mixed_errors():
    # ref 5 words. hyp: "das" ok, "war"(sub), "ein" ok, "test" deleted,
    # "heute" ok, "morgen" inserted → checked against jiwer's alignment.
    ref = "das ist ein test heute"
    hyp = "das war ein heute morgen"
    r = compute_wer(ref, hyp)
    # jiwer aligns: ist→war (sub), test→(del), +morgen (ins) OR similar;
    # total edits / 5 must be a sane multiple of 0.2.
    assert r.wer_strict == pytest.approx((r.sub + r.ins + r.delete) / 5)
    assert r.sub + r.ins + r.delete >= 2


def test_german_number_expansion():
    # strict-de expands digits via num2words(de): "3" == "drei".
    r = compute_wer("ich habe 3 äpfel", "ich habe drei äpfel")
    assert r.wer_strict == pytest.approx(0.0)
    assert "drei" in normalize_strict_de("ich habe 3 äpfel")


def test_eszett_folds_to_ss():
    assert normalize_strict_de("Straße") == "strasse"
    r = compute_wer("auf der straße", "auf der strasse")
    assert r.wer_strict == pytest.approx(0.0)


def test_filler_words_stripped_in_strict():
    # "äh" is a generic German filler → stripped in strict-de.
    assert "äh" not in normalize_strict_de("das äh ist ein test").split()
    r = compute_wer("das äh ist ein test", "das ist ein test")
    assert r.wer_strict == pytest.approx(0.0)


def test_punctuation_stripped_strict_but_kept_verbatim():
    assert normalize_strict_de("Hallo, Welt!") == "hallo welt"
    assert "," in normalize_verbatim("Hallo, Welt!")
    # permissive lowercases + strips punctuation but no number/filler munging.
    assert normalize_permissive("Es sind 3 Äpfel!") == "es sind 3 äpfel"


# ── Private-name-removal proof (the key public-safety change) ───────────────


def test_spoken_name_not_stripped_by_default():
    """A human name is content, not a speaker tag — must survive normalization.

    Proves the removal of the hardcoded ``phil|chris|helena`` from the source
    regex: with no caller-supplied labels, "Helena" is preserved.
    """
    out = normalize_strict_de("[Helena] hat heute angerufen")
    assert "helena" in out.split()

    # And a bare spoken name is likewise preserved.
    assert "helena" in normalize_strict_de("helena kommt später").split()


def test_name_stripped_only_when_supplied_as_explicit_label():
    """When the caller explicitly declares "Helena" a speaker label, [Helena]
    is scrubbed — opt-in, not hardcoded."""
    out = normalize_strict_de(
        "[Helena] hat heute angerufen", speaker_labels=["Helena"]
    )
    assert "helena" not in out.split()
    assert out == "hat heute angerufen"


def test_explicit_label_changes_wer_but_default_counts_the_name():
    ref = "[Helena] das ist ein test"
    hyp = "das ist ein test"
    # Default: [Helena] → "helena" survives → ref has 5 words, hyp 4 → 1 del/5.
    default = compute_wer(ref, hyp)
    assert default.wer_strict == pytest.approx(0.2)
    # Explicit label: [Helena] scrubbed → both sides "das ist ein test" → 0.
    labeled = compute_wer(ref, hyp, speaker_labels=["Helena"])
    assert labeled.wer_strict == pytest.approx(0.0)


def test_generic_speaker_tags_still_stripped_by_default():
    # Generic (non-name) bracketed tags remain stripped without any config.
    assert normalize_strict_de("[Sprecher0] hallo welt") == "hallo welt"
    assert normalize_strict_de("[speaker 1] hallo welt") == "hallo welt"
