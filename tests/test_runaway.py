"""Looping hypotheses are counted beside the WER, never removed from it (#62).

Phi-4-multimodal returned up to 16,000 characters of "sehr, sehr, …" on a
handful of 3,332 spc-test clips; those insertions outweighed every other error
in the corpus. ``runaway_stats`` names such utterances and their share of the corpus
edits under the published normalization and alignment.
"""

from __future__ import annotations

import pytest

from raven_eval_core.flozi_wer import corpus_wer_pct, runaway_stats, utterance_word_errors


def test_a_loop_is_counted_and_its_edit_share_is_a_share_of_the_wer() -> None:
    refs = ["das ist gut", "wir gehen heute nach hause", "ja"]
    hyps = ["das ist gut", "wir gehen heute nach hause", "sehr " * 40]
    got = runaway_stats(refs, hyps)
    assert (got.n_runaway, got.n_utterances) == (1, 3)
    assert got.runaway_pct == pytest.approx(100 / 3)
    # Every edit of this corpus is the loop's.
    assert got.edit_share_pct == pytest.approx(100.0)
    assert got.total_edits == sum(e for e, _ in utterance_word_errors(refs, hyps))
    # The WER still counts it: the loop is named, not removed.
    assert corpus_wer_pct(refs, hyps) > 400


def test_the_rule_is_strictly_more_than_ratio_times_the_reference() -> None:
    refs = ["eins zwei"]
    assert runaway_stats(refs, ["a b c d e f"]).n_runaway == 0  # exactly 3x
    assert runaway_stats(refs, ["a b c d e f g"]).n_runaway == 1
    assert runaway_stats(refs, ["a b c d e f g"], length_ratio=4).n_runaway == 0


def test_lengths_are_counted_after_the_published_normalization() -> None:
    # Punctuation is not a word: ", , , ," adds nothing to the hypothesis.
    assert runaway_stats(["hallo"], ["hallo , , , , , ,"]).n_runaway == 0


def test_a_perfect_corpus_has_no_edit_share() -> None:
    got = runaway_stats(["a b"], ["a b"])
    assert (got.n_runaway, got.edit_share_pct) == (0, None)
    assert runaway_stats([], []).n_utterances == 0
