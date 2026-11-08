"""Cross-utterance leakage gate (raven_eval_core.leakage, raven.eval#89)."""

from __future__ import annotations

from raven_eval_core.leakage import (
    LEAK_MAX_SHARE_PCT,
    LEAK_MIN_FLAGGED,
    LeakResult,
    cross_utterance_leaks,
)

A = "im gesellschaftsrecht findet sich keine entsprechung für diese regel"
B = "der edle könig selber verschloss die kammertür hinter sich und ging"
C = "morgen früh fahren wir mit dem zug nach hamburg an die see"


def test_a_transcript_of_its_own_clip_is_not_flagged() -> None:
    res = cross_utterance_leaks([A, B, C], [A, B, "morgen fahren wir mit dem bus"])
    assert res.flagged == ()


def test_a_transcript_continuing_with_another_clip_is_flagged_with_its_source() -> None:
    # The #89 shape: the right sentence, then the words of the next request's clip.
    res = cross_utterance_leaks([A, B, C], [f"{A} {B}", B, C])
    assert res.flagged == (0,) and res.sources == (1,)


def test_the_match_is_on_normalized_words_not_on_punctuation_or_case() -> None:
    leaked = "Im Gesellschaftsrecht findet sich keine Entsprechung. Der edle König, selber verschloss die Kammertür!"
    assert cross_utterance_leaks([A, B], [leaked, B]).flagged == (0,)


def test_a_run_shorter_than_six_words_is_not_text_of_another_clip() -> None:
    assert cross_utterance_leaks([A, B], [f"{A} der edle könig selber verschloss", B]).flagged == ()


def test_a_run_shared_with_the_own_reference_is_not_a_leak() -> None:
    # Corpora repeat sentences (FLEURS reads one sentence by several speakers): a
    # correct transcript of a duplicate matches the other reference too.
    res = cross_utterance_leaks([A, A, B], [A, A, B])
    assert res.flagged == ()


def test_failed_requests_and_empty_transcripts_are_not_flagged() -> None:
    assert cross_utterance_leaks([A, B], [None, ""]).flagged == ()


def test_the_gate_needs_both_a_share_and_a_count() -> None:
    def result(n_flagged: int, n: int) -> LeakResult:
        return LeakResult(tuple(range(n_flagged)), tuple(range(n_flagged)), n)

    # A 20-clip smoke with two natural duplicates is 10 %, but only two clips.
    assert not result(LEAK_MIN_FLAGGED, 20).failed
    # The largest clean share of round 4: 22 of 1968 (Whisper-turbo, VoxPopuli).
    assert not result(22, 1968).failed
    assert result(int(1968 * LEAK_MAX_SHARE_PCT / 100) + 1, 1968).failed
    # The leaked Voxtral run of #89: 473 of its first 646 clips.
    assert result(473, 646).failed


def test_lists_of_different_length_are_refused() -> None:
    import pytest

    with pytest.raises(ValueError, match="differ in length"):
        cross_utterance_leaks([A], [A, B])
