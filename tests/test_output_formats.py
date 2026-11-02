"""MOSS-Transcribe-Diarize output: the markup is parsed away, the speech stays.

The forms below come from two places: the earlier internal harness's normaliser
tests (decimal point and decimal comma, integer seconds, case and inner
whitespace in the tag), and the model's real output on the 2026-11 sweep, where
it also broke its own markup (an unclosed timestamp, a stray ``[``).
"""

from __future__ import annotations

import pytest

from raven_asr.config import KNOWN_MODELS, OUTPUT_FORMATS
from raven_asr.output_formats import Segment, parse_moss, parser_for
from raven_eval_core.flozi_wer import corpus_wer_pct

REAL = (
    "[4.18][S01] Für die besten Aussichten auf Hongkong sollten Sie die Insel "
    "verlassen und zum gegenüberliegenden[10.96][10.96][S01] Ufer von Koh Luhm "
    "fahren.[13.81]"
)


def test_real_output_parses_to_spoken_text_and_two_turns():
    parsed = parse_moss(REAL)
    assert parsed.text == (
        "Für die besten Aussichten auf Hongkong sollten Sie die Insel verlassen "
        "und zum gegenüberliegenden Ufer von Koh Luhm fahren."
    )
    assert parsed.segments == [
        Segment("S01", 4.18, 10.96, "Für die besten Aussichten auf Hongkong sollten "
                "Sie die Insel verlassen und zum gegenüberliegenden"),
        Segment("S01", 10.96, 13.81, "Ufer von Koh Luhm fahren."),
    ]


def test_two_speakers_keep_their_labels():
    parsed = parse_moss("[0.48][S01]guten tag[3.12][3.12][S02]hallo[4.0]")
    assert parsed.text == "guten tag hallo"
    assert [(s.speaker, s.start_s, s.end_s, s.text) for s in parsed.segments] == [
        ("S01", 0.48, 3.12, "guten tag"), ("S02", 3.12, 4.0, "hallo"),
    ]


@pytest.mark.parametrize(
    "raw",
    [
        "[0,48][S01]hallo welt[3,12]",  # decimal comma
        "[12][S01]hallo welt[13]",  # integer seconds
        "[0.48][s01]hallo welt[3.12]",  # lower-case tag
        "[ 0.48 ][ S01 ]hallo welt[ 3.12 ]",  # inner whitespace
        "[S01] hallo welt",  # tag without timestamps
        "[0.48] hallo welt [3.12]",  # timestamps without a tag
        "hallo welt",  # no markup at all
    ],
)
def test_markup_forms(raw: str):
    assert parse_moss(raw).text == "hallo welt"


def test_decimal_comma_timestamps_are_read_as_seconds():
    (seg,) = parse_moss("[0,48][S01]hallo[3,12]").segments
    assert (seg.start_s, seg.end_s) == (0.48, 3.12)


def test_a_tag_with_no_text_yields_no_turn_and_no_words():
    # Real spc-test output: the second speaker opens and closes on nothing.
    parsed = parse_moss("[5.81][5.81][S01] und wieder zurück.[6.81][6.81][S02][6.83]")
    assert parsed.text == "und wieder zurück."
    assert [s.speaker for s in parsed.segments] == ["S01"]


def test_repeated_tags_do_not_duplicate_words():
    assert parse_moss("[1.0][1.0][S01][S01] hallo[2.0][2.0]").text == "hallo"


@pytest.mark.parametrize(
    ("raw", "text"),
    [
        # A timestamp whose ] never came (avemio, voxpopuli).
        ("[0.35[S01]Mehrere Schulbücher wurden.[5.24]", "Mehrere Schulbücher wurden."),
        ("[S01] wir wollen zugang[4.33", "wir wollen zugang"),
        # A lone [ that opens nothing (voxpopuli).
        ("[S01] hochwertige[[S01] ausbildung[5.21]", "hochwertige ausbildung"),
    ],
)
def test_broken_markup_the_model_emitted(raw: str, text: str):
    assert parse_moss(raw).text == text


@pytest.mark.parametrize("word", ["[Musik]", "[Heidi]", "[Günter Rudolph]"])
def test_bracketed_words_are_left_to_the_scoring_contract(word: str):
    # A closed bracket around words is something the model wrote, not markup.
    assert word in parse_moss(f"[0.1][S01] hallo {word}[2.0]").text


def test_numbers_inside_speech_are_not_timestamps():
    assert parse_moss("[0.1][S01] Standard 802.11 und 3,5 Prozent[4.0]").text == (
        "Standard 802.11 und 3,5 Prozent"
    )


def test_parsed_output_scores_like_the_plain_transcript():
    ref = "guten tag hallo"
    assert corpus_wer_pct([ref], [parse_moss("[0.48][S01]guten tag[3.12][S02]hallo").text]) == 0
    # The bug this fixes: scored raw, every timestamp and tag is an insertion.
    assert corpus_wer_pct([ref], ["[0.48][S01]guten tag[3.12][S02]hallo"]) > 0


def test_moss_is_bound_to_its_parser_and_no_other_model_is():
    assert KNOWN_MODELS["OpenMOSS-Team/MOSS-Transcribe-Diarize"].output_format == (
        "moss-timed-speakers"
    )
    assert {m for m, s in KNOWN_MODELS.items() if s.output_format} == set(OUTPUT_FORMATS)
    for fmt in OUTPUT_FORMATS.values():
        assert parser_for(fmt) is not None
    assert parser_for(None) is None
    with pytest.raises(KeyError):
        parser_for("no-such-format")
