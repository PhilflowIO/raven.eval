"""MOSS-Transcribe-Diarize output: the markup is parsed away, the speech stays.

The forms below come from two places: the earlier internal harness's normaliser
tests (decimal point and decimal comma, integer seconds, case and inner
whitespace in the tag), and the model's real output on the 2026-11 sweep, where
it also broke its own markup (an unclosed timestamp, a stray ``[``).
"""

from __future__ import annotations

import pytest

from raven_asr.config import KNOWN_MODELS, OUTPUT_FORMATS
from raven_asr.output_formats import (
    Segment,
    parse_moss,
    parse_nemotron,
    parse_vibevoice,
    parser_for,
)
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


def test_moss_and_vibevoice_are_bound_to_their_parsers_and_no_other_model_is():
    assert KNOWN_MODELS["OpenMOSS-Team/MOSS-Transcribe-Diarize"].output_format == (
        "moss-timed-speakers"
    )
    assert KNOWN_MODELS["microsoft/VibeVoice-ASR-HF"].output_format == "vibevoice-json"
    assert {m for m, s in KNOWN_MODELS.items() if s.output_format} == set(OUTPUT_FORMATS)
    for fmt in OUTPUT_FORMATS.values():
        assert parser_for(fmt) is not None
    assert parser_for(None) is None
    with pytest.raises(KeyError):
        parser_for("no-such-format")


# ── VibeVoice-ASR ────────────────────────────────────────────────────────────
# The shape is the model card's own example (microsoft/VibeVoice-ASR-HF,
# "Speaker-timestamped transcription"), with German words.
VV = ('[{"Start":0,"End":15.43,"Speaker":0,"Content":"Guten Tag und willkommen."},'
      '{"Start":15.43,"End":21.05,"Speaker":1,"Content":"Danke, schön hier zu sein."}]')


def test_vibevoice_turns_become_spoken_text_and_segments():
    parsed = parse_vibevoice(VV)
    assert parsed.text == "Guten Tag und willkommen. Danke, schön hier zu sein."
    assert parsed.segments == [
        Segment("S00", 0.0, 15.43, "Guten Tag und willkommen."),
        Segment("S01", 15.43, 21.05, "Danke, schön hier zu sein."),
    ]


def test_vibevoice_json_escapes_are_decoded():
    parsed = parse_vibevoice('[{"Start":0,"End":1,"Speaker":0,"Content":"Gr\\u00fc\\u00dfe \\"Anna\\""}]')
    assert parsed.text == 'Grüße "Anna"'


def test_vibevoice_answer_cut_by_the_output_cap_keeps_every_word_it_has():
    raw = VV[:-1] + ',{"Start":21.05,"End":30,"Speaker":0,"Content":"Und dann sa'
    parsed = parse_vibevoice(raw)
    assert parsed.text.endswith("schön hier zu sein. Und dann sa")
    assert parsed.segments[-1] == Segment(None, None, None, "Und dann sa")


def test_vibevoice_plain_and_empty_answers():
    assert parse_vibevoice("  nur  Text ") .text == "nur Text"
    assert parse_vibevoice("nur Text").segments == []
    assert parse_vibevoice("[]").text == ""
    # An empty turn carries no words and no segment.
    assert parse_vibevoice('[{"Start":0,"End":1,"Speaker":0,"Content":""}]').text == ""


def test_vibevoice_non_speech_tags_are_not_spoken():
    raw = (
        'assistant\n[{"Start":0.0,"End":4.0,"Speaker":0,"Content":"[Noise] Guten Tag."},'
        '{"Start":4.0,"End":12.0,"Content":"[Silence]"},'
        '{"Start":12.0,"End":13.0,"Speaker":0,"Content":"[Unintelligible Speech]"},'
        '{"Start":13.0,"End":14.0,"Speaker":0,"Content":"Danke [Noise] sehr."},'
        '{"Start":14.0,"End":15.0,"Speaker":0,"Content":"[Noise] [Noise]"},'
        '{"Start":15.0,"End":16.0,"Speaker":0,"Content":"[Mus'
    )
    parsed = parse_vibevoice(raw)
    assert parsed.text == "Guten Tag. Danke sehr."
    assert [s.text for s in parsed.segments] == ["Guten Tag.", "Danke sehr."]
    assert parsed.segments[0].speaker == "S00"
    assert parsed.segments[0].end_s == 4.0


def test_vibevoice_answer_of_only_tags_is_no_speech():
    raw = '[{"Start":0,"End":9,"Speaker":0,"Content":"[Silence]"},{"Start":9,"End":10,"Speaker":0,"Content":"[Noise]"}]'
    parsed = parse_vibevoice(raw)
    assert parsed.text == ""
    assert parsed.segments == []


def test_vibevoice_brackets_outside_the_tag_list_stay_for_the_scoring_contract():
    raw = '[{"Start":0,"End":1,"Speaker":0,"Content":"[Heidi] kam."}]'
    assert parse_vibevoice(raw).text == "[Heidi] kam."


def test_vibevoice_loop_on_words_keeps_every_word_for_the_runaway_count():
    turn = '{"Start":0,"End":1,"Speaker":0,"Content":"und dann"}'
    tag = '{"Start":0,"End":1,"Speaker":0,"Content":"[Noise]"}'
    raw = "[" + ",".join([tag] * 5 + [turn] * 40) + "]"
    parsed = parse_vibevoice(raw)
    assert parsed.text == " ".join(["und dann"] * 40)
    from raven_eval_core.flozi_wer import runaway_stats
    stats = runaway_stats(["und dann auch"], [parsed.text], 3.0)
    assert stats.n_runaway == 1


def test_vibevoice_parsed_output_scores_like_the_plain_transcript():
    ref = "Guten Tag und willkommen. Danke, schön hier zu sein."
    assert corpus_wer_pct([ref], [parse_vibevoice(VV).text]) == 0
    assert corpus_wer_pct([ref], [VV]) > 0


# ------------------------------------------------------------ Nemotron locale tags

NEMOTRON_REAL = ("Man steigt in der Ziffer und sinkt im Werte. <de-DE> Es ist wie bei den "
                 "Schulden weniger ist da immer mehr. <de-DE>")


def test_every_nemotron_locale_tag_is_removed_not_only_the_last():
    assert parse_nemotron(NEMOTRON_REAL).text == (
        "Man steigt in der Ziffer und sinkt im Werte. Es ist wie bei den "
        "Schulden weniger ist da immer mehr.")
    assert parse_nemotron(NEMOTRON_REAL).segments == []


def test_a_tag_between_two_words_leaves_a_gap():
    assert parse_nemotron("gut<de-DE>Morgen").text == "gut Morgen"


def test_text_that_only_looks_like_a_tag_stays():
    assert parse_nemotron("ich <3 dich <b> <de> <DE-de>").text == "ich <3 dich <b> <de> <DE-de>"


def test_parsing_twice_changes_nothing():
    once = parse_nemotron(NEMOTRON_REAL).text
    assert parse_nemotron(once).text == once


def test_both_prompt_conditioned_nemotron_models_are_bound_to_the_parser():
    for model in ("nvidia/nemotron-3.5-asr-streaming-0.6b",
                  "nvidia/nemotron-labs-unified-asr-0.6b"):
        assert OUTPUT_FORMATS[model] == "nemotron-locale-tags"
        assert KNOWN_MODELS[model].output_format == "nemotron-locale-tags"
        assert parser_for(OUTPUT_FORMATS[model]) is parse_nemotron
