"""What a model's server returns, turned into what was spoken.

Most models answer a transcription request with the transcript and nothing else.
Some wrap it in markup of their own — MOSS-Transcribe-Diarize interleaves
timestamps and speaker tags with the words:

    [4.18][S01] Für die besten Aussichten … gegenüberliegenden[10.96][10.96][S01] Ufer von …[13.81]

VibeVoice-ASR answers with a JSON list of speaker turns:

    [{"Start":0,"End":4.2,"Speaker":0,"Content":"Guten Tag."}, …]

That markup is how one model formats its output, not speech, and scoring it
counts every ``[4.18]`` and ``[S01]`` as an inserted word (MOSS scored ~100 %
WER on sets it transcribes almost perfectly). It is parsed here, at the
model-spec layer, and not in the scoring contract: ``benchmark.config.yaml``
normalises *speech* the same way for every model and does not learn vendor
output formats.

A model's format is a registry fact (``raven_asr.config.OUTPUT_FORMATS``). The
adapter applies it to every response; the runner then records

    prediction       the spoken text only — what WER is computed on
    raw_prediction   the server's output, byte for byte, for audit and re-parse
    segments         the speaker turns the markup carried (when it carried any)

``raven_asr.reparse`` applies the same parser to predictions recorded before a
model had one, from the raw text they hold, without running inference again.

Stdlib only: ``reparse`` and the tests import this without the ``asr`` extra.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "OUTPUT_PARSERS", "ParsedOutput", "Segment", "parse_moss", "parse_vibevoice",
    "parser_for",
]


@dataclass(frozen=True)
class Segment:
    """One speaker turn carried by the markup.

    ``start_s`` / ``end_s`` are None where the markup did not bound the turn;
    ``speaker`` is None for text the model emitted before any speaker tag.
    """

    speaker: str | None
    start_s: float | None
    end_s: float | None
    text: str

    def as_json(self) -> dict[str, object]:
        return {
            "speaker": self.speaker, "start_s": self.start_s,
            "end_s": self.end_s, "text": self.text,
        }


@dataclass(frozen=True)
class ParsedOutput:
    """A model output split into the spoken text and the structure around it."""

    text: str
    segments: list[Segment] = field(default_factory=list)


#: One MOSS markup token. A timestamp in seconds — ``[4.18]``, ``[3,12]`` (a
#: decimal comma, as some locales print it) or ``[12]`` — or a speaker tag
#: (``[S01]``; case and inner whitespace tolerated): the forms the earlier
#: internal harness stripped in its normaliser. Plus the two ways the model was
#: seen to break its own markup on the 2026-11 sweep (~20 of 23,830 outputs): a
#: decimal timestamp whose ``]`` never came (``[0.35[S01]``, ``…[4.33`` at the
#: end), and a lone ``[`` that opens nothing (``hochwertige[[S01]``) — a ``[``
#: with no ``]`` before the next ``[`` or the end of the output.
#:
#: Nothing broader: a bracketed word (``[Musik]``, ``[Heidi]``) has a closing
#: bracket and is left for the scoring contract, as for any other model — the
#: model did write it, and the contract decides what counts as speech.
_MOSS_MARKUP = re.compile(
    r"\[\s*(?P<ts>\d+[.,]\d+)\s*\]?"
    r"|\[\s*(?P<its>\d+)\s*\]"
    r"|\[\s*[Ss](?P<spk>\d+)\s*\]"
    r"|(?P<stray>\[)(?=[^\]]*(?:\[|$))"
)
_WHITESPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def parse_moss(raw: str) -> ParsedOutput:
    """MOSS-Transcribe-Diarize output → spoken text + speaker segments.

    The model emits ``[start][Snn] text [end]`` per turn, so the end of one turn
    and the start of the next sit side by side (``…[10.96][10.96][S01] …``).
    Markup is replaced by a space, never glued: a turn boundary falls between
    words. A tag with no text after it yields no segment and no text; text with
    no markup at all comes back unchanged (whitespace collapsed) with no
    segments — a plain answer is still a valid MOSS answer.
    """
    pieces: list[str] = []
    segments: list[Segment] = []
    last_ts: float | None = None
    # The open turn: speaker, start, collected text.
    speaker: str | None = None
    start: float | None = None
    words: list[str] = []
    is_open = False

    def close(end: float | None) -> None:
        nonlocal is_open, words
        text = _clean(" ".join(words))
        if is_open and text:
            segments.append(Segment(speaker, start, end, text))
        is_open, words = False, []

    pos = 0
    for m in _MOSS_MARKUP.finditer(raw):
        between = raw[pos:m.start()]
        pos = m.end()
        if between.strip():
            if not is_open:
                speaker, start, is_open = None, last_ts, True
            words.append(between)
            pieces.append(between)
        stamp = m.group("ts") or m.group("its")
        if stamp is not None:
            ts = float(stamp.replace(",", "."))
            if is_open and _clean(" ".join(words)):
                close(ts)
            last_ts = ts
        elif m.group("spk") is not None:
            close(None)
            speaker, start, is_open = f"S{m.group('spk')}", last_ts, True
        # A stray "[" only separates; it carries neither time nor speaker.
    tail = raw[pos:]
    if tail.strip():
        if not is_open:
            speaker, start, is_open = None, last_ts, True
        words.append(tail)
        pieces.append(tail)
    close(None)
    return ParsedOutput(text=_clean(" ".join(pieces)), segments=segments)


#: One turn object of a VibeVoice answer: a ``{…}`` with no nested braces (the
#: turns are flat), and the ``Content`` string of a turn the answer broke off in
#: (``max_tokens`` reached mid-string), which has no closing quote.
_VV_OBJECT = re.compile(r"\{[^{}]*\}")
_VV_OPEN_CONTENT = re.compile(r'"Content"\s*:\s*"((?:[^"\\]|\\.)*)$', re.DOTALL)


def _vv_float(value: object) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _vv_speaker(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        return f"S{int(value):02d}"
    return str(value)


def _vv_string(body: str) -> str:
    """A JSON string body (without quotes) decoded; a broken escape kept verbatim."""
    try:
        return str(json.loads(f'"{body}"'))
    except ValueError:
        return body


#: The non-speech tags VibeVoice-ASR writes into ``Content``, as observed on 3,204
#: German clips (2026-11-09): its way of labelling a stretch with no words, not
#: anything spoken, and scoring them would count each one as an inserted word.
#: A closed list, not "anything in brackets": ``[Heidi]`` would be a word the
#: model wrote, left for the scoring contract like for any other model.
_VV_NON_SPEECH = re.compile(
    r"\[\s*(?:noise|silence|music|lyric|click|speech|environmental sounds"
    r"|unintelligible speech)\s*\]",
    re.IGNORECASE,
)


_VV_BROKEN_TAG = re.compile(r"\[[^\[\]]*$")


def _vv_spoken(content: str) -> str:
    """A turn's ``Content`` without non-speech tags (a tag between words is a gap)."""
    return _clean(_VV_NON_SPEECH.sub(" ", content))


def parse_vibevoice(raw: str) -> ParsedOutput:
    """VibeVoice-ASR output → spoken text + speaker segments.

    The model card calls the answer "a string that resembles a JSON structure"
    (https://huggingface.co/microsoft/VibeVoice-ASR-HF, "Speaker-timestamped
    transcription"): a list of ``{"Start", "End", "Speaker", "Content"}`` turns.
    The spoken text is the turns' ``Content`` joined by a space, in order.

    Parsed turn by turn, not as one document, so an answer cut off by the output
    cap keeps every complete turn plus the words of the turn it broke off in. A
    turn that is not valid JSON is skipped as a whole — it carries no Content the
    parser could trust. An answer with no turn object at all (the model answered
    in plain text) comes back unchanged, whitespace collapsed, with no segments.

    The model labels non-speech with ``[Noise]``-style tags, as whole turns and
    inside turns. Those are dropped, and a turn left with no words yields no
    segment; ``raw_prediction`` keeps every one. A model that loops on words
    keeps them all and stays visible to the runaway count (``wer.runaway``).
    """
    pieces: list[str] = []
    segments: list[Segment] = []
    found = raw.strip() == "[]"  # no speech: an empty turn list, not the text "[]"
    end = 0
    for m in _VV_OBJECT.finditer(raw):
        found = True
        end = m.end()
        try:
            turn = json.loads(m.group(0))
        except ValueError:
            continue
        if not isinstance(turn, dict):
            continue
        text = _vv_spoken(str(turn.get("Content") or ""))
        if not text:
            continue
        pieces.append(text)
        segments.append(Segment(
            _vv_speaker(turn.get("Speaker")), _vv_float(turn.get("Start")),
            _vv_float(turn.get("End")), text,
        ))
    if tail := _VV_OPEN_CONTENT.search(raw[end:]):
        found = True
        # The cap can stop the model mid-tag: a "[" still open at the end of the
        # answer starts a tag that lost its name, not a word.
        text = _vv_spoken(_VV_BROKEN_TAG.sub("", _vv_string(tail.group(1))))
        if text:
            pieces.append(text)
            segments.append(Segment(None, None, None, text))
    if not found:
        return ParsedOutput(text=_clean(raw))
    return ParsedOutput(text=_clean(" ".join(pieces)), segments=segments)


#: Output format name → parser. The names are what ``raven_asr.config`` binds a
#: model to; a model with no entry there returns its transcript as-is.
OUTPUT_PARSERS: dict[str, Callable[[str], ParsedOutput]] = {
    "moss-timed-speakers": parse_moss,
    "vibevoice-json": parse_vibevoice,
}


def parser_for(output_format: str | None) -> Callable[[str], ParsedOutput] | None:
    """The parser for ``output_format``, None for a plain-text model."""
    if output_format is None:
        return None
    try:
        return OUTPUT_PARSERS[output_format]
    except KeyError:
        raise KeyError(
            f"unknown output format {output_format!r}; have {sorted(OUTPUT_PARSERS)}"
        ) from None
