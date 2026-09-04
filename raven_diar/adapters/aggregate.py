"""Fold word/utterance-level speaker labels into diarization turns — ONE way.

Hosted diarizers (AssemblyAI, Deepgram, Speechmatics, …) do not return turns.
They return *words* or *utterances*, each carrying a speaker label, and leave the
folding to the caller. That folding is a scoring decision, not a formatting
detail: how long a silence you bridge before starting a new turn moves miss and
false-alarm directly. If every adapter folded its own way, a DER *difference*
between two providers would partly measure our two folding rules rather than the
two models — exactly the confound this toolkit exists to prevent.

So the rule lives here, once, and every hosted adapter calls it with the shared
default. An adapter that needs a different threshold is making a benchmark claim
and must say so in its published row; it does not get to do it quietly.

The local-GPU adapters (pyannote, sortformer) do NOT use this: their backends
already emit turns, and re-folding them would be a second, invisible opinion.

That asymmetry is real and is measured rather than hand-waved. A hosted row is a
FIXED POINT of this function — folding already-folded turns changes nothing, so
it reads 0.000 pp — which is the direct evidence that the step is reconstruction
and not tuning. A local row does move (between -0.19 and -1.77 pp across the
committed artifacts, signed both ways and different per corpus), so a hosted-
vs-local comparison shares gold, scorer and collars but not this one step.
``raven_diar.analysis.folding_sensitivity`` reports the residue for any row, and
``BENCHMARKS.md`` states it rather than claiming an identical protocol. Applying
the folding uniformly instead was considered and rejected: it would move six
published numbers and break the two vendor reproductions this benchmark rests on.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from raven_eval_core.der import DiarSegment

#: Silence (seconds) bridged inside one speaker's turn before the turn is split.
#:
#: 0.5 s is the shared contract for every hosted adapter. It sits above normal
#: inter-word gaps and short breaths (which would otherwise shred one utterance
#: into dozens of turns and inflate false alarm at collar 0.0) and below a
#: conversational pause a human transcript would render as a new turn. It is a
#: *constant*, not a tuning knob: per-provider or per-dataset values would make
#: providers incomparable, which is the thing this module prevents.
DEFAULT_GAP_MERGE_S: float = 0.5


@dataclass(frozen=True)
class LabelledSpan:
    """One word or utterance with a speaker label, as a hosted API returns it."""

    start: float
    end: float
    speaker: str


#: Accepted input shape: a :class:`LabelledSpan` or a plain ``(start, end, speaker)``.
SpanLike = LabelledSpan | tuple[float, float, str]


def _normalise(span: SpanLike) -> LabelledSpan:
    if isinstance(span, LabelledSpan):
        return span
    start, end, speaker = span
    return LabelledSpan(float(start), float(end), str(speaker))


def spans_to_turns(
    spans: Iterable[SpanLike],
    *,
    gap_merge_s: float = DEFAULT_GAP_MERGE_S,
) -> list[DiarSegment]:
    """Fold labelled word/utterance spans into ``(start, end, speaker)`` turns.

    Merging is done **per speaker**, not over the globally time-sorted stream:
    conversational audio interleaves and overlaps speakers, and a single
    interjected word from B must not chop A's turn in two. Two spans of the same
    speaker merge when the silence between them is ``<= gap_merge_s``; overlapping
    or out-of-order same-speaker spans (word timings from hosted APIs are not
    always monotonic) therefore merge too, and the turn takes the later end.

    Zero- and negative-length spans are dropped: they carry no scoreable time and
    an RTTM line with duration 0 is malformed.

    Args:
        spans: Labelled word/utterance spans, in any order.
        gap_merge_s: Max silence bridged within one turn. Must be >= 0. Leave at
            :data:`DEFAULT_GAP_MERGE_S` unless you are publishing the deviation.

    Returns:
        Turns sorted by (start, end, speaker) — the order ``to_rttm`` expects.
    """
    if gap_merge_s < 0:
        raise ValueError(f"gap_merge_s must be >= 0, got {gap_merge_s!r}")

    by_speaker: dict[str, list[LabelledSpan]] = {}
    for raw in spans:
        span = _normalise(raw)
        if span.end <= span.start:
            continue
        by_speaker.setdefault(span.speaker, []).append(span)

    turns: list[DiarSegment] = []
    for speaker, group in by_speaker.items():
        group.sort(key=lambda s: (s.start, s.end))
        start, end = group[0].start, group[0].end
        for span in group[1:]:
            if span.start - end <= gap_merge_s:      # negative gap == overlap
                end = max(end, span.end)
            else:
                turns.append((start, end, speaker))
                start, end = span.start, span.end
        turns.append((start, end, speaker))

    turns.sort(key=lambda t: (t[0], t[1], t[2]))
    return turns
