"""Numeric entity hit rate — did the transcript get the numbers right?

A wrong digit in a date, an amount or a count is the most expensive error a
meeting transcript can make, and WER barely registers it: "23" for "22" is one
substitution among hundreds of words. This score isolates exactly those tokens.
The split follows KugelAudio's open German TTS benchmark protocol, which scores
entity correctness separately from CER for the same reason.

What an entity is
-----------------
A run of digits, with the separators that give it its shape kept inside it:
``[0-9]+([.,:][0-9]+)*``. So "23", "3,5" (decimal comma), "12.03.2024" (date),
"10:30" (time) and "1990" are one entity each. A separator only joins when a
digit follows it, so the ordinal "3." in "am 3. Mai" is "3", and "2-3" is two
entities, "2" and "3". One canonicalization: a German thousands grouping
("1.000", "2.500.000" — dot-separated groups of exactly three digits after a
first group of one to three) drops its dots, so "1.000" and "tausend" are both
"1000". Amounts, percentages and units are scored on their number: "5 %",
"5 Prozent" and "fünf Prozent" each carry the entity "5"; the unit word is
WER's business. Letters around a digit run are not part of it ("90er" -> "90").

Where it is read
----------------
On the flozi-strict normalization (``raven_eval_core.flozi_wer``), which is
what turns spoken numbers into digits (``alpha2digit``: "dreiundzwanzig" ->
"23", "drei komma fünf" -> "3,5"), but **before its final punctuation strip**
(:func:`~raven_eval_core.flozi_wer.normalize_flozi_before_punctuation_strip`).
The strip is harmless for WER and fatal here: it fuses "3,5" into "35", "2-3"
into "23", "1:0" into "10" and "12.03." into "1203", so a transcript that says
"35" would be credited with the reference's "3,5". Reading one step earlier is
what keeps the guarantee below; every other step is flozi's, verbatim.

Reference and hypothesis are normalized **independently**, each on its own
text — never with knowledge of the other — and an entity is a hit only on exact
string equality. A different number therefore cannot become a hit: it would
need two different spoken numbers to normalize to the same digits, and
``tests/test_metric_contract.py`` pins that ``alpha2digit`` maps every German
number word from 3 to 2999 onto its own digits and nothing else.

What it does not credit (a lower bound, never a flattering one)
--------------------------------------------------------------
Some formatting differences cost a hit although the number was right: flozi
leaves "null", "eins" and "zwei" standing alone as words (``alpha2digit``'s
threshold), so a reference "2" against a spoken "zwei" is a miss; "10:30"
against "zehn Uhr dreißig" ("10 Uhr 30") is a miss; "fünf Millionen" becomes
"5000000" while "5 Mio." stays "5". The asymmetry is deliberate: a formatting
choice can only lower the hit rate, a wrong number can never raise it.

Matching and aggregation
------------------------
Per utterance, as a multiset: an entity that occurs twice in the reference must
occur twice in the hypothesis to score twice (``Σ min(count_ref, count_hyp)``).
An entity the hypothesis adds is not counted — that is an insertion, and WER's.
The corpus hit rate is Σ hits / Σ reference entities over the whole subset.
Utterances whose reference holds no entity contribute nothing, not even to the
denominator, and are reported only as the difference between ``n_samples`` and
``n_utterances_with_entities``. On a read-speech corpus that is most of them,
so the entity count — not the utterance count — is what a hit rate rests on.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .flozi_wer import _check_aligned, normalize_flozi_before_punctuation_strip

#: The two conventions that define this score, restated in
#: ``benchmark.config.yaml`` → ``entity`` and asserted equal there by
#: ``tests/test_metric_contract.py``.
ENTITY_NORMALIZATION = "flozi-strict-before-punctuation-strip"
ENTITY_MATCHING = "multiset-per-utterance"

_NUMBER_RE = re.compile(r"[0-9]+(?:[.,:][0-9]+)*")
_THOUSANDS_RE = re.compile(r"[0-9]{1,3}(?:\.[0-9]{3})+")


def extract_numeric_entities(text: str) -> list[str]:
    """The numeric entities of one raw text, in order of appearance."""
    out: list[str] = []
    for m in _NUMBER_RE.finditer(normalize_flozi_before_punctuation_strip(text)):
        span = m.group()
        if _THOUSANDS_RE.fullmatch(span):
            span = span.replace(".", "")
        out.append(span)
    return out


def utterance_entity_hits(
    references: list[str], predictions: list[str]
) -> list[tuple[int, int]]:
    """Per-utterance ``(hits, reference entities)``, one pair per utterance.

    Utterances without a reference entity come back as ``(0, 0)``; whoever
    aggregates decides to skip them (:func:`entity_units` does). ``Σhits /
    Σentities`` over the result IS the published hit rate.
    """
    _check_aligned(references, predictions)
    units: list[tuple[int, int]] = []
    for ref, hyp in zip(references, predictions, strict=True):
        ref_entities = Counter(extract_numeric_entities(ref))
        hyp_entities = Counter(extract_numeric_entities(hyp))
        hits = sum((ref_entities & hyp_entities).values())
        units.append((hits, sum(ref_entities.values())))
    return units


def entity_units(
    references: list[str], predictions: list[str]
) -> list[tuple[int, int]]:
    """:func:`utterance_entity_hits` restricted to utterances that carry an entity.

    The sampling units of the interval. Resampling the entity-free utterances
    too would leave the ratio unchanged on average but let a resample draw none
    of the few that matter — on a set with three such utterances, about one
    resample in twenty — and score it 0 %, dragging the interval's floor down
    for a reason that has nothing to do with the model.
    """
    return [u for u in utterance_entity_hits(references, predictions) if u[1] > 0]


@dataclass(frozen=True)
class EntityResult:
    """Corpus numeric-entity score of one subset.

    ``hit_rate_pct`` is ``None`` when the references hold no entity at all: a
    subset without numbers has no hit rate, and 0 % would claim every number
    was wrong.
    """

    hit_rate_pct: float | None
    n_hits: int
    n_entities: int
    n_utterances_with_entities: int


def entity_hit_rate(references: list[str], predictions: list[str]) -> EntityResult:
    """Σ hits / Σ reference entities (%) over one paired list."""
    units = entity_units(references, predictions)
    n_hits = sum(h for h, _ in units)
    n_entities = sum(n for _, n in units)
    return EntityResult(
        hit_rate_pct=n_hits / n_entities * 100.0 if n_entities else None,
        n_hits=n_hits,
        n_entities=n_entities,
        n_utterances_with_entities=len(units),
    )
