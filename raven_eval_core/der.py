"""Diarization Error Rate (DER) scoring — the who-spoke-when metric.

A DESCRIPTIVE, comparative benchmark dimension — never a pass/fail gate. DER
answers "who-spoke-when": of the total reference speech time, what fraction did
the diarizing provider miss, hallucinate, or attribute to the wrong speaker.

Implementation
--------------
DER is computed with ``pyannote.metrics`` (``DiarizationErrorRate``), the
de-facto standard implementation of the NIST md-eval DER — optimal speaker
mapping via the Hungarian algorithm, ``(missed + false_alarm + confusion) /
total_reference_speech``. We do NOT reimplement it (use the standard lib for a
standard metric). ``pyannote.metrics`` is lightweight here (pyannote-core +
scikit-learn + scipy + numpy — no torch).

Segment model
-------------
Throughout, a *segment* is a ``(start_s, end_s, speaker_label)`` 3-tuple of
floats/str. ``ref_segments`` come from the per-speaker RTTM ground truth;
``hyp_segments`` are the diarizer's hypothesis.

The collar (default 0.25 s) is the NIST-standard forgiveness window around each
reference speaker boundary — diarization boundaries are inherently fuzzy, and
scoring the exact ms around a turn change penalizes every system equally and
uninformatively. Pass ``collar=0.0, skip_overlap=False`` for an un-forgiven,
overlap-inclusive score.

Extraction note (raven.eval public)
-----------------------------------
This module is the secret-free extraction of Raven's ``der_metrics.py``. The
private ``der_coupled_wer`` path (and its ``wer_metrics`` import + the
``CoupledWerResult`` / ``_ref_speaker_at`` / ``_strip_text`` helpers it used) was
intentionally dropped: this package is purely ``(reference, hypothesis) ->
number``. ``assign_word_speakers`` is kept because it is a self-contained,
dependency-free data-shaping helper with no private coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyannote.core import Annotation, Segment

# pyannote >= 4 exposes DER under .diarization; keep the import narrow.
from pyannote.metrics.diarization import DiarizationErrorRate

# A diarization segment: (start_s, end_s, speaker_label).
DiarSegment = tuple[float, float, str]

# A diarization segment WITH text: (start_s, end_s, speaker_label, text).
DiarSegmentText = tuple[float, float, str, str]

# A timestamped ASR word: (start_s, end_s, text). Emitted by an ASR that carries
# word timestamps but NO speaker labels — the input to assign_word_speakers.
TimedWord = tuple[float, float, str]


def _to_annotation(segments: list[DiarSegment]) -> Annotation:
    """Build a pyannote ``Annotation`` from (start, end, speaker) tuples.

    Zero-/negative-length segments are dropped — pyannote treats them as empty
    support and they carry no diarization signal.
    """
    ann = Annotation()
    for start, end, speaker in segments:
        if end > start:
            ann[Segment(float(start), float(end))] = str(speaker)
    return ann


def compute_der(
    ref_segments: list[DiarSegment],
    hyp_segments: list[DiarSegment],
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> float:
    """Diarization Error Rate as a fraction (0.0 = perfect, can exceed 1.0).

    ``ref_segments`` / ``hyp_segments`` are lists of (start_s, end_s, speaker)
    tuples. Speaker labels need NOT agree between ref and hyp — pyannote finds
    the optimal label mapping (Hungarian) before scoring, so a provider that
    emits ``Sprecher 0`` for the GT's ``alice`` is not penalized for the label
    name, only for boundary/attribution errors.

    ``collar`` is the forgiveness window (s) around each reference boundary
    (default 0.25, NIST md-eval standard). ``skip_overlap`` (default False)
    excludes regions where >1 reference speaker is active from scoring; keep it
    False to score overlapped speech too. Pass ``collar=0.0, skip_overlap=False``
    for the strictest, overlap-inclusive score.

    Edge cases:
      * empty ref  → 0.0 if hyp is also empty, else 1.0 (everything is false
        alarm; we return 1.0 as the conventional "completely wrong" value).
      * empty hyp on non-empty ref → 1.0 (everything missed).
    """
    if not ref_segments:
        return 0.0 if not hyp_segments else 1.0
    if not hyp_segments:
        return 1.0

    reference = _to_annotation(ref_segments)
    hypothesis = _to_annotation(hyp_segments)
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    return float(metric(reference, hypothesis))


@dataclass(frozen=True)
class DerComponents:
    """DER plus its NIST md-eval decomposition, as *fractions* of reference speech.

    ``der == miss + false_alarm + confusion`` holds exactly (all four are divided
    by the same ``total_ref`` — the scored reference-speech duration *after* the
    collar is removed). ``total_ref`` is kept in **seconds** so a corpus can be
    aggregated correctly (sum the seconds, divide once) rather than by averaging
    per-file rates, which would silently weight a 5 s clip like a 5 min meeting.
    """

    der: float
    miss: float
    false_alarm: float
    confusion: float
    total_ref: float  # seconds of scored reference speech (post-collar)


def _components_from_detail(detail: dict) -> DerComponents:
    total = float(detail["total"])
    if total <= 0.0:
        # Degenerate (empty reference after collar) — no rate is defined.
        return DerComponents(
            der=float(detail["diarization error rate"]),
            miss=0.0,
            false_alarm=0.0,
            confusion=0.0,
            total_ref=0.0,
        )
    return DerComponents(
        der=float(detail["diarization error rate"]),
        miss=float(detail["missed detection"]) / total,
        false_alarm=float(detail["false alarm"]) / total,
        confusion=float(detail["confusion"]) / total,
        total_ref=total,
    )


def compute_der_components(
    ref_segments: list[DiarSegment],
    hyp_segments: list[DiarSegment],
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> DerComponents:
    """Single-pair DER with its miss / false-alarm / confusion breakdown.

    Same semantics + edge cases as :func:`compute_der`; additionally exposes the
    three error components (as fractions of reference speech). ``miss`` is speech
    the diarizer failed to attribute to anyone, ``false_alarm`` is speech it
    invented, ``confusion`` is speech attributed to the wrong speaker after the
    optimal Hungarian label mapping.
    """
    if not ref_segments:
        der = 0.0 if not hyp_segments else 1.0
        return DerComponents(der=der, miss=0.0, false_alarm=0.0, confusion=0.0,
                             total_ref=0.0)
    if not hyp_segments:
        total = sum(float(e) - float(s) for s, e, _ in ref_segments if e > s)
        return DerComponents(der=1.0, miss=1.0, false_alarm=0.0, confusion=0.0,
                             total_ref=total)

    reference = _to_annotation(ref_segments)
    hypothesis = _to_annotation(hyp_segments)
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    detail = metric(reference, hypothesis, detailed=True)
    return _components_from_detail(detail)


def compute_der_corpus_detailed(
    pairs: list[tuple[list[DiarSegment], list[DiarSegment]]],
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> tuple[DerComponents, list[DerComponents]]:
    """Corpus DER **and** the per-file components it was accumulated from.

    One scoring pass produces both: the NIST-correct corpus aggregate
    (``Σ(miss+fa+conf) / Σ(total)``) and the list of per-file
    :class:`DerComponents`, aligned index-for-index with ``pairs``.

    The per-file list is what makes the corpus number *auditable* rather than
    merely stated — a reader can see which files carry the error, re-aggregate
    under a different convention (a file-mean, the convention the ETH
    diarization benchmark uses), resample it for a confidence interval, or split
    it by reference speaker count. Every one of those questions is unanswerable
    from a single corpus scalar, which is why this is the primary entry point and
    :func:`compute_der_corpus` is the thin wrapper.

    A pair with both sides empty is skipped by the corpus accumulator (nothing to
    score) but still yields an all-zero row, so indices stay aligned.

    An empty ``pairs`` yields an all-zero aggregate and an empty list.
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    per_file: list[DerComponents] = []
    scored = 0
    for ref_segments, hyp_segments in pairs:
        if not ref_segments and not hyp_segments:
            per_file.append(DerComponents(der=0.0, miss=0.0, false_alarm=0.0,
                                          confusion=0.0, total_ref=0.0))
            continue
        detail = metric(
            _to_annotation(ref_segments), _to_annotation(hyp_segments),
            detailed=True,
        )
        per_file.append(_components_from_detail(detail))
        scored += 1
    if scored == 0:
        return (
            DerComponents(der=0.0, miss=0.0, false_alarm=0.0, confusion=0.0,
                          total_ref=0.0),
            per_file,
        )
    acc = metric[:]  # accumulated components across all files
    aggregate = _components_from_detail({
        "diarization error rate": float(abs(metric)),
        "missed detection": float(acc["missed detection"]),
        "false alarm": float(acc["false alarm"]),
        "confusion": float(acc["confusion"]),
        "total": float(acc["total"]),
    })
    return aggregate, per_file


def compute_der_corpus(
    pairs: list[tuple[list[DiarSegment], list[DiarSegment]]],
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> DerComponents:
    """Corpus DER over many (ref, hyp) file pairs — the NIST-correct aggregation.

    Accumulates one ``DiarizationErrorRate`` across every file so the reported
    DER is ``Σ(miss+fa+conf) / Σ(total)`` over the whole corpus — NOT the mean of
    per-file DERs. This is how CALLHOME / DIHARD / the pyannote model cards report
    a dataset number, and how ``scripts/verify.py`` re-scores committed RTTMs.

    An empty ``pairs`` yields an all-zero result (nothing scored). See
    :func:`compute_der_corpus_detailed` when the per-file breakdown is wanted.
    """
    aggregate, _ = compute_der_corpus_detailed(
        pairs, collar=collar, skip_overlap=skip_overlap
    )
    return aggregate


def file_mean_der(per_file: list[DerComponents]) -> float:
    """Unweighted mean of per-file DERs — the OTHER aggregation convention.

    Not comparable to :func:`compute_der_corpus`: this weights a 30 s clip like a
    50 min meeting, where the corpus aggregate weights by scored reference speech.
    Both are legitimate and both are published in the literature — the ETH
    diarization benchmark (arXiv 2509.26177, Table 3 caption) averages over
    samples, the pyannote model cards and CALLHOME/DIHARD report the corpus
    figure. The convention moves the number by a third of a percentage point on
    our own German CALLHOME row, so a DER is only comparable once it is stated.

    Files that scored no reference speech (``total_ref == 0``) are excluded: they
    have no defined rate and would otherwise pull the mean toward zero.
    """
    rates = [c.der for c in per_file if c.total_ref > 0.0]
    if not rates:
        return 0.0
    return sum(rates) / len(rates)


def assign_word_speakers(
    diar_turns: list[DiarSegment],
    words: list[TimedWord],
) -> list[DiarSegmentText]:
    """Attach a diarization speaker to each timestamped word (WhisperX algorithm).

    ``diar_turns`` are (start, end, speaker) turns from a diarizer's hypothesis
    RTTM; ``words`` are (start, end, text) from an ASR that has word timestamps
    but no speaker labels. This is exactly WhisperX's ``assign_word_speakers``
    merge: each word is assigned to the diarization turn with the greatest
    temporal overlap; a word overlapping no turn falls back to the turn whose
    midpoint is nearest, so no word is silently dropped.

    Returns one (start, end, speaker, text) 4-tuple per word. An empty
    ``diar_turns`` (or ``words``) yields an empty list (nothing to attach).
    """
    if not words or not diar_turns:
        return []
    out: list[DiarSegmentText] = []
    for w_start, w_end, text in words:
        ws, we = float(w_start), float(w_end)
        best_spk: str | None = None
        best_overlap = 0.0
        for t_start, t_end, spk in diar_turns:
            overlap = min(we, float(t_end)) - max(ws, float(t_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = str(spk)
        if best_spk is None:
            # No turn overlaps this word → nearest turn by midpoint distance.
            w_mid = (ws + we) / 2.0
            best_dist = float("inf")
            for t_start, t_end, spk in diar_turns:
                t_mid = (float(t_start) + float(t_end)) / 2.0
                dist = abs(w_mid - t_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_spk = str(spk)
        out.append((ws, we, str(best_spk), text))
    return out


# ── RTTM parsing ──────────────────────────────────────────────────────────


def parse_rttm(text: str) -> list[DiarSegment]:
    """Parse standard NIST RTTM into (start_s, end_s, speaker) segments.

    RTTM ``SPEAKER`` line layout (10 fields):

        SPEAKER <file-id> <chan> <start> <dur> <NA> <NA> <speaker> <NA> <NA>
           0        1        2       3      4    5    6      7        8   9

    Non-``SPEAKER`` lines (e.g. ``SPKR-INFO``) and blank/comment lines are
    ignored. Durations are converted to absolute end times (start + dur).
    """
    segments: list[DiarSegment] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        if len(fields) < 8:
            continue
        try:
            start = float(fields[3])
            dur = float(fields[4])
        except ValueError:
            continue
        speaker = fields[7]
        segments.append((start, start + dur, speaker))
    return segments


def load_rttm(path: Path) -> list[DiarSegment]:
    """Read an RTTM file into (start_s, end_s, speaker) segments."""
    return parse_rttm(Path(path).read_text())


def to_rttm(segments: list[DiarSegment], file_id: str = "fixture") -> str:
    """Serialize segments back to RTTM (roundtrip / test helper)."""
    lines = []
    for start, end, speaker in segments:
        dur = float(end) - float(start)
        lines.append(
            f"SPEAKER {file_id} 1 {float(start):.3f} {dur:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>"
        )
    return "\n".join(lines) + ("\n" if lines else "")
