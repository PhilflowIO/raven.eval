"""Tests for the instruments that read a DER artifact past its corpus scalar.

Each one guards a property a published paragraph now leans on: that a bucket DER
is the same kind of quantity as the headline DER, that an interval is
reproducible from its seed, that a boundary offset is speaker-aware, and that the
backfill tool cannot turn a red re-score green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raven_diar.adapters.aggregate import spans_to_turns
from raven_diar.analysis import (
    aggregate_der,
    bootstrap_der_ci,
    boundary_report,
    der_by_speaker_count,
    folding_sensitivity,
    paired_bootstrap_delta,
    reference_overlap,
    speaker_bucket,
)
from raven_diar.rescore import backfill
from raven_diar.score import DerScore, score_segment_pairs
from raven_eval_core.der import DerComponents, file_mean_der, to_rttm


def _write_rttm(path: Path, segments: list[tuple[float, float, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_rttm(segments, file_id=path.stem), encoding="utf-8")
    return path


# ── the aggregation identity ─────────────────────────────────────────────────


def test_aggregate_over_all_rows_equals_corpus_der():
    """Σerr/Σtotal over every file row IS the published corpus number.

    The property that makes a bucket or a resample comparable to the headline:
    they are the same estimator on a subset, not a different statistic.
    """
    pairs = [
        ([(0.0, 10.0, "A")], [(0.0, 8.0, "A")]),
        ([(0.0, 60.0, "A"), (60.0, 120.0, "B")], [(0.0, 120.0, "A")]),
        ([(0.0, 5.0, "A")], [(0.0, 5.0, "A")]),
    ]
    score = score_segment_pairs("t", pairs, file_ids=["a", "b", "c"])
    assert aggregate_der(list(score.per_file), "classic") == pytest.approx(
        score.der_classic, abs=1e-9
    )
    assert aggregate_der(list(score.per_file), "full") == pytest.approx(
        score.der_full, abs=1e-9
    )


def test_file_mean_differs_from_corpus_when_lengths_differ():
    """The two conventions are genuinely different numbers, not a rounding."""
    components = [
        DerComponents(der=0.5, miss=0.5, false_alarm=0.0, confusion=0.0,
                      total_ref=1.0),
        DerComponents(der=0.1, miss=0.1, false_alarm=0.0, confusion=0.0,
                      total_ref=99.0),
    ]
    assert file_mean_der(components) == pytest.approx(0.3)
    # corpus: (0.5*1 + 0.1*99) / 100 = 0.104 — the long file dominates, as it must.
    err = sum(c.miss * c.total_ref for c in components)
    total = sum(c.total_ref for c in components)
    assert err / total == pytest.approx(0.104)


def test_file_mean_ignores_files_with_no_scored_speech():
    """A file with zero scored reference speech has no rate to average in."""
    components = [
        DerComponents(der=0.2, miss=0.2, false_alarm=0.0, confusion=0.0,
                      total_ref=10.0),
        DerComponents(der=0.0, miss=0.0, false_alarm=0.0, confusion=0.0,
                      total_ref=0.0),
    ]
    assert file_mean_der(components) == pytest.approx(0.2)


# ── confidence intervals ──────────────────────────────────────────────────────


def _rows(n: int, *, miss_s: float = 1.0, total_s: float = 10.0):
    pairs = [
        ([(0.0, total_s, "A")], [(0.0, total_s - miss_s, "A")]) for _ in range(n)
    ]
    return list(score_segment_pairs("t", pairs).per_file)


def test_interval_brackets_the_point_estimate():
    rows = _rows(20)
    ci = bootstrap_der_ci(rows, "full", resamples=500)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.n == 20


def _heterogeneous_rows(n: int = 12):
    """Files of differing difficulty — the only case a resample can move."""
    pairs = [
        ([(0.0, 10.0, "A")], [(0.0, 10.0 - i % 5, "A")]) for i in range(n)
    ]
    return list(score_segment_pairs("t", pairs).per_file)


def test_interval_is_reproducible_from_its_seed():
    """A published interval that moves between two runs is not a number."""
    rows = _heterogeneous_rows()
    a = bootstrap_der_ci(rows, "full", resamples=400, seed=7)
    b = bootstrap_der_ci(rows, "full", resamples=400, seed=7)
    c = bootstrap_der_ci(rows, "full", resamples=400, seed=8)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert (a.lo, a.hi) != (c.lo, c.hi)


def test_interval_on_identical_files_is_degenerate():
    """Files that all score alike leave nothing to resample — width 0, not noise."""
    ci = bootstrap_der_ci(_rows(8), "full", resamples=200)
    assert ci.hi - ci.lo == pytest.approx(0.0, abs=1e-9)


def test_single_file_yields_no_interval():
    ci = bootstrap_der_ci(_rows(1), "full", resamples=200)
    assert ci.resamples == 0 and ci.lo == ci.hi == ci.point


def test_paired_delta_of_a_model_against_itself_is_zero():
    rows = _rows(10)
    delta = paired_bootstrap_delta(rows, rows, "full", resamples=300)
    assert delta.point == pytest.approx(0.0, abs=1e-9)
    assert delta.lo <= 0.0 <= delta.hi


def test_paired_delta_only_uses_files_both_models_scored():
    """An unpaired file cannot inform a paired comparison and is dropped."""
    a = _rows(3)
    b = list(score_segment_pairs(
        "t",
        [([(0.0, 10.0, "A")], [(0.0, 9.0, "A")])],
        file_ids=[a[0].file_id],
    ).per_file)
    delta = paired_bootstrap_delta(a, b, "full", resamples=100)
    assert delta.n == 1


# ── DER by reference speaker count ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, "1"), (2, "2"), (4, "4"), (5, "5+"), (21, "5+"), (0, "5+")],
)
def test_speaker_bucket_labels(n: int, expected: str):
    assert speaker_bucket(n) == expected


def test_buckets_split_by_reference_not_hypothesis_speakers():
    """The axis is how hard the FILE is, which the reference defines."""
    pairs = [
        ([(0.0, 10.0, "A")], [(0.0, 5.0, "X"), (5.0, 10.0, "Y")]),          # ref 1
        ([(0.0, 5.0, "A"), (5.0, 10.0, "B")], [(0.0, 10.0, "X")]),          # ref 2
    ]
    rows = list(score_segment_pairs("t", pairs, file_ids=["a", "b"]).per_file)
    assert [b.bucket for b in der_by_speaker_count(rows, "full")] == ["1", "2"]


def test_empty_buckets_are_omitted_not_zeroed():
    """No file at 5+ speakers means no row — not a model scoring 0 there."""
    rows = list(score_segment_pairs(
        "t", [([(0.0, 10.0, "A")], [(0.0, 10.0, "A")])], file_ids=["a"]
    ).per_file)
    assert [b.bucket for b in der_by_speaker_count(rows, "full")] == ["1"]


def test_buckets_reaggregate_to_the_corpus_number():
    """Every file is in exactly one bucket, so the buckets rebuild the total."""
    pairs = [
        ([(0.0, 20.0, "A")], [(0.0, 18.0, "A")]),
        ([(0.0, 10.0, "A"), (10.0, 20.0, "B")], [(0.0, 20.0, "A")]),
        ([(0.0, 6.0, "A"), (6.0, 12.0, "B"), (12.0, 18.0, "C")],
         [(0.0, 18.0, "A")]),
    ]
    score = score_segment_pairs("t", pairs, file_ids=["a", "b", "c"])
    buckets = der_by_speaker_count(list(score.per_file), "full")
    err = sum(b.der / 100.0 * b.scored_speech_s for b in buckets)
    total = sum(b.scored_speech_s for b in buckets)
    assert err / total * 100.0 == pytest.approx(score.der_full, abs=1e-6)


# ── reference overlap ────────────────────────────────────────────────────────


def test_reference_overlap_on_a_hand_checkable_file(tmp_path: Path):
    """Two speakers, 10 s each, overlapping for 2 s, inside a 18 s extent.

    speech (union) = 18 s, speaker-time = 20 s, overlap = 2 s, wall clock = 18 s.
    """
    gold = _write_rttm(tmp_path / "f1.rttm",
                       [(0.0, 10.0, "A"), (8.0, 18.0, "B")])
    stats = reference_overlap([gold])
    assert stats.overlap_s == pytest.approx(2.0)
    assert stats.speech_s == pytest.approx(18.0)
    assert stats.speaker_time_s == pytest.approx(20.0)
    assert stats.of_speech == pytest.approx(2 / 18 * 100)
    assert stats.of_speaker_time == pytest.approx(2 / 20 * 100)
    assert stats.of_wallclock == pytest.approx(2 / 18 * 100)


def test_reference_overlap_denominators_are_ordered(tmp_path: Path):
    """of_speaker_time < of_speech, always: overlap counts twice below the line."""
    gold = _write_rttm(tmp_path / "f1.rttm",
                       [(0.0, 10.0, "A"), (5.0, 15.0, "B")])
    stats = reference_overlap([gold])
    assert stats.of_speaker_time < stats.of_speech


def test_reference_overlap_of_a_single_speaker_is_zero(tmp_path: Path):
    gold = _write_rttm(tmp_path / "f1.rttm", [(0.0, 5.0, "A"), (6.0, 9.0, "A")])
    assert reference_overlap([gold]).overlap_s == pytest.approx(0.0)


# ── boundary offsets ─────────────────────────────────────────────────────────


def test_perfect_hypothesis_has_zero_boundary_offset(tmp_path: Path):
    segs = [(0.0, 5.0, "A"), (6.0, 11.0, "B")]
    gold = _write_rttm(tmp_path / "gold" / "f1.rttm", segs)
    hyp = _write_rttm(tmp_path / "hyp" / "f1.rttm", segs)
    report = boundary_report([(gold, hyp)])
    assert report.offset_median_ms == pytest.approx(0.0)
    assert report.within_250ms == pytest.approx(100.0)
    assert report.missed_short_s + report.missed_long_s == pytest.approx(0.0)


def test_uniform_shift_shows_up_as_that_shift(tmp_path: Path):
    """A hypothesis late by 100 ms everywhere reads as a 100 ms median offset."""
    gold = _write_rttm(tmp_path / "gold" / "f1.rttm",
                       [(0.0, 5.0, "A"), (6.0, 11.0, "B")])
    hyp = _write_rttm(tmp_path / "hyp" / "f1.rttm",
                      [(0.1, 5.1, "A"), (6.1, 11.1, "B")])
    report = boundary_report([(gold, hyp)])
    assert report.offset_median_ms == pytest.approx(100.0, abs=1e-6)


def test_a_clean_label_swap_is_not_an_error(tmp_path: Path):
    """Diarization is label-permutation invariant, and so is this instrument.

    Naming the first speaker B instead of A is not a mistake — the optimal
    mapping undoes it, exactly as it does inside DER.
    """
    gold = _write_rttm(tmp_path / "gold" / "f1.rttm",
                       [(0.0, 5.0, "A"), (20.0, 25.0, "B")])
    swapped = _write_rttm(tmp_path / "hyp" / "f1.rttm",
                          [(0.0, 5.0, "B"), (20.0, 25.0, "A")])
    assert boundary_report([(gold, swapped)]).offset_median_ms == pytest.approx(0.0)


def test_boundaries_are_credited_per_speaker_not_globally(tmp_path: Path):
    """Boundaries at the right TIMES but on the wrong speaker are not credited.

    A speaker-blind proxy would call this hypothesis perfect: every reference
    boundary time (0, 5, 20, 25) appears in it. Attributing the second turn to
    the first speaker instead leaves the reference's second speaker with no
    counterpart at all, which is what the unmatched count is for.
    """
    gold = _write_rttm(tmp_path / "gold" / "f1.rttm",
                       [(0.0, 5.0, "A"), (20.0, 25.0, "B")])
    one_speaker = _write_rttm(tmp_path / "hyp" / "f1.rttm",
                              [(0.0, 5.0, "A"), (20.0, 25.0, "A")])
    report = boundary_report([(gold, one_speaker)])
    assert report.unmatched_boundaries == 2
    assert report.missed_long_s == pytest.approx(5.0)


def test_missed_speech_is_split_by_reference_segment_length(tmp_path: Path):
    """The instrument behind "the vendor only drops short backchannels".

    One 0.2 s interjection and one 30 s turn, both entirely unrecognised: the
    split must attribute 0.2 s to short and 30 s to long, which is what makes the
    backchannel reading falsifiable.
    """
    gold = _write_rttm(tmp_path / "gold" / "f1.rttm",
                       [(0.0, 30.0, "A"), (40.0, 40.2, "B")])
    hyp = _write_rttm(tmp_path / "hyp" / "f1.rttm", [(60.0, 61.0, "A")])
    report = boundary_report([(gold, hyp)])
    assert report.missed_short_s == pytest.approx(0.2, abs=1e-6)
    assert report.missed_long_s == pytest.approx(30.0, abs=1e-6)
    assert report.missed_short_share < 1.0


# ── the backfill guard ───────────────────────────────────────────────────────


def _artifact(tmp_path: Path, expected: dict) -> Path:
    model = tmp_path / "run" / "diarizer"
    _write_rttm(model / "gold" / "d" / "f1.rttm", [(0.0, 10.0, "A")])
    _write_rttm(model / "hyp" / "d" / "f1.rttm", [(0.0, 8.0, "A")])
    (model / "expected.json").write_text(json.dumps(expected))
    return model


def test_backfill_adds_only_the_missing_fields(tmp_path: Path):
    model = _artifact(tmp_path, {"d": {"der_full": 20.0}})
    added, conflicts = backfill(model)
    assert not conflicts
    written = json.loads((model / "expected.json").read_text())
    assert written["d"]["der_full"] == 20.0          # untouched
    assert set(written["d"]) == set(DerScore.EXPECTED_FIELDS)
    assert "d.der_classic" in added


def test_backfill_refuses_to_overwrite_a_disagreeing_value(tmp_path: Path):
    """The guard: this tool may never be the way a red re-score goes green."""
    model = _artifact(tmp_path, {"d": {"der_full": 99.0}})
    _added, conflicts = backfill(model)
    assert conflicts and any("der_full" in c for c in conflicts)
    # The file on disk is untouched: a conflict aborts the whole write, so the
    # fields that WOULD have been fine are not written either.
    assert json.loads((model / "expected.json").read_text()) == {
        "d": {"der_full": 99.0}
    }


def test_backfill_is_idempotent(tmp_path: Path):
    model = _artifact(tmp_path, {"d": {"der_full": 20.0}})
    backfill(model)
    first = (model / "expected.json").read_text()
    added, conflicts = backfill(model)
    assert not added and not conflicts
    assert (model / "expected.json").read_text() == first


# ── folding sensitivity (#5449) ──────────────────────────────────────────────


def test_already_folded_turns_are_a_fixed_point(tmp_path: Path):
    """The evidence that the shared folding is reconstruction, not tuning.

    A hypothesis that has been through ``spans_to_turns`` does not move when it
    goes through again, so a hosted row's 0.000 pp residue is not a coincidence
    of that corpus — it is the property that lets the step be applied to one side
    of a hosted-vs-local comparison without it being an opinion about the model.
    """
    gold = _write_rttm(tmp_path / "gold" / "f1.rttm",
                       [(0.0, 5.0, "A"), (6.0, 11.0, "B")])
    words = [(0.0, 0.4, "A"), (0.6, 1.0, "A"), (3.0, 5.0, "A"), (6.0, 11.0, "B")]
    hyp = _write_rttm(tmp_path / "hyp" / "f1.rttm",
                      spans_to_turns(words, gap_merge_s=0.5))
    result = folding_sensitivity([(gold, hyp)], "t")
    assert result["delta_classic"] == pytest.approx(0.0, abs=1e-9)
    assert result["delta_full"] == pytest.approx(0.0, abs=1e-9)


def test_unfolded_turns_do_move(tmp_path: Path):
    """A local-shaped hypothesis with sub-gap silences is not a fixed point."""
    gold = _write_rttm(tmp_path / "gold" / "f1.rttm", [(0.0, 10.0, "A")])
    shredded = [(0.0, 1.0, "A"), (1.2, 2.0, "A"), (2.3, 10.0, "A")]
    hyp = _write_rttm(tmp_path / "hyp" / "f1.rttm", shredded)
    result = folding_sensitivity([(gold, hyp)], "t")
    assert result["delta_full"] != pytest.approx(0.0, abs=1e-9)
