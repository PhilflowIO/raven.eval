"""DER regression tests on tiny, hand-computable synthetic inputs.

DER = (missed + false_alarm + confusion) / total_reference_speech.
All exact-value cases use ``collar=0.0`` so the arithmetic is hand-verifiable
(the default 0.25 collar removes a forgiveness window around each reference
boundary, which is tested separately for sanity, not exact value).
"""

from __future__ import annotations

import pytest

from raven_eval_core import (
    compute_der,
    compute_der_components,
    compute_der_corpus,
    parse_rttm,
    to_rttm,
)


def test_perfect_match_is_zero():
    ref = [(0.0, 5.0, "A"), (5.0, 10.0, "B")]
    hyp = [(0.0, 5.0, "X"), (5.0, 10.0, "Y")]  # different label names
    assert compute_der(ref, hyp, collar=0.0) == pytest.approx(0.0)


def test_pure_miss():
    # ref: A speaks 0-10 (10 s). hyp: A speaks 0-8 → 2 s missed. DER = 2/10.
    ref = [(0.0, 10.0, "A")]
    hyp = [(0.0, 8.0, "A")]
    assert compute_der(ref, hyp, collar=0.0) == pytest.approx(0.2, abs=1e-6)


def test_pure_confusion():
    # ref: A 0-10, B 10-20 (20 s). hyp: single speaker Z 0-20.
    # optimal map sends Z to one ref speaker → the other 10 s is confusion.
    # DER = 10/20 = 0.5.
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    hyp = [(0.0, 20.0, "Z")]
    assert compute_der(ref, hyp, collar=0.0) == pytest.approx(0.5, abs=1e-6)


def test_pure_false_alarm():
    # ref: A 0-10 (10 s). hyp: A 0-10 AND B 10-15 (5 s of hallucinated speech).
    # false_alarm = 5, DER = 5/10 = 0.5.
    ref = [(0.0, 10.0, "A")]
    hyp = [(0.0, 10.0, "A"), (10.0, 15.0, "B")]
    assert compute_der(ref, hyp, collar=0.0) == pytest.approx(0.5, abs=1e-6)


def test_speaker_permutation_does_not_change_der():
    # Hungarian optimal mapping: swapping hyp label names must not move DER.
    ref = [(0.0, 5.0, "A"), (5.0, 10.0, "B")]
    hyp_a = [(0.0, 5.0, "X"), (5.0, 10.0, "Y")]
    hyp_b = [(0.0, 5.0, "Y"), (5.0, 10.0, "X")]  # names swapped
    der_a = compute_der(ref, hyp_a, collar=0.0)
    der_b = compute_der(ref, hyp_b, collar=0.0)
    assert der_a == pytest.approx(der_b)
    assert der_a == pytest.approx(0.0)

    # And a fully-relabeled but time-identical hyp scores identically.
    hyp_c = [(0.0, 5.0, "999"), (5.0, 10.0, "foo")]
    assert compute_der(ref, hyp_c, collar=0.0) == pytest.approx(der_a)


def test_collar_changes_number_but_stays_sane():
    # Same miss case as test_pure_miss. collar=0.25 removes a forgiveness window
    # around reference boundaries, so the scored number differs from 0.2 but
    # stays a sane fraction.
    ref = [(0.0, 10.0, "A")]
    hyp = [(0.0, 8.0, "A")]
    der_0 = compute_der(ref, hyp, collar=0.0)
    der_25 = compute_der(ref, hyp, collar=0.25)
    assert der_0 == pytest.approx(0.2, abs=1e-6)
    assert der_25 != pytest.approx(0.2)
    assert 0.0 <= der_25 <= 1.0


def test_skip_overlap_parameter_accepted_and_sane():
    # Overlap region: ref A 0-10, B 5-10 (overlapped speech 5-10).
    ref = [(0.0, 10.0, "A"), (5.0, 10.0, "B")]
    hyp = [(0.0, 10.0, "A")]  # B entirely missed
    der_with_overlap = compute_der(ref, hyp, collar=0.0, skip_overlap=False)
    der_skip_overlap = compute_der(ref, hyp, collar=0.0, skip_overlap=True)
    # Excluding overlap regions changes what is scored → different numbers.
    assert der_with_overlap != pytest.approx(der_skip_overlap)
    assert 0.0 <= der_with_overlap <= 2.0
    assert 0.0 <= der_skip_overlap <= 2.0


def test_empty_edge_cases():
    assert compute_der([], [], collar=0.0) == pytest.approx(0.0)
    assert compute_der([], [(0.0, 1.0, "A")], collar=0.0) == pytest.approx(1.0)
    assert compute_der([(0.0, 1.0, "A")], [], collar=0.0) == pytest.approx(1.0)


def test_rttm_roundtrip():
    segments = [(0.0, 3.5, "alice"), (3.5, 7.25, "bob"), (7.25, 10.0, "alice")]
    rttm = to_rttm(segments, file_id="demo")
    parsed = parse_rttm(rttm)
    assert len(parsed) == len(segments)
    for (s0, e0, spk0), (s1, e1, spk1) in zip(segments, parsed):
        assert s1 == pytest.approx(s0, abs=1e-3)
        assert e1 == pytest.approx(e0, abs=1e-3)
        assert spk1 == spk0


def test_components_decompose_der():
    # pure confusion 10s over 20s -> der 0.5, all confusion.
    comp = compute_der_components(
        [(0.0, 10.0, "A"), (10.0, 20.0, "B")], [(0.0, 20.0, "Z")], collar=0.0
    )
    assert comp.der == pytest.approx(0.5, abs=1e-6)
    assert comp.confusion == pytest.approx(0.5, abs=1e-6)
    assert comp.miss == pytest.approx(0.0, abs=1e-6)
    assert comp.false_alarm == pytest.approx(0.0, abs=1e-6)
    assert comp.total_ref == pytest.approx(20.0, abs=1e-6)
    # miss + fa + conf == der exactly (same denominator).
    assert comp.miss + comp.false_alarm + comp.confusion == pytest.approx(comp.der)


def test_components_pure_miss_and_fa():
    miss = compute_der_components([(0.0, 10.0, "A")], [(0.0, 8.0, "A")], collar=0.0)
    assert miss.miss == pytest.approx(0.2, abs=1e-6)
    assert miss.false_alarm == pytest.approx(0.0, abs=1e-6)

    fa = compute_der_components(
        [(0.0, 10.0, "A")], [(0.0, 10.0, "A"), (10.0, 15.0, "B")], collar=0.0
    )
    assert fa.false_alarm == pytest.approx(0.5, abs=1e-6)
    assert fa.miss == pytest.approx(0.0, abs=1e-6)


def test_components_empty_edge_cases():
    assert compute_der_components([], [], collar=0.0).der == pytest.approx(0.0)
    assert compute_der_components([], [(0.0, 1.0, "A")], collar=0.0).der == pytest.approx(1.0)
    empty_hyp = compute_der_components([(0.0, 10.0, "A")], [], collar=0.0)
    assert empty_hyp.der == pytest.approx(1.0)
    assert empty_hyp.miss == pytest.approx(1.0)
    assert empty_hyp.total_ref == pytest.approx(10.0)


def test_corpus_aggregation_is_error_weighted_not_mean():
    # file1: der 0.5 (conf 10/20); file2: der 0.2 (miss 2/10).
    # NIST-correct corpus = (10+2)/(20+10) = 0.4, NOT mean(0.5,0.2)=0.35.
    pairs = [
        ([(0.0, 10.0, "A"), (10.0, 20.0, "B")], [(0.0, 20.0, "Z")]),
        ([(0.0, 10.0, "A")], [(0.0, 8.0, "A")]),
    ]
    corpus = compute_der_corpus(pairs, collar=0.0)
    assert corpus.der == pytest.approx(0.4, abs=1e-6)
    assert corpus.total_ref == pytest.approx(30.0, abs=1e-6)
    assert corpus.miss + corpus.false_alarm + corpus.confusion == pytest.approx(corpus.der)


def test_corpus_empty_is_zero():
    assert compute_der_corpus([], collar=0.0).der == pytest.approx(0.0)
    assert compute_der_corpus([([], [])], collar=0.0).der == pytest.approx(0.0)


def test_parse_rttm_ignores_non_speaker_lines():
    rttm = (
        "; a comment\n"
        "SPKR-INFO demo 1 <NA> <NA> <NA> unknown alice <NA> <NA>\n"
        "SPEAKER demo 1 0.000 2.500 <NA> <NA> alice <NA> <NA>\n"
        "\n"
        "SPEAKER demo 1 2.500 2.500 <NA> <NA> bob <NA> <NA>\n"
    )
    parsed = parse_rttm(rttm)
    assert parsed == [(0.0, 2.5, "alice"), (2.5, 5.0, "bob")]
