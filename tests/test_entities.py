"""Numeric entity hit rate (raven_eval_core.entities).

The "a wrong number is never a hit" guarantee is a contract and lives in
tests/test_metric_contract.py; this file pins the definition around it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raven_asr.analysis import entity_interval, paired_entity_delta
from raven_eval_core.bootstrap import ratio_pct
from raven_eval_core.entities import (
    entity_hit_rate,
    entity_units,
    extract_numeric_entities,
    utterance_entity_hits,
)
from raven_eval_core.flozi_wer import (
    normalize_flozi,
    normalize_flozi_before_punctuation_strip,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


@pytest.mark.parametrize(
    ("text", "entities"),
    [
        ("keine Zahl hier", []),
        ("es kamen dreiundzwanzig Leute", ["23"]),
        ("3,5 Prozent", ["3,5"]),                      # decimal comma stays one entity
        ("drei komma fünf Prozent", ["3,5"]),
        ("0,5 Liter", ["0,5"]),
        ("am 12.03.2024 um 10:30 Uhr", ["12.03.2024", "10:30"]),
        ("am 3. Mai", ["3"]),                          # ordinal dot is not a separator
        ("am dritten Mai", ["3"]),
        ("1.000 Euro", ["1000"]),                      # German thousands grouping
        ("2.500.000 Einwohner", ["2500000"]),
        ("tausend Euro", ["1000"]),
        ("5 % und 5 Prozent", ["5", "5"]),             # the unit is WER's business
        ("€ 5", ["5"]),
        ("2-3 Tage", ["2", "3"]),                      # a hyphen separates
        ("die 90er Jahre", ["90"]),
        ("[Musik] 12 Leute", ["12"]),                  # flozi's bracket strip applies
    ],
)
def test_extraction(text: str, entities: list[str]) -> None:
    assert extract_numeric_entities(text) == entities


def test_normalize_flozi_is_the_early_stage_plus_the_strip() -> None:
    """One normalization, read at two points — not a second one."""
    text = "Am 12.03.2024 kamen dreiundzwanzig Leute, d.h. 3,5 % [lacht]"
    early = normalize_flozi_before_punctuation_strip(text)
    assert "12.03.2024" in early and "3,5" in early
    assert normalize_flozi(text) == "Am 12032024 kamen 23 Leute dh 35"


def test_multiset_an_entity_twice_must_appear_twice() -> None:
    (hits, n), = utterance_entity_hits(["3 mal 3 ist 9"], ["3 mal ist 9"])
    assert (hits, n) == (2, 3)


def test_an_extra_number_in_the_hypothesis_is_not_a_hit() -> None:
    (hits, n), = utterance_entity_hits(["es waren 5"], ["es waren 5 oder 5 oder 6"])
    assert (hits, n) == (1, 1)


def test_order_within_an_utterance_does_not_matter() -> None:
    (hits, n), = utterance_entity_hits(["von 3 bis 7"], ["7 bis 3"])
    assert (hits, n) == (2, 2)


def test_decimals_are_matched_whole() -> None:
    refs = ["1,5 Liter", "1,5 Liter", "1,5 Liter"]
    hyps = ["eins komma fünf Liter", "1,6 Liter", "15 Liter"]
    assert [h for h, _ in utterance_entity_hits(refs, hyps)] == [1, 0, 0]


def test_utterances_without_entities_contribute_nothing() -> None:
    refs = ["guten Morgen", "es kamen 23 Leute", "auf Wiedersehen", "um 10 Uhr"]
    hyps = ["guten Abend 7", "es kamen 22 Leute", "tschüss", "um 10 Uhr"]
    assert utterance_entity_hits(refs, hyps) == [(0, 0), (0, 1), (0, 0), (1, 1)]
    assert entity_units(refs, hyps) == [(0, 1), (1, 1)]
    result = entity_hit_rate(refs, hyps)
    assert result.hit_rate_pct == 50.0
    assert (result.n_hits, result.n_entities, result.n_utterances_with_entities) == (1, 2, 2)


def test_a_subset_without_numbers_has_no_hit_rate() -> None:
    """0 % would claim every number was wrong; there were none."""
    result = entity_hit_rate(["guten Morgen"], ["guten Morgen 5"])
    assert result.hit_rate_pct is None
    assert (result.n_entities, result.n_utterances_with_entities) == (0, 0)


def test_misaligned_inputs_raise() -> None:
    with pytest.raises(ValueError, match="must align"):
        utterance_entity_hits(["a", "b"], ["a"])


# ── The interval: around the printed hit rate, over the utterances it is made of ──


def _pred(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_published_hit_rate_decomposes_exactly_per_utterance():
    """Σhits/Σentities over the interval's units IS the hit rate, on every artifact."""
    paths = sorted(ARTIFACTS.rglob("predictions_*.jsonl"))
    assert paths
    for p in paths:
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
        refs = [r["reference"] for r in rows]
        hyps = [r["prediction"] for r in rows]
        result = entity_hit_rate(refs, hyps)
        if result.hit_rate_pct is not None:
            assert ratio_pct(entity_units(refs, hyps)) == result.hit_rate_pct


def test_interval_resamples_only_utterances_with_an_entity(tmp_path: Path):
    rows = [{"sample_id": f"s{i}", "reference": "es waren 23" if i < 4 else "hallo",
             "prediction": "es waren 23" if i % 2 else "es waren 22"}
            for i in range(40)]
    ci = entity_interval(_pred(tmp_path, "a", rows), resamples=500)
    assert ci is not None
    assert ci.n == 4
    assert ci.point == 50.0
    assert ci.lo <= ci.point <= ci.hi


def test_no_interval_without_entities(tmp_path: Path):
    p = _pred(tmp_path, "a", [{"sample_id": "s0", "reference": "hallo", "prediction": "5"}])
    assert entity_interval(p, resamples=50) is None


def test_paired_entity_delta_against_itself_is_zero(tmp_path: Path):
    rows = [{"sample_id": f"s{i}", "reference": f"es waren {i + 3}",
             "prediction": f"es waren {i + 3 + i % 2}"} for i in range(10)]
    p = _pred(tmp_path, "a", rows)
    d = paired_entity_delta(p, p, resamples=200)
    assert d is not None
    assert d.point == 0.0 and d.lo <= 0.0 <= d.hi


def test_paired_entity_delta_refuses_different_references(tmp_path: Path):
    a = _pred(tmp_path, "a", [{"sample_id": "s0", "reference": "es waren 3",
                               "prediction": "es waren 3"}])
    b = _pred(tmp_path, "b", [{"sample_id": "s0", "reference": "es waren 4",
                               "prediction": "es waren 4"}])
    with pytest.raises(ValueError, match="different references"):
        paired_entity_delta(a, b, resamples=50)
