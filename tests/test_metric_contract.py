"""The scoring contract and the metric core must agree — both ways.

``tests/test_dataset_contract.py`` does this for the *dataset* list. This file
does it for the *metrics* themselves, which is the other half of the same
promise: ``benchmark.config.yaml`` is published so "third parties see the precise
rules our values were computed under", and that only holds if the declared rules
are the rules the code actually applies.

Two directions, both failing the build:

  * a metric block in the config with no implementation in ``raven_eval_core`` →
    the contract advertises a number nobody can compute;
  * a metric implemented in ``raven_eval_core`` and absent from the config → a
    number published under rules the committed contract does not describe.

Plus the sharper third check that motivated this file: for BLEU, the config does
not merely *name* the metric, it pins the conventions (tokenizer, case, smoothing,
n-gram order) that move the score by whole points. Those are asserted field-by-
field against the module constants, so editing one side alone turns CI red.

Pure and offline: reads one YAML file and imports the metric core. No downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from raven_eval_core import SCORED_METRICS  # noqa: E402
from raven_eval_core import bleu as bleu_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "benchmark.config.yaml"

def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _declared_metrics() -> set[str]:
    """The metric blocks the contract declares, read from its own `metrics:` list.

    This used to be "every top-level key except `datasets`". That inverted the
    burden of proof: any new top-level block — publication rules, provenance
    notes — silently became a metric that these tests then demanded an
    implementation for, and the failure surfaced far from its cause. Naming the
    metrics explicitly makes an unlisted block simply not a metric, and
    :func:`test_every_declared_metric_has_a_config_block` keeps the list from
    naming something that does not exist.
    """
    return set(_config()["metrics"])


def test_every_declared_metric_has_a_config_block() -> None:
    config = _config()
    missing = sorted(name for name in config["metrics"] if name not in config)
    assert not missing, (
        f"benchmark.config.yaml lists metric(s) {missing} under `metrics:` that "
        f"have no block of their own — the contract names rules it does not state."
    )


def test_non_metric_blocks_are_not_mistaken_for_metrics() -> None:
    """A new top-level block must not become a metric by being written down."""
    config = _config()
    assert "dialect_publication_rules" in config, (
        "expected a non-metric top-level block to exist, so this test actually "
        "exercises the distinction it is guarding"
    )
    assert "dialect_publication_rules" not in _declared_metrics()


def test_every_declared_metric_is_implemented() -> None:
    orphaned = sorted(_declared_metrics() - SCORED_METRICS)
    assert not orphaned, (
        f"benchmark.config.yaml declares metric block(s) {orphaned} that "
        f"raven_eval_core.SCORED_METRICS does not implement — the contract "
        f"advertises a number nobody can compute. Implement the scorer or drop "
        f"the block."
    )


def test_every_implemented_metric_is_declared() -> None:
    undocumented = sorted(SCORED_METRICS - _declared_metrics())
    assert not undocumented, (
        f"metric(s) {undocumented} are implemented in raven_eval_core but have no "
        f"block in benchmark.config.yaml — a number measured with them would be "
        f"published under rules the committed contract does not describe."
    )


def test_every_metric_module_is_registered() -> None:
    """A scorer module on disk that nobody declared is invisible drift."""
    on_disk = {p.stem for p in (REPO_ROOT / "raven_eval_core").glob("*.py")}
    # flozi_wer.py is the published WER path, wer.py the diagnostic lens: two
    # modules, one declared metric. Map the module names onto metric names.
    module_to_metric = {
        "der": "der", "wer": "wer", "flozi_wer": "wer", "bleu": "bleu",
        "entities": "entity",
    }
    # Not a metric of its own: the resampler behind `der.uncertainty` and
    # `wer.uncertainty`, whose settings those blocks declare.
    shared_by_metrics = {"bootstrap"}
    # Not scorers at all: the strict loader for the contract this test reads,
    # the record of what produced a run, and the reader of the `systems:` block
    # (training-data disclosure per system). None computes a number.
    not_scorers = {"contract", "run_manifest", "systems"}
    unmapped = sorted(
        on_disk - {"__init__"} - set(module_to_metric) - shared_by_metrics
        - not_scorers
    )
    assert not unmapped, (
        f"scorer module(s) {unmapped} exist under raven_eval_core/ but map to no "
        f"metric — add them to module_to_metric here (and to SCORED_METRICS + "
        f"benchmark.config.yaml if they are a new metric)."
    )
    assert set(module_to_metric.values()) == SCORED_METRICS


# ── BLEU conventions: the config restates them, the module applies them ──────


def _bleu_published_variant() -> dict:
    variants = _config()["bleu"]["variants"]
    published = [v for v in variants if v["name"] == "published"]
    assert len(published) == 1, (
        f"bleu.variants must contain exactly one 'published' variant, found "
        f"{[v['name'] for v in variants]}."
    )
    return published[0]


@pytest.mark.parametrize(
    ("config_field", "constant_name"),
    [
        ("tokenize", "BLEU_TOKENIZE"),
        ("lowercase", "BLEU_LOWERCASE"),
        ("smooth_method", "BLEU_SMOOTH_METHOD"),
        ("effective_order", "BLEU_EFFECTIVE_ORDER"),
        ("max_ngram_order", "BLEU_MAX_NGRAM_ORDER"),
    ],
)
def test_bleu_convention_matches_the_implementation(
    config_field: str, constant_name: str
) -> None:
    declared = _bleu_published_variant()[config_field]
    implemented = getattr(bleu_mod, constant_name)
    assert declared == implemented, (
        f"benchmark.config.yaml declares bleu.{config_field}={declared!r} but "
        f"raven_eval_core.bleu.{constant_name}={implemented!r}. Every one of these "
        f"moves BLEU by whole points; the published number and the contract must "
        f"not describe different scorers."
    )


def test_bleu_is_declared_corpus_level() -> None:
    """Published BLEU is corpus-level; the sentence score is a diagnostic only."""
    block = _config()["bleu"]
    assert block["aggregation"] == "corpus"
    assert block["scorer"] == "sacrebleu"
    assert "sentence" in block["also_reported"]


def test_bleu_signature_reflects_the_declared_conventions() -> None:
    """The string we publish next to a BLEU number encodes the pinned rules."""
    variant = _bleu_published_variant()
    sig = bleu_mod.bleu_signature()
    assert f"tok:{variant['tokenize']}" in sig
    assert f"smooth:{variant['smooth_method']}" in sig
    assert ("case:mixed" if not variant["lowercase"] else "case:lc") in sig
    assert ("eff:no" if not variant["effective_order"] else "eff:yes") in sig
    assert "version:" in sig, "the signature must carry the sacrebleu version"


def test_wer_uncertainty_matches_the_public_contract() -> None:
    """Every published WER interval is computed under the declared settings."""
    from raven_asr.config import (
        BOOTSTRAP_CONFIDENCE,
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
    )

    unc = _config()["wer"]["uncertainty"]
    assert unc["resamples"] == BOOTSTRAP_RESAMPLES
    assert unc["seed"] == BOOTSTRAP_SEED
    assert unc["confidence"] == BOOTSTRAP_CONFIDENCE


def test_region_breakdown_matches_the_public_contract() -> None:
    """``dialect_region_breakdown`` names the corpora and the floor the code applies."""
    from raven_asr.analysis import REGION_PARSERS
    from raven_asr.config import REGION_MIN_N

    block = _config()["dialect_region_breakdown"]
    assert block["min_n"] == REGION_MIN_N
    assert set(block["datasets"]) == set(REGION_PARSERS)
    assert "dialect_region_breakdown" not in _declared_metrics()


# ── Training-data disclosure: the `systems:` block ───────────────────────────


def _raw_system_entries() -> list[tuple[str, dict]]:
    """Every full entry of the raw block, read without the parser under test."""
    return [
        (f"systems.{family}.{key}", entry)
        for family, entries in _config()["systems"].items()
        for key, entry in entries.items()
        if "same_as" not in entry
    ]


def test_systems_block_is_not_a_metric() -> None:
    assert "systems" in _config()
    assert "systems" not in _declared_metrics()


def test_systems_statuses_come_from_the_allowed_set() -> None:
    """Read from the raw YAML, so a parser bug cannot wave a value through."""
    from raven_eval_core.systems import CORPUS_SEEN_VALUES, TRAINING_DATA_STATUSES

    assert TRAINING_DATA_STATUSES == ("offengelegt", "teilweise", "nicht offengelegt")
    assert CORPUS_SEEN_VALUES == ("ja", "nein", "unbekannt")
    entries = _raw_system_entries()
    assert entries, "the `systems:` block is empty"
    offenders = []
    for where, entry in entries:
        if entry["training_data"]["status"] not in TRAINING_DATA_STATUSES:
            offenders.append(f"{where}: status {entry['training_data']['status']!r}")
        for dataset, seen in entry["test_corpus_seen"].items():
            if seen["seen"] not in CORPUS_SEEN_VALUES:
                offenders.append(f"{where}.{dataset}: seen {seen['seen']!r}")
    assert not offenders, offenders


def test_every_ja_and_nein_carries_a_source_and_a_quote() -> None:
    """`ja` and `nein` are claims about a vendor; an unsourced one is a guess."""
    offenders, claims = [], 0
    for where, entry in _raw_system_entries():
        for dataset, seen in entry["test_corpus_seen"].items():
            if seen["seen"] == "unbekannt":
                continue
            claims += 1
            if not str(seen.get("source", "")).startswith("https://") or not seen.get("quote"):
                offenders.append(f"{where}.test_corpus_seen.{dataset}")
    assert claims, "no `ja`/`nein` in the block — this test would guard nothing"
    assert not offenders, f"`ja`/`nein` without source URL + verbatim quote: {offenders}"


def test_every_system_entry_names_what_was_checked_and_when() -> None:
    import datetime

    for where, entry in _raw_system_entries():
        training = entry["training_data"]
        assert isinstance(training["checked"], datetime.date), where
        assert training["sources"], f"{where}: no page was recorded as checked"
        assert all(s["url"].startswith("https://") for s in training["sources"]), where


def test_systems_block_validates_against_the_registries() -> None:
    """Keys are registry keys and corpora are dataset selectors — no second list."""
    from raven_asr.config import KNOWN_MODELS
    from raven_diar.config import KNOWN_DIARIZERS
    from test_published_table import published_systems

    systems = published_systems()
    assert set(systems["wer"]) <= set(KNOWN_MODELS)
    assert set(systems["der"]) <= set(KNOWN_DIARIZERS)
    # An alias resolves to the entry of the checkpoint it serves.
    alias = systems["wer"]["modal/parakeet"]
    assert alias.same_as == "primeline/parakeet-primeline"
    assert alias.status == systems["wer"][alias.same_as].status


def _parse(block: dict):
    from raven_eval_core.systems import parse_systems

    return parse_systems(
        block,
        known_systems={"wer": {"a/model", "served/elsewhere", "other/model"}, "der": {"diar"}},
        known_corpora={"wer": {"fleurs"}, "der": {"ami"}},
        same_checkpoint={
            "wer": {"a/model": "ckpt", "served/elsewhere": "ckpt", "other/model": "x"},
        },
    )


def _entry(**seen) -> dict:
    import datetime

    return {
        "training_data": {
            "status": "nicht offengelegt",
            "checked": datetime.date(2026, 10, 2),
            "sources": [{"url": "https://example.org/card"}],
        },
        "test_corpus_seen": {"fleurs": seen or {"seen": "unbekannt"}},
    }


def test_the_parser_accepts_a_minimal_entry_and_an_alias() -> None:
    parsed = _parse({"wer": {"a/model": _entry(), "served/elsewhere": {"same_as": "a/model"}}})
    assert parsed["wer"]["a/model"].corpus_seen("fleurs").seen == "unbekannt"
    assert parsed["wer"]["served/elsewhere"].same_as == "a/model"
    with pytest.raises(KeyError):  # absent is an error, never a silent `unbekannt`
        parsed["wer"]["a/model"].corpus_seen("mls-de")


@pytest.mark.parametrize(
    ("block", "complaint"),
    [
        ({"wer": {"not/registered": _entry()}}, "not a registry key"),
        ({"wer": {"a/model": _entry(seen="ja")}}, "needs a `source`"),
        ({"wer": {"a/model": _entry(seen="nein", source="https://example.org")}}, "needs a `source`"),
        ({"wer": {"a/model": _entry(seen="vielleicht")}}, "must be one of"),
        ({"wer": {"a/model": {**_entry(), "test_corpus_seen": {"ami": {"seen": "unbekannt"}}}}},
         "not a WER dataset"),
        ({"wer": {"a/model": _entry(), "other/model": {"same_as": "a/model"}}},
         "does not bind both keys to one checkpoint"),
        ({"asr": {}}, "unknown family"),
    ],
)
def test_the_parser_rejects_what_the_contract_forbids(block: dict, complaint: str) -> None:
    from raven_eval_core.systems import SystemsContractError

    with pytest.raises(SystemsContractError, match=complaint):
        _parse(block)


# ── Numeric entities: a wrong number must never be scored as the right one ────


def test_entity_conventions_match_the_implementation() -> None:
    from raven_eval_core import entities

    block = _config()["entity"]
    assert block["normalization"] == entities.ENTITY_NORMALIZATION
    assert block["matching"] == entities.ENTITY_MATCHING
    assert block["aggregation"] == "corpus"


def test_entity_uncertainty_matches_the_public_contract() -> None:
    from raven_asr.config import (
        BOOTSTRAP_CONFIDENCE,
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
    )

    unc = _config()["entity"]["uncertainty"]
    assert unc["resamples"] == BOOTSTRAP_RESAMPLES
    assert unc["seed"] == BOOTSTRAP_SEED
    assert unc["confidence"] == BOOTSTRAP_CONFIDENCE


def _hit_rate(reference: str, prediction: str) -> float | None:
    from raven_eval_core.entities import entity_hit_rate

    return entity_hit_rate([reference], [prediction]).hit_rate_pct


@pytest.mark.parametrize(
    ("reference", "prediction"),
    [
        # The misrecognition the score exists to catch: one digit off.
        ("es kamen dreiundzwanzig Leute", "es kamen zweiundzwanzig Leute"),
        ("es kamen 23 Leute", "es kamen zweiundzwanzig Leute"),
        ("die Rate liegt bei 3,5 Prozent", "die Rate liegt bei 3,6 Prozent"),
        # flozi's punctuation strip would make each of these pairs one string:
        # "3,5" and "35", "2-3" and "23", "12.03." and "1203", "1:0" and "10".
        ("die Rate liegt bei 3,5 Prozent", "die Rate liegt bei 35 Prozent"),
        ("die Rate liegt bei 35 Prozent", "die Rate liegt bei 3,5 Prozent"),
        ("es waren 23 Leute", "es waren 2-3 Leute"),
        ("am 1203 eingereicht", "am 12.03. eingereicht"),
        ("das Spiel endete 10", "das Spiel endete 1:0"),
        ("um 1030 Uhr", "um 10:30 Uhr"),
        ("am 12.03.2024", "am 12.03.2023"),
    ],
)
def test_a_different_number_is_a_miss(reference: str, prediction: str) -> None:
    """No normalization step may map a misrecognized number onto the reference's."""
    assert _hit_rate(reference, prediction) == 0.0


@pytest.mark.parametrize(
    ("reference", "prediction"),
    [
        ("es kamen 23 Leute", "es kamen dreiundzwanzig Leute"),
        ("es kamen dreiundzwanzig Leute", "es kamen 23 Leute"),
        ("die Rate liegt bei 3,5 %", "die Rate liegt bei drei komma fünf Prozent"),
        ("es kostet 1.000 Euro", "es kostet tausend Euro"),
        ("im Jahr 1990", "im Jahr neunzehnhundertneunzig"),
    ],
)
def test_the_same_number_in_another_form_is_a_hit(reference: str, prediction: str) -> None:
    assert _hit_rate(reference, prediction) == 100.0


def test_spoken_numbers_map_onto_their_own_digits_and_no_other() -> None:
    """The guarantee behind every miss above, over the whole everyday range.

    A hit is exact string equality of independently normalized texts, so a
    wrong number could only score if two different spoken numbers normalized to
    the same digits. ``alpha2digit`` is the one step that maps words onto
    digits; this pins it as injective, number by number. 0, 1 and 2 stay words
    in flozi's normalization (``alpha2digit``'s threshold) and so carry no
    entity — a miss against a digit, never a false hit.
    """
    from num2words import num2words
    from raven_eval_core.entities import extract_numeric_entities

    for n in range(3, 3000):
        spoken = num2words(n, lang="de")
        assert extract_numeric_entities(f"es waren {spoken} Leute") == [str(n)], spoken
    for n in range(3):
        assert extract_numeric_entities(num2words(n, lang="de")) == []


def test_reference_and_hypothesis_are_normalized_independently() -> None:
    """The hypothesis is never read in the light of the reference, or vice versa."""
    from raven_eval_core.entities import extract_numeric_entities, utterance_entity_hits

    ref, hyp = "zweiundzwanzig und 3,5", "dreiundzwanzig und 35"
    (hits, n), = utterance_entity_hits([ref], [hyp])
    assert (hits, n) == (0, 2)
    assert extract_numeric_entities(ref) == ["22", "3,5"]
    assert extract_numeric_entities(hyp) == ["23", "35"]
