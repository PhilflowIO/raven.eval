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

# Top-level config keys that are not metric blocks.
NON_METRIC_KEYS = frozenset({"datasets"})


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _declared_metrics() -> set[str]:
    return set(_config()) - set(NON_METRIC_KEYS)


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
    module_to_metric = {"der": "der", "wer": "wer", "flozi_wer": "wer", "bleu": "bleu"}
    unmapped = sorted(on_disk - {"__init__"} - set(module_to_metric))
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
