"""The scoring contract cannot carry a key nothing reads.

``benchmark.config.yaml`` is published as the rules every number was computed
under. A permissive loader reads a mistyped key as "not set" and the file goes on
claiming a rule nothing applies; the strict schema makes that a load error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from raven_eval_core.contract import CONTRACT_PATH, load_contract, resolved_contract



def test_the_committed_contract_loads_under_the_strict_schema():
    contract = load_contract()
    assert contract.metrics == ["der", "wer", "bleu", "entity"]
    assert [v.name for v in contract.der.variants] == ["full", "classic"]


@pytest.mark.parametrize(
    "path",
    [(), ("der",), ("der", "variants", 0), ("wer", "uncertainty"), ("datasets", "wer", 0)],
    ids=["top-level", "metric-block", "list-item", "nested", "dataset-entry"],
)
def test_the_contract_rejects_an_unknown_key(tmp_path: Path, path):
    doc = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    node = doc
    for step in path:
        node = node[step]
    node["colar"] = 0.25  # the typo a permissive loader reads as "not set"
    tampered = tmp_path / "benchmark.config.yaml"
    tampered.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValidationError, match="colar"):
        load_contract(tampered)


def test_the_contract_rejects_a_missing_key(tmp_path: Path):
    doc = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    del doc["der"]["turn_gap_merge_s"]
    tampered = tmp_path / "benchmark.config.yaml"
    tampered.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValidationError, match="turn_gap_merge_s"):
        load_contract(tampered)


def test_the_resolved_contract_is_plain_data_with_defaults_filled_in():
    """What the run manifest hashes: every key present, optional ones as null."""
    resolved = resolved_contract()
    assert resolved["der"]["turn_gap_merge_s"] == 0.5
    callhome = next(d for d in resolved["datasets"]["der"] if d["id"] == "callhome-de")
    assert callhome["hf"] == "talkbank/callhome" and callhome["metric"] is None
