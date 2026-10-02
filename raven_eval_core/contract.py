"""Strict loader for the scoring contract (``benchmark.config.yaml``).

The contract is published so a third party sees the exact rules a number was
computed under. A key it does not expect is therefore never harmless: a typo
(``colar: 0.25``) would be read as "no collar given" by anything that looks the
real key up, and the file would go on claiming a rule nothing applies. Every
model here forbids unknown keys, so such a file fails to load instead.

This is what a Tier-2 run resolves its config from; the hash recorded in the run
manifest (``raven_eval_core.run_manifest``) covers the model returned here plus
the harness's own model/dataset pins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict

CONTRACT_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent / "benchmark.config.yaml"
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Uncertainty(_Strict):
    method: str
    resamples: int
    seed: int
    confidence: float
    comparisons: str


class DerVariant(_Strict):
    name: str
    collar: float
    skip_overlap: bool


class DerAggregation(_Strict):
    primary: str
    also_reported: list[str]


class DerContract(_Strict):
    scorer_primary: str
    variants: list[DerVariant]
    # Keyed by diarizer label; the inner keys are that adapter's own settings
    # (raven_diar adapters' ``run_config``), which only the adapter can name.
    model_configuration: dict[str, dict[str, str | int | float | bool]]
    speaker_mapping: str
    vad: str
    aggregation: DerAggregation
    uncertainty: Uncertainty
    turn_gap_merge_s: float


class WerContract(_Strict):
    scorer: str
    normalization: str
    alignment: str
    also_reported: list[str]
    uncertainty: Uncertainty


class BleuVariant(_Strict):
    name: str
    tokenize: str
    lowercase: bool
    smooth_method: str
    effective_order: bool
    max_ngram_order: int


class BleuContract(_Strict):
    scorer: str
    variants: list[BleuVariant]
    aggregation: str
    signature: str
    also_reported: list[str]


class EntityContract(_Strict):
    scorer: str
    normalization: str
    entity: str
    matching: str
    aggregation: str
    uncertainty: Uncertainty


class DatasetEntry(_Strict):
    id: str
    license: str
    language_tag: str
    variety_label: str
    # Required but nullable: "the source documents no locality" has to be stated.
    locality: str | None
    hf: str | None = None
    config: str | None = None
    zenodo: str | None = None
    doi: str | None = None
    version: str | None = None
    metric: str | None = None
    acquisition: str | None = None
    representativeness: str | None = None
    eligible_for_aggregate: bool | None = None


class Datasets(_Strict):
    der: list[DatasetEntry]
    wer: list[DatasetEntry]


class DialectRegionBreakdown(_Strict):
    datasets: list[str]
    region: str
    source_of_region: str
    min_n: int
    wer: str
    bleu: str
    coverage: str


class BenchmarkContract(_Strict):
    metrics: list[str]
    der: DerContract
    wer: WerContract
    bleu: BleuContract
    entity: EntityContract
    datasets: Datasets
    dialect_publication_rules: list[str]
    dialect_region_breakdown: DialectRegionBreakdown


def load_contract(path: Path = CONTRACT_PATH) -> BenchmarkContract:
    """Parse and validate the contract; raises on any unknown or missing key."""
    return BenchmarkContract.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def resolved_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """The validated contract as plain JSON-able data, defaults filled in."""
    return load_contract(path).model_dump(mode="json")
