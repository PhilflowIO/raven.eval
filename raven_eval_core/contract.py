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

import datetime as _dt
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


# ── Training-data disclosure per benchmarked system: the `systems:` block ────
#
# What each vendor documents about its training data, and whether that includes
# the corpora the system is scored on here. The values are published beside
# every result row, so what they mean is part of the contract:
#
#   status   offengelegt        the vendor names the training corpora
#            teilweise          part of the lineage is documented, part is not
#            nicht offengelegt  closed API, or a card that names no corpus
#   seen     ja         documented training on that corpus — its train split or
#                       the same source. NOT a claim that the test recordings
#                       themselves were trained on.
#            nein       the vendor states the training set as a closed list and
#                       the corpus is not in it
#            unbekannt  nothing documented either way. Never a guess: an
#                       open-ended list ("including …") cannot support `nein`.
#
# Open weights settle none of this — they ship without their training data and
# the data cannot be read back out of them. The admissible sources are model
# cards, papers and dataset documentation, which is why `ja` and `nein` must
# carry a source URL and a verbatim quote.

TrainingDataStatus = Literal["offengelegt", "teilweise", "nicht offengelegt"]
CorpusSeenValue = Literal["ja", "nein", "unbekannt"]
#: The metric families of ``datasets:`` — a system belongs to exactly one.
SYSTEM_FAMILIES: Final[tuple[str, ...]] = ("wer", "der")


class Source(_Strict):
    """One page that was read, and what on it supports the claim."""

    url: str = Field(pattern=r"^https://")
    #: Verbatim from the page. None only where the finding is an absence — a
    #: page that says nothing about training data has nothing to quote.
    quote: str | None = None
    note: str | None = None


class TrainingData(_Strict):
    status: TrainingDataStatus
    checked: _dt.date
    # Even `nicht offengelegt` names the page that was checked.
    sources: list[Source] = Field(min_length=1)
    note: str | None = None

    @model_validator(mode="after")
    def _a_disclosure_is_quoted(self) -> TrainingData:
        if self.status != "nicht offengelegt" and not any(s.quote for s in self.sources):
            raise ValueError(
                f"status {self.status!r} asserts a disclosure but no source "
                f"carries a verbatim quote"
            )
        return self


class CorpusSeen(_Strict):
    """Whether a system's documented training data includes one test corpus."""

    seen: CorpusSeenValue
    source: str | None = Field(default=None, pattern=r"^https://")
    quote: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _a_claim_is_evidenced(self) -> CorpusSeen:
        if self.seen != "unbekannt" and not (self.source and self.quote):
            raise ValueError(
                f"`{self.seen}` is a claim and needs a `source` URL plus a "
                f"verbatim `quote`; without one it is `unbekannt`"
            )
        return self


class SystemEntry(_Strict):
    """Either a full disclosure, or `same_as` another key of the same family."""

    training_data: TrainingData | None = None
    test_corpus_seen: dict[str, CorpusSeen] | None = None
    #: This registry key serves the same checkpoint as the named key and shares
    #: its entry. Checked against the registry by :func:`resolve_systems`.
    same_as: str | None = None

    @model_validator(mode="after")
    def _one_shape(self) -> SystemEntry:
        full = self.training_data is not None and bool(self.test_corpus_seen)
        partial = self.training_data is not None or self.test_corpus_seen is not None
        if self.same_as is not None and partial:
            raise ValueError("`same_as` stands alone")
        if self.same_as is None and not full:
            raise ValueError(
                "needs `training_data` and a non-empty `test_corpus_seen`, or `same_as`"
            )
        return self


class Systems(_Strict):
    # Keyed by registry key: raven_asr.config.KNOWN_MODELS under `wer`,
    # raven_diar.config.KNOWN_DIARIZERS under `der`.
    wer: dict[str, SystemEntry]
    der: dict[str, SystemEntry]


class BenchmarkContract(_Strict):
    metrics: list[str]
    der: DerContract
    wer: WerContract
    bleu: BleuContract
    entity: EntityContract
    datasets: Datasets
    dialect_publication_rules: list[str]
    dialect_region_breakdown: DialectRegionBreakdown
    systems: Systems


def load_contract(path: Path = CONTRACT_PATH) -> BenchmarkContract:
    """Parse and validate the contract; raises on any unknown or missing key."""
    return BenchmarkContract.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def resolved_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """The validated SCORING contract as plain JSON-able data, defaults filled in.

    ``systems`` is left out. This is what a run manifest hashes as the rules a
    number was computed under, and what a vendor's model card says about
    training data is not such a rule: re-reading a card must not change the
    config hash of every run.
    """
    return load_contract(path).model_dump(mode="json", exclude={"systems"})


class SystemsContractError(ValueError):
    """`systems:` disagrees with the code registries. Carries every violation."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(
            "benchmark.config.yaml `systems:` is invalid:\n  " + "\n  ".join(problems)
        )


@dataclass(frozen=True)
class SystemDisclosure:
    """One registry key's disclosure, with a `same_as` alias already followed."""

    family: str
    key: str
    training_data: TrainingData
    test_corpus_seen: Mapping[str, CorpusSeen]
    same_as: str | None = None

    @property
    def status(self) -> str:
        return self.training_data.status

    def corpus_seen(self, dataset: str) -> CorpusSeen:
        """The recorded value for ``dataset`` — never a default.

        A missing entry is an error rather than ``unbekannt``: "nobody looked"
        and "we looked and the vendor says nothing" are different facts, and
        only the second may be published.
        """
        try:
            return self.test_corpus_seen[dataset]
        except KeyError:
            raise KeyError(
                f"systems.{self.family}.{self.key} has no test_corpus_seen entry "
                f"for {dataset!r} (has: {sorted(self.test_corpus_seen)})"
            ) from None


def resolve_systems(
    systems: Systems,
    *,
    known_systems: Mapping[str, Collection[str]],
    known_corpora: Mapping[str, Collection[str]],
    same_checkpoint: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, dict[str, SystemDisclosure]]:
    """Bind the ``systems:`` block to the code registries.

    The schema above checks the block's shape; this checks that it names only
    things that exist. ``known_systems`` and ``known_corpora`` map each family
    (``wer`` / ``der``) to the registry keys and dataset selectors in code, so
    the block cannot grow into a second model list. ``same_checkpoint`` maps
    each family's registry keys to the checkpoint they serve and is what a
    ``same_as`` alias is checked against.

    The registries are arguments because this package is the dependency-light
    core and imports neither harness. Raises :class:`SystemsContractError`
    listing every violation at once.
    """
    problems: list[str] = []
    out: dict[str, dict[str, SystemDisclosure]] = {}
    for family in SYSTEM_FAMILIES:
        entries: dict[str, SystemEntry] = getattr(systems, family)
        out[family] = {}
        for key, entry in entries.items():
            where = f"systems.{family}.{key}"
            if key not in known_systems[family]:
                problems.append(
                    f"{where}: not a registry key — add the system to the "
                    f"{family.upper()} registry first, this block lists no models "
                    f"of its own"
                )
            elif entry.same_as is None:
                foreign = sorted(set(entry.test_corpus_seen) - set(known_corpora[family]))
                if foreign:
                    problems.append(
                        f"{where}.test_corpus_seen: {foreign} not a {family.upper()} "
                        f"dataset of this repo"
                    )
                out[family][key] = SystemDisclosure(
                    family, key, entry.training_data, entry.test_corpus_seen
                )
        checkpoints = (same_checkpoint or {}).get(family, {})
        for key, entry in entries.items():
            if entry.same_as is None or key not in known_systems[family]:
                continue
            where, target = f"systems.{family}.{key}", out[family].get(entry.same_as)
            if target is None or target.same_as is not None:
                problems.append(
                    f"{where}: same_as {entry.same_as!r} is not a full entry of "
                    f"systems.{family}"
                )
            elif checkpoints.get(key) is None or checkpoints.get(key) != checkpoints.get(entry.same_as):
                problems.append(
                    f"{where}: same_as {entry.same_as!r}, but the registry does not "
                    f"bind both keys to one checkpoint ({checkpoints.get(key)!r} vs "
                    f"{checkpoints.get(entry.same_as)!r})"
                )
            else:
                out[family][key] = SystemDisclosure(
                    family, key, target.training_data, target.test_corpus_seen,
                    same_as=entry.same_as,
                )
    if problems:
        raise SystemsContractError(problems)
    return out
