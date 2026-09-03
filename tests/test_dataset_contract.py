"""The scoring contract and the code must name the same datasets — both ways.

``benchmark.config.yaml`` is published as raven.eval's scoring contract: "the
config is committed so third parties see the precise rules our DER/WER values
were computed under". That only holds if the config and the harness agree. It had
drifted in both directions at once — three WER ids with no loader anywhere in the
repo, and an implemented, measured, artifact-carrying DER dataset
(``voxconverse-test``) that the contract never mentioned.

These tests close both directions:

  * a config id with no registered loader → the contract promises a run nobody
    can perform;
  * a registered loader missing from the config → a published number produced
    under rules the contract does not describe.

Every assertion names the offending id. The tests are pure and offline: they read
one YAML file and two dataclass registries, download nothing, and import no
loader module (so they run without the ``asr`` extra installed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from raven_asr.config import WER_DATASETS  # noqa: E402
from raven_asr.datasets import NON_LOADER_MODULES, WER_LOADERS  # noqa: E402
from raven_diar.config import DER_DATASETS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "benchmark.config.yaml"

# raven_diar/datasets has no NON_LOADER_MODULES of its own; keep the exclusion
# list local so a shared base added there later shows up as a deliberate edit.
DIAR_NON_LOADER_MODULES = frozenset({"__init__", "base"})


def _config_ids(metric: str) -> set[str]:
    doc = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    entries = doc["datasets"][metric]
    return {entry["id"] for entry in entries}


def _module_names(package_dir: Path, excluded: frozenset[str]) -> set[str]:
    return {p.stem for p in package_dir.glob("*.py")} - set(excluded)


# ── WER ──────────────────────────────────────────────────────────────────────


def test_every_wer_config_id_has_a_registered_loader() -> None:
    promised = _config_ids("wer")
    implemented = set(WER_DATASETS)
    orphaned = sorted(promised - implemented)
    assert not orphaned, (
        f"benchmark.config.yaml promises WER dataset(s) {orphaned} with no entry "
        f"in raven_asr.config.WER_DATASETS — the contract advertises a run "
        f"nobody can reproduce. Implement the loader (ADR-app-0054: reconcile by "
        f"implementing, not by deleting) or drop the id."
    )


def test_every_registered_wer_dataset_is_in_the_config() -> None:
    promised = _config_ids("wer")
    implemented = set(WER_DATASETS)
    undocumented = sorted(implemented - promised)
    assert not undocumented, (
        f"WER dataset(s) {undocumented} are implemented in "
        f"raven_asr.config.WER_DATASETS but absent from benchmark.config.yaml — "
        f"a number measured on them would be published under rules the committed "
        f"contract does not describe. Add them to datasets.wer."
    )


def test_wer_registry_and_loader_registry_agree() -> None:
    specs = set(WER_DATASETS)
    loaders = set(WER_LOADERS)
    assert specs == loaders, (
        f"raven_asr.config.WER_DATASETS and raven_asr.datasets.WER_LOADERS "
        f"disagree: only in WER_DATASETS {sorted(specs - loaders)}, "
        f"only in WER_LOADERS {sorted(loaders - specs)}."
    )


def test_every_wer_loader_module_exists_and_is_registered() -> None:
    package_dir = REPO_ROOT / "raven_asr" / "datasets"
    on_disk = _module_names(package_dir, NON_LOADER_MODULES)
    referenced = {spec.loader for spec in WER_DATASETS.values()}

    missing = sorted(referenced - on_disk)
    assert not missing, (
        f"WER_DATASETS points at loader module(s) {missing} that do not exist "
        f"under {package_dir.relative_to(REPO_ROOT)}."
    )

    unregistered = sorted(on_disk - referenced)
    assert not unregistered, (
        f"loader module(s) {unregistered} exist under "
        f"{package_dir.relative_to(REPO_ROOT)} but no WER_DATASETS entry uses "
        f"them — either register them (and add the id to benchmark.config.yaml) "
        f"or list them in raven_asr.datasets.NON_LOADER_MODULES if they are "
        f"shared infrastructure."
    )


def test_every_wer_dataset_pins_a_source_and_a_license() -> None:
    """ADR-app-0054: reproducibility is a property of the acquisition path."""
    for dataset_id, spec in sorted(WER_DATASETS.items()):
        assert spec.source, f"{dataset_id}: no source recorded"
        assert spec.license, f"{dataset_id}: no license recorded"
        assert spec.durability in {"doi", "hf", "vendor"}, (
            f"{dataset_id}: durability {spec.durability!r} is not one of "
            f"doi / hf / vendor"
        )
        assert spec.subsets, f"{dataset_id}: no subset selectors recorded"


# ── DER ──────────────────────────────────────────────────────────────────────


def test_every_der_config_id_has_a_registered_loader() -> None:
    promised = _config_ids("der")
    implemented = set(DER_DATASETS)
    orphaned = sorted(promised - implemented)
    assert not orphaned, (
        f"benchmark.config.yaml promises DER dataset(s) {orphaned} with no entry "
        f"in raven_diar.config.DER_DATASETS."
    )


def test_every_registered_der_dataset_is_in_the_config() -> None:
    promised = _config_ids("der")
    implemented = set(DER_DATASETS)
    undocumented = sorted(implemented - promised)
    assert not undocumented, (
        f"DER dataset(s) {undocumented} are implemented in "
        f"raven_diar.config.DER_DATASETS but absent from benchmark.config.yaml. "
        f"voxconverse-test is the exact case that made this test exist: measured, "
        f"published in BENCHMARKS.md, with a committed artifacts/ directory, and "
        f"nowhere in the contract."
    )


def test_every_der_loader_module_exists_and_is_registered() -> None:
    package_dir = REPO_ROOT / "raven_diar" / "datasets"
    on_disk = _module_names(package_dir, DIAR_NON_LOADER_MODULES)
    referenced = {spec.loader for spec in DER_DATASETS.values()}

    missing = sorted(referenced - on_disk)
    assert not missing, (
        f"DER_DATASETS points at loader module(s) {missing} that do not exist "
        f"under {package_dir.relative_to(REPO_ROOT)}."
    )

    unregistered = sorted(on_disk - referenced)
    assert not unregistered, (
        f"loader module(s) {unregistered} exist under "
        f"{package_dir.relative_to(REPO_ROOT)} but no DER_DATASETS entry uses them."
    )


def test_every_der_dataset_pins_a_revision() -> None:
    for dataset_id, spec in sorted(DER_DATASETS.items()):
        assert spec.revision, (
            f"{dataset_id}: no pinned revision — a floating source lets a "
            f"published DER drift with an upstream branch."
        )
