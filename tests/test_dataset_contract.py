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

import re
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


def test_every_wer_dataset_pins_its_references() -> None:
    """A WER is a statement about one fixed set of references.

    Either an upstream revision names that set, or — for a source with no
    version history, like a vendor share link — the archive digest does. Neither
    means the published number drifts with whatever upstream serves next.
    """
    for dataset_id, spec in sorted(WER_DATASETS.items()):
        pinned = spec.revision and spec.revision.lower() not in {
            "main", "master", "head", "latest"
        }
        assert pinned or spec.sha256, (
            f"{dataset_id}: neither a pinned revision nor an archive sha256 — "
            f"its references can change underneath a published WER."
        )


# ── what is spoken: language_tag / variety_label / locality ─────────────────
#
# A deliberately small allowlist rather than a full BCP 47 parser. Every tag
# here is a plain ``language[-REGION]``; anything richer would be a new kind of
# claim and should be a deliberate edit to this list. The region allowlist is
# what actually stops pseudo tags: "de-BY" and "gsw-BE" are WELL-FORMED BCP 47
# (Belarus, Belgium), so no syntax check can catch a state or canton smuggled
# into the region slot — only a list of the countries we mean can.

#: ISO 639 language subtags in use: German, Swiss German, Bavarian, English.
ALLOWED_LANGUAGES = frozenset({"de", "gsw", "bar", "en"})
#: ISO 3166-1 alpha-2 countries a German-variety corpus here may name.
ALLOWED_REGIONS = frozenset({"DE", "AT", "CH"})
LANGUAGE_TAG_RE = re.compile(r"^(?P<language>[a-z]{2,3})(?:-(?P<region>[A-Z]{2}))?$")


def language_tag_problem(tag: object) -> str:
    """Why ``tag`` is not an acceptable language_tag, or ``""``."""
    if not isinstance(tag, str):
        return f"{tag!r} is not a string"
    m = LANGUAGE_TAG_RE.match(tag)
    if not m:
        return f"{tag!r} is not of the form language[-REGION]"
    if m.group("language") not in ALLOWED_LANGUAGES:
        return f"language subtag {m.group('language')!r} not in {sorted(ALLOWED_LANGUAGES)}"
    region = m.group("region")
    if region is not None and region not in ALLOWED_REGIONS:
        return (
            f"region subtag {region!r} is not one of the countries "
            f"{sorted(ALLOWED_REGIONS)} — a region subtag names a country, never "
            f"a federal state or canton"
        )
    return ""


def _all_config_entries() -> list[tuple[str, dict]]:
    doc = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return [(f"{metric}/{entry['id']}", entry)
            for metric, entries in doc["datasets"].items() for entry in entries]


@pytest.mark.parametrize(
    "tag", ["de", "de-DE", "de-AT", "de-CH", "gsw-CH", "gsw", "bar", "en"]
)
def test_valid_language_tags_pass(tag: str) -> None:
    assert language_tag_problem(tag) == ""


@pytest.mark.parametrize(
    "tag",
    [
        "de-BY",      # Bavaria as a region subtag — is Belarus
        "gsw-BE",     # Bern as a region subtag — is Belgium
        "de-BAY",     # three-letter pseudo region
        "de-ba",      # xSID's own variety label, lower-case region
        "de_DE",      # POSIX locale, not BCP 47
        "Swiss German",
        "",
        None,
    ],
)
def test_pseudo_and_malformed_tags_are_rejected(tag: object) -> None:
    assert language_tag_problem(tag) != ""


def test_every_dataset_declares_language_variety_and_locality() -> None:
    problems: list[str] = []
    for where, entry in _all_config_entries():
        missing = [k for k in ("language_tag", "variety_label", "locality")
                   if k not in entry]
        if missing:
            problems.append(f"{where}: missing {missing}")
            continue
        if why := language_tag_problem(entry["language_tag"]):
            problems.append(f"{where}: {why}")
        label = entry["variety_label"]
        if not isinstance(label, str) or not label.strip():
            problems.append(f"{where}: variety_label must be a non-empty string")
        locality = entry["locality"]
        if locality is not None and (not isinstance(locality, str)
                                     or not locality.strip()):
            problems.append(f"{where}: locality must be null or a non-empty string")
    assert not problems, "\n".join(problems)


def test_dialect_corpora_carry_a_dialect_language_tag() -> None:
    """A dialect corpus tagged plain ``de`` would read as Standard German."""
    from raven_asr.config import DIALECT_DATASET_IDS

    tags = {entry["id"]: entry["language_tag"]
            for where, entry in _all_config_entries() if where.startswith("wer/")}
    # The control spur is in DIALECT_DATASET_IDS (barred from aggregates) but
    # its variety IS Standard German; it is the one dialect-set id tagged de.
    for dataset_id in sorted(DIALECT_DATASET_IDS - {"xsid-de-control"}):
        assert tags[dataset_id].split("-")[0] in {"gsw", "bar"}, (
            f"{dataset_id}: dialect corpus tagged {tags[dataset_id]!r}"
        )
    assert tags["xsid-de-control"] == "de"
