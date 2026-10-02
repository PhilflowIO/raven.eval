"""The published tables must equal the artifacts they link to.

`make verify` proves that `expected.json` equals what re-scoring the committed
RTTMs produces. Nothing proved that `BENCHMARKS.md` equals `expected.json`, so
the verified chain stopped one step short of the thing a reader actually reads: a
typo, a stale row left behind by a re-run, or a number nudged by hand would have
passed CI. Today the table and the artifacts agree — that is care, not a
mechanism, and care does not survive the next campaign.

This closes the last link in both directions: every published row must resolve to
a committed artifact and match it, and every committed DER artifact must appear
in the table. The second direction is the one that catches a measurement quietly
dropped from the page. The WER table carries the same guard — it had the same gap
and it is the same twenty lines.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO_ROOT / "BENCHMARKS.md"
ARTIFACTS = REPO_ROOT / "artifacts"

#: A table row links its run as `[label](./artifacts/<run>/<model>/)`.
_RUN_LINK = re.compile(r"\]\(\./artifacts/(?P<run>[^/)]+)/(?P<model>[^/)]+)/\)")

#: The table prints two decimals, so a cell may differ from the committed value
#: by up to half of the last printed place. Anything larger is a real mismatch.
ROUNDING_TOLERANCE = 0.005 + 1e-9

#: Fixtures prove the mechanism and are deliberately absent from the page.
FIXTURE_PREFIXES = ("_demo",)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _number(cell: str) -> float:
    """Parse a table cell that may be bolded (`**16.08**`)."""
    return float(cell.replace("*", "").strip())


#: Column layout of each published table, as `{cell index: expected.json field}`.
#: Named here rather than sliced inline so a column added to the page without a
#: field behind it fails loudly instead of shifting every index by one.
_DER_COLUMNS = {
    2: "der_full", 3: "miss", 4: "fa", 5: "conf",
    6: "der_classic", 7: "miss_classic", 8: "fa_classic", 9: "conf_classic",
    10: "der_classic_filemean",
}
# model | dataset | 9 numbers | CI | n | run | Trainingsdaten | kennt Testkorpus
_DER_WIDTH, _DER_N = 16, 12
_WER_COLUMNS = {2: "wer_pct", 4: "cer_pct", 6: "entity_hit_rate_pct"}
#: "[lo, hi]" cells, each bound to its two expected.json fields.
_WER_INTERVALS = {3: ("wer_ci_lo", "wer_ci_hi"), 7: ("entity_ci_lo", "entity_ci_hi")}
_WER_N_ENTITIES = 8
_WER_WIDTH, _WER_N = 13, 5
# model | subset | wer | CI | cer | n | numbers hit | CI | numbers | run | ref
#       | Trainingsdaten | kennt Testkorpus
#: The two disclosure cells close every result row, in both tables. They are
#: appended rather than inserted so no numeric column index above moves.
_DISCLOSURE_CELLS = {"training_data": -2, "corpus_seen": -1}


def _linked_rows() -> list[tuple[str, list[str], Path]]:
    """Every markdown table row in BENCHMARKS.md that links an artifact dir."""
    out = []
    for line in BENCHMARKS.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        link = _RUN_LINK.search(line)
        if not link:
            continue
        artifact = ARTIFACTS / link.group("run") / link.group("model")
        out.append((line, _cells(line), artifact))
    return out


def published_rows(metric: str = "der") -> list[dict]:
    """Rows of one published table, parsed from the committed markdown.

    Which table a row belongs to is decided by the artifact it links, not by
    where it sits on the page: a DER artifact has a ``gold/`` subtree, a WER one
    has ``predictions_*.jsonl``. That keeps the two tables from being told apart
    by heading order, which is exactly the kind of thing an edit reshuffles.
    """
    is_der = metric == "der"
    width, n_idx = (_DER_WIDTH, _DER_N) if is_der else (_WER_WIDTH, _WER_N)
    columns = _DER_COLUMNS if is_der else _WER_COLUMNS
    rows: list[dict] = []
    for line, cells, artifact in _linked_rows():
        looks_der = (artifact / "gold").is_dir()
        if looks_der != is_der:
            continue
        if len(cells) != width:
            raise AssertionError(
                f"{metric.upper()} table row has {len(cells)} cells, expected "
                f"{width} — the row format changed and this guard was not "
                f"updated:\n  {line}"
            )
        values = {field: _number(cells[i]) for i, field in columns.items()}
        if not is_der:
            # "[lo, hi]" — an interval is a published number like the point.
            for i, (lo_field, hi_field) in _WER_INTERVALS.items():
                lo, hi = cells[i].strip("[]").split(",")
                values[lo_field], values[hi_field] = _number(lo), _number(hi)
        rows.append({
            "model_cell": cells[0],
            "dataset_cell": cells[1],
            "values": values,
            "n": int(cells[n_idx]),
            "n_entities": None if is_der else int(cells[_WER_N_ENTITIES]),
            "artifact": artifact,
            "metric": metric,
            **{name: cells[i] for name, i in _DISCLOSURE_CELLS.items()},
        })
    return rows


def committed_der_artifacts() -> list[Path]:
    """Every committed DER artifact that is a product number, not a fixture."""
    return sorted(
        p.parent
        for p in ARTIFACTS.rglob("expected.json")
        if (p.parent / "gold").is_dir()
        and any((p.parent / "gold").rglob("*.rttm"))
        and not p.parent.relative_to(ARTIFACTS).parts[0].startswith(FIXTURE_PREFIXES)
    )


def _dataset_for(row: dict, expected: dict) -> str:
    """Which dataset in the artifact this table row is about.

    The dataset cell carries prose ("callhome-de (German, telephone)",
    "voxconverse (**test**)"), so it is matched by its leading identifier against
    the keys the artifact actually holds rather than parsed.
    """
    head = row["dataset_cell"].split("(")[0].strip().replace("*", "")
    candidates = [d for d in expected if d == head or d.startswith(head)]
    if len(candidates) == 1:
        return candidates[0]
    # "voxconverse (**test**)" -> voxconverse-test; disambiguate by the artifact.
    if len(expected) == 1:
        return next(iter(expected))
    raise AssertionError(
        f"cannot map table dataset {row['dataset_cell']!r} onto artifact "
        f"{row['artifact'].relative_to(REPO_ROOT)} (has {sorted(expected)})"
    )


def test_the_table_is_not_empty():
    """A guard that silently matches zero rows guards nothing."""
    assert len(published_rows()) >= len(committed_der_artifacts())


@pytest.mark.parametrize(
    "row", published_rows(), ids=lambda r: f"{r['artifact'].name}-{r['dataset_cell'][:18]}"
)
def test_every_published_row_matches_its_artifact(row: dict):
    """Each printed number equals the committed one, to the printed precision."""
    expected_path = row["artifact"] / "expected.json"
    assert expected_path.exists(), (
        f"BENCHMARKS.md links {row['artifact'].relative_to(REPO_ROOT)} but there "
        f"is no expected.json there — a published number with no artifact."
    )
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    dataset = _dataset_for(row, expected)
    committed = expected[dataset]

    mismatches = [
        f"{field}: table {printed}, expected.json {committed[field]}"
        for field, printed in row["values"].items()
        if abs(printed - float(committed[field])) > ROUNDING_TOLERANCE
    ]
    assert not mismatches, (
        f"{row['artifact'].relative_to(REPO_ROOT)} [{dataset}] — the published "
        f"table disagrees with the artifact it links to:\n  "
        + "\n  ".join(mismatches)
    )


@pytest.mark.parametrize(
    "row", published_rows(), ids=lambda r: f"{r['artifact'].name}-{r['dataset_cell'][:18]}"
)
def test_every_published_n_matches_the_committed_file_count(row: dict):
    """`n` is a claim about how much data the row rests on."""
    expected = json.loads((row["artifact"] / "expected.json").read_text())
    dataset = _dataset_for(row, expected)
    n_gold = len(list((row["artifact"] / "gold" / dataset).glob("*.rttm")))
    assert row["n"] == n_gold, (
        f"{row['artifact'].relative_to(REPO_ROOT)} [{dataset}] publishes n="
        f"{row['n']} but {n_gold} gold RTTMs are committed."
    )


@pytest.mark.parametrize(
    "row", published_rows("wer"),
    ids=lambda r: f"{r['artifact'].name}-{r['dataset_cell'][:18]}",
)
def test_every_published_wer_row_matches_its_artifact(row: dict):
    """The WER table has the same gap and closes the same way."""
    expected = json.loads((row["artifact"] / "expected.json").read_text())
    subset = _dataset_for(row, expected)
    committed = expected[subset]
    mismatches = [
        f"{field}: table {printed}, expected.json {committed[field]}"
        for field, printed in row["values"].items()
        if abs(printed - float(committed[field])) > ROUNDING_TOLERANCE
    ]
    assert not mismatches, (
        f"{row['artifact'].relative_to(REPO_ROOT)} [{subset}] — the published "
        f"table disagrees with the artifact it links to:\n  "
        + "\n  ".join(mismatches)
    )
    n_lines = sum(
        1 for line in (row["artifact"] / f"predictions_{subset}.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    )
    assert row["n"] == n_lines, (
        f"{row['artifact'].relative_to(REPO_ROOT)} [{subset}] publishes n="
        f"{row['n']} but {n_lines} predictions are committed."
    )
    # The count a hit rate rests on is a claim like n, and matched exactly.
    assert row["n_entities"] == committed["n_entities"], (
        f"{row['artifact'].relative_to(REPO_ROOT)} [{subset}] publishes "
        f"{row['n_entities']} numbers, expected.json commits {committed['n_entities']}."
    )


def test_every_committed_artifact_appears_in_the_table():
    """The direction that catches a measurement quietly dropped from the page."""
    linked = {row["artifact"].resolve() for row in published_rows()}
    missing = [
        p.relative_to(REPO_ROOT)
        for p in committed_der_artifacts()
        if p.resolve() not in linked
    ]
    assert not missing, (
        "committed DER artifacts that BENCHMARKS.md does not publish: "
        f"{missing} — either publish them or say why they are held back."
    )


def test_every_committed_artifact_is_traceable_to_a_pinned_revision():
    """Gold is only as trustworthy as the revision it was drawn from.

    A checksum over `artifacts/*/gold/` was the obvious guard here and is the
    wrong one: `make verify` already re-scores from those exact bytes, so a gold
    edit moves every number and fails the build, and a manifest committed in the
    same change as the gold it covers proves nothing extra. What a checksum
    cannot tell you — and this can — is *where the reference came from*. Every
    artifact must name the pinned dataset and model revision it was produced
    under, so a disputed row can be re-derived from upstream rather than trusted.
    """
    offenders = []
    for artifact in committed_der_artifacts():
        summary_path = artifact / "summary.json"
        if not summary_path.exists():
            offenders.append(f"{artifact.relative_to(REPO_ROOT)}: no summary.json")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for field in ("dataset_revision", "model_revision"):
            value = summary.get(field)
            if not value or str(value).lower() in {
                "", "none", "latest", "main", "master", "head"
            }:
                offenders.append(
                    f"{artifact.relative_to(REPO_ROOT)}: {field}={value!r}"
                )
    assert not offenders, (
        "committed DER artifacts whose provenance is not pinned: " + str(offenders)
    )



# ── Training-data disclosure: no published system without an entry ───────────
#
# BENCHMARKS.md promises that training-data overlap is named beside the rows it
# affects. That was kept for one model, as prose. It is now two cells on every
# result row, read from `systems:` in benchmark.config.yaml, and these tests are
# what keeps a new row — or a new artifact — from being published without them.

README = REPO_ROOT / "README.md"


def published_systems() -> dict:
    """`systems:` from the contract, validated against the code registries.

    The binding to the registries lives here because this is the one place that
    may import both harnesses; ``raven_eval_core.contract`` takes them as
    arguments so the metric core keeps depending on neither.
    """
    from raven_asr.config import FLOZI_SUBSETS, KNOWN_MODELS, WER_DATASETS
    from raven_diar.config import DER_DATASETS, KNOWN_DIARIZERS
    from raven_eval_core.contract import load_contract, resolve_systems

    return resolve_systems(
        load_contract().systems,
        known_systems={"wer": KNOWN_MODELS, "der": KNOWN_DIARIZERS},
        # A WER row is published per subset where a corpus has several, so the
        # selectors the harness accepts are the corpus keys, not only the ids.
        known_corpora={"wer": {*WER_DATASETS, *FLOZI_SUBSETS}, "der": DER_DATASETS},
        same_checkpoint={
            "wer": {key: spec.model_id for key, spec in KNOWN_MODELS.items()},
            "der": {key: spec.model_id for key, spec in KNOWN_DIARIZERS.items()},
        },
    )


def _system_key(row: dict) -> str:
    """The registry key a row is about: the model cell up to any ` (licence)`."""
    return row["model_cell"].split(" (")[0].strip()


def _all_published_rows() -> list[dict]:
    return published_rows("der") + published_rows("wer")


@pytest.mark.parametrize(
    "row", _all_published_rows(),
    ids=lambda r: f"{r['artifact'].name}-{r['dataset_cell'][:18]}",
)
def test_every_published_row_states_its_training_data(row: dict):
    """Both disclosure cells equal what the contract records for that system."""
    systems = published_systems()[row["metric"]]
    key = _system_key(row)
    assert key in systems, (
        f"BENCHMARKS.md publishes a {row['metric'].upper()} row for {key!r} but "
        f"benchmark.config.yaml has no systems.{row['metric']}.{key} entry. Read "
        f"the vendor's model card, record what it says about training data, then "
        f"publish the row."
    )
    entry = systems[key]
    expected = json.loads((row["artifact"] / "expected.json").read_text())
    dataset = _dataset_for(row, expected)
    seen = entry.corpus_seen(dataset).seen
    printed = {name: row[name].replace("*", "").strip() for name in _DISCLOSURE_CELLS}
    assert printed == {"training_data": entry.status, "corpus_seen": seen}, (
        f"{key} on {dataset}: the row's last two cells must read "
        f"`| {entry.status} | {seen} |` (systems.{row['metric']}.{key}), "
        f"the table prints {printed}."
    )


def test_every_committed_artifact_has_a_training_data_entry():
    """An artifact is a published measurement even before it has a table row."""
    from raven_asr.config import KNOWN_MODELS
    from raven_diar.config import KNOWN_DIARIZERS

    systems = published_systems()
    by_label = {
        "wer": {spec.label: key for key, spec in KNOWN_MODELS.items()},
        "der": {spec.label: key for key, spec in KNOWN_DIARIZERS.items()},
    }
    checked, missing = 0, []
    for expected_path in sorted(ARTIFACTS.rglob("expected.json")):
        artifact = expected_path.parent
        if artifact.relative_to(ARTIFACTS).parts[0].startswith(FIXTURE_PREFIXES):
            continue
        family = "der" if (artifact / "gold").is_dir() else "wer"
        where = artifact.relative_to(REPO_ROOT)
        key = by_label[family].get(artifact.name)
        if key is None or key not in systems[family]:
            missing.append(f"{where}: no systems.{family} entry for {artifact.name!r}")
            continue
        for dataset in json.loads(expected_path.read_text(encoding="utf-8")):
            checked += 1
            if dataset not in systems[family][key].test_corpus_seen:
                missing.append(f"{where}: systems.{family}.{key} says nothing about {dataset!r}")
    assert checked, "no committed artifact was examined — this guard matched nothing"
    assert not missing, (
        "committed artifacts without a training-data disclosure:\n  " + "\n  ".join(missing)
    )


def _table_after(text: str, marker: str) -> list[list[str]]:
    """Body rows of the first markdown table following ``marker``."""
    lines = text[text.index(marker):].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    body = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        body.append(_cells(line))
    return body


def test_the_sources_table_matches_the_contract():
    """Status, date and every source link per published system, as recorded."""
    systems = published_systems()
    published = {(r["metric"], _system_key(r)) for r in _all_published_rows()}
    rows = {
        cells[0].strip("`"): cells
        for cells in _table_after(BENCHMARKS.read_text(encoding="utf-8"), "<!-- systems-sources -->")
    }
    assert set(rows) == {key for _, key in published}, (
        "the sources table must list exactly the systems with a published row"
    )
    for family, key in sorted(published):
        entry, cells = systems[family][key], rows[key]
        assert cells[1] == entry.status, f"{key}: sources table prints {cells[1]!r}"
        training = entry.training_data
        assert cells[2] == training.checked.isoformat(), f"{key}: checked date {cells[2]!r}"
        absent = [s.url for s in training.sources if f"({s.url})" not in cells[3]]
        assert not absent, f"{key}: sources table omits {absent}"


#: The README headline rows carry no model name, so which system and which
#: corpora a row stands for is stated here. A row added to that table without a
#: line below fails, which is the point.
_README_ROWS = {
    "VoxConverse test (EN, in-the-wild)": ("der", "pyannote-community-1", ["voxconverse-test"]),
    "VoxConverse dev (EN)": ("der", "pyannote-community-1", ["voxconverse"]),
    "CALLHOME-de (DE, telephone)": ("der", "pyannote-community-1", ["callhome-de"]),
    "Tuda-De / CommonVoice / MLS (DE)": (
        "wer", "primeline/parakeet-primeline",
        ["Tuda-De", "common_voice_19_0", "multilingual_librispeech"],
    ),
}


def test_the_readme_headline_table_states_training_data():
    """The first table a reader meets carries the same two columns."""
    systems = published_systems()
    rows = _table_after(README.read_text(encoding="utf-8"), "<!-- headline-table -->")
    assert rows, "README headline table not found"
    for cells in rows:
        assert cells[1] in _README_ROWS, (
            f"README headline row {cells[1]!r} is not mapped to a system in "
            f"_README_ROWS — say which system and corpora it stands for."
        )
        family, key, datasets = _README_ROWS[cells[1]]
        entry = systems[family][key]
        # One value where the corpora agree, else one per corpus in the order
        # the dataset cell names them.
        values = [entry.corpus_seen(d).seen for d in datasets]
        seen = values[0] if len(set(values)) == 1 else " / ".join(values)
        assert cells[-2:] == [entry.status, seen], (
            f"README row {cells[1]!r} must end `| {entry.status} | {seen} |`, "
            f"it prints {cells[-2:]}."
        )
