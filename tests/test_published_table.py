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
_DER_WIDTH, _DER_N = 14, 12          # model | dataset | 9 numbers | CI | n | run
_WER_COLUMNS = {2: "wer_pct", 3: "cer_pct"}
_WER_WIDTH, _WER_N = 7, 4            # model | subset | wer | cer | n | run | ref


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
        rows.append({
            "model_cell": cells[0],
            "dataset_cell": cells[1],
            "values": {field: _number(cells[i]) for i, field in columns.items()},
            "n": int(cells[n_idx]),
            "artifact": artifact,
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
