"""Per-region WER/BLEU inside a dialect corpus (#32).

Each test guards one sentence the breakdown makes: that every utterance is in
exactly one region, that a region's number is the corpus scorer run on that
region's utterances and nothing else, and that a region too small to measure
shows its n and a note instead of a number.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from raven_asr.analysis import REGION_PARSERS, by_region
from raven_asr.config import REGION_MIN_N
from raven_eval_core.bleu import corpus_bleu_score
from raven_eval_core.flozi_wer import corpus_wer_pct

REPO_ROOT = Path(__file__).resolve().parents[1]
FHNW_ARTIFACTS = sorted(
    (REPO_ROOT / "artifacts").glob("*/*/predictions_fhnw-all-dialects.jsonl")
)


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "predictions_fhnw-all-dialects.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _row(region: str, i: int, ref: str, hyp: str) -> dict:
    return {"reference": ref, "prediction": hyp, "latency_s": 0.0,
            "sample_id": f"fhnw-all-dialects-{region}-{i:08d}-aaaa-bbbb"}


@pytest.mark.parametrize("path", FHNW_ARTIFACTS, ids=lambda p: p.parent.name)
def test_region_n_sum_to_the_dataset_n(path: Path) -> None:
    n_dataset = sum(1 for x in path.read_text(encoding="utf-8").splitlines() if x)
    regions = by_region(path, resamples=10)
    assert sum(r.n for r in regions) == n_dataset
    assert len({r.region for r in regions}) == len(regions)  # one row per region


def test_committed_artifacts_exist_to_split() -> None:
    assert FHNW_ARTIFACTS, "no committed fhnw-all-dialects artifact"


@pytest.mark.parametrize("path", FHNW_ARTIFACTS, ids=lambda p: p.parent.name)
def test_a_region_gets_a_number_exactly_when_it_reaches_the_minimum(path: Path) -> None:
    """Below the minimum n and a note only; at or above it, WER and BLEU."""
    for r in by_region(path, resamples=10):
        if r.n < REGION_MIN_N:
            assert r.wer is None and r.bleu is None
            assert str(REGION_MIN_N) in r.note
        else:
            assert r.wer is not None and r.bleu is not None


def test_a_region_number_is_the_corpus_scorer_on_that_region(tmp_path: Path) -> None:
    zh = [_row("ZH", i, "wir gehen heute nach hause", "wir gehen heute heim")
          for i in range(4)]
    be = [_row("BE", i, "das wetter ist schön", "das wetter ist schön")
          for i in range(3)]
    path = _write(tmp_path, zh + be)
    regions = {r.region: r for r in by_region(path, min_n=3, resamples=50)}

    assert [r.region for r in by_region(path, min_n=3, resamples=50)] == ["ZH", "BE"]
    for code, members in (("ZH", zh), ("BE", be)):
        refs = [m["reference"] for m in members]
        hyps = [m["prediction"] for m in members]
        row = regions[code]
        assert row.n == len(members)
        assert row.wer is not None and row.bleu is not None
        assert row.wer.point == pytest.approx(corpus_wer_pct(refs, hyps))
        assert row.wer.n == len(members)
        assert row.bleu == pytest.approx(corpus_bleu_score(refs, hyps))
    assert regions["ZH"].name == "Zürich"


def test_below_min_n_shows_n_and_a_note_not_a_number(tmp_path: Path) -> None:
    path = _write(tmp_path, [_row("ZH", 0, "a b", "a b"), _row("ZH", 1, "c d", "c")])
    (row,) = by_region(path, min_n=3, resamples=10)
    assert row.n == 2
    assert row.wer is None and row.bleu is None
    assert "n < 3" in row.note


def test_an_unmapped_region_is_reported_raw(tmp_path: Path) -> None:
    (row,) = by_region(_write(tmp_path, [_row("xx", 0, "a", "a")]), resamples=10)
    assert row.region == row.name == "xx"


def test_an_unreadable_id_fails_instead_of_dropping_the_line(tmp_path: Path) -> None:
    rows = [_row("ZH", 0, "a", "a"),
            {"reference": "b", "prediction": "b", "sample_id": "something-else"}]
    with pytest.raises(ValueError):
        by_region(_write(tmp_path, rows), resamples=10)


def test_lines_without_sample_id_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        by_region(_write(tmp_path, [{"reference": "a", "prediction": "a"}]))


def test_a_corpus_without_regions_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "predictions_spc-test.jsonl"
    path.write_text(json.dumps(_row("ZH", 0, "a", "a")) + "\n", encoding="utf-8")
    with pytest.raises(KeyError):
        by_region(path)


def test_only_dialect_corpora_are_split() -> None:
    from raven_asr.config import DIALECT_DATASET_IDS

    assert set(REGION_PARSERS) <= DIALECT_DATASET_IDS


def _benchmarks() -> list[str]:
    return (REPO_ROOT / "BENCHMARKS.md").read_text(encoding="utf-8").splitlines()


def _cells(line: str) -> list[str]:
    return [c.strip().rstrip("¹") for c in line.strip().strip("|").split("|")]


def _n_lines(path: Path) -> int:
    return sum(1 for x in path.read_text(encoding="utf-8").splitlines() if x)


def test_published_canton_counts_equal_the_artifacts() -> None:
    """The n row of the sample-run table is read off those artifacts, not typed.

    That table spreads the cantons across its columns and describes every
    committed FHNW run over the same first utterances — those whose line count
    is the sum of its n row.
    """
    text = _benchmarks()
    header_at = next(i for i, line in enumerate(text)
                     if line.startswith("| canton |") and "| n |" not in line)
    codes = _cells(text[header_at])[1:]
    counts = [int(c) for c in _cells(text[header_at + 2])[1:]]
    published = dict(zip(codes, counts, strict=True))
    sample_runs = [p for p in FHNW_ARTIFACTS if _n_lines(p) == sum(counts)]
    assert sample_runs
    for path in sample_runs:
        assert {r.region: r.n for r in by_region(path, resamples=10)} == published


_ARTIFACT_REF = re.compile(r"ARTIFACT=(artifacts/[^/\s`]+/[^/\s`]+)")


def _region_tables() -> list[tuple[Path, list[list[str]]]]:
    """Every one-canton-per-row table, bound to the artifact named just above it.

    Binding by the `make analyse ARTIFACT=…` reference the text gives for the
    table makes the link explicit: a table without one fails to parse.
    """
    text = _benchmarks()
    out = []
    for i, line in enumerate(text):
        if not (line.startswith("| canton |") and "| n |" in line):
            continue
        ref = next(m for j in range(i - 1, -1, -1)
                   if (m := _ARTIFACT_REF.search(text[j])))
        rows = []
        for row in text[i + 2:]:
            if not row.startswith("|"):
                break
            rows.append(_cells(row))
        out.append((REPO_ROOT / ref.group(1), rows))
    return out


def test_published_region_tables_equal_their_artifacts() -> None:
    """Every printed canton row — n, WER, interval, BLEU — is the scorer's output."""
    tables = _region_tables()
    assert tables
    for artifact, rows in tables:
        (path,) = artifact.glob("predictions_fhnw-all-dialects.jsonl")
        computed = {r.region: r for r in by_region(path)}
        assert [row[0] for row in rows] == list(computed)  # every region, n order
        for code, n, wer, interval, bleu in rows:
            r = computed[code]
            assert int(n) == r.n
            if r.wer is None or r.bleu is None:
                assert wer == interval == bleu == "—"
                continue
            lo, hi = (float(x) for x in interval.strip("[]").split(","))
            for printed, value in ((wer, r.wer.point), (lo, r.wer.lo),
                                   (hi, r.wer.hi), (bleu, r.bleu)):
                assert float(printed) == pytest.approx(value, abs=0.005 + 1e-9)
