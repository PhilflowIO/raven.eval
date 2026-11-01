"""A run reads its corpus from a pinned local snapshot; streaming is opt-in.

No network here: ``huggingface_hub.snapshot_download`` is replaced by a fake that
serves a tmp directory holding real (tiny) parquet files, so the read path —
``datasets`` over local parquet, audio left undecoded, soundfile decoding — is
the one a real run takes.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("datasets")
pytest.importorskip("pyarrow")
pytest.importorskip("soundfile")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from raven_asr import prefetch as prefetch_mod
from raven_asr import runner
from raven_asr.config import WER_DATASETS
from raven_asr.datasets.fleurs_de import FleursDeLoader
from raven_asr.datasets.flozi_mixed_evals import DATA_FILES, FloziMixedEvalsLoader

REV = WER_DATASETS["fleurs"].revision


def _wav(value: float) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.full(160, value, dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


class _FakeHub:
    """Stands in for ``snapshot_download``: a cache that may or may not be filled."""

    def __init__(self, root: Path, populate: dict[str, list[dict[str, Any]]]) -> None:
        self.root = root
        self.populate = populate
        self.calls: list[dict[str, Any]] = []
        self.cached = False

    def __call__(self, **kw: Any) -> str:
        from huggingface_hub.errors import LocalEntryNotFoundError

        self.calls.append(kw)
        if kw["local_files_only"] and not self.cached:
            raise LocalEntryNotFoundError("not cached")
        if not kw["local_files_only"]:
            for rel, rows in self.populate.items():
                _write_parquet(self.root / rel, rows)
            self.cached = True
        return str(self.root)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeHub:
    import huggingface_hub

    rows = [
        {"audio": {"bytes": _wav(0.1 * i), "path": f"{i}.wav"},
         "raw_transcription": f"satz {i}", "transcription": f"satz {i}"}
        for i in range(5)
    ]
    fake = _FakeHub(tmp_path / "snap", {FleursDeLoader.hf_data_files[0]: rows})
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return fake


def test_a_run_fetches_the_pinned_split_once_then_reads_it_locally(hub: _FakeHub) -> None:
    loader = FleursDeLoader(streaming=False, revision=REV)
    samples = list(loader.iter_samples("fleurs"))
    assert [s.sample_id for s in samples] == [f"fleurs-{i}" for i in range(5)]
    assert [s.reference for s in samples] == [f"satz {i}" for i in range(5)]
    assert loader._source == "local-snapshot"
    # Looked up offline first, then fetched: only the split's files, at the pin.
    assert [c["local_files_only"] for c in hub.calls] == [True, False]
    for call in hub.calls:
        assert call["revision"] == REV
        assert call["repo_type"] == "dataset"
        assert call["allow_patterns"] == list(FleursDeLoader.hf_data_files)

    # A second run finds the snapshot and never asks the Hub for anything.
    hub.calls.clear()
    again = FleursDeLoader(streaming=False, revision=REV)
    assert len(list(again.iter_samples("fleurs", limit=2))) == 2
    assert [c["local_files_only"] for c in hub.calls] == [True]


def test_streaming_is_explicit_and_skips_the_snapshot(
    hub: _FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    import datasets

    seen: list[dict[str, Any]] = []

    class _Streamed:
        def cast_column(self, *_a: Any) -> _Streamed:
            return self

        def __iter__(self) -> Any:
            return iter([])

    def fake_load(*args: Any, **kw: Any) -> _Streamed:
        seen.append(kw)
        return _Streamed()

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    loader = FleursDeLoader(streaming=True, revision=REV)
    assert list(loader.iter_samples("fleurs")) == []
    assert seen[0]["streaming"] is True and seen[0]["revision"] == REV
    assert hub.calls == []
    assert loader._source == "hf-streaming"


def test_the_runner_no_longer_forces_streaming() -> None:
    for dataset_id in ("fleurs", "mls-de", "voxpopuli-de", "avemio-german-mixed-test"):
        loader, _ = runner._iter_loader_for_subset(dataset_id)
        assert loader._streaming is False, dataset_id
        loader, _ = runner._iter_loader_for_subset(dataset_id, streaming=True)
        assert loader._streaming is True, dataset_id


def test_flozi_looks_up_the_pinned_revision_not_any_cached_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    rev = WER_DATASETS["german-mixed"].revision
    rows = [{"audio": {"bytes": _wav(0.0), "path": "x.wav"}, "references": "hallo",
             "from": "Tuda-De"}]
    fake = _FakeHub(tmp_path / "snap", {rel: rows for rel in DATA_FILES})
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    loader = FloziMixedEvalsLoader(revision=rev)
    assert len(list(loader.iter_samples("Tuda-De"))) == 25
    assert {c["revision"] for c in fake.calls} == {rev}
    assert len(DATA_FILES) == 25


def test_an_incomplete_snapshot_is_completed_not_trusted(hub: _FakeHub) -> None:
    hub.root.mkdir(parents=True)
    hub.cached = True  # the snapshot dir exists, but its file never arrived
    loader = FleursDeLoader(streaming=False, revision=REV)
    assert len(list(loader.iter_samples("fleurs"))) == 5
    assert [c["local_files_only"] for c in hub.calls] == [True, False]


def test_prefetch_covers_every_registered_dataset() -> None:
    for dataset_id in WER_DATASETS:
        loader_cls = runner.load_loader_class(dataset_id)
        assert callable(getattr(loader_cls, "prefetch", None)), dataset_id
    assert prefetch_mod._dataset_ids("all") == list(WER_DATASETS)
    assert prefetch_mod._dataset_ids("Tuda-De") == ["german-mixed"]


def test_prefetch_cli_reports_failure_in_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake(dataset_id: str) -> list[str]:
        if dataset_id == "mls-de":
            raise OSError("disk full")
        return [f"/cache/{dataset_id}"]

    monkeypatch.setattr(prefetch_mod, "prefetch", fake)
    assert prefetch_mod.main(["--dataset", "fleurs"]) == 0
    assert "[fleurs] ready" in capsys.readouterr().out
    assert prefetch_mod.main(["--dataset", "all"]) == 1
    assert "mls-de" in capsys.readouterr().err


def test_prefetch_fetches_the_hf_split(hub: _FakeHub) -> None:
    paths = prefetch_mod.prefetch("fleurs")
    assert paths == [str(hub.root / FleursDeLoader.hf_data_files[0])]
    assert Path(paths[0]).is_file()
