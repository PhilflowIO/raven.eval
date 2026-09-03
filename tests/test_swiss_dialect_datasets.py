"""Guards for the Swiss dialect corpora that the generic contract test cannot make.

``tests/test_dataset_contract.py`` proves the config and the code name the same
datasets. Three things specific to these two corpora are not covered by that and
are the ones that would silently corrupt a published number:

  * the acquisition path is a URL plus a digest, not a Hub revision — so every
    artefact must actually carry both, and a wrong digest must raise rather than
    hand back bytes;
  * the licence claim on SPC is an *inference*, and NOTICE must keep saying so;
  * dialects are never aggregated — not with each other, not into an overall.

These tests are offline and import no heavy dependency: the loader modules pull
``pyarrow`` / ``soundfile`` lazily inside ``iter_samples``, so importing them to
read their pins costs nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from raven_asr.config import (
    DIALECT_DATASET_IDS,
    FLOZI_SUBSETS,
    WER_DATASETS,
    resolve_wer_dataset,
)
from raven_asr.datasets import WER_LOADERS
from raven_asr.datasets.fhnw_all_dialects import ARCHIVE, sample_id_for
from raven_asr.datasets.local_archive import (
    ChecksumMismatch,
    RemoteArtifact,
    ensure_artifact,
    sha256_of,
    verify,
)
from raven_asr.datasets.spc_test import HF_REVISION, SHARDS

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# ── the corpora are registered like every other dataset ──────────────────────


def test_dialect_ids_are_registered_datasets_and_loaders() -> None:
    for dataset_id in sorted(DIALECT_DATASET_IDS):
        assert dataset_id in WER_DATASETS, f"{dataset_id}: no WerDatasetSpec"
        assert dataset_id in WER_LOADERS, f"{dataset_id}: no loader registered"


def test_dialect_corpora_declare_the_metric_they_need() -> None:
    """Translation-shaped corpora must say so; the scorer must not guess."""
    for dataset_id in sorted(DIALECT_DATASET_IDS):
        assert WER_DATASETS[dataset_id].metric == "bleu+wer", (
            f"{dataset_id}: Swiss German audio against a Standard German "
            f"reference is translation-shaped; WER alone penalises a correct "
            f"translation for not being a transliteration."
        )


# ── no aggregation across dialects, and none into an overall ─────────────────


def test_no_dialect_id_can_be_absorbed_by_the_german_mixed_aggregate() -> None:
    overlap = sorted(DIALECT_DATASET_IDS & set(FLOZI_SUBSETS))
    assert not overlap, (
        f"{overlap} are both dialect corpora and `german-mixed` subsets — the "
        f'"All" aggregation over Tuda-De / MLS / Common Voice would absorb them '
        f"into a figure that describes no corpus that exists."
    )


def test_each_dialect_corpus_resolves_only_to_itself() -> None:
    """A dialect run yields a per-corpus number and nothing wider."""
    for dataset_id in sorted(DIALECT_DATASET_IDS):
        spec = WER_DATASETS[dataset_id]
        assert spec.subsets == (dataset_id,), (
            f"{dataset_id}: subsets {spec.subsets} — a dialect corpus owns "
            f"exactly its own subset, so nothing can average it with another."
        )
        assert resolve_wer_dataset(dataset_id) == (dataset_id, dataset_id)


# ── acquisition: URL + digest, verified ──────────────────────────────────────


def _artifacts() -> list[RemoteArtifact]:
    return [*SHARDS, ARCHIVE]


def test_every_dialect_artifact_pins_a_url_a_digest_and_a_size() -> None:
    for artifact in _artifacts():
        assert artifact.url.startswith("https://"), artifact.filename
        assert SHA256_RE.match(artifact.sha256), (
            f"{artifact.filename}: sha256 {artifact.sha256!r} is not 64 hex chars"
        )
        assert artifact.size_bytes > 0, artifact.filename
        assert artifact.durability in {"doi", "hf", "vendor"}, artifact.filename


def test_spc_shard_urls_carry_the_pinned_revision() -> None:
    """A `resolve/main` URL would let a re-upload move a published number."""
    for shard in SHARDS:
        assert f"/resolve/{HF_REVISION}/" in shard.url, shard.url
        assert "/resolve/main/" not in shard.url, shard.url
    assert WER_DATASETS["spc-test"].revision == HF_REVISION


def test_fhnw_records_its_durability_liability() -> None:
    spec = WER_DATASETS["fhnw-all-dialects"]
    assert spec.durability == "vendor", (
        "the Dropbox share link has no version history; ranking it above "
        "'vendor' would hide the liability ADR-app-0054 requires us to track"
    )
    assert spec.sha256 == ARCHIVE.sha256
    assert "mirror" in ARCHIVE.note.lower()


def test_a_wrong_digest_raises_instead_of_returning_bytes(tmp_path: Path) -> None:
    """The whole point of the pin: substitution fails loudly."""
    payload = b"not the corpus"
    artifact = RemoteArtifact(
        url="https://example.invalid/x.bin",
        filename="x.bin",
        sha256="0" * 64,
        size_bytes=len(payload),
    )
    (tmp_path / "x.bin").write_bytes(payload)
    with pytest.raises(ChecksumMismatch):
        verify(tmp_path / "x.bin", artifact)
    with pytest.raises(ChecksumMismatch):
        ensure_artifact(artifact, tmp_path)


def test_ensure_artifact_fetches_when_absent_and_verifies(tmp_path: Path) -> None:
    payload = b"corpus bytes"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    artifact = RemoteArtifact(
        url="https://example.invalid/y.bin",
        filename="y.bin",
        sha256=digest,
        size_bytes=len(payload),
    )
    calls: list[str] = []

    def fake_fetch(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(payload)

    path = ensure_artifact(artifact, tmp_path, fetcher=fake_fetch)
    assert path.read_bytes() == payload
    assert calls == [artifact.url]
    assert sha256_of(path) == digest

    # Second call trusts the marker and does not re-fetch.
    ensure_artifact(artifact, tmp_path, fetcher=fake_fetch)
    assert calls == [artifact.url]


def test_a_tampered_local_mirror_is_caught_on_the_next_call(tmp_path: Path) -> None:
    payload = b"corpus bytes"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    artifact = RemoteArtifact(
        url="https://example.invalid/z.bin",
        filename="z.bin",
        sha256=digest,
        size_bytes=len(payload),
    )
    ensure_artifact(artifact, tmp_path, fetcher=lambda _u, d: d.write_bytes(payload))
    (tmp_path / "z.bin").write_bytes(b"swapped bytes!!")
    with pytest.raises(ChecksumMismatch):
        ensure_artifact(artifact, tmp_path)


# ── the licence inference must stay an inference ─────────────────────────────


def test_notice_marks_the_spc_licence_as_inferred_and_carries_both() -> None:
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Swiss Parliaments Corpus" in notice
    assert "All Swiss German Dialects Test Set" in notice
    assert "INFERRED, NOT STATED" in notice, (
        "the i4ds/SPC_test card carries no license tag; NOTICE must not present "
        "MIT as a licence read off the artefact we download"
    )
    # The grant itself, not a reference to it.
    assert "Permission is hereby granted, free of charge" in notice
    assert "WITHOUT WARRANTY OF ANY KIND" in notice


def test_spc_spec_license_string_carries_the_caveat() -> None:
    assert "inferred" in WER_DATASETS["spc-test"].license.lower()


# ── small pure helper ────────────────────────────────────────────────────────


def test_sample_id_keeps_the_dialect_region() -> None:
    assert (
        sample_id_for({"accent": "ZH", "path": "abc-123.flac"})
        == "fhnw-all-dialects-ZH-abc-123"
    )
    assert sample_id_for({"path": "abc.flac"}) == "fhnw-all-dialects-xx-abc"
