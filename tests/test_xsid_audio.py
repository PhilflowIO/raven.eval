"""xSID-audio loader: the padding trap, the checksum gate, the control pair.

Three things here can silently produce a wrong published number rather than an
error, so each gets a test:

  * the upstream ``_1`` vs ``_001`` padding mismatch drops 99 of 300 Bavarian
    validation clips with no exception raised anywhere;
  * a mirror file that is merely *present* is not a verified artifact — the
    sha256 must be checked on every call, not only after a download;
  * ``xsid-bar`` without ``xsid-de-control`` is a number about one voice, and
    a dialect probe inside a general average silently moves that average.

Everything except the two ``real_corpus`` tests is pure and offline.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from raven_asr.config import DIALECT_DATASET_IDS, WER_DATASETS
from raven_asr.datasets import WER_LOADERS, load_loader_class

xsid = pytest.importorskip(
    "raven_asr.datasets.xsid_audio", reason="needs the asr extra (numpy)"
)


# ── The upstream padding defect ──────────────────────────────────────────────


def test_member_key_normalizes_zero_padding() -> None:
    """The whole point: ``_1`` and ``_001`` must land on the same key."""
    assert xsid._member_key("de-ba/valid/xsid_de-ba_valid_1.wav") == xsid._member_key(
        "de-ba/valid/xsid_de-ba_valid_001.wav"
    )
    assert xsid._member_key("a/b_1.wav") == "a/b_1.wav"
    assert xsid._member_key("a/b_007.wav") == "a/b_7.wav"


def test_member_key_does_not_merge_distinct_clips() -> None:
    """Normalizing padding must not also collapse different indices or stems."""
    assert xsid._member_key("x_01.wav") != xsid._member_key("x_02.wav")
    assert xsid._member_key("valid_1.wav") != xsid._member_key("test_1.wav")


def test_member_key_passes_through_unmatched_names() -> None:
    assert xsid._member_key("de-ba/README.txt") == "de-ba/README.txt"
    assert xsid._member_key("no-index.wav") == "no-index.wav"


def test_literal_join_would_drop_clips_but_member_key_does_not() -> None:
    """Reproduces the defect in miniature: TSV padding vs zip padding.

    Without the normalizer 2 of 3 clips vanish silently — no error, just a
    shorter corpus and a WER computed on the survivors.
    """
    tsv_paths = ["v_1.wav", "v_2.wav", "v_10.wav"]
    zip_members = ["v_001.wav", "v_002.wav", "v_010.wav"]

    literal_hits = [p for p in tsv_paths if p in set(zip_members)]
    assert literal_hits == [], "the naive join is supposed to miss everything here"

    keyed = {xsid._member_key(m): m for m in zip_members}
    normalized_hits = [keyed[xsid._member_key(p)] for p in tsv_paths]
    assert normalized_hits == zip_members


# ── The checksum gate ────────────────────────────────────────────────────────


def test_artifact_url_is_the_documented_zenodo_content_url() -> None:
    assert xsid.artifact_url("de-ba.zip") == (
        "https://zenodo.org/api/records/21605015/files/de-ba.zip/content"
    )
    with pytest.raises(KeyError):
        xsid.artifact_url("not-a-file.zip")


def test_ensure_artifact_rejects_bytes_that_do_not_match_the_manifest(
    tmp_path: Path,
) -> None:
    """A file of the right NAME is not the artifact. Present ≠ verified."""
    (tmp_path / "xsid_de_test.tsv").write_bytes(b"ID\tText\n1\twrong bytes\n")
    with pytest.raises(RuntimeError, match="failed verification"):
        xsid.ensure_artifact("xsid_de_test.tsv", tmp_path)


def test_ensure_artifact_fetches_when_absent_and_verifies_the_result(
    tmp_path: Path,
) -> None:
    payload = b"ID\tText\tIntent\tAudio\tText_original\n"
    digest = hashlib.sha256(payload).hexdigest()
    name = "xsid_de_valid.tsv"

    calls: list[str] = []

    def fake_fetch(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(payload)

    # Patch the manifest entry so the fake payload is the "correct" bytes.
    spec = xsid.ARTIFACTS[name]
    patched = xsid._Artifact(size=len(payload), md5=spec.md5, sha256=digest)
    original = dict(xsid.ARTIFACTS)
    xsid.ARTIFACTS[name] = patched
    try:
        out = xsid.ensure_artifact(name, tmp_path / "sub", fetcher=fake_fetch)
        assert out.read_bytes() == payload
        assert calls == [xsid.artifact_url(name)]
        # Second call must NOT re-fetch, but must re-verify.
        xsid.ensure_artifact(name, tmp_path / "sub", fetcher=fake_fetch)
        assert len(calls) == 1
    finally:
        xsid.ARTIFACTS.clear()
        xsid.ARTIFACTS.update(original)


def test_every_manifest_entry_carries_a_full_sha256() -> None:
    for name, spec in xsid.ARTIFACTS.items():
        assert len(spec.sha256) == 64, f"{name}: sha256 is not 64 hex chars"
        assert len(spec.md5) == 32, f"{name}: md5 is not 32 hex chars"
        assert spec.size > 0, f"{name}: no recorded size"


def test_sha256_of_streams_a_file(tmp_path: Path) -> None:
    blob = b"x" * (3 << 20)  # larger than the 1 MiB read chunk
    p = tmp_path / "blob.bin"
    p.write_bytes(blob)
    assert xsid.sha256_of(p) == hashlib.sha256(blob).hexdigest()


# ── Acquisition path is documented, not implicit ─────────────────────────────


def test_corpora_dir_env_precedence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(xsid.XSID_CORPORA_DIR_ENV, raising=False)
    monkeypatch.delenv(xsid.BAVARIAN_CORPORA_DIR_ENV, raising=False)
    monkeypatch.setenv(xsid.BENCH_CORPORA_DIR_ENV, str(tmp_path))
    assert xsid.corpora_dir() == tmp_path / "bavarian"

    monkeypatch.setenv(xsid.BAVARIAN_CORPORA_DIR_ENV, str(tmp_path / "bav"))
    assert xsid.corpora_dir() == tmp_path / "bav"

    monkeypatch.setenv(xsid.XSID_CORPORA_DIR_ENV, str(tmp_path / "x"))
    assert xsid.corpora_dir() == tmp_path / "x"


def test_no_env_at_all_still_resolves_to_a_writable_cache(monkeypatch) -> None:
    """A clean checkout with no environment must be able to download."""
    for env in (
        xsid.XSID_CORPORA_DIR_ENV,
        xsid.BAVARIAN_CORPORA_DIR_ENV,
        xsid.BENCH_CORPORA_DIR_ENV,
    ):
        monkeypatch.delenv(env, raising=False)
    assert xsid.corpora_dir() == (
        Path.home() / ".cache" / "raven-eval" / "corpora" / "bavarian"
    )


def test_required_artifacts_always_include_the_german_reference_tsvs() -> None:
    """Both ids need the ``de`` TSVs: one is scored against them, one IS them."""
    bar = xsid.XsidBavarianLoader()
    assert set(bar.required_artifacts()) == {
        "de-ba.zip",
        "xsid_de-ba_test.tsv",
        "xsid_de-ba_valid.tsv",
        "xsid_de_test.tsv",
        "xsid_de_valid.tsv",
    }
    control = xsid.XsidGermanControlLoader()
    assert set(control.required_artifacts()) == {
        "de.zip",
        "xsid_de_test.tsv",
        "xsid_de_valid.tsv",
    }


def test_a_different_zenodo_version_is_refused_not_silently_accepted() -> None:
    """The checksums describe v0.2 only; another version needs its own manifest."""
    with pytest.raises(ValueError, match="pinned Zenodo version"):
        xsid.XsidBavarianLoader(revision="0.1")
    # The pinned version, and "unpinned", are both fine.
    assert xsid.XsidBavarianLoader(revision=xsid.ZENODO_VERSION)._revision == "0.2"
    assert xsid.XsidBavarianLoader()._revision == "0.2"


# ── The control pair and the publication guardrails ──────────────────────────


def test_the_pair_is_registered_together() -> None:
    """Bavarian without its control spur is a number about one voice."""
    assert "xsid-bar" in WER_LOADERS and "xsid-de-control" in WER_LOADERS
    assert "xsid-bar" in WER_DATASETS and "xsid-de-control" in WER_DATASETS
    assert (
        WER_DATASETS["xsid-bar"].loader == WER_DATASETS["xsid-de-control"].loader
    ), "the pair must stay in one module so neither can be retired alone"


def test_both_ids_are_low_representativeness_and_out_of_every_aggregate() -> None:
    for dataset_id in ("xsid-bar", "xsid-de-control"):
        spec = WER_DATASETS[dataset_id]
        assert spec.representativeness == "low", (
            f"{dataset_id}: one speaker, read speech — 'low' is what forbids a "
            f"winner mark on the published row"
        )
        assert spec.eligible_for_aggregate is False, (
            f"{dataset_id}: a dialect probe inside a cross-dataset average "
            f"silently moves that average"
        )


# Every dialect corpus in the repo, Swiss and Bavarian. Written out rather than
# derived, so that adding a dialect corpus and forgetting to bar it from the
# aggregates fails here instead of quietly shifting a published average.
ALL_DIALECT_DATASET_IDS = {
    "spc-test",
    "fhnw-all-dialects",
    "xsid-bar",
    "xsid-de-control",
}


def test_no_other_dataset_was_accidentally_excluded_from_aggregates() -> None:
    """Guard the flag from becoming a quiet way to hide a bad number."""
    excluded = {k for k, v in WER_DATASETS.items() if not v.eligible_for_aggregate}
    assert excluded == ALL_DIALECT_DATASET_IDS


def test_aggregate_exclusion_is_exactly_the_dialect_corpora() -> None:
    """`DIALECT_DATASET_IDS` is a derived view, and must stay one.

    Two hand-maintained lists of "which corpora are dialect corpora" is how one
    falls behind the other, invisibly: a forgotten id does not crash, it just
    enters an average it does not belong in. If a dataset is ever barred from
    aggregates for some reason *other* than being a dialect corpus, this test is
    where that has to be decided deliberately rather than discovered later.
    """
    assert DIALECT_DATASET_IDS == ALL_DIALECT_DATASET_IDS


def test_the_pair_pins_a_doi_and_carries_the_no_synthesis_condition() -> None:
    for dataset_id in ("xsid-bar", "xsid-de-control"):
        spec = WER_DATASETS[dataset_id]
        assert spec.durability == "doi"
        assert spec.revision == xsid.ZENODO_VERSION
        assert "CC-BY-SA-4.0" in spec.license
        assert "synthesis" in spec.license, (
            f"{dataset_id}: the authors' condition is not a licence term and "
            f"does not travel on its own — name it on the spec"
        )


def test_notice_carries_the_condition_verbatim() -> None:
    """Paraphrasing it is the failure mode; the words are the obligation."""
    notice = (Path(__file__).resolve().parents[1] / "NOTICE").read_text(
        encoding="utf-8"
    )
    # The NOTICE wraps the sentence, so compare on collapsed whitespace.
    flat = " ".join(notice.split())
    assert " ".join(xsid.NO_SYNTHESIS_CONDITION.split()) in flat
    assert "10.5281/zenodo.21605015" in notice


def test_metric_hints_describe_the_two_different_tasks() -> None:
    assert xsid.XsidBavarianLoader.metric_hint == "bleu+wer"  # dialect → standard
    assert xsid.XsidGermanControlLoader.metric_hint == "wer"  # plain dictation


def test_loader_classes_resolve_through_the_registry() -> None:
    assert load_loader_class("xsid-bar") is xsid.XsidBavarianLoader
    assert load_loader_class("xsid-de-control") is xsid.XsidGermanControlLoader


def test_unknown_subset_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown subset"):
        next(xsid.XsidBavarianLoader().iter_samples("fleurs"))


# ── Against synthetic corpus bytes (no network, real zip/TSV plumbing) ───────


def _write_fake_corpus(root: Path) -> None:
    """A miniature corpus with the real padding mismatch baked in."""
    import numpy as np
    import soundfile as sf

    root.mkdir(parents=True, exist_ok=True)
    for variety in ("de-ba", "de"):
        buf = root / f"{variety}.zip"
        with zipfile.ZipFile(buf, "w") as zf:
            for split, n in (("test", 2), ("valid", 2)):
                for i in range(1, n + 1):
                    wav = root / "tmp.wav"
                    sf.write(wav, np.zeros(160, dtype="float32"), 16000)
                    # zip uses ZERO-PADDED names …
                    zf.writestr(
                        f"{variety}/{split}/xsid_{variety}_{split}_{i:03d}.wav",
                        wav.read_bytes(),
                    )
                    # … plus an AppleDouble side-file that must be ignored.
                    zf.writestr(
                        f"__MACOSX/{variety}/{split}/._x_{i:03d}.wav", b"junk"
                    )
        for split, n in (("test", 2), ("valid", 2)):
            rows = ["ID\tText\tIntent\tAudio\tText_original"]
            for i in range(1, n + 1):
                text = f"{variety} {split} {i}"
                # … while the TSV uses UNPADDED names — the upstream defect.
                rows.append(
                    f"{split}-{i}\t{text}\tintent{i}\t"
                    f"{variety}/{split}/xsid_{variety}_{split}_{i}.wav\torig"
                )
            (root / f"xsid_{variety}_{split}.tsv").write_text(
                "\n".join(rows) + "\n", encoding="utf-8"
            )
    (root / "tmp.wav").unlink()


def test_iter_samples_joins_across_the_padding_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("soundfile")
    base = tmp_path / "bavarian"
    _write_fake_corpus(base / xsid.SUBDIR)
    monkeypatch.setenv(xsid.XSID_CORPORA_DIR_ENV, str(base))
    # prepare() would checksum the synthetic bytes against the real manifest.
    monkeypatch.setattr(
        xsid._XsidAudioLoader, "prepare", lambda self, *a, **k: base / xsid.SUBDIR
    )

    samples = list(xsid.XsidBavarianLoader().iter_samples("xsid-bar"))
    assert len(samples) == 4, "every unpadded TSV row must find its padded member"
    # Reference is the STANDARD GERMAN text, not the dialect text.
    assert samples[0].reference.startswith("de ")
    assert samples[0].metadata["reference_dialect"].startswith("de-ba ")
    assert samples[0].metadata["dialect_region"] == "oberbayern-laendlich"
    assert samples[0].subset == "xsid-bar"
    assert all("__MACOSX" not in s.metadata["clip"] for s in samples)

    control = list(xsid.XsidGermanControlLoader().iter_samples("xsid-de-control"))
    assert len(control) == 4
    # The control scores German audio against the same German text.
    assert control[0].reference == control[0].metadata["reference_dialect"]

    assert len(list(xsid.XsidBavarianLoader().iter_samples("xsid-bar", limit=3))) == 3


# ── Against the real bytes, when a mirror is present ─────────────────────────

_REAL_ROOT = xsid.corpora_dir() / xsid.SUBDIR
_HAVE_REAL = all((_REAL_ROOT / n).exists() for n in xsid.ARTIFACTS)
real_corpus = pytest.mark.skipif(
    not _HAVE_REAL,
    reason=f"no xSID mirror under {_REAL_ROOT} (set XSID_CORPORA_DIR)",
)


@real_corpus
def test_real_valid_split_resolves_all_300_bavarian_clips() -> None:
    """The regression the padding fix exists for: 300, not 201."""
    loader = xsid.XsidBavarianLoader()
    root = loader.prepare()
    with zipfile.ZipFile(root / "de-ba.zip") as zf:
        raw_members = {
            n
            for n in zf.namelist()
            if n.endswith(".wav") and not n.startswith("__MACOSX/")
        }
    keyed = {xsid._member_key(n) for n in raw_members}

    rows = xsid._tsv_rows(root / "xsid_de-ba_valid.tsv")
    assert len(rows) == 300
    resolved = [r for r in rows if xsid._member_key(r["Audio"]) in keyed]
    assert len(resolved) == 300, (
        f"only {len(resolved)}/300 Bavarian validation clips resolved — the "
        f"_1 vs _001 padding fix regressed"
    )
    # Compare against the RAW member names: that is the join a literal path
    # lookup performs, and it is the one that loses 99 clips without an error.
    naive = [r for r in rows if r["Audio"] in raw_members]
    assert len(naive) == 201, (
        f"expected the upstream padding defect to cost exactly 99 of 300 "
        f"validation clips on a literal join; got {len(naive)} — the corpus "
        f"changed, so re-check the pinned Zenodo version"
    )


@real_corpus
def test_real_corpus_yields_800_paired_utterances_per_variety() -> None:
    for loader, expected_min in (
        (xsid.XsidBavarianLoader(), 800),
        (xsid.XsidGermanControlLoader(), 800),
    ):
        n = sum(1 for _ in loader.iter_samples(loader.subset))
        assert n == expected_min, f"{loader.name}: {n} samples, expected 800"
