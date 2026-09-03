"""xSID-audio loader — Bavarian read speech + its Standard German control spur.

WHY this corpus is in the public benchmark: it is the only dialect factor Raven
publishes that a stranger can re-derive. Bavarian audio alone is not a dialect
statement — the same person recorded both varieties, so the interesting figure is
the DELTA ``xsid-bar`` minus ``xsid-de-control``. Ship both ids or neither.

Provenance
    source        Zenodo record 21605015 — https://zenodo.org/records/21605015
    doi           10.5281/zenodo.21605015
    version       0.2, published 2026-07-27 by Verena Blaschke, Miriam Winkler
                  and Barbara Plank (MaiNLP, LMU Munich); audio companion to the
                  ACL 2026 paper "Standard-to-Dialect Transfer Trends Differ
                  across Text and Speech" (aclanthology.org/2026.acl-long.309/).
    acquisition   every artifact is fetched by :meth:`prepare` from the documented
                  Zenodo content URL and its sha256 verified on EVERY call —
                  never trusted because a file of the right name is on disk. A
                  local mirror (``XSID_CORPORA_DIR`` / ``BAVARIAN_CORPORA_DIR`` /
                  ``BENCH_CORPORA_DIR``) is a fast path, not the only path.
    durability    rank "doi" — a real archive with a DOI and version history, so
                  unlike a vendor link the pinned bytes stay retrievable.

Licence: **CC BY-SA 4.0** (Zenodo API ``metadata.license.id = cc-by-sa-4.0``,
verified 2026-09-02). The authors add one restriction that is NOT a licence term
and therefore does not travel with the licence — it is reproduced verbatim in
:data:`NO_SYNTHESIS_CONDITION` and in ``/NOTICE``. We only run recognition.

MEASUREMENT CAVEATS — these must travel with any published number:
  * **Read speech**, not spontaneous: sentences read aloud from the xSID intent
    corpus; no disfluencies, no overlap, no meeting acoustics.
  * **ONE speaker for both varieties** (Zenodo description + README). A number
    from this set is a PROBE (does the model collapse on Bavarian at all?), not a
    representative Bavarian benchmark — it cannot separate speaker idiosyncrasy
    from dialect effect. ``representativeness = "low"`` encodes exactly that, and
    ``benchmark.config.yaml`` forbids a winner mark on these rows.
  * **Never aggregate across dialect areas.** A mean spanning Swiss German and
    Bavarian describes no population. Dialect ids also stay out of any ``overall``
    average.
  * Short utterances (voice-assistant commands), so per-item WER is noisy.

Scoring shape: Bavarian audio is scored against the PARALLEL Standard German
sentence for the same ``ID`` in ``xsid_de_<split>.tsv`` — a translation-style
task, hence ``metric_hint = "bleu+wer"``. BLEU itself is not implemented in this
repo yet; until it lands these rows carry WER only. The dialectal transcription is
kept in ``Sample.metadata['reference_dialect']`` so a dialect-faithful score can
be computed later without a re-fetch.

Upstream defect this loader fixes (verified 2026-09-02): ``xsid_de-ba_valid.tsv``
references clips as ``..._valid_1.wav`` while ``de-ba.zip`` stores
``..._valid_001.wav``. A literal path join silently drops 99 of the 300 Bavarian
validation clips. Both sides go through :func:`_member_key`, so the join is on
(stem, integer index) rather than on padding upstream does not guarantee.

TSV columns (verified 2026-09-02): ``ID, Text, Intent, Audio, Text_original``.
``Text`` is the recorded wording, ``Text_original`` the original xSID string —
we score against ``Text``, the thing that was actually spoken. The zips are read
member-wise via ``zipfile`` (no extraction), so a ``--limit`` run touches only
the members it needs instead of unpacking 430 MB.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .base import Sample

# ── Provenance pins ──────────────────────────────────────────────────────────

ZENODO_RECORD: Final[str] = "21605015"
ZENODO_DOI: Final[str] = "10.5281/zenodo.21605015"
#: Record VERSION, not a git sha — this is the pin a published number claims.
ZENODO_VERSION: Final[str] = "0.2"
ZENODO_URL: Final[str] = f"https://zenodo.org/records/{ZENODO_RECORD}"
_CONTENT_URL: Final[str] = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{{name}}/content"
)

#: Verbatim, from the Zenodo record description and the corpus README. NOT a
#: licence term — CC BY-SA 4.0 does not carry it, so it does not propagate on its
#: own and must be restated wherever this audio goes. Mirrored in /NOTICE.
NO_SYNTHESIS_CONDITION: Final[str] = (
    "We share the audio recordings for research on processing spoken language "
    "data, but do not permit their use in the context of speech synthesis or "
    "voice cloning."
)


@dataclass(frozen=True)
class _Artifact:
    """One Zenodo file: its size and the sha256 a fetch must reproduce."""

    size: int
    #: md5 as reported by the Zenodo API (re-verified after download, 2026-09-02).
    md5: str
    #: sha256 computed locally over the verified bytes — what we check on.
    sha256: str


#: Every artifact this loader reads, with the checksums recorded in the mirror's
#: SOURCES.md (Zenodo-API md5 re-verified locally 2026-09-02; sha256 ours).
ARTIFACTS: Final[dict[str, _Artifact]] = {
    "de-ba.zip": _Artifact(
        size=209_309_866,
        md5="828a13112562562887add9b5160644e1",
        sha256="ca553ec8762230851b426520547ea22af1661e11d6ae164dd9371ae297d28a7e",
    ),
    "de.zip": _Artifact(
        size=220_966_920,
        md5="10a2d23acec7773fd0dc29d484d0cd3b",
        sha256="ad2bb6cd39dbae975624aefbe6eeefb31d9a41a6a26a12ae170f20d3c4c591aa",
    ),
    "xsid_de-ba_test.tsv": _Artifact(
        size=70_188,
        md5="2aaa58f5f6cf7a69aa47ed766deede33",
        sha256="3e5d3812341daa580e5f107eae7a3e30ba4079607ab68f213e2b1881b3803ee4",
    ),
    "xsid_de-ba_valid.tsv": _Artifact(
        size=42_429,
        md5="b004d375d5b30e8640a005d4651a1acb",
        sha256="c1a91ec8aec1b515255e637b734dcd524bece7263c02e193aebbbe19df76788e",
    ),
    "xsid_de_test.tsv": _Artifact(
        size=71_470,
        md5="fbf954585cdaaa682927e6ac659fbd47",
        sha256="9840e4d4dda81a34e56c4bacfd3d0ed380ae5749d6cb79e4904c75e30f09569f",
    ),
    "xsid_de_valid.tsv": _Artifact(
        size=43_717,
        md5="31e7ab6acbdd11f87c77839331efd8fc",
        sha256="675201c0e610db30315c4c615c5b1fcc10d9376d79c54bbbb232bea7be04975b",
    ),
}

# ── Layout ───────────────────────────────────────────────────────────────────

SUBDIR: Final[str] = "xsid-audio"
#: test first: the larger split (500 vs 300) and the one the paper reports.
SPLITS: Final[tuple[str, ...]] = ("test", "valid")
#: The scoring reference is ALWAYS the Standard German TSV, whatever variety's
#: audio we iterate — that is what makes the Bavarian row a translation task.
REFERENCE_VARIETY: Final[str] = "de"

#: Full-path override for this corpus alone.
XSID_CORPORA_DIR_ENV: Final[str] = "XSID_CORPORA_DIR"
#: Region override shared with the internal harness's Bavarian mirror.
BAVARIAN_CORPORA_DIR_ENV: Final[str] = "BAVARIAN_CORPORA_DIR"
#: Root of all locally mirrored bench corpora, one subdirectory per region.
BENCH_CORPORA_DIR_ENV: Final[str] = "BENCH_CORPORA_DIR"


def corpora_dir() -> Path:
    """Resolve where the xSID artifacts live (and are fetched to).

    Precedence, most specific first::

        XSID_CORPORA_DIR        full path to the dir holding ``xsid-audio/``
        BAVARIAN_CORPORA_DIR    same, for the whole Bavarian region
        BENCH_CORPORA_DIR       <root>/bavarian
        default                 ~/.cache/raven-eval/corpora/bavarian

    The default is a cache dir on purpose: a clean checkout with no environment
    at all must still be able to run, by downloading. The env vars only make an
    existing mirror a fast path.
    """
    for env_var in (XSID_CORPORA_DIR_ENV, BAVARIAN_CORPORA_DIR_ENV):
        override = os.environ.get(env_var)
        if override:
            return Path(override)
    root = os.environ.get(BENCH_CORPORA_DIR_ENV)
    if root:
        return Path(root) / "bavarian"
    return Path.home() / ".cache" / "raven-eval" / "corpora" / "bavarian"


# ── Acquisition ──────────────────────────────────────────────────────────────


def artifact_url(name: str) -> str:
    """Documented Zenodo content URL of one artifact."""
    if name not in ARTIFACTS:
        raise KeyError(f"unknown xSID artifact {name!r}; known: {sorted(ARTIFACTS)}")
    return _CONTENT_URL.format(name=name)


def sha256_of(path: Path) -> str:
    """Stream-hash a file (the zips are ~200 MB — never read them whole)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_fetch(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` via a temp file, so a partial download never
    masquerades as a finished artifact."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)


def ensure_artifact(name: str, base: Path, *, fetcher=_http_fetch) -> Path:
    """Return a verified local path to ``name``, downloading it if absent.

    The sha256 is checked on EVERY call, not only after a download: a mirror is
    just a cache, and a published number must not rest on "a file of that name
    was on disk". A checksum mismatch is fatal — we do not silently re-download
    over bytes somebody may have edited on purpose.
    """
    spec = ARTIFACTS[name]
    dest = base / name
    if not dest.exists():
        base.mkdir(parents=True, exist_ok=True)
        url = artifact_url(name)
        print(f"[xsid] fetching {name} ({spec.size / 1e6:.1f} MB) from {url}")
        fetcher(url, dest)
    actual = sha256_of(dest)
    if actual != spec.sha256:
        raise RuntimeError(
            f"xSID artifact {dest} failed verification: sha256 {actual}, "
            f"expected {spec.sha256} (Zenodo record {ZENODO_RECORD} v{ZENODO_VERSION}, "
            f"DOI {ZENODO_DOI}). Delete the file to re-fetch, or point "
            f"{XSID_CORPORA_DIR_ENV} at an unmodified mirror."
        )
    return dest


# ── Upstream padding defect ──────────────────────────────────────────────────

_CLIP_INDEX_RE = re.compile(r"^(?P<stem>.*_)(?P<idx>\d+)\.wav$")


def _member_key(name: str) -> str:
    """Normalize a clip path so the TSV and the zip agree on zero-padding.

    ``xsid_de-ba_valid.tsv`` says ``..._valid_1.wav``; ``de-ba.zip`` stores
    ``..._valid_001.wav``. Matching on the literal string drops 99 of the 300
    Bavarian validation clips without any error. Both sides pass through here, so
    the join key is (stem, integer index).
    """
    m = _CLIP_INDEX_RE.match(name)
    if not m:
        return name
    return f"{m.group('stem')}{int(m.group('idx')):d}.wav"


def _tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def _decode(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode an in-memory WAV to float32 mono (soundfile sniffs the container)."""
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sr)


# ── Loaders ──────────────────────────────────────────────────────────────────


class _XsidAudioLoader:
    """Shared body; subclasses pick which variety's audio is iterated."""

    # --- subclass contract ---------------------------------------------------
    name: str = ""
    #: == benchmark.config.yaml datasets.wer[].id
    subset: str = ""
    variety: str = "de-ba"
    dialect_region: str = ""
    metric_hint: str = "wer"

    # --- provenance (uniform across every loader in this repo) ---------------
    hf_dataset_id: str = f"zenodo.org/records/{ZENODO_RECORD}"  # not HF-hosted
    hf_config: str | None = None
    hf_split: str = "test+valid"
    audio_column: str = "Audio"
    text_column: str = "Text"
    license: str = "CC-BY-SA-4.0"
    source_url: str = ZENODO_URL
    #: Single speaker, read speech — a probe, never a representative benchmark.
    representativeness: str = "low"

    def __init__(self, streaming: bool = False, revision: str | None = None) -> None:
        # ``streaming`` is accepted for the shared runner signature and ignored:
        # the zips are read member-wise, so nothing is materialized up front.
        self._streaming = streaming
        # The runner threads WerDatasetSpec.revision here. For a Zenodo record
        # that is the record VERSION; a mismatch means the caller asked for a
        # different publication than the checksums in this module describe.
        if revision is not None and revision != ZENODO_VERSION:
            raise ValueError(
                f"{self.name}: pinned Zenodo version is {ZENODO_VERSION!r}, caller "
                f"asked for {revision!r}. The sha256 manifest in this module "
                f"describes exactly v{ZENODO_VERSION} of record {ZENODO_RECORD}; "
                f"another version needs its own checksums."
            )
        self._revision = revision or ZENODO_VERSION

    # ----- acquisition -------------------------------------------------------

    def required_artifacts(self) -> list[str]:
        """The artifacts this variety needs — its own zip plus both TSV sets.

        The Standard German TSVs are required for BOTH ids: the Bavarian row is
        scored against them, and the control row IS them.
        """
        names = [f"{self.variety}.zip"]
        for variety in dict.fromkeys((self.variety, REFERENCE_VARIETY)):
            names += [f"xsid_{variety}_{split}.tsv" for split in SPLITS]
        return names

    def prepare(self, base: Path | None = None, *, fetcher=_http_fetch) -> Path:
        """Fetch (if needed) and verify every artifact; return the corpus dir."""
        root = (base or corpora_dir()) / SUBDIR
        for name in self.required_artifacts():
            ensure_artifact(name, root, fetcher=fetcher)
        return root

    # ----- iteration ---------------------------------------------------------

    def iter_samples(self, subset: str, limit: int | None = None) -> Iterator[Sample]:
        if subset not in {self.subset, "All"}:
            raise ValueError(
                f"unknown subset {subset!r} for {self.name}; expected "
                f"{self.subset!r} or 'All'"
            )
        root = self.prepare()
        emitted = 0
        with zipfile.ZipFile(root / f"{self.variety}.zip") as zf:
            members = {
                _member_key(n): n
                for n in zf.namelist()
                # __MACOSX/ AppleDouble side-files shadow the real members.
                if n.endswith(".wav") and not n.startswith("__MACOSX/")
            }
            for split in SPLITS:
                rows = _tsv_rows(root / f"xsid_{self.variety}_{split}.tsv")
                german = {
                    r["ID"]: r
                    for r in _tsv_rows(root / f"xsid_{REFERENCE_VARIETY}_{split}.tsv")
                }
                for row in rows:
                    ref_row = german.get(row["ID"])
                    if ref_row is None:
                        # No parallel Standard German sentence → no WER reference.
                        # Skip rather than score against the dialect text, which
                        # would silently turn a translation into a dictation.
                        continue
                    member = members.get(_member_key(row["Audio"]))
                    if member is None:
                        continue
                    audio, sr = _decode(zf.read(member))
                    yield Sample(
                        audio=audio,
                        sample_rate=sr,
                        reference=str(ref_row["Text"]),
                        sample_id=f"{self.name}#{emitted}",
                        subset=self.subset,
                        metadata={
                            "clip": member,
                            "split": split,
                            "xsid_id": row["ID"],
                            "intent": row.get("Intent", ""),
                            "dialect_region": self.dialect_region,
                            # Dialectal transcription of the same utterance — a
                            # dialect-faithful score can use this without refetching.
                            "reference_dialect": str(row["Text"]),
                        },
                    )
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return


class XsidBavarianLoader(_XsidAudioLoader):
    """Bavarian audio scored against the parallel Standard German text."""

    name = "xsid-bar"
    subset = "xsid-bar"
    variety = "de-ba"
    dialect_region = "oberbayern-laendlich"
    metric_hint = "bleu+wer"  # Bavarian audio → Standard German text


class XsidGermanControlLoader(_XsidAudioLoader):
    """Standard German control: same speaker, same sentences, no dialect.

    Registered on purpose and not optional. Both halves share the single speaker
    whose voice would otherwise be an unmeasurable confounder, so the Bavarian
    number is only interpretable as a delta against this one. Publishing
    ``xsid-bar`` without ``xsid-de-control`` is worse than publishing neither.
    """

    name = "xsid-de-control"
    subset = "xsid-de-control"
    variety = "de"
    dialect_region = "hochdeutsch"
    metric_hint = "wer"  # plain dictation: German audio → German text
