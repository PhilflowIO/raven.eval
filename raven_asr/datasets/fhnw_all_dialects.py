"""FHNW "All Swiss German Dialects Test Set" — every region, not just Bern.

FHNW i4ds, SwissText 2021 shared task 3 ("Swiss German Speech to Standard German
Text"): 5,750 utterances / 12.72 h with a dialect distribution close to the real
distribution in Switzerland, paired with Standard German sentence text in an
adapted Common Voice layout. It is the corpus that makes a claim about
*Schweizerdeutsch* honest rather than a claim about Bern, which is why
ADR-app-0054 works this exact set through as its example.

Like SPC this is translation-shaped (Swiss German audio → Standard German text),
so it declares ``metric="bleu+wer"`` in ``raven_asr.config.WER_DATASETS``.

Provenance
    source        https://www.dropbox.com/s/rfmjqkdjox7xstq/clickworker_collection_1.zip?dl=1
    acquisition   the zip is fetched by URL and verified against
                  :data:`ARCHIVE`'s sha256 before it is unpacked
    durability    rank 3 ("vendor") — a **tracked liability**, see below
    license       MIT, verified verbatim against ``README.txt`` inside the
                  archive ("License:     MIT"), quoted in ``/NOTICE``

Durability — the one thing wrong with this corpus
    The problem is not citability. There is a direct URL and a checksum, so a
    stranger can fetch and verify the exact bytes every published number was
    measured on. The problem is that a Dropbox share link is not an archive: no
    version history, no institutional commitment, and it can disappear without
    notice, at which point a published number becomes unreproducible through no
    action of ours.

    ADR-app-0054 decides this case explicitly, and this loader implements that
    decision rather than reopening it:

      * keep the corpus — dropping it costs the breadth that makes the Swiss
        claim honest, and the fix is procurement, not deletion;
      * verify the sha256 on **every** acquisition, so a silent substitution
        fails loudly instead of shifting a number (:func:`ensure_artifact`);
      * pursue a durable mirror — the MIT licence permits self-hosting, and a
        Zenodo deposit would add a DOI and version history. **Still outstanding**
        as of 2026-09-03; ``durability="vendor"`` in the dataset spec is the
        machine-readable form of that liability;
      * only if the link dies *before* the mirror exists is the number marked on
        the page as measured-but-no-longer-externally-reproducible. Not
        pre-emptively.

Layout, after acquisition, under ``<swiss>/fhnw-all-dialects/Clickworker_Test_Set/``
    ``public.tsv``  columns ``client_id, path, sentence, up_votes, down_votes,
                    age, gender, accent``. ``accent`` is the speaker's dialect
                    region as a canton code; per the archive README, BS speakers
                    are labelled BL because the collection did not distinguish
                    the two. We read the public half only — ``private.tsv`` was
                    the hidden shared-task half and scores identically.
    ``clips.tar``   FLAC clips at ``clips/<uuid>.flac``. Members are extracted
                    lazily, so a ``--limit 3`` run reads three clips out of a
                    1.6 GB tar rather than unpacking it.

Sample rate: the clips decode at 44.1 kHz (measured on the first clips of the
public half, 2026-09-03). As with SPC we yield the native rate rather than a
convenient 16 kHz — the adapters build a WAV header from what the loader reports,
so a wrong number here is transcribed noise, not a rounding difference.

The dialect region travels in the ``sample_id`` (``fhnw-all-dialects-ZH-<clip>``)
because :class:`~raven_asr.datasets.base.Sample` is a fixed five-field contract
shared by every loader in this repo and must not grow a sixth field for one
corpus. Per-region scoring columns therefore parse the id; a richer sample
contract is a separate change with a separate blast radius.
"""

from __future__ import annotations

import csv
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from .base import Sample
from .local_archive import RemoteArtifact, ensure_artifact, swiss_corpora_dir
from .local_archive import decode_audio_bytes as _decode

DATASET_ID: Final[str] = "fhnw-all-dialects"
SOURCE_URL: Final[str] = (
    "https://www.dropbox.com/s/rfmjqkdjox7xstq/clickworker_collection_1.zip?dl=1"
)
SUBDIR: Final[Path] = Path("fhnw-all-dialects")
UNPACKED_DIR: Final[str] = "Clickworker_Test_Set"

#: The distribution archive, pinned by digest. Size and sha256 recorded in
#: ``raven-bench-corpora/swiss/SOURCES.md`` on 2026-09-01, the day the bytes the
#: published numbers were measured on were fetched.
ARCHIVE: Final[RemoteArtifact] = RemoteArtifact(
    url=SOURCE_URL,
    filename="clickworker_collection_1.zip",
    sha256="7ca2492143b6d418ca42d6407f939ce8d2c53f66999e05906931ccefe6ff3148",
    size_bytes=1_603_017_371,
    durability="vendor",
    note=(
        "Dropbox share link: citable and checksummed, but not an archive. "
        "Tracked liability per ADR-app-0054 — a durable mirror (self-hosted or "
        "a Zenodo deposit with a DOI) is outstanding."
    ),
)

TSV: Final[str] = "public.tsv"
TAR: Final[str] = "clips.tar"
TEXT_COLUMN: Final[str] = "sentence"

ALL_SUBSETS: Final[str] = "All"


class FhnwAllDialectsLoader:
    """Yields FHNW all-dialects utterances from the checksum-verified archive."""

    name = DATASET_ID
    subset = DATASET_ID
    hf_dataset_id = ""  # not on the Hub — the Dropbox distribution is the source
    hf_config = None
    hf_split = "test"
    audio_column = "path"
    text_column = TEXT_COLUMN
    license = "MIT"
    source_url = SOURCE_URL

    def __init__(self, streaming: bool = False, revision: str | None = None) -> None:
        # Accepted-and-ignored: the runner constructs every loader this way.
        # There is no upstream revision to pin here — the sha256 in ARCHIVE *is*
        # the pin, which is exactly why a vendor link needs one.
        del streaming
        if revision:
            raise ValueError(
                f"{DATASET_ID}: this corpus has no upstream revision to pin "
                f"(got {revision!r}); it is pinned by the archive sha256 "
                f"{ARCHIVE.sha256}."
            )

    # ----- acquisition -------------------------------------------------------

    def _prepare(self) -> Path:
        """Return the unpacked corpus dir, fetching and unzipping if needed."""
        region = swiss_corpora_dir()
        unpacked = region / SUBDIR / UNPACKED_DIR
        if (unpacked / TSV).is_file() and (unpacked / TAR).is_file():
            return unpacked

        archive = ensure_artifact(ARCHIVE, region)
        target = region / SUBDIR
        target.mkdir(parents=True, exist_ok=True)
        print(f"[{DATASET_ID}] unpacking {archive.name} into {target}")
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                # The archive ships a single top-level Clickworker_Test_Set/
                # directory; refuse anything that would escape it.
                if member.startswith("/") or ".." in Path(member).parts:
                    raise RuntimeError(
                        f"{DATASET_ID}: refusing to extract unsafe path {member!r} "
                        f"from {archive}"
                    )
            zf.extractall(target)

        missing = [rel for rel in (TSV, TAR) if not (unpacked / rel).is_file()]
        if missing:
            raise RuntimeError(
                f"{DATASET_ID}: {archive} unpacked into {target} but {missing} "
                f"are missing under {UNPACKED_DIR}/ — the archive layout changed."
            )
        return unpacked

    # ----- public contract ---------------------------------------------------

    def iter_samples(self, subset: str, limit: int | None = None) -> Iterator[Sample]:
        if subset not in (self.subset, ALL_SUBSETS):
            raise ValueError(
                f"unknown subset {subset!r} for loader {self.name!r}; "
                f"expected one of {(self.subset, ALL_SUBSETS)}"
            )
        root = self._prepare()

        with (root / TSV).open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))

        emitted = 0
        with tarfile.open(root / TAR) as tar:
            for row in rows:
                reference = str(row.get(TEXT_COLUMN) or "")
                if not reference:
                    continue
                member_name = f"clips/{row['path']}"
                try:
                    member = tar.getmember(member_name)
                except KeyError:
                    # Listed in the TSV, absent from the tar: skip this clip
                    # rather than abort a run over the other 5,749.
                    continue
                payload = tar.extractfile(member)
                if payload is None:
                    continue
                audio, sample_rate = _decode(payload.read())
                yield Sample(
                    audio=audio,
                    sample_rate=sample_rate,
                    reference=reference,
                    sample_id=sample_id_for(row),
                    subset=self.subset,
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def sample_id_for(row: dict[str, str]) -> str:
    """``fhnw-all-dialects-<region>-<clip stem>`` — region survives in the id.

    Kept a module-level function so a per-region scorer can parse ids with the
    same code that produced them instead of re-deriving the convention.
    """
    region = (row.get("accent") or "xx").strip() or "xx"
    clip = Path(row.get("path") or "").stem or "unknown"
    return f"{DATASET_ID}-{region}-{clip}"
