"""Fetch-and-verify acquisition for corpora that are loose archives, not HF repos.

Every other WER loader in this repo reaches its corpus through
``datasets.load_dataset`` at a pinned HF revision — the Hub is content-addressed,
so the revision *is* the checksum. The Swiss dialect corpora are not shaped like
that: they are loose files behind a plain URL (parquet shards on the Hub's
``resolve`` endpoint, a zip on a Dropbox share). For those, the pin has to be an
explicit sha256, and the harness has to do the fetching itself.

ADR-app-0054 states the rule this module implements: *reproducibility is a
property of the acquisition path, not of the licence*. Concretely:

  * every artefact carries URL + sha256 + byte size, recorded in
    :class:`RemoteArtifact` next to the loader that consumes it;
  * :func:`ensure_artifact` verifies the digest on acquisition and refuses to
    hand back bytes that do not match, so a silent upstream substitution fails
    loudly instead of shifting a published number;
  * a local mirror is a *fast path*, never the only path. A missing file is
    downloaded; a present file is verified once and marked.

Why the marker file: these artefacts are 0.4-1.6 GB each, and re-hashing ~2.4 GB
on every ``--limit 3`` smoke run would make the fast path slower than the
download it avoids. So the digest is checked once per file and the result
recorded in a sibling ``.<name>.sha256`` marker; the marker is only trusted while
the file's size and mtime still match what was verified. Delete the marker (or
pass ``force=True``) to force a re-hash.

Directory resolution deliberately honours the same env vars the internal
flow.raven harness uses (``BENCH_CORPORA_DIR`` / ``SWISS_CORPORA_DIR``), so a
workstation that already mirrors these corpora is picked up without copying
2.4 GB a second time. A clean checkout with no env set falls back to a cache dir
under the user's home and downloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Root for every locally materialised bench corpus, one subdirectory per region
# ("swiss/", ...). Each region dir is expected to look like the layout the
# loaders document; nothing outside this module writes to it.
CORPORA_DIR_ENV = "RAVEN_EVAL_CORPORA_DIR"
# Same env var name the internal flow.raven harness uses, honoured second so an
# existing workstation mirror is reused rather than re-downloaded.
BENCH_CORPORA_DIR_ENV = "BENCH_CORPORA_DIR"
SWISS_CORPORA_DIR_ENV = "SWISS_CORPORA_DIR"

DEFAULT_CORPORA_DIR = Path.home() / ".cache" / "raven.eval" / "corpora"

_CHUNK = 1 << 20

Fetcher = Callable[[str, Path], None]


def corpora_root() -> Path:
    """Resolve the root under which fetched corpora are materialised."""
    for env in (CORPORA_DIR_ENV, BENCH_CORPORA_DIR_ENV):
        override = os.environ.get(env)
        if override:
            return Path(override).expanduser()
    return DEFAULT_CORPORA_DIR


def region_corpora_dir(region: str, env_var: str) -> Path:
    """Resolve one region's corpus dir; the region env var wins as a full path."""
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    return corpora_root() / region


def swiss_corpora_dir() -> Path:
    """Resolve the Swiss corpora dir (``SWISS_CORPORA_DIR`` or ``<root>/swiss``)."""
    return region_corpora_dir("swiss", SWISS_CORPORA_DIR_ENV)


@dataclass(frozen=True)
class RemoteArtifact:
    """One fetchable file, pinned by digest rather than by an upstream revision.

    ``url`` is the acquisition path a stranger can follow; ``sha256`` is what the
    published number was measured on. ``durability`` mirrors the ranking in
    ADR-app-0054 — "doi" (DOI / institutional archive) > "hf" (Hub file at a
    pinned revision) > "vendor" (vendor-hosted share link: usable, citable, but
    with no version history and no institutional commitment, so it is carried as
    a tracked liability and never silently relied on).
    """

    url: str
    filename: str
    sha256: str
    size_bytes: int
    durability: str = "vendor"
    note: str = ""


def sha256_of(path: Path) -> str:
    """Streamed sha256 of a file (these artefacts do not fit in memory)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.sha256")


def _read_marker(path: Path) -> dict[str, object] | None:
    marker = _marker_path(path)
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _write_marker(path: Path, digest: str) -> None:
    stat = path.stat()
    _marker_path(path).write_text(
        json.dumps(
            {"sha256": digest, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            indent=0,
        ),
        encoding="utf-8",
    )


def _marker_still_valid(path: Path, expected: str) -> bool:
    data = _read_marker(path)
    if data is None or data.get("sha256") != expected:
        return False
    stat = path.stat()
    return data.get("size") == stat.st_size and data.get("mtime_ns") == stat.st_mtime_ns


def http_fetch(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` through a temp file.

    The temp file matters: a partially downloaded 1.6 GB zip that carries the
    final name would be verified, fail, and look like upstream corruption rather
    than an interrupted download.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "raven.eval/0.1"})
    with urllib.request.urlopen(request, timeout=300) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=_CHUNK)
    tmp.replace(dest)


class ChecksumMismatch(RuntimeError):
    """A fetched or mirrored artefact does not carry the pinned digest."""


def verify(path: Path, artifact: RemoteArtifact) -> str:
    """Hash ``path`` and raise :class:`ChecksumMismatch` unless it matches."""
    actual = sha256_of(path)
    if actual != artifact.sha256:
        raise ChecksumMismatch(
            f"{path} does not carry the pinned digest for {artifact.filename}:\n"
            f"  expected sha256 {artifact.sha256}\n"
            f"  actual   sha256 {actual}\n"
            f"  source   {artifact.url}\n"
            "The published number was measured on the expected bytes. Either the "
            "upstream artefact was replaced (in which case the number needs "
            "re-measuring, not a relaxed check) or the local copy is damaged — "
            "delete it and let the harness re-fetch."
        )
    return actual


def ensure_artifact(
    artifact: RemoteArtifact,
    directory: Path,
    *,
    fetcher: Fetcher = http_fetch,
    force_verify: bool = False,
) -> Path:
    """Return a local path to ``artifact``, downloading it if absent.

    Verifies the sha256 on first sight of a file and records the result; a
    subsequent call trusts the marker only while size and mtime are unchanged.
    """
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / artifact.filename

    if dest.is_file():
        if not force_verify and _marker_still_valid(dest, artifact.sha256):
            return dest
        print(f"[corpora] verifying {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        verify(dest, artifact)
        _write_marker(dest, artifact.sha256)
        return dest

    print(
        f"[corpora] fetching {artifact.filename} "
        f"({artifact.size_bytes / 1e6:.0f} MB) from {artifact.url}"
    )
    fetcher(artifact.url, dest)
    verify(dest, artifact)
    _write_marker(dest, artifact.sha256)
    return dest


def decode_audio_bytes(raw: bytes) -> tuple["object", int]:
    """Decode an in-memory audio container (FLAC/WAV/MP3/…) to float32 mono.

    soundfile sniffs the container from the bytes, so this stays format-agnostic.
    Returns the file's **native** sample rate — no resampling here. That is
    deliberate: the SPC FLAC payloads are 48 kHz even though the HF card declares
    a 16 kHz ``Audio()`` feature, and silently pretending otherwise would send
    every adapter a mislabelled WAV header.
    """
    import io

    import numpy as np
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype="float32"), int(sr)


__all__ = [
    "CORPORA_DIR_ENV",
    "ChecksumMismatch",
    "RemoteArtifact",
    "corpora_root",
    "decode_audio_bytes",
    "ensure_artifact",
    "http_fetch",
    "sha256_of",
    "swiss_corpora_dir",
    "verify",
]
