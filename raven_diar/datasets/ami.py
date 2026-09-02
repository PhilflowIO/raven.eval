"""AMI loader — 4-speaker meetings, gold from the canonical prepared RTTMs.

AMI (Augmented Multi-party Interaction, CC-BY-4.0) is the standard multi-speaker
meeting benchmark. Rather than re-derive RTTMs from the raw NXT annotations, this
loader consumes the community-canonical prepared gold from
``pyannote/AMI-diarization-setup`` (the ``only_words`` RTTMs used by the pyannote
and SpeechBrain recipes), pinned by commit. Audio is the Mix-Headset channel,
fetched per meeting from the Edinburgh AMI corpus mirror — the same URLs that
repo's ``download_ami.sh`` uses, restricted to the meetings of the selected
split so a test-split run pulls 16 files (~1.1 GB), not the whole corpus.

Layout after :meth:`prepare` (under ``<root>/ami/``)::

    ami/
      _setup/                         pinned clone of AMI-diarization-setup
      rttms/<split>/*.rttm            gold (copied from only_words/rttms/<split>)
      audio/<file_id>.Mix-Headset.wav fetched from the AMI mirror (CC-BY-4.0)

:meth:`prepare` is idempotent: gold is copied once, audio is fetched only for
meetings whose wav is missing. A download that fails is reported and the file is
left absent, so :meth:`iter_files` emits ``audio_path=None`` for it and the
runner skips it loudly (never a silent pass).
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from .base import DiarFile

DATASET_ID = "ami"
SETUP_REPO = "https://github.com/pyannote/AMI-diarization-setup.git"
# Mirror used by the setup repo's download_ami.sh (http:// there; the host now
# 301-redirects to https, so we go straight to https).
AUDIO_MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
AUDIO_HINT = (
    "download <file_id>.Mix-Headset.wav from "
    f"{AUDIO_MIRROR}/<file_id>/audio/ and place it under <root>/ami/audio/"
)

Fetcher = Callable[[str, Path], None]


def audio_url(file_id: str) -> str:
    """Mirror URL of one meeting's Mix-Headset wav (matches download_ami.sh)."""
    return f"{AUDIO_MIRROR}/{file_id}/audio/{file_id}.Mix-Headset.wav"


def _http_fetch(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` via a temp file so a partial download never
    masquerades as a finished wav."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)


def _head_of(clone: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _checkout_pinned(clone: Path, revision: str | None) -> None:
    """Ensure ``clone`` is the setup repo checked out at ``revision``.

    Clones shallowly when absent. When the clone already exists (a previous
    run, or a copy placed by hand) its HEAD is verified against the pin and
    moved there if it differs — a stale clone must never feed gold RTTMs from
    a different commit than the one the published number claims. GitHub serves
    reachable commits by full SHA to a shallow fetch, so a pinned 40-hex
    revision resolves without pulling the whole history.
    """
    if not clone.is_dir():
        subprocess.run(
            ["git", "clone", "--depth", "1", SETUP_REPO, str(clone)], check=True
        )
    if not revision:
        return
    head = _head_of(clone)
    if head == revision or head.startswith(revision):
        return
    subprocess.run(
        ["git", "-C", str(clone), "fetch", "--depth", "1", "origin", revision],
        check=True,
    )
    subprocess.run(["git", "-C", str(clone), "checkout", revision], check=True)
    head = _head_of(clone)
    if not head.startswith(revision):
        raise RuntimeError(
            f"AMI setup clone at {clone} is at {head}, expected pinned {revision}"
        )


class AMILoader:
    """Yields (Mix-Headset audio, prepared gold RTTM) pairs for AMI."""

    name = DATASET_ID
    dataset_id = DATASET_ID

    def __init__(self, split: str = "test", fetcher: Fetcher = _http_fetch) -> None:
        self._split = split
        self._fetch = fetcher

    def _base(self, root: Path) -> Path:
        return root / "ami"

    def _prepare_gold(self, base: Path, revision: str | None) -> Path:
        rttms_root = base / "rttms"
        split_dir = rttms_root / self._split
        if split_dir.is_dir() and any(split_dir.glob("*.rttm")):
            return split_dir
        _checkout_pinned(base / "_setup", revision)
        clone = base / "_setup"
        # Canonical gold lives under only_words/rttms/{train,dev,test}.
        src = clone / "only_words" / "rttms" / self._split
        if not src.is_dir():
            raise FileNotFoundError(
                f"expected prepared RTTMs at {src}; AMI-diarization-setup layout "
                f"changed or split '{self._split}' unknown — check the pinned revision"
            )
        split_dir.mkdir(parents=True, exist_ok=True)
        for rttm in src.glob("*.rttm"):
            (split_dir / rttm.name).write_bytes(rttm.read_bytes())
        return split_dir

    def _prepare_audio(self, base: Path, split_dir: Path) -> None:
        audio_dir = base / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        wanted = sorted(p.stem for p in split_dir.glob("*.rttm"))
        missing = [fid for fid in wanted if not (audio_dir / f"{fid}.Mix-Headset.wav").exists()]
        if not missing:
            return
        print(f"[ami] fetching {len(missing)}/{len(wanted)} Mix-Headset wavs from the AMI mirror")
        failed: list[str] = []
        for fid in missing:
            dest = audio_dir / f"{fid}.Mix-Headset.wav"
            try:
                self._fetch(audio_url(fid), dest)
                print(f"[ami]   {fid}: {dest.stat().st_size / 1e6:.1f} MB")
            except Exception as exc:  # noqa: BLE001 — report, keep going, skip loudly later
                failed.append(fid)
                print(f"[ami]   {fid}: download FAILED ({exc})")
        if failed:
            print(
                f"[ami] {len(failed)} file(s) without audio will be skipped: "
                f"{', '.join(failed)} — {AUDIO_HINT}"
            )

    def prepare(self, root: Path, revision: str | None = None) -> None:
        base = self._base(root)
        base.mkdir(parents=True, exist_ok=True)
        split_dir = self._prepare_gold(base, revision)
        self._prepare_audio(base, split_dir)

    def iter_files(self, root: Path, limit: int | None = None) -> Iterator[DiarFile]:
        base = self._base(root)
        rttms_dir = base / "rttms" / self._split
        audio_dir = base / "audio"
        if not rttms_dir.is_dir():
            raise FileNotFoundError(
                f"no gold RTTMs at {rttms_dir} — run prepare() first "
                f"(git clone of {SETUP_REPO})"
            )
        for emitted, rttm in enumerate(sorted(rttms_dir.glob("*.rttm"))):
            file_id = rttm.stem
            wav = audio_dir / f"{file_id}.Mix-Headset.wav"
            yield DiarFile(
                file_id=file_id,
                audio_path=wav if wav.exists() else None,
                gold_rttm_path=rttm,
                dataset=DATASET_ID,
            )
            if limit is not None and emitted + 1 >= limit:
                return
