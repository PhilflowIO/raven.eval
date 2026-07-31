"""AMI loader — 4-speaker meetings, gold from the canonical prepared RTTMs.

AMI (Augmented Multi-party Interaction, CC-BY-4.0) is the standard multi-speaker
meeting benchmark. Rather than re-derive RTTMs from the raw NXT annotations, this
loader consumes the community-canonical prepared gold from
``pyannote/AMI-diarization-setup`` (the ``only_words`` RTTMs used by the pyannote
and SpeechBrain recipes), pinned by commit. Audio is the Mix-Headset channel,
downloaded by the caller via that repo's ``download_ami.sh`` (large, not
redistributed here).

Layout after :meth:`prepare` (under ``<root>/ami/``)::

    ami/
      rttms/<split>/*.rttm            gold (cloned from the pinned setup repo)
      audio/<file_id>.Mix-Headset.wav caller-downloaded (download_ami.sh)

:meth:`prepare` clones the pinned setup repo for the RTTMs and prints the audio
download step; :meth:`iter_files` pairs each gold RTTM with its Mix-Headset wav
and emits ``audio_path=None`` when the wav is absent (loud skip, not silent pass).
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

from .base import DiarFile

DATASET_ID = "ami"
SETUP_REPO = "https://github.com/pyannote/AMI-diarization-setup.git"
AUDIO_HINT = (
    "run the setup repo's ./pyannote/download_ami.sh (Mix-Headset), then place "
    "<file_id>.Mix-Headset.wav files under <root>/ami/audio/"
)


class AMILoader:
    """Yields (Mix-Headset audio, prepared gold RTTM) pairs for AMI."""

    name = DATASET_ID
    dataset_id = DATASET_ID

    def __init__(self, split: str = "test") -> None:
        self._split = split

    def _base(self, root: Path) -> Path:
        return root / "ami"

    def prepare(self, root: Path, revision: str | None = None) -> None:
        base = self._base(root)
        rttms_root = base / "rttms"
        if not rttms_root.is_dir():
            base.mkdir(parents=True, exist_ok=True)
            clone = base / "_setup"
            if not clone.is_dir():
                subprocess.run(
                    ["git", "clone", "--depth", "1", SETUP_REPO, str(clone)],
                    check=True,
                )
                if revision:
                    subprocess.run(
                        ["git", "-C", str(clone), "fetch", "--depth", "1",
                         "origin", revision],
                        check=True,
                    )
                    subprocess.run(
                        ["git", "-C", str(clone), "checkout", revision], check=True
                    )
            # Canonical gold lives under only_words/rttms/{train,dev,test}.
            src_root = clone / "only_words" / "rttms"
            if not src_root.is_dir():
                raise FileNotFoundError(
                    f"expected prepared RTTMs at {src_root}; AMI-diarization-setup "
                    f"layout changed — check the pinned revision"
                )
            for split in ("train", "dev", "test"):
                src = src_root / split
                if src.is_dir():
                    dst = rttms_root / split
                    dst.mkdir(parents=True, exist_ok=True)
                    for rttm in src.glob("*.rttm"):
                        (dst / rttm.name).write_bytes(rttm.read_bytes())
        audio_dir = base / "audio"
        if not audio_dir.is_dir() or not any(audio_dir.glob("*.wav")):
            print(
                f"[ami] gold RTTMs ready under {rttms_root}. Audio is NOT "
                f"auto-downloaded — {AUDIO_HINT}"
            )

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
