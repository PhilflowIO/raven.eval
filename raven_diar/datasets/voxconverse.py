"""VoxConverse loader — gold RTTMs shipped verbatim, the fastest path to a real DER.

VoxConverse (Chung et al., Interspeech 2020) publishes its diarization ground
truth as ready-to-score RTTM files directly in the upstream repo — no conversion,
no per-speaker segment assembly. This makes it the quickest end-to-end DER smoke:
point the diarizer at a ``.wav`` and score its hypothesis against the matching
``.rttm``.

Layout after :meth:`prepare` (under ``<root>/voxconverse/``)::

    voxconverse/
      labels/dev/*.rttm      gold RTTMs  (cloned from the pinned upstream tag)
      labels/test/*.rttm
      audio/<file_id>.wav    caller-downloaded audio (see the printed instructions)

Gold RTTMs are fetched from the pinned upstream release (``revision``); audio is
NOT (the wavs are large and hosted separately) — :meth:`prepare` prints the exact
download step and :meth:`iter_files` emits ``audio_path=None`` for any file whose
wav is missing, so the runner skips it loudly rather than scoring nothing.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

from .base import DiarFile

DATASET_ID = "voxconverse"
UPSTREAM_REPO = "https://github.com/joonson/voxconverse.git"
# Audio zips (dev/test) — hosted on the Oxford VGG mirror; caller downloads them.
AUDIO_HINT = (
    "https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_dev_wav.zip  "
    "(and voxconverse_test_wav.zip); unzip the .wav files into "
    "<root>/voxconverse/audio/"
)


class VoxConverseLoader:
    """Yields (audio, gold-RTTM) pairs from VoxConverse's shipped RTTMs."""

    name = DATASET_ID
    dataset_id = DATASET_ID

    def __init__(self, split: str = "dev") -> None:
        # VoxConverse ships gold for dev + test; dev is the usual tuning split.
        self._split = split

    def _base(self, root: Path) -> Path:
        return root / "voxconverse"

    def prepare(self, root: Path, revision: str | None = None) -> None:
        base = self._base(root)
        labels_root = base / "labels"
        rev = revision or "v0.3"
        if not labels_root.is_dir():
            base.mkdir(parents=True, exist_ok=True)
            clone = base / "_repo"
            if not clone.is_dir():
                # Shallow clone at the pinned tag/commit — gold RTTMs only.
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", rev,
                     UPSTREAM_REPO, str(clone)],
                    check=True,
                )
            # The upstream repo keeps gold under dev/ and test/ (RTTM dirs).
            labels_root.mkdir(parents=True, exist_ok=True)
            for split in ("dev", "test"):
                src = clone / split
                if src.is_dir():
                    dst = labels_root / split
                    dst.mkdir(parents=True, exist_ok=True)
                    for rttm in src.glob("*.rttm"):
                        (dst / rttm.name).write_bytes(rttm.read_bytes())
        audio_dir = base / "audio"
        if not audio_dir.is_dir() or not any(audio_dir.glob("*.wav")):
            print(
                f"[voxconverse] gold RTTMs ready under {labels_root}. "
                f"Audio is NOT auto-downloaded — fetch it yourself:\n  {AUDIO_HINT}"
            )

    def iter_files(self, root: Path, limit: int | None = None) -> Iterator[DiarFile]:
        base = self._base(root)
        labels_dir = base / "labels" / self._split
        audio_dir = base / "audio"
        if not labels_dir.is_dir():
            raise FileNotFoundError(
                f"no gold RTTMs at {labels_dir} — run prepare() first "
                f"(git clone of {UPSTREAM_REPO})"
            )
        for emitted, rttm in enumerate(sorted(labels_dir.glob("*.rttm"))):
            file_id = rttm.stem
            wav = audio_dir / f"{file_id}.wav"
            yield DiarFile(
                file_id=file_id,
                audio_path=wav if wav.exists() else None,
                gold_rttm_path=rttm,
                dataset=DATASET_ID,
            )
            if limit is not None and emitted + 1 >= limit:
                return


#: Registry entry point (see raven_diar.registry) — ``DiarDatasetSpec.loader``.
LOADER = VoxConverseLoader
