"""CALLHOME-de loader — the German anchor. Gold from speaker segments → RTTM.

CALLHOME German (``talkbank/callhome``, config ``deu``) is 2-speaker German
telephone speech with gold speaker turns. The HF dataset (in the
diarizers-community processing shape) carries, per example, aligned lists of
segment starts/ends and speaker labels — the exact inputs an RTTM needs. This
loader's job is the *conversion*: ``(timestamps_start, timestamps_end, speakers)``
→ standard NIST RTTM lines. That converter (:func:`segments_from_row`) is a pure,
network-free function so it can be unit-tested against a synthetic row today, even
though the download + full run happens later on the user's box.

Layout after :meth:`prepare` (under ``<root>/callhome-de/``)::

    callhome-de/
      audio/<file_id>.wav    materialised from the HF audio column
      gold/<file_id>.rttm    written from segments_from_row(example)

The audio is materialised locally from the HF dataset (which the caller downloads
under TalkBank's cite-to-use terms); only the derived RTTM text is ever a Tier-1
commit candidate.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from raven_eval_core.der import DiarSegment, to_rttm

from .base import DiarFile

DATASET_ID = "callhome-de"
HF_DATASET = "talkbank/callhome"
HF_CONFIG = "deu"
TARGET_SAMPLE_RATE = 16000

# Field-name variants seen across the talkbank / diarizers-community shapes.
_START_KEYS = ("timestamps_start", "starts", "start", "segment_start")
_END_KEYS = ("timestamps_end", "ends", "end", "segment_end")
_SPEAKER_KEYS = ("speakers", "speaker", "labels", "label")


def _first_present(row: dict, keys: tuple[str, ...]) -> object | None:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def decode_audio(audio: object) -> tuple[object | None, int]:
    """Return ``(mono_waveform, sample_rate)`` from either ``datasets`` shape.

    ``datasets`` < 4 handed back a ``{"array", "sampling_rate"}`` dict; >= 4
    returns a torchcodec ``AudioDecoder`` instead. Supporting both keeps this
    loader from being silently pinned to one ``datasets`` major: the previous
    dict-only check wrote ZERO wavs on datasets 5.x, so every file was skipped
    as "missing audio" and the run aborted having scored nothing.
    """
    if audio is None:
        return None, TARGET_SAMPLE_RATE
    if isinstance(audio, dict):
        arr = audio.get("array")
        if arr is None:
            return None, TARGET_SAMPLE_RATE
        return arr, int(audio.get("sampling_rate") or TARGET_SAMPLE_RATE)
    get_all_samples = getattr(audio, "get_all_samples", None)
    if get_all_samples is None:
        return None, TARGET_SAMPLE_RATE
    samples = get_all_samples()
    data = samples.data  # torch tensor, (channels, frames)
    arr = data.numpy() if hasattr(data, "numpy") else data
    if getattr(arr, "ndim", 1) == 2:
        arr = arr.mean(axis=0)  # downmix to mono
    return arr, int(samples.sample_rate)


def segments_from_row(row: dict) -> list[DiarSegment]:
    """Convert one CALLHOME/diarizers example into (start, end, speaker) segments.

    Accepts the standard diarizers-community shape (parallel
    ``timestamps_start`` / ``timestamps_end`` / ``speakers`` lists) and a few
    common key aliases. Speaker labels are stringified as-is (DER's Hungarian
    mapping makes the label *names* irrelevant — only the partition matters).
    Zero-/negative-length turns are dropped. Raises ``ValueError`` if the three
    lists are absent or mismatched in length (a malformed row must fail loudly,
    never silently score against half a reference).
    """
    starts = _first_present(row, _START_KEYS)
    ends = _first_present(row, _END_KEYS)
    speakers = _first_present(row, _SPEAKER_KEYS)
    if starts is None or ends is None or speakers is None:
        raise ValueError(
            "CALLHOME row missing segment fields; expected one of "
            f"{_START_KEYS} / {_END_KEYS} / {_SPEAKER_KEYS}, got keys {list(row)}"
        )
    starts = list(starts)
    ends = list(ends)
    speakers = list(speakers)
    if not (len(starts) == len(ends) == len(speakers)):
        raise ValueError(
            f"CALLHOME row segment lists mismatched: "
            f"{len(starts)} starts / {len(ends)} ends / {len(speakers)} speakers"
        )
    segments: list[DiarSegment] = []
    for s, e, spk in zip(starts, ends, speakers, strict=True):
        s_f, e_f = float(s), float(e)
        if e_f > s_f:
            segments.append((s_f, e_f, str(spk)))
    return segments


def _file_id(example: dict, idx: int) -> str:
    for key in ("id", "file_id", "recording_id", "name"):
        if example.get(key):
            return str(example[key])
    return f"callhome-deu-{idx:04d}"


class CallhomeDeLoader:
    """Materialises CALLHOME-de audio + derives gold RTTMs from speaker segments."""

    name = DATASET_ID
    dataset_id = DATASET_ID

    def __init__(self, split: str = "data") -> None:
        # talkbank/callhome exposes its examples under a single split; kept
        # overridable for the diarizers-community mirrors that use "test".
        self._split = split

    def _base(self, root: Path) -> Path:
        return root / "callhome-de"

    def prepare(self, root: Path, revision: str | None = None) -> None:
        base = self._base(root)
        audio_dir = base / "audio"
        gold_dir = base / "gold"
        if gold_dir.is_dir() and any(gold_dir.glob("*.rttm")):
            return  # already materialised
        import soundfile as sf  # asr/diar extra
        from datasets import load_dataset  # asr/diar extra

        audio_dir.mkdir(parents=True, exist_ok=True)
        gold_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(
            HF_DATASET, HF_CONFIG, split=self._split, revision=revision
        )
        for idx, example in enumerate(ds):
            fid = _file_id(example, idx)
            segments = segments_from_row(example)
            (gold_dir / f"{fid}.rttm").write_text(
                to_rttm(segments, file_id=fid), encoding="utf-8"
            )
            waveform, sample_rate = decode_audio(example.get("audio"))
            if waveform is not None:
                sf.write(
                    audio_dir / f"{fid}.wav",
                    waveform,
                    sample_rate,
                    subtype="PCM_16",
                )

    def iter_files(self, root: Path, limit: int | None = None) -> Iterator[DiarFile]:
        base = self._base(root)
        gold_dir = base / "gold"
        audio_dir = base / "audio"
        if not gold_dir.is_dir():
            raise FileNotFoundError(
                f"no gold RTTMs at {gold_dir} — run prepare() first "
                f"(load_dataset('{HF_DATASET}', '{HF_CONFIG}'))"
            )
        for emitted, rttm in enumerate(sorted(gold_dir.glob("*.rttm"))):
            fid = rttm.stem
            wav = audio_dir / f"{fid}.wav"
            yield DiarFile(
                file_id=fid,
                audio_path=wav if wav.exists() else None,
                gold_rttm_path=rttm,
                dataset=DATASET_ID,
            )
            if limit is not None and emitted + 1 >= limit:
                return


#: Registry entry point (see raven_diar.registry) — ``DiarDatasetSpec.loader``.
LOADER = CallhomeDeLoader
