"""Long recordings turned into the requests production would send, cached.

A corpus directory holds one recording per id: ``<id>.opus`` (any container
ffmpeg reads), ``<id>.gt.txt`` (the reference transcript), ``<id>.manifest.json``
(licence, duration, slot, transcript kind) and optionally ``<id>.vtt`` (timed
cues). :func:`prepare` decodes a recording once to mono 16 kHz PCM16 WAV with
production's ffmpeg call (raven_asr.load.vad.decode_and_detect), runs
silencedetect in the same pass, and caches under ``<cache>/<id>/``:

  source.wav   the decoded working copy (never leaves the host; not committed)
  plan.json    silences, speech islands, both workloads' blocks (VAD blocks with
               islands over 30 s split at their quietest instant — see
               raven_asr.load.vad), what the unsplit VAD plan would have sent,
               which path production would take, the sha256 of the source file

A request's audio is the spliced PCM of its block, framed as a 16 kHz mono
PCM16 WAV in memory (:meth:`Prepared.wav_bytes`) — the bytes production uploads.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import vad
from .vad import SAMPLE_RATE, Block

PLAN_SCHEMA = "raven.load-plan/1"
KINDS = ("vad", "window")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_data_offset(path: Path) -> tuple[int, int]:
    """``(byte offset, byte length)`` of the PCM in a RIFF/WAVE file's data chunk."""
    with path.open("rb") as fh:
        head = fh.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"{path}: not a RIFF/WAVE file")
        while True:
            hdr = fh.read(8)
            if len(hdr) < 8:
                raise ValueError(f"{path}: no data chunk")
            cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid == b"data":
                start = fh.tell()
                # ffmpeg writing a stream may leave 0xFFFFFFFF; trust the file size.
                end = path.stat().st_size
                return start, min(size, end - start) if size else end - start
            fh.seek(size + (size & 1), 1)


def wav_header(n_frames: int, rate: int = SAMPLE_RATE) -> bytes:
    """A 44-byte header for mono PCM16 at ``rate``."""
    data = n_frames * 2
    return (b"RIFF" + struct.pack("<I", 36 + data) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", data))


@dataclass
class Recording:
    """One long recording of the corpus and its reference."""

    rid: str
    audio: Path
    reference: Path
    manifest: dict
    vtt: Path | None

    @property
    def dialect(self) -> bool:
        """Whether the audio is dialect against a Standard German reference.

        Read from the manifest's ``slot`` (``E-CH-Mundart`` for Basel's Grosser Rat).
        """
        return "mundart" in str(self.manifest.get("slot", "")).lower()


def find_recordings(corpus: Path, ids: list[str]) -> list[Recording]:
    out = []
    for rid in ids:
        audio = next((p for p in sorted(corpus.glob(f"{rid}.*"))
                      if p.suffix in (".opus", ".mp3", ".mp4", ".wav", ".flac", ".m4a")), None)
        if audio is None:
            raise FileNotFoundError(f"{corpus}: no audio for {rid}")
        ref = corpus / f"{rid}.gt.txt"
        man = corpus / f"{rid}.manifest.json"
        vtt = corpus / f"{rid}.vtt"
        out.append(Recording(rid, audio, ref, json.loads(man.read_text()) if man.exists() else {},
                             vtt if vtt.exists() else None))
    return out


@dataclass
class Prepared:
    """A recording's cached working copy and both workloads."""

    recording: Recording
    wav: Path
    duration_s: float
    plan: dict
    _pcm: np.ndarray | None = None

    @property
    def pcm(self) -> np.ndarray:
        if self._pcm is None:
            off, length = wav_data_offset(self.wav)
            self._pcm = np.memmap(self.wav, dtype="<i2", mode="r", offset=off,
                                  shape=(length // 2,))
        return self._pcm

    def blocks(self, kind: str) -> list[Block]:
        return [Block.from_json(b) for b in self.plan["workloads"][kind]]

    def wav_bytes(self, block: Block) -> bytes:
        parts = [np.asarray(self.pcm[a:b]).tobytes() for a, b in block.frame_ranges]
        n = sum(b - a for a, b in block.frame_ranges)
        return wav_header(n) + b"".join(parts)


def prepare(rec: Recording, cache: Path) -> Prepared:
    """Decode, detect silences and plan both workloads once; reuse the cache after."""
    d = cache / rec.rid
    d.mkdir(parents=True, exist_ok=True)
    wav = d / "source.wav"
    plan_path = d / "plan.json"
    src_sha = sha256_file(rec.audio)
    params = {"noise_floor_db": vad.NOISE_FLOOR_DB, "min_silence_s": vad.MIN_SILENCE_S,
              "pad_s": vad.PAD_S, "min_island_s": vad.MIN_ISLAND_S,
              "merge_gap_s": vad.MERGE_GAP_S, "max_block_s": vad.MAX_BLOCK_S,
              "window_s": vad.WINDOW_S, "window_leeway": vad.WINDOW_LEEWAY,
              "split_long_islands_s": vad.MAX_BLOCK_S, "quiet_window_s": vad.QUIET_WINDOW_S}
    if plan_path.exists() and wav.exists():
        plan = json.loads(plan_path.read_text())
        if plan.get("source_sha256") == src_sha and plan.get("params") == params:
            return Prepared(rec, wav, plan["duration_s"], plan)
    stderr = vad.decode_and_detect(str(rec.audio), str(wav))
    _off, length = wav_data_offset(wav)
    total_frames = length // 2
    duration = total_frames / SAMPLE_RATE
    silences = vad.parse_silence_log(stderr, duration)
    islands = vad.build_speech_islands(silences, duration)
    speech = sum(e - s for s, e in islands)
    off, _ = wav_data_offset(wav)
    pcm = np.memmap(wav, dtype="<i2", mode="r", offset=off, shape=(total_frames,))
    vad_split = vad.vad_blocks(islands, total_frames,
                               cut_at=lambda lo, hi: vad.quietest_instant(pcm, lo, hi))
    plan = {
        "schema": PLAN_SCHEMA,
        "recording": rec.rid,
        "source": rec.audio.name,
        "source_sha256": src_sha,
        "params": params,
        "duration_s": duration,
        "speech_s": speech,
        "speech_ratio": speech / duration if duration else 0.0,
        "production_path": vad.production_path(duration, islands),
        "n_silences": len(silences),
        "islands": islands,
        "vad_unsplit": vad.unsplit_stats(islands),
        "workloads": {
            "vad": [b.as_json() for b in vad_split],
            "window": [b.as_json() for b in vad.window_blocks(duration, silences, total_frames)],
        },
    }
    plan_path.write_text(json.dumps(plan) + "\n")
    return Prepared(rec, wav, duration, plan)


@dataclass(frozen=True)
class Item:
    """One request of a run: which recording, which workload, which block."""

    rid: str
    kind: str
    block: Block

    @property
    def key(self) -> str:
        return f"{self.rid}/{self.kind}/{self.block.index}"


def all_items(prepared: list[Prepared], kind: str) -> list[Item]:
    """Every block of ``kind`` of every recording, recording by recording, in time order."""
    return [Item(p.recording.rid, kind, b) for p in prepared for b in p.blocks(kind)]


def sweep_slice(prepared: list[Prepared], kind: str, n_requests: int, seed: int) -> list[Item]:
    """A fixed, seeded mix of ``n_requests`` blocks across the recordings.

    Every step of a sweep sends the same slice, so steps differ by N only.
    """
    items = all_items(prepared, kind)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n_requests]
