"""Production's speech compaction and dense-track windows, reimplemented.

Raven (flow.raven, master 2026-11-09) transcribes a meeting after it ends, one
LiveKit egress track per speaker. Before upload a track goes one of two ways:

* **VAD blocks** (``api/app/providers/_vad.py``: ``parse_silence_log``,
  ``build_speech_islands``, ``plan_blocks``, ``_cut_blocks``,
  ``_build_plan_sync``, ``map_block_time``; defaults in ``api/app/config.py``
  ``stt_vad_*``). ffmpeg ``silencedetect`` at −40 dB with silences of at least
  0.6 s; the complement is padded by 0.35 s on both sides, islands closer than
  0.5 s merge, islands under 0.25 s are dropped; consecutive islands are
  spliced into blocks of at most 30 s of *speech* (an island longer than that
  is its own block, never cut). Each block keeps its piecewise map block time →
  track time, derived from the sample counts actually copied.
* **Whole file** when the track is dense (speech ratio > 0.6) or shorter than
  60 s (``plan_segmentation`` returns None). The GPU box's server then cuts it
  near every 600 s at the silence whose midpoint is closest to the boundary,
  within ±20 % (flow.raven-deployment ``server/philsai-stt/parakeet_service/
  chunker.py``, ``_pick_split_points``; ``MAX_CHUNK_DURATION_SEC=600``).

Both are reproduced here as pure functions over the silence list, so a load test
can send exactly the request shapes production sends. The functions mirror the
production ones line for line; where this module differs it says so. Parliament
recordings are dense, so production would send them whole — the harness
prepares BOTH shapes from every recording and records which one production
would have picked (:func:`production_path`).
"""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

#: flow.raven api/app/config.py, stt_vad_* (master 2026-11-09).
NOISE_FLOOR_DB = -40.0
MIN_SILENCE_S = 0.6
PAD_S = 0.35
MIN_ISLAND_S = 0.25
MERGE_GAP_S = 0.5
MAX_BLOCK_S = 30.0
MAX_SPEECH_RATIO = 0.6
MIN_SOURCE_S = 60.0
#: flow.raven-deployment server/philsai-stt docker-compose: MAX_CHUNK_DURATION_SEC.
WINDOW_S = 600.0
WINDOW_LEEWAY = 0.2

SAMPLE_RATE = 16000

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[0-9.]+)")

#: (block_start, block_end, source_start), seconds — production's ``Piece``.
Piece = tuple[float, float, float]


def parse_silence_log(stderr: str, duration: float) -> list[tuple[float, float]]:
    """ffmpeg ``silencedetect`` stderr → ``[(start, end), ...]`` (_vad.py, same name).

    A trailing ``silence_start`` without its ``silence_end`` is closed at
    ``duration``.
    """
    silences: list[tuple[float, float]] = []
    pending: float | None = None
    for line in stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending = max(0.0, float(start_match.group(1)))
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending is not None:
            end = min(duration, float(end_match.group(1)))
            if end > pending:
                silences.append((pending, end))
            pending = None
    if pending is not None and duration > pending:
        silences.append((pending, duration))
    return silences


def build_speech_islands(
    silences: Sequence[tuple[float, float]],
    duration: float,
    *,
    pad: float = PAD_S,
    min_island: float = MIN_ISLAND_S,
    merge_gap: float = MERGE_GAP_S,
) -> list[tuple[float, float]]:
    """Complement of the silences, padded, then merged (_vad.py, same name).

    The pad is applied before the merge, as in production: a word onset sits a
    few tens of ms inside what silencedetect calls silence.
    """
    if duration <= 0:
        return []
    speech: list[tuple[float, float]] = []
    cursor = 0.0
    for sil_start, sil_end in sorted(silences):
        if sil_start > cursor:
            speech.append((cursor, min(sil_start, duration)))
        cursor = max(cursor, sil_end)
    if cursor < duration:
        speech.append((cursor, duration))
    padded = [(max(0.0, s - pad), min(duration, e + pad)) for s, e in speech if e > s]
    merged: list[tuple[float, float]] = []
    for start, end in padded:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(s, e) for s, e in merged if e - s >= min_island]


def plan_blocks(
    islands: Sequence[tuple[float, float]], *, max_block_seconds: float = MAX_BLOCK_S
) -> list[list[tuple[float, float]]]:
    """Group islands into blocks of at most ``max_block_seconds`` of speech.

    Production's ``plan_blocks`` with its Parakeet arguments (no join-gap cap, no
    solo minimum — those are the bounded planner's, #9508). An island longer
    than the cap becomes its own block rather than being split.
    """
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    current_len = 0.0
    for start, end in islands:
        island_len = end - start
        if current and current_len + island_len > max_block_seconds:
            blocks.append(current)
            current, current_len = [], 0.0
        current.append((start, end))
        current_len += island_len
    if current:
        blocks.append(current)
    return blocks


def block_pieces(
    islands: Sequence[tuple[float, float]], total_frames: int, rate: int = SAMPLE_RATE
) -> tuple[tuple[tuple[int, int], ...], tuple[Piece, ...]]:
    """The sample ranges a block copies and its block→source map.

    Returns ``(frame_ranges, pieces)``: ``frame_ranges`` are the ``[first, last)``
    source samples spliced in order, ``pieces`` the ``(block_start, block_end,
    source_start)`` map derived from those sample counts — production's
    ``_cut_blocks`` rounding (floor the start, ceil the end, clamp to the file),
    so the map cannot drift from the bytes the model hears.
    """
    ranges: list[tuple[int, int]] = []
    pieces: list[Piece] = []
    written = 0
    for island_start, island_end in islands:
        first = max(0, min(total_frames, math.floor(island_start * rate)))
        last = max(first, min(total_frames, math.ceil(island_end * rate)))
        if last <= first:
            continue
        ranges.append((first, last))
        block_start = written / rate
        written += last - first
        pieces.append((block_start, written / rate, first / rate))
    return tuple(ranges), tuple(pieces)


def map_block_time(pieces: Sequence[Piece], t: float, *, is_end: bool = False) -> float:
    """A block-timeline instant on the source timeline (_vad.py, same name).

    A value exactly on a splice resolves to the island it belongs to: an end to
    the island it finishes, a start to the island it opens.
    """
    if not pieces:
        return max(0.0, t)
    for index, (block_start, block_end, source_start) in enumerate(pieces):
        if t < block_start:
            return source_start
        last = index == len(pieces) - 1
        if t < block_end or (t <= block_end and (is_end or last)):
            return source_start + (t - block_start)
    tail_start, tail_end, tail_source = pieces[-1]
    return tail_source + (tail_end - tail_start)


def remap_word(pieces: Sequence[Piece], start: float, end: float) -> tuple[float, float]:
    """One word's block times on the source axis, kept inside ONE island.

    Production's ``remap_segments`` for a word (#7836): when the plain map of
    start and end stays within the word's own length, the shift is exact;
    otherwise the word straddles a splice and is clipped to the island holding
    most of it (``_dominant_piece`` + ``_clip_to_piece``).
    """
    if not pieces:
        return max(0.0, start), max(0.0, end)
    block_end = pieces[-1][1]
    start = min(max(start, 0.0), block_end)
    end = min(max(end, start), block_end)
    s = map_block_time(pieces, start)
    e = map_block_time(pieces, end, is_end=True)
    if e - s <= end - start + 1e-6:
        return s, max(s, e)

    def overlap(p: Piece) -> float:
        return min(end, p[1]) - max(start, p[0])

    best = max(pieces, key=overlap)
    if overlap(best) <= 0:
        best = next((p for p in pieces if start < p[1]), pieces[-1])
    b0, b1, src = best
    cs = min(max(start, b0), b1)
    ce = min(max(end, cs), b1)
    return src + (cs - b0), src + (ce - b0)


def pick_split_points(
    duration: float,
    silences: Sequence[tuple[float, float]],
    *,
    window_s: float = WINDOW_S,
    leeway: float = WINDOW_LEEWAY,
) -> list[float]:
    """Where the GPU box's server cuts a whole-file upload (chunker.py, ``_pick_split_points``).

    For each target boundary the silence midpoint closest to it within ±20 % of
    the window; none in range → a hard cut at the target. The next target is
    one window after the cut.
    """
    points: list[float] = []
    target = float(window_s)
    span = window_s * leeway
    while target < duration - span:
        lo, hi = target - span, target + span
        mids = [(s + e) / 2 for s, e in silences if lo <= (s + e) / 2 <= hi]
        cut = min(mids, key=lambda m: abs(m - target)) if mids else target
        points.append(cut)
        target = cut + window_s
    return points


def windows(
    duration: float, silences: Sequence[tuple[float, float]], **kw
) -> list[tuple[float, float]]:
    """The ``[(start, end), ...]`` windows of a whole-file upload."""
    cuts = [0.0, *pick_split_points(duration, silences, **kw), duration]
    return [(a, b) for a, b in zip(cuts, cuts[1:], strict=False) if b > a]


def production_path(duration: float, islands: Sequence[tuple[float, float]]) -> str:
    """Which path flow.raven's ``plan_segmentation`` would take for this track.

    ``"vad"`` or ``"whole-file"`` with the reason, from the same guards: no
    islands, speech ratio above 0.6 (dense), or a source under 60 s.
    """
    if not islands:
        return "whole-file (no speech islands)"
    ratio = sum(e - s for s, e in islands) / duration if duration else 0.0
    if ratio > MAX_SPEECH_RATIO:
        return f"whole-file (dense track, speech ratio {ratio:.2f} > {MAX_SPEECH_RATIO})"
    if duration < MIN_SOURCE_S:
        return f"whole-file (short source, {duration:.0f} s < {MIN_SOURCE_S:.0f} s)"
    return "vad"


def decode_and_detect(src: str, wav_out: str) -> str:
    """Decode ``src`` to mono 16 kHz PCM16 WAV and run silencedetect in one pass.

    Production's ``_decode_and_detect``: the same filter string and output
    shape; returns the stderr carrying the silence log.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info",
        "-i", src, "-map", "0:a:0",
        "-af", f"silencedetect=noise={NOISE_FLOOR_DB}dB:d={MIN_SILENCE_S}",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "wav", "-y", wav_out,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {proc.stderr[-400:]}")
    return proc.stderr


@dataclass(frozen=True)
class Block:
    """One request of a workload: spliced source ranges and their time map."""

    index: int
    frame_ranges: tuple[tuple[int, int], ...]
    pieces: tuple[Piece, ...]

    @property
    def audio_s(self) -> float:
        return sum(b - a for a, b in self.frame_ranges) / SAMPLE_RATE

    @property
    def source_start(self) -> float:
        return self.pieces[0][2] if self.pieces else 0.0

    @property
    def source_end(self) -> float:
        if not self.pieces:
            return 0.0
        b0, b1, src = self.pieces[-1]
        return src + (b1 - b0)

    def as_json(self) -> dict:
        return {"index": self.index, "frame_ranges": [list(r) for r in self.frame_ranges],
                "pieces": [list(p) for p in self.pieces]}

    @classmethod
    def from_json(cls, d: dict) -> Block:
        return cls(d["index"], tuple(tuple(r) for r in d["frame_ranges"]),  # type: ignore[arg-type]
                   tuple(tuple(p) for p in d["pieces"]))  # type: ignore[arg-type]


def vad_blocks(islands: Sequence[tuple[float, float]], total_frames: int) -> list[Block]:
    """Production's VAD blocks of one track (no density guard: see the module doc)."""
    out: list[Block] = []
    for planned in plan_blocks(islands):
        ranges, pieces = block_pieces(planned, total_frames)
        if pieces:
            out.append(Block(len(out), ranges, pieces))
    return out


def window_blocks(duration: float, silences: Sequence[tuple[float, float]],
                  total_frames: int) -> list[Block]:
    """The whole-file path's ~600 s windows as blocks of one piece each."""
    out: list[Block] = []
    for start, end in windows(duration, silences):
        ranges, pieces = block_pieces([(start, end)], total_frames)
        if pieces:
            out.append(Block(len(out), ranges, pieces))
    return out
