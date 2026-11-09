"""Cut live audio of many sessions into the frames NeMo's streaming pipeline steps.

NeMo's own request streamer (``ContinuousBatchedRequestStreamer``) reads whole
files and so knows where each one ends. A live session does not: audio arrives
in real time and the end is announced afterwards. This module is the live
counterpart, with the streamer's frame rules:

* one frame per stream per pipeline step, ``frame_samples`` long, the first one
  flagged ``is_first`` (the pipeline creates the stream's state on it);
* the last frame flagged ``is_last`` and zero-padded to ``frame_samples`` with
  ``length`` the number of real samples (``pad_last_frame``; the pipeline then
  frees the state);
* when the audio ends exactly on a frame boundary, the frame that held the last
  samples was already stepped without ``is_last``: the stream then closes with
  one padded frame of a single zero sample, 1/16000 s, the least audio that
  lets the pipeline finalize.

When more streams are ready than one step takes, the ones whose oldest unsent
audio has waited longest go first, so an overloaded GPU delays every stream
alike instead of starving some. Pure numpy: no torch, testable anywhere.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FrameSpec:
    stream_id: int
    samples: np.ndarray  # float32, exactly frame_samples long
    is_first: bool
    is_last: bool
    length: int  # real samples in ``samples``


@dataclass
class _Live:
    chunks: list[np.ndarray] = field(default_factory=list)
    buffered: int = 0
    waiting_since: float | None = None
    sent_first: bool = False
    ended: bool = False


class FrameCutter:
    """Thread-safe: websocket handlers append, the stepping thread takes frames."""

    def __init__(self, frame_samples: int, clock=time.monotonic) -> None:
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        self.frame_samples = frame_samples
        self._clock = clock
        self._live: dict[int, _Live] = {}
        self._lock = threading.Lock()
        self.ready_event = threading.Event()

    def open(self, stream_id: int) -> None:
        with self._lock:
            if stream_id in self._live:
                raise ValueError(f"stream {stream_id} is already open")
            self._live[stream_id] = _Live()

    def append(self, stream_id: int, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        with self._lock:
            live = self._live[stream_id]
            if live.ended:
                raise ValueError(f"stream {stream_id} has ended")
            live.chunks.append(np.asarray(samples, dtype=np.float32))
            live.buffered += samples.size
            if live.waiting_since is None:
                live.waiting_since = self._clock()
            if live.buffered >= self.frame_samples:
                self.ready_event.set()

    def end(self, stream_id: int) -> None:
        with self._lock:
            live = self._live.get(stream_id)
            if live is None or live.ended:
                return
            live.ended = True
            if live.waiting_since is None:
                live.waiting_since = self._clock()
            self.ready_event.set()

    def is_open(self, stream_id: int) -> bool:
        with self._lock:
            return stream_id in self._live

    def __len__(self) -> int:
        with self._lock:
            return len(self._live)

    def _take(self, live: _Live, n: int) -> np.ndarray:
        """The first ``n`` buffered samples, removed from the buffer."""
        out = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            head = live.chunks[0]
            k = min(n - filled, head.size)
            out[filled:filled + k] = head[:k]
            filled += k
            if k == head.size:
                live.chunks.pop(0)
            else:
                live.chunks[0] = head[k:]
        live.buffered -= n
        return out

    def take(self, max_frames: int) -> tuple[list[FrameSpec], list[int]]:
        """``(frames, closed_empty)``: at most one frame from each ready stream,
        at most ``max_frames`` in all, and the streams that ended without any
        audio (closed here: the pipeline never saw them, nothing to finalize)."""
        with self._lock:
            ready = [
                (live.waiting_since, sid) for sid, live in self._live.items()
                if live.buffered >= self.frame_samples
                or (live.ended and (live.buffered or live.sent_first))
            ]
            ready.sort()
            frames: list[FrameSpec] = []
            for _, sid in ready[:max_frames]:
                live = self._live[sid]
                n = min(live.buffered, self.frame_samples)
                samples = np.zeros(self.frame_samples, dtype=np.float32)
                if n:
                    samples[:n] = self._take(live, n)
                is_last = live.ended and live.buffered == 0
                length = n if n else 1  # see module docstring: boundary close
                frames.append(FrameSpec(sid, samples, not live.sent_first, is_last, length))
                live.sent_first = True
                if is_last:
                    del self._live[sid]
                else:
                    live.waiting_since = self._clock() if live.buffered else None
            empty = [sid for sid, live in self._live.items()
                     if live.ended and not live.buffered and not live.sent_first]
            for sid in empty:
                del self._live[sid]
            still_ready = any(
                live.buffered >= self.frame_samples or live.ended
                for live in self._live.values()
            )
            if not still_ready:
                self.ready_event.clear()
            return frames, empty
