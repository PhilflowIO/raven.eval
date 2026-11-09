"""The seeded utterance pool the live sessions play.

Every utterance of every source corpus that is selected is decoded once, turned
into 16 kHz mono PCM16 (the realtime protocol's format) and put into one list in
a seeded order. Session ``i`` of a step with ``n`` sessions starts at position
``i * len(pool) // n`` and plays forward, wrapping around: sessions play
different utterances, and a step plays the same utterances whenever it is
repeated with the same seed. The pool is identified by the sha256 of its id
order, recorded in every curve.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from ..config import WER_DATASETS
from ..datasets import load_loader_class
from ..datasets.base import Sample

SAMPLE_RATE = 16000

#: The default pool: all of FLEURS German test (read sentences, ~2.9 h) and a
#: seeded 2,000 of the avemio German mixed test (Common Voice + MLS, ~4 h), so
#: 128 sessions x 180 s fit without one session repeating another's audio.
DEFAULT_SOURCES: tuple[tuple[str, int | None], ...] = (
    ("fleurs", None),
    ("avemio-german-mixed-test", 2000),
)
DEFAULT_SEED = 20261109


@dataclass(frozen=True)
class Utterance:
    sample_id: str
    subset: str
    reference: str
    pcm16: bytes  # 16 kHz mono little-endian int16

    @property
    def duration_s(self) -> float:
        return len(self.pcm16) / 2 / SAMPLE_RATE


def to_pcm16(audio: np.ndarray, sample_rate: int) -> bytes:
    """Float audio in [-1, 1] at any rate -> 16 kHz mono PCM16 bytes."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        import soxr

        audio = soxr.resample(audio, sample_rate, SAMPLE_RATE, quality="VHQ")
    return (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def select(samples: Iterable[Sample], take: int | None, rng: random.Random) -> list[Sample]:
    """All samples, or a seeded ``take`` of them (in corpus order before the shuffle)."""
    rows = list(samples)
    if take is not None and take < len(rows):
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), take))]
    return rows


def build_pool(
    sources: tuple[tuple[str, int | None], ...] = DEFAULT_SOURCES,
    seed: int = DEFAULT_SEED,
    loader_for=None,
) -> tuple[list[Utterance], dict]:
    """``(pool, spec)``. ``loader_for(dataset_id)`` -> iterable of Samples (tests inject one)."""
    rng = random.Random(seed)
    pool: list[Utterance] = []
    counts: dict[str, int] = {}
    revisions: dict[str, str | None] = {}
    for dataset_id, take in sources:
        if loader_for is None:
            spec = WER_DATASETS[dataset_id]
            loader = load_loader_class(dataset_id)(streaming=False, revision=spec.revision)
            samples: Iterable[Sample] = loader.iter_samples(spec.subsets[0])
            revisions[dataset_id] = spec.revision
        else:
            samples = loader_for(dataset_id)
            revisions[dataset_id] = None
        rows = select(samples, take, rng)
        for s in rows:
            pool.append(Utterance(s.sample_id, s.subset, s.reference,
                                  to_pcm16(s.audio, s.sample_rate)))
        counts[dataset_id] = len(rows)
    rng.shuffle(pool)
    ids = "\n".join(u.sample_id for u in pool).encode()
    spec = {
        "seed": seed,
        "sources": [{"dataset": d, "take": t, "selected": counts[d], "revision": revisions[d]}
                    for d, t in sources],
        "n_utterances": len(pool),
        "audio_s": round(sum(u.duration_s for u in pool), 1),
        "ids_sha256": hashlib.sha256(ids).hexdigest(),
    }
    return pool, spec


def session_playlist(pool: list[Utterance], session: int, n_sessions: int) -> Iterator[Utterance]:
    """Session ``session`` of ``n_sessions``: its start offset, then forward, wrapping."""
    if not pool:
        raise ValueError("empty pool")
    start = session * len(pool) // n_sessions
    i = start
    while True:
        yield pool[i % len(pool)]
        i += 1
