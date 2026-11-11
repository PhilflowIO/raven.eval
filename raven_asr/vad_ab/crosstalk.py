"""Cross-track crosstalk flag: a voice on my track that is someone else's, louder on theirs.

No single-channel detector tells a foreground voice from a background one
(Silero maintainers, silero-vad#687). Raven has what the meeting literature
assumes instead — one simultaneous track per participant (ICSI: energy
normalisation + cross-correlation, Pfau et al. 2001; Wrigley et al. 2005). So a
frame on track A is vetoed as crosstalk when, at the same time,

* the partner's detector says speech on the partner's own track,
* the partner's track is at least :data:`MIN_LEAD_DB` louder than A, and
* A's and the partner's level envelopes move together (Pearson correlation of
  the dB levels over :data:`CORR_WINDOW_S` at least :data:`MIN_CORR`).

A real overlap — A talks while the partner talks — keeps A's own voice on A's
track at A's level, so the partner's lead is small and A is not vetoed. That is
the case the flag must not break; Testset A contains it on purpose.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .frames import HOP

MIN_LEAD_DB = 10.0
MIN_CORR = 0.5
CORR_WINDOW_S = 1.0
LEVEL_SMOOTH_FRAMES = 5
_FLOOR_DB = -100.0


def frame_levels(pcm16: np.ndarray, n: int, rate: int = 16000) -> np.ndarray:
    """dBFS per 10 ms grid frame, energy-smoothed over :data:`LEVEL_SMOOTH_FRAMES` frames."""
    hop = int(HOP * rate)
    x = np.zeros(n * hop, dtype=np.float64)
    m = min(x.size, len(pcm16))
    x[:m] = np.asarray(pcm16[:m], dtype=np.float64) / 32768.0
    energy = np.square(x).reshape(n, hop).mean(axis=1)
    k = LEVEL_SMOOTH_FRAMES
    if k > 1:
        energy = np.convolve(energy, np.ones(k) / k, mode="same")
    return 10 * np.log10(np.maximum(energy, 10 ** (_FLOOR_DB / 10)))


def rolling_corr(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Pearson correlation of ``a`` and ``b`` over a centred window of ``window`` frames."""
    n = a.size
    half = window // 2

    def csum(v: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(v, dtype=np.float64)))

    sa, sb, saa, sbb, sab = csum(a), csum(b), csum(a * a), csum(b * b), csum(a * b)
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    cnt = (hi - lo).astype(np.float64)
    ma = (sa[hi] - sa[lo]) / cnt
    mb = (sb[hi] - sb[lo]) / cnt
    cov = (sab[hi] - sab[lo]) / cnt - ma * mb
    va = (saa[hi] - saa[lo]) / cnt - ma * ma
    vb = (sbb[hi] - sbb[lo]) / cnt - mb * mb
    den = np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 1e-9, cov / den, 0.0)
    return r


def veto(own_levels: np.ndarray, partners: Sequence[tuple[np.ndarray, np.ndarray]], *,
         min_lead_db: float = MIN_LEAD_DB, min_corr: float = MIN_CORR,
         corr_window_s: float = CORR_WINDOW_S) -> np.ndarray:
    """Grid mask of frames on this track that are another participant's voice.

    ``partners`` are ``(levels, speech_mask)`` of every other track of the
    meeting on the same grid.
    """
    n = own_levels.size
    out = np.zeros(n, dtype=bool)
    window = max(3, int(round(corr_window_s / HOP)))
    for levels, speech in partners:
        m = min(n, levels.size, speech.size)
        lead = levels[:m] - own_levels[:m]
        corr = rolling_corr(own_levels[:m], levels[:m], window)
        out[:m] |= speech[:m] & (lead >= min_lead_db) & (corr >= min_corr)
    return out
