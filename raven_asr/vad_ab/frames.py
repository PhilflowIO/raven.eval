"""The 10 ms scoring grid: labels, hysteresis, per-track counts, ROC-AUC, equal-FA picks.

Everything here is pure numpy so it is tested in CI without any engine.

Scoring is done on the islands production would send, not on raw frame
decisions: a frame counts as "sent" when it lies inside an island (after pad,
merge and minimum island). That is the product quantity — what Parakeet hears
and what the GPU pays for. The raw frame scores are scored separately as a
ROC-AUC, which is independent of the planner and of any threshold.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

#: Grid step for labels, decisions and scores, seconds.
HOP = 0.01

Interval = tuple[float, float]


def n_frames(duration: float) -> int:
    return max(0, int(math.ceil(duration / HOP - 1e-9)))


def to_grid(intervals: Iterable[Sequence[float]], n: int) -> np.ndarray:
    """Boolean mask of ``n`` frames, True where a frame's centre lies in any interval."""
    mask = np.zeros(n, dtype=bool)
    for start, end in intervals:
        first = max(0, int(math.ceil(start / HOP - 0.5)))
        last = min(n, int(math.ceil(end / HOP - 0.5)))
        if last > first:
            mask[first:last] = True
    return mask


def merge_intervals(intervals: Iterable[Sequence[float]], gap: float = 0.0) -> list[Interval]:
    """Sorted union; intervals closer than ``gap`` seconds are joined."""
    out: list[list[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in intervals):
        if end <= start:
            continue
        if out and start - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(s, e) for s, e in out]


def hold(values: np.ndarray, hop: float, n: int) -> np.ndarray:
    """Resample per-frame ``values`` at ``hop`` seconds onto ``n`` grid frames (sample and hold).

    Grid frame ``i`` takes the value of the source frame containing its centre;
    past the last source frame the last value holds.
    """
    values = np.asarray(values)
    if values.size == 0:
        return np.zeros(n, dtype=np.float32)
    centres = (np.arange(n) + 0.5) * HOP
    idx = np.minimum(values.size - 1, np.floor(centres / hop).astype(np.int64))
    return values[idx]


def hysteresis(scores: np.ndarray, on: float, off: float) -> np.ndarray:
    """Speech starts where ``score >= on`` and lasts until ``score < off`` (Silero's rule).

    Vectorised: every frame that is decisive (above ``on`` or below ``off``) sets
    the state, every frame in between keeps the last decisive state; before the
    first decisive frame the state is non-speech.
    """
    s = np.asarray(scores)
    if off > on:
        raise ValueError("off threshold above on threshold")
    events = np.where(s >= on, 1, np.where(s < off, 0, -1)).astype(np.int8)
    idx = np.where(events >= 0, np.arange(events.size), -1)
    idx = np.maximum.accumulate(idx) if idx.size else idx
    out = np.zeros(events.size, dtype=bool)
    known = idx >= 0
    out[known] = events[idx[known]] == 1
    return out


def track_counts(sent: np.ndarray, target: np.ndarray, crosstalk: np.ndarray | None = None) -> dict:
    """Seconds of speech, missed speech, non-speech, false alarm and crosstalk sent.

    ``sent`` is the grid mask of what the planner's islands cover; ``target`` the
    reference speech of the track's owner; ``crosstalk`` the reference speech of
    another person audible on this track. Crosstalk frames that are also target
    frames (a real overlap) count as target speech, not as crosstalk.
    """
    n = min(sent.size, target.size)
    sent, target = sent[:n], target[:n]
    miss = target & ~sent
    fa = ~target & sent
    out = {
        "duration_s": n * HOP,
        "speech_s": float(target.sum()) * HOP,
        "missed_s": float(miss.sum()) * HOP,
        "nonspeech_s": float((~target).sum()) * HOP,
        "false_alarm_s": float(fa.sum()) * HOP,
        "sent_s": float(sent.sum()) * HOP,
    }
    if crosstalk is not None:
        xt = crosstalk[:n] & ~target
        out["crosstalk_s"] = float(xt.sum()) * HOP
        out["crosstalk_sent_s"] = float((xt & sent).sum()) * HOP
    return out


def summarise(counts: Iterable[dict]) -> dict:
    """Corpus totals and rates from per-track counts (time-weighted, not track-averaged)."""
    keys = ("duration_s", "speech_s", "missed_s", "nonspeech_s", "false_alarm_s", "sent_s",
            "crosstalk_s", "crosstalk_sent_s")
    tot = {k: 0.0 for k in keys}
    tracks = 0
    for c in counts:
        tracks += 1
        for k in keys:
            tot[k] += c.get(k, 0.0)
    hours = tot["duration_s"] / 3600 if tot["duration_s"] else float("nan")
    return {
        "tracks": tracks,
        "audio_h": round(hours, 4),
        "missed_pct": _pct(tot["missed_s"], tot["speech_s"]),
        "false_alarm_rate_pct": _pct(tot["false_alarm_s"], tot["nonspeech_s"]),
        "false_alarm_s_per_h": round(tot["false_alarm_s"] / hours, 1) if not math.isnan(hours) else None,
        "sent_share_pct": _pct(tot["sent_s"], tot["duration_s"]),
        "crosstalk_sent_pct": _pct(tot["crosstalk_sent_s"], tot["crosstalk_s"]),
        "totals_s": {k: round(v, 2) for k, v in tot.items()},
    }


def _pct(num: float, den: float) -> float | None:
    return round(100.0 * num / den, 3) if den > 0 else None


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the ROC curve of raw frame scores (Mann-Whitney U, ties averaged)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = int(labels.sum())
    neg = labels.size - pos
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    # Average rank per tie group (ranks 1..N).
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sorted_scores.size]))
    avg = (starts + ends + 1) / 2.0
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.repeat(avg, ends - starts)
    u = ranks[labels].sum() - pos * (pos + 1) / 2.0
    return float(u / (pos * neg))


def best_at_false_alarm(curve: Sequence[dict], max_fa_rate_pct: float) -> dict | None:
    """The swept operating point with the lowest miss whose FA rate is at most ``max_fa_rate_pct``.

    ``curve`` holds one :func:`summarise` result per threshold, each with its
    ``threshold``. Only measured points are returned — no interpolation between
    thresholds, so the answer is a setting that exists.
    """
    ok = [p for p in curve if p["false_alarm_rate_pct"] is not None
          and p["missed_pct"] is not None and p["false_alarm_rate_pct"] <= max_fa_rate_pct + 1e-9]
    if not ok:
        return None
    return min(ok, key=lambda p: (p["missed_pct"], p["false_alarm_rate_pct"]))


def pipeline_auc(curve: Sequence[dict]) -> float | None:
    """Area under (FA rate, 1 − miss) over the swept planner outputs, ends pinned at (0,0) and (1,1).

    The upper envelope is used: a point beaten by another with lower FA and
    lower miss does not pull the area down.
    """
    pts = [(p["false_alarm_rate_pct"] / 100.0, 1.0 - p["missed_pct"] / 100.0) for p in curve
           if p["false_alarm_rate_pct"] is not None and p["missed_pct"] is not None]
    if not pts:
        return None
    pts = sorted([(0.0, 0.0), *pts, (1.0, 1.0)])
    envelope: list[tuple[float, float]] = []
    best = -1.0
    for fa, hit in pts:
        best = max(best, hit)
        envelope.append((fa, best))
    xs = np.array([p[0] for p in envelope])
    ys = np.array([p[1] for p in envelope])
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0))


def miss_at_frame_fpr(scores: np.ndarray, labels: np.ndarray, fpr_pct: float) -> float | None:
    """Missed speech frames (%) of the raw detector when its frame false-alarm rate is ``fpr_pct``.

    The detector's own operating curve, before the planner pads and merges: the
    threshold is the ``1 − fpr`` quantile of the scores on non-speech frames,
    and a speech frame counts as found when its score is above it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or not labels.any():
        return None
    thr = np.quantile(scores[~labels], 1.0 - fpr_pct / 100.0, method="higher")
    return round(100.0 * float(np.mean(scores[labels] <= thr)), 3)
