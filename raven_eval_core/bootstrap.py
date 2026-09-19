"""Percentile bootstrap for a corpus ratio — the one interval every metric shares.

Both published error rates are the same shape of estimator: a sum of errors over
a sum of reference units, aggregated across the whole corpus (DER: error seconds
over scored speech seconds, per file; WER: word edits over reference words, per
utterance). So one resampler serves both. Each sampling unit contributes a
``(numerator, denominator)`` pair, and every resample re-aggregates
``Σnum / Σden`` — the estimator that is published, never a mean of per-unit rates.

``bootstrap_ratio_ci``
    How precise is one number? Resamples units with replacement.

``paired_bootstrap_ratio_delta``
    Is a gap between two models real? One resample draws a set of units and
    scores BOTH models on it, so per-unit difficulty cancels. An interval
    excluding zero licenses "A is ahead of B"; one spanning zero means the
    ranking is a coin flip on this corpus.

``pair_by_id``
    Aligns two models' units by id and refuses to pair around a gap: a unit only
    one model has is a coverage problem to fix, not a row to drop quietly — a
    comparison on a silently shrunk set looks exactly as confident as a full one.

Seeded and pure-stdlib, so a published interval is reproducible to the digit on
any machine. The draw order (``randrange`` per unit per resample) is part of that
contract: changing it moves every published interval.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

#: One sampling unit: (errors, reference size), in whatever unit the metric uses.
Unit = tuple[float, float]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap interval, all in percent."""

    point: float
    lo: float
    hi: float
    n: int
    resamples: int
    seed: int

    @property
    def half_width(self) -> float:
        """Half the interval width — the ± a headline number should carry."""
        return (self.hi - self.lo) / 2.0

    def format(self) -> str:
        return (f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] "
                f"(±{self.half_width:.3f}, n={self.n})")


def ratio_pct(units: Sequence[Unit]) -> float:
    """``Σnum / Σden`` in percent; 0.0 for an empty or zero-size denominator."""
    num = sum(u[0] for u in units)
    den = sum(u[1] for u in units)
    if den <= 0.0:
        return 0.0
    return num / den * 100.0


def _percentiles(draws: list[float], confidence: float) -> tuple[float, float]:
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    resamples = len(draws)
    return (draws[int(tail * (resamples - 1))],
            draws[int((1.0 - tail) * (resamples - 1))])


def bootstrap_ratio_ci(
    units: Sequence[Unit],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> Interval:
    """Percentile bootstrap over ``units`` for the corpus ratio ``Σnum/Σden``."""
    point = ratio_pct(units)
    n = len(units)
    if n < 2:
        return Interval(point=point, lo=point, hi=point, n=n, resamples=0, seed=seed)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        draws.append(ratio_pct([units[rng.randrange(n)] for _ in range(n)]))
    lo, hi = _percentiles(draws, confidence)
    return Interval(point=point, lo=lo, hi=hi, n=n, resamples=resamples, seed=seed)


def paired_bootstrap_ratio_delta(
    pairs: Sequence[tuple[Unit, Unit]],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> Interval:
    """Interval on ``ratio(a) - ratio(b)`` over units both models were scored on."""
    n = len(pairs)
    point = ratio_pct([a for a, _ in pairs]) - ratio_pct([b for _, b in pairs])
    if n < 2:
        return Interval(point=point, lo=point, hi=point, n=n, resamples=0, seed=seed)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(ratio_pct([pairs[i][0] for i in idx])
                     - ratio_pct([pairs[i][1] for i in idx]))
    lo, hi = _percentiles(draws, confidence)
    return Interval(point=point, lo=lo, hi=hi, n=n, resamples=resamples, seed=seed)


class UnpairedUnitsError(ValueError):
    """Two models were not scored on the same units."""


def pair_by_id(a: Mapping[str, T], b: Mapping[str, T]) -> list[tuple[T, T]]:
    """Align two models' units by id, in ``a``'s order; raise on any unmatched id."""
    only_a = [k for k in a if k not in b]
    only_b = [k for k in b if k not in a]
    if only_a or only_b:
        raise UnpairedUnitsError(
            f"models were not scored on the same units — only in A: "
            f"{only_a[:5]}{'…' if len(only_a) > 5 else ''} ({len(only_a)}), "
            f"only in B: {only_b[:5]}{'…' if len(only_b) > 5 else ''} "
            f"({len(only_b)}). A paired comparison over the intersection would "
            "quietly answer a different question; re-run the missing units."
        )
    return [(a[k], b[k]) for k in a]
