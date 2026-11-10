"""The step rule both capacity harnesses share: percentiles and when to stop.

A capacity curve doubles the load N step by step — live sessions in
:mod:`raven_asr.stream`, requests in flight in :mod:`raven_asr.load` — and each
step decides for itself whether it stayed within the harness's rule
(``within_threshold``). The curve keeps doubling while every step passes; the
first step that breaks the rule is recorded and so is ONE step beyond it, then
the run stops. The curve keeps every step, so any other threshold can be read
off it later.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default), None for no values."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def pcts(values: Sequence[float]) -> dict[str, float | None]:
    """``{"p50", "p95", "p99"}`` of ``values``, rounded to 0.1 ms."""
    return {f"p{q}": (None if (v := percentile(values, q)) is None else round(v, 4))
            for q in (50, 95, 99)}


def next_action(steps: Sequence[dict]) -> str:
    """'continue' | 'one_more' | 'stop' after the steps so far."""
    failed = [i for i, s in enumerate(steps) if not s["within_threshold"]]
    if not failed:
        return "continue"
    return "one_more" if failed[0] == len(steps) - 1 else "stop"


def max_n_within(steps: Sequence[dict]) -> int | None:
    """The largest N before the first step that broke the rule."""
    best = None
    for s in steps:
        if not s["within_threshold"]:
            break
        best = s["n"]
    return best
