"""The offline capacity curve: requests in flight N against one server.

Each step keeps exactly N requests on the wire (closed loop: a worker sends its
next block the moment its last answer arrives) until the step's fixed slice of
blocks is used up. Every step sends the same slice
(raven_asr.load.workload.sweep_slice), so steps differ by N only.

Per step: RTFx (audio seconds answered per wall second, over the whole step and
over its steady phase — until the last block was sent, while all N workers were
busy), latency percentiles, latency over the request's answer budget, errors,
timeouts, aborts, GPU (raven_asr.probes: VRAM max, utilization, vLLM gauges)
and the harness's own CPU.

The rule (raven_asr.capacity): a step passes when every block was answered —
no error, no timeout, no abort — and the p95 of latency / answer budget is at
most 1, i.e. 95 % of blocks came back within the read timeout production would
have given them (``min(300, 60 + audio_s)``). The sweep stops one step after the
first step that fails.
"""

from __future__ import annotations

import asyncio
import resource
import time
from collections.abc import Callable, Sequence

import httpx

from ..capacity import pcts, percentile
from ..probes import Sampler
from .client import Outcome, answer_budget_s, transcribe
from .workload import Item

DEFAULT_STEPS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
#: A step that runs longer than this is cut; its unsent and unanswered blocks
#: count as aborted (the step then fails the rule).
STEP_CAP_S = 3600.0


def slice_size(kind: str, n: int, available: int) -> int:
    """Blocks per step: enough for every worker to send many, capped by the corpus.

    VAD blocks (≤ 30 s of speech): max(256, 16 N). 600 s windows: max(32, 4 N).
    """
    want = max(256, 16 * n) if kind == "vad" else max(32, 4 * n)
    return min(want, available)


def cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


async def run_items(
    url: str,
    model: str,
    items: Sequence[Item],
    wav_of: Callable[[Item], bytes],
    n: int,
    ask: str,
    *,
    retries: int = 0,
    cap_s: float = STEP_CAP_S,
    on_done: Callable[[Item, Outcome], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[Outcome], float, float]:
    """Send ``items`` with ``n`` in flight; ``(outcomes in item order, t0, t_last_sent)``."""
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(len(items)):
        queue.put_nowait(i)
    results: list[Outcome | None] = [None] * len(items)
    t0 = time.monotonic()
    last_sent = [t0]
    limits = httpx.Limits(max_connections=n + 4, max_keepalive_connections=n + 4)

    async with httpx.AsyncClient(limits=limits, transport=transport) as client:

        async def worker() -> None:
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                it = items[i]
                last_sent[0] = time.monotonic()
                wav = wav_of(it)
                out = await transcribe(client, url, model, it.key, wav, it.block.audio_s,
                                       ask, retries=retries)
                results[i] = out
                if on_done is not None:
                    on_done(it, out)

        tasks = [asyncio.create_task(worker()) for _ in range(n)]
        _done, pending = await asyncio.wait(tasks, timeout=cap_s)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    outs = []
    for i, r in enumerate(results):
        if r is None:
            it = items[i]
            r = Outcome(key=it.key, audio_s=it.block.audio_s,
                        budget_s=answer_budget_s(it.block.audio_s), t_start=t0)
        outs.append(r)
    return outs, t0, last_sent[0]


def step_summary(n: int, outs: Sequence[Outcome], t0: float, t_last_sent: float,
                 probes: dict, cpu_s: float) -> dict:
    """One step of the curve from its outcomes (pure; tests feed it by hand)."""
    ok = [o for o in outs if o.status == "ok" and o.t_end is not None]
    ends = [o.t_end for o in outs if o.t_end is not None]
    t_end = max(ends) if ends else t0
    wall = max(t_end - t0, 1e-9)
    audio_ok = sum(o.audio_s for o in ok)
    steady = max(t_last_sent - t0, 0.0)
    audio_steady = sum(o.audio_s for o in ok if o.t_end is not None and o.t_end <= t_last_sent)
    lats = [o.latency_s for o in ok if o.latency_s is not None]
    ratios = [o.latency_s / o.budget_s for o in ok if o.latency_s is not None]
    kinds: dict[str, int] = {}
    for o in outs:
        kinds[o.status] = kinds.get(o.status, 0) + 1
    errors = kinds.get("http_error", 0) + kinds.get("conn_error", 0)
    timeouts = kinds.get("timeout", 0)
    aborts = kinds.get("aborted", 0)
    p95_ratio = percentile(ratios, 95)
    passed = (bool(ok) and len(ok) == len(outs) and p95_ratio is not None and p95_ratio <= 1.0)
    batch = [o.extra["batch_size"] for o in ok if "batch_size" in o.extra]
    return {
        "n": n,
        "requests": len(outs),
        "answered": len(ok),
        "errors": errors,
        "timeouts": timeouts,
        "aborts": aborts,
        "error_kinds": sorted({f"{o.status}: {(o.error or '')[:100]}" for o in outs
                               if o.status != "ok"}),
        "audio_s": round(audio_ok, 1),
        "wall_s": round(wall, 2),
        "rtfx": round(audio_ok / wall, 2),
        "steady_s": round(steady, 2),
        "rtfx_steady": round(audio_steady / steady, 2) if steady > 1.0 else None,
        "latency_s": pcts(lats),
        "latency_over_budget": pcts(ratios),
        "latency_per_audio_s_p50": (None if not ok else round(
            percentile([o.latency_s / o.audio_s for o in ok if o.audio_s > 0], 50) or 0.0, 4)),
        "server_batch_size_mean": round(sum(batch) / len(batch), 2) if batch else None,
        "probes": probes,
        "harness_cpu_pct_of_core": round(100.0 * cpu_s / wall, 1),
        "within_threshold": passed,
    }


async def run_step(url: str, model: str, items: Sequence[Item], wav_of, n: int, ask: str,
                   metrics_url: str | None) -> tuple[dict, list[Outcome]]:
    c0 = cpu_seconds()
    with Sampler(metrics_url) as sampler:
        outs, t0, t_last = await run_items(url, model, items, wav_of, n, ask)
    return step_summary(n, outs, t0, t_last, sampler.summary(), cpu_seconds() - c0), outs


def _f(v, fmt: str = "{:.2f}") -> str:
    return "–" if v is None else fmt.format(v)


def render_table(curve: dict) -> str:
    """The short Markdown table of one sweep."""
    lines = [
        f"### {curve['model']} — offline capacity, {curve['kind']} blocks",
        "",
        (f"{curve['hardware_label']}; timestamps asked: {curve['timestamps']['ask']}; "
         "rule: no error/timeout and p95 latency/answer-budget ≤ 1 "
         "(budget = min(300, 60 + audio s))"),
        "",
        ("| N | requests | RTFx | RTFx steady | p50 s | p95 s | p99 s | p95 lat/budget "
         "| errors | timeouts | aborts | GPU util mean % | VRAM max MiB | harness CPU % |"),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in curve["steps"]:
        p = s["probes"]
        lines.append(
            f"| {s['n']} | {s['requests']} | {_f(s['rtfx'], '{:.0f}')} "
            f"| {_f(s['rtfx_steady'], '{:.0f}')} | {_f(s['latency_s']['p50'])} "
            f"| {_f(s['latency_s']['p95'])} | {_f(s['latency_s']['p99'])} "
            f"| {_f(s['latency_over_budget']['p95'])} | {s['errors']} | {s['timeouts']} "
            f"| {s['aborts']} | {_f(p.get('util_mean_pct'), '{:.0f}')} "
            f"| {_f(p.get('memory_used_max_mib'), '{:.0f}')} "
            f"| {_f(s['harness_cpu_pct_of_core'], '{:.0f}')} |")
    best = curve["max_n_within_threshold"] or "none"
    lines += ["", f"Max sustained N: **{best}** ({curve['stopped_because']})."]
    return "\n".join(lines) + "\n"
