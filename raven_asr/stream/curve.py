"""From per-utterance records to the capacity curve and its table.

A step's latency percentiles run over every utterance of every session that
completed (``transcription.done`` arrived). An utterance that errored or timed
out counts as an error; a session that ended on an error is a drop. WER is the
offline lens (raven_eval_core, strict-de corpus WER) over the completed
utterances, and — when the model's offline predictions are given — the offline
WER of exactly the same utterances, so the two differ by the streaming mode only.

The step rule: keep doubling N while p95 final latency stays within the
threshold and nothing errored; the first step that breaks either rule is
recorded and so is ONE step beyond it, then the run stops. The curve keeps every
step, so any other threshold can be read off it later.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from raven_eval_core.wer import corpus_wer_strict_de_pct

from ..capacity import pcts as _pcts
from ..capacity import percentile
from .session import UtteranceRecord

DEFAULT_STEPS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_THRESHOLD_S = 1.0


def load_offline(pred_dir: Path | None) -> dict[str, str]:
    """``sample_id -> prediction`` from every ``predictions_*.jsonl`` in ``pred_dir``."""
    out: dict[str, str] = {}
    if pred_dir is None:
        return out
    for path in sorted(Path(pred_dir).glob("predictions_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("prediction") is not None:
                    out[row["sample_id"]] = row["prediction"]
    return out


def step_summary(
    n: int,
    records: Iterable[UtteranceRecord],
    session_errors: Sequence[str | None],
    probes: dict,
    offline: dict[str, str],
    threshold_s: float,
    spoken: Callable[[str], str] = lambda raw: raw,
) -> dict:
    """One step of the curve. ``spoken`` turns a model's raw output into the
    spoken text exactly as the offline scorer does (raven_asr.output_formats:
    the Nemotron prompt models' locale tags, for one), applied to the streaming
    transcripts and to the offline predictions alike."""
    recs = list(records)
    done = [r for r in recs if r.error is None and r.t_done is not None]
    errored = [r for r in recs if r.error is not None]
    finals = [r.final_s for r in done if r.final_s is not None]
    ttfps = [r.ttfp_s for r in done if r.ttfp_s is not None]
    wer = off = None
    if done:
        refs = [r.reference for r in done]
        try:
            wer = round(corpus_wer_strict_de_pct(refs, [spoken(r.text or "") for r in done]), 3)
        except ValueError:
            wer = None
        same = [r for r in done if r.sample_id in offline]
        if same and len(same) == len(done):
            off = round(corpus_wer_strict_de_pct(
                [r.reference for r in same], [spoken(offline[r.sample_id]) for r in same]), 3)
    drops = sum(1 for e in session_errors if e is not None)
    p95 = percentile(finals, 95)
    passed = bool(done) and not errored and drops == 0 and p95 is not None and p95 <= threshold_s
    return {
        "n": n,
        "utterances": len(recs),
        "completed": len(done),
        "errors": len(errored),
        "drops": drops,
        "error_kinds": sorted({(r.error or "")[:120] for r in errored}),
        "audio_s": round(sum(r.duration_s for r in done), 1),
        "ttfp_s": _pcts(ttfps),
        "final_s": _pcts(finals),
        "send_lag_max_s": round(max((r.send_lag_s for r in recs), default=0.0), 4),
        "wer_strict_pct": wer,
        "offline_wer_strict_pct": off,
        "probes": probes,
        "within_threshold": passed,
    }


def _f(v, fmt: str = "{:.2f}") -> str:
    return "–" if v is None else fmt.format(v)


def render_table(curve: dict) -> str:
    """The short Markdown table of one curve (CPU columns for a CPU curve)."""
    cpu = curve.get("device") == "cpu"
    prec = curve.get("precision") or {}
    # Curves recorded with per-process CPU get the busiest client process (and
    # server) per step: ~100 % there means one event loop, not the GPU, was the cap.
    procs = [k for k in ("client", "server")
             if any(s["probes"].get(f"cpu_{k}_proc_max_pct") is not None for s in curve["steps"])]
    lines = [
        f"### {curve['model']} — live capacity",
        "",
        (f"{curve['latency']}; chunk {curve['chunk_ms']} ms; {curve['hardware_label']}; "
         f"{'CPU' if cpu else 'GPU'}, weights {prec.get('weights', '?')}; "
         f"threshold p95 final ≤ {curve['params']['threshold_s']} s"),
        "",
        ("| N | p50 final s | p95 final s | p99 final s | p95 first partial s | errors | drops "
         + ("| WER stream % | WER offline % | CPU util mean % | server RSS max MiB |"
            if cpu else "| WER stream % | WER offline % | GPU util mean % | VRAM max MiB |")
         + "".join(f" {k} CPU busiest proc % |" for k in procs)),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(procs),
    ]
    for s in curve["steps"]:
        p = s["probes"]
        lines.append(
            f"| {s['n']} | {_f(s['final_s']['p50'])} | {_f(s['final_s']['p95'])} "
            f"| {_f(s['final_s']['p99'])} | {_f(s['ttfp_s']['p95'])} | {s['errors']} "
            f"| {s['drops']} | {_f(s['wer_strict_pct'])} | {_f(s['offline_wer_strict_pct'])} "
            f"| {_f(p.get('cpu_util_mean_pct' if cpu else 'util_mean_pct'), '{:.0f}')} "
            f"| {_f(p.get('server_rss_max_mib' if cpu else 'memory_used_max_mib'), '{:.0f}')} |"
            + "".join(f" {_f(p.get(f'cpu_{k}_proc_max_pct'), '{:.0f}')} |" for k in procs)
        )
    best = curve["max_n_within_threshold"] or "none"
    lines += ["", (f"Max N with p95 final ≤ {curve['params']['threshold_s']} s and no errors: "
                   f"**{best}** ({curve['stopped_because']}).")]
    return "\n".join(lines) + "\n"
