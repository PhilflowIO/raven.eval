"""Throughput of a WER run: audio seconds transcribed per wall-clock second.

Per-utterance ``latency_s`` (in ``predictions_*.jsonl``) is what one request
took. Under a request pool and a batching server it is not throughput: a clip
that waits behind nine others in the pool, or in the server's batch, reports
the wait as its latency, and the sum of latencies says nothing about how fast
the corpus went through. Throughput is measured on the run as a whole instead.

A *pass* is one invocation's work on one subset. A subset resumed after a crash
or a retry has several, each appended as one line to
``throughput_<subset>.jsonl`` beside the predictions the moment it ends — also
when it ends early, flagged ``interrupted``. The summary's ``throughput`` block
adds the passes up: RTFx is their audio over their wall time.

Only what a pass actually transcribed counts: utterances answered from the
resume log are not requested and add neither audio nor time, and a failed
request adds no audio (a timeout that fails fast would otherwise read as
speed). The clock starts at a pass's first request, so a resumed pass that
first skips through hours of already-transcribed clips is not charged for it.

Throughput is a measurement of one run on one machine, not a function of the
committed predictions: ``make verify`` cannot re-derive it and does not try.
It holds the block to its own arithmetic (:func:`throughput_problem`).

Stdlib only — the verifier and ``promote`` import this without the runner's
inference stack.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

#: Free-text label for the hardware a self-hosted model ran on, set by the
#: operator (e.g. ``"1x RTX 3090 24GB"``). Never detected: a host name or
#: address in a public artifact names infrastructure, not hardware.
HARDWARE_ENV = "RAVEN_EVAL_HARDWARE"

#: Fields of one pass line, in the order they are written.
PASS_FIELDS = (
    "started_at", "finished_at", "wall_s", "n_requests", "n_ok", "audio_s",
    "concurrency", "adapter", "hardware", "interrupted",
)

# Relative + absolute slack for re-deriving a rounded ratio from rounded inputs.
_REL_TOL = 1e-3
_ABS_TOL = 1e-3


def throughput_path(out_dir: Path, safe_subset: str) -> Path:
    """The pass log of one subset; ``safe_subset`` is already filename-safe."""
    return out_dir / f"throughput_{safe_subset}.jsonl"


def hardware_label() -> str | None:
    """The operator's hardware label, or None when unset or blank."""
    label = os.environ.get(HARDWARE_ENV, "").strip()
    return label or None


def append_pass(path: Path, record: dict[str, Any]) -> None:
    """Append one pass line and force it to disk.

    Called on the way out of a pass that may be dying (SIGTERM), so the line is
    fsynced rather than left in a buffer the exit could drop.
    """
    line = {f: record[f] for f in PASS_FIELDS}
    with path.open("a", encoding="utf-8") as fh:
        if fh.tell() and not _ends_with_newline(path):
            fh.write("\n")  # never glue a line onto a torn one
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


def read_passes(path: Path) -> list[dict[str, Any]]:
    """Every complete pass line of ``path``, oldest first; [] if there is none.

    A torn line (the process died mid-write) is skipped, like the predictions
    log's.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    passes: list[dict[str, Any]] = []
    for raw in text.splitlines():
        try:
            line = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(line, dict) and all(f in line for f in PASS_FIELDS):
            passes.append(line)
    return passes


def _one_or_list(values: list[Any]) -> Any:
    distinct = list(dict.fromkeys(values))
    return distinct[0] if len(distinct) == 1 else distinct


def summarize(passes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The ``throughput`` block of one subset's summary row, or None without passes.

    ``concurrency`` and ``hardware`` are a single value when every pass shares
    it and the list of distinct values otherwise. Throughput depends on both, so
    passes that differ in either add up to a number that describes no single
    setup: ``comparable`` is then false and ``comparable_reason`` says why.
    """
    if not passes:
        return None
    total_wall = round(sum(float(p["wall_s"]) for p in passes), 3)
    total_audio = round(sum(float(p["audio_s"]) for p in passes), 3)
    n_ok = sum(int(p["n_ok"]) for p in passes)
    concurrency = _one_or_list([p["concurrency"] for p in passes])
    hardware = _one_or_list([p["hardware"] for p in passes])
    reasons = []
    if isinstance(concurrency, list):
        reasons.append(f"passes ran at different concurrency {concurrency}")
    if isinstance(hardware, list):
        reasons.append(f"passes ran on different hardware {hardware}")
    return {
        "passes": passes,
        "total_wall_s": total_wall,
        "total_audio_s": total_audio,
        "n_ok": n_ok,
        "rtfx": round(total_audio / total_wall, 4) if total_wall > 0 else None,
        "utterances_per_s": round(n_ok / total_wall, 4) if total_wall > 0 else None,
        "concurrency": concurrency,
        "hardware": hardware,
        "comparable": not reasons,
        "comparable_reason": "; ".join(reasons) or None,
    }


def _close(got: object, want: float) -> bool:
    return isinstance(got, (int, float)) and not isinstance(got, bool) and math.isclose(
        float(got), want, rel_tol=_REL_TOL, abs_tol=_ABS_TOL
    )


def throughput_problem(block: object) -> str:
    """Why a committed ``throughput`` block contradicts itself, or ``""``.

    Holds the totals to the sum of the passes and the ratios to the totals,
    within rounding. Says nothing about whether the run was as fast as claimed —
    no artifact can.
    """
    if block is None:
        return ""
    if not isinstance(block, dict):
        return "throughput is not an object"
    passes = block.get("passes")
    if not isinstance(passes, list) or not passes:
        return "throughput.passes is not a non-empty list"
    try:
        wall = sum(float(p["wall_s"]) for p in passes)
        audio = sum(float(p["audio_s"]) for p in passes)
        n_ok = sum(int(p["n_ok"]) for p in passes)
    except (KeyError, TypeError, ValueError):
        return "throughput.passes holds a line without numeric wall_s/audio_s/n_ok"
    problems = []
    if not _close(block.get("total_wall_s"), wall):
        problems.append(f"total_wall_s={block.get('total_wall_s')!r} vs passes {wall:.3f}")
    if not _close(block.get("total_audio_s"), audio):
        problems.append(f"total_audio_s={block.get('total_audio_s')!r} vs passes {audio:.3f}")
    if block.get("n_ok") != n_ok:
        problems.append(f"n_ok={block.get('n_ok')!r} vs passes {n_ok}")
    total_wall, total_audio = block.get("total_wall_s"), block.get("total_audio_s")
    if _close(total_wall, wall) and _close(total_audio, audio) and float(total_wall) > 0:  # type: ignore[arg-type]
        rtfx = float(total_audio) / float(total_wall)  # type: ignore[arg-type]
        if not _close(block.get("rtfx"), rtfx):
            problems.append(f"rtfx={block.get('rtfx')!r} vs total_audio_s/total_wall_s {rtfx:.4f}")
        ups = n_ok / float(total_wall)  # type: ignore[arg-type]
        if not _close(block.get("utterances_per_s"), ups):
            problems.append(
                f"utterances_per_s={block.get('utterances_per_s')!r} vs n_ok/total_wall_s {ups:.4f}"
            )
    return "throughput: " + "; ".join(problems) if problems else ""
