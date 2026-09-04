#!/usr/bin/env python
"""Measure how offline Sortformer's GPU memory grows with recording length.

This is the script behind ONE claim in BENCHMARKS.md: that the AMI cell for
``sortformer-4spk-v1`` is empty *by construction* rather than by configuration.
The offline checkpoint attends over the whole recording, so activation memory
grows with the square of the duration; the shortest AMI test meeting is 14
minutes, which is already past what a 24 GB card can hold. That sentence decided
which numbers this repo publishes, and until now nothing in the repo reproduced
it.

**This is a Tier-3 diagnostic and it is not a published number.** Peak allocation
depends on the GPU, the driver, the torch build and the NeMo release; the curve
is evidence for a decision, not a result to compare against. Everything that
makes it non-comparable is therefore written into the output JSON — device name,
total VRAM, torch/NeMo versions, the checkpoint revision — so a reader can see at
a glance that it is one machine's answer.

What it does: take one real recording, cut prefixes of increasing length, run the
diarizer on each, and record ``torch.cuda.max_memory_allocated()``. Out-of-memory
is a DATA POINT, not a crash — it is the entire point of the exercise, so it is
caught, recorded with the duration that provoked it, and the sweep stops there.

Usage (needs a GPU and the ``sortformer`` extra):

    uv run --extra sortformer python scripts/measure_sortformer_memory.py \\
        --audio data/diar/callhome-de/audio/callhome-deu-0000.wav \\
        --out diagnostics/sortformer-4spk-v1-memory.json

The default duration ladder stops at 12 minutes because that is where a 24 GB
card is expected to fail; ``--durations`` overrides it for a bigger card.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raven_diar.config import KNOWN_DIARIZERS  # noqa: E402
from raven_diar.registry import DIARIZER_ADAPTERS  # noqa: E402

#: Seconds of audio per measurement. Quadratic growth means the interesting part
#: is the top of the ladder, so the steps are even rather than dense-at-the-start.
DEFAULT_DURATIONS = (120, 240, 360, 480, 600, 720)

#: Which diarizer the claim is about. The streaming v2 checkpoint is deliberately
#: also runnable here (``--model``), because "memory does not track duration" is
#: the other half of the same claim and deserves the same evidence.
DEFAULT_MODEL = "sortformer-4spk-v1"


def _cut_prefix(source: Path, seconds: float, dest: Path) -> float:
    """Write the first ``seconds`` of ``source`` to ``dest``; return real length.

    Returns the length actually written, which is shorter than asked when the
    source recording runs out — a truncated point measured as if it were the
    full duration would bend the curve downwards exactly where it matters.
    """
    import soundfile as sf  # sortformer extra

    with sf.SoundFile(str(source)) as f:
        rate = f.samplerate
        frames = min(int(seconds * rate), len(f))
        data = f.read(frames=frames, dtype="float32")
    sf.write(str(dest), data, rate)
    return frames / rate


def _fit_quadratic_through_origin(points: list[tuple[float, float]]) -> float | None:
    """Least-squares ``a`` for ``peak_gb ≈ a · duration_s²``.

    Through the origin on purpose: the constant term is the model's own resident
    weights, which do not scale with duration and are reported separately as the
    idle baseline. Fitting an intercept here would let a large constant hide the
    growth term the claim is about.
    """
    num = sum(d * d * gb for d, gb in points)
    den = sum((d * d) ** 2 for d, gb in points)
    return num / den if den else None


def _device_facts() -> dict[str, object]:
    import torch  # sortformer extra

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device — this measurement is about GPU memory and a CPU run "
            "would silently answer a different question"
        )
    props = torch.cuda.get_device_properties(0)
    try:
        nemo_version = __import__("nemo").__version__
    except Exception:  # pragma: no cover - provenance is best-effort
        nemo_version = None
    return {
        "device_name": props.name,
        "device_total_gb": round(props.total_memory / 1024**3, 3),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nemo": nemo_version,
        "driver": _nvidia_driver(),
        "python": platform.python_version(),
        "host": platform.node(),
    }


def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except Exception:  # pragma: no cover - provenance is best-effort
        return None
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None


def measure(
    *,
    audio: Path,
    model_key: str,
    durations: tuple[int, ...],
    workdir: Path,
) -> dict[str, object]:
    """Run the duration ladder and return the committable curve."""
    import torch  # sortformer extra

    spec = KNOWN_DIARIZERS[model_key]
    diarizer = DIARIZER_ADAPTERS.resolve(spec.adapter)(
        provider_id=spec.label, model_id=spec.model_id, revision=spec.revision
    )
    facts = _device_facts()

    # Load the checkpoint before the first measurement and record what it costs
    # resident, so every peak below is "weights + activations" with the weights
    # part known rather than folded into the growth term.
    workdir.mkdir(parents=True, exist_ok=True)
    warmup = workdir / "warmup.wav"
    _cut_prefix(audio, 5.0, warmup)
    diarizer.diarize(warmup)
    torch.cuda.synchronize()
    baseline_gb = torch.cuda.memory_allocated() / 1024**3

    points: list[dict[str, object]] = []
    for wanted in durations:
        clip = workdir / f"clip-{wanted}s.wav"
        actual = _cut_prefix(audio, float(wanted), clip)
        if actual < wanted - 1:
            print(
                f"source is only {actual:.0f}s — stopping before {wanted}s rather "
                f"than measuring a shorter clip as if it were longer",
                file=sys.stderr,
            )
            break
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            diarizer.diarize(clip)
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            points.append({
                "duration_s": round(actual, 3),
                "peak_allocated_gb": round(peak_gb, 4),
                "status": "ok",
            })
            print(f"{actual:6.0f}s  peak {peak_gb:7.3f} GB")
        except torch.cuda.OutOfMemoryError as exc:
            points.append({
                "duration_s": round(actual, 3),
                "peak_allocated_gb": None,
                "status": "oom",
                "error": str(exc).splitlines()[0],
            })
            print(f"{actual:6.0f}s  OOM — this is the answer, not a failure")
            break
        finally:
            clip.unlink(missing_ok=True)
    warmup.unlink(missing_ok=True)

    ok = [
        (float(p["duration_s"]), float(p["peak_allocated_gb"]))
        for p in points if p["status"] == "ok"
    ]
    coefficient = _fit_quadratic_through_origin(ok)
    oom_at = next((p["duration_s"] for p in points if p["status"] == "oom"), None)

    return {
        "tier": 3,
        "not_a_published_number": (
            "Peak allocation is a property of this GPU, driver, torch and NeMo "
            "build, not of the model alone. This curve is the evidence for why "
            "the v1 AMI cell in BENCHMARKS.md is empty; it is not comparable to "
            "any other measurement and must never be quoted as a result."
        ),
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "key": model_key,
            "model_id": spec.model_id,
            "revision": spec.revision,
            "label": spec.label,
        },
        "audio_source": audio.name,
        "environment": facts,
        "resident_after_load_gb": round(baseline_gb, 4),
        "points": points,
        "fit": {
            "form": "peak_allocated_gb = a * duration_s**2",
            "a_gb_per_s2": coefficient,
            "through_origin": True,
            "n_points": len(ok),
        },
        "oom_at_s": oom_at,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure-sortformer-memory", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio", type=Path, required=True,
                        help="one recording at least as long as the last duration")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"diarizer key (default {DEFAULT_MODEL})")
    parser.add_argument("--durations", type=int, nargs="+", default=list(DEFAULT_DURATIONS),
                        help="seconds per measurement, ascending")
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the JSON curve")
    parser.add_argument("--workdir", type=Path, default=Path("results/memory-curve"),
                        help="scratch dir for the cut clips (removed as it goes)")
    args = parser.parse_args(argv)

    if args.model not in KNOWN_DIARIZERS:
        parser.error(
            f"unknown --model {args.model!r}; see raven_diar.config.KNOWN_DIARIZERS"
        )
    if sorted(args.durations) != args.durations:
        parser.error("--durations must be ascending: the sweep stops at the first OOM")
    if not args.audio.is_file():
        parser.error(f"no such audio file: {args.audio}")

    curve = measure(
        audio=args.audio,
        model_key=args.model,
        durations=tuple(args.durations),
        workdir=args.workdir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(curve, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    fit = curve["fit"]["a_gb_per_s2"]
    if fit is not None:
        print(f"fit: peak ≈ {fit:.3g} GB/s² · duration²")
    if curve["oom_at_s"] is not None:
        print(f"OOM at {curve['oom_at_s']:.0f}s on {curve['environment']['device_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
