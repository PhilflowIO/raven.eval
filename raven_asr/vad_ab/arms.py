"""The detectors under test, each reduced to a frame score and a frame decision.

Every arm answers two things for one track:

* ``score10`` — a per-10 ms score where higher means "more speech" (dBFS peak,
  dB over the track floor, or a model probability). Only used for ROC-AUC.
* ``islands(threshold)`` — the islands production would build when this arm
  decides what is speech at ``threshold``: the arm's frame decision becomes
  silences (:func:`raven_asr.load.vad.silences_from_mask`, or silencedetect's
  own sample rule for arm 0), and pad / merge / minimum island are production's
  (:func:`raven_asr.load.vad.build_speech_islands`).

Engines (onnxruntime, torch, fireredvad, NeMo, pyannote) are imported only
inside the engine that needs them, so this module and the scoring are testable
without any of them. Every engine runs single-threaded, so the CPU seconds per
audio hour are comparable and correspond to one core.
"""

from __future__ import annotations

import math
import os
import resource
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from raven_asr.load import vad as planner

from .frames import HOP, hold, hysteresis, n_frames

SAMPLE_RATE = 16000

#: Silero's off threshold sits 0.15 below its on threshold (its own default
#: 0.5 / 0.35; the research report's recall setting 0.3 / 0.15 keeps the gap).
SILERO_OFF_DELTA = 0.15

#: Arm 3's decision copy is brought to this level: the 95th percentile of the
#: track's 50 ms window levels (its loud speech, not its room) lands here. The
#: gain is never negative and at most +40 dB, and only the detector hears it.
NORMALISE_TARGET_DB = -20.0
NORMALISE_PERCENTILE = 95
NORMALISE_MAX_GAIN_DB = 40.0

_PROB_SWEEP = (0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
               0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


@dataclass(frozen=True)
class Arm:
    """One detector setting. ``default`` is its production-candidate threshold."""

    name: str
    kind: str  # "silencedetect" | "adaptive" | "neural"
    default: float
    sweep: tuple[float, ...]
    engine: str | None = None  # neural only: "silero" | "firered" | "marblenet" | "pyannote"
    off_delta: float = 0.0  # neural only: on − off of the hysteresis
    normalise: bool = False
    note: str = ""
    params: dict = field(default_factory=dict)

    def off(self, on: float) -> float:
        return max(0.0, on - self.off_delta)


ARMS: dict[str, Arm] = {a.name: a for a in (
    Arm("0-silencedetect", "silencedetect", -40.0,
        tuple(float(x) for x in np.arange(-75.0, -12.4, 2.5)),
        note="ffmpeg silencedetect noise=-40dB d=0.6 (today)"),
    Arm("1-adaptive-floor", "adaptive", 0.0, tuple(float(x) for x in np.arange(-20.0, 20.1, 2.0)),
        note="per-track floor: 10th pct of open-mic 50 ms levels + 8 dB, held in [-60, -40] dB"),
    Arm("2-silero-0.5", "neural", 0.5, _PROB_SWEEP, engine="silero", off_delta=SILERO_OFF_DELTA,
        note="Silero VAD v6.2 ONNX, on 0.5 / off 0.35"),
    Arm("3-silero-0.3-norm", "neural", 0.3, _PROB_SWEEP, engine="silero",
        off_delta=SILERO_OFF_DELTA, normalise=True,
        note="Silero VAD v6.2 ONNX, on 0.3 / off 0.15, decision copy normalised to -20 dBFS (p95)"),
    Arm("4-firered", "neural", 0.4, _PROB_SWEEP, engine="firered",
        note="FireRedVAD offline (VAD/), 5-frame smoothed probability, single threshold 0.4 (its default)"),
    Arm("5-marblenet", "neural", 0.5, _PROB_SWEEP, engine="marblenet",
        note="NVIDIA Frame-VAD Multilingual MarbleNet v2.0, speech probability per 20 ms, threshold 0.5"),
    Arm("6-pyannote-seg", "neural", 0.5, _PROB_SWEEP, engine="pyannote",
        note="pyannote segmentation-3.0, max speaker activity, threshold 0.5 (optional recall ceiling)"),
)}


# ── audio ────────────────────────────────────────────────────────────────────


def read_pcm16(path: str) -> np.ndarray:
    """Mono 16 kHz int16 samples of a WAV (the decoded track production plans on)."""
    import soundfile as sf

    data, rate = sf.read(path, dtype="int16", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: {rate} Hz, expected {SAMPLE_RATE}")
    return np.ascontiguousarray(data[:, 0])


def normalise_gain_db(pcm16: np.ndarray) -> float:
    """Arm 3's static gain: bring the 95th percentile of the 50 ms levels to −20 dBFS."""
    levels = [lv for lv in planner.window_levels(pcm16) if lv > planner.DIGITAL_SILENCE_DB]
    if not levels:
        return 0.0
    loud = float(np.percentile(levels, NORMALISE_PERCENTILE))
    return float(min(NORMALISE_MAX_GAIN_DB, max(0.0, NORMALISE_TARGET_DB - loud)))


# ── engines ──────────────────────────────────────────────────────────────────


class SileroOnnx:
    """Silero VAD v6 ONNX with its own I/O — no torch (the shape production would ship).

    Mirrors ``silero_vad.utils_vad.OnnxWrapper`` at 16 kHz: 512-sample windows,
    each prefixed by the previous window's last 64 samples, a (2, 1, 128) state
    carried across windows, the trailing partial window zero-padded.
    """

    WINDOW = 512
    CONTEXT = 64

    def __init__(self, model_path: str | None = None):
        import onnxruntime as ort

        if model_path is None:
            import silero_vad

            model_path = os.path.join(os.path.dirname(silero_vad.__file__), "data", "silero_vad.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])

    def probs(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        w, c = self.WINDOW, self.CONTEXT
        n = int(math.ceil(len(x) / w))
        padded = np.zeros(n * w, dtype=np.float32)
        padded[:len(x)] = x
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, c), dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        out = np.empty(n, dtype=np.float32)
        run = self.session.run
        for i in range(n):
            frame = np.concatenate((context, padded[i * w:(i + 1) * w][None, :]), axis=1)
            prob, state = run(None, {"input": frame, "state": state, "sr": sr})
            out[i] = prob.reshape(-1)[0]
            context = frame[:, -c:]
        return out, w / SAMPLE_RATE


class FireRed:
    """FireRedVAD's offline model; probabilities smoothed as its own post-processor does."""

    def __init__(self, model_dir: str):
        import torch
        from fireredvad.vad import FireRedVad, FireRedVadConfig

        torch.set_num_threads(1)
        self.cfg = FireRedVadConfig()
        self.vad = FireRedVad.from_pretrained(model_dir, self.cfg)

    def probs(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        import torch

        pcm16 = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)
        with torch.no_grad():
            _, probs = self.vad.detect((pcm16, SAMPLE_RATE), do_postprocess=False)
        p = np.asarray(probs, dtype=np.float32).reshape(-1)
        return smooth(p, self.cfg.smooth_window_size), 0.01


def smooth(p: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average over ``window`` frames (FireRedVAD's ``smooth_window_size``)."""
    if window <= 1 or p.size == 0:
        return p
    c = np.cumsum(np.concatenate(([0.0], p.astype(np.float64))))
    idx = np.arange(1, p.size + 1)
    lo = np.maximum(0, idx - window)
    return ((c[idx] - c[lo]) / (idx - lo)).astype(np.float32)


class MarbleNet:
    """NVIDIA Frame-VAD Multilingual MarbleNet v2.0 through NeMo, speech probability per 20 ms."""

    def __init__(self, nemo_path: str):
        import torch
        from nemo.collections.asr.models import EncDecFrameClassificationModel

        torch.set_num_threads(1)
        # The checkpoint carries no weight for the training loss; strict=False skips only that.
        self.model = EncDecFrameClassificationModel.restore_from(nemo_path, map_location="cpu", strict=False)
        self.model.eval()

    def probs(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        import torch

        sig = torch.from_numpy(np.asarray(x, dtype=np.float32))[None, :]
        length = torch.tensor([sig.shape[1]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_signal=sig, input_signal_length=length)
        prob = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy().astype(np.float32)
        return prob, 0.02


class PyannoteSeg:
    """pyannote segmentation-3.0 as a speech detector: the most active speaker's probability."""

    def __init__(self, model: str = "pyannote/segmentation-3.0"):
        import torch
        from pyannote.audio import Inference, Model

        torch.set_num_threads(1)
        m = Model.from_pretrained(model, token=os.environ.get("HF_TOKEN"))
        self.inference = Inference(m, window="sliding", duration=10.0, step=2.5)

    def probs(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        import torch

        out = self.inference({"waveform": torch.from_numpy(np.asarray(x, np.float32))[None, :],
                              "sample_rate": SAMPLE_RATE})
        data = np.asarray(out.data, dtype=np.float32)
        return data.max(axis=1), float(out.sliding_window.step)


_ENGINES: dict[str, object] = {}


def engine(name: str) -> object:
    """One engine per process, built on first use from the environment's model paths."""
    if name not in _ENGINES:
        if name == "silero":
            _ENGINES[name] = SileroOnnx(os.environ.get("SILERO_ONNX"))
        elif name == "firered":
            _ENGINES[name] = FireRed(os.environ["FIRERED_VAD_DIR"])
        elif name == "marblenet":
            _ENGINES[name] = MarbleNet(os.environ["MARBLENET_NEMO"])
        elif name == "pyannote":
            _ENGINES[name] = PyannoteSeg()
        else:
            raise KeyError(name)
    return _ENGINES[name]


# ── one arm over one track ───────────────────────────────────────────────────


def _cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _children_cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def ffmpeg_silencedetect_cpu(wav_path: str) -> float:
    """CPU seconds the silencedetect filter itself costs on this file.

    Production runs it inside the decode it needs anyway, so its cost is the
    decode with the filter minus the same decode without it.
    """
    if shutil.which("ffmpeg") is None:
        return float("nan")

    def run(af: str) -> float:
        before = _children_cpu()
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
                        "-i", wav_path, "-af", af, "-f", "null", "-"], check=True)
        return _children_cpu() - before

    with_filter = run(f"silencedetect=noise={planner.NOISE_FLOOR_DB}dB:d={planner.MIN_SILENCE_S}")
    without = run("anull")
    return max(0.0, with_filter - without)


def neural_islands(score10: np.ndarray, arm: Arm, on: float, duration: float,
                   veto: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Islands from a neural arm's grid scores at ``on`` (optionally minus vetoed frames)."""
    mask = hysteresis(score10, on, arm.off(on))
    if veto is not None:
        mask = mask & ~veto[:mask.size]
    return planner.build_speech_islands(planner.silences_from_mask(mask, HOP), duration)


def run_arm(arm: Arm, wav_path: str, *, probs_fn: Callable | None = None) -> dict:
    """Score one track with one arm: grid scores, CPU seconds, islands per swept threshold."""
    pcm = read_pcm16(wav_path)
    duration = len(pcm) / SAMPLE_RATE
    n = n_frames(duration)
    out: dict = {"arm": arm.name, "wav": wav_path, "duration_s": duration}

    if arm.kind == "silencedetect":
        x = pcm.astype(np.float32) / 32768.0
        islands = {}
        for thr in sorted({*arm.sweep, arm.default}):
            sil = planner.silencedetect_silences(x, SAMPLE_RATE, noise_db=thr)
            islands[thr] = planner.build_speech_islands(sil, duration)
        hop_n = int(HOP * SAMPLE_RATE)
        pad = np.zeros(n * hop_n, dtype=np.float32)
        pad[:min(len(x), pad.size)] = np.abs(x[:pad.size])
        peak = pad.reshape(n, hop_n).max(axis=1)
        score10 = 20 * np.log10(np.maximum(peak, 1e-6))
        out["cpu_s"] = ffmpeg_silencedetect_cpu(wav_path)

    elif arm.kind == "adaptive":
        t0 = _cpu()
        levels = planner.window_levels(pcm)
        floor, default_islands = planner.adaptive_islands(levels, duration)
        out["cpu_s"] = _cpu() - t0
        out["floor_db"] = floor
        islands = {0.0: default_islands}
        base = planner.adaptive_floor(levels)
        for off in arm.sweep:
            if off != 0.0:
                islands[off] = planner.adaptive_islands(levels, duration, offset_db=off)[1]
        score10 = hold(np.asarray(levels, dtype=np.float32) - base, planner.LEVEL_WINDOW_S, n)

    else:
        x = pcm.astype(np.float32) / 32768.0
        if arm.normalise:
            gain_db = normalise_gain_db(pcm)
            out["gain_db"] = gain_db
            x = np.clip(x * (10 ** (gain_db / 20.0)), -1.0, 1.0).astype(np.float32)
        fn = probs_fn or engine(arm.engine).probs  # type: ignore[union-attr]
        t0, w0 = _cpu(), time.perf_counter()
        probs, hop = fn(x)
        out["cpu_s"] = _cpu() - t0
        out["wall_s"] = time.perf_counter() - w0
        score10 = hold(probs, hop, n).astype(np.float32)
        islands = {thr: neural_islands(score10, arm, thr, duration)
                   for thr in sorted({*arm.sweep, arm.default})}

    out["islands"] = {repr(float(k)): v for k, v in islands.items()}
    out["score10"] = np.asarray(score10, dtype=np.float32)
    return out
