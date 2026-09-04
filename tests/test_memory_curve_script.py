"""Tests for scripts/measure_sortformer_memory.py — the GPU-free half of it.

The measurement itself needs a 24 GB card and the NeMo stack, so what is guarded
here is everything that could quietly make the resulting curve WRONG rather than
absent: the quadratic fit, the prefix cutter that must not report a truncated
clip as a full-length one, and the argument checks that keep the ladder ordered.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_SPEC = importlib.util.spec_from_file_location(
    "measure_sortformer_memory",
    Path(__file__).resolve().parent.parent / "scripts" / "measure_sortformer_memory.py",
)
mem = importlib.util.module_from_spec(_SPEC)
sys.modules["measure_sortformer_memory"] = mem
_SPEC.loader.exec_module(mem)


def test_importing_the_script_pulls_no_gpu_stack():
    """It resolves adapters through the lazy registry, like the runner does."""
    assert "torch" not in sys.modules
    assert "nemo" not in sys.modules


def test_fit_recovers_a_known_quadratic():
    a = 3.5e-5
    points = [(d, a * d * d) for d in (120, 240, 360, 480, 600)]
    fitted, rms = mem._fit_through_origin(points, 2)
    assert fitted == pytest.approx(a, rel=1e-9)
    assert rms == pytest.approx(0.0, abs=1e-9)
    assert mem._fit_through_origin([], 2) is None


def test_fit_is_through_the_origin_not_least_squares_with_an_intercept():
    """A constant offset must NOT be absorbed into the growth coefficient.

    Resident weights are reported separately; if the fit had an intercept it
    could trade the two off and understate the term the claim rests on.
    """
    a, offset = 3.5e-5, 1.0
    points = [(d, a * d * d + offset) for d in (120, 240, 360, 480, 600)]
    fitted, _rms = mem._fit_through_origin(points, 2)
    assert fitted > a  # the offset is visibly pushed into the coefficient


def test_the_growth_regime_tells_the_two_checkpoints_apart():
    """The whole claim is quadratic-vs-not, so the script must decide it.

    Only ever fitting a quadratic would report a coefficient for the streaming
    checkpoint too, and that coefficient would mean nothing.
    """
    quadratic = [(d, 3.5e-5 * d * d) for d in (120, 240, 360, 480, 600)]
    assert mem._growth_regime(quadratic)["prefers"] == "quadratic"

    linear = [(d, 5e-4 * d) for d in (120, 360, 600, 1200, 1800)]
    regime = mem._growth_regime(linear)
    assert regime["prefers"] == "linear"
    assert regime["fits"]["linear"]["a"] == pytest.approx(5e-4, rel=1e-9)
    # Both laws are always reported, so the verdict can be checked, not trusted.
    assert regime["fits"]["quadratic"]["rms_residual_gb"] > 0
    assert regime["n_points"] == 5

    assert mem._growth_regime([])["prefers"] is None


def test_prefix_cut_reports_the_length_it_actually_wrote(tmp_path: Path):
    rate = 16000
    source = tmp_path / "src.wav"
    sf.write(str(source), np.zeros(10 * rate, dtype="float32"), rate)

    dest = tmp_path / "cut.wav"
    assert mem._cut_prefix(source, 4.0, dest) == pytest.approx(4.0)
    with sf.SoundFile(str(dest)) as f:
        assert len(f) == 4 * rate

    # Asking for more than the source has must report the SHORT length, so the
    # caller can refuse the point instead of plotting it at the wrong duration.
    assert mem._cut_prefix(source, 30.0, dest) == pytest.approx(10.0)


def test_a_descending_ladder_is_rejected(capsys):
    """The sweep stops at the first OOM, so order is part of the contract."""
    with pytest.raises(SystemExit):
        mem.main(["--audio", __file__, "--out", "x.json", "--durations", "600", "120"])
    assert "ascending" in capsys.readouterr().err


def test_an_unknown_model_is_rejected_before_any_gpu_work(capsys):
    with pytest.raises(SystemExit):
        mem.main(["--audio", __file__, "--out", "x.json", "--model", "not-a-diarizer"])
    assert "unknown --model" in capsys.readouterr().err
