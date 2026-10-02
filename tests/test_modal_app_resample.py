"""The Modal adapter must deliver 16 kHz, whatever rate the corpus has."""

from __future__ import annotations

import asyncio
import io
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from raven_asr.adapters import modal_app
from raven_asr.adapters.modal_app import MODAL_INPUT_SAMPLE_RATE, ModalAppAdapter


class _FakeFunction:
    """Stands in for a Modal function; enforces the apps' 16 kHz contract."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def remote(self, audio_bytes: bytes, sample_rate: int) -> str:
        audio, header_sr = sf.read(io.BytesIO(audio_bytes))
        if sample_rate != 16_000 or header_sr != 16_000:
            raise ValueError(f"Only 16 kHz audio supported, got {sample_rate}")
        self.calls.append((sample_rate, header_sr, len(audio)))
        return "ok"


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> tuple[ModalAppAdapter, _FakeFunction]:
    fake = _FakeFunction()
    monkeypatch.setattr(ModalAppAdapter, "_lookup_function", lambda self, name: fake)
    return (
        ModalAppAdapter(provider_id="modal-parakeet", model_id="m", app_name="a"),
        fake,
    )


def test_native_44k_clip_reaches_the_app_as_16k(adapter: Any) -> None:
    ad, fake = adapter
    sr = 44_100
    t = np.arange(sr * 2) / sr
    clip = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    result = asyncio.run(ad.atranscribe(clip, sr))

    assert result.text == "ok"
    sent_sr, header_sr, n = fake.calls[0]
    assert (sent_sr, header_sr) == (MODAL_INPUT_SAMPLE_RATE, MODAL_INPUT_SAMPLE_RATE)
    assert n == 2 * MODAL_INPUT_SAMPLE_RATE  # duration preserved
    assert result.raw["input_sample_rate"] == sr
    assert result.raw["sent_sample_rate"] == MODAL_INPUT_SAMPLE_RATE


def test_16k_clip_is_passed_through_unchanged() -> None:
    clip = np.random.default_rng(0).standard_normal(16_000).astype(np.float32)
    assert modal_app.to_input_rate(clip, 16_000) is clip


def test_resampling_keeps_the_signal() -> None:
    sr = 44_100
    t = np.arange(sr) / sr
    out = modal_app.to_input_rate(np.sin(2 * np.pi * 440 * t).astype(np.float32), sr)
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = np.argmax(spectrum) * MODAL_INPUT_SAMPLE_RATE / len(out)
    assert abs(peak_hz - 440) < 2
