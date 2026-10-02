"""Modal serverless GPU adapter (async via ``.remote.aio``).

For long-form audio (clips longer than ``longform_threshold_s``), routes
to a dedicated ``transcribe_longform`` function on the same Modal app.
Chunking strategy lives **inside** the Modal app (Silero-VAD-aligned
splits with decoder-prefix carryover and language=de for encoder-decoder
models; FastConformer local-attention single-call for parakeet). The
adapter side does no chunking, no stitching, no overlap math — that
class of bug is structurally impossible here.

Every Modal app accepts **16 kHz** mono PCM16 only and rejects anything else.
Loaders yield audio at the corpus's native rate (the Swiss corpora are
44.1 kHz), so the adapter converts at this boundary — the one place that knows
the remote's input contract — with soxr at its "VHQ" setting. The hosted
adapters keep sending native audio: each vendor resamples on its own side, and
converting for them here would change what their published numbers measured.

The previous adapter-side ``plan_chunks`` / ``stitch_transcripts`` path
was removed after internal review, once we confirmed (52.2 % best_k=0
rate) that the longest-common-suffix stitcher duplicated overlap regions
on >50 % of boundary pairs. No ``best_k=1`` band-aid was introduced —
root-cause fix only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..retry import with_retry
from .base import TranscribeResult, encode_wav_pcm16

if TYPE_CHECKING:  # pragma: no cover
    pass


logger = logging.getLogger(__name__)

# The only input rate the Modal STT apps accept (they raise on any other).
MODAL_INPUT_SAMPLE_RATE: int = 16_000


def to_input_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample mono float audio to ``MODAL_INPUT_SAMPLE_RATE`` (no-op if equal)."""
    if sample_rate == MODAL_INPUT_SAMPLE_RATE:
        return audio
    import soxr

    return np.asarray(
        soxr.resample(
            np.asarray(audio, dtype=np.float32),
            sample_rate,
            MODAL_INPUT_SAMPLE_RATE,
            quality="VHQ",
        ),
        dtype=np.float32,
    )


# Per-provider default threshold above which the adapter routes to the
# Modal-side ``transcribe_longform`` function. Below this threshold, the
# short-clip ``transcribe`` is used. The Modal-side functions internally
# decide whether to chunk (Silero-VAD) or single-call (parakeet local-attn).
DEFAULT_LONGFORM_THRESHOLD_S: dict[str, float] = {
    "modal-parakeet": 180.0,
    "modal-qwen3-asr": 180.0,
    "modal-voxtral-2507": 180.0,
    "modal-voxtral-2602": 180.0,
}
_DEFAULT_LONGFORM_FALLBACK_S: float = 180.0


class ModalAppAdapter:
    """Invokes a Modal function via the Modal Python SDK."""

    def __init__(
        self,
        *,
        provider_id: str,
        model_id: str,
        app_name: str,
        function_name: str = "transcribe",
        longform_function_name: str | None = "transcribe_longform",
        longform_threshold_s: float | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self._app_name = app_name
        self._function_name = function_name
        self._longform_function_name = longform_function_name
        self._longform_threshold_s = (
            longform_threshold_s
            if longform_threshold_s is not None
            else DEFAULT_LONGFORM_THRESHOLD_S.get(
                provider_id, _DEFAULT_LONGFORM_FALLBACK_S
            )
        )
        self._fn = self._lookup_function(self._function_name)
        self._fn_longform: object | None = (
            self._lookup_function(longform_function_name)
            if longform_function_name
            else None
        )

    def _lookup_function(self, name: str) -> object:
        import modal

        return modal.Function.from_name(self._app_name, name)

    async def _invoke_remote(
        self, fn: object, wav_bytes: bytes, sample_rate: int
    ) -> str:
        fn_any: Any = fn
        remote = fn_any.remote
        aio = getattr(remote, "aio", None)
        if aio is not None:
            text = await aio(wav_bytes, sample_rate)
        else:
            text = await asyncio.to_thread(remote, wav_bytes, sample_rate)
        return str(text).strip()

    @with_retry()
    async def _single_call(
        self, audio: np.ndarray, sample_rate: int, *, longform: bool = False
    ) -> TranscribeResult:
        fn = self._fn_longform if longform else self._fn
        if fn is None:
            raise RuntimeError(
                f"[{self.provider_id}] longform requested but "
                f"longform_function_name is not configured"
            )
        function_name = (
            self._longform_function_name if longform else self._function_name
        )
        wav_bytes = encode_wav_pcm16(
            to_input_rate(audio, sample_rate), MODAL_INPUT_SAMPLE_RATE
        )
        started = time.monotonic()
        text = await self._invoke_remote(fn, wav_bytes, MODAL_INPUT_SAMPLE_RATE)
        latency = time.monotonic() - started
        return TranscribeResult(
            text=text,
            latency_s=latency,
            raw={
                "app": self._app_name,
                "function": function_name,
                "longform": longform,
                "input_sample_rate": sample_rate,
                "sent_sample_rate": MODAL_INPUT_SAMPLE_RATE,
            },
        )

    async def atranscribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> TranscribeResult:
        duration_s = len(audio) / sample_rate
        use_longform = (
            self._fn_longform is not None
            and duration_s > self._longform_threshold_s
        )
        if use_longform:
            logger.info(
                "[%s] routing to %s: %.1fs audio (threshold=%.0fs)",
                self.provider_id,
                self._longform_function_name,
                duration_s,
                self._longform_threshold_s,
            )
        return await self._single_call(audio, sample_rate, longform=use_longform)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        return asyncio.run(self.atranscribe(audio, sample_rate))
