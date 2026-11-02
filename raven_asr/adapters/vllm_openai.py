"""OpenAI-compatible ``/v1/audio/transcriptions`` adapter (async).

Talks to the two self-hosted server types the harness measures — a NeMo-style
server and a vLLM server (or a shim in front of one). Both take the same
multipart request; which of the two a model is served by is a registry fact
(``raven_asr.config.KNOWN_MODELS``), carried here as the *name* of the env var
holding the endpoint. The URL itself never enters this repository.

Three properties a long unattended run depends on:

* **One HTTP client per adapter instance**, opened lazily inside the running
  event loop and closed by :meth:`aclose`. A client per request pays a TCP
  handshake per utterance and, at a few thousand utterances an hour, leaves the
  host with a socket churn the measurement has nothing to do with.
* **16 kHz mono PCM16 WAV** on the wire, resampled here. The self-hosted servers
  are built around that rate; sending a 48 kHz Swiss clip and letting each server
  convert (or not) would make the input differ by server.
* **Served-model check.** A server that answers for a different model than the
  run is labelled with produces plausible transcripts under the wrong name. When
  the response carries ``x_raven.model`` and it differs from the posted model,
  :class:`~raven_asr.adapters.base.ServedModelMismatchError` aborts the run. A
  server that sends no echo at all is warned about once and trusted.
* **Serving identity.** The first response's ``x_raven`` engine / recipe /
  batching (:mod:`raven_asr.serving`) is pinned as :attr:`serving` and recorded
  with each pass; a later response with a different block raises
  :class:`~raven_asr.serving.ServingChangedError` (a mismatch, exit 3). A server
  that reports none pins ``None`` and is warned about once.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx
import numpy as np

from .. import serving as serving_block
from ..retry import with_retry
from .base import (
    ServedModelMismatchError,
    ServingChangedError,
    TranscribeResult,
    encode_wav_pcm16,
    resample_mono,
)

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "de"
DEFAULT_TIMEOUT_S = 300.0
#: The rate every self-hosted endpoint is sent, whatever the corpus's native rate.
INPUT_SAMPLE_RATE = 16_000

#: Optional Bearer token shared by both self-hosted server types.
DEFAULT_API_KEY_ENV = "BENCH_STT_API_KEY"


class VllmOpenAIAdapter:
    """Posts 16 kHz WAV bytes to ``{base_url}/audio/transcriptions``."""

    def __init__(
        self,
        *,
        provider_id: str,
        model_id: str,
        base_url_env: str,
        api_key_env: str | None = DEFAULT_API_KEY_ENV,
        language: str = DEFAULT_LANGUAGE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        base_url = os.environ.get(base_url_env)
        if not base_url:
            raise RuntimeError(
                f"{base_url_env} is not set — expected the base URL of an "
                f"OpenAI-compatible endpoint including /v1 (docs/TIER2-KEYS.md)"
            )
        self.provider_id = provider_id
        self.model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._api_key = os.environ.get(api_key_env) if api_key_env else None
        self._language = language
        self._timeout_s = timeout_s
        # Only tests pass one (httpx.MockTransport); None is the network.
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._warned_no_echo = False
        #: The serving block this adapter is pinned to (None: none reported).
        self.serving: dict[str, object] | None = None
        self._serving_pinned = False
        self._pinned_by_resume = False

    def _http(self) -> httpx.AsyncClient:
        """The adapter's one client, created on first use inside the running loop.

        The runner bounds concurrency with its own semaphore, so the connection
        pool is left unbounded rather than silently capping it a second time.
        """
        if self._client is None:
            headers = (
                {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            )
            self._client = httpx.AsyncClient(
                timeout=self._timeout_s,
                headers=headers,
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                ),
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client; the adapter can be used again afterwards."""
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()

    def _check_served_model(self, body: dict[str, object]) -> None:
        x_raven = body.get("x_raven")
        served = x_raven.get("model") if isinstance(x_raven, dict) else None
        if served is None:
            if not self._warned_no_echo:
                self._warned_no_echo = True
                logger.warning(
                    "[%s] the endpoint does not report which model it serves "
                    "(no x_raven.model in the response); trusting that it serves "
                    "%s", self.provider_id, self.model_id,
                )
            return
        if served != self.model_id:
            raise ServedModelMismatchError(
                f"[{self.provider_id}] the endpoint serves {served!r}, but this "
                f"run measures {self.model_id!r}. Load the right model on the "
                f"server, or point the endpoint env var at the server that has it."
            )

    @property
    def serving_known(self) -> bool:
        """Whether :attr:`serving` is pinned (a response arrived, or a resume primed it)."""
        return self._serving_pinned

    def expect_serving(self, block: dict[str, object]) -> None:
        """Pin the serving block a resumed subset's earlier passes recorded.

        Called before the first request: the first answer from another setup
        then aborts the run before any transcript of it is added to the subset.
        """
        if self._serving_pinned and block != self.serving:
            raise ServingChangedError(
                f"[{self.provider_id}] this run is served by "
                f"{serving_block.describe(self.serving)}, but earlier passes of the "
                f"subset being resumed recorded {serving_block.describe(block)}. "
                "Restore that setup, or remove the subset's output to measure it "
                "again."
            )
        self.serving, self._serving_pinned = block, True
        self._pinned_by_resume = True

    def _check_serving(self, body: dict[str, object]) -> None:
        block = serving_block.from_x_raven(body.get("x_raven"))
        if not self._serving_pinned:
            self.serving, self._serving_pinned = block, True
            if block is None:
                logger.warning(
                    "[%s] the endpoint does not report its serving recipe "
                    "(no x_raven.engine/recipe/batching); results will record "
                    "serving: null", self.provider_id,
                )
            return
        if block != self.serving and self._pinned_by_resume:
            raise ServingChangedError(
                f"[{self.provider_id}] earlier passes of the subset being resumed "
                f"were served by {serving_block.describe(self.serving)}, this "
                f"endpoint answers with {serving_block.describe(block)}. Restore "
                "that setup, or remove the subset's output to measure it again."
            )
        if block != self.serving:
            raise ServingChangedError(
                f"[{self.provider_id}] the endpoint's serving changed mid-run: "
                f"pinned {serving_block.describe(self.serving)}, now "
                f"{serving_block.describe(block)}. Transcripts from two setups "
                "cannot share one score; restart the run on one setup."
            )

    @with_retry()
    async def atranscribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> TranscribeResult:
        wav_bytes = encode_wav_pcm16(
            resample_mono(audio, sample_rate, INPUT_SAMPLE_RATE), INPUT_SAMPLE_RATE
        )
        files = {"file": ("clip.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model_id, "language": self._language}
        started = time.monotonic()
        r = await self._http().post(
            f"{self._base_url}/audio/transcriptions", files=files, data=data
        )
        r.raise_for_status()
        latency = time.monotonic() - started
        body: dict[str, object] = r.json()
        self._check_served_model(body)
        self._check_serving(body)
        text = str(body.get("text", "")).strip()
        return TranscribeResult(
            text=text,
            latency_s=latency,
            raw={
                "body": body,
                "input_sample_rate": sample_rate,
                "sent_sample_rate": INPUT_SAMPLE_RATE,
            },
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscribeResult:
        async def _once() -> TranscribeResult:
            try:
                return await self.atranscribe(audio, sample_rate)
            finally:
                await self.aclose()

        return asyncio.run(_once())
