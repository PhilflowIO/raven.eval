"""Deepgram diarization adapter (behind the ``diar-hosted`` extra).

The first **hosted** diarizer in the harness, and therefore the first caller of
the shared word→turn aggregator in :mod:`raven_diar.adapters.aggregate`. That is
deliberate: Deepgram does not return speaker *turns*, it returns *words* each
carrying a ``speaker`` index, and the folding of words into turns is a scoring
decision. Doing that folding here, privately, would make a DER *difference*
between two hosted providers partly a difference between two folding rules. So
this module contributes zero folding logic of its own — it maps the response to
``(start, end, speaker)`` spans and hands them to ``spans_to_turns``.

Adding it touched a module and one :class:`~raven_diar.config.DiarizerSpec`
entry; the runner's dispatch was not edited (see ``raven_diar/registry.py``).

**No diarization-only endpoint exists.** Deepgram diarizes as a flag on the
*pre-recorded transcription* request (``POST /v1/listen``), so every DER file
also pays for a transcription. There is no separate diarization price line;
verified against <https://deepgram.com/pricing> on 2026-09-03, where diarization
is included in the pre-recorded base rate (0.258 $/h at the budgeted tier) and
only the *streaming* tab charges a diarization surcharge. This adapter therefore
uses the pre-recorded endpoint exclusively.

**Two pins, because two models run.** A published number must not drift under a
vendor's silent model update, so both halves of the request are pinned:

* ``model`` — the ASR model, which produces the word timings the turns are built
  from. Pinned to an explicit option string (``nova-3-general``), not the
  floating family alias ``nova-3``.
* ``diarize_model`` — the diarizer itself. Deepgram versions its diarizers
  (``v1`` / ``v2``, ``latest`` resolving to the newest GA batch model) and
  documents ``diarize_model`` as *both* the enable flag and the version
  selector; the deprecated boolean ``diarize=true`` always routes to v1 and
  setting both is rejected. So ``DiarizerSpec.revision`` carries the
  ``diarize_model`` value — that is this adapter's analogue of an HF revision
  hash. See <https://developers.deepgram.com/docs/diarization>.

Deepgram also echoes ``metadata.diarize_info`` (``model_uuid`` + ``arch``) on any
request where a diarizer actually ran. We keep it in :attr:`DiarizeResult.raw`,
which turns the pin into evidence rather than an assertion — and its *absence* is
how the API signals "the diarizer did not run", which we raise on rather than
silently score as a one-speaker file.

``httpx`` is imported lazily inside the request path so importing this module for
the registry pulls nothing, keeping the Tier-1 re-score path dependency-free.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from raven_eval_core.der import DiarSegment

from .aggregate import DEFAULT_GAP_MERGE_S, LabelledSpan, spans_to_turns
from .base import DiarizeResult

DEFAULT_BASE_URL = "https://api.deepgram.com/v1"
#: ASR model whose word timings the turns are folded from. An explicit option
#: string, not the family alias ``nova-3``, which Deepgram is free to repoint.
DEFAULT_MODEL = "nova-3-general"
#: Default ``diarize_model``: the newest GA *batch* diarizer, pinned by version
#: rather than ``latest`` so a published DER cannot move when Deepgram ships v3.
DEFAULT_DIARIZE_MODEL = "v2"
DEFAULT_LANGUAGE = "de"
DEFAULT_TIMEOUT_S = 600.0

#: Env override for the shared gap threshold. Exists ONLY so the calibration
#: sweep documented in ``adapters/aggregate.py`` can be run without editing code;
#: a run that sets it is publishing a deviation and must say so in its row.
GAP_MERGE_ENV = "DIAR_GAP_MERGE_S"


def _resolve_gap_merge(explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(GAP_MERGE_ENV)
    return float(raw) if raw else DEFAULT_GAP_MERGE_S


def words_to_spans(body: dict) -> list[LabelledSpan]:
    """Extract ``(start, end, speaker)`` word spans from a Deepgram response.

    Pure and network-free so the response contract is unit-testable without an
    API key. Words missing a ``speaker`` label are dropped: an unlabelled word
    carries no diarization information, and inventing a speaker for it would
    manufacture confusion the model never claimed.

    Raises:
        ValueError: if the diarizer did not run (no ``metadata.diarize_info``) or
            the response carries no word list at all — both are failures that
            must not be scored as "this file had one speaker".
    """
    metadata = body.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("diarize_info"), dict
    ):
        raise ValueError(
            "Deepgram response has no metadata.diarize_info — the diarizer did "
            "not run for this request (check the diarize_model parameter). "
            "Refusing to score an undiarized transcript."
        )
    results = body.get("results")
    channels = results.get("channels") if isinstance(results, dict) else None
    if not isinstance(channels, list) or not channels:
        raise ValueError("Deepgram response has no results.channels")
    first = channels[0]
    alternatives = first.get("alternatives") if isinstance(first, dict) else None
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("Deepgram response has no alternatives")
    words = alternatives[0].get("words") if isinstance(alternatives[0], dict) else None
    if not isinstance(words, list):
        raise ValueError("Deepgram response has no word list")

    spans: list[LabelledSpan] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        speaker = word.get("speaker")
        if speaker is None:
            continue
        start, end = word.get("start"), word.get("end")
        if start is None or end is None:
            continue
        spans.append(LabelledSpan(float(start), float(end), f"speaker_{speaker}"))
    return spans


class DeepgramDiarizer:
    """Diarizer adapter wrapping Deepgram's pre-recorded ``/v1/listen``."""

    def __init__(
        self,
        provider_id: str = "deepgram-nova-3",
        model_id: str = DEFAULT_MODEL,
        revision: str | None = None,
        api_key_env: str = "DEEPGRAM_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        language: str = DEFAULT_LANGUAGE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        gap_merge_s: float | None = None,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set — required for the Deepgram diarizer "
                "(docs/TIER2-DER-KEYS.md). This repo ships env-var names only."
            )
        self.provider_id = provider_id
        self.model_id = model_id
        # Revision precedence: explicit arg > env override > the pinned default.
        # NOT "latest": a floating alias would let a published DER drift under a
        # vendor model update, which is the one thing a benchmark may not do.
        self.revision = (
            revision or os.environ.get("DEEPGRAM_DIARIZE_MODEL") or DEFAULT_DIARIZE_MODEL
        )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._timeout_s = timeout_s
        self._gap_merge_s = _resolve_gap_merge(gap_merge_s)

    async def _alisten(self, audio_bytes: bytes) -> dict:
        import httpx  # diar-hosted extra

        from raven_asr.retry import with_retry  # shared transient-failure policy

        @with_retry()
        async def _post() -> dict:
            headers = {
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "audio/wav",
            }
            params = {
                "model": self.model_id,
                "language": self._language,
                # Enables diarization AND selects its version in one parameter;
                # never combined with the deprecated boolean `diarize`, which
                # Deepgram rejects when both are set.
                "diarize_model": self.revision,
            }
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(
                    f"{self._base_url}/listen",
                    content=audio_bytes,
                    params=params,
                    headers=headers,
                )
            response.raise_for_status()
            return response.json()

        return await _post()

    def diarize(self, audio_path: Path) -> DiarizeResult:
        audio_bytes = Path(audio_path).read_bytes()
        started = time.perf_counter()
        body = asyncio.run(self._alisten(audio_bytes))
        latency_s = time.perf_counter() - started
        spans = words_to_spans(body)
        # The ONE folding path (adapters/aggregate.py) — no private variant here.
        segments: list[DiarSegment] = spans_to_turns(
            spans, gap_merge_s=self._gap_merge_s
        )
        metadata = body.get("metadata", {})
        return DiarizeResult(
            segments=segments,
            latency_s=latency_s,
            raw={
                "model_id": self.model_id,
                "revision": self.revision,
                "language": self._language,
                "gap_merge_s": self._gap_merge_s,
                "n_words": len(spans),
                # Evidence that the pin took effect, straight from the vendor.
                "diarize_info": metadata.get("diarize_info"),
                "model_info": metadata.get("model_info"),
                "request_id": metadata.get("request_id"),
            },
        )


#: Registry entry point (see raven_diar.registry) — the runner resolves
#: ``DiarizerSpec.adapter == "deepgram"`` to this attribute.
ADAPTER = DeepgramDiarizer
