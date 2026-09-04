"""AssemblyAI speaker-diarization adapter (hosted API, behind the ``assemblyai`` extra).

The first *hosted* diarizer in this repo, and therefore the first one that pays
for something it does not use: **AssemblyAI has no diarization-only endpoint**.
Diarization is the ``speaker_labels`` flag on a normal transcription request, so
every DER file also buys a German transcript. That is a cost fact, not an
implementation detail — it is why the per-hour rate below includes the
transcription base price (see ``docs/TIER2-DER-KEYS.md``).

Shape of one call (all three steps are the documented v2 REST flow):

1. ``POST /v2/upload`` — the local wav, raw body. Returns a short-lived
   ``upload_url`` usable only by this account.
2. ``POST /v2/transcript`` — ``{audio_url, speaker_labels: true,
   language_code, speech_models}``.
3. ``GET /v2/transcript/{id}`` until ``status`` is ``completed`` or ``error``.

**The folding is NOT ours to invent.** AssemblyAI returns ``utterances`` — its
own turns, produced by its own private folding rule. Consuming those would make
a DER *difference* against another provider partly a difference of two vendors'
folding heuristics rather than of two diarization models. So this adapter goes
one level finer, to the word-level speaker labels, and folds them with the
shared :mod:`raven_diar.adapters.aggregate` helper at the shared threshold —
the same one every hosted adapter uses. There is deliberately no second folding
path in this file.

**The pin.** AssemblyAI publishes no immutable model version (no hash, no dated
tag): the only selector the API accepts is an alias in ``speech_models``. Worse,
the *default* is a fallback chain, ``["universal-3-5-pro", "universal-2"]`` —
two different models, so an unpinned published DER could silently come from
either. This adapter therefore sends a **single-element** list (no fallback) and
asserts the response's ``speech_model_used`` is that same alias, failing the run
rather than scoring a model we did not ask for. The alias is still an alias: if
AssemblyAI re-trains behind ``universal-3-5-pro``, the number moves and nothing
in the API tells us. That is stated plainly in the docs rather than dressed up
as a version pin.

``httpx`` is imported lazily inside the request helpers so importing this module
(as the registry does when *listing* adapters) pulls no dependency at all — the
Tier-1 re-score path stays install-free.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from .aggregate import DEFAULT_GAP_MERGE_S, LabelledSpan, spans_to_turns
from .base import DiarizeResult

logger = logging.getLogger("raven_diar.adapters.assemblyai")

DEFAULT_BASE_URL = "https://api.assemblyai.com"
#: The alias we send in ``speech_models`` AND assert back in ``speech_model_used``.
DEFAULT_MODEL_ALIAS = "universal-3-5-pro"
#: CALLHOME-de, AMI and VoxConverse-de are German/English speech, never "detect".
#: A benchmark must not let a language detector vary the model path per file.
#: No language DEFAULT here either — see the note in adapters/deepgram.py.
#: The kwarg is named ``language`` like every other adapter's (the runner
#: builds them all with the same kwargs); ``language_code`` stays the name
#: of the field the vendor's request body uses.
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_POLL_INTERVAL_S = 5.0
#: Ceiling on the polling loop for ONE file. CALLHOME calls are ~10 min of audio;
#: an hour without a terminal status means the job is wedged, not slow.
DEFAULT_MAX_POLL_S = 3600.0

#: HTTP statuses worth retrying — transient by definition. Mirrors
#: ``raven_asr.retry.RETRYABLE_STATUS``; kept local because that decorator is
#: async-only and this adapter is deliberately synchronous (the runner is).
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _resolve_api_key(explicit: str | None, env: str) -> str:
    if explicit:
        return explicit
    value = os.environ.get(env)
    if not value:
        raise RuntimeError(
            f"{env} is not set — required for the AssemblyAI diarizer "
            f"(see docs/TIER2-DER-KEYS.md)."
        )
    return value


def _ms_to_s(value: Any) -> float:
    """AssemblyAI reports every timestamp in integer milliseconds."""
    return float(value) / 1000.0


def labelled_spans(body: dict[str, Any]) -> list[LabelledSpan]:
    """Extract word-level ``(start, end, speaker)`` spans from a completed transcript.

    Preference order, finest first — the finer the input, the less of AssemblyAI's
    own folding survives into our turns:

    1. ``utterances[].words[]`` — the documented diarized shape. A word without
       its own ``speaker`` inherits the utterance's label.
    2. top-level ``words[]`` that carry a ``speaker`` — the same words, for
       responses that omit the utterance grouping.
    3. ``utterances[]`` themselves, only if an utterance carries no words at all.

    Raises ``ValueError`` when none of the three yields a speaker-labelled span:
    a transcript that came back without diarization must fail loudly, never be
    scored as an empty hypothesis (which would post a flattering miss-only DER).
    """
    spans: list[LabelledSpan] = []
    utterances = body.get("utterances") or []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        utt_speaker = utterance.get("speaker")
        words = utterance.get("words") or []
        used_a_word = False
        for word in words:
            if not isinstance(word, dict):
                continue
            speaker = word.get("speaker") or utt_speaker
            if speaker is None or word.get("start") is None or word.get("end") is None:
                continue
            spans.append(
                LabelledSpan(
                    _ms_to_s(word["start"]), _ms_to_s(word["end"]), str(speaker)
                )
            )
            used_a_word = True
        if not used_a_word and utt_speaker is not None:
            if utterance.get("start") is not None and utterance.get("end") is not None:
                spans.append(
                    LabelledSpan(
                        _ms_to_s(utterance["start"]),
                        _ms_to_s(utterance["end"]),
                        str(utt_speaker),
                    )
                )
    if spans:
        return spans

    for word in body.get("words") or []:
        if not isinstance(word, dict):
            continue
        speaker = word.get("speaker")
        if speaker is None or word.get("start") is None or word.get("end") is None:
            continue
        spans.append(
            LabelledSpan(_ms_to_s(word["start"]), _ms_to_s(word["end"]), str(speaker))
        )
    if spans:
        return spans

    raise ValueError(
        "AssemblyAI transcript carries no speaker-labelled words or utterances — "
        "was speaker_labels enabled? Scoring this as an empty hypothesis would "
        "publish a miss-only DER, so the run stops here."
    )


class AssemblyAIDiarizer:
    """Diarizer adapter wrapping AssemblyAI's ``speaker_labels`` transcription."""

    def __init__(
        self,
        provider_id: str = "assemblyai-universal-3-5-pro",
        model_id: str = f"assemblyai/{DEFAULT_MODEL_ALIAS}",
        revision: str | None = None,
        # Keyword-only AND required: the corpus decides the language, and a
        # missing one must be a TypeError at construction, not a 100 %-miss
        # run that looks like a bad model.
        *,
        language: str,
        api_key: str | None = None,
        api_key_env: str = "ASSEMBLYAI_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        gap_merge_s: float = DEFAULT_GAP_MERGE_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        max_poll_s: float = DEFAULT_MAX_POLL_S,
        max_retries: int = 4,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        # ``revision`` IS the model alias here: the vendor offers no other pin.
        # Falls back to the tail of model_id so a spec that only names the model
        # still sends something explicit — never the API's silent fallback chain.
        self.revision = revision or model_id.rsplit("/", 1)[-1]
        self._api_key = _resolve_api_key(api_key, api_key_env)
        self._base_url = base_url.rstrip("/")
        self._language_code = language
        self._gap_merge_s = gap_merge_s
        self._timeout_s = timeout_s
        self._poll_interval_s = poll_interval_s
        self._max_poll_s = max_poll_s
        self._max_retries = max_retries

    # ── HTTP plumbing ────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {"authorization": self._api_key}

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """One request with bounded exponential backoff on transient failures.

        A 120-file sweep costs real money; a single 502 in file 80 must not throw
        away the run. Non-retryable statuses (401, 400, …) raise immediately —
        retrying a bad key just burns time.
        """
        import httpx  # assemblyai extra — lazy so listing adapters imports nothing

        url = f"{self._base_url}{path}"
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(timeout=self._timeout_s) as client:
                    response = client.request(
                        method, url, headers=self._headers(), **kwargs
                    )
                if response.status_code in RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"{response.status_code} from {path}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in RETRYABLE_STATUS:
                    raise
                last_exc = exc
                if attempt == self._max_retries:
                    break
                logger.warning(
                    "AssemblyAI %s %s failed (%s); retry %d/%d in %.1fs",
                    method, path, exc, attempt + 1, self._max_retries, delay,
                )
                time.sleep(delay)
                delay *= 2
        assert last_exc is not None
        raise last_exc

    def _upload(self, audio_path: Path) -> str:
        body = self._request("POST", "/v2/upload", content=audio_path.read_bytes())
        upload_url = body.get("upload_url")
        if not upload_url:
            raise RuntimeError(f"AssemblyAI upload returned no upload_url: {body}")
        return str(upload_url)

    def _submit(self, upload_url: str) -> str:
        payload = {
            "audio_url": upload_url,
            "speaker_labels": True,
            "language_code": self._language_code,
            # Single element => no fallback chain. See the module docstring.
            "speech_models": [self.revision],
        }
        body = self._request("POST", "/v2/transcript", json=payload)
        transcript_id = body.get("id")
        if not transcript_id:
            raise RuntimeError(f"AssemblyAI submit returned no transcript id: {body}")
        return str(transcript_id)

    def _poll(self, transcript_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._max_poll_s
        while True:
            body = self._request("GET", f"/v2/transcript/{transcript_id}")
            status = body.get("status")
            if status == "completed":
                return body
            if status == "error":
                raise RuntimeError(
                    f"AssemblyAI transcript {transcript_id} failed: {body.get('error')}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"AssemblyAI transcript {transcript_id} still {status!r} after "
                    f"{self._max_poll_s:.0f}s — treating as wedged, not slow."
                )
            time.sleep(self._poll_interval_s)

    # ── the protocol ─────────────────────────────────────────────────────────

    def diarize(self, audio_path: Path) -> DiarizeResult:
        started = time.perf_counter()
        upload_url = self._upload(audio_path)
        transcript_id = self._submit(upload_url)
        body = self._poll(transcript_id)
        latency_s = time.perf_counter() - started
        return self.build_result(body, latency_s=latency_s, transcript_id=transcript_id)

    def build_result(
        self,
        body: dict[str, Any],
        *,
        latency_s: float,
        transcript_id: str | None = None,
    ) -> DiarizeResult:
        """Turn a completed transcript body into scoreable turns (pure, testable)."""
        self._assert_model_used(body)
        segments = spans_to_turns(
            labelled_spans(body), gap_merge_s=self._gap_merge_s
        )
        return DiarizeResult(
            segments=segments,
            latency_s=latency_s,
            raw={
                "model_id": self.model_id,
                "revision": self.revision,
                "speech_model_used": body.get("speech_model_used"),
                "language_code": body.get("language_code"),
                "audio_duration_s": body.get("audio_duration"),
                "transcript_id": transcript_id,
                "gap_merge_s": self._gap_merge_s,
            },
        )

    def _assert_model_used(self, body: dict[str, Any]) -> None:
        """Fail the file if the vendor served a model we did not pin.

        ``speech_models`` is a *routing* parameter: AssemblyAI documents that it
        falls back to another model for unsupported languages. A published row
        must name the model that actually produced it, so a mismatch is an error,
        not a warning. A response that omits the field entirely is accepted with a
        warning — older API surfaces do not report it and refusing would make the
        adapter brittle against a field we do not control.
        """
        used = body.get("speech_model_used")
        if used is None:
            logger.warning(
                "AssemblyAI response carries no speech_model_used; the published "
                "row can only claim the requested alias %r.", self.revision,
            )
            return
        if str(used) != self.revision:
            raise RuntimeError(
                f"AssemblyAI served speech_model_used={used!r} but the pin is "
                f"{self.revision!r} — the vendor re-routed the request (language "
                f"support?). Refusing to score a model the row would misname."
            )


#: Registry entry point (see raven_diar.registry) — the runner resolves
#: ``DiarizerSpec.adapter == "assemblyai"`` to this attribute.
ADAPTER = AssemblyAIDiarizer
