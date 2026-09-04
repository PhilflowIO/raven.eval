"""AssemblyAI diarizer adapter — offline tests (no network, no key, no GPU).

Three things are guarded, in descending order of how much a published number
depends on them:

1. **The adapter folds through the SHARED aggregator.** A second folding path
   inside a hosted adapter would make a DER *difference* between two providers
   partly a difference of two folding rules. The test does not merely check the
   output shape — it asserts the shared helper is the code that produced it.
2. **The model is pinned and the pin is enforced.** AssemblyAI's default is a
   two-model fallback chain, so an unpinned run could publish a number from
   either model. The adapter must send exactly one alias and reject a response
   that names a different one.
3. **A transcript without speaker labels fails loudly.** Scoring it as an empty
   hypothesis would post a flattering miss-only DER.
"""

from __future__ import annotations

import pytest

from raven_diar.adapters import assemblyai as aai
from raven_diar.adapters.aggregate import DEFAULT_GAP_MERGE_S
from raven_diar.config import KNOWN_DIARIZERS
from raven_diar.registry import DIARIZER_ADAPTERS

API_KEY_ENV = "ASSEMBLYAI_API_KEY"


@pytest.fixture
def diarizer(monkeypatch) -> aai.AssemblyAIDiarizer:
    monkeypatch.setenv(API_KEY_ENV, "test-key-not-a-real-secret")
    spec = KNOWN_DIARIZERS["assemblyai-universal-3-5-pro"]
    return aai.AssemblyAIDiarizer(
        provider_id=spec.label, model_id=spec.model_id, revision=spec.revision
    )


def _completed_body(**overrides) -> dict:
    """A minimal completed transcript in the documented diarized shape."""
    body = {
        "status": "completed",
        "speech_model_used": "universal-3-5-pro",
        "language_code": "de",
        "audio_duration": 4,
        "utterances": [
            {
                "speaker": "A",
                "start": 0,
                "end": 900,
                "words": [
                    {"text": "guten", "speaker": "A", "start": 0, "end": 400},
                    {"text": "tag", "speaker": "A", "start": 450, "end": 900},
                ],
            },
            {
                "speaker": "B",
                "start": 2000,
                "end": 3000,
                "words": [
                    {"text": "hallo", "speaker": "B", "start": 2000, "end": 3000},
                ],
            },
        ],
    }
    body.update(overrides)
    return body


# ── registration (the seam: module + spec, no dispatcher edit) ───────────────


def test_the_spec_is_registered_and_resolves_without_an_api_key():
    """Resolving must not need the key — listing adapters is a directory walk."""
    spec = KNOWN_DIARIZERS["assemblyai-universal-3-5-pro"]
    assert spec.adapter == "assemblyai"
    assert DIARIZER_ADAPTERS.resolve(spec.adapter) is aai.AssemblyAIDiarizer
    assert "assemblyai" in DIARIZER_ADAPTERS.available()


def test_the_spec_pins_a_single_model_alias():
    spec = KNOWN_DIARIZERS["assemblyai-universal-3-5-pro"]
    assert spec.revision == "universal-3-5-pro"
    assert spec.model_id.endswith(spec.revision)


def test_a_missing_api_key_fails_with_the_env_var_name(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        aai.AssemblyAIDiarizer()


# ── span extraction (ms → s, the documented shapes) ──────────────────────────


def test_word_level_spans_are_read_from_utterances_in_seconds():
    spans = aai.labelled_spans(_completed_body())
    assert [(s.start, s.end, s.speaker) for s in spans] == [
        (0.0, 0.4, "A"),
        (0.45, 0.9, "A"),
        (2.0, 3.0, "B"),
    ]


def test_a_word_without_its_own_label_inherits_the_utterance_speaker():
    body = _completed_body(
        utterances=[
            {
                "speaker": "A",
                "start": 0,
                "end": 500,
                "words": [{"text": "ja", "start": 0, "end": 500}],
            }
        ]
    )
    assert [s.speaker for s in aai.labelled_spans(body)] == ["A"]


def test_top_level_words_are_the_fallback_when_utterances_are_absent():
    body = _completed_body(
        utterances=[],
        words=[
            {"text": "so", "speaker": "A", "start": 100, "end": 600},
            {"text": "unlabelled", "start": 700, "end": 900},  # dropped: no speaker
        ],
    )
    assert [(s.start, s.end, s.speaker) for s in aai.labelled_spans(body)] == [
        (0.1, 0.6, "A")
    ]


def test_an_utterance_without_words_still_contributes_its_own_span():
    body = _completed_body(
        utterances=[{"speaker": "A", "start": 1000, "end": 2500, "words": []}]
    )
    assert [(s.start, s.end, s.speaker) for s in aai.labelled_spans(body)] == [
        (1.0, 2.5, "A")
    ]


def test_a_transcript_without_any_speaker_labels_raises():
    body = _completed_body(utterances=[], words=[])
    with pytest.raises(ValueError, match="no speaker-labelled"):
        aai.labelled_spans(body)


# ── the folding contract: the SHARED aggregator, not a private one ───────────


def test_the_adapter_folds_through_the_shared_aggregator(diarizer, monkeypatch):
    """Not "the turns look right" — "the shared helper is what produced them"."""
    calls: list[dict] = []
    real = aai.spans_to_turns

    def _spy(spans, *, gap_merge_s):
        calls.append({"n_spans": len(list(spans)), "gap_merge_s": gap_merge_s})
        return real(spans, gap_merge_s=gap_merge_s)

    monkeypatch.setattr(aai, "spans_to_turns", _spy)
    diarizer.build_result(_completed_body(), latency_s=0.0)

    assert len(calls) == 1, "a hosted adapter must fold exactly once, and shared"
    assert calls[0]["gap_merge_s"] == DEFAULT_GAP_MERGE_S


def test_words_merge_into_turns_at_the_shared_threshold(diarizer):
    result = diarizer.build_result(_completed_body(), latency_s=1.5)
    # The two "A" words are 0.05 s apart → one turn; "B" is far away → its own.
    assert result.segments == [(0.0, 0.9, "A"), (2.0, 3.0, "B")]
    assert result.latency_s == 1.5
    assert result.raw["gap_merge_s"] == DEFAULT_GAP_MERGE_S


def test_the_result_feeds_the_untouched_scorer(diarizer):
    """No change to DiarizeResult / score.py — prove it end-to-end."""
    from raven_diar.score import score_segment_pairs
    from raven_eval_core.der import to_rttm

    result = diarizer.build_result(_completed_body(), latency_s=0.0)
    assert to_rttm(result.segments, file_id="f1").count("SPEAKER") == 2
    gold = [(0.0, 0.9, "spk1"), (2.0, 3.0, "spk2")]
    assert score_segment_pairs("demo", [(gold, result.segments)]).der_full == 0.0


# ── the pin is enforced, not decorative ──────────────────────────────────────


def test_a_rerouted_model_fails_the_file(diarizer):
    body = _completed_body(speech_model_used="universal-2")
    with pytest.raises(RuntimeError, match="universal-2"):
        diarizer.build_result(body, latency_s=0.0)


def test_a_response_without_the_field_is_accepted_with_a_warning(diarizer, caplog):
    body = _completed_body()
    body.pop("speech_model_used")
    with caplog.at_level("WARNING"):
        result = diarizer.build_result(body, latency_s=0.0)
    assert result.raw["speech_model_used"] is None
    assert "speech_model_used" in caplog.text


def test_the_submit_payload_pins_one_model_and_enables_diarization(
    diarizer, monkeypatch
):
    """The request body is where the pin actually happens — assert it verbatim."""
    seen: dict = {}

    def _fake_request(method, path, **kwargs):
        seen["method"], seen["path"] = method, path
        seen["json"] = kwargs.get("json")
        return {"id": "t-123"}

    monkeypatch.setattr(diarizer, "_request", _fake_request)
    assert diarizer._submit("https://cdn.assemblyai.com/upload/abc") == "t-123"
    assert seen["method"] == "POST" and seen["path"] == "/v2/transcript"
    assert seen["json"] == {
        "audio_url": "https://cdn.assemblyai.com/upload/abc",
        "speaker_labels": True,
        "language_code": "de",
        "speech_models": ["universal-3-5-pro"],  # ONE element: no fallback chain
    }


def test_an_errored_transcript_raises_with_the_vendor_message(diarizer, monkeypatch):
    monkeypatch.setattr(
        diarizer, "_request",
        lambda *a, **k: {"status": "error", "error": "audio too short"},
    )
    with pytest.raises(RuntimeError, match="audio too short"):
        diarizer._poll("t-123")


def test_importing_the_adapter_pulls_no_http_client():
    """Lazy httpx: listing/resolving adapters must work without the extra."""
    import subprocess
    import sys
    from pathlib import Path

    probe = (
        "import sys, raven_diar.adapters.assemblyai;"
        "print('httpx' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False", out.stdout
