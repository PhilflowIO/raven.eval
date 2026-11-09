"""The vLLM shim reports the served model and its serving setup (serving/vllm/shim.py).

Moved from the operator's private deployment repo together with the shim.
raven.eval aborts when `x_raven.model` differs from the model it meant to
measure — the only guard against a forgotten model swap producing numbers
under the wrong name — and records `x_raven.engine/.recipe/.batching` as the
run's serving block. Pinned against the served file, with the vLLM upstream
mocked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

SERVING = Path(__file__).resolve().parents[1] / "serving"
SHIM = SERVING / "vllm" / "shim.py"


def _upstream(served: list[str], calls: list[str], bodies: list[bytes] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if bodies is not None and request.method == "POST":
            bodies.append(request.read())
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in served]})
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.28.0"})
        if request.url.path == "/v1/audio/transcriptions":
            return httpx.Response(200, json={"text": "hallo welt"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200, json={"choices": [{"message": {"content": " hallo "}}], "usage": {"t": 1}}
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture()
def load_shim(monkeypatch):
    def _load(served: list[str]):
        monkeypatch.delenv("SHIM_FAMILY", raising=False)
        monkeypatch.delenv("STT_API_KEY", raising=False)
        monkeypatch.delenv("RAVEN_SERVE_MAX_NUM_SEQS", raising=False)
        spec = importlib.util.spec_from_file_location("raven_serving_shim", SHIM)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["raven_serving_shim"] = mod
        spec.loader.exec_module(mod)
        calls: list[str] = []
        mod._http = httpx.AsyncClient(transport=_upstream(served, calls))
        return mod, calls

    return _load


def _post(mod, model: str):
    return TestClient(mod.app).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"RIFF....", "audio/wav")},
        data={"model": model, "language": "de"},
    )


def test_passthrough_reports_served_model(load_shim):
    mod, _ = load_shim(["Qwen/Qwen3-ASR-1.7B"])
    r = _post(mod, "Qwen/Qwen3-ASR-1.7B")
    assert r.status_code == 200
    assert r.json()["text"] == "hallo welt"
    assert r.json()["x_raven"]["model"] == "Qwen/Qwen3-ASR-1.7B"


def test_reports_what_is_served_not_what_was_asked(load_shim):
    # The point of the field: a stale server must be visible to the client.
    mod, _ = load_shim(["primeline/whisper-large-v3-german"])
    r = _post(mod, "openai/whisper-large-v3")
    assert r.json()["x_raven"]["model"] == "primeline/whisper-large-v3-german"


def test_chat_translate_reports_base_model_not_lora(load_shim):
    # vLLM lists the base model first and LoRA modules after it.
    mod, _ = load_shim(["microsoft/Phi-4-multimodal-instruct", "speech"])
    r = _post(mod, "microsoft/Phi-4-multimodal-instruct")
    assert r.status_code == 200
    assert r.json()["text"] == "hallo"
    assert r.json()["x_raven"]["model"] == "microsoft/Phi-4-multimodal-instruct"


def test_served_model_is_cached_not_fetched_per_request(load_shim):
    mod, calls = load_shim(["Qwen/Qwen3-ASR-1.7B"])
    for _ in range(5):
        assert _post(mod, "Qwen/Qwen3-ASR-1.7B").status_code == 200
    assert calls.count("/v1/models") == 1
    assert calls.count("/version") == 1
    assert calls.count("/v1/audio/transcriptions") == 5


def _recipe(rid: str) -> dict:
    sys.path.insert(0, str(SERVING))
    from recipe import recipe_sha256

    return {"id": rid, "sha256": recipe_sha256(rid)}


def test_reports_engine_recipe_and_batching(load_shim):
    mod, _ = load_shim(["Qwen/Qwen3-ASR-1.7B"])
    x = _post(mod, "Qwen/Qwen3-ASR-1.7B").json()["x_raven"]
    assert x["engine"] == "vllm 0.28.0"
    assert x["recipe"] == _recipe("qwen3-asr-1.7b")
    assert x["batching"] == {"max_num_seqs": 16}


def test_batching_follows_the_launch_override(load_shim, monkeypatch):
    mod, _ = load_shim(["OpenMOSS-Team/MOSS-Transcribe-Diarize"])
    assert _post(mod, "OpenMOSS-Team/MOSS-Transcribe-Diarize").json()["x_raven"]["batching"] == {
        "max_num_seqs": 2  # the recipe default
    }
    mod2, _ = load_shim(["OpenMOSS-Team/MOSS-Transcribe-Diarize"])
    monkeypatch.setenv("RAVEN_SERVE_MAX_NUM_SEQS", "16")
    assert _post(mod2, "OpenMOSS-Team/MOSS-Transcribe-Diarize").json()["x_raven"]["batching"] == {
        "max_num_seqs": 16
    }


def test_the_transformers_stand_in_reports_its_own_engine(load_shim, monkeypatch):
    mod, _ = load_shim(["moonshine-ai/moonshine-streaming-small-de"])
    monkeypatch.setenv("RAVEN_SERVE_MAX_NUM_SEQS", "16")  # does not apply: generate() is serial

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "5.3.0", "engine": "transformers 5.3.0 (torch 2.9.0)"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "moonshine-ai/moonshine-streaming-small-de"}]})
        return httpx.Response(200, json={"text": "hallo"})

    mod._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    x = _post(mod, "moonshine-ai/moonshine-streaming-small-de").json()["x_raven"]
    assert x["engine"] == "transformers 5.3.0 (torch 2.9.0)"
    assert x["recipe"] == _recipe("moonshine-streaming-small-de")
    assert x["batching"] == {"max_num_seqs": 1}


def test_an_unknown_model_is_refused_rather_than_decoded_under_engine_defaults(load_shim):
    """No recipe, no max_tokens: the engine default would decide loop cost again."""
    mod, calls = load_shim(["someone/unlisted-model"])
    r = _post(mod, "someone/unlisted-model")
    assert r.status_code == 503 and "decoding.max_tokens" in r.text
    assert "/v1/audio/transcriptions" not in calls


def test_reports_the_recipe_decoding(load_shim):
    mod, _ = load_shim(["Qwen/Qwen3-ASR-1.7B"])
    x = _post(mod, "Qwen/Qwen3-ASR-1.7B").json()["x_raven"]
    assert x["decoding"] == {"mode": "greedy", "max_tokens": 1024}


def test_reports_the_recipe_precision(load_shim):
    mod, _ = load_shim(["primeline/whisper-large-v3-german"])
    x = _post(mod, "primeline/whisper-large-v3-german").json()["x_raven"]
    # Served in bf16, the format its checkpoint stores (#82; an fp16 cast before).
    assert x["precision"] == {
        "weights": "bf16", "activations": "bf16", "scheme": "none", "tensor_parallel": 1,
        "checkpoint": {"source": "published", "dtype": "bf16"},
    }


def _load_recording(load_shim, served: str):
    mod, calls = load_shim([served])
    bodies: list[bytes] = []
    mod._http = httpx.AsyncClient(transport=_upstream([served], calls, bodies))
    return mod, bodies


def test_passthrough_pins_greedy_and_the_recipe_cap_upstream(load_shim):
    mod, bodies = _load_recording(load_shim, "Qwen/Qwen3-ASR-1.7B")
    assert _post(mod, "Qwen/Qwen3-ASR-1.7B").status_code == 200
    (body,) = bodies
    for name, value in (("temperature", b"0.0"), ("repetition_penalty", b"1.0"),
                        ("max_completion_tokens", b"1024")):
        assert f'name="{name}"'.encode() in body
        assert value in body.split(f'name="{name}"'.encode())[1][:40]


def test_chat_translate_pins_greedy_and_the_recipe_cap_upstream(load_shim):
    import json as _json

    mod, bodies = _load_recording(load_shim, "microsoft/Phi-4-multimodal-instruct")
    assert _post(mod, "microsoft/Phi-4-multimodal-instruct").status_code == 200
    payload = _json.loads(bodies[0])
    assert (payload["temperature"], payload["repetition_penalty"], payload["max_tokens"]) == (
        0.0, 1.0, 1024)


def test_a_sampling_temperature_is_refused(load_shim):
    mod, calls = load_shim(["Qwen/Qwen3-ASR-1.7B"])
    r = TestClient(mod.app).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"RIFF....", "audio/wav")},
        data={"model": "Qwen/Qwen3-ASR-1.7B", "temperature": "0.7"},
    )
    assert r.status_code == 400
    assert "/v1/audio/transcriptions" not in calls


def test_vibevoice_is_told_german_as_its_context_sentence(load_shim):
    """VibeVoice-ASR runs on vLLM's Transformers backend, which has no transcription
    endpoint, and it has no language parameter. Its chat template turns the one text
    part into "extra info" context, the channel Microsoft documents for metadata, so
    the recipe's decoding.language_forced makes language=de that sentence and the
    template writes the instruction around it (raven.eval#104)."""
    import json as _json

    mod, bodies = _load_recording(load_shim, "microsoft/VibeVoice-ASR-HF")
    r = _post(mod, "microsoft/VibeVoice-ASR-HF")
    assert r.status_code == 200
    payload = _json.loads(bodies[0])
    assert payload["model"] == "microsoft/VibeVoice-ASR-HF"
    (message,) = payload["messages"]
    assert [part["type"] for part in message["content"]] == ["text", "input_audio"]
    assert message["content"][0]["text"] == "The audio is in German."
    assert message["content"][1]["input_audio"]["format"] == "wav"
    assert (payload["temperature"], payload["repetition_penalty"], payload["max_tokens"]) == (
        0.0, 1.0, 1024)
    x_raven = r.json()["x_raven"]
    assert x_raven["recipe"] == _recipe("vibevoice-asr")
    assert x_raven["decoding"]["language_forced"] is True


def test_vibevoice_request_prompt_and_language_share_the_one_context_part(load_shim):
    import json as _json

    mod, bodies = _load_recording(load_shim, "microsoft/VibeVoice-ASR-HF")
    r = TestClient(mod.app).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"RIFF....", "audio/wav")},
        data={"model": "microsoft/VibeVoice-ASR-HF", "language": "de",
              "prompt": "About VibeVoice"},
    )
    assert r.status_code == 200
    (message,) = _json.loads(bodies[0])["messages"]
    assert message["content"][0] == {
        "type": "text", "text": "About VibeVoice The audio is in German."}


@pytest.mark.parametrize("language", ["", "xx"])
def test_vibevoice_refuses_a_language_it_cannot_be_told(load_shim, language):
    """The recipe forces the language: a request without one, or with a code that has
    no context sentence, must fail loudly instead of running unforced."""
    mod, bodies = _load_recording(load_shim, "microsoft/VibeVoice-ASR-HF")
    r = TestClient(mod.app).post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"RIFF....", "audio/wav")},
        data={"model": "microsoft/VibeVoice-ASR-HF", "language": language},
    )
    assert r.status_code == 400
    assert not [b for b in bodies if b"chat" in b or b"messages" in b]


def test_vibevoice_without_language_forced_sends_the_audio_alone(load_shim, monkeypatch):
    """Without the recipe flag nothing is added: the template writes its plain
    instruction (the pre-#104 request)."""
    import json as _json

    mod, bodies = _load_recording(load_shim, "microsoft/VibeVoice-ASR-HF")
    monkeypatch.setattr(mod, "recipe_decoding",
                        lambda served: {"mode": "greedy", "max_tokens": 1024})
    r = _post(mod, "microsoft/VibeVoice-ASR-HF")
    assert r.status_code == 200
    (message,) = _json.loads(bodies[0])["messages"]
    assert [part["type"] for part in message["content"]] == ["input_audio"]


@pytest.mark.parametrize("model", [
    "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
    "PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16",
])
def test_nemotron_transcribe_gets_the_cards_benchmark_request(load_shim, model):
    """NeMo's SpeechLM plugin serves Nemotron-3.5-Transcribe as a chat model: the
    card's verbatim ASR instruction plus the audio as an audio_url, reasoning off,
    greedy, the recipe's output cap. "Language: as spoken": language=de is not
    forwarded, as the card names no wording that forces a language. (Built
    directly: the serving recipe lands with the quantized checkpoint, and the
    shim refuses a model without one.)"""
    mod, _ = load_shim([model])
    family = mod._detect_family(model)
    assert family == "nemotron-transcribe"
    payload = mod.CHAT_BUILDERS[family](
        model=model, audio_b64="UklGRg==", audio_format="wav", language="de", language_forced=False,
        user_prompt="", max_tokens=1024,
    )
    assert payload["model"] == model
    (message,) = payload["messages"]
    assert message["role"] == "user"
    text, audio = message["content"]
    assert text == {"type": "text", "text": mod.NEMOTRON_TRANSCRIBE_PROMPT}
    assert "Language: as spoken." in text["text"] and "German" not in text["text"]
    assert audio == {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,UklGRg=="}}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert (payload["temperature"], payload["repetition_penalty"], payload["max_tokens"]) == (
        0.0, 1.0, 1024)


@pytest.mark.parametrize("model", [
    "openai/whisper-large-v3-turbo", "ibm-granite/granite-speech-5.0-2b",
    "mistralai/Voxtral-Mini-4B-Realtime-2602",
    "ibm-granite/granite-speech-5.0-470m-turboctc",
])
def test_the_new_transcription_models_pass_through(load_shim, model):
    mod, calls = load_shim([model])
    mod._frames_after_prompt = lambda *_a: 200  # the engine's audio stack is not installed here
    assert _post(mod, model).status_code == 200
    assert "/v1/audio/transcriptions" in calls and "/v1/chat/completions" not in calls


VOXTRAL_RT = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def _sent_max_tokens(body: bytes) -> bytes:
    return body.split(b'name="max_completion_tokens"')[1][:40].split(b"\r\n")[2]


def test_a_streaming_model_is_stopped_where_its_audio_ends(load_shim):
    """raven.eval#89: past its last audio frame Voxtral Realtime decodes on zero or
    another request's audio, so the shim caps every request at the frames left."""
    mod, bodies = _load_recording(load_shim, VOXTRAL_RT)
    seen: list[tuple] = []

    def frames(audio, model, revision, language):
        seen.append((audio, model, revision, language))
        return 57

    mod._frames_after_prompt = frames
    assert _post(mod, VOXTRAL_RT).status_code == 200
    assert _sent_max_tokens(bodies[0]) == b"57"
    # Computed for the served model at its pinned revision, on the uploaded bytes.
    assert seen == [(b"RIFF....", VOXTRAL_RT, "2769294da9567371363522aac9bbcfdd19447add", "de")]


def test_the_audio_end_never_raises_the_recipe_cap(load_shim):
    mod, bodies = _load_recording(load_shim, VOXTRAL_RT)
    mod._frames_after_prompt = lambda *_a: 5000
    assert _post(mod, VOXTRAL_RT).status_code == 200
    assert _sent_max_tokens(bodies[0]) == b"1024"


def test_audio_too_short_for_the_prompt_is_refused(load_shim):
    mod, bodies = _load_recording(load_shim, VOXTRAL_RT)
    mod._frames_after_prompt = lambda *_a: 0
    r = _post(mod, VOXTRAL_RT)
    assert r.status_code == 400 and "too short" in r.text
    assert bodies == []


def test_models_without_the_flag_do_not_decode_the_audio(load_shim):
    mod, bodies = _load_recording(load_shim, "Qwen/Qwen3-ASR-1.7B")

    def boom(*_a):
        raise AssertionError("only stop_at_audio_end recipes look at the audio")

    mod._frames_after_prompt = boom
    assert _post(mod, "Qwen/Qwen3-ASR-1.7B").status_code == 200
    assert _sent_max_tokens(bodies[0]) == b"1024"


def test_the_flag_is_reported_with_the_decoding(load_shim):
    mod, _ = _load_recording(load_shim, VOXTRAL_RT)
    mod._frames_after_prompt = lambda *_a: 57
    x = _post(mod, VOXTRAL_RT).json()["x_raven"]
    assert x["decoding"] == {"mode": "greedy", "max_tokens": 1024, "stop_at_audio_end": True}
