"""The coupled live harness: endpoint cut, batch upload at the endpoint, step summary."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("websockets")

from raven_asr.stream import coupled
from raven_asr.stream.pool import SAMPLE_RATE, Utterance, speech_offset_s
from raven_asr.stream.session import UtteranceRecord


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return 0.3 * np.sin(2 * np.pi * 220 * t)


def _pcm(*parts: np.ndarray) -> bytes:
    x = np.concatenate(parts)
    return (x * 32767).round().astype("<i2").tobytes()


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds))


def test_speech_offset_is_where_the_last_loud_frame_ends():
    pcm = _pcm(_silence(0.5), _tone(1.0), _silence(2.0))
    assert speech_offset_s(pcm) == pytest.approx(1.5, abs=0.021)
    assert speech_offset_s(_pcm(_silence(1.0))) == pytest.approx(1.0)


def test_endpoint_cuts_long_trailing_silence_and_pads_short_one():
    long_tail = Utterance("a", "fleurs", "x", _pcm(_tone(1.0), _silence(3.0)))
    cut, end = coupled.endpointed(long_tail, 0.7)
    assert end == pytest.approx(1.0, abs=0.021)
    assert cut.duration_s == pytest.approx(end + 0.7, abs=1e-3)
    short_tail = Utterance("b", "fleurs", "x", _pcm(_tone(1.0), _silence(0.1)))
    cut, end = coupled.endpointed(short_tail, 0.7)
    assert cut.duration_s == pytest.approx(end + 0.7, abs=1e-3)
    assert cut.pcm16[-200:] == b"\x00" * 200


async def _realtime_server(received: list):
    from websockets.asyncio.server import serve

    async def handler(ws):
        await ws.send(json.dumps({"type": "session.created"}))
        async for raw in ws:
            ev = json.loads(raw)
            if ev["type"] == "input_audio_buffer.append":
                received.append(ev["audio"])
            elif ev["type"] == "input_audio_buffer.commit" and ev.get("final"):
                await ws.send(json.dumps({"type": "transcription.done", "text": "hallo welt"}))

    return await serve(handler, "127.0.0.1", 0)


def test_batch_final_leaves_at_the_endpoint_with_the_same_audio(monkeypatch):
    import httpx

    seen: list[tuple[float, int, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        seen.append((0.0, len(body), dict(request.headers)))
        return httpx.Response(200, json={"text": "hallo welt"})

    utt = Utterance("u0", "fleurs", "hallo welt", _pcm(_tone(0.4), _silence(1.0)))

    async def go():
        server = await _realtime_server([])
        port = server.sockets[0].getsockname()[1]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            recs, finals, err = await coupled.coupled_session(
                f"ws://127.0.0.1:{port}", "org/m", [utt], 0, 1, chunk_ms=100, duration_s=0.01,
                start_delay_s=0.0, done_timeout_s=5.0, pause_s=0.3,
                final_url="http://final/transcribe", client=client, final_timeout_s=5.0)
        server.close()
        return recs, finals, err

    recs, finals, err = asyncio.run(go())
    assert err is None and len(recs) == 1 and len(finals) == 1
    f, r = finals[0], recs[0]
    assert f.status == "ok" and f.text == "hallo welt"
    assert f.t_due - f.t_speech_end == pytest.approx(0.3, abs=0.03)
    assert f.t_sent >= f.t_due - 1e-3  # not before the speaker paused
    assert f.t_sent - f.t_due < 0.1
    # the batch model gets exactly the audio the stream model got, as WAV
    assert seen and seen[0][1] > int((0.4 + 0.3) * SAMPLE_RATE) * 2
    assert r.duration_s == pytest.approx(f.audio_s)


def _rec(i, t_done, text="hallo welt", err=None):
    return UtteranceRecord(0, i, f"u{i}", "fleurs", "hallo welt", 2.0, t_start=10.0 * i,
                           t_first_send=10.0 * i + 0.1, t_audio_end=10.0 * i + 2.0,
                           t_done=t_done, text=text, error=err)


def _fin(i, after, status="ok", text="hallo welt"):
    f = coupled.FinalRecord(0, i, f"u{i}", 2.0, t_speech_end=10.0 * i + 1.3,
                            t_due=10.0 * i + 2.0, t_sent=10.0 * i + 2.0,
                            t_final_done=10.0 * i + 1.3 + after, status=status, text=text)
    return f


def test_summary_counts_empty_answers_as_errors_and_applies_the_rule():
    recs = [_rec(i, 10.0 * i + 2.05) for i in range(4)]
    finals = [_fin(i, 0.8) for i in range(4)]
    s = coupled.coupled_summary(4, recs, finals, [None], {}, 1.0, lambda t: t, True)
    assert s["within_threshold"] is True
    assert s["final"]["after_speech_s"]["p95"] == pytest.approx(0.8)
    assert s["stream"]["after_speech_s"]["p50"] == pytest.approx(0.75)
    assert s["paired"]["stream_vs_final_wer_pct"] == 0.0

    finals[2] = _fin(2, 0.5, status="empty", text="")
    s = coupled.coupled_summary(4, recs, finals, [None], {}, 1.0, lambda t: t, True)
    assert s["final"]["errors"] == 1 and s["within_threshold"] is False

    recs[1] = _rec(1, 12.05, text="")
    s = coupled.coupled_summary(4, recs, [_fin(i, 0.8) for i in range(4)], [None], {}, 1.0,
                                lambda t: t, False)
    assert s["stream"]["empty"] == 1 and s["within_threshold"] is False


def test_summary_rule_fails_on_a_slow_p95():
    recs = [_rec(i, 10.0 * i + 2.05) for i in range(20)]
    finals = [_fin(i, 0.8 if i < 17 else 3.0) for i in range(20)]
    s = coupled.coupled_summary(20, recs, finals, [None], {}, 1.0, lambda t: t, True)
    assert s["final"]["after_speech_s"]["p95"] > 1.0 and s["within_threshold"] is False
    table = coupled.render_table({
        "model": "org/m", "case": "batch busy", "latency": "x", "hardware_label": "card",
        "final_url": "http://x", "params": {"endpoint_pause_s": 0.7, "threshold_s": 1.0},
        "steps": [s]})
    assert "| 20 | 20 |" in table


def test_the_slots128_variant_differs_from_the_published_recipe_in_slots_only():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))
    import streaming

    streams, _ = streaming.load_streams()
    s = streams["nvidia-nemotron-3.5-asr-streaming-0.6b-slots128"]
    pub = streams["nvidia-nemotron-3.5-asr-streaming-0.6b"]
    assert s["num_slots"] == 128 and s["batch_size"] <= s["num_slots"]
    keep = ("num_slots", "latency", "notes")
    assert {k: v for k, v in s.items() if k not in keep} == \
        {k: v for k, v in pub.items() if k not in keep}
