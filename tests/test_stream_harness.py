"""The capacity harness against a mock realtime server: pacing, timings, metrics, rules."""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest

pytest.importorskip("websockets")

from raven_asr.datasets.base import Sample
from raven_asr.stream.curve import (
    load_offline,
    max_n_within,
    next_action,
    percentile,
    render_table,
    step_summary,
)
from raven_asr.stream.pool import (
    SAMPLE_RATE,
    Utterance,
    build_pool,
    session_playlist,
    to_pcm16,
)
from raven_asr.stream.probes import parse_gauges
from raven_asr.stream.session import UtteranceRecord, run_session


def _utt(i: int, seconds: float) -> Utterance:
    return Utterance(f"u{i}", "fleurs", f"hallo welt {i}",
                     (np.zeros(int(SAMPLE_RATE * seconds))).astype("<i2").tobytes())


async def _mock_server(final_delay_s: float, received: list, fail_after: int | None = None):
    """A realtime server: echoes 'hallo welt' once audio flows, done after a delay."""
    from websockets.asyncio.server import serve

    async def handler(ws):
        await ws.send(json.dumps({"type": "session.created"}))
        utterances = 0
        async for raw in ws:
            ev = json.loads(raw)
            if ev["type"] == "session.update":
                assert ev["model"] == "org/m"
            elif ev["type"] == "input_audio_buffer.commit" and not ev.get("final"):
                utterances += 1
                if fail_after is not None and utterances > fail_after:
                    await ws.send(json.dumps({"type": "error", "error": "boom", "code": "x"}))
                    continue
                sent_text = False
            elif ev["type"] == "input_audio_buffer.append":
                received.append(len(base64.b64decode(ev["audio"])))
                if not sent_text:
                    sent_text = True
                    await ws.send(json.dumps({"type": "transcription.delta", "delta": "hallo "}))
            elif ev["type"] == "input_audio_buffer.commit" and ev.get("final"):
                await asyncio.sleep(final_delay_s)
                await ws.send(json.dumps({"type": "transcription.done", "text": "hallo welt"}))

    return await serve(handler, "127.0.0.1", 0)


def test_session_paces_in_real_time_and_measures_final_latency():
    received: list[int] = []

    async def go():
        server = await _mock_server(0.05, received)
        port = server.sockets[0].getsockname()[1]
        playlist = iter([_utt(0, 0.3), _utt(1, 0.25), _utt(2, 0.3)])
        t = asyncio.get_running_loop().time()
        recs, err = await run_session(f"ws://127.0.0.1:{port}", "org/m", playlist, session=0,
                                      chunk_ms=100, duration_s=0.4)
        wall = asyncio.get_running_loop().time() - t
        server.close()
        return recs, err, wall

    recs, err, wall = asyncio.run(go())
    assert err is None
    assert [r.sample_id for r in recs] == ["u0", "u1"]  # u1 started before 0.4 s; u2 did not
    assert wall >= 0.55  # real time: 0.3 s + 0.25 s of audio, not as fast as possible
    assert received[:3] == [3200, 3200, 3200]  # 100 ms of PCM16 per chunk
    for r in recs:
        assert r.text == "hallo welt"
        assert 0.04 <= r.final_s < 0.5
        assert r.ttfp_s is not None and r.ttfp_s >= 0.09  # the first chunk leaves after 100 ms
        assert r.t_audio_end - r.t_start >= r.duration_s - 0.02


def test_server_error_ends_the_session_as_a_drop():
    async def go():
        server = await _mock_server(0.0, [], fail_after=1)
        port = server.sockets[0].getsockname()[1]
        playlist = iter([_utt(i, 0.1) for i in range(5)])
        out = await run_session(f"ws://127.0.0.1:{port}", "org/m", playlist, session=3,
                                chunk_ms=50, duration_s=5.0)
        server.close()
        return out

    recs, err = asyncio.run(go())
    assert err is not None and "boom" in err
    assert recs[0].error is None and recs[-1].error is not None


def test_unreachable_server_is_a_drop_without_records():
    recs, err = asyncio.run(run_session("ws://127.0.0.1:9", "org/m", iter([_utt(0, 0.1)]),
                                        session=0, chunk_ms=50, duration_s=1.0))
    assert recs == [] and err.startswith("connect")


def _rec(i, final_s, err=None, text="hallo welt"):
    r = UtteranceRecord(0, i, f"s{i}", "fleurs", "Hallo Welt", 2.0, t_start=0.0,
                        t_first_send=0.1, t_audio_end=2.0, t_first_text=0.5,
                        t_done=None if err else 2.0 + final_s, text=None if err else text,
                        error=err)
    return r


def test_step_summary_percentiles_wer_and_rule():
    recs = [_rec(i, 0.1 * (i + 1)) for i in range(10)]
    s = step_summary(4, recs, [None] * 4, {}, {f"s{i}": "hallo welt" for i in range(10)}, 1.0)
    assert s["completed"] == 10 and s["errors"] == 0 and s["drops"] == 0
    assert s["final_s"]["p50"] == pytest.approx(0.55)
    assert s["final_s"]["p95"] == pytest.approx(0.955)
    assert s["wer_strict_pct"] == 0.0 and s["offline_wer_strict_pct"] == 0.0
    assert s["within_threshold"]
    s = step_summary(4, recs + [_rec(99, 0, err="timeout")], [None, "x", None, None], {}, {}, 1.0)
    assert not s["within_threshold"] and s["errors"] == 1 and s["drops"] == 1
    assert s["offline_wer_strict_pct"] is None  # offline covers none of the utterances


def test_step_rule_records_one_step_beyond():
    ok, bad = {"within_threshold": True, "n": 1}, {"within_threshold": False, "n": 2}
    assert next_action([ok]) == "continue"
    assert next_action([ok, bad]) == "one_more"
    assert next_action([ok, bad, dict(bad, n=4)]) == "stop"
    assert max_n_within([dict(ok, n=1), dict(ok, n=2), dict(bad, n=4), dict(ok, n=8)]) == 2
    assert max_n_within([bad]) is None


def test_percentile_matches_numpy():
    xs = [0.3, 1.2, 0.7, 0.1, 0.9, 2.5]
    for q in (50, 95, 99):
        assert percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)))
    assert percentile([], 50) is None


def test_pool_is_seeded_and_sessions_start_apart():
    def loader_for(ds):
        n = 10 if ds == "fleurs" else 30
        return [Sample(np.zeros(800, dtype=np.float32), 16000, "r", f"{ds}-{i}", ds)
                for i in range(n)]

    a, spec_a = build_pool(seed=1, loader_for=loader_for)
    _b, spec_b = build_pool(seed=1, loader_for=loader_for)
    c, _ = build_pool(seed=2, loader_for=loader_for)
    assert spec_a["ids_sha256"] == spec_b["ids_sha256"] != _ids(c)
    assert spec_a["n_utterances"] == len(a) == 40
    first = [next(session_playlist(a, i, 4)).sample_id for i in range(4)]
    assert len(set(first)) == 4


def _ids(pool):
    import hashlib

    return hashlib.sha256("\n".join(u.sample_id for u in pool).encode()).hexdigest()


def test_to_pcm16_resamples_to_16k():
    pytest.importorskip("soxr")
    pcm = to_pcm16(np.zeros(44100, dtype=np.float32), 44100)
    assert len(pcm) == 2 * 16000


def test_parse_vllm_gauges():
    text = ('# HELP x\nvllm:num_requests_running{model_name="m"} 12.0\n'
            'vllm:num_requests_waiting{model_name="m"} 3.0\n'
            'vllm:kv_cache_usage_perc{model_name="m"} 0.42\nother 1\n')
    assert parse_gauges(text) == {"running": 12.0, "waiting": 3.0, "kv_cache_usage": 0.42}


def test_offline_predictions_and_table(tmp_path):
    (tmp_path / "predictions_fleurs.jsonl").write_text(
        json.dumps({"sample_id": "s1", "prediction": "a"}) + "\n"
        + json.dumps({"sample_id": "s2", "prediction": None}) + "\n")
    assert load_offline(tmp_path) == {"s1": "a"}
    step = step_summary(1, [_rec(0, 0.2)], [None], {"util_mean_pct": 40.0,
                                                     "memory_used_max_mib": 9000}, {}, 1.0)
    curve = {"model": "org/m", "latency": "480 ms", "chunk_ms": 80, "hardware_label": "1x GPU",
             "params": {"threshold_s": 1.0}, "steps": [step], "max_n_within_threshold": 1,
             "stopped_because": "every step stepped"}
    table = render_table(curve)
    assert "| 1 | 0.20 | 0.20 |" in table and "**1**" in table


def test_wer_reads_raw_output_through_the_offline_parser():
    """Both sides go through the model's output parser, as offline scoring does."""
    from raven_asr.stream.__main__ import spoken_text

    spoken = spoken_text("nvidia/nemotron-3.5-asr-streaming-0.6b")
    rec = _rec(0, 0.1, text="Hallo. <de-DE> Welt. <de-DE>")
    rec.reference = "Hallo. Welt."
    offline = {"s0": "Hallo. <de-DE> Welt."}
    raw = step_summary(1, [rec], [None], {}, offline, 1.0)
    parsed = step_summary(1, [rec], [None], {}, offline, 1.0, spoken)
    assert raw["wer_strict_pct"] > 0 and raw["offline_wer_strict_pct"] > 0
    assert parsed["wer_strict_pct"] == 0.0 and parsed["offline_wer_strict_pct"] == 0.0
    assert spoken_text("mistralai/Voxtral-Mini-4B-Realtime-2602")("a <de-DE>") == "a <de-DE>"


def test_rescore_recomputes_wer_and_keeps_latencies(tmp_path):
    from raven_asr.stream.__main__ import main

    rec = _rec(0, 0.3, text="Hallo <de-DE> Welt")
    step = step_summary(1, [rec], [None], {}, {}, 1.0)
    curve = {"model": "nvidia/nemotron-3.5-asr-streaming-0.6b", "latency": "l", "chunk_ms": 560,
             "hardware_label": "h", "params": {"threshold_s": 1.0}, "steps": [step],
             "max_n_within_threshold": 1, "stopped_because": "x"}
    (tmp_path / "curve.json").write_text(json.dumps(curve))
    (tmp_path / "utterances_n1.jsonl").write_text(json.dumps(rec.as_json()) + "\n")
    (tmp_path / "off").mkdir()
    (tmp_path / "off" / "predictions_fleurs.jsonl").write_text(
        json.dumps({"sample_id": "s0", "prediction": "Hallo <de-DE> Welt"}) + "\n")
    assert step["wer_strict_pct"] > 0
    assert main(["--rescore", "--out", str(tmp_path), "--offline-predictions",
                 str(tmp_path / "off")]) == 0
    out = json.loads((tmp_path / "curve.json").read_text())
    assert out["steps"][0]["wer_strict_pct"] == 0.0
    assert out["steps"][0]["offline_wer_strict_pct"] == 0.0
    assert out["steps"][0]["final_s"] == step["final_s"]
    assert out["output_format"] == "nemotron-locale-tags"
