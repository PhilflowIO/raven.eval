"""The capacity harness against a mock realtime server: pacing, timings, metrics, rules."""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest

pytest.importorskip("websockets")

from raven_asr.capacity import max_n_within, next_action, percentile
from raven_asr.datasets.base import Sample
from raven_asr.probes import (
    group_cpu_pct,
    parse_gauges,
    parse_pid_stat,
    parse_proc_stat,
)
from raven_asr.stream.__main__ import start_delays
from raven_asr.stream.curve import (
    load_offline,
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


def test_proc_stat_busy_and_total():
    busy, total = parse_proc_stat("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3 4\n")
    assert (busy, total) == (150.0, 1000.0)


def test_cpu_curve_table_shows_cpu_columns_and_precision():
    step = step_summary(1, [_rec(0, 0.2)], [None], {"cpu_util_mean_pct": 37.0,
                                                     "server_rss_max_mib": 812.0}, {}, 1.0)
    curve = {"model": "m", "latency": "l", "chunk_ms": 100, "hardware_label": "CPU, 24 cores",
             "device": "cpu", "precision": {"weights": "int8"}, "params": {"threshold_s": 1.0},
             "steps": [step], "max_n_within_threshold": 1, "stopped_because": "x"}
    table = render_table(curve)
    assert "CPU util mean %" in table and "| 37 | 812 |" in table
    assert "CPU, weights int8" in table


# ── per-process CPU and the multi-process client (#119) ─────────────────────


def test_pid_stat_reads_utime_and_stime_after_the_command_name():
    # pid (comm with ") " inside) state ... utime=14th, stime=15th field overall
    line = "42 (python (x) y) S 1 42 42 0 -1 4194304 10 0 0 0 250 50 0 0 20 0 3 0\n"
    assert parse_pid_stat(line, 100.0) == pytest.approx(3.0)
    assert parse_pid_stat("garbage", 100.0) is None


def test_group_cpu_counts_processes_seen_at_both_ends_in_one_core_units():
    prev = {"1/task/1": 10.0, "1/task/2": 5.0, "1/task/3": 1.0}
    cur = {"1/task/1": 10.5, "1/task/2": 5.25, "1/task/4": 9.0}  # 3 ended, 4 is new: neither counts
    assert group_cpu_pct(prev, cur, 0.5) == (150.0, 100.0)
    assert group_cpu_pct({}, cur, 0.5) is None


def test_start_delays_are_seeded_per_step_and_within_one_chunk():
    a = start_delays(8, 160, 7)
    assert a == start_delays(8, 160, 7) and a != start_delays(16, 160, 7)[:8]
    assert all(0 <= d <= 0.16 for d in a)


def test_client_pool_splits_sessions_over_processes_and_keeps_the_order():
    import threading

    from raven_asr.stream.__main__ import ClientPool

    pool = [_utt(i, 0.2) for i in range(6)]
    clients = ClientPool(pool, 2)  # fork before the server thread exists
    ready, holder = threading.Event(), {}

    def serve() -> None:
        loop = asyncio.new_event_loop()
        holder["server"] = loop.run_until_complete(_mock_server(0.0, []))
        holder["loop"] = loop
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    ready.wait(5)
    port = holder["server"].sockets[0].getsockname()[1]
    try:
        assert len(set(clients.pids())) == 3  # this process and two workers
        records, errors = clients.run_step(f"ws://127.0.0.1:{port}", "org/m", 3, chunk_ms=100,
                                           duration_s=0.1, done_timeout_s=5.0, seed=1)
    finally:
        clients.close()
        holder["loop"].call_soon_threadsafe(holder["loop"].stop)
    assert errors == [None, None, None]
    assert sorted({r.session for r in records}) == [0, 1, 2]
    by_session = {r.session: r.sample_id for r in records if r.index == 0}
    assert by_session == {i: next(session_playlist(pool, i, 3)).sample_id for i in range(3)}


def test_table_shows_the_busiest_client_and_server_process_when_recorded():
    step = step_summary(1, [_rec(0, 0.2)], [None], {
        "util_mean_pct": 40.0, "memory_used_max_mib": 9000,
        "cpu_client_thread_max_pct": 97.4, "cpu_server_thread_max_pct": 61.0}, {}, 1.0)
    curve = {"model": "org/m", "latency": "l", "chunk_ms": 80, "hardware_label": "1x GPU",
             "params": {"threshold_s": 1.0}, "steps": [step], "max_n_within_threshold": 1,
             "stopped_because": "x"}
    table = render_table(curve)
    assert "client CPU busiest thread % | server CPU busiest thread % |" in table
    assert "| 40 | 9000 | 97 | 61 |" in table
    header, sep = table.splitlines()[4:6]
    assert header.count("|") == sep.count("|")


def test_an_out_of_memory_error_event_is_an_error_not_an_empty_transcript():
    """#124: the NeMo server's error event for a failed step, as the harness reads it."""
    async def go():
        from websockets.asyncio.server import serve

        async def handler(ws):
            await ws.send(json.dumps({"type": "session.created"}))
            async for raw in ws:
                if json.loads(raw).get("final"):
                    await ws.send(json.dumps({
                        "type": "error", "code": "out_of_memory",
                        "error": "pipeline step of 512 streams failed: the GPU ran out of memory"}))

        server = await serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        out = await run_session(f"ws://127.0.0.1:{port}", "org/m", iter([_utt(0, 0.1)]),
                                session=0, chunk_ms=50, duration_s=1.0)
        server.close()
        return out

    recs, err = asyncio.run(go())
    assert "out_of_memory" in err
    assert recs[0].error.startswith("server error: out_of_memory") and recs[0].text is None
    step = step_summary(1, recs, [err], {}, {"u0": "hallo welt 0"}, 1.0)
    assert step["errors"] == 1 and step["drops"] == 1 and step["completed"] == 0
    assert not step["within_threshold"]


def test_a_step_the_server_reported_failing_breaks_the_rule_and_is_on_the_table():
    from raven_asr.stream.__main__ import apply_health

    step = step_summary(512, [_rec(0, -7.0, text="")], [None], {}, {}, 1.0)
    assert step["within_threshold"]  # an empty "done" before the audio ended looks perfect
    apply_health(step, {"ok": False, "error": "pipeline step failed: CUDA out of memory"})
    assert not step["within_threshold"] and "out of memory" in step["server_error"]
    assert apply_health({"within_threshold": True}, {"ok": True, "error": None}) == \
        {"within_threshold": True}
    assert apply_health({"within_threshold": True}, None) == {"within_threshold": True}
    curve = {"model": "m", "latency": "l", "chunk_ms": 560, "hardware_label": "1x GPU",
             "params": {"threshold_s": 1.0}, "steps": [step], "max_n_within_threshold": None,
             "stopped_because": "x"}
    assert "N = 512: the server reported `pipeline step failed" in render_table(curve)
    assert max_n_within([step]) is None


def test_threads_lists_this_process_and_its_threads():
    import os
    import threading

    from raven_asr.probes import proc_cpu_s, threads

    stop = threading.Event()
    t = threading.Thread(target=lambda: (sum(range(3_000_000)), stop.wait()))
    t.start()
    try:
        keys = threads(os.getpid())
        assert f"{os.getpid()}/task/{os.getpid()}" in keys and len(keys) >= 2
        # per thread, not the main thread's /proc/<pid>/stat (the whole process)
        assert sum(proc_cpu_s(k) or 0.0 for k in keys) <= proc_cpu_s(str(os.getpid())) + 0.05
    finally:
        stop.set()
        t.join()
    assert threads(2**31 - 1) == []


def test_sampler_measures_thread_cpu_over_whole_intervals_only():
    import os
    import time

    from raven_asr.probes import Sampler, threads

    with Sampler(interval_s=0.05, processes={"me": lambda: threads(os.getpid())}) as s:
        end = time.monotonic() + 0.4
        while time.monotonic() < end:  # one busy thread: at most ~100 % of one core
            pass
    vals = s.proc_cpu["me"]
    assert vals and all(busiest <= 150.0 for _, busiest in vals)


# ── first words after speech onset (#130) ───────────────────────────────────


def test_speech_onset_is_the_leading_silence_of_the_audio():
    from raven_asr.stream.pool import speech_onset_s

    t = np.arange(int(SAMPLE_RATE * 1.0)) / SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    hiss = 0.0005 * np.random.default_rng(0).standard_normal(int(SAMPLE_RATE * 1.5))
    audio = np.concatenate([hiss, tone, hiss[:8000]])
    assert speech_onset_s(to_pcm16(audio, SAMPLE_RATE)) == pytest.approx(1.5, abs=0.021)
    assert speech_onset_s(to_pcm16(tone, SAMPLE_RATE)) == 0.0
    assert speech_onset_s(to_pcm16(np.zeros(SAMPLE_RATE), SAMPLE_RATE)) == 0.0
    assert speech_onset_s(b"") == 0.0


def test_first_words_are_counted_from_speech_onset_not_utterance_start():
    r = UtteranceRecord(0, 0, "s", "fleurs", "x", 10.0, t_start=100.0, t_first_text=103.6,
                        t_done=110.2, t_audio_end=110.0, text="x", speech_onset_s=2.5)
    assert r.ttfp_s == pytest.approx(3.6)
    assert r.first_text_after_speech_s == pytest.approx(1.1)
    assert r.as_json()["first_text_after_speech_s"] == pytest.approx(1.1)
    legacy = UtteranceRecord(0, 0, "s", "fleurs", "x", 10.0, t_start=0.0, t_first_text=1.0)
    assert legacy.first_text_after_speech_s is None  # curves recorded before #130


def test_step_and_table_report_first_words_after_speech():
    recs = []
    for i in range(10):
        r = _rec(i, 0.1)
        r.speech_onset_s = 0.2  # _rec: t_start 0, first text at 0.5
        recs.append(r)
    step = step_summary(1, recs, [None], {"util_mean_pct": 10.0, "memory_used_max_mib": 1}, {}, 1.0)
    assert step["first_text_after_speech_s"]["p95"] == pytest.approx(0.3)
    assert step["ttfp_s"]["p95"] == pytest.approx(0.5)
    curve = {"model": "m", "latency": "l", "chunk_ms": 80, "hardware_label": "1x GPU",
             "params": {"threshold_s": 1.0}, "steps": [step], "max_n_within_threshold": 1,
             "stopped_because": "x"}
    table = render_table(curve)
    assert "p95 first words after speech s" in table and "| 0.50 | 0.30 | 0.30 |" in table
    header, sep = table.splitlines()[4:6]
    assert header.count("|") == sep.count("|")


def test_pool_records_each_utterances_onset():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    audio = np.concatenate([np.zeros(SAMPLE_RATE // 2), 0.3 * np.sin(2 * np.pi * 200 * t)])

    def loader(_):
        return [Sample(audio=audio, sample_rate=SAMPLE_RATE, reference="hallo", sample_id="a1",
                       subset="fleurs")]

    pool, _ = build_pool(sources=(("fleurs", None),), loader_for=loader)
    assert pool[0].onset_s == pytest.approx(0.5, abs=0.021)
