"""The offline load test's pure logic: VAD blocks, windows, stitching, loops, the stop rule."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from raven_asr.capacity import max_n_within, next_action
from raven_asr.load import score as sc
from raven_asr.load import vad
from raven_asr.load.client import (
    ASK_JSON,
    ASK_WORDS,
    Outcome,
    answer_budget_s,
    probe_timestamps,
)
from raven_asr.load.sweep import run_items, slice_size, step_summary
from raven_asr.load.timing import onset_offsets, parse_vtt
from raven_asr.load.workload import Item, wav_header

from raven_asr.adapters.base import parse_word_timestamps

# ── VAD port (flow.raven api/app/providers/_vad.py) ──────────────────────────


def test_parse_silence_log_closes_a_trailing_silence_at_the_end():
    log = ("[silencedetect] silence_start: 1.5\n[silencedetect] silence_end: 3.0 | d: 1.5\n"
           "[silencedetect] silence_start: 8.25\n")
    assert vad.parse_silence_log(log, 10.0) == [(1.5, 3.0), (8.25, 10.0)]


def test_islands_are_padded_before_they_are_merged():
    # Speech 0-1.0, 1.9-5.0 (silence 1.0-1.9 is 0.9 s long). Padded by 0.35 s the
    # islands touch (1.35 vs 1.55: gap 0.2 <= 0.5) and merge into one.
    islands = vad.build_speech_islands([(1.0, 1.9), (5.0, 10.0)], 10.0)
    assert islands == [(0.0, 5.35)]


def test_islands_shorter_than_the_minimum_are_dropped():
    # A 0.1 s blip padded to 0.8 s survives; production keeps islands >= 0.25 s
    # after padding, so only a zero-length speech gap disappears.
    islands = vad.build_speech_islands([(0.0, 4.0), (4.1, 10.0)], 10.0, pad=0.0)
    assert islands == []


def test_blocks_hold_at_most_30_s_of_speech_and_never_cut_an_island():
    islands = [(0.0, 10.0), (12.0, 22.0), (25.0, 35.0), (40.0, 85.0), (90.0, 95.0)]
    blocks = vad.plan_blocks(islands)
    assert blocks == [[(0.0, 10.0), (12.0, 22.0), (25.0, 35.0)], [(40.0, 85.0)], [(90.0, 95.0)]]


def test_block_pieces_are_derived_from_the_samples_copied():
    ranges, pieces = vad.block_pieces([(1.0, 2.5), (4.0, 4.5)], total_frames=16000 * 10)
    assert ranges == ((16000, 40000), (64000, 72000))
    assert pieces == ((0.0, 1.5, 1.0), (1.5, 2.0, 4.0))


def test_map_block_time_resolves_a_splice_by_start_or_end():
    pieces = ((0.0, 1.5, 1.0), (1.5, 2.0, 4.0))
    assert vad.map_block_time(pieces, 1.5) == 4.0          # a start opens the next island
    assert vad.map_block_time(pieces, 1.5, is_end=True) == 2.5  # an end finishes this one
    assert vad.map_block_time(pieces, 1.75) == 4.25


def test_remap_word_keeps_a_straddling_word_inside_one_island():
    pieces = ((0.0, 1.5, 1.0), (1.5, 2.0, 4.0))
    # 1.2-1.9 s straddles the splice; most of it lies in the second island.
    s, e = vad.remap_word(pieces, 1.2, 1.9)
    assert (s, e) == (4.0, 4.4)
    assert vad.remap_word(pieces, 0.2, 0.6) == (1.2, 1.6)


def test_window_cuts_take_the_silence_closest_to_each_600_s_boundary():
    silences = [(530.0, 532.0), (590.0, 592.0), (1300.0, 1302.0)]
    # 591 is closest to 600; the next target is 591 + 600 = 1191, nothing within
    # ±120 s of it (1301 is 110 s away: inside) -> 1301.
    assert vad.pick_split_points(2000.0, silences) == [591.0, 1301.0]
    # No silence near a boundary: a hard cut at the target.
    assert vad.pick_split_points(1500.0, []) == [600.0, 1200.0]
    # A remainder within the leeway (≤ 120 s) rides with the last window.
    assert vad.windows(1300.0, []) == [(0.0, 600.0), (600.0, 1300.0)]


def test_production_path_follows_the_density_and_length_guards():
    assert vad.production_path(600.0, [(0.0, 500.0)]).startswith("whole-file (dense")
    assert vad.production_path(30.0, [(0.0, 10.0)]).startswith("whole-file (short")
    assert vad.production_path(600.0, [(0.0, 100.0)]) == "vad"


def test_long_islands_are_split_in_the_second_half_of_the_ceiling():
    # 70 s island, ceiling 30: cuts searched in [15, 30], then [cut+15, cut+30].
    pieces = vad.split_long_islands([(0.0, 70.0)], max_seconds=30.0, cut_at=lambda lo, hi: lo + 1)
    assert pieces == [(0.0, 16.0), (16.0, 32.0), (32.0, 48.0), (48.0, 70.0)]
    # A cut_at answer outside the window is clamped into it.
    assert vad.split_long_islands([(0.0, 40.0)], max_seconds=30.0, cut_at=lambda lo, hi: 99.0) == [
        (0.0, 30.0), (30.0, 40.0)]


def test_quietest_instant_finds_the_pause():
    import numpy as np

    pcm = np.full(16000 * 4, 8000, dtype=np.int16)
    pcm[int(2.5 * 16000):int(2.6 * 16000)] = 0  # a 100 ms breath at 2.5 s
    t = vad.quietest_instant(pcm, 2.0, 3.0)
    assert 2.5 <= t <= 2.6


def test_vad_blocks_carry_their_own_time_map():
    islands = [(0.0, 20.0), (30.0, 45.0)]
    blocks = vad.vad_blocks(islands, total_frames=16000 * 60)
    assert len(blocks) == 2
    assert blocks[1].source_start == 30.0 and blocks[1].audio_s == 15.0
    assert vad.Block.from_json(json.loads(json.dumps(blocks[1].as_json()))) == blocks[1]


def test_wav_header_is_a_valid_pcm16_header():
    import io
    import wave

    raw = wav_header(4) + b"\x00\x01" * 4
    with wave.open(io.BytesIO(raw)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (
            1, 2, 16000, 4)


# ── stitching and long-form scoring ──────────────────────────────────────────


def test_answers_are_stitched_in_time_order_and_a_failed_block_is_deleted():
    ref = "eins zwei drei vier fünf sechs sieben acht"
    answers = [
        sc.Answer(1, 10.0, 20.0, "fünf sechs"),
        sc.Answer(0, 0.0, 10.0, "eins zwei drei vier"),
        sc.Answer(2, 20.0, 30.0, None),  # failed after its retry
    ]
    s = sc.score_recording(ref, answers, window_words=4)
    assert s["deletions"] == 2 and s["insertions"] == 0 and s["substitutions"] == 0
    assert s["failed_answers"] == 1
    # Windows partition the edits.
    assert sum(w["errors"] for w in s["windows"]) == s["errors"]
    assert [w["ref_words"] for w in s["windows"]] == [4, 4]
    assert s["windows"][1]["t_start"] == 10.0


def test_windows_flag_runaway_and_dropout():
    ref = " ".join(["wort"] * 8)
    hyp_loop = " ".join(["wort"] * 4 + ["sehr"] * 12)
    s = sc.score_recording(ref, [sc.Answer(0, 0.0, 10.0, hyp_loop)], window_words=8)
    assert s["runaway_windows"] == 1
    assert s["loop_count"] == 1 and s["loops"][0]["gram"] == "sehr"
    s2 = sc.score_recording(ref, [sc.Answer(0, 0.0, 10.0, "wort wort")], window_words=8)
    assert s2["dropout_windows"] == 1


def test_text_with_no_reference_under_it_is_a_runaway_answer():
    ref = "guten morgen meine damen und herren"
    answers = [sc.Answer(0, 0.0, 5.0, "guten morgen meine damen und herren"),
               sc.Answer(1, 50.0, 55.0, "")]
    s = sc.score_recording(ref, answers)
    assert s["runaway_answers"]["n"] == 0
    assert s["empty_answers"] == 1


def test_loop_spans_need_three_back_to_back_repeats_over_twelve_words():
    assert sc.loop_spans("ja ja ja ja".split()) == []
    assert sc.loop_spans(("das ist so " * 4).split()) == [(0, 3, 4)]
    assert sc.loop_spans(("a b c d " * 3).split()) == [(0, 4, 3)]
    words = "vorher".split() + ["sehr"] * 15 + "nachher".split()
    assert sc.loop_spans(words) == [(1, 1, 15)]


def test_paired_delta_uses_the_same_reference_windows():
    ref = " ".join(f"w{i}" for i in range(40))
    good = {"r": sc.score_recording(ref, [sc.Answer(0, 0, 1, ref)], window_words=10)}
    bad_hyp = " ".join(f"w{i}" if i % 4 else "x" for i in range(40))
    bad = {"r": sc.score_recording(ref, [sc.Answer(0, 0, 1, bad_hyp)], window_words=10)}
    d = sc.paired_delta(bad, good)
    assert d["delta_pp"] == 25.0 and d["windows"] == 4
    assert d["wer_pct"] == 25.0 and d["baseline_wer_pct"] == 0.0


def test_vtt_onsets_compare_model_word_starts_with_cue_starts():
    vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nGuten Morgen\n\n"
           "00:00:03.000 --> 00:00:05.000\nmeine Damen\n")
    cues = parse_vtt(vtt)
    words = [{"word": "Guten", "start": 1.2, "end": 1.5},
             {"word": "Morgen", "start": 1.5, "end": 2},
             {"word": "meine", "start": 2.9, "end": 3.2}, {"word": "Damen", "start": 3.2, "end": 4}]
    assert [round(x, 3) for x in onset_offsets(cues, words)] == [0.2, -0.1]


# ── word timestamps in answers ───────────────────────────────────────────────


def test_word_timestamps_are_read_from_both_server_shapes():
    nemo = {"text": "a b", "timestamps": {"word": [{"word": "a", "start": 0.1, "end": 0.2},
                                                   {"word": "b", "start": "x", "end": 0.3}]}}
    assert parse_word_timestamps(nemo) == [{"word": "a", "start": 0.1, "end": 0.2}]
    oai = {"text": "a", "words": [{"word": "a", "start": 1, "end": 0.5}]}
    assert parse_word_timestamps(oai) == [{"word": "a", "start": 1.0, "end": 1.0}]
    assert parse_word_timestamps({"text": "a"}) is None


def test_timestamp_probe_falls_back_to_json_when_verbose_json_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        if b"verbose_json" in body:
            return httpx.Response(502, json={"detail": {"upstream_status": 400}})
        return httpx.Response(200, json={"text": "hallo", "x_raven": {"model": "m"}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await probe_timestamps(c, "http://x/v1/audio/transcriptions", "m", b"w", 1.0)

    res = asyncio.run(go())
    assert res["ask"] == ASK_JSON and res["word_timestamps"] is False
    assert res["refusal"].startswith("HTTP 502") and res["x_raven"] == {"model": "m"}


# ── the sweep and its stop rule ──────────────────────────────────────────────


def test_answer_budget_is_productions_read_timeout():
    assert answer_budget_s(10.0) == 70.0
    assert answer_budget_s(600.0) == 300.0


def _out(key: str, audio: float, lat: float, status: str = "ok", t0: float = 0.0) -> Outcome:
    return Outcome(key=key, audio_s=audio, budget_s=answer_budget_s(audio), t_start=t0,
                   t_end=t0 + lat, status=status)


def test_a_step_passes_only_without_errors_and_within_the_budget():
    good = [_out(str(i), 30.0, 2.0, t0=i * 0.5) for i in range(20)]
    s = step_summary(4, good, 0.0, 9.5, {}, 1.0)
    assert s["within_threshold"] and s["errors"] == 0 and s["answered"] == 20
    assert s["rtfx"] == round(600.0 / 11.5, 2)
    slow = [_out(str(i), 30.0, 95.0) for i in range(20)]  # budget 90 s
    assert not step_summary(4, slow, 0.0, 0.0, {}, 1.0)["within_threshold"]
    one_timeout = good[:-1] + [_out("t", 30.0, 90.0, status="timeout")]
    s = step_summary(4, one_timeout, 0.0, 9.5, {}, 1.0)
    assert not s["within_threshold"] and s["timeouts"] == 1


def test_sweep_stops_one_step_after_the_first_failure():
    steps = [{"n": 1, "within_threshold": True}, {"n": 2, "within_threshold": True}]
    assert next_action(steps) == "continue"
    steps.append({"n": 4, "within_threshold": False})
    assert next_action(steps) == "one_more" and max_n_within(steps) == 2
    steps.append({"n": 8, "within_threshold": True})
    assert next_action(steps) == "stop" and max_n_within(steps) == 2


def test_slice_size_scales_with_n_and_is_capped_by_the_corpus():
    assert slice_size("vad", 1, 10_000) == 256 and slice_size("vad", 64, 10_000) == 1024
    assert slice_size("window", 64, 200) == 200


def test_run_items_keeps_exactly_n_in_flight_and_maps_timeouts():
    in_flight = [0, 0]

    async def handler(request: httpx.Request) -> httpx.Response:
        in_flight[0] += 1
        in_flight[1] = max(in_flight[1], in_flight[0])
        await asyncio.sleep(0.01)
        in_flight[0] -= 1
        if b"slow" in request.read():
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json={"text": "ok", "words": [
            {"word": "ok", "start": 0.0, "end": 0.1}]})

    blk = vad.Block(0, ((0, 16000),), ((0.0, 1.0, 0.0),))
    items = [Item("r", "vad", vad.Block(i, blk.frame_ranges, blk.pieces)) for i in range(12)]
    wav = {it.key: (b"slow" if it.block.index == 5 else b"fast") for it in items}
    outs, _t0, _t1 = asyncio.run(run_items(
        "http://x/v1/audio/transcriptions", "m", items, lambda it: wav[it.key], 3, ASK_WORDS,
        transport=httpx.MockTransport(handler)))
    assert in_flight[1] == 3
    assert [o.status for o in outs].count("timeout") == 1 and outs[5].status == "timeout"
    assert outs[0].words == [{"word": "ok", "start": 0.0, "end": 0.1}]


# ── named load variants ──────────────────────────────────────────────────────


def test_load_variants_set_only_host_knobs_and_say_why():
    from raven_asr.load.variants import load_variants

    variants = load_variants()
    assert "primeline-parakeet@prod-batch4" in variants
    for v in variants.values():
        assert v["reason"] and v["env"]


def test_a_variant_naming_a_model_setting_is_refused(tmp_path):
    from raven_asr.load.variants import RECIPES_FILE, VariantError, load_variants

    bad = tmp_path / "v.toml"
    bad.write_text('[variants."primeline-parakeet@x"]\nbase = "primeline-parakeet"\n'
                   'env = { RAVEN_SERVE_MAX_NUM_SEQS = "8" }\nreason = "r"\n')
    with pytest.raises(VariantError, match="not nemo host knobs"):
        load_variants(bad, RECIPES_FILE)


def test_launch_takes_the_nemo_batching_knobs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))
    import launch
    from recipe import load_recipes

    rid = "primeline-parakeet"
    _argv, env = launch.build(load_recipes()[rid], rid, {"RAVEN_SERVE_NEMO_MAX_BATCH": "4",
                                                       "RAVEN_SERVE_NEMO_BATCH_WAIT_MS": "15"},
                              python="python", checkpoint="/x.nemo")
    assert env["NEMO_MAX_BATCH"] == "4" and env["NEMO_BATCH_WAIT_MS"] == "15"
    _argv, env = launch.build(load_recipes()[rid], rid, {}, python="python", checkpoint="/x.nemo")
    assert env["NEMO_MAX_BATCH"] == "16" and env["NEMO_BATCH_WAIT_MS"] == "20"


# ── the transformers stand-in's CTC word times ───────────────────────────────


def test_ctc_word_times_come_from_the_frames_the_labels_fire_on():
    pytest.importorskip("fastapi")
    path = Path(__file__).resolve().parents[1] / "serving" / "vllm" / "transformers_server.py"
    spec = importlib.util.spec_from_file_location("raven_transformers_server", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["raven_transformers_server"] = mod
    spec.loader.exec_module(mod)
    blank = 0
    # frames: _ HA HA _ LO _ _ WE LT LT _   (HA, WE open words)
    frames = [0, 1, 1, 0, 2, 0, 0, 3, 4, 4, 0]
    spell = {1: "ha", 2: "llo", 3: "we", 4: "lt"}
    words = mod.ctc_word_times(frames, blank, lambda t: t in (1, 3),
                               lambda ids: "".join(spell[i] for i in ids), 0.08)
    assert words == [{"word": "hallo", "start": 0.08, "end": 0.4},
                     {"word": "welt", "start": 0.56, "end": 0.8}]


def test_words_on_silence_count_as_a_runaway_answer():
    ref = "guten morgen meine damen und herren"
    answers = [sc.Answer(0, 0.0, 5.0, ref), sc.Answer(1, 50.0, 55.0, "applaus applaus danke schön")]
    s = sc.score_recording(ref, answers)
    assert s["runaway_answers"]["n"] == 1 and s["insertions"] == 4


def test_reference_past_the_end_of_the_audio_is_set_apart():
    ref = " ".join(f"w{i}" for i in range(40))
    heard = " ".join(f"w{i}" for i in range(20))
    # The model talked until the last second of a 100 s recording: the second
    # half of the reference was never in the audio.
    s = sc.score_recording(ref, [sc.Answer(0, 0.0, 100.0, heard)], window_words=10,
                           duration_s=100.0)
    assert s["wer_pct"] == 50.0 and s["wer_pct_audible"] == 0.0
    assert s["windows_beyond_audio"] == 2 and s["dropout_windows"] == 0
    # A model that stopped at 30 s of a 1000 s recording keeps its deletions.
    s = sc.score_recording(ref, [sc.Answer(0, 0.0, 30.0, heard)], window_words=10,
                           duration_s=1000.0)
    assert s["wer_pct_audible"] == 50.0 and s["windows_beyond_audio"] == 0
