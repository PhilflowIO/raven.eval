"""Speech-detector A/B (raven_asr.vad_ab): the second island source, the ports, the scoring."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest
from raven_asr.vad_ab import arms as A
from raven_asr.vad_ab import crosstalk as xt
from raven_asr.vad_ab import frames as F
from raven_asr.vad_ab import report as R
from raven_asr.vad_ab import testset_a as T

from raven_asr.load import vad

RATE = 16000


def _tone(seconds: float, db: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (10 ** (db / 20) * np.sqrt(2) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ── second island source ─────────────────────────────────────────────────────


def test_silences_from_mask_keeps_only_runs_of_at_least_min_silence():
    mask = np.ones(300, dtype=bool)
    mask[50:109] = False   # 0.59 s: too short
    mask[150:210] = False  # 0.60 s: a silence
    mask[250:] = False     # trailing 0.5 s: too short
    assert vad.silences_from_mask(mask, 0.01) == [(1.5, 2.1)]


def test_mask_silences_feed_the_production_island_builder():
    mask = np.zeros(1000, dtype=bool)
    mask[200:400] = True
    islands = vad.build_speech_islands(vad.silences_from_mask(mask, 0.01), 10.0)
    assert islands == [(pytest.approx(1.65), pytest.approx(4.35))]


def test_silencedetect_replica_counts_a_run_of_quiet_samples():
    x = np.concatenate([_tone(1.0, -20), np.zeros(int(0.7 * RATE), np.float32), _tone(1.0, -20),
                        np.full(int(0.5 * RATE), 0.005, np.float32)])
    sil = vad.silencedetect_silences(x, RATE, noise_db=-40)
    # The tone crosses zero, but never for 0.6 s; the 0.7 s gap is the one silence,
    # the trailing 0.5 s at -46 dBFS is too short.
    assert len(sil) == 1
    assert sil[0][0] == pytest.approx(1.0, abs=1e-3)
    assert sil[0][1] == pytest.approx(1.7, abs=1e-3)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_silencedetect_replica_matches_ffmpeg(tmp_path):
    import soundfile as sf

    rng = np.random.default_rng(0)
    parts = []
    for _ in range(6):
        parts.append(_tone(float(rng.uniform(0.5, 2.0)), float(rng.uniform(-35, -15))))
        parts.append((rng.standard_normal(int(rng.uniform(0.3, 1.5) * RATE)) * 10 ** (-55 / 20)).astype(np.float32))
    x = np.concatenate(parts)
    src = tmp_path / "x.wav"
    sf.write(src, x, RATE, subtype="PCM_16")
    pcm, _ = sf.read(src, dtype="int16")
    ours = vad.silencedetect_silences(pcm.astype(np.float32) / 32768.0, RATE, noise_db=-40)
    theirs = vad.parse_silence_log(vad.decode_and_detect(str(src), str(tmp_path / "o.wav")), len(pcm) / RATE)
    assert len(ours) == len(theirs)
    for (a, b), (c, d) in zip(ours, theirs, strict=True):
        assert a == pytest.approx(c, abs=2e-3)
        assert b == pytest.approx(d, abs=2e-3)


# ── adaptive floor port ──────────────────────────────────────────────────────


def test_adaptive_floor_sits_8_db_over_a_room_plateau_inside_its_clamp():
    levels = [-62.0] * 60 + [-30.0] * 40
    assert vad.adaptive_floor(levels) == pytest.approx(-54.0)
    assert vad.adaptive_floor([-45.0] * 60 + [-20.0] * 40) == -40.0  # held at the speech floor
    assert vad.adaptive_floor([-80.0] * 60 + [-20.0] * 40) == -60.0  # held at the lowest floor


def test_adaptive_floor_without_a_plateau_is_the_lowest():
    levels = list(np.linspace(-70, -20, 100))
    assert vad.adaptive_floor(levels) == -60.0


def test_adaptive_islands_fall_back_to_the_lowest_floor_for_an_open_microphone():
    # Room at -45 with a voice at -42: nothing rises 8 dB above the room, but the
    # microphone was open — production plans the track again at -60.
    levels = [-45.0] * 200 + [-42.0] * 40 + [-45.0] * 200
    floor, islands = vad.adaptive_islands(levels, 22.0)
    assert floor == -60.0
    assert islands  # the whole open stretch, not nothing


def test_window_levels_match_a_full_scale_sine():
    lv = vad.window_levels((_tone(0.2, -20) * 32767).astype(np.int16))
    assert len(lv) == 4
    assert lv[0] == pytest.approx(-20.0, abs=0.1)


# ── grid, hysteresis, counts, AUC ────────────────────────────────────────────


def test_to_grid_uses_frame_centres():
    m = F.to_grid([(0.005, 0.025)], 5)
    assert m.tolist() == [True, True, False, False, False]


def test_hysteresis_holds_between_the_thresholds():
    s = np.array([0.1, 0.6, 0.4, 0.3, 0.2, 0.4, 0.6, 0.36])
    assert F.hysteresis(s, 0.5, 0.35).tolist() == [False, True, True, False, False, False, True, True]


def test_hold_maps_32_ms_frames_onto_the_10_ms_grid():
    out = F.hold(np.array([0.0, 1.0]), 0.032, 7)
    assert out.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_counts_treat_a_real_overlap_as_speech_not_crosstalk():
    target = np.zeros(100, bool)
    target[10:30] = True
    cross = np.zeros(100, bool)
    cross[25:60] = True  # 25-30 overlaps the owner
    sent = np.zeros(100, bool)
    sent[20:50] = True
    c = F.track_counts(sent, target, cross)
    assert c["missed_s"] == pytest.approx(0.10)
    assert c["false_alarm_s"] == pytest.approx(0.20)
    assert c["crosstalk_s"] == pytest.approx(0.30)
    assert c["crosstalk_sent_s"] == pytest.approx(0.20)
    s = F.summarise([c])
    assert s["missed_pct"] == pytest.approx(50.0)
    assert s["crosstalk_sent_pct"] == pytest.approx(66.667, abs=1e-2)


def test_roc_auc_matches_the_pairwise_definition():
    rng = np.random.default_rng(1)
    scores = rng.integers(0, 5, 200).astype(float)
    labels = rng.random(200) < 0.4
    pos, neg = scores[labels], scores[~labels]
    brute = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    assert F.roc_auc(scores, labels) == pytest.approx(brute)
    assert F.roc_auc(np.arange(10.0), np.arange(10) >= 5) == 1.0


def test_equal_false_alarm_pick_never_exceeds_the_budget():
    curve = [{"threshold": t, "missed_pct": m, "false_alarm_rate_pct": f}
             for t, m, f in ((0.1, 1.0, 30.0), (0.3, 3.0, 12.0), (0.5, 6.0, 9.0), (0.7, 9.0, 4.0))]
    assert F.best_at_false_alarm(curve, 10.0)["threshold"] == 0.5
    assert F.best_at_false_alarm(curve, 3.0) is None


def test_pipeline_auc_of_a_perfect_point_is_one():
    assert F.pipeline_auc([{"missed_pct": 0.0, "false_alarm_rate_pct": 0.0}]) == pytest.approx(1.0)


def test_miss_at_frame_fpr():
    scores = np.concatenate([np.arange(100.0), np.arange(50.0, 150.0)])
    labels = np.concatenate([np.zeros(100, bool), np.ones(100, bool)])
    # 10 % FPR -> threshold at the 90th percentile of the negatives (90): 41 of 100 positives <= 90.
    assert F.miss_at_frame_fpr(scores, labels, 10.0) == pytest.approx(41.0)


def test_firered_smoothing_is_a_trailing_mean():
    out = A.smooth(np.array([0.0, 1.0, 1.0, 0.0, 0.0], np.float32), 2)
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.0])


# ── arms over a real WAV (engines injected) ──────────────────────────────────


def _speech_like_wav(tmp_path):
    import soundfile as sf

    x = np.concatenate([np.zeros(RATE * 3, np.float32), _tone(2.0, -50), np.zeros(RATE * 3, np.float32),
                        _tone(2.0, -25), np.zeros(RATE * 3, np.float32)])
    noise = np.random.default_rng(2).standard_normal(x.size).astype(np.float32) * 10 ** (-70 / 20)
    p = tmp_path / "t.wav"
    sf.write(p, x + noise, RATE, subtype="PCM_16")
    return str(p)


def test_silencedetect_arm_loses_the_minus_50_dbfs_voice(tmp_path):
    out = A.run_arm(A.ARMS["0-silencedetect"], _speech_like_wav(tmp_path))
    islands = out["islands"][repr(-40.0)]
    assert len(islands) == 1 and islands[0][0] > 7.5  # only the -25 dBFS voice
    assert out["score10"].size == 1300


def test_adaptive_arm_keeps_both_voices(tmp_path):
    out = A.run_arm(A.ARMS["1-adaptive-floor"], _speech_like_wav(tmp_path))
    assert len(out["islands"][repr(0.0)]) == 2


def test_neural_arm_turns_probabilities_into_production_islands(tmp_path):
    def probs(x):
        frames = int(np.ceil(len(x) / 512))
        p = np.zeros(frames, np.float32)
        t = np.arange(frames) * 0.032
        p[((t >= 3) & (t < 5)) | ((t >= 8) & (t < 10))] = 0.9
        return p, 0.032

    out = A.run_arm(A.ARMS["2-silero-0.5"], _speech_like_wav(tmp_path), probs_fn=probs)
    isl = out["islands"][repr(0.5)]
    assert len(isl) == 2
    assert isl[0][0] == pytest.approx(3 - 0.35, abs=0.05)


def test_normalisation_gain_lifts_a_quiet_track_to_minus_20_dbfs():
    pcm = (_tone(2.0, -50) * 32767).astype(np.int16)
    assert A.normalise_gain_db(pcm) == pytest.approx(30.0, abs=0.2)
    assert A.normalise_gain_db((_tone(2.0, -10) * 32767).astype(np.int16)) == 0.0


# ── crosstalk flag ───────────────────────────────────────────────────────────


def test_crosstalk_veto_flags_a_correlated_leak_but_not_a_real_overlap():
    rng = np.random.default_rng(3)
    env = -30 + 10 * np.sin(np.linspace(0, 20, 300)) + rng.normal(0, 1, 300)
    partner_levels = np.concatenate([env, env])
    partner_speech = np.ones(600, bool)
    own = np.concatenate([env - 20,                      # leak: same envelope, 20 dB down
                          -32 + rng.normal(0, 3, 300)])  # own voice at a similar level
    v = xt.veto(own, [(partner_levels, partner_speech)])
    assert v[20:280].mean() > 0.9
    assert v[320:580].mean() < 0.1


def test_rolling_corr_of_identical_series_is_one():
    a = np.sin(np.linspace(0, 10, 200))
    assert np.allclose(xt.rolling_corr(a, a, 21)[15:185], 1.0)


# ── Testset A pieces ─────────────────────────────────────────────────────────


def test_textgrid_words_reads_mfa_long_format(tmp_path):
    tg = tmp_path / "u.TextGrid"
    tg.write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 2\n'
                  'tiers? <exists>\nsize = 2\nitem []:\n    item [1]:\n        class = "IntervalTier"\n'
                  '        name = "words"\n        xmin = 0\n        xmax = 2\n        intervals: size = 3\n'
                  '        intervals [1]:\n            xmin = 0\n            xmax = 0.3\n            text = ""\n'
                  '        intervals [2]:\n            xmin = 0.3\n            xmax = 0.9\n            text = "hallo"\n'
                  '        intervals [3]:\n            xmin = 0.9\n            xmax = 2\n            text = "welt"\n'
                  '    item [2]:\n        class = "IntervalTier"\n        name = "phones"\n        xmin = 0\n'
                  '        xmax = 2\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n'
                  '            xmax = 2\n            text = "a"\n')
    assert T.textgrid_words(str(tg)) == [(0.3, 0.9, "hallo"), (0.9, 2.0, "welt")]


def test_meeting_grid_pairs_all_cells_with_neighbouring_levels():
    grid = T.meeting_grid()
    assert len(grid) == 15
    cells = [c for pair in grid for c in pair]
    assert len(set(cells)) == 30
    assert all(abs(a[0] - b[0]) <= 10 for a, b in grid)


def test_plan_timeline_fills_both_shares_inside_the_meeting():
    rng = np.random.default_rng(4)
    pool = [T.Utterance(f"u{i}", "", "", 10.0, ((0.2, 9.8),)) for i in range(200)]
    events = T.plan_timeline(rng, pool, (0.2, 0.1))
    per = [sum(u.duration for s, u, _ in events if s == spk) for spk in (0, 1)]
    assert per[0] >= 0.2 * T.TRACK_S and per[1] >= 0.1 * T.TRACK_S
    assert all(start + u.duration <= T.TRACK_S for _, u, start in events)
    overlaps = 0
    for seed in range(10):
        pool = [T.Utterance(f"u{i}", "", "", 10.0, ((0.2, 9.8),)) for i in range(200)]
        ev = T.plan_timeline(np.random.default_rng(seed), pool, (0.2, 0.2))
        overlaps += sum(1 for i in range(1, len(ev)) if ev[i][2] < ev[i - 1][2] + ev[i - 1][1].duration)
    assert overlaps >= 3  # both talking at once happens, on purpose


def test_active_level_counts_only_the_speech():
    x = np.concatenate([_tone(1.0, -30), np.zeros(RATE, np.float32)])
    assert T.active_level_db(x, [(0.0, 1.0)]) == pytest.approx(-30.0, abs=0.05)


def test_chain_lag_finds_a_known_delay():
    rng = np.random.default_rng(5)
    a = (rng.standard_normal(RATE * 25) * 3000).astype(np.int16)
    b = np.concatenate([np.zeros(80, np.int16), a[:-80]])
    assert T.chain_lag_s(a, b) == pytest.approx(80 / RATE)


# ── report over fake results ─────────────────────────────────────────────────


def test_compare_picks_each_arm_at_the_reference_false_alarm(tmp_path):
    tracks = [{"id": "t1", "wav": "", "duration_s": 10.0, "target": [(2.0, 4.0)], "crosstalk": None,
               "meta": {"set": "B"}}]
    runs = {
        "0-silencedetect": {"-40.0": [(1.65, 4.35)], "-50.0": [(0.0, 10.0)]},
        "2-silero-0.5": {"0.5": [(2.5, 4.0)], "0.2": [(1.8, 4.2)]},
    }
    for arm, isl in runs.items():
        d = tmp_path / arm
        d.mkdir()
        (d / "t1.json").write_text(json.dumps({"duration_s": 10.0, "cpu_s": 1.0, "islands": isl}))
        np.save(d / "t1.npy", np.linspace(0, 1, 1000).astype(np.float32))
    cmp = R.compare(tracks, str(tmp_path), list(runs))
    assert cmp["reference_fa_rate_pct"] == pytest.approx(8.75)
    eq = cmp["arms"]["2-silero-0.5"]["equal_fa"]["at_reference_fa"]
    assert eq["threshold"] == 0.2 and eq["missed_pct"] == 0.0
    assert "| 2-silero-0.5 |" in R.markdown("B", cmp)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_ffmpeg_silencedetect_cpu_is_measured(tmp_path):
    p = _speech_like_wav(tmp_path)
    assert A.ffmpeg_silencedetect_cpu(p) >= 0.0
    subprocess.run(["true"], check=True)


def test_deletion_scorer_blocks_and_reference():
    from raven_asr.vad_ab import deletions as Dl

    pcm = np.arange(RATE * 10, dtype=np.int16)
    blocks = Dl.block_wavs(pcm, [(1.0, 2.0), (3.0, 4.5)])
    assert len(blocks) == 1 and blocks[0][1] == pytest.approx(2.5)
    assert len(blocks[0][0]) == 44 + 2 * int(2.5 * RATE)
    track = {"utterances": [{"role": "own", "start_s": 5.0, "text": "zwei"},
                            {"role": "leak", "start_s": 1.0, "text": "fremd"},
                            {"role": "own", "start_s": 1.0, "text": "eins"}]}
    assert Dl.reference_text(track) == "eins zwei"


def test_foreign_insertions_count_only_leaked_words_once_each():
    from raven_asr.vad_ab import deletions as Dl

    from raven_asr.load.score import align

    ref = ["guten", "morgen"]
    hyp = ["guten", "und", "und", "morgen", "zug", "quatsch"]
    ops = align(ref, hyp)
    # "und" leaked once: one of its two insertions is foreign; "zug" leaked; "quatsch" never said.
    assert Dl.foreign_insertions(ops, hyp, ["und", "zug"]) == 2
    assert Dl.foreign_insertions(ops, hyp, []) == 0
    track = {"utterances": [{"role": "leak", "start_s": 2.0, "text": "b"},
                            {"role": "own", "start_s": 0.0, "text": "x"},
                            {"role": "leak", "start_s": 1.0, "text": "a"}]}
    assert Dl.leaked_text(track) == "a b"


def test_word_summary_reports_rates_per_level():
    from raven_asr.vad_ab import deletions as Dl

    def tr(level, ref, d, i, f):
        return {"ref_words": ref, "delete": d, "insert": i, "foreign_insert": f, "substitute": 0,
                "equal": ref - d, "blocks": 1, "failed_blocks": 0, "sent_s": 1.0, "level_dbfs": level}

    s = Dl.summarise_words({"a": tr(-20.0, 100, 1, 10, 8), "b": tr(-55.0, 100, 9, 30, 20)})
    assert s["deletion_pct"] == 5.0 and s["foreign_insertion_pct"] == 14.0
    assert list(s["deletion_pct_by_level"]) == ["-20.0", "-55.0"]
    assert s["foreign_insertion_pct_by_level"] == {"-20.0": 8.0, "-55.0": 20.0}


def test_crosstalk_islands_flag_on_removes_the_leak_and_off_matches_the_arm(tmp_path):
    """Two-track meeting: B's track carries A's voice 20 dB down. Flag on drops it; off is the plain arm."""
    arm = A.ARMS["3-silero-0.3-norm"]
    rng = np.random.default_rng(7)
    dur, n = 6.0, 600
    env = np.repeat(rng.uniform(0.2, 1.0, 60), RATE // 10)  # A's level moves every 100 ms
    voice = env * np.sin(2 * np.pi * 220 * np.arange(env.size) / RATE)
    pcm = {"a": (0.3 * voice * 32767).astype(np.int16),
           "b": (0.03 * voice * 32767).astype(np.int16)}  # -20 dB, same envelope
    speech = np.ones(n, np.float32)
    (tmp_path / arm.name).mkdir()
    tracks = []
    for tid in ("a", "b"):
        np.save(tmp_path / arm.name / f"{tid}.npy", speech)
        (tmp_path / arm.name / f"{tid}.json").write_text(json.dumps({"duration_s": dur}))
        tracks.append({"id": tid, "wav": f"{tid}.wav", "meta": {"meeting": "m"}})
    isl = R.crosstalk_islands(tracks, str(tmp_path), arm.name, read=lambda t: pcm[t["id"]])
    assert isl["a"]["on"] == isl["a"]["off"] == A.neural_islands(speech, arm, arm.default, dur)
    assert isl["b"]["veto"][50:550].mean() > 0.9
    assert isl["b"]["on"] == []
