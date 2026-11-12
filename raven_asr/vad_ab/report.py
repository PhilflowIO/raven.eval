"""Turn per-track arm outputs into the comparison: default setting, equal false alarm, AUC, cost.

Results layout written by ``python -m raven_asr.vad_ab run``::

    <results>/<arm>/<track id>.json   islands per swept threshold, CPU seconds, duration
    <results>/<arm>/<track id>.npy    the arm's 10 ms grid scores

Comparison rule: detectors are compared at equal false-alarm rate — each arm's
swept setting with the lowest miss whose false-alarm rate does not exceed arm
0's at its production setting — and by frame ROC-AUC. Never at equal threshold.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from raven_asr.load import vad as planner

from . import crosstalk as xt
from .arms import ARMS, neural_islands, read_pcm16
from .frames import (
    best_at_false_alarm,
    miss_at_frame_fpr,
    n_frames,
    pipeline_auc,
    roc_auc,
    summarise,
    to_grid,
    track_counts,
)

REFERENCE_ARM = "0-silencedetect"


def load_arm(results: str, arm: str, track_ids: Sequence[str]) -> dict[str, dict]:
    out = {}
    for tid in track_ids:
        j = Path(results, arm, f"{tid}.json")
        if j.exists():
            d = json.loads(j.read_text())
            d["score10"] = np.load(Path(results, arm, f"{tid}.npy"))
            out[tid] = d
    return out


def _labels(track: dict, n: int) -> tuple[np.ndarray, np.ndarray | None]:
    target = to_grid(track["target"], n)
    cross = to_grid(track["crosstalk"], n) if track.get("crosstalk") is not None else None
    return target, cross


def _counts_for(track: dict, islands, n: int) -> dict:
    target, cross = _labels(track, n)
    c = track_counts(to_grid(islands, n), target, cross)
    c["blocks"] = len(planner.plan_blocks(islands))
    return c


def evaluate_arm(tracks: Sequence[dict], runs: dict[str, dict], arm_name: str) -> dict:
    """Curve over the sweep, the production setting, frame AUC, CPU cost, density guard, blocks."""
    arm = ARMS[arm_name]
    tracks = [t for t in tracks if t["id"] in runs]
    if not tracks:
        return {"arm": arm_name, "tracks": 0}
    keys = sorted({k for t in tracks for k in runs[t["id"]]["islands"]}, key=float)
    per_thr: dict[str, list[dict]] = defaultdict(list)
    for t in tracks:
        r = runs[t["id"]]
        n = n_frames(r["duration_s"])
        for k in keys:
            if k in r["islands"]:
                c = _counts_for(t, r["islands"][k], n)
                c["track"] = t["id"]
                per_thr[k].append(c)
    curve = []
    for k in keys:
        s = summarise(per_thr[k])
        s["threshold"] = float(k)
        s["dense_track_share_pct"] = round(100.0 * np.mean(
            [c["sent_s"] / c["duration_s"] > planner.MAX_SPEECH_RATIO for c in per_thr[k]]), 1)
        s["blocks_per_audio_h"] = round(sum(c["blocks"] for c in per_thr[k]) / s["audio_h"], 1)
        curve.append(s)
    default = next(p for p in curve if abs(p["threshold"] - arm.default) < 1e-9)
    scores = np.concatenate([runs[t["id"]]["score10"][:n_frames(runs[t["id"]]["duration_s"])]
                             for t in tracks])
    labels = np.concatenate([_labels(t, n_frames(runs[t["id"]]["duration_s"]))[0] for t in tracks])
    cpu = sum(runs[t["id"]].get("cpu_s", float("nan")) for t in tracks)
    hours = sum(runs[t["id"]]["duration_s"] for t in tracks) / 3600
    return {
        "arm": arm_name, "note": arm.note, "tracks": len(tracks), "audio_h": round(hours, 3),
        "default_threshold": arm.default, "default": default, "curve": curve,
        "frame_auc": roc_auc(scores, labels), "pipeline_auc": pipeline_auc(curve),
        "frame_miss_at_fpr_pct": {f"{f:g}": miss_at_frame_fpr(scores, labels, f) for f in (5.0, 10.0)},
        "cpu_s_per_audio_h": round(cpu / hours, 2) if hours else None,
        "per_track_default": {c["track"]: c for c in per_thr[repr(float(arm.default))]},
        "per_track": {k: {c["track"]: c for c in v} for k, v in per_thr.items()},
    }


def breakdown(tracks: Sequence[dict], per_track: dict[str, dict], key: str) -> dict:
    """Miss % (and crosstalk sent %) per value of ``meta[key]``."""
    groups: dict = defaultdict(list)
    for t in tracks:
        if t["id"] in per_track:
            groups[t["meta"].get(key)].append(per_track[t["id"]])
    return {str(g): summarise(v) for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def compare(tracks: Sequence[dict], results: str, arms: Sequence[str],
            fa_points: Sequence[float] = (5.0, 10.0, 20.0)) -> dict:
    """All arms on one manifest, with the equal-false-alarm picks."""
    ids = [t["id"] for t in tracks]
    evals = {a: evaluate_arm(tracks, load_arm(results, a, ids), a) for a in arms}
    ref = evals.get(REFERENCE_ARM, {}).get("default")
    ref_fa = ref["false_alarm_rate_pct"] if ref else None
    out = {"reference_arm": REFERENCE_ARM, "reference_fa_rate_pct": ref_fa, "arms": {}}
    for a, e in evals.items():
        if not e.get("tracks"):
            out["arms"][a] = e
            continue
        picks = {}
        if ref_fa is not None:
            picks["at_reference_fa"] = best_at_false_alarm(e["curve"], ref_fa)
        for fa in fa_points:
            picks[f"at_fa_{fa:g}pct"] = best_at_false_alarm(e["curve"], fa)
        e["equal_fa"] = {k: _slim(v) for k, v in picks.items()}
        if tracks and tracks[0]["meta"].get("set") == "A":
            pt = e["per_track_default"]
            e["by_level_default"] = breakdown(tracks, pt, "level_dbfs")
            e["by_snr_default"] = breakdown(tracks, pt, "snr_db")
            ref_pick = picks.get("at_reference_fa")
            if ref_pick:
                pt_eq = e["per_track"][repr(float(ref_pick["threshold"]))]
                e["by_level_equal_fa"] = breakdown(tracks, pt_eq, "level_dbfs")
                e["by_snr_equal_fa"] = breakdown(tracks, pt_eq, "snr_db")
        e["curve"] = [_slim(p) for p in e["curve"]]
        e.pop("per_track", None)
        out["arms"][a] = e
    return out


def _slim(p: dict | None) -> dict | None:
    if p is None:
        return None
    return {k: p[k] for k in ("threshold", "missed_pct", "false_alarm_rate_pct", "false_alarm_s_per_h",
                              "crosstalk_sent_pct", "sent_share_pct", "dense_track_share_pct",
                              "blocks_per_audio_h") if k in p}


def crosstalk_islands(tracks: Sequence[dict], results: str, arm_name: str, threshold: float | None = None,
                      *, read: Callable[[dict], np.ndarray] = lambda t: read_pcm16(t["wav"]),
                      floor_db: float | None = None) -> dict[str, dict]:
    """Per track: the arm's islands with the cross-track flag off and on, and the veto mask.

    Partners are the other tracks of the same meeting; a partner's speech mask
    is its own detector output at ``threshold`` (default: the arm's production
    setting). ``read`` returns a track's PCM16 samples. Shared by the frame
    A/B (:func:`crosstalk_ab`) and the word count (``deletions``), so both
    measure the same islands. ``floor_db`` turns on the floor guard
    (:func:`raven_asr.vad_ab.crosstalk.veto`): levels are read from the raw
    samples, so the normalisation gain never lifts a quiet track over the floor.
    Returns ``{track id: {"n", "off", "on", "veto", "own_level_db"}}``.
    """
    arm = ARMS[arm_name]
    if arm.kind != "neural":
        raise ValueError("the cross-track flag is wired for frame-probability arms")
    thr = arm.default if threshold is None else threshold
    by_id = {t["id"]: t for t in tracks}
    runs = load_arm(results, arm_name, list(by_id))
    from .frames import hysteresis

    levels, masks = {}, {}
    for tid, r in runs.items():
        n = n_frames(r["duration_s"])
        levels[tid] = xt.frame_levels(read(by_id[tid]), n)
        masks[tid] = hysteresis(r["score10"][:n], thr, arm.off(thr))
    out: dict[str, dict] = {}
    for t in tracks:
        tid = t["id"]
        if tid not in runs:
            continue
        r = runs[tid]
        n = n_frames(r["duration_s"])
        partners = [(levels[p["id"]], masks[p["id"]]) for p in tracks
                    if p["id"] != tid and p["meta"].get("meeting") == t["meta"].get("meeting")
                    and p["id"] in runs]
        v = xt.veto(levels[tid], partners, own_speech=masks[tid], floor_db=floor_db)
        out[tid] = {"n": n, "veto": v, "own_level_db": xt.own_speech_level(levels[tid], masks[tid]),
                    "off": neural_islands(r["score10"][:n], arm, thr, r["duration_s"]),
                    "on": neural_islands(r["score10"][:n], arm, thr, r["duration_s"], veto=v)}
    return out


def crosstalk_ab(tracks: Sequence[dict], results: str, arm_name: str, threshold: float | None = None,
                 *, read: Callable[[str], np.ndarray] = read_pcm16) -> dict:
    """The winner with the cross-track flag off and on (Testset A, two tracks per meeting)."""
    arm = ARMS[arm_name]
    thr = arm.default if threshold is None else threshold
    isl = crosstalk_islands(tracks, results, arm_name, thr, read=lambda t: read(t["wav"]))
    off, on = [], []
    for t in tracks:
        tid = t["id"]
        if tid not in isl:
            continue
        x = isl[tid]
        n = x["n"]
        c_off, c_on = _counts_for(t, x["off"], n), _counts_for(t, x["on"], n)
        target, _ = _labels(t, n)
        c_on["vetoed_target_s"] = float((x["veto"] & target).sum()) * 0.01
        c_off["track"] = c_on["track"] = tid
        off.append(c_off)
        on.append(c_on)
    res = {"arm": arm_name, "threshold": thr, "flag_off": summarise(off), "flag_on": summarise(on),
           "params": {"min_lead_db": xt.MIN_LEAD_DB, "min_corr": xt.MIN_CORR,
                      "corr_window_s": xt.CORR_WINDOW_S}}
    res["flag_on"]["vetoed_target_s"] = round(sum(c["vetoed_target_s"] for c in on), 2)
    res["by_level_on"] = breakdown(tracks, {c["track"]: c for c in on}, "level_dbfs")
    res["by_level_off"] = breakdown(tracks, {c["track"]: c for c in off}, "level_dbfs")
    return res


def markdown(name: str, cmp: dict) -> str:
    """The comparison as tables (German headers: the report goes to flow.raven)."""
    lines = [f"### {name}", "",
             f"Bezug: Arm 0 bei −40 dB, Fehlalarmrate {cmp['reference_fa_rate_pct']} %.", "",
             "| Arm | verpasst % (Default) | Fehlalarm s/h (Default) | verpasst % bei gleicher FA wie Arm 0 "
             "| Schwelle dort | Frame-AUC | Frame-verpasst % bei 5 % FPR | CPU s/h | Spuren >60 % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a, e in cmp["arms"].items():
        if not e.get("tracks"):
            lines.append(f"| {a} | – | – | – | – | – | – | – | – |")
            continue
        d = e["default"]
        eq = e["equal_fa"].get("at_reference_fa") or {}
        lines.append(
            f"| {a} | {d['missed_pct']:.2f} | {d['false_alarm_s_per_h']:.0f} | "
            f"{eq.get('missed_pct', float('nan')):.2f} | {eq.get('threshold', float('nan')):g} | "
            f"{(e['frame_auc'] or float('nan')):.3f} | "
            f"{(e['frame_miss_at_fpr_pct']['5'] if e['frame_miss_at_fpr_pct']['5'] is not None else float('nan')):.2f} | "
            f"{e['cpu_s_per_audio_h']:.1f} | {d['dense_track_share_pct']:.0f} % |")
    return "\n".join(lines) + "\n"
