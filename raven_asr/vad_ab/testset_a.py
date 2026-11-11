"""Testset A: synthetic German two-person meetings with known speech, levels, noise and leakage.

Why synthetic: the question is how a detector behaves for a quiet speaker in a
noisy room whose neighbour leaks into the microphone. That needs speech whose
level, noise and leakage are set, not guessed, and labels that are exact. Real
meetings confirm the winner later (flow.raven#11879).

Recipe (research report 2026-11-10 §4):

* Speech: FLEURS German (``google/fleurs`` ``de_de``, CC BY 4.0). Labels are the
  word intervals from forced alignment of the clean utterance (Montreal Forced
  Aligner, ``german_mfa``); words closer than :data:`WORD_MERGE_GAP_S` are one
  stretch of speech.
* Meetings of two people, ten minutes, one track per person. Each track's owner
  speaks at a fixed level (active speech level over the aligned words) from
  :data:`LEVELS_DBFS` — Raven's loss sits at −50 dBFS — and the track carries
  noise at a fixed SNR from :data:`SNRS_DB` relative to that level. The 30
  (level, SNR) cells are paired by level into 15 meetings, so the two people of
  a meeting are within one level step of each other.
* Leakage: the partner's utterances are also on this track, 15–25 dB below the
  owner's level (uniform per utterance), labelled as crosstalk, not as speech.
  About one switch in seven overlaps (both talk), which is real speech for the
  owner and must not be cut.
* Noise: DEMAND environments (CC BY 4.0, channel 1) and MUSAN ``noise`` (CC BY
  4.0), alternating per track. WHAM! is not used (CC BY-NC).
* Browser chain: WebRTC audio processing noise suppression (level 2, AGC off so
  the level grid is the level on the egress track) and Opus 32 kbps, decoded
  back to 16 kHz PCM — the shape of a LiveKit egress track. No echo canceller:
  leakage through the air is not echo. The chain's delay is measured per track
  by cross-correlation and the labels are shifted by it.

Nothing here is committed as audio; the manifest records every placement,
gain and seed so the set can be rebuilt.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frames import merge_intervals

SAMPLE_RATE = 16000
TRACK_S = 600.0
LEVELS_DBFS = (-20.0, -30.0, -40.0, -45.0, -50.0, -55.0)
SNRS_DB = (-5.0, 0.0, 5.0, 10.0, 20.0)
LEAK_DB = (15.0, 25.0)
SPEECH_SHARE = (0.10, 0.30)
OVERLAP_P = 0.15
SWITCH_P = 0.75
WORD_MERGE_GAP_S = 0.2
CROSSFADE_S = 0.05
NS_LEVEL = 2
OPUS_KBPS = 32
DEMAND_ENVS = ("OOFFICE", "OMEETING", "OHALLWAY", "PCAFETER", "DKITCHEN", "DLIVING", "STRAFFIC", "TBUS")


# ── utterances: FLEURS rows + forced alignment ───────────────────────────────


def write_fleurs_corpus(parquet: str, out_dir: str) -> int:
    """FLEURS rows → ``<out>/fleurs/<row>.wav`` + ``.lab`` (raw transcription) for MFA."""
    import pyarrow.parquet as pq
    import soundfile as sf

    spk = Path(out_dir) / "fleurs"
    spk.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet, columns=["id", "audio", "raw_transcription", "gender"])
    n = 0
    for i, row in enumerate(table.to_pylist()):
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            continue
        stem = f"fleurs-{i:04d}-{row['id']}"
        sf.write(spk / f"{stem}.wav", audio[:, 0], SAMPLE_RATE, subtype="PCM_16")
        (spk / f"{stem}.lab").write_text(row["raw_transcription"].strip() + "\n", encoding="utf-8")
        n += 1
    return n


_INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s*xmin = ([0-9.eE+-]+)\s*xmax = ([0-9.eE+-]+)\s*text = \"((?:[^\"]|\"\")*)\"")


def textgrid_words(path: str, tier: str = "words") -> list[tuple[float, float, str]]:
    """Non-empty intervals of one tier of a long-format TextGrid (what MFA writes)."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"item \[\d+\]:", text)
    for block in blocks:
        if re.search(r'name = "' + re.escape(tier) + r'"', block):
            return [(float(a), float(b), t.replace('""', '"')) for a, b, t in _INTERVAL_RE.findall(block)
                    if t.strip() and t.strip() not in ("<eps>", "sil", "sp")]
    return []


@dataclass(frozen=True)
class Utterance:
    uid: str
    wav: str
    text: str
    duration: float
    speech: tuple[tuple[float, float], ...]  # merged word intervals, utterance time


def load_utterances(corpus_dir: str, align_dir: str) -> list[Utterance]:
    """Every corpus utterance that has an alignment with at least one word."""
    import soundfile as sf

    out: list[Utterance] = []
    for wav in sorted(Path(corpus_dir).rglob("*.wav")):
        grid = next(iter(Path(align_dir).rglob(f"{wav.stem}.TextGrid")), None)
        if grid is None:
            continue
        words = textgrid_words(str(grid))
        if not words:
            continue
        info = sf.info(str(wav))
        out.append(Utterance(wav.stem, str(wav), wav.with_suffix(".lab").read_text("utf-8").strip(),
                             info.frames / info.samplerate,
                             tuple(merge_intervals([(s, e) for s, e, _ in words], WORD_MERGE_GAP_S))))
    return out


# ── composition ──────────────────────────────────────────────────────────────


def meeting_grid() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The 30 (level, SNR) cells paired into 15 two-person meetings, neighbours by level."""
    cells = [(lv, snr) for lv in LEVELS_DBFS for snr in SNRS_DB]
    return [(cells[i], cells[i + 1]) for i in range(0, len(cells), 2)]


def plan_timeline(rng: np.random.Generator, pool: list[Utterance], shares: tuple[float, float],
                  total_s: float = TRACK_S) -> list[tuple[int, Utterance, float]]:
    """``[(speaker 0|1, utterance, start_s), ...]`` filling each speaker's share of the meeting.

    Each speaker takes utterances from the end of ``pool`` (which is consumed)
    until their labelled speech reaches their share. Turns mostly alternate; on a switch the next
    speaker sometimes starts before the previous one finishes (overlap). The
    pauses are exponential draws scaled so the meeting ends just inside
    ``total_s``; if the speech alone does not fit, utterances are dropped from
    the longer side.
    """
    own: list[list[Utterance]] = [[], []]
    for spk in (0, 1):
        got = 0.0
        while pool and got < shares[spk] * total_s:
            u = pool.pop()
            own[spk].append(u)
            got += sum(e - s for s, e in u.speech) or u.duration  # labelled speech, not file length
    while True:
        order: list[int] = []
        left = [len(own[0]), len(own[1])]
        last: int | None = None
        while left[0] or left[1]:
            if last is not None and left[1 - last] and (not left[last] or rng.random() < SWITCH_P):
                spk = 1 - last
            elif last is not None and left[last]:
                spk = last
            else:
                spk = 0 if left[0] >= left[1] else 1
            order.append(spk)
            left[spk] -= 1
            last = spk
        seq = [(spk, own[spk][sum(1 for o in order[:i] if o == spk)]) for i, spk in enumerate(order)]
        steps: list[float] = []  # >0: pause weight, <0: overlap seconds
        for i in range(1, len(seq)):
            prev_u = seq[i - 1][1]
            if seq[i][0] != seq[i - 1][0] and rng.random() < OVERLAP_P:
                steps.append(-min(float(rng.uniform(0.3, 1.5)), max(0.0, prev_u.duration - 0.5)))
            else:
                steps.append(float(rng.exponential(1.0)) + 1e-3)
        lead, tail = float(rng.uniform(0.5, 3.0)), 0.5
        speech = sum(u.duration for _, u in seq)
        overlap = -sum(st for st in steps if st < 0)
        free = total_s - lead - tail - speech + overlap
        if free >= 0.3 * max(1, sum(1 for st in steps if st > 0)) or not seq:
            break
        longer = 0 if sum(u.duration for u in own[0]) >= sum(u.duration for u in own[1]) else 1
        own[longer].pop()
    weight = sum(st for st in steps if st > 0)
    events: list[tuple[int, Utterance, float]] = []
    t = lead
    for i, (spk, u) in enumerate(seq):
        if i:
            st = steps[i - 1]
            t = t + seq[i - 1][1].duration + (st if st < 0 else st / weight * free)
        events.append((spk, u, t))
    return events


def active_level_db(x: np.ndarray, speech: Sequence[tuple[float, float]], rate: int = SAMPLE_RATE) -> float:
    """RMS level in dBFS over the speech stretches only (the level an utterance is set to)."""
    parts = [x[int(s * rate):int(math.ceil(e * rate))] for s, e in speech]
    v = np.concatenate(parts) if parts else x
    rms = float(np.sqrt(np.mean(np.square(v, dtype=np.float64)))) if v.size else 0.0
    return 20 * math.log10(max(rms, 1e-10))


def crossfade_concat(pieces: list[np.ndarray], n: int, fade: int) -> np.ndarray:
    """Join pieces with linear cross-fades until ``n`` samples."""
    out = np.zeros(0, dtype=np.float32)
    for p in pieces:
        p = p.astype(np.float32)
        if out.size >= fade and p.size > fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            out[-fade:] = out[-fade:] * (1 - ramp) + p[:fade] * ramp
            out = np.concatenate((out, p[fade:]))
        else:
            out = np.concatenate((out, p))
        if out.size >= n:
            break
    return out[:n]


def make_noise(rng: np.random.Generator, files: Sequence[str], n: int) -> np.ndarray:
    """``n`` samples of noise from random excerpts of ``files``, unit RMS."""
    import soundfile as sf

    pieces: list[np.ndarray] = []
    total = 0
    while total < n + SAMPLE_RATE:
        data, rate = sf.read(files[int(rng.integers(len(files)))], dtype="float32", always_2d=True)
        x = data[:, 0]
        if rate != SAMPLE_RATE:
            x = _resample(x, rate)
        if x.size < SAMPLE_RATE:
            continue
        start = int(rng.integers(0, max(1, x.size // 2)))
        pieces.append(x[start:])
        total += x.size - start
    noise = crossfade_concat(pieces, n, int(CROSSFADE_S * SAMPLE_RATE))
    rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    return noise / max(rms, 1e-10)


def _resample(x: np.ndarray, rate: int) -> np.ndarray:
    import soxr

    return soxr.resample(x, rate, SAMPLE_RATE).astype(np.float32)


def webrtc_ns(pcm16: np.ndarray, level: int = NS_LEVEL) -> np.ndarray:
    """WebRTC audio processing noise suppression on 10 ms frames, AGC off."""
    from webrtc_noise_gain import AudioProcessor

    proc = AudioProcessor(0, level)
    hop = SAMPLE_RATE // 100
    n = (len(pcm16) // hop) * hop
    out = np.empty(n, dtype=np.int16)
    for i in range(0, n, hop):
        res = proc.Process10ms(pcm16[i:i + hop].tobytes())
        out[i:i + hop] = np.frombuffer(res.audio, dtype=np.int16)
    return out


def opus_roundtrip(pcm16: np.ndarray, kbps: int = OPUS_KBPS) -> np.ndarray:
    """Encode to Opus at ``kbps`` (VoIP mode) and decode back to 16 kHz mono PCM."""
    with tempfile.TemporaryDirectory() as tmp:
        raw, ogg, wav = (os.path.join(tmp, f) for f in ("in.raw", "x.ogg", "out.wav"))
        pcm16.tofile(raw)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "s16le", "-ar", str(SAMPLE_RATE),
                        "-ac", "1", "-i", raw, "-c:a", "libopus", "-b:a", f"{kbps}k",
                        "-application", "voip", ogg], check=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", ogg, "-ar", str(SAMPLE_RATE),
                        "-ac", "1", "-c:a", "pcm_s16le", wav], check=True)
        import soundfile as sf

        data, _ = sf.read(wav, dtype="int16")
    return np.asarray(data, dtype=np.int16)


def chain_lag_s(before: np.ndarray, after: np.ndarray, max_lag_s: float = 0.1) -> float:
    """Delay the processing chain added, by cross-correlation on the loudest 20 s."""
    win = 20 * SAMPLE_RATE
    if before.size <= win:
        start = 0
    else:
        energy = np.convolve(np.square(before.astype(np.float64)), np.ones(SAMPLE_RATE), "valid")
        start = int(np.clip(np.argmax(energy) - win // 2, 0, before.size - win))
    a = before[start:start + win].astype(np.float64)
    b = after[start:start + win].astype(np.float64)
    if a.size < win or b.size < win:
        return 0.0
    max_lag = int(max_lag_s * SAMPLE_RATE)
    spec = np.fft.rfft(b, 2 * win) * np.conj(np.fft.rfft(a, 2 * win))
    corr = np.fft.irfft(spec)
    lags = np.concatenate((corr[:max_lag + 1], corr[-max_lag:]))
    idx = int(np.argmax(lags))
    lag = idx if idx <= max_lag else idx - (2 * max_lag + 1)
    return lag / SAMPLE_RATE


def _load_utt(u: Utterance) -> np.ndarray:
    import soundfile as sf

    x, _ = sf.read(u.wav, dtype="float32")
    return x


def build_meeting(m: int, cells, pool: list[Utterance], noises: Sequence[tuple[str, list[str]]],
                  out_dir: str, seed: int, *, process: bool = True) -> list[dict]:
    """Render one meeting's two tracks; return their manifest entries."""
    import soundfile as sf

    rng = np.random.default_rng([seed, m])
    shares = (float(rng.uniform(*SPEECH_SHARE)), float(rng.uniform(*SPEECH_SHARE)))
    events = plan_timeline(rng, pool, shares)
    n = int(TRACK_S * SAMPLE_RATE)
    tracks: list[dict] = []
    for spk in (0, 1):
        level, snr = cells[spk]
        noise_name, noise_files = noises[(2 * m + spk) % len(noises)]
        sig = np.zeros(n, dtype=np.float64)
        target: list[tuple[float, float]] = []
        crosstalk: list[tuple[float, float]] = []
        placed: list[dict] = []
        for who, utt, start in events:
            x = _load_utt(utt)
            own = who == spk
            want = level if own else level - float(rng.uniform(*LEAK_DB))
            gain = 10 ** ((want - active_level_db(x, utt.speech)) / 20)
            i0 = int(round(start * SAMPLE_RATE))
            seg = x[: max(0, min(x.size, n - i0))] * gain
            sig[i0:i0 + seg.size] += seg
            spans = [(start + s, start + e) for s, e in utt.speech]
            (target if own else crosstalk).extend(spans)
            placed.append({"utt": utt.uid, "text": utt.text, "start_s": round(start, 3),
                           "role": "own" if own else "leak", "level_dbfs": round(want, 2)})
        noise = make_noise(rng, noise_files, n) * 10 ** ((level - snr) / 20)
        mix = sig + noise
        clipped = float(np.mean(np.abs(mix) >= 1.0))
        pcm = np.clip(np.round(mix * 32767), -32768, 32767).astype(np.int16)
        lag = 0.0
        if process:
            out = opus_roundtrip(webrtc_ns(pcm))
            lag = chain_lag_s(pcm, out)
            pcm = out
        tid = f"a-m{m:02d}-{spk}"
        path = os.path.join(out_dir, f"{tid}.wav")
        sf.write(path, pcm, SAMPLE_RATE, subtype="PCM_16")
        tracks.append({
            "id": tid, "wav": path, "duration_s": len(pcm) / SAMPLE_RATE,
            "target": _shifted(target, lag), "crosstalk": _shifted(crosstalk, lag),
            "meta": {"set": "A", "meeting": f"m{m:02d}", "partner": f"a-m{m:02d}-{1 - spk}",
                     "level_dbfs": level, "snr_db": snr, "noise": noise_name,
                     "speech_share_planned": round(shares[spk], 3), "chain_lag_s": lag,
                     "clipped_share": clipped, "processed": process},
            "utterances": placed,
        })
    return tracks


def _shifted(intervals, lag: float) -> list[tuple[float, float]]:
    return [(round(s + lag, 3), round(e + lag, 3)) for s, e in merge_intervals(intervals)]


def noise_sources(demand_dir: str, musan_dir: str) -> list[tuple[str, list[str]]]:
    """Alternating DEMAND environments (channel 1) and the MUSAN noise pool."""
    demand = [(f"demand-{env}", [str(p)]) for env in DEMAND_ENVS
              for p in [Path(demand_dir) / env / "ch01.wav"] if p.exists()]
    musan = sorted(str(p) for p in Path(musan_dir).rglob("*.wav"))
    if not demand or not musan:
        raise FileNotFoundError("need DEMAND */ch01.wav and MUSAN noise/*.wav")
    out: list[tuple[str, list[str]]] = []
    for env in demand:
        out.append(env)
        out.append(("musan-noise", musan))
    return out


def build(corpus_dir: str, align_dir: str, demand_dir: str, musan_dir: str, out_dir: str,
          seed: int = 20261111, *, process: bool = True) -> dict:
    """Render all meetings and write ``<out>/manifest.json``."""
    os.makedirs(out_dir, exist_ok=True)
    utts = load_utterances(corpus_dir, align_dir)
    rng = np.random.default_rng(seed)
    pool = [utts[i] for i in rng.permutation(len(utts))]
    noises = noise_sources(demand_dir, musan_dir)
    tracks: list[dict] = []
    for m, cells in enumerate(meeting_grid()):
        tracks.extend(build_meeting(m, cells, pool, noises, out_dir, seed, process=process))
    manifest = {"set": "A", "seed": seed, "aligned_utterances": len(utts),
                "unused_utterances": len(pool), "recipe": {
                    "levels_dbfs": LEVELS_DBFS, "snrs_db": SNRS_DB, "leak_db": LEAK_DB,
                    "speech_share": SPEECH_SHARE, "overlap_p": OVERLAP_P, "track_s": TRACK_S,
                    "word_merge_gap_s": WORD_MERGE_GAP_S, "webrtc_ns_level": NS_LEVEL,
                    "opus_kbps": OPUS_KBPS if process else None},
                "tracks": tracks}
    Path(out_dir, "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def callhome_manifest(callhome_dir: str, out_path: str) -> dict:
    """Testset B: CALLHOME German mono mixes, speech = union of all gold RTTM turns."""
    import soundfile as sf

    tracks = []
    for wav in sorted(Path(callhome_dir, "audio").glob("*.wav")):
        rttm = Path(callhome_dir, "gold", f"{wav.stem}.rttm")
        turns = []
        for line in rttm.read_text().splitlines():
            f = line.split()
            if f and f[0] == "SPEAKER":
                turns.append((float(f[3]), float(f[3]) + float(f[4])))
        info = sf.info(str(wav))
        tracks.append({"id": wav.stem, "wav": str(wav), "duration_s": info.frames / info.samplerate,
                       "target": merge_intervals(turns), "crosstalk": None,
                       "meta": {"set": "B", "corpus": "callhome-de"}})
    manifest = {"set": "B", "tracks": tracks}
    Path(out_path).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
