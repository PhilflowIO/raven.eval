"""Words lost after transcription: an arm's islands as Parakeet blocks, deletions against the owner's text.

For each Testset A track and arm, the islands at the arm's production setting
become production's blocks (``plan_blocks``, ≤ 30 s of speech, spliced from the
source samples), are sent to a Parakeet server, and the block texts in track
order form the hypothesis. The reference is the text of the owner's own
utterances in start order. Deletions are reference words the hypothesis lacks —
words lost because the detector never sent them (or the model dropped them).
Leaked partner words that were sent show up as insertions; those that match
the partner text leaked into this track are counted as foreign words. Arm
``oracle`` sends the labelled own speech through the same planner: the
ASR-only floor. ``<arm>+xt`` is the arm with the cross-track crosstalk flag
on (:func:`raven_asr.vad_ab.report.crosstalk_islands`, partners from the same
meeting) — the flag's word effect: own words it costs, foreign words it removes.

    python -m raven_asr.vad_ab.deletions --manifest A/manifest.json --results R \
        --wav-dir A --arms oracle,0-silencedetect,3-silero-0.3-norm --url http://…/v1/audio/transcriptions --out D.json
    python -m raven_asr.vad_ab.deletions … --arms 3-silero-0.3-norm,3-silero-0.3-norm+xt --out X.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np

from raven_asr.load import vad as planner
from raven_asr.load.client import ASK_JSON, transcribe
from raven_asr.load.score import align, edits, tokens
from raven_asr.load.workload import wav_header

from .arms import ARMS, read_pcm16
from .frames import HOP, n_frames, to_grid

#: Arm-name suffix for "this arm with the cross-track crosstalk flag on".
XT_SUFFIX = "+xt"


def track_islands(track: dict, results: str, arm: str) -> list[tuple[float, float]]:
    if arm == "oracle":
        n = n_frames(track["duration_s"])
        mask = to_grid(track["target"], n)
        return planner.build_speech_islands(planner.silences_from_mask(mask, HOP), track["duration_s"])
    d = json.loads(Path(results, arm, f"{track['id']}.json").read_text())
    return [tuple(x) for x in d["islands"][repr(float(ARMS[arm].default))]]


def flagged_islands(tracks: list[dict], results: str, arm: str, wav_dir: str) -> dict[str, list]:
    """``<arm>+xt``: per track, the base arm's islands at its default with the crosstalk veto applied."""
    from .report import crosstalk_islands

    base = arm.removesuffix(XT_SUFFIX)
    isl = crosstalk_islands(tracks, results, base,
                            read=lambda t: read_pcm16(str(Path(wav_dir, Path(t["wav"]).name))))
    return {tid: x["on"] for tid, x in isl.items()}


def block_wavs(pcm: np.ndarray, islands) -> list[tuple[bytes, float]]:
    out = []
    for planned in planner.plan_blocks(islands):
        ranges, _ = planner.block_pieces(planned, len(pcm))
        if not ranges:
            continue
        audio = np.concatenate([pcm[a:b] for a, b in ranges]).astype("<i2")
        out.append((wav_header(len(audio)) + audio.tobytes(), len(audio) / planner.SAMPLE_RATE))
    return out


def reference_text(track: dict) -> str:
    own = sorted((u for u in track["utterances"] if u["role"] == "own"), key=lambda u: u["start_s"])
    return " ".join(u["text"] for u in own)


def leaked_text(track: dict) -> str:
    """The partner's words leaked into this track (role ``leak``), in start order."""
    leak = sorted((u for u in track["utterances"] if u["role"] == "leak"), key=lambda u: u["start_s"])
    return " ".join(u["text"] for u in leak)


def foreign_insertions(ops: Sequence, hyp: Sequence[str], leaked: Sequence[str]) -> int:
    """Inserted hypothesis words that are the partner's: matched against the leaked words as a multiset.

    Bag-of-words, not an alignment: a leaked word used once can claim one
    insertion. An inserted word the partner never said (an ASR artefact) is not
    foreign; a common word ("und") the partner did say counts — at most as
    often as it leaked.
    """
    pool = Counter(leaked)
    hits = 0
    for o in ops:
        if o.kind == "insert" and pool[hyp[o.hyp]] > 0:
            pool[hyp[o.hyp]] -= 1
            hits += 1
    return hits


def summarise_words(per_track: dict) -> dict:
    """Totals, rates against the owner's reference words, and per owner level."""
    ref = sum(v["ref_words"] for v in per_track.values())
    keys = ("delete", "insert", "foreign_insert", "substitute", "equal", "blocks", "failed_blocks", "sent_s")
    tot = {k: sum(v[k] for v in per_track.values()) for k in keys}
    by_level: dict = {}
    for v in per_track.values():
        g = by_level.setdefault(str(v["level_dbfs"]), {"ref": 0, "delete": 0, "insert": 0, "foreign_insert": 0})
        for k in ("delete", "insert", "foreign_insert"):
            g[k] += v[k]
        g["ref"] += v["ref_words"]

    def pct(x: float, d: float) -> float:
        return round(100 * x / d, 2) if d else 0.0

    levels = sorted(by_level.items(), key=lambda kv: -float(kv[0]))
    return {
        "ref_words": ref, **tot,
        "deletion_pct": pct(tot["delete"], ref),
        "insertion_pct": pct(tot["insert"], ref),
        "foreign_insertion_pct": pct(tot["foreign_insert"], ref),
        "wer_pct": pct(tot["delete"] + tot["insert"] + tot["substitute"], ref),
        "deletion_pct_by_level": {k: pct(g["delete"], g["ref"]) for k, g in levels},
        "insertion_pct_by_level": {k: pct(g["insert"], g["ref"]) for k, g in levels},
        "foreign_insertion_pct_by_level": {k: pct(g["foreign_insert"], g["ref"]) for k, g in levels},
        "per_track": per_track,
    }


def plan(tracks: list[dict], a: argparse.Namespace) -> dict:
    """No server: per arm the blocks and seconds a run would send (readiness check, GPU-time estimate)."""
    out: dict = {"arms": {}}
    for arm in a.arms.split(","):
        flagged = flagged_islands(tracks, a.results, arm, a.wav_dir) if arm.endswith(XT_SUFFIX) else None
        blocks = sent = done = 0
        for t in tracks:
            if flagged is not None and t["id"] not in flagged:
                continue
            islands = flagged[t["id"]] if flagged is not None else track_islands(t, a.results, arm)
            b = block_wavs(read_pcm16(str(Path(a.wav_dir, Path(t["wav"]).name))), islands)
            blocks += len(b)
            sent += sum(s for _, s in b)
            done += 1
        out["arms"][arm] = {"tracks": done, "blocks": blocks, "sent_s": round(sent, 1)}
        print(f"{arm}: {done} tracks, {blocks} blocks, {sent:.0f} s to send", flush=True)
    return out


async def run(a: argparse.Namespace) -> dict:
    manifest = json.loads(Path(a.manifest).read_text())
    tracks = manifest["tracks"]
    sem = asyncio.Semaphore(a.concurrency)
    out: dict = {"url": a.url, "arms": {}}
    if a.plan_only:
        return plan(tracks, a)
    async with httpx.AsyncClient() as client:
        for arm in a.arms.split(","):
            per_track = {}
            flagged = (flagged_islands(tracks, a.results, arm, a.wav_dir)
                       if arm.endswith(XT_SUFFIX) else None)

            async def one(t: dict, arm: str = arm, flagged: dict | None = flagged,
                          per_track: dict = per_track) -> None:
                if flagged is not None and t["id"] not in flagged:
                    return
                pcm = read_pcm16(str(Path(a.wav_dir, Path(t["wav"]).name)))
                islands = flagged[t["id"]] if flagged is not None else track_islands(t, a.results, arm)
                blocks = block_wavs(pcm, islands)

                async def send(i: int, wav: bytes, sec: float):
                    async with sem:
                        return await transcribe(client, a.url, a.model, f"{t['id']}-{i}", wav, sec,
                                                ASK_JSON, retries=1)

                outs = await asyncio.gather(*(send(i, w, s) for i, (w, s) in enumerate(blocks)))
                hyp = " ".join(o.text or "" for o in outs)
                ref_t, hyp_t = tokens(reference_text(t)), tokens(hyp)
                ops = align(ref_t, hyp_t)
                e = edits(ops)
                per_track[t["id"]] = {"ref_words": len(ref_t), "hyp_words": len(hyp_t), **e,
                                      "foreign_insert": foreign_insertions(ops, hyp_t, tokens(leaked_text(t))),
                                      "blocks": len(blocks), "sent_s": round(sum(s for _, s in blocks), 1),
                                      "failed_blocks": sum(1 for o in outs if o.status != "ok"),
                                      "level_dbfs": t["meta"]["level_dbfs"], "snr_db": t["meta"]["snr_db"]}

            await asyncio.gather(*(one(t) for t in tracks))
            out["arms"][arm] = summarise_words(per_track)
            r = out["arms"][arm]
            print(f"{arm}: D {r['deletion_pct']} %  I {r['insertion_pct']} %  "
                  f"foreign {r['foreign_insertion_pct']} %  WER {r['wer_pct']} %  failed {r['failed_blocks']}\n"
                  f"  D by level {r['deletion_pct_by_level']}\n"
                  f"  foreign by level {r['foreign_insertion_pct_by_level']}", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m raven_asr.vad_ab.deletions")
    p.add_argument("--manifest", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--wav-dir", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--url", default="")
    p.add_argument("--model", default="parakeet")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", default="")
    p.add_argument("--plan-only", action="store_true",
                   help="no server: print blocks and seconds each arm would send")
    a = p.parse_args(argv)
    if not a.plan_only and not (a.out and a.url):
        p.error("--url and --out are required unless --plan-only")
    asyncio.run(run(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
