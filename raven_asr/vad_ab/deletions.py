"""Words lost after transcription: an arm's islands as Parakeet blocks, deletions against the owner's text.

For each Testset A track and arm, the islands at the arm's production setting
become production's blocks (``plan_blocks``, ≤ 30 s of speech, spliced from the
source samples), are sent to a Parakeet server, and the block texts in track
order form the hypothesis. The reference is the text of the owner's own
utterances in start order. Deletions are reference words the hypothesis lacks —
words lost because the detector never sent them (or the model dropped them).
Leaked partner words that were sent show up as insertions. Arm ``oracle``
sends the labelled own speech through the same planner: the ASR-only floor.

    python -m raven_asr.vad_ab.deletions --manifest A/manifest.json --results R \
        --wav-dir A --arms oracle,0-silencedetect,3-silero-0.3-norm --url http://…/v1/audio/transcriptions --out D.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import numpy as np

from raven_asr.load import vad as planner
from raven_asr.load.client import ASK_JSON, transcribe
from raven_asr.load.score import align, edits, tokens
from raven_asr.load.workload import wav_header

from .arms import ARMS, read_pcm16
from .frames import HOP, n_frames, to_grid


def track_islands(track: dict, results: str, arm: str) -> list[tuple[float, float]]:
    if arm == "oracle":
        n = n_frames(track["duration_s"])
        mask = to_grid(track["target"], n)
        return planner.build_speech_islands(planner.silences_from_mask(mask, HOP), track["duration_s"])
    d = json.loads(Path(results, arm, f"{track['id']}.json").read_text())
    return [tuple(x) for x in d["islands"][repr(float(ARMS[arm].default))]]


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


async def run(a: argparse.Namespace) -> dict:
    manifest = json.loads(Path(a.manifest).read_text())
    tracks = manifest["tracks"]
    sem = asyncio.Semaphore(a.concurrency)
    out: dict = {"url": a.url, "arms": {}}
    async with httpx.AsyncClient() as client:
        for arm in a.arms.split(","):
            per_track = {}

            async def one(t: dict, arm: str = arm) -> None:
                pcm = read_pcm16(str(Path(a.wav_dir, Path(t["wav"]).name)))
                blocks = block_wavs(pcm, track_islands(t, a.results, arm))

                async def send(i: int, wav: bytes, sec: float):
                    async with sem:
                        return await transcribe(client, a.url, a.model, f"{t['id']}-{i}", wav, sec,
                                                ASK_JSON, retries=1)

                outs = await asyncio.gather(*(send(i, w, s) for i, (w, s) in enumerate(blocks)))
                hyp = " ".join(o.text or "" for o in outs)
                ref_t, hyp_t = tokens(reference_text(t)), tokens(hyp)
                e = edits(align(ref_t, hyp_t))
                per_track[t["id"]] = {"ref_words": len(ref_t), "hyp_words": len(hyp_t), **e,
                                      "blocks": len(blocks), "sent_s": round(sum(s for _, s in blocks), 1),
                                      "failed_blocks": sum(1 for o in outs if o.status != "ok"),
                                      "level_dbfs": t["meta"]["level_dbfs"], "snr_db": t["meta"]["snr_db"]}

            await asyncio.gather(*(one(t) for t in tracks))
            ref = sum(v["ref_words"] for v in per_track.values())
            tot = {k: sum(v[k] for v in per_track.values())
                   for k in ("delete", "insert", "substitute", "equal", "blocks", "failed_blocks", "sent_s")}
            by_level: dict = {}
            for v in per_track.values():
                g = by_level.setdefault(str(v["level_dbfs"]), {"ref": 0, "delete": 0})
                g["ref"] += v["ref_words"]
                g["delete"] += v["delete"]
            out["arms"][arm] = {
                "ref_words": ref, **tot,
                "deletion_pct": round(100 * tot["delete"] / ref, 2),
                "insertion_pct": round(100 * tot["insert"] / ref, 2),
                "wer_pct": round(100 * (tot["delete"] + tot["insert"] + tot["substitute"]) / ref, 2),
                "deletion_pct_by_level": {k: round(100 * g["delete"] / g["ref"], 2) for k, g in sorted(by_level.items())},
                "per_track": per_track,
            }
            r = out["arms"][arm]
            print(f"{arm}: D {r['deletion_pct']} %  I {r['insertion_pct']} %  WER {r['wer_pct']} %  "
                  f"failed {r['failed_blocks']}  by level {r['deletion_pct_by_level']}", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m raven_asr.vad_ab.deletions")
    p.add_argument("--manifest", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--wav-dir", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="parakeet")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", required=True)
    asyncio.run(run(p.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
