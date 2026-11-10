"""How well a model's word times land: onsets against timed subtitle cues.

For a recording with a WebVTT file (the Bundestag's own subtitles), every cue
says when its first word was spoken, to subtitle precision. A model's words are
put on the recording's timeline — block time through production's piece map
(raven_asr.load.vad.remap_word) — and aligned to the cue text under the
published lens; wherever a cue's first word is matched exactly by a model word,
``model word start − cue start`` is one onset offset. The distribution of those
offsets (median, spread, share within ±0.5/1/2 s) is the coarse timing check.
It measures the model against the subtitler, not against ground truth: the
subtitle timing has its own error, which is the same for every model, so the
comparison between models holds while the absolute offset is only indicative.

No forced alignment is involved: only the times the server itself returned.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..capacity import percentile
from .score import align, tokens

_TS = re.compile(r"(\d+):(\d\d):(\d\d)[.,](\d{3})\s*-->\s*(\d+):(\d\d):(\d\d)[.,](\d{3})")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def parse_vtt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _TS.search(lines[i])
        if not m:
            i += 1
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        body = []
        i += 1
        while i < len(lines) and lines[i].strip():
            body.append(lines[i].strip())
            i += 1
        cues.append(Cue(start, end, " ".join(body)))
    return cues


def onset_offsets(cues: Sequence[Cue], words: Sequence[dict]) -> list[float]:
    """``model start − cue start`` for every cue whose first word the model matched.

    ``words`` are ``{"word", "start", "end"}`` on the recording's timeline, in order.
    """
    ref: list[str] = []
    cue_first: list[int] = []
    for c in cues:
        t = tokens(c.text)
        if t:
            cue_first.append(len(ref))
            ref.extend(t)
        else:
            cue_first.append(-1)
    hyp: list[str] = []
    hyp_t: list[float] = []
    for w in words:
        for tok in tokens(str(w["word"])):
            hyp.append(tok)
            hyp_t.append(float(w["start"]))
    ref_to_hyp: dict[int, int] = {}
    for o in align(ref, hyp):
        if o.kind == "equal" and o.ref is not None and o.hyp is not None:
            ref_to_hyp[o.ref] = o.hyp
    out = []
    for c, r in zip(cues, cue_first, strict=True):
        if r >= 0 and r in ref_to_hyp:
            out.append(hyp_t[ref_to_hyp[r]] - c.start)
    return out


def summarize(offsets: Sequence[float], n_cues: int) -> dict:
    if not offsets:
        return {"cues": n_cues, "matched": 0}
    absd = [abs(x) for x in offsets]

    def share(t: float) -> float:
        return round(100.0 * sum(1 for x in absd if x <= t) / len(absd), 1)

    return {
        "cues": n_cues,
        "matched": len(offsets),
        "median_s": round(percentile(list(offsets), 50) or 0.0, 3),
        "p10_s": round(percentile(list(offsets), 10) or 0.0, 3),
        "p90_s": round(percentile(list(offsets), 90) or 0.0, 3),
        "abs_median_s": round(percentile(absd, 50) or 0.0, 3),
        "abs_p90_s": round(percentile(absd, 90) or 0.0, 3),
        "within_0_5s_pct": share(0.5),
        "within_1s_pct": share(1.0),
        "within_2s_pct": share(2.0),
    }


def vtt_check(vtt: Path, words: Sequence[dict]) -> dict:
    cues = parse_vtt(vtt.read_text(encoding="utf-8"))
    return summarize(onset_offsets(cues, words), len(cues))
