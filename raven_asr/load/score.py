"""Long-form accuracy: stitch a recording's answers, align once, read it in windows.

Stitching. A recording's answered blocks are put in time order (block source
start) and their texts joined — what production stores as the track's text.
A block that failed after its retry contributes nothing: its speech counts as
deleted, exactly as a transcript with that hole would read.

Lens. The published one, flozi-strict (raven_eval_core.flozi_wer): each block's
text and the reference go through ``normalize_flozi`` and jiwer's
``wer_standardize``; the recording is then aligned ONCE, reference against the
stitched hypothesis (jiwer.process_words; ~100k words align in about a second).
Per-recording WER = edits / reference words of that one alignment. Parliament
protocols are edited, not verbatim, so the absolute WER has a floor well above
zero; compare models with each other, paired.

Windows. The reference is cut into consecutive windows of ``WINDOW_REF_WORDS``
words (~5 min of speech at parliament pace). Every edit belongs to one window —
substitutions and deletions by their reference word, insertions by the reference
word they precede — so windows partition the recording's edits, and the same
partition holds for every model (it is defined on the reference only). That
makes windows the unit of the paired bootstrap (raven_eval_core.bootstrap)
between two models on one recording or on the corpus. Each window carries the
time span of the blocks its hypothesis words came from.

Loops. :func:`loop_spans` finds an n-gram repeated back to back (n ≤ 10) at
least 3 times covering at least 12 words — "sehr sehr sehr …" twelve times, or a
four-word phrase three times running. Run on every answer's own words and, as a
baseline for how often edited speech itself trips the rule, on every reference
window. A window whose hypothesis is more than 1.5x (runaway) or less than 0.5x
(dropout) its reference length is flagged too.

Runaway. Each answer gets the reference words aligned to its own words (plus
the deletions between them) and raven_eval_core.flozi_wer.runaway_stats — the
leaderboard's rule, hypothesis words > 3x reference words — counts the answers
that ran away and their share of the recording's edits. An answer with words
but no reference word at all (text on applause or silence) is a runaway by that
rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jiwer
from jiwer.transformations import wer_standardize

from raven_eval_core.bootstrap import bootstrap_ratio_ci, paired_bootstrap_ratio_delta
from raven_eval_core.flozi_wer import normalize_flozi, runaway_stats

WINDOW_REF_WORDS = 600
LOOP_MAX_N = 10
LOOP_MIN_REPS = 3
LOOP_MIN_WORDS = 12
RUNAWAY_WINDOW_RATIO = 1.5
DROPOUT_WINDOW_RATIO = 0.5
RESAMPLES = 2000
SEED = 20261110
CONFIDENCE = 0.95


def tokens(text: str) -> list[str]:
    """Words under the published lens: normalize_flozi, then jiwer's wer_standardize."""
    norm = normalize_flozi(text or "")
    if not norm.strip():
        return []
    out = wer_standardize(norm)
    return [w for sent in out for w in sent]


@dataclass(frozen=True)
class Op:
    """One aligned position: kind (equal|substitute|delete|insert), ref index, hyp index."""

    kind: str
    ref: int | None
    hyp: int | None


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[Op]:
    """Word-level alignment of two token lists, one Op per position."""
    if not ref:
        return [Op("insert", None, j) for j in range(len(hyp))]
    if not hyp:
        return [Op("delete", i, None) for i in range(len(ref))]
    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    ops: list[Op] = []
    for ch in out.alignments[0]:
        if ch.type in ("equal", "substitute"):
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                ops.append(Op(ch.type, ch.ref_start_idx + k, ch.hyp_start_idx + k))
        elif ch.type == "delete":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                ops.append(Op("delete", ch.ref_start_idx + k, None))
        else:
            for k in range(ch.hyp_end_idx - ch.hyp_start_idx):
                ops.append(Op("insert", None, ch.hyp_start_idx + k))
    return ops


def edits(ops: Sequence[Op]) -> dict[str, int]:
    c = {"substitute": 0, "delete": 0, "insert": 0, "equal": 0}
    for o in ops:
        c[o.kind] += 1
    return c


def anchor_ref(ops: Sequence[Op], n_ref: int) -> list[int]:
    """The reference index each op belongs to (an insertion: the next reference word)."""
    out = [0] * len(ops)
    nxt = n_ref - 1 if n_ref else 0
    for k in range(len(ops) - 1, -1, -1):
        if ops[k].ref is not None:
            nxt = ops[k].ref  # type: ignore[assignment]
        out[k] = nxt
    return out


def loop_spans(words: Sequence[str], *, max_n: int = LOOP_MAX_N, min_reps: int = LOOP_MIN_REPS,
               min_words: int = LOOP_MIN_WORDS) -> list[tuple[int, int, int]]:
    """``[(start, n, reps), ...]``: n-grams repeated back to back past the loop rule.

    Scans left to right; at each position the shortest n whose back-to-back
    repetition satisfies ``reps ≥ min_reps`` and ``n·reps ≥ min_words`` is taken
    and the scan jumps past it, so one loop is reported once.
    """
    spans: list[tuple[int, int, int]] = []
    i, L = 0, len(words)
    while i < L:
        hit = None
        for n in range(1, max_n + 1):
            if i + 2 * n > L:
                break
            gram = list(words[i:i + n])
            reps = 1
            while i + (reps + 1) * n <= L and list(words[i + reps * n:i + (reps + 1) * n]) == gram:
                reps += 1
            if reps >= min_reps and n * reps >= min_words:
                hit = (i, n, reps)
                break
        if hit:
            spans.append(hit)
            i += hit[1] * hit[2]
        else:
            i += 1
    return spans


@dataclass
class Answer:
    """One block's answer placed on the recording: order key, time span, text."""

    index: int
    source_start: float
    source_end: float
    text: str | None  # None: the block failed after its retry


def score_recording(reference: str, answers: Sequence[Answer], *,
                    window_words: int = WINDOW_REF_WORDS) -> dict:
    """Every accuracy number of one recording from its reference and its answers."""
    answers = sorted(answers, key=lambda a: (a.source_start, a.index))
    ref = tokens(reference)
    hyp: list[str] = []
    owner: list[int] = []  # hyp token -> position in ``answers``
    per_answer: list[list[str]] = []
    for k, a in enumerate(answers):
        t = tokens(a.text or "")
        per_answer.append(t)
        hyp.extend(t)
        owner.extend([k] * len(t))
    ops = align(ref, hyp)
    c = edits(ops)
    n_ref = len(ref)
    errs = c["substitute"] + c["delete"] + c["insert"]
    anchors = anchor_ref(ops, n_ref)

    # Windows over the reference.
    n_win = max(1, -(-n_ref // window_words))
    win = [{"i": w, "ref_words": 0, "substitute": 0, "delete": 0, "insert": 0, "hyp_words": 0,
            "answers": set()} for w in range(n_win)]
    for o, r in zip(ops, anchors, strict=True):
        w = min(r // window_words, n_win - 1)
        if o.ref is not None:
            win[w]["ref_words"] += 1
        if o.kind != "equal":
            win[w][o.kind] += 1
        if o.hyp is not None:
            win[w]["hyp_words"] += 1
            win[w]["answers"].add(owner[o.hyp])
    windows = []
    for w in win:
        e = w["substitute"] + w["delete"] + w["insert"]
        ans = sorted(w.pop("answers"))
        ratio = w["hyp_words"] / w["ref_words"] if w["ref_words"] else None
        ref_words_w = ref[w["i"] * window_words:(w["i"] + 1) * window_words]
        windows.append({
            **w, "errors": e,
            "wer_pct": round(100.0 * e / w["ref_words"], 2) if w["ref_words"] else None,
            "len_ratio": None if ratio is None else round(ratio, 3),
            "runaway": ratio is not None and ratio > RUNAWAY_WINDOW_RATIO,
            "dropout": ratio is not None and ratio < DROPOUT_WINDOW_RATIO,
            "t_start": round(answers[ans[0]].source_start, 1) if ans else None,
            "t_end": round(answers[ans[-1]].source_end, 1) if ans else None,
            "ref_loops": len(loop_spans(ref_words_w)),
        })

    # Reference words aligned to each answer, for the runaway rule.
    first = [None] * len(answers)
    last = [None] * len(answers)
    for o in ops:
        if o.hyp is not None and o.ref is not None:
            k = owner[o.hyp]
            first[k] = o.ref if first[k] is None else first[k]
            last[k] = o.ref
    a_refs, a_hyps = [], []
    loops = []
    for k, a in enumerate(answers):
        if a.text is None:
            continue
        r = ref[first[k]:last[k] + 1] if first[k] is not None else []
        a_refs.append(" ".join(r))
        a_hyps.append(" ".join(per_answer[k]))
        for s, n, reps in loop_spans(per_answer[k]):
            loops.append({"answer": a.index, "t": round(a.source_start, 1), "n": n, "reps": reps,
                          "gram": " ".join(per_answer[k][s:s + n])})
    run = runaway_stats(a_refs, a_hyps) if a_refs else None
    units = [(float(w["errors"]), float(w["ref_words"])) for w in windows]
    ci = bootstrap_ratio_ci(units, resamples=RESAMPLES, seed=SEED, confidence=CONFIDENCE)
    return {
        "ref_words": n_ref,
        "hyp_words": len(hyp),
        "answers": len(answers),
        "failed_answers": sum(1 for a in answers if a.text is None),
        "empty_answers": sum(1 for k, a in enumerate(answers)
                             if a.text is not None and not per_answer[k]),
        "substitutions": c["substitute"],
        "deletions": c["delete"],
        "insertions": c["insert"],
        "errors": errs,
        "wer_pct": round(100.0 * errs / n_ref, 3) if n_ref else None,
        "wer_ci95": [round(ci.lo, 3), round(ci.hi, 3)],
        "insertion_rate_pct": round(100.0 * c["insert"] / n_ref, 3) if n_ref else None,
        "deletion_rate_pct": round(100.0 * c["delete"] / n_ref, 3) if n_ref else None,
        "hyp_over_ref": round(len(hyp) / n_ref, 4) if n_ref else None,
        "window_ref_words": window_words,
        "windows": windows,
        "runaway_windows": sum(1 for w in windows if w["runaway"]),
        "dropout_windows": sum(1 for w in windows if w["dropout"]),
        "worst_window_wer_pct": max((w["wer_pct"] for w in windows if w["wer_pct"] is not None),
                                    default=None),
        "loops": loops,
        "loop_count": len(loops),
        "ref_loop_count": sum(w["ref_loops"] for w in windows),
        "runaway_answers": None if run is None else {
            "n": run.n_runaway, "of": run.n_utterances, "pct": round(run.runaway_pct, 3),
            "edit_share_pct": None if run.edit_share_pct is None else round(run.edit_share_pct, 3)},
    }


def window_units(scores: dict[str, dict]) -> dict[str, tuple[float, float]]:
    """``{"<recording>/<window>": (errors, ref words)}`` for the bootstrap."""
    return {f"{rid}/{w['i']}": (float(w["errors"]), float(w["ref_words"]))
            for rid, s in scores.items() for w in s["windows"]}


def corpus_interval(scores: dict[str, dict]) -> dict:
    units = list(window_units(scores).values())
    ci = bootstrap_ratio_ci(units, resamples=RESAMPLES, seed=SEED, confidence=CONFIDENCE)
    return {"wer_pct": round(ci.point, 3), "ci95": [round(ci.lo, 3), round(ci.hi, 3)],
            "windows": ci.n}


def paired_delta(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """WER(a) − WER(b) in points with its paired 95 % interval over shared windows."""
    ua, ub = window_units(a), window_units(b)
    keys = [k for k in ua if k in ub]
    if len(keys) != len(ua) or len(keys) != len(ub):
        raise ValueError("the two runs were not scored on the same recordings")
    d = paired_bootstrap_ratio_delta([(ua[k], ub[k]) for k in keys], resamples=RESAMPLES,
                                     seed=SEED, confidence=CONFIDENCE)
    return {"delta_pp": round(d.point, 3), "ci95": [round(d.lo, 3), round(d.hi, 3)],
            "windows": d.n}
