"""Cross-utterance leakage: transcripts that carry text of OTHER clips of the corpus.

A model cannot know the wording of a clip it was never sent. When a transcript
holds a run of words that occurs in another utterance's reference of the same
corpus but not in its own, the most likely source is the serving stack, not the
model: audio or decoder state of one request reaching another (raven.eval#89 —
Voxtral-Mini-4B-Realtime on vLLM 0.33.0 continued each transcript with the audio
of the next request in the batch). Such a run must never be scored silently, for
any model, so the harness checks every finished subset with this.

The rule, per utterance: tokenize reference and hypothesis the way the published
WER does (flozi-strict normalization, lowercased); the utterance is *leaked* when
its hypothesis contains a run of :data:`LEAK_NGRAM` consecutive tokens that is
part of at least one other reference of the subset and not of its own.

A clean model is not at zero. Corpora repeat themselves — FLEURS reads the same
FLoRes sentence by several speakers, read-speech corpora reuse prompts, a set's
references disagree on spelling — so a correct transcript can match a near
duplicate's 6-gram where its own reference spells it differently. Calibrated
2026-11-08 on every finished predictions file of round 4 on the second bench
host (8 models, 8 corpora) and every committed artifact: clean runs flag 0-22
utterances, at most 1.12 % of a set; the leaked Voxtral run of #89 flagged 473
of 646 (73 %).

The check is a gate, not a metric: it does not change any number, it decides
whether a subset may be recorded as complete at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .flozi_wer import normalize_flozi

#: Length of the word run that counts as copied text. Six German words in the
#: same order are specific enough that a chance match across different sentences
#: is rare, and short enough to catch a single leaked clause.
LEAK_NGRAM = 6

#: A subset fails when more than this share of its utterances is leaked.
#: Calibration (module docstring): the highest clean share was 1.12 % (22 of
#: 1968, Whisper-large-v3-turbo on VoxPopuli), most sets flag none; the leaked
#: Voxtral-Mini-4B-Realtime run (raven.eval#89) was 73 %, and 40-55 % on a
#: 20-clip FLEURS smoke at concurrency 2-8.
LEAK_MAX_SHARE_PCT = 2.0

#: ... and more than this many utterances are leaked. A small subset (a smoke
#: run of 20 clips, a 30-clip control set) must not fail on one or two natural
#: duplicates that are a large share only because the denominator is small.
LEAK_MIN_FLAGGED = 5


@dataclass(frozen=True)
class LeakResult:
    """Leaked utterances of one subset and whether the subset passes.

    ``flagged``      indices (into the given lists) of leaked utterances
    ``sources``      per flagged index, the index of one other utterance whose
                     reference holds the leaked run (for the error message)
    ``n_utterances`` utterances checked
    """

    flagged: tuple[int, ...]
    sources: tuple[int, ...]
    n_utterances: int

    @property
    def n_flagged(self) -> int:
        return len(self.flagged)

    @property
    def share_pct(self) -> float:
        return 0.0 if not self.n_utterances else 100.0 * self.n_flagged / self.n_utterances

    @property
    def failed(self) -> bool:
        return self.n_flagged > LEAK_MIN_FLAGGED and self.share_pct > LEAK_MAX_SHARE_PCT


def _tokens(text: str | None) -> list[str]:
    return normalize_flozi(text or "").lower().split()


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[k : k + n]) for k in range(len(tokens) - n + 1)}


def cross_utterance_leaks(
    references: Sequence[str],
    predictions: Sequence[str | None],
    n: int = LEAK_NGRAM,
) -> LeakResult:
    """Utterances whose hypothesis holds an ``n``-word run of another reference only.

    ``predictions[i]`` may be None (a failed request): it is never flagged.
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"references ({len(references)}) and predictions ({len(predictions)}) differ in length"
        )
    own = [_ngrams(_tokens(r), n) for r in references]
    # n-gram -> the first two utterances whose reference holds it: enough to
    # tell "only my own reference" from "also another one".
    holders: dict[tuple[str, ...], list[int]] = {}
    for i, grams in enumerate(own):
        for g in grams:
            seen = holders.setdefault(g, [])
            if len(seen) < 2:
                seen.append(i)
    flagged: list[int] = []
    sources: list[int] = []
    for i, pred in enumerate(predictions):
        if pred is None:
            continue
        hyp = _tokens(pred)
        for k in range(len(hyp) - n + 1):
            g = tuple(hyp[k : k + n])
            if g in own[i]:
                continue
            other = next((j for j in holders.get(g, ()) if j != i), None)
            if other is not None:
                flagged.append(i)
                sources.append(other)
                break
    return LeakResult(tuple(flagged), tuple(sources), len(references))
