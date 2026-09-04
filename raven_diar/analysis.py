"""Instruments that read a committed DER artifact and answer what one DER cannot.

A corpus DER is a single scalar over a whole dataset. Four questions that decide
how it may be *read* are unanswerable from it, and each one had a claim in
``BENCHMARKS.md`` standing on it before this module existed:

``bootstrap_der_ci``
    How precise is it? A 0.05 pp agreement with a vendor's published figure means
    nothing if resampling the same 16 meetings moves the number by 2 pp. Files
    are the sampling unit; each resample re-aggregates ``Σerr/Σtotal``, never a
    mean of file DERs, so the interval is around the same estimator we publish.

``paired_bootstrap_delta``
    Is a *gap* between two models real? Resamples the two models on the **same**
    files jointly, so per-file difficulty cancels — the honest test behind any
    "X is ahead of Y" sentence, and the only thing that licenses the word
    "systematic" under ADR-app-0036.

``der_by_speaker_count``
    Where does it fail? Diarizers degrade with speaker count, and a corpus number
    hides that: the ETH benchmark reports Sortformer jumping from ~13 % at ≤4
    speakers to ~23 % at 5+. The reference speaker count is already in every
    committed gold RTTM.

``reference_overlap`` / ``boundary_report``
    What kind of error is it? An overlap fraction characterises the corpus; a
    boundary-offset distribution and a miss-by-segment-length split test whether
    a miss-dominated row really means "the vendor overhears short backchannels"
    or whether the missed speech sits in long segments the ASR simply dropped.

Everything here is derived from the committed ``gold``/``hyp`` RTTMs with the
same core scorer the published numbers use — no GPU, no network, no second
implementation of DER. The bootstrap is seeded, so a published interval is
reproducible to the digit.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from raven_eval_core.der import DiarSegment, load_rttm

from .config import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COLLARS,
)
from .adapters.aggregate import DEFAULT_GAP_MERGE_S, spans_to_turns
from .score import FileScore, score_rttm_pairs, score_segment_pairs

#: Resamples / seed / level for every published interval, from the scoring
#: contract (``benchmark.config.yaml`` → ``der.uncertainty``, mirrored in
#: ``raven_diar.config``) rather than chosen here: an interval is a published
#: quantity and its settings belong in the contract, like the collar.
DEFAULT_RESAMPLES: int = BOOTSTRAP_RESAMPLES
DEFAULT_SEED: int = BOOTSTRAP_SEED

#: Reference segments shorter than this are the backchannel/interjection regime
#: — the shape a transcript-derived diarizer is accused of dropping (#5460).
SHORT_SEGMENT_S: float = 0.5


# ── confidence intervals (#5452) ──────────────────────────────────────────────


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap interval, all in percent."""

    point: float
    lo: float
    hi: float
    n: int
    resamples: int
    seed: int

    @property
    def half_width(self) -> float:
        """Half the interval width — the ± a headline number should carry."""
        return (self.hi - self.lo) / 2.0

    def format(self) -> str:
        return (f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] "
                f"(±{self.half_width:.3f}, n={self.n})")


def _collar_fields(collar_name: str) -> tuple[str, str, str, str]:
    """Field names on :class:`FileScore` carrying the seconds for one collar."""
    if collar_name not in COLLARS:
        raise KeyError(f"unknown collar variant {collar_name!r}; have {sorted(COLLARS)}")
    suffix = "full" if collar_name == "full" else "classic"
    return (f"miss_{suffix}_s", f"fa_{suffix}_s", f"conf_{suffix}_s",
            f"total_{suffix}_s")


def _err_total(rows: Sequence[FileScore], collar_name: str) -> tuple[float, float]:
    """(Σ error seconds, Σ scored reference seconds) over ``rows``."""
    miss_f, fa_f, conf_f, total_f = _collar_fields(collar_name)
    err = sum(getattr(r, miss_f) + getattr(r, fa_f) + getattr(r, conf_f)
              for r in rows)
    total = sum(getattr(r, total_f) for r in rows)
    return err, total


def aggregate_der(rows: Sequence[FileScore], collar_name: str = "classic") -> float:
    """NIST-correct corpus DER (percent) over an arbitrary subset of file rows.

    The one aggregation used everywhere below — a bucket, a bootstrap resample
    and the published corpus figure are all ``Σerr/Σtotal`` over their rows, so a
    bucket number and the headline number are the same kind of quantity.
    """
    err, total = _err_total(rows, collar_name)
    if total <= 0.0:
        return 0.0
    return err / total * 100.0


def bootstrap_der_ci(
    rows: Sequence[FileScore],
    collar_name: str = "classic",
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Interval:
    """Percentile bootstrap over **files** for one model/dataset DER.

    The file is the sampling unit because it is the unit the corpus was drawn in:
    CALLHOME-de is 120 independent telephone calls, AMI test is 16 meetings. The
    interval answers "if we had drawn 16 other AMI meetings, how different could
    this number have been" — which is exactly the question a 0.05 pp agreement
    with a vendor's published figure invites and cannot answer.

    Each resample re-aggregates ``Σerr/Σtotal`` rather than averaging file DERs,
    so the resampled quantity is the estimator we publish and not a cousin of it.
    """
    scored = [r for r in rows if getattr(r, _collar_fields(collar_name)[3]) > 0.0]
    point = aggregate_der(scored, collar_name)
    n = len(scored)
    if n < 2:
        return Interval(point=point, lo=point, hi=point, n=n,
                        resamples=0, seed=seed)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        sample = [scored[rng.randrange(n)] for _ in range(n)]
        draws.append(aggregate_der(sample, collar_name))
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    lo = draws[int(tail * (resamples - 1))]
    hi = draws[int((1.0 - tail) * (resamples - 1))]
    return Interval(point=point, lo=lo, hi=hi, n=n, resamples=resamples, seed=seed)


def paired_bootstrap_delta(
    rows_a: Sequence[FileScore],
    rows_b: Sequence[FileScore],
    collar_name: str = "classic",
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Interval:
    """Interval on ``DER(a) - DER(b)`` over the files both models scored.

    Paired: one resample draws a set of *files* and scores both models on that
    same set, so file difficulty cancels and the interval is about the models.
    An interval excluding zero is what licenses a claim that one model is ahead;
    an interval spanning zero means the ranking is a coin flip on this corpus,
    however many decimals separate the two point estimates.

    Rows are matched by ``file_id``; files present for only one model are dropped
    (an unpaired file cannot inform a paired comparison).
    """
    by_id_b = {r.file_id: r for r in rows_b}
    pairs = [(a, by_id_b[a.file_id]) for a in rows_a if a.file_id in by_id_b]
    total_f = _collar_fields(collar_name)[3]
    pairs = [(a, b) for a, b in pairs
             if getattr(a, total_f) > 0.0 and getattr(b, total_f) > 0.0]
    n = len(pairs)
    point = (aggregate_der([a for a, _ in pairs], collar_name)
             - aggregate_der([b for _, b in pairs], collar_name))
    if n < 2:
        return Interval(point=point, lo=point, hi=point, n=n,
                        resamples=0, seed=seed)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        sample_a = [pairs[i][0] for i in idx]
        sample_b = [pairs[i][1] for i in idx]
        draws.append(aggregate_der(sample_a, collar_name)
                     - aggregate_der(sample_b, collar_name))
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    lo = draws[int(tail * (resamples - 1))]
    hi = draws[int((1.0 - tail) * (resamples - 1))]
    return Interval(point=point, lo=lo, hi=hi, n=n, resamples=resamples, seed=seed)


# ── DER by reference speaker count (#5455) ────────────────────────────────────

#: Bucket labels for reference speaker counts. "5+" is one bucket because that is
#: where every published breakdown puts the cliff and because no public German
#: set here has enough 6- or 7-speaker files to carry its own row.
SPEAKER_BUCKETS: tuple[str, ...] = ("1", "2", "3", "4", "5+")


def speaker_bucket(n_speakers: int) -> str:
    """Bucket label for a reference speaker count."""
    return str(n_speakers) if 1 <= n_speakers <= 4 else "5+"


@dataclass(frozen=True)
class BucketRow:
    """DER within one reference-speaker-count bucket."""

    bucket: str
    n_files: int
    der: float
    scored_speech_s: float


def der_by_speaker_count(
    rows: Sequence[FileScore], collar_name: str = "classic"
) -> list[BucketRow]:
    """Corpus DER within each reference-speaker-count bucket, in bucket order.

    Buckets with no file are omitted rather than printed as zero — an absent
    bucket is "this corpus has no such file", which is a different statement from
    "this model scores 0 there".
    """
    total_f = _collar_fields(collar_name)[3]
    grouped: dict[str, list[FileScore]] = {}
    for row in rows:
        grouped.setdefault(speaker_bucket(row.n_speakers_ref), []).append(row)
    out: list[BucketRow] = []
    for bucket in SPEAKER_BUCKETS:
        members = grouped.get(bucket)
        if not members:
            continue
        out.append(BucketRow(
            bucket=bucket,
            n_files=len(members),
            der=aggregate_der(members, collar_name),
            scored_speech_s=sum(getattr(r, total_f) for r in members),
        ))
    return out


# ── reference corpus statistics (#5453) ───────────────────────────────────────


def _to_annotation(segments: Iterable[DiarSegment]) -> Annotation:
    ann = Annotation()
    for start, end, speaker in segments:
        if end > start:
            ann[Segment(float(start), float(end))] = str(speaker)
    return ann


@dataclass(frozen=True)
class OverlapStats:
    """Overlapped-speech statistics of a reference corpus, all fractions in percent.

    Three denominators, because "the overlap fraction" names none of them and the
    three differ by more than a percentage point on the same corpus:

    * ``of_speech`` — overlapped seconds / seconds where **anyone** speaks. The
      usual reading of "how much of the speech is overlapped".
    * ``of_speaker_time`` — overlapped seconds / Σ per-speaker durations. Larger,
      because overlapped time is counted once in the numerator and twice in the
      denominator.
    * ``of_wallclock`` — overlapped seconds / recording extent, silence included.
      The smallest, and the one that depends on how much silence a corpus keeps.

    ``per_file_mean`` / ``per_file_sd`` are the unweighted per-file mean and
    standard deviation of ``of_speech`` — quoted because a corpus mean hides that
    the per-file spread is several percentage points wide.
    """

    n_files: int
    overlap_s: float
    speech_s: float
    speaker_time_s: float
    wallclock_s: float
    of_speech: float
    of_speaker_time: float
    of_wallclock: float
    per_file_mean: float
    per_file_sd: float


def reference_overlap(gold_files: Sequence[Path]) -> OverlapStats:
    """Overlapped-speech statistics over a set of gold RTTMs."""
    overlap_s = speech_s = speaker_time_s = wallclock_s = 0.0
    per_file: list[float] = []
    for path in gold_files:
        ann = _to_annotation(load_rttm(path))
        if not ann:
            continue
        timeline: Timeline = ann.get_timeline()
        f_speech = timeline.support().duration()
        f_overlap = ann.get_overlap().duration()
        f_speaker = sum(seg.duration for seg in timeline)
        f_wall = ann.get_timeline().extent().duration
        overlap_s += f_overlap
        speech_s += f_speech
        speaker_time_s += f_speaker
        wallclock_s += f_wall
        if f_speech > 0.0:
            per_file.append(f_overlap / f_speech * 100.0)
    n = len(per_file)
    return OverlapStats(
        n_files=n,
        overlap_s=overlap_s,
        speech_s=speech_s,
        speaker_time_s=speaker_time_s,
        wallclock_s=wallclock_s,
        of_speech=overlap_s / speech_s * 100.0 if speech_s else 0.0,
        of_speaker_time=(overlap_s / speaker_time_s * 100.0
                         if speaker_time_s else 0.0),
        of_wallclock=overlap_s / wallclock_s * 100.0 if wallclock_s else 0.0,
        per_file_mean=statistics.fmean(per_file) if per_file else 0.0,
        per_file_sd=statistics.stdev(per_file) if n > 1 else 0.0,
    )


# ── boundary precision + where the misses sit (#5460) ─────────────────────────


@dataclass(frozen=True)
class BoundaryReport:
    """Speaker-aware boundary offsets and the length regime the misses sit in.

    ``offset_*`` are milliseconds between a reference speaker-turn boundary (each
    segment start and end) and the **nearest boundary of the same speaker** in
    the hypothesis, after pyannote's optimal (Hungarian) label mapping. Speaker-
    aware by construction: a boundary is only credited against the speaker it
    belongs to, so a diarizer cannot score well by emitting boundaries anywhere.
    Collar-free, because this is not a DER — the collar exists to forgive exactly
    the quantity measured here.

    ``unmatched_boundaries`` counts reference boundaries whose speaker has no
    counterpart in the mapped hypothesis at all; they carry no offset and are
    reported rather than dropped silently.

    ``missed_short_s`` / ``missed_long_s`` split reference speech that the mapped
    hypothesis never covers by the length of the reference segment it sits in
    (``SHORT_SEGMENT_S``). This is the instrument for the "the vendor overhears
    backchannels" reading of a miss-dominated row: that reading predicts the
    missed seconds concentrate in the short regime.
    """

    n_files: int
    n_boundaries: int
    unmatched_boundaries: int
    offset_median_ms: float
    offset_mean_ms: float
    offset_p90_ms: float
    within_250ms: float      # percent of matched boundaries
    missed_short_s: float
    missed_long_s: float

    @property
    def missed_short_share(self) -> float:
        """Percent of uncovered reference speech sitting in short segments."""
        total = self.missed_short_s + self.missed_long_s
        return self.missed_short_s / total * 100.0 if total else 0.0


def _mapped_hypothesis(reference: Annotation, hypothesis: Annotation) -> Annotation:
    """Relabel ``hypothesis`` with pyannote's optimal mapping onto ``reference``."""
    mapping = DiarizationErrorRate().optimal_mapping(reference, hypothesis)
    return hypothesis.rename_labels(mapping=mapping)


def _boundaries(ann: Annotation, label: str) -> list[float]:
    """Sorted start+end times of one speaker's merged turns."""
    support = ann.label_timeline(label).support()
    times: set[float] = set()
    for seg in support:
        times.add(seg.start)
        times.add(seg.end)
    return sorted(times)


def boundary_report(
    rttm_pairs: Sequence[tuple[Path, Path]],
    *,
    short_segment_s: float = SHORT_SEGMENT_S,
) -> BoundaryReport:
    """Boundary offsets + missed-speech length split over (gold, hyp) RTTM pairs."""
    offsets: list[float] = []
    unmatched = 0
    missed_short = missed_long = 0.0
    n_files = 0
    for gold_path, hyp_path in rttm_pairs:
        reference = _to_annotation(load_rttm(gold_path))
        hypothesis = _to_annotation(load_rttm(hyp_path))
        if not reference:
            continue
        n_files += 1
        mapped = _mapped_hypothesis(reference, hypothesis) if hypothesis else Annotation()
        hyp_labels = set(mapped.labels())
        for label in reference.labels():
            ref_timeline = reference.label_timeline(label).support()
            if label not in hyp_labels:
                unmatched += 2 * len(ref_timeline)
                missed_long += sum(
                    seg.duration for seg in ref_timeline
                    if seg.duration >= short_segment_s
                )
                missed_short += sum(
                    seg.duration for seg in ref_timeline
                    if seg.duration < short_segment_s
                )
                continue
            hyp_times = _boundaries(mapped, label)
            hyp_support = mapped.label_timeline(label).support()
            for seg in ref_timeline:
                for t in (seg.start, seg.end):
                    offsets.append(
                        min(abs(t - h) for h in hyp_times) * 1000.0
                    )
                covered = hyp_support.crop(seg, mode="intersection").duration()
                uncovered = max(0.0, seg.duration - covered)
                if seg.duration < short_segment_s:
                    missed_short += uncovered
                else:
                    missed_long += uncovered
    offsets.sort()
    n = len(offsets)
    return BoundaryReport(
        n_files=n_files,
        n_boundaries=n + unmatched,
        unmatched_boundaries=unmatched,
        offset_median_ms=statistics.median(offsets) if offsets else 0.0,
        offset_mean_ms=statistics.fmean(offsets) if offsets else 0.0,
        offset_p90_ms=offsets[int(0.9 * (n - 1))] if offsets else 0.0,
        within_250ms=(sum(1 for o in offsets if o <= 250.0) / n * 100.0
                      if n else 0.0),
        missed_short_s=missed_short,
        missed_long_s=missed_long,
    )


# ── artifact plumbing + CLI ───────────────────────────────────────────────────


def artifact_rttm_pairs(model_dir: Path, dataset: str) -> list[tuple[Path, Path]]:
    """(gold, hyp) RTTM paths for one dataset of a committed artifact directory."""
    gold_dir = model_dir / "gold" / dataset
    hyp_dir = model_dir / "hyp" / dataset
    pairs: list[tuple[Path, Path]] = []
    for gold in sorted(gold_dir.glob("*.rttm")):
        hyp = hyp_dir / gold.name
        if not hyp.exists():
            raise FileNotFoundError(f"missing hypothesis RTTM {hyp}")
        pairs.append((gold, hyp))
    return pairs


def artifact_datasets(model_dir: Path) -> list[str]:
    """Dataset names present under a committed artifact's ``gold/``."""
    gold_root = model_dir / "gold"
    if not gold_root.is_dir():
        return []
    return sorted(p.name for p in gold_root.iterdir() if p.is_dir())


def folding_sensitivity(
    rttm_pairs: Sequence[tuple[Path, Path]],
    dataset: str,
    *,
    gap_merge_s: float = DEFAULT_GAP_MERGE_S,
) -> dict[str, float]:
    """How much this row moves if the shared turn folding is applied to it too.

    Hosted adapters must fold: their APIs return labelled *words*, and turns have
    to be reconstructed before anything can be scored. Local diarizers already
    emit turns, so folding them again would be a second opinion we impose rather
    than a reconstruction we cannot avoid — which is why
    ``raven_diar/adapters/aggregate.py`` exempts them.

    The consequence is that "both providers' turns come from the same shared
    aggregator" is true of two hosted rows compared with each other and *not* of
    a hosted row compared with a local one. This function measures the residue
    instead of leaving the reader to wonder about it: it re-scores the committed
    hypothesis after passing it through the same folding, and reports the shift.

    A hosted row is a fixed point (the folding is idempotent on already-folded
    turns) and reads exactly 0.000. A local row moves by however much its own
    turn granularity differs from the folded one — which is corpus- and
    model-specific and signed in both directions, so this is not a correction
    that could simply be applied to everything.
    """
    raw = [(load_rttm(g), load_rttm(h)) for g, h in rttm_pairs]
    folded = [(g, spans_to_turns(h, gap_merge_s=gap_merge_s)) for g, h in raw]
    before = score_segment_pairs(dataset, raw)
    after = score_segment_pairs(dataset, folded)
    return {
        "gap_merge_s": gap_merge_s,
        "der_full": before.der_full,
        "der_full_folded": after.der_full,
        "delta_full": after.der_full - before.der_full,
        "der_classic": before.der_classic,
        "der_classic_folded": after.der_classic,
        "delta_classic": after.der_classic - before.der_classic,
    }


def analyse(
    model_dir: Path,
    dataset: str,
    collar_name: str = "classic",
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Run every instrument on one dataset of one committed artifact."""
    pairs = artifact_rttm_pairs(model_dir, dataset)
    score = score_rttm_pairs(dataset, pairs)
    rows = list(score.per_file)
    ci = bootstrap_der_ci(rows, collar_name, resamples=resamples, seed=seed)
    overlap = reference_overlap([g for g, _ in pairs])
    boundary = boundary_report(pairs)
    return {
        "model": model_dir.name,
        "dataset": dataset,
        "collar": collar_name,
        "n_files": score.n_files,
        "der_corpus": (score.der_classic if collar_name == "classic"
                       else score.der_full),
        "der_file_mean": (score.der_classic_filemean if collar_name == "classic"
                          else score.der_full_filemean),
        "interval": {"point": ci.point, "lo": ci.lo, "hi": ci.hi,
                     "half_width": ci.half_width, "n": ci.n,
                     "resamples": ci.resamples, "seed": ci.seed},
        "by_speaker_count": [
            {"bucket": b.bucket, "n_files": b.n_files, "der": b.der,
             "scored_speech_s": b.scored_speech_s}
            for b in der_by_speaker_count(rows, collar_name)
        ],
        "reference_overlap": {
            "of_speech": overlap.of_speech,
            "of_speaker_time": overlap.of_speaker_time,
            "of_wallclock": overlap.of_wallclock,
            "per_file_mean": overlap.per_file_mean,
            "per_file_sd": overlap.per_file_sd,
            "overlap_s": overlap.overlap_s,
            "speech_s": overlap.speech_s,
        },
        "folding_sensitivity": folding_sensitivity(pairs, dataset),
        "boundary": {
            "n_boundaries": boundary.n_boundaries,
            "unmatched": boundary.unmatched_boundaries,
            "median_ms": boundary.offset_median_ms,
            "mean_ms": boundary.offset_mean_ms,
            "p90_ms": boundary.offset_p90_ms,
            "within_250ms": boundary.within_250ms,
            "missed_short_s": boundary.missed_short_s,
            "missed_long_s": boundary.missed_long_s,
            "missed_short_share": boundary.missed_short_share,
        },
    }


def _print_human(report: dict[str, object]) -> None:
    iv = report["interval"]                      # type: ignore[index]
    ov = report["reference_overlap"]             # type: ignore[index]
    bd = report["boundary"]                      # type: ignore[index]
    print(f"{report['model']} · {report['dataset']} · collar {report['collar']} "
          f"· n={report['n_files']}")
    print(f"  DER corpus      {report['der_corpus']:.3f} %   "
          f"file-mean {report['der_file_mean']:.3f} %")
    print(f"  95 % CI         [{iv['lo']:.3f}, {iv['hi']:.3f}] "
          f"(±{iv['half_width']:.3f}, {iv['resamples']} resamples, seed {iv['seed']})")
    buckets = report["by_speaker_count"]         # type: ignore[index]
    if buckets:
        cells = "   ".join(
            f"{b['bucket']}: {b['der']:.2f} (n={b['n_files']})" for b in buckets
        )
        print(f"  by ref speakers {cells}")
    print(f"  ref overlap     {ov['of_speech']:.2f} % of speech   "
          f"{ov['of_speaker_time']:.2f} % of speaker-time   "
          f"{ov['of_wallclock']:.2f} % of wall clock   "
          f"(per file {ov['per_file_mean']:.2f} ± {ov['per_file_sd']:.2f})")
    print(f"  boundary offset median {bd['median_ms']:.0f} ms   "
          f"mean {bd['mean_ms']:.0f} ms   p90 {bd['p90_ms']:.0f} ms   "
          f"≤250 ms {bd['within_250ms']:.1f} %")
    print(f"  uncovered ref speech: short(<{SHORT_SEGMENT_S} s) "
          f"{bd['missed_short_s']:.0f} s ({bd['missed_short_share']:.1f} %)   "
          f"long {bd['missed_long_s']:.0f} s")
    fs = report["folding_sensitivity"]           # type: ignore[index]
    print(f"  folding residue if the shared {fs['gap_merge_s']} s turn folding "
          f"were applied here too: "
          f"{fs['delta_classic']:+.3f} pp @0.25   {fs['delta_full']:+.3f} pp @0.0"
          + ("   (fixed point — already folded)"
             if abs(fs["delta_classic"]) < 1e-9 else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-diar-analysis", description=__doc__)
    parser.add_argument(
        "model_dir", type=Path,
        help="a committed artifact dir: artifacts/<run>/<model>/",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="dataset under gold/ (default: every dataset in the artifact)",
    )
    parser.add_argument(
        "--collar", default="classic", choices=sorted(COLLARS),
        help="collar variant to analyse (default: classic = 0.25 s)",
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json", action="store_true", help="emit JSON, not a table")
    args = parser.parse_args(argv)

    datasets = ([args.dataset] if args.dataset
                else artifact_datasets(args.model_dir))
    if not datasets:
        print(f"FAIL: no gold/<dataset>/ under {args.model_dir}", file=sys.stderr)
        return 2
    reports = [
        analyse(args.model_dir, ds, args.collar,
                resamples=args.resamples, seed=args.seed)
        for ds in datasets
    ]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for report in reports:
            _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
