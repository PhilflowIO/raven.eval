"""DER scoring for a set of (gold, hyp) RTTM pairs — both collars + breakdown.

Light by construction: imports only ``raven_eval_core.der`` (pyannote.metrics, no
torch), so this is the shared metric core between the Tier-2 runner (which writes
these numbers) and the Tier-1 re-scorer (``scripts/verify.py``, which recomputes
them) — they cannot drift.

Reported per dataset:
  * ``der_full``     — corpus DER at collar 0.0, skip_overlap False  (pyannote / DIHARD)
  * ``der_classic``  — corpus DER at collar 0.25, skip_overlap False (NIST / CALLHOME)
  * ``miss`` / ``fa`` / ``conf`` — the md-eval decomposition at collar **0.0**
    (``miss + fa + conf == der_full`` by construction)
  * ``miss_classic`` / ``fa_classic`` / ``conf_classic`` — the same decomposition
    at collar **0.25** (``… == der_classic``). Both exist because a decomposition
    printed next to a DER of a *different* collar is not a decomposition of that
    number — it is three unrelated figures in adjacent columns.
  * ``der_full_filemean`` / ``der_classic_filemean`` — the same files under the
    OTHER aggregation convention (unweighted mean of per-file DERs). Not
    comparable to the corpus figures; published because other benchmarks report
    it and a DER is only comparable once its aggregation is stated.
  * ``per_file`` — one :class:`FileScore` per scored file, which is what makes
    the corpus scalar auditable: confidence intervals, a split by reference
    speaker count and any boundary analysis all need the file rows, not the
    aggregate.

All rate values are **percentages** (fraction × 100), matching the WER convention
and the ``±0.05 pp`` verify tolerance; ``*_s`` fields are seconds. Corpus
aggregation is NIST-correct (Σ(errors)/Σ(total) over files), delegated to
``compute_der_corpus_detailed``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from raven_eval_core.der import (
    DerComponents,
    DiarSegment,
    compute_der_corpus_detailed,
    file_mean_der,
    load_rttm,
)

from .config import COLLARS

# A (gold_segments, hyp_segments) pair.
SegmentPair = tuple[list[DiarSegment], list[DiarSegment]]


@dataclass(frozen=True)
class FileScore:
    """One file's contribution to the corpus DER, at both collars.

    The error components are kept in **seconds**, not as rates. That is the
    aggregation-safe form: a resample or a speaker-count bucket re-derives its
    DER as ``Σ(miss_s+fa_s+conf_s) / Σ(total_s)`` over the selected rows, which
    is the same NIST-correct arithmetic the corpus figure uses. Storing rates
    would only permit a file-mean, and would silently weight a 30 s clip like a
    50 min meeting.

    ``der_*`` are percentages, redundant with the seconds and carried because a
    reader scanning the file list wants the rate, not to divide by hand.
    ``n_speakers_ref`` is the number of distinct speaker labels in the *reference*
    (not the hypothesis) — the axis the field reports difficulty on.
    """

    file_id: str
    n_speakers_ref: int
    # collar 0.0 ("full")
    der_full: float
    miss_full_s: float
    fa_full_s: float
    conf_full_s: float
    total_full_s: float
    # collar 0.25 ("classic")
    der_classic: float
    miss_classic_s: float
    fa_classic_s: float
    conf_classic_s: float
    total_classic_s: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _file_score(
    file_id: str,
    ref_segments: list[DiarSegment],
    full: DerComponents,
    classic: DerComponents,
) -> FileScore:
    """Assemble one :class:`FileScore` from the two per-collar component sets."""
    return FileScore(
        file_id=file_id,
        n_speakers_ref=len({spk for _, _, spk in ref_segments}),
        der_full=full.der * 100.0,
        miss_full_s=full.miss * full.total_ref,
        fa_full_s=full.false_alarm * full.total_ref,
        conf_full_s=full.confusion * full.total_ref,
        total_full_s=full.total_ref,
        der_classic=classic.der * 100.0,
        miss_classic_s=classic.miss * classic.total_ref,
        fa_classic_s=classic.false_alarm * classic.total_ref,
        conf_classic_s=classic.confusion * classic.total_ref,
        total_classic_s=classic.total_ref,
    )


@dataclass(frozen=True)
class DerScore:
    """Per-dataset DER result, all rate fields in percent (except ``n_files``)."""

    dataset: str
    n_files: int
    der_full: float          # corpus, collar 0.0
    der_classic: float       # corpus, collar 0.25
    miss: float              # component of der_full  (collar 0.0)
    fa: float                # component of der_full  (collar 0.0)
    conf: float              # component of der_full  (collar 0.0)
    miss_classic: float      # component of der_classic (collar 0.25)
    fa_classic: float        # component of der_classic (collar 0.25)
    conf_classic: float      # component of der_classic (collar 0.25)
    der_full_filemean: float     # unweighted mean of per-file DER, collar 0.0
    der_classic_filemean: float  # unweighted mean of per-file DER, collar 0.25
    per_file: tuple[FileScore, ...] = field(default=(), repr=False)

    #: Fields written to ``expected.json`` and re-checked by ``scripts/verify.py``.
    EXPECTED_FIELDS = (
        "der_full", "der_classic", "miss", "fa", "conf",
        "miss_classic", "fa_classic", "conf_classic",
        "der_full_filemean", "der_classic_filemean",
    )

    def expected_entry(self) -> dict[str, float]:
        """The ``expected.json`` value for this dataset (percent, rounded)."""
        return {f: round(float(getattr(self, f)), 4) for f in self.EXPECTED_FIELDS}

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["per_file"] = [f.as_dict() for f in self.per_file]
        return out


def score_segment_pairs(
    dataset: str,
    pairs: list[SegmentPair],
    file_ids: list[str] | None = None,
) -> DerScore:
    """Score already-parsed (gold, hyp) segment pairs → :class:`DerScore`.

    ``file_ids`` labels the per-file rows; when omitted the rows are numbered
    positionally, which is enough for an in-memory analysis but not for a
    committed artifact — :func:`score_rttm_pairs` always passes real ids.
    """
    full_cfg = COLLARS["full"]
    classic_cfg = COLLARS["classic"]
    full, full_files = compute_der_corpus_detailed(
        pairs,
        collar=float(full_cfg["collar"]),
        skip_overlap=bool(full_cfg["skip_overlap"]),
    )
    classic, classic_files = compute_der_corpus_detailed(
        pairs,
        collar=float(classic_cfg["collar"]),
        skip_overlap=bool(classic_cfg["skip_overlap"]),
    )
    ids = file_ids if file_ids is not None else [str(i) for i in range(len(pairs))]
    if len(ids) != len(pairs):
        raise ValueError(
            f"file_ids has {len(ids)} entries for {len(pairs)} pairs"
        )
    per_file = tuple(
        _file_score(fid, ref, f, c)
        for fid, (ref, _hyp), f, c in zip(ids, pairs, full_files, classic_files,
                                          strict=True)
    )
    return DerScore(
        dataset=dataset,
        n_files=len(pairs),
        der_full=full.der * 100.0,
        der_classic=classic.der * 100.0,
        miss=full.miss * 100.0,
        fa=full.false_alarm * 100.0,
        conf=full.confusion * 100.0,
        miss_classic=classic.miss * 100.0,
        fa_classic=classic.false_alarm * 100.0,
        conf_classic=classic.confusion * 100.0,
        der_full_filemean=file_mean_der(full_files) * 100.0,
        der_classic_filemean=file_mean_der(classic_files) * 100.0,
        per_file=per_file,
    )


def score_rttm_pairs(
    dataset: str, rttm_pairs: list[tuple[Path, Path]]
) -> DerScore:
    """Load (gold_rttm, hyp_rttm) file pairs and score them → :class:`DerScore`."""
    pairs: list[SegmentPair] = [
        (load_rttm(gold), load_rttm(hyp)) for gold, hyp in rttm_pairs
    ]
    file_ids = [gold.stem for gold, _ in rttm_pairs]
    return score_segment_pairs(dataset, pairs, file_ids=file_ids)


def score_pair(gold_rttm: Path, hyp_rttm: Path, dataset: str = "pair") -> DerScore:
    """Convenience: score a single (gold, hyp) RTTM pair."""
    return score_rttm_pairs(dataset, [(gold_rttm, hyp_rttm)])
