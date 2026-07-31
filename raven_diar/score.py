"""DER scoring for a set of (gold, hyp) RTTM pairs — both collars + breakdown.

Light by construction: imports only ``raven_eval_core.der`` (pyannote.metrics, no
torch), so this is the shared metric core between the Tier-2 runner (which writes
these numbers) and the Tier-1 re-scorer (``scripts/verify.py``, which recomputes
them) — they cannot drift.

Reported per dataset:
  * ``der_full``     — DER at collar 0.0, skip_overlap False  (pyannote / DIHARD)
  * ``der_classic``  — DER at collar 0.25, skip_overlap False (NIST / CALLHOME)
  * ``miss`` / ``fa`` / ``conf`` — the md-eval decomposition at collar 0.0
    (the un-forgiven, hand-verifiable partition; by construction
    ``miss + fa + conf == der_full``).

All values are **percentages** (fraction × 100), matching the WER convention and
the ``±0.05 pp`` verify tolerance. Corpus aggregation is NIST-correct
(Σ(errors)/Σ(total) over files), delegated to ``compute_der_corpus``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from raven_eval_core.der import (
    DiarSegment,
    compute_der_corpus,
    load_rttm,
)

from .config import COLLARS

# A (gold_segments, hyp_segments) pair.
SegmentPair = tuple[list[DiarSegment], list[DiarSegment]]


@dataclass(frozen=True)
class DerScore:
    """Per-dataset DER result, all fields in percent (except ``n_files``)."""

    dataset: str
    n_files: int
    der_full: float      # collar 0.0
    der_classic: float   # collar 0.25
    miss: float          # component of der_full
    fa: float            # component of der_full
    conf: float          # component of der_full

    def expected_entry(self) -> dict[str, float]:
        """The ``expected.json`` value for this dataset (percent, rounded)."""
        return {
            "der_full": round(self.der_full, 4),
            "der_classic": round(self.der_classic, 4),
            "miss": round(self.miss, 4),
            "fa": round(self.fa, 4),
            "conf": round(self.conf, 4),
        }

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def score_segment_pairs(dataset: str, pairs: list[SegmentPair]) -> DerScore:
    """Score already-parsed (gold, hyp) segment pairs → :class:`DerScore`."""
    full_cfg = COLLARS["full"]
    classic_cfg = COLLARS["classic"]
    full = compute_der_corpus(
        pairs,
        collar=float(full_cfg["collar"]),
        skip_overlap=bool(full_cfg["skip_overlap"]),
    )
    classic = compute_der_corpus(
        pairs,
        collar=float(classic_cfg["collar"]),
        skip_overlap=bool(classic_cfg["skip_overlap"]),
    )
    return DerScore(
        dataset=dataset,
        n_files=len(pairs),
        der_full=full.der * 100.0,
        der_classic=classic.der * 100.0,
        miss=full.miss * 100.0,
        fa=full.false_alarm * 100.0,
        conf=full.confusion * 100.0,
    )


def score_rttm_pairs(
    dataset: str, rttm_pairs: list[tuple[Path, Path]]
) -> DerScore:
    """Load (gold_rttm, hyp_rttm) file pairs and score them → :class:`DerScore`."""
    pairs: list[SegmentPair] = [
        (load_rttm(gold), load_rttm(hyp)) for gold, hyp in rttm_pairs
    ]
    return score_segment_pairs(dataset, pairs)


def score_pair(gold_rttm: Path, hyp_rttm: Path, dataset: str = "pair") -> DerScore:
    """Convenience: score a single (gold, hyp) RTTM pair."""
    return score_rttm_pairs(dataset, [(gold_rttm, hyp_rttm)])
