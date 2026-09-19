"""Read a committed WER artifact past its corpus scalar: how precise, and is a gap real.

The WER counterpart of ``raven_diar.analysis``, on the same resampler
(``raven_eval_core.bootstrap``) and the same contract settings
(``benchmark.config.yaml`` → ``wer.uncertainty``):

``wer_interval``
    How precise is one published WER? A percentile bootstrap over utterances,
    each resample re-aggregating ``Σedits / Σref_words`` — the published
    estimator. On 100 utterances the answer is routinely several points.

``paired_wer_delta``
    Is a gap between two models real? Both models are resampled on the SAME
    utterances, matched by ``sample_id``, so utterance difficulty cancels. An
    interval that spans zero means the two numbers do not rank the models.

Two lenses, because the page makes claims in both: ``flozi-strict`` (the
published table, ``raven_eval_core.flozi_wer``) and ``strict-de`` (the benchmark
page's length-weighted lens, ``raven_eval_core.corpus_wer_strict_de_pct``). Each
decomposes into per-utterance units whose ratio IS the corpus number, so an
interval is always around the exact figure that is printed.

Pure Python, no GPU, no network — reads only ``predictions_*.jsonl``.

    make analyse ARTIFACT=artifacts/<run>/<model>
    make analyse ARTIFACT=artifacts/<run-a>/<model-a> COMPARE=artifacts/<run-b>/<model-b>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from raven_eval_core.bootstrap import (
    Interval,
    Unit,
    bootstrap_ratio_ci,
    pair_by_id,
    paired_bootstrap_ratio_delta,
)
from raven_eval_core.flozi_wer import utterance_word_errors
from raven_eval_core.wer import utterance_strict_de_units

from .config import BOOTSTRAP_CONFIDENCE, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED

#: Lens name → per-utterance decomposition of that lens's corpus WER.
LENSES: dict[str, Callable[[list[str], list[str]], list[tuple[float, float]]]] = {
    "flozi-strict": utterance_word_errors,  # type: ignore[dict-item]
    "strict-de": utterance_strict_de_units,  # type: ignore[dict-item]
}
PUBLISHED_LENS = "flozi-strict"

_PRED_RE = re.compile(r"^predictions_(?P<subset>.+)\.jsonl$")


def _rows(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    failed = [r for r in rows if r.get("error") is not None or r["prediction"] is None]
    if failed:
        raise ValueError(
            f"{path}: {len(failed)} failed request(s) — an interval over the clips "
            "that worked would be around the wrong number"
        )
    return rows


def _units(rows: list[dict], lens: str) -> list[Unit]:
    try:
        decompose = LENSES[lens]
    except KeyError:
        raise KeyError(f"unknown lens {lens!r}; have {sorted(LENSES)}") from None
    return [(float(e), float(n)) for e, n in decompose(
        [r["reference"] for r in rows], [r["prediction"] for r in rows]
    )]


def wer_interval(
    path: Path,
    lens: str = PUBLISHED_LENS,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Interval:
    """Bootstrap interval around the corpus WER of one predictions file."""
    return bootstrap_ratio_ci(_units(_rows(path), lens), resamples=resamples,
                              seed=seed, confidence=confidence)


def paired_wer_delta(
    path_a: Path,
    path_b: Path,
    lens: str = PUBLISHED_LENS,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Interval:
    """Interval on ``WER(a) - WER(b)`` over the same utterances.

    Utterances are matched by ``sample_id``; a file without ids cannot be paired
    and raises. The same id must carry the same reference on both sides —
    otherwise the two runs read different data under one name, and the
    difference would be about the data.
    """
    rows_a, rows_b = _rows(path_a), _rows(path_b)
    for path, rows in ((path_a, rows_a), (path_b, rows_b)):
        if any("sample_id" not in r for r in rows):
            raise ValueError(f"{path}: lines without sample_id cannot be paired")
    by_a = {r["sample_id"]: r for r in rows_a}
    by_b = {r["sample_id"]: r for r in rows_b}
    matched = pair_by_id(by_a, by_b)
    differing = [a["sample_id"] for a, b in matched if a["reference"] != b["reference"]]
    if differing:
        raise ValueError(
            f"{len(differing)} sample_id(s) carry different references in the two "
            f"runs (first: {differing[0]}) — they did not read the same data"
        )
    units_a = _units([a for a, _ in matched], lens)
    units_b = _units([b for _, b in matched], lens)
    return paired_bootstrap_ratio_delta(
        list(zip(units_a, units_b, strict=True)),
        resamples=resamples, seed=seed, confidence=confidence,
    )


def artifact_subsets(model_dir: Path) -> dict[str, Path]:
    """``{subset: predictions path}`` of one artifact dir."""
    out: dict[str, Path] = {}
    for p in sorted(model_dir.glob("predictions_*.jsonl")):
        m = _PRED_RE.match(p.name)
        if m:
            out[m.group("subset")] = p
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-asr-analysis", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="artifacts/<run>/<model>/")
    parser.add_argument("--compare", type=Path, default=None,
                        help="a second artifact dir: paired interval on WER(model) - WER(compare)")
    parser.add_argument("--lens", default=PUBLISHED_LENS, choices=sorted(LENSES))
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    subsets = artifact_subsets(args.model_dir)
    if not subsets:
        print(f"FAIL: no predictions_*.jsonl under {args.model_dir}", file=sys.stderr)
        return 2
    other = artifact_subsets(args.compare) if args.compare else {}
    for subset, path in subsets.items():
        ci = wer_interval(path, args.lens, resamples=args.resamples, seed=args.seed)
        print(f"{args.model_dir.name} · {subset} · {args.lens}")
        print(f"  WER {ci.point:.3f} %   95 % CI [{ci.lo:.3f}, {ci.hi:.3f}] "
              f"(±{ci.half_width:.3f}, n={ci.n}, {ci.resamples} resamples, seed {ci.seed})")
        if args.compare:
            if subset not in other:
                print(f"  vs {args.compare.name}: no {subset} predictions there")
                continue
            d = paired_wer_delta(path, other[subset], args.lens,
                                 resamples=args.resamples, seed=args.seed)
            verdict = ("spans zero — no ranking" if d.lo <= 0.0 <= d.hi
                       else "excludes zero")
            print(f"  minus {args.compare.name}: {d.point:+.3f} pp   "
                  f"95 % CI [{d.lo:+.3f}, {d.hi:+.3f}] (n={d.n}) — {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
