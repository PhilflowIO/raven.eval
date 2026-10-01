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

``by_region``
    Where inside a dialect corpus does it fail? "Swiss German" is not one
    evaluation category: FHNW spans every Swiss dialect region and writes each
    utterance's canton into its ``sample_id``. Per region: n, corpus WER with the
    same bootstrap interval, corpus BLEU with n and no interval (BLEU has none
    yet). A region under ``REGION_MIN_N`` utterances shows its n and a note,
    never a number. Region rows sit next to the corpus number and are never
    averaged into one — the corpus figure pools utterances, it is not a mean of
    regions.

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

from dataclasses import dataclass

from raven_eval_core.bleu import corpus_bleu_score
from raven_eval_core.bootstrap import (
    Interval,
    Unit,
    bootstrap_ratio_ci,
    pair_by_id,
    paired_bootstrap_ratio_delta,
)
from raven_eval_core.flozi_wer import utterance_word_errors
from raven_eval_core.wer import utterance_strict_de_units

from .config import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    REGION_MIN_N,
)
from .datasets import fhnw_all_dialects

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


# ── per region, inside one dialect corpus (#32) ─────────────────────────────

#: subset → (region parser over sample_id, code → display name). Mirrors
#: ``benchmark.config.yaml`` → ``dialect_region_breakdown.datasets``; the
#: contract test asserts the two name the same corpora.
REGION_PARSERS: dict[str, tuple[Callable[[str], str], Callable[[str], str]]] = {
    fhnw_all_dialects.DATASET_ID: (fhnw_all_dialects.region_of,
                                   fhnw_all_dialects.region_name),
}


@dataclass(frozen=True)
class RegionRow:
    """One region of one dialect corpus.

    ``wer`` and ``bleu`` are None exactly when ``n`` is under the minimum; then
    ``note`` says why and the row carries its n only. ``bleu`` is a point
    estimate: BLEU has no interval in this repo yet, so none is invented here.
    """

    region: str          # code as it stands in the sample_id
    name: str            # display name, or the code itself when unmapped
    n: int
    wer: Interval | None
    bleu: float | None
    note: str = ""


def by_region(
    path: Path,
    lens: str = PUBLISHED_LENS,
    *,
    min_n: int = REGION_MIN_N,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> list[RegionRow]:
    """WER (with interval) and BLEU per dialect region of one predictions file.

    Regions come from each line's ``sample_id`` via the corpus's own parser, so
    the split is the one the loader wrote. Rows are ordered by n, largest first.
    Raises if the subset has no region parser, if a line has no ``sample_id``,
    or if an id carries no readable region (the parser raises) — so every
    utterance lands in exactly one region and the region n sum to the corpus n.
    A breakdown that lost utterances would look exactly as authoritative as one
    that did not.
    """
    subset = _subset_of(path)
    if subset not in REGION_PARSERS:
        raise KeyError(f"{subset!r} has no region parser; have {sorted(REGION_PARSERS)}")
    parse, name = REGION_PARSERS[subset]
    rows = _rows(path)
    if any("sample_id" not in r for r in rows):
        raise ValueError(f"{path}: lines without sample_id cannot be split by region")

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(parse(r["sample_id"]), []).append(r)

    out: list[RegionRow] = []
    for code, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        n = len(members)
        if n < min_n:
            out.append(RegionRow(
                region=code, name=name(code), n=n, wer=None, bleu=None,
                note=f"n < {min_n}: too few utterances for a number",
            ))
            continue
        out.append(RegionRow(
            region=code, name=name(code), n=n,
            wer=bootstrap_ratio_ci(_units(members, lens), resamples=resamples,
                                   seed=seed, confidence=confidence),
            bleu=corpus_bleu_score([r["reference"] for r in members],
                                   [r["prediction"] for r in members]),
        ))
    return out


def _subset_of(path: Path) -> str:
    m = _PRED_RE.match(path.name)
    if not m:
        raise ValueError(f"{path.name} is not a predictions_<subset>.jsonl file")
    return m.group("subset")


def _print_regions(regions: list[RegionRow], min_n: int) -> None:
    print(f"  by region (min n {min_n}; no winner marks, never averaged; "
          f"BLEU has no interval):")
    for row in regions:
        label = f"{row.region} {row.name}" if row.name != row.region else row.region
        if row.wer is None or row.bleu is None:
            print(f"    {label:<36} n={row.n:<5} {row.note}")
        else:
            print(f"    {label:<36} n={row.n:<5} WER {row.wer.point:.3f} % "
                  f"[{row.wer.lo:.3f}, {row.wer.hi:.3f}]   BLEU {row.bleu:.2f}")


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
        if subset in REGION_PARSERS:
            _print_regions(by_region(path, args.lens, resamples=args.resamples,
                                     seed=args.seed), REGION_MIN_N)
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
