"""Optional cross-check: does dscore agree with our pyannote.metrics DER?

Makes the dscore note in ``BENCHMARKS.md`` / ``benchmark.config.yaml`` REAL — a
runnable check (``make dscore-check``) that scores the SAME gold/hyp RTTMs with
nryant/dscore and asserts its DER agrees with ``raven_eval_core.der`` within a
small tolerance, at **both** published collars.

The collar is the whole point of this check, and it was wrong
-------------------------------------------------------------
This module used to pass no collar flag at all and describe dscore's default as
"0.25 with ``--score_overlaps`` on". Three things were wrong with that sentence
and each would have made the first real run fail:

  1. dscore's ``--collar`` default is **0.0**, not 0.25 (``score.py``,
     ``--collar ... default=0.0``). Comparing our ``classic`` variant against it
     compares collar 0.25 with collar 0.0 — 3.2 pp on the first CALLHOME-de file
     alone, six times the 0.5 pp tolerance below.
  2. ``--score_overlaps`` does not exist. The flag is ``--ignore_overlaps``, and
     *not* passing it is what scores overlapped speech, which is what our
     ``skip_overlap: false`` wants. That half was accidentally right.
  3. **The two tools mean different things by "collar".** pyannote.metrics
     centres a window of total width ``collar`` on each boundary
     (``pyannote/metrics/utils.py``: ``Segment(t - .5*collar, t + .5*collar)``),
     while md-eval — which dscore wraps — applies ``collar`` to *each side*. So
     ``dscore --collar X`` corresponds to ``pyannote collar=2X``, and a naive
     ``--collar 0.25`` would silently score a doubled forgiveness window.

Measured on ``artifacts/2026-07-31-callhome-de/.../callhome-deu-0000.rttm``
against dscore @ ``e02f949``:

    pyannote collar=0.00  24.7724   dscore --collar 0.0    24.77
    pyannote collar=0.25  21.5256   dscore --collar 0.125  21.50
    pyannote collar=0.50  21.4404   dscore --collar 0.25   21.40

which is the conversion above, confirmed end to end. :data:`DSCORE_COLLAR_FACTOR`
carries it, and ``tests/test_dscore_check.py`` asserts the flag is really passed
— that test needs no dscore checkout, so the bug cannot come back the same way.

Why this is OPTIONAL, not a pinned dependency
----------------------------------------------
dscore (github.com/nryant/dscore) is a **repository of scripts**, not a
pip-installable package — there is no PyPI wheel to add to ``pyproject`` extras
and pin cleanly. So instead of faking a dependency, this check is gated on a
local checkout the user pins by commit:

    git clone https://github.com/nryant/dscore ~/dscore
    git -C ~/dscore checkout <DSCORE_PINNED_COMMIT>
    export DSCORE_DIR=~/dscore
    make dscore-check GOLD=... HYP=...

It also needs dscore's own runtime deps (``numpy``, ``scipy``, ``intervaltree``,
``tabulate``) and ``perl`` for the bundled ``md-eval-22.pl``.

When ``DSCORE_DIR`` is unset / missing, the check SKIPS with a clear message (exit
0, ``skipped``) — it never silently passes as if it ran, and it is deliberately
NOT wired into CI (CI has no dscore checkout, and pinning it as a dep is exactly
what's infeasible).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from raven_eval_core.der import compute_der_corpus, load_rttm

from .config import COLLARS

# Pin dscore by commit for reproducibility. This is the commit the cross-check
# was validated against on 2026-09-04; bump it deliberately (not floating) when
# re-verifying. The previous value, "f2d33d3", is not a commit in that repository
# at all — a pin nobody had ever resolved, which is the same class of defect as
# the missing collar flag: a claim that had never been executed.
DSCORE_PINNED_COMMIT = "e02f949ac6592279300a2c33d03daf9e0c12fd27"

#: ``dscore --collar X`` == ``pyannote collar = 2X``. md-eval applies the collar
#: to each side of a boundary; pyannote.metrics centres a window of that total
#: width on it. Halving is not a fudge factor — it is the unit conversion, and
#: without it the cross-check compares two different forgiveness windows.
DSCORE_COLLAR_FACTOR = 0.5

# Agreement tolerance between dscore and pyannote.metrics, in percentage points.
# The two implement the same md-eval DER; small deltas come from collar edge
# handling and dscore's 10 ms scoring step. 0.5 pp is tight enough to catch a
# real methodology divergence and loose enough for that quantisation.
CROSSCHECK_TOLERANCE_PCT = 0.5


def _dscore_dir() -> Path | None:
    d = os.environ.get("DSCORE_DIR")
    if not d:
        return None
    p = Path(d).expanduser()
    return p if (p / "score.py").exists() else None


def dscore_argv(
    gold_rttm: Path, hyp_rttm: Path, dscore_dir: Path, collar: float
) -> list[str]:
    """The exact command line used to score one pair with dscore.

    Split out so a test can assert the collar really reaches dscore, converted,
    without needing a checkout of it. ``--ignore_overlaps`` is deliberately NOT
    passed: our contract scores overlapped speech (``skip_overlap: false``).

    RTTM paths are resolved to absolute: ``score.py`` is run with ``cwd`` set to
    the dscore checkout so it can find its bundled ``md-eval-22.pl``, and a
    relative path handed to it there resolves against the wrong directory. That
    is only visible once the check is actually executed, which it never had been.
    """
    return [
        sys.executable, str(dscore_dir / "score.py"),
        "-r", str(Path(gold_rttm).resolve()),
        "-s", str(Path(hyp_rttm).resolve()),
        "--collar", f"{collar * DSCORE_COLLAR_FACTOR:g}",
    ]


def run_dscore(
    gold_rttm: Path, hyp_rttm: Path, dscore_dir: Path, collar: float
) -> float:
    """Run dscore's score.py on one (gold, hyp) pair → overall DER (percent).

    ``collar`` is given in OUR units (pyannote's total-width convention) and
    converted on the way out.
    """
    proc = subprocess.run(
        dscore_argv(gold_rttm, hyp_rttm, dscore_dir, collar),
        capture_output=True, text=True, cwd=str(dscore_dir),
        check=False,  # parse the DER table even if dscore exits nonzero
    )
    out = proc.stdout + "\n" + proc.stderr
    # dscore prints a table; the OVERALL row's first float is the DER (percent).
    for line in out.splitlines():
        if "OVERALL" in line.upper():
            nums = re.findall(r"[0-9]+\.[0-9]+", line)
            if nums:
                return float(nums[0])
    raise RuntimeError(
        f"could not parse dscore OVERALL DER from output:\n{out[-2000:]}"
    )


def crosscheck(
    gold_rttm: Path,
    hyp_rttm: Path,
    tolerance_pct: float = CROSSCHECK_TOLERANCE_PCT,
) -> dict[str, object]:
    """Compare dscore vs pyannote.metrics DER on one pair, at BOTH collars.

    Both, because a cross-check at one collar only proves agreement at that
    collar — and the collar is exactly where the two implementations differ in
    convention. Returns ``{status, variants: {name: {...}}, detail}``. ``status``
    is ``"skipped"`` when no ``DSCORE_DIR`` checkout is present, else ``"pass"``
    if every variant agrees within tolerance and ``"fail"`` otherwise.
    """
    pair = [(load_rttm(gold_rttm), load_rttm(hyp_rttm))]
    ours = {
        name: compute_der_corpus(
            pair,
            collar=float(cfg["collar"]),
            skip_overlap=bool(cfg["skip_overlap"]),
        ).der * 100.0
        for name, cfg in COLLARS.items()
    }

    dscore_dir = _dscore_dir()
    if dscore_dir is None:
        return {
            "status": "skipped",
            "variants": {n: {"ours_pct": v} for n, v in ours.items()},
            "detail": "DSCORE_DIR unset or has no score.py — cross-check skipped "
            f"(pin nryant/dscore @ {DSCORE_PINNED_COMMIT[:7]}, export DSCORE_DIR).",
        }

    variants: dict[str, dict[str, float]] = {}
    worst = 0.0
    for name, cfg in COLLARS.items():
        collar = float(cfg["collar"])
        theirs = run_dscore(gold_rttm, hyp_rttm, dscore_dir, collar)
        delta = abs(ours[name] - theirs)
        worst = max(worst, delta)
        variants[name] = {
            "collar": collar,
            "dscore_collar": collar * DSCORE_COLLAR_FACTOR,
            "ours_pct": ours[name],
            "dscore_pct": theirs,
            "delta_pct": delta,
        }
    return {
        "status": "pass" if worst <= tolerance_pct else "fail",
        "variants": variants,
        "detail": f"worst |Δ|={worst:.3f} pp across {len(variants)} collar(s) "
                  f"(tol {tolerance_pct})",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-diar-dscore-check", description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="gold RTTM")
    parser.add_argument("--hyp", type=Path, required=True, help="hypothesis RTTM")
    parser.add_argument("--tolerance", type=float, default=CROSSCHECK_TOLERANCE_PCT)
    args = parser.parse_args(argv)

    result = crosscheck(args.gold, args.hyp, tolerance_pct=args.tolerance)
    for name, v in result["variants"].items():           # type: ignore[union-attr]
        if "dscore_pct" in v:
            print(f"  {name:8s} pyannote collar={v['collar']:.2f} -> "
                  f"{v['ours_pct']:.3f} % | dscore --collar "
                  f"{v['dscore_collar']:g} -> {v['dscore_pct']:.3f} % | "
                  f"Δ {v['delta_pct']:.3f} pp")
        else:
            print(f"  {name:8s} pyannote -> {v['ours_pct']:.3f} % (not cross-checked)")
    print(result["detail"])
    if result["status"] == "skipped":
        print("dscore cross-check SKIPPED (optional — see module docstring).")
        return 0
    if result["status"] == "pass":
        print("dscore cross-check PASS: pyannote.metrics and dscore agree.")
        return 0
    print("dscore cross-check FAIL: DER disagreement exceeds tolerance.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
