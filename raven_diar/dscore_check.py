"""Optional cross-check: does dscore agree with our pyannote.metrics DER?

Makes the "dscore cross-check planned for Etappe 5" note in ``BENCHMARKS.md`` /
``benchmark.config.yaml`` REAL — a runnable check (``make dscore-check``) that
scores the SAME gold/hyp RTTMs with nryant/dscore and asserts its DER agrees with
``raven_eval_core.der`` within a small tolerance.

Why this is OPTIONAL, not a pinned dependency
----------------------------------------------
dscore (github.com/nryant/dscore) is a **repository of scripts**, not a
pip-installable package — there is no PyPI wheel to add to ``pyproject`` extras
and pin cleanly. So instead of faking a dependency, this check is gated on a
local checkout the user pins by commit:

    git clone https://github.com/nryant/dscore ~/dscore
    git -C ~/dscore checkout <DSCORE_PINNED_COMMIT>
    export DSCORE_DIR=~/dscore
    make dscore-check                     # or: python -m raven_diar.dscore_check ...

When ``DSCORE_DIR`` is unset / missing, the check SKIPS with a clear message (exit
0, ``skipped``) — it never silently passes as if it ran, and it is deliberately
NOT wired into CI (CI has no dscore checkout, and pinning it as a dep is exactly
what's infeasible). The scoring collar it uses is dscore's default (0.25,
``--score_overlaps`` on) to line up with our ``classic`` variant.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from raven_eval_core.der import compute_der_corpus, load_rttm

# Pin dscore by commit for reproducibility. This is the commit the cross-check is
# validated against; bump it deliberately (not floating) when re-verifying.
DSCORE_PINNED_COMMIT = "f2d33d3"  # nryant/dscore, master @ 2023 (score.py CLI)

# Agreement tolerance between dscore and pyannote.metrics, in percentage points.
# The two implement the same md-eval DER; small deltas come from collar edge
# handling. 0.5 pp is tight enough to catch a real methodology divergence.
CROSSCHECK_TOLERANCE_PCT = 0.5

_DER_LINE_RE = re.compile(r"(?:OVERALL|\*\*\*\s*OVERALL).*?([0-9]+\.[0-9]+)")


def _dscore_dir() -> Path | None:
    d = os.environ.get("DSCORE_DIR")
    if not d:
        return None
    p = Path(d).expanduser()
    return p if (p / "score.py").exists() else None


def run_dscore(gold_rttm: Path, hyp_rttm: Path, dscore_dir: Path) -> float:
    """Run dscore's score.py on one (gold, hyp) pair → overall DER (percent)."""
    proc = subprocess.run(
        [sys.executable, str(dscore_dir / "score.py"),
         "-r", str(gold_rttm), "-s", str(hyp_rttm)],
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
    """Compare dscore vs pyannote.metrics DER on one pair.

    Returns ``{status, ours_pct, dscore_pct?, delta_pct?, detail}``. ``status`` is
    ``"skipped"`` when no ``DSCORE_DIR`` checkout is present, ``"pass"`` /
    ``"fail"`` otherwise. Uses the ``classic`` collar (0.25) to match dscore's
    default.
    """
    ours = (
        compute_der_corpus(
            [(load_rttm(gold_rttm), load_rttm(hyp_rttm))],
            collar=0.25,
            skip_overlap=False,
        ).der
        * 100.0
    )
    dscore_dir = _dscore_dir()
    if dscore_dir is None:
        return {
            "status": "skipped",
            "ours_pct": ours,
            "detail": "DSCORE_DIR unset or has no score.py — cross-check skipped "
            f"(pin nryant/dscore @ {DSCORE_PINNED_COMMIT}, export DSCORE_DIR).",
        }
    theirs = run_dscore(gold_rttm, hyp_rttm, dscore_dir)
    delta = abs(ours - theirs)
    return {
        "status": "pass" if delta <= tolerance_pct else "fail",
        "ours_pct": ours,
        "dscore_pct": theirs,
        "delta_pct": delta,
        "detail": f"|Δ|={delta:.3f} pp (tol {tolerance_pct})",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-diar-dscore-check", description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="gold RTTM")
    parser.add_argument("--hyp", type=Path, required=True, help="hypothesis RTTM")
    parser.add_argument("--tolerance", type=float, default=CROSSCHECK_TOLERANCE_PCT)
    args = parser.parse_args(argv)

    result = crosscheck(args.gold, args.hyp, tolerance_pct=args.tolerance)
    print(result)
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
