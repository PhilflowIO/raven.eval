"""The dscore cross-check must actually cross-check.

The check had never run: it passed no collar flag, so it compared our collar-0.25
DER against dscore's collar-0.0 default, and it named a pinned commit that does
not exist in that repository. Neither defect is reachable by a test that needs a
dscore checkout — CI has none, and that is exactly why both survived. So the
tests here stand in a fake ``score.py``: they exercise the command line and the
parsing, which is where both bugs lived.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from raven_diar.config import COLLARS
from raven_diar.dscore_check import (
    DSCORE_COLLAR_FACTOR,
    DSCORE_PINNED_COMMIT,
    crosscheck,
    dscore_argv,
)
from raven_eval_core.der import to_rttm

#: A dscore stand-in: prints the OVERALL row dscore prints, echoing back the
#: collar it was given so a test can see what actually arrived.
FAKE_SCORE_PY = '''\
import sys
collar = 0.0
for i, a in enumerate(sys.argv):
    if a == "--collar":
        collar = float(sys.argv[i + 1])
print("*** OVERALL ***    {:.2f}  25.74  0.66".format(100.0 * collar + 1.0))
'''


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    gold = tmp_path / "gold.rttm"
    hyp = tmp_path / "hyp.rttm"
    gold.write_text(to_rttm([(0.0, 10.0, "A"), (10.0, 20.0, "B")], file_id="f1"))
    hyp.write_text(to_rttm([(0.0, 9.0, "A"), (11.0, 20.0, "B")], file_id="f1"))
    return gold, hyp


@pytest.fixture
def fake_dscore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "dscore"
    d.mkdir()
    (d / "score.py").write_text(FAKE_SCORE_PY)
    monkeypatch.setenv("DSCORE_DIR", str(d))
    return d


def test_the_collar_reaches_dscore_at_all(pair, tmp_path: Path):
    """The bug: no --collar in the command line, so dscore used its 0.0 default."""
    gold, hyp = pair
    argv = dscore_argv(gold, hyp, tmp_path, collar=0.25)
    assert "--collar" in argv


def test_the_collar_is_converted_to_dscore_units(pair, tmp_path: Path):
    """dscore --collar X == pyannote collar 2X — md-eval applies it per side.

    Measured against dscore @ e02f949 on a real CALLHOME-de file: our collar 0.25
    reads 21.53 and dscore reads 21.50 at --collar 0.125, 21.40 at --collar 0.25.
    Passing our number through unconverted scores a doubled forgiveness window.
    """
    gold, hyp = pair
    argv = dscore_argv(gold, hyp, tmp_path, collar=0.25)
    assert argv[argv.index("--collar") + 1] == "0.125"
    assert DSCORE_COLLAR_FACTOR == 0.5


def test_overlapped_speech_is_scored_not_ignored(pair, tmp_path: Path):
    """Our contract is skip_overlap=false, so --ignore_overlaps must be absent.

    ``--score_overlaps``, which the old docstring named, is not a dscore flag at
    all; passing it would have made dscore exit on an unrecognised argument.
    """
    gold, hyp = pair
    argv = dscore_argv(gold, hyp, tmp_path, collar=0.0)
    assert "--ignore_overlaps" not in argv
    assert "--score_overlaps" not in argv


def test_every_published_collar_is_cross_checked(pair, fake_dscore):
    """One collar proves agreement at one collar — and the collar is the risk."""
    gold, hyp = pair
    result = crosscheck(gold, hyp, tolerance_pct=1e9)
    assert set(result["variants"]) == set(COLLARS)
    for name, cfg in COLLARS.items():
        v = result["variants"][name]
        assert v["collar"] == pytest.approx(float(cfg["collar"]))
        assert v["dscore_collar"] == pytest.approx(
            float(cfg["collar"]) * DSCORE_COLLAR_FACTOR
        )


def test_the_fake_dscore_sees_the_converted_collar(pair, fake_dscore):
    """End-to-end through subprocess: the stand-in echoes 100*collar + 1."""
    gold, hyp = pair
    result = crosscheck(gold, hyp, tolerance_pct=1e9)
    # classic = 0.25 our units -> 0.125 dscore units -> 100*0.125 + 1 = 13.50
    assert result["variants"]["classic"]["dscore_pct"] == pytest.approx(13.50)
    assert result["variants"]["full"]["dscore_pct"] == pytest.approx(1.00)


def test_a_disagreement_fails_rather_than_passing_quietly(pair, fake_dscore):
    gold, hyp = pair
    result = crosscheck(gold, hyp, tolerance_pct=0.5)
    assert result["status"] == "fail"


def test_no_checkout_skips_and_says_so(pair, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DSCORE_DIR", raising=False)
    gold, hyp = pair
    result = crosscheck(gold, hyp)
    assert result["status"] == "skipped"
    assert "DSCORE_DIR" in str(result["detail"])
    # Our own numbers are still reported — a skip is not an absence of data.
    assert set(result["variants"]) == set(COLLARS)


def test_the_pinned_commit_is_shaped_like_a_commit():
    """The old pin, "f2d33d3", is not a commit in nryant/dscore at all.

    A short hash is also not a pin worth having: it cannot be verified without
    network access and it can become ambiguous. Full hash, or it is a note.
    """
    assert re.fullmatch(r"[0-9a-f]{40}", DSCORE_PINNED_COMMIT)
