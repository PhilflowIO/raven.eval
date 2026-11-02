"""Re-deriving recorded MOSS predictions without inference (raven_asr.reparse).

The fixture is a real excerpt of a MOSS-Transcribe-Diarize run recorded before
the model had a parser — ``prediction`` holds the server's markup — two lines
each of fleurs (CC-BY-4.0), spc-test (MIT) and voxpopuli-de (CC0); attribution
in /NOTICE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("tqdm")

from raven_asr import runner
from raven_asr.reparse import NoParserError, main, reparse_run
from raven_eval_core import corpus_wer_strict_de_pct

FIXTURE = Path(__file__).parent / "fixtures" / "moss-transcribe-diarize"
MOSS = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
SUBSETS = ("fleurs", "spc-test", "voxpopuli-de")


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _recorded_run(tmp_path: Path, model: str = MOSS) -> Path:
    """A runner output dir as the sweep left it: raw markup scored as speech."""
    out = tmp_path / "moss-transcribe-diarize"
    out.mkdir()
    spec = runner.KNOWN_MODELS[model]
    manifest = runner._run_manifest(spec, None, None)
    entries: list = []
    for subset in SUBSETS:
        outcomes = [
            runner._Outcome(
                sample_id=line["sample_id"], reference=line["reference"],
                duration_s=line["duration_s"], prediction=line["prediction"],
                latency_s=line["latency_s"],
            )
            for line in _lines(FIXTURE / f"predictions_{subset}.jsonl")
        ]
        runner._record_subset(
            out_dir=out, spec=spec, result=runner._RunResult(subset, outcomes),
            entries=entries, limit=None, manifest=manifest,
        )
    return out


def _rows(out: Path) -> dict[str, dict]:
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return {r["subset"]: r for r in summary["results"]}


def test_reparse_strips_markup_keeps_raw_and_rescores(tmp_path: Path):
    out = _recorded_run(tmp_path)
    before = _rows(out)
    raw = {s: _lines(out / f"predictions_{s}.jsonl") for s in SUBSETS}

    reports = reparse_run(out)

    assert {r.subset: r.n_changed for r in reports} == {s: 2 for s in SUBSETS}
    after = _rows(out)
    for subset in SUBSETS:
        lines = _lines(out / f"predictions_{subset}.jsonl")
        for old, new in zip(raw[subset], lines, strict=True):
            assert new["raw_prediction"] == old["prediction"]  # byte for byte
            assert "[S0" not in new["prediction"]
            assert isinstance(new["segments"], list)
            assert {k: new[k] for k in ("reference", "sample_id", "duration_s", "latency_s")} == {
                k: old[k] for k in ("reference", "sample_id", "duration_s", "latency_s")
            }
        # The published flozi-strict lens drops bracketed text itself, so raw
        # markup cost it only the words it glued together; never more after.
        assert after[subset]["wer_pct"] <= before[subset]["wer_pct"]
        # The strict-de lens read every timestamp and tag as a word (~100 % WER).
        assert corpus_wer_strict_de_pct(
            [x["reference"] for x in lines], [x["prediction"] for x in lines]
        ) < corpus_wer_strict_de_pct(
            [x["reference"] for x in raw[subset]], [x["prediction"] for x in raw[subset]]
        )
        marker = json.loads((out / f".done_{subset}.json").read_text(encoding="utf-8"))
        assert marker["wer_pct"] == pytest.approx(after[subset]["wer_pct"], abs=1e-4)
    # "schutz und[4.62][4.62]eintrittsgebühren" (fleurs-586) no longer glues
    # two words into one error.
    assert after["fleurs"]["wer_pct"] < before["fleurs"]["wer_pct"]

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["output_format"] == "moss-timed-speakers"
    (step,) = summary["post_hoc"]
    assert step["step"] == "parse" and step["subsets"] == list(SUBSETS)
    # What the run measured is not re-derived: same manifest as recorded.
    assert summary["run_manifest"]["git"] == runner._run_manifest(
        runner.KNOWN_MODELS[MOSS], None, None)["git"]


def test_reparse_is_idempotent(tmp_path: Path):
    out = _recorded_run(tmp_path)
    reparse_run(out)
    first = {s: (out / f"predictions_{s}.jsonl").read_bytes() for s in SUBSETS}
    reports = reparse_run(out)
    assert all(r.n_changed == 0 for r in reports)
    assert {s: (out / f"predictions_{s}.jsonl").read_bytes() for s in SUBSETS} == first
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["post_hoc"]) == 1  # nothing applied, nothing recorded


def test_dry_run_writes_nothing(tmp_path: Path):
    out = _recorded_run(tmp_path)
    snapshot = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    reports = reparse_run(out, dry_run=True)
    assert sum(r.n_changed for r in reports) == 6
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == snapshot


def test_a_model_without_a_parser_or_a_cap_is_refused_untouched(tmp_path: Path):
    # A hosted API: no output format, no serving recipe.
    out = _recorded_run(tmp_path, model="xai/grok-voice-transcribe-2.0")
    snapshot = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    with pytest.raises(NoParserError):
        reparse_run(out)
    assert main([str(out)]) == 2
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == snapshot


def test_failed_lines_stay_failed(tmp_path: Path):
    out = _recorded_run(tmp_path)
    path = out / "predictions_fleurs.jsonl"
    lines = _lines(path)
    lines.append({"reference": "x", "prediction": None, "latency_s": None,
                  "sample_id": "fleurs-9", "duration_s": 1.0, "error": "ReadTimeout: "})
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
                    encoding="utf-8")
    reparse_run(out)
    assert _lines(path)[-1] == lines[-1]


def test_new_lines_match_what_a_live_run_writes(tmp_path: Path):
    """A reparsed line and a line the adapter parsed live are the same line."""
    out = _recorded_run(tmp_path)
    reparse_run(out)
    line = _lines(out / "predictions_fleurs.jsonl")[0]
    from raven_asr.output_formats import parse_moss

    parsed = parse_moss(line["raw_prediction"])
    live = runner._prediction_line(runner._Outcome(
        sample_id=line["sample_id"], reference=line["reference"],
        duration_s=line["duration_s"], prediction=parsed.text,
        latency_s=line["latency_s"], raw_prediction=line["raw_prediction"],
        segments=[s.as_json() for s in parsed.segments],
    ))
    assert list(live) == list(line) and live == line



# ── Post-hoc output-length cap (#62) ─────────────────────────────────────────

PHI4 = "microsoft/Phi-4-multimodal-instruct"
PHI4_FIXTURE = Path(__file__).parent / "fixtures" / "phi-4-multimodal"


class _WordTokenizer:
    """A stand-in tokenizer: one token per whitespace-separated piece."""

    calls = 0

    def __init__(self, cap) -> None:
        type(self).calls += 1
        self.cap = cap

    def encode(self, text: str) -> list[int]:
        self._pieces = text.split(" ")
        return list(range(len(self._pieces)))

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._pieces[i] for i in ids)


def _phi4_run(tmp_path: Path) -> Path:
    """A Phi-4 spc-test excerpt as the 2026-11 sweep recorded it (one loop)."""
    out = tmp_path / "phi-4-multimodal"
    out.mkdir()
    spec = runner.KNOWN_MODELS[PHI4]
    outcomes = [
        runner._Outcome(sample_id=x["sample_id"], reference=x["reference"],
                        duration_s=x["duration_s"], prediction=x["prediction"],
                        latency_s=x["latency_s"])
        for x in _lines(PHI4_FIXTURE / "predictions_spc-test.jsonl")
    ]
    runner._record_subset(out_dir=out, spec=spec,
                          result=runner._RunResult("spc-test", outcomes), entries=[],
                          limit=None, manifest=runner._run_manifest(spec, None, None))
    return out


def test_the_cap_cuts_only_the_loop_and_records_itself(tmp_path: Path):
    out = _phi4_run(tmp_path)
    before = _rows(out)["spc-test"]
    raw = _lines(out / "predictions_spc-test.jsonl")
    _WordTokenizer.calls = 0

    (report,) = reparse_run(out, tokenizer=_WordTokenizer)

    assert (report.n_changed, report.n_capped) == (1, 1)
    assert _WordTokenizer.calls == 1  # loaded once, and only for the long line
    lines = _lines(out / "predictions_spc-test.jsonl")
    for old, new in zip(raw, lines, strict=True):
        if new.get("capped_post_hoc"):
            assert new["raw_prediction"] == old["prediction"]  # server output kept
            assert len(new["prediction"].split(" ")) == 1024
            assert "segments" not in new  # Phi-4 has no output format
        else:
            assert new == old  # short lines are not touched at all
    after = _rows(out)["spc-test"]
    assert after["wer_pct"] < before["wer_pct"]
    assert after["n_runaway"] == before["n_runaway"] == 1  # still a loop, still counted
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    (step,) = summary["post_hoc"]
    assert step["step"] == "cap" and step["capped_post_hoc"] is True
    assert step["max_tokens"] == 1024 and step["exact"] is False
    assert step["tokenizer"]["revision"] == "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
    assert step["n_capped"] == {"spc-test": 1}


def test_the_cap_is_idempotent(tmp_path: Path):
    out = _phi4_run(tmp_path)
    reparse_run(out, tokenizer=_WordTokenizer)
    first = (out / "predictions_spc-test.jsonl").read_bytes()
    (report,) = reparse_run(out, tokenizer=_WordTokenizer)
    assert report.n_changed == 0
    assert (out / "predictions_spc-test.jsonl").read_bytes() == first


def test_no_cap_leaves_a_plain_model_with_nothing_to_do(tmp_path: Path):
    out = _phi4_run(tmp_path)
    with pytest.raises(NoParserError):
        reparse_run(out, cap=False)


def test_moss_is_capped_before_it_is_parsed(tmp_path: Path):
    """The cap acts on the server's output, markup included, then the parse."""
    out = _recorded_run(tmp_path)
    path = out / "predictions_fleurs.jsonl"
    lines = _lines(path)
    lines[0]["prediction"] = "[0.0][S01] " + "sehr " * 3000 + "[9.0]"
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
                    encoding="utf-8")
    reparse_run(out, tokenizer=_WordTokenizer)
    line = _lines(path)[0]
    assert line["capped_post_hoc"] is True
    assert line["raw_prediction"].endswith("[9.0]")
    assert "[" not in line["prediction"] and line["prediction"].count("sehr") == 1023
    steps = [s["step"] for s in json.loads((out / "summary.json").read_text())["post_hoc"]]
    assert steps == ["cap", "parse"]
