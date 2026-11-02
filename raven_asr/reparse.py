"""Re-derive recorded predictions from the server output they hold — no inference.

A model's output format (:mod:`raven_asr.output_formats`) can be learnt after
the model was measured: MOSS-Transcribe-Diarize was run and scored with its
timestamp and speaker markup still in ``prediction``. Its transcripts are
right; only their reading was wrong. Re-running inference to fix a parse would
spend GPU hours on an answer the predictions file already holds.

So this tool takes a runner output dir (``results/.../<label>/``, the layout
``raven_asr.reproduce`` writes and ``raven_asr.promote`` reads) and, for every
``predictions_<subset>.jsonl`` in it:

* takes the server's output of each line — ``raw_prediction`` when the line has
  one, else ``prediction`` (a line recorded before the model had a parser holds
  nothing else);
* parses it with the model's registered parser into ``prediction`` (spoken text),
  ``raw_prediction`` (unchanged server output) and ``segments``;
* rewrites the file atomically (temp file, fsync, rename);
* re-scores every subset it touched through the runner's own scoring
  (:func:`raven_asr.runner.rescore_recorded`): resume markers, ``summary.json``
  and ``model-index.yaml``, with a ``post_hoc`` entry naming what was applied.

It is idempotent: a second run parses the same raw text into the same lines.
It refuses a model without a registered output format — there is nothing to
re-derive, and rewriting its predictions would only blur what the server said.
Failed lines (``"prediction": null``) are left as they are.

Run it on a run's output before ``raven_asr.promote``; a committed artifact is
never rewritten here (``make verify`` holds those to their seal):

    uv run --extra asr python -m raven_asr.reparse results/<run>/<label>
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import KNOWN_MODELS, ModelSpec
from .output_formats import ParsedOutput, parser_for

_PRED_RE = re.compile(r"^predictions_(?P<subset>.+)\.jsonl$")


class NoParserError(ValueError):
    """The run's model has no output format, so there is nothing to re-derive."""


@dataclass(frozen=True)
class SubsetReport:
    subset: str
    n_lines: int
    n_changed: int


def spec_for_run(run_dir: Path) -> ModelSpec:
    """The registry entry of the model a runner output dir was measured with."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    model_id = summary.get("model_id")
    if model_id in KNOWN_MODELS:
        return KNOWN_MODELS[model_id]
    for spec in KNOWN_MODELS.values():
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"{run_dir}: model {model_id!r} is not in raven_asr.config.KNOWN_MODELS")


def reparse_line(line: dict[str, Any], parse) -> dict[str, Any]:
    """One predictions line re-derived from its server output."""
    if line.get("error") is not None or not isinstance(line.get("prediction"), str):
        return line
    raw = line.get("raw_prediction")
    if not isinstance(raw, str):
        raw = line["prediction"]
    parsed: ParsedOutput = parse(raw)
    out: dict[str, Any] = {}
    for key, value in line.items():
        if key in ("raw_prediction", "segments"):
            continue
        out[key] = value
        if key == "prediction":
            out["prediction"] = parsed.text
    # Same key order the runner writes: raw and segments after the base fields.
    out["raw_prediction"] = raw
    out["segments"] = [seg.as_json() for seg in parsed.segments]
    return out


def _write_atomic(path: Path, lines: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def reparse_run(run_dir: Path, *, dry_run: bool = False) -> list[SubsetReport]:
    """Re-derive every predictions file in ``run_dir``; re-score what changed.

    Raises :class:`NoParserError` for a model without an output format and
    ``FileNotFoundError`` for a dir that is not a runner output.
    """
    if not (run_dir / "summary.json").is_file():
        raise FileNotFoundError(f"no summary.json in {run_dir} — not a runner output dir")
    spec = spec_for_run(run_dir)
    parse = parser_for(spec.output_format)
    if parse is None:
        raise NoParserError(
            f"{spec.model_id} has no output format in raven_asr.config.OUTPUT_FORMATS: "
            "its predictions are the server's text already, nothing to re-derive"
        )
    reports: list[SubsetReport] = []
    changed_subsets: list[str] = []
    for path in sorted(run_dir.glob("predictions_*.jsonl")):
        m = _PRED_RE.match(path.name)
        assert m
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
        new = [reparse_line(line, parse) for line in lines]
        n_changed = sum(1 for a, b in zip(lines, new, strict=True) if a != b)
        reports.append(SubsetReport(m.group("subset"), len(lines), n_changed))
        if n_changed and not dry_run:
            _write_atomic(path, new)
            changed_subsets.append(m.group("subset"))
    if changed_subsets:
        # Imported here: the runner pulls the inference stack (numpy, tqdm),
        # which a dry run has no use for.
        from .runner import rescore_recorded

        rescore_recorded(
            run_dir, spec, changed_subsets,
            {
                "step": "parse",
                "output_format": spec.output_format,
                "tool": "raven_asr.reparse",
                "subsets": changed_subsets,
                "at": datetime.datetime.now(datetime.timezone.utc)
                .isoformat(timespec="seconds").replace("+00:00", "Z"),
            },
        )
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-asr-reparse", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="runner output dir(s), each holding summary.json + predictions")
    parser.add_argument("--dry-run", action="store_true",
                        help="report how many lines would change, write nothing")
    args = parser.parse_args(argv)
    status = 0
    for run_dir in args.run_dirs:
        try:
            reports = reparse_run(run_dir, dry_run=args.dry_run)
        except (NoParserError, FileNotFoundError, KeyError) as exc:
            print(f"SKIP {run_dir}: {exc}", file=sys.stderr)
            status = 2
            continue
        verb = "would change" if args.dry_run else "changed"
        for r in reports:
            print(f"{run_dir}: {r.subset}: {verb} {r.n_changed} of {r.n_lines} line(s)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
