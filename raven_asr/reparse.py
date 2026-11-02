"""Re-derive recorded predictions from the server output they hold — no inference.

Two things about a model's output can be settled after the model was measured,
and both are functions of the server output a run already recorded:

* **its output format** (:mod:`raven_asr.output_formats`). MOSS-Transcribe-
  Diarize was run and scored with its timestamp and speaker markup still in
  ``prediction``. Its transcripts are right; only their reading was wrong.
* **its output-length cap** (``serving/recipes.toml`` → ``decoding.max_tokens``).
  Until the cap was part of the recipe, a looping model generated until an
  engine default stopped it — 4,096 tokens for Phi-4, other limits elsewhere —
  and that default decided how much a loop cost. Decoding is greedy, so the
  first N tokens of the longer output are what a server capped at N returns;
  the recorded output is cut to N tokens of the model's own tokenizer, at the
  recipe's pinned revision.

Re-running inference for either would spend GPU hours on an answer the
predictions file already holds. So this tool takes a runner output dir
(``results/.../<label>/``, the layout ``raven_asr.reproduce`` writes and
``raven_asr.promote`` reads) and, for every ``predictions_<subset>.jsonl``:

* takes the server's output of each line — ``raw_prediction`` when the line has
  one, else ``prediction`` (a line recorded before holds nothing else);
* cuts it to the recipe's ``max_tokens`` when it is longer, marking the line
  ``"capped_post_hoc": true``;
* parses the result with the model's output format, if it has one;
* writes ``prediction`` (what is scored), ``raw_prediction`` (the server output,
  unchanged) and, for a parsed model, ``segments``, atomically;
* re-scores every subset it touched through the runner's own scoring
  (:func:`raven_asr.runner.rescore_recorded`): resume markers, ``summary.json``
  and ``model-index.yaml``, with a ``post_hoc`` entry per step applied.

**The cap is an approximation, and says so** (``"exact": false`` in its
``post_hoc`` entry). What a server caps is generated token ids; what a run
records is their decoded text, which is re-tokenized here, and a tokenizer need
not return the ids a model generated (on the 2026-11 Phi-4 run every looping
answer re-tokenized to 4,096 tokens — 4,095 once — against a server cap of
4,096). vLLM also splits audio longer than a model's clip limit and caps each
piece, where this caps the joined output. The next run with the recipe's cap
applied server-side is the exact measurement.

It is idempotent: a second run derives the same lines from the same raw text.
It refuses a model with neither an output format nor a recipe cap — there is
nothing to re-derive. Failed lines (``"prediction": null``) are left as they are.

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
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .config import KNOWN_MODELS, ModelSpec
from .output_formats import ParsedOutput, parser_for

_PRED_RE = re.compile(r"^predictions_(?P<subset>.+)\.jsonl$")
RECIPES_FILE = Path(__file__).resolve().parents[1] / "serving" / "recipes.toml"


class NoParserError(ValueError):
    """The run's model has neither an output format nor a recipe cap."""


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class Cap:
    """The recipe's output-length cap and where its tokenizer comes from."""

    max_tokens: int
    repo: str
    revision: str


@dataclass(frozen=True)
class SubsetReport:
    subset: str
    n_lines: int
    n_changed: int
    n_capped: int = 0


@dataclass
class _Plan:
    parse: Callable[[str], ParsedOutput] | None
    cap: Cap | None
    tokenizer: Callable[[], Tokenizer] | None
    _tok: Tokenizer | None = field(default=None, repr=False)

    def tok(self) -> Tokenizer:
        if self._tok is None:
            assert self.tokenizer is not None
            self._tok = self.tokenizer()
        return self._tok


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


def recipe_cap(model_id: str, recipes_file: Path = RECIPES_FILE) -> Cap | None:
    """``decoding.max_tokens`` of the model's serving recipe, or None.

    None for a hosted model (no recipe) and for a NeMo recipe, whose length bound
    is structural (max symbols per frame) rather than a token cap.
    """
    try:
        recipes = tomllib.loads(recipes_file.read_text(encoding="utf-8"))["recipes"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None
    for recipe in recipes.values():
        if recipe.get("model") == model_id:
            max_tokens = (recipe.get("decoding") or {}).get("max_tokens")
            if isinstance(max_tokens, int):
                return Cap(max_tokens, recipe["model"], recipe["revision"])
    return None


class _HFTokenizer:
    """The model's own tokenizer at the recipe revision.

    The ``tokenizer.json`` the repo ships, read with ``tokenizers``, where there
    is one; otherwise (Qwen3-ASR, some Whisper fine-tunes ship vocab + merges
    only) ``transformers.AutoTokenizer`` builds it from the repo's files, never
    running remote code. :attr:`source` names which, for the ``post_hoc`` record.
    """

    def __init__(self, cap: Cap) -> None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            path = hf_hub_download(cap.repo, "tokenizer.json", revision=cap.revision)
        except EntryNotFoundError:
            import transformers

            tok = transformers.AutoTokenizer.from_pretrained(
                cap.repo, revision=cap.revision, trust_remote_code=False
            )
            self._encode = lambda text: list(tok.encode(text, add_special_tokens=False))
            self._decode = lambda ids: tok.decode(ids)
            self.source = f"transformers {transformers.__version__} AutoTokenizer"
            return
        from tokenizers import Tokenizer as _Tok

        tok = _Tok.from_file(path)
        self._encode = lambda text: list(tok.encode(text, add_special_tokens=False).ids)
        self._decode = lambda ids: tok.decode(ids)
        self.source = "tokenizer.json"

    def encode(self, text: str) -> list[int]:
        return self._encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._decode(ids)


def cap_text(raw: str, max_tokens: int, tok: Callable[[], Tokenizer]) -> tuple[str, bool]:
    """``raw`` cut to ``max_tokens`` tokens, and whether it was cut.

    Every tokenizer these models use spends at least one UTF-8 byte per token,
    so a text of at most ``max_tokens`` bytes cannot exceed the cap and is
    returned without loading a tokenizer at all — almost every line.
    """
    if len(raw.encode("utf-8")) <= max_tokens:
        return raw, False
    tokenizer = tok()
    ids = tokenizer.encode(raw)
    if len(ids) <= max_tokens:
        return raw, False
    return tokenizer.decode(ids[:max_tokens]).strip(), True


def reparse_line(line: dict[str, Any], plan: _Plan) -> tuple[dict[str, Any], bool]:
    """One predictions line re-derived from its server output; and whether it was capped."""
    if line.get("error") is not None or not isinstance(line.get("prediction"), str):
        return line, False
    raw = line.get("raw_prediction")
    if not isinstance(raw, str):
        raw = line["prediction"]
    text, capped = (raw, False) if plan.cap is None else cap_text(
        raw, plan.cap.max_tokens, plan.tok
    )
    parsed: ParsedOutput | None = None if plan.parse is None else plan.parse(text)
    if parsed is None and not capped and "raw_prediction" not in line:
        return line, False  # a plain model's line under the cap: nothing to say
    out: dict[str, Any] = {}
    for key, value in line.items():
        if key in ("raw_prediction", "segments", "capped_post_hoc"):
            continue
        out[key] = value
        if key == "prediction":
            out["prediction"] = parsed.text if parsed is not None else text
    # Same key order the runner writes: raw and segments after the base fields.
    out["raw_prediction"] = raw
    if parsed is not None:
        out["segments"] = [seg.as_json() for seg in parsed.segments]
    if capped:
        out["capped_post_hoc"] = True
    return out, capped


def _write_atomic(path: Path, lines: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _now() -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds").replace("+00:00", "Z"))


def reparse_run(
    run_dir: Path,
    *,
    dry_run: bool = False,
    cap: bool = True,
    tokenizer: Callable[[Cap], Tokenizer] = _HFTokenizer,
    recipes_file: Path = RECIPES_FILE,
) -> list[SubsetReport]:
    """Re-derive every predictions file in ``run_dir``; re-score what changed.

    Raises :class:`NoParserError` for a model with nothing to re-derive and
    ``FileNotFoundError`` for a dir that is not a runner output. ``tokenizer``
    builds the model's tokenizer from its :class:`Cap` (tests pass a fake); it
    is only called when a line is long enough to need it.
    """
    if not (run_dir / "summary.json").is_file():
        raise FileNotFoundError(f"no summary.json in {run_dir} — not a runner output dir")
    spec = spec_for_run(run_dir)
    the_cap = recipe_cap(spec.model_id, recipes_file) if cap else None
    plan = _Plan(
        parse=parser_for(spec.output_format),
        cap=the_cap,
        tokenizer=None if the_cap is None else (lambda: tokenizer(the_cap)),
    )
    if plan.parse is None and plan.cap is None:
        raise NoParserError(
            f"{spec.model_id} has no output format in raven_asr.config.OUTPUT_FORMATS "
            "and no decoding.max_tokens in serving/recipes.toml: nothing to re-derive"
        )
    reports: list[SubsetReport] = []
    changed: list[str] = []
    capped: dict[str, int] = {}
    # Every file is derived before any is written: a tokenizer that cannot be
    # loaded halfway through must not leave some subsets rewritten and unscored.
    pending: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in sorted(run_dir.glob("predictions_*.jsonl")):
        m = _PRED_RE.match(path.name)
        assert m
        subset = m.group("subset")
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
        results = [reparse_line(line, plan) for line in lines]
        new = [r[0] for r in results]
        n_capped = sum(1 for r in results if r[1])
        n_changed = sum(1 for a, b in zip(lines, new, strict=True) if a != b)
        reports.append(SubsetReport(subset, len(lines), n_changed, n_capped))
        if n_capped:
            capped[subset] = n_capped
        if n_changed and not dry_run:
            pending.append((path, new))
            changed.append(subset)
    if not changed:
        return reports
    for path, new in pending:
        _write_atomic(path, new)
    steps: list[dict[str, Any]] = []
    if plan.cap is not None and capped:
        steps.append({
            "step": "cap",
            "capped_post_hoc": True,
            "max_tokens": plan.cap.max_tokens,
            "tokenizer": {"repo": plan.cap.repo, "revision": plan.cap.revision,
                          "source": getattr(plan._tok, "source", None)},
            "n_capped": capped,
            "exact": False,
            "why_not_exact": (
                "cut on the re-tokenized decoded text, not on the generated token "
                "ids; audio vLLM splits is capped per piece by a server, once here"
            ),
            "tool": "raven_asr.reparse",
            "at": _now(),
        })
    if plan.parse is not None:
        steps.append({
            "step": "parse",
            "output_format": spec.output_format,
            "tool": "raven_asr.reparse",
            "subsets": changed,
            "at": _now(),
        })
    # Imported here: the runner pulls the inference stack (numpy, tqdm), which a
    # dry run has no use for.
    from .runner import rescore_recorded

    rescore_recorded(run_dir, spec, changed, steps)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raven-asr-reparse", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="runner output dir(s), each holding summary.json + predictions")
    parser.add_argument("--dry-run", action="store_true",
                        help="report how many lines would change, write nothing")
    parser.add_argument("--no-cap", action="store_true",
                        help="parse only; leave output length as recorded")
    args = parser.parse_args(argv)
    status = 0
    for run_dir in args.run_dirs:
        try:
            reports = reparse_run(run_dir, dry_run=args.dry_run, cap=not args.no_cap)
        except (NoParserError, FileNotFoundError, KeyError, OSError) as exc:
            print(f"SKIP {run_dir}: {exc}", file=sys.stderr)
            status = 2
            continue
        verb = "would change" if args.dry_run else "changed"
        for r in reports:
            print(f"{run_dir}: {r.subset}: {verb} {r.n_changed} of {r.n_lines} "
                  f"line(s), {r.n_capped} capped")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
