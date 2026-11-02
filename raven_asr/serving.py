"""Which serving recipe answered a self-hosted run, and holding it fixed.

A self-hosted model's transcripts depend on more than its weights: the engine
version, the launch flags and the server's batching all reach the output (one
measurement: server-side batching changed 2 of 862 FLEURS transcripts of
Parakeet-TDT-0.6B-v3 by one character). The servers in ``serving/`` therefore
report, in every response's ``x_raven``:

    engine    "vllm 0.28.0" / "nemo 3.0.0 (torch ...)" / "transformers ... (torch ...)"
    recipe    {"id", "sha256"}: the recipe in serving/recipes.toml and a hash
              over it and every file that executes it (serving/recipe.py)
    batching  NeMo {"max_batch", "wait_ms", "max_audio_s"}; vLLM {"max_num_seqs"}

These three are the run's *serving block*. The adapter pins the first block it
sees; every later response must carry the same one. A different block means the
endpoint changed under the run — a restart on another engine, another recipe
revision, another batch size — and the run aborts exactly like a served-model
mismatch (:class:`ServingChangedError`, ``reproduce`` exit 3).

Why abort rather than record and mark the row "not comparable": a WER is the
mean over one subset, and a subset transcribed half under one setup and half
under another describes neither. Throughput can be summed across differing
passes and flagged; a transcript cannot be un-mixed. The same rule holds across
invocations: a resumed subset whose earlier passes recorded another block is
refused at its first answer, before anything of it is recorded
(:func:`of_passes` primes the adapter), so the operator decides —
restore the old setup, or clear the subset's output and measure it again.

A server that reports none of the three (one started from an older copy of the
recipe) yields a ``null`` block and one warning; its results still stand, they
just cannot name their serving conditions.

Stdlib only — ``promote`` and ``scripts/verify.py`` import this.
"""

from __future__ import annotations

import json
from typing import Any

#: The ``x_raven`` keys that make up the serving block, in recorded order.
SERVING_KEYS = ("engine", "recipe", "batching")


class ServedModelMismatchError(RuntimeError):
    """The endpoint reports serving a different model than the run is labelled with.

    Not a property of one utterance but of the whole endpoint, so the runner lets
    it abort the run instead of recording it as a failed clip: every transcript
    after it would carry another model's output under this model's name.
    (Re-exported by :mod:`raven_asr.adapters.base`, where adapters import it.)
    """


class ServingChangedError(ServedModelMismatchError):
    """The endpoint's serving block differs from the one this run is pinned to.

    A subclass of :class:`ServedModelMismatchError` on purpose: it is the same
    kind of fact — about the endpoint, not about one clip — and it takes the
    same path out of the run (no transcript recorded, ``reproduce`` exit 3).
    """


def from_x_raven(x_raven: object) -> dict[str, Any] | None:
    """The serving block in a response's ``x_raven``, or None if it carries none."""
    if not isinstance(x_raven, dict) or not any(k in x_raven for k in SERVING_KEYS):
        return None
    return {k: x_raven.get(k) for k in SERVING_KEYS}


def describe(block: dict[str, Any] | None) -> str:
    """One line for an error message."""
    return "none reported" if block is None else json.dumps(block, sort_keys=True)


def of_passes(passes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The one serving block every recorded pass agrees on, or None.

    Passes written before serving was recorded carry no ``serving`` key and are
    not counted. Raises :class:`ServingChangedError` when the recorded passes
    disagree — such a subset has no single serving block to report.
    """
    recorded = [p["serving"] for p in passes if "serving" in p]
    distinct: list[dict[str, Any] | None] = []
    for block in recorded:
        if block not in distinct:
            distinct.append(block)
    if len(distinct) > 1:
        raise ServingChangedError(
            "the recorded passes were served by different setups: "
            + " vs ".join(describe(b) for b in distinct)
        )
    return distinct[0] if distinct else None


def serving_problem(row: dict[str, Any]) -> str:
    """Why a committed summary row's ``serving`` block contradicts its passes, or ``""``.

    The block is a recording, not something an artifact can re-derive; what is
    checkable is that it is well formed and equals what the row's throughput
    passes recorded. A row from before serving was recorded has no key at all
    and is not failed for it.
    """
    if "serving" not in row:
        return ""
    block = row["serving"]
    if block is not None:
        if not isinstance(block, dict) or set(block) != set(SERVING_KEYS):
            return f"serving: expected null or an object with keys {list(SERVING_KEYS)}"
        recipe = block.get("recipe")
        if recipe is not None and not (
            isinstance(recipe, dict)
            and isinstance(recipe.get("id"), str)
            and isinstance(recipe.get("sha256"), str)
            and len(recipe["sha256"]) == 64
        ):
            return "serving.recipe: expected null or {id, sha256 (64 hex)}"
    tp = row.get("throughput")
    passes = tp.get("passes") if isinstance(tp, dict) else None
    if not isinstance(passes, list):
        return ""
    try:
        recorded = of_passes(passes)
    except ServingChangedError as exc:
        return f"serving: {exc}"
    if any("serving" in p for p in passes) and recorded != block:
        return f"serving: {describe(block)} vs passes {describe(recorded)}"
    return ""
