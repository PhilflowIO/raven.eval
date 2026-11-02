"""The serving recipe of the self-hosted models: load it, name it, hash it.

``recipes.toml`` beside this file holds one declarative recipe per self-hosted
model. A recipe plus the code that executes it (the launcher, the server or
shim that answers, the engine pins) is what decided how a model was served, so
the *identity* of a recipe is its id plus a SHA-256 over exactly those inputs
(:func:`recipe_sha256`). The servers report that identity in every response
(``x_raven.recipe``) and the harness records it next to each result, so an
artifact names the serving conditions it was measured under.

Stdlib only: the NeMo server, the shim and the launcher import this from their
own engine environments, and the harness tests import it without any of them.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

#: The ``serving/`` directory. Every path below is relative to it.
SERVING_DIR = Path(__file__).resolve().parent
RECIPES_FILE = SERVING_DIR / "recipes.toml"

#: Engines a recipe may name, and the files each one runs through. A change to
#: any of them changes what a request goes through, so each is part of the hash
#: of every recipe on that engine. ``transformers`` is the stand-in server for a
#: model vLLM cannot load (``vllm/transformers_server.py``, one file for every
#: such model; the recipe's ``model_class`` says which transformers class loads
#: it); it sits behind the same shim as the vLLM models.
ENGINE_FILES: dict[str, tuple[str, ...]] = {
    "vllm": ("recipe.py", "launch.py", "vllm/shim.py", "vllm/requirements.txt"),
    "transformers": (
        "recipe.py", "launch.py", "vllm/shim.py", "vllm/transformers_server.py",
        "vllm/requirements.txt",
    ),
    "nemo": (
        "recipe.py", "launch.py", "nemo/server.py", "nemo/batcher.py",
        "nemo/requirements.txt",
    ),
}

#: Keys every recipe must carry, and keys only some engines accept.
#: ``revision`` is the full HF commit the recipe serves. A repo id alone is a
#: moving target: a re-upload under the same id swaps the weights (and can break
#: loading) without anything in the recipe changing.
REQUIRED_KEYS = frozenset({
    "model", "revision", "engine", "weights_gb", "gpu_mem", "chunking", "decoding",
})
ENGINE_KEYS: dict[str, frozenset[str]] = {
    "vllm": frozenset({
        "dtype", "max_num_seqs", "max_model_len", "trust_remote_code", "args",
        "snapshot_ignore", "serve_from_snapshot", "lora_modules", "notes",
    }),
    "transformers": frozenset({"model_class", "max_num_seqs", "notes"}),
    "nemo": frozenset({
        "checkpoint_file", "max_batch", "batch_wait_ms", "batch_max_audio_s",
        "longform", "notes",
    }),
}
#: Engine keys a recipe on that engine must carry. NeMo's
#: ``ASRModel.from_pretrained`` takes no revision, so a NeMo recipe names the
#: .nemo file in the repo; the launcher downloads it at ``revision`` and the
#: server restores exactly that file.
#: A transformers recipe names the class that loads its model: the stand-in
#: serves only classes it knows how to decode (transformers_server.DECODERS).
ENGINE_REQUIRED: dict[str, frozenset[str]] = {
    "nemo": frozenset({"checkpoint_file"}),
    "transformers": frozenset({"model_class"}),
}
_REVISION = re.compile(r"[0-9a-f]{40}")

#: How every recipe decodes. Greedy only: with sampling a transcript is a draw,
#: and two runs of one recipe would not be the same measurement.
DECODING_MODES = frozenset({"greedy"})
#: What a recipe's ``decoding`` table carries per engine.
#:   vllm / transformers  max_tokens — the most tokens the server may generate
#:                        for one request. The shim sends it with every request
#:                        (vLLM: max_completion_tokens / max_tokens), together
#:                        with temperature 0 and repetition_penalty 1.0, so a
#:                        generation_config.json default can never apply.
#:   nemo                 the checkpoint's own decoding config, which the server
#:                        does not change; it refuses to start when the restored
#:                        checkpoint disagrees. Transducers (strategy
#:                        greedy_batch) emit at most max_symbols tokens per
#:                        encoder frame; Canary (strategy beam, beam_size 1 —
#:                        greedy) at most max_generation_delta tokens beyond the
#:                        encoder length. Those are their length bounds.
DECODING_KEYS: dict[str, frozenset[str]] = {
    "vllm": frozenset({"mode", "max_tokens"}),
    "transformers": frozenset({"mode", "max_tokens"}),
    "nemo": frozenset({"mode", "strategy", "max_symbols", "beam_size", "len_pen",
                       "max_generation_delta"}),
}
DECODING_REQUIRED: dict[str, frozenset[str]] = {
    "vllm": frozenset({"mode", "max_tokens"}),
    "transformers": frozenset({"mode", "max_tokens"}),
    "nemo": frozenset({"mode", "strategy"}),
}


def _decoding_problem(engine: str, decoding: object) -> str:
    if not isinstance(decoding, dict):
        return "decoding must be a table"
    missing = DECODING_REQUIRED[engine] - decoding.keys()
    if missing:
        return f"decoding needs {sorted(missing)} on {engine}"
    unknown = decoding.keys() - DECODING_KEYS[engine]
    if unknown:
        return f"decoding keys {sorted(unknown)} do not apply to {engine}"
    if decoding["mode"] not in DECODING_MODES:
        return f"decoding.mode must be one of {sorted(DECODING_MODES)}, got {decoding['mode']!r}"
    if "max_tokens" in decoding and not (
        isinstance(decoding["max_tokens"], int) and decoding["max_tokens"] > 0
    ):
        return "decoding.max_tokens must be a positive integer"
    return ""


class RecipeError(ValueError):
    """A recipe is missing, ambiguous or malformed."""


def load_recipes(path: Path = RECIPES_FILE) -> dict[str, dict[str, Any]]:
    """Every recipe, keyed by recipe id, validated against the engine's keys."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    recipes = data.get("recipes")
    if not isinstance(recipes, dict) or not recipes:
        raise RecipeError(f"{path}: no [recipes.<id>] tables")
    models: dict[str, str] = {}
    for rid, recipe in recipes.items():
        missing = REQUIRED_KEYS - recipe.keys()
        if missing:
            raise RecipeError(f"recipe {rid!r}: missing {sorted(missing)}")
        engine = recipe["engine"]
        if engine not in ENGINE_FILES:
            raise RecipeError(f"recipe {rid!r}: unknown engine {engine!r}")
        unknown = recipe.keys() - REQUIRED_KEYS - ENGINE_KEYS[engine]
        if unknown:
            raise RecipeError(f"recipe {rid!r}: keys {sorted(unknown)} do not apply to {engine}")
        missing = ENGINE_REQUIRED.get(engine, frozenset()) - recipe.keys()
        if missing:
            raise RecipeError(f"recipe {rid!r}: {engine} needs {sorted(missing)}")
        if problem := _decoding_problem(engine, recipe["decoding"]):
            raise RecipeError(f"recipe {rid!r}: {problem}")
        if not _REVISION.fullmatch(str(recipe["revision"])):
            raise RecipeError(
                f"recipe {rid!r}: revision must be the full 40-hex HF commit the "
                f"recipe serves, got {recipe.get('revision')!r}"
            )
        model = recipe["model"]
        if model in models:
            raise RecipeError(f"model {model!r} has two recipes: {models[model]!r} and {rid!r}")
        models[model] = rid
    return recipes


def resolve(name: str, recipes: dict[str, dict[str, Any]] | None = None) -> str:
    """The recipe id for ``name``, which is a recipe id or a served model id."""
    recipes = load_recipes() if recipes is None else recipes
    if name in recipes:
        return name
    for rid, recipe in recipes.items():
        if recipe["model"] == name:
            return rid
    raise RecipeError(f"no recipe for {name!r} in {RECIPES_FILE.name}")


def recipe_sha256(rid: str, recipes: dict[str, dict[str, Any]] | None = None) -> str:
    """SHA-256 over one recipe and every file its engine runs through.

    The recipe enters as canonical JSON, so reordering ``recipes.toml`` or
    editing another model's recipe leaves this one's hash alone; each file
    enters as its path and its bytes.
    """
    recipes = load_recipes() if recipes is None else recipes
    recipe = recipes[rid]
    h = hashlib.sha256()
    h.update(b"recipe\0" + rid.encode() + b"\0")
    h.update(json.dumps(recipe, sort_keys=True, ensure_ascii=False).encode() + b"\0")
    for rel in ENGINE_FILES[recipe["engine"]]:
        h.update(b"file\0" + rel.encode() + b"\0")
        h.update((SERVING_DIR / rel).read_bytes() + b"\0")
    return h.hexdigest()


def identity(name: str) -> dict[str, str] | None:
    """``{"id", "sha256"}`` for ``name`` (recipe id or model id), or None.

    None when no recipe matches or the recipe files are not readable — a server
    started outside a raven.eval checkout then says so instead of guessing.
    """
    try:
        recipes = load_recipes()
        rid = resolve(name, recipes)
        return {"id": rid, "sha256": recipe_sha256(rid, recipes)}
    except (OSError, RecipeError, tomllib.TOMLDecodeError):
        return None


def decoding(name: str) -> dict[str, Any] | None:
    """The ``decoding`` table of ``name``'s recipe, as servers report it, or None.

    Reported in every response as ``x_raven.decoding`` and recorded with the
    serving block, so a run names its decoding mode and output-length bound.
    """
    try:
        recipes = load_recipes()
        return dict(recipes[resolve(name, recipes)]["decoding"])
    except (OSError, RecipeError, tomllib.TOMLDecodeError):
        return None
