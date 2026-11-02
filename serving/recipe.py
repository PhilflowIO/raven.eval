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
import tomllib
from pathlib import Path
from typing import Any

#: The ``serving/`` directory. Every path below is relative to it.
SERVING_DIR = Path(__file__).resolve().parent
RECIPES_FILE = SERVING_DIR / "recipes.toml"

#: Engines a recipe may name, and the files each one runs through. A change to
#: any of them changes what a request goes through, so each is part of the hash
#: of every recipe on that engine. ``transformers`` is the stand-in server for a
#: model vLLM cannot load; it sits behind the same shim as the vLLM models.
ENGINE_FILES: dict[str, tuple[str, ...]] = {
    "vllm": ("recipe.py", "launch.py", "vllm/shim.py", "vllm/requirements.txt"),
    "transformers": (
        "recipe.py", "launch.py", "vllm/shim.py", "vllm/moonshine_shim.py",
        "vllm/requirements.txt",
    ),
    "nemo": (
        "recipe.py", "launch.py", "nemo/server.py", "nemo/batcher.py",
        "nemo/requirements.txt",
    ),
}

#: Keys every recipe must carry, and keys only some engines accept.
REQUIRED_KEYS = frozenset({"model", "engine", "weights_gb", "gpu_mem", "chunking"})
ENGINE_KEYS: dict[str, frozenset[str]] = {
    "vllm": frozenset({
        "dtype", "max_num_seqs", "max_model_len", "trust_remote_code", "args",
        "snapshot_ignore", "serve_from_snapshot", "lora_modules", "notes",
    }),
    "transformers": frozenset({"max_num_seqs", "notes"}),
    "nemo": frozenset({
        "checkpoint_file", "max_batch", "batch_wait_ms", "batch_max_audio_s",
        "longform", "notes",
    }),
}


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
