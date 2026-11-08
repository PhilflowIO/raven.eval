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
    "precision",
})
ENGINE_KEYS: dict[str, frozenset[str]] = {
    "vllm": frozenset({
        "max_num_seqs", "max_model_len", "trust_remote_code", "args",
        "snapshot_ignore", "serve_from_snapshot", "lora_modules", "vllm_plugins", "notes",
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
#:                        stop_at_audio_end (vllm only, optional) — the model
#:                        decodes one position per audio frame and has no audio
#:                        past the last one (Voxtral Realtime): the shim lowers
#:                        max_tokens per request to the frames left after the
#:                        prompt, so generation ends where the audio ends
#:                        (shim._frames_after_prompt, raven.eval#89).
#:   nemo                 the checkpoint's own decoding config, which the server
#:                        does not change; it refuses to start when the restored
#:                        checkpoint disagrees. Transducers (strategy
#:                        greedy_batch) emit at most max_symbols tokens per
#:                        encoder frame; Canary (strategy beam, beam_size 1 —
#:                        greedy) at most max_generation_delta tokens beyond the
#:                        encoder length. Those are their length bounds.
DECODING_KEYS: dict[str, frozenset[str]] = {
    "vllm": frozenset({"mode", "max_tokens", "stop_at_audio_end"}),
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
    if "stop_at_audio_end" in decoding and decoding["stop_at_audio_end"] is not True:
        return "decoding.stop_at_audio_end is either true or absent"
    return ""


#: Precision: what number format the served weights and the computation are in,
#: how they got there, and over how many GPUs one model instance is split. Two
#: runs of one model at different precision measure different things, so every
#: recipe declares it — nothing is left to an engine default — and every
#: response reports it (``x_raven.precision``).
#:
#:   weights          the format the weights are held in while serving
#:   activations      the format the computation runs in (vLLM: --dtype)
#:   scheme           "none" for plain floating point (weights == activations);
#:                    otherwise the quantization scheme, e.g. "fp8-w8a16-marlin"
#:   checkpoint       where the served weights come from:
#:                      {source = "published", dtype}  the repo at ``revision``
#:                        as its publisher ships it; dtype = what it stores, so a
#:                        cast (dtype != weights) is on record
#:                      {source = "vendor", repo, revision, dtype}  a quantized
#:                        checkpoint the model's own vendor publishes
#:                      {source = "third-party", repo, revision, dtype}  a quantized
#:                        checkpoint someone other than the vendor publishes
#:                        (e.g. RedHatAI's llm-compressor builds); repo names who
#:                      {source = "self", recipe, dtype[, repo, revision]}  one we
#:                        made; recipe names the script or recipe that produced it
#:   tensor_parallel  GPUs one model instance is split over; the launcher
#:                    requires exactly that many in CUDA_VISIBLE_DEVICES
#:   control          optional: the id of the quantized recipe this one is the
#:                    unquantized control of. A control serves the same model in
#:                    plain floating point (scheme "none"), measured on FLEURS-de
#:                    only, so the published table can put the paired difference
#:                    quantized - control directly under the quantized row.
PRECISION_KEYS = frozenset({"weights", "activations", "scheme", "checkpoint", "tensor_parallel"})
PRECISION_OPTIONAL = frozenset({"control"})
#: Plain floating-point formats, with the torch / vLLM name each loads as.
FLOAT_FORMATS: dict[str, str] = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
WEIGHT_FORMATS = frozenset({*FLOAT_FORMATS, "fp8-e4m3", "fp8-e5m2", "int8", "int4", "nf4", "mxfp4"})
ACTIVATION_FORMATS = frozenset({*FLOAT_FORMATS, "fp8-e4m3", "int8"})
CHECKPOINT_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # source: (required, optional)
    "published": (frozenset({"source", "dtype"}), frozenset()),
    "vendor": (frozenset({"source", "repo", "revision", "dtype"}), frozenset()),
    "third-party": (frozenset({"source", "repo", "revision", "dtype"}), frozenset()),
    "self": (frozenset({"source", "recipe", "dtype"}), frozenset({"repo", "revision"})),
}
_SCHEME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
#: vLLM flags the precision table owns; a recipe's ``args`` may not set them.
PRECISION_FLAGS = ("--dtype", "--tensor-parallel-size", "-tp")
#: The kernel a quantized vLLM scheme names, and the environment that makes vLLM
#: 0.33.0 pick it. vLLM ranks its kernels per layer type and takes the first one
#: the GPU supports; the scheme is only true if that first one is the named one.
#:   fp8-w8a16-marlin  FP8 weights, 16-bit activations. Below compute capability
#:                     8.9 compressed-tensors' W8A8-FP8 falls back to weight-only
#:                     CompressedTensorsW8A16Fp8 (compressed_tensors.py:896-912),
#:                     whose CUDA kernels are [Humming, Marlin] in that order
#:                     (kernels/linear/__init__.py:522-526). vLLM installs
#:                     humming-kernels itself (requirements/cuda.txt:33), so Marlin
#:                     runs only with Humming disabled (__init__.py:653).
#: The launcher sets this environment; a quantized vLLM scheme without an entry
#: here is refused, so no quantized row is left to vLLM's kernel ranking.
VLLM_SCHEME_ENV: dict[str, dict[str, str]] = {
    "fp8-w8a16-marlin": {"VLLM_DISABLED_KERNELS": "HummingFP8ScaledMMLinearKernel"},
}


def _checkpoint_problem(ckpt: object, scheme: str, model: str, revision: str) -> str:
    if not isinstance(ckpt, dict):
        return "precision.checkpoint must be a table"
    source = ckpt.get("source")
    if source not in CHECKPOINT_KEYS:
        return f"precision.checkpoint.source must be one of {sorted(CHECKPOINT_KEYS)}, got {source!r}"
    required, optional = CHECKPOINT_KEYS[source]
    if missing := required - ckpt.keys():
        return f"precision.checkpoint ({source}) needs {sorted(missing)}"
    if unknown := ckpt.keys() - required - optional:
        return f"precision.checkpoint keys {sorted(unknown)} do not apply to source {source!r}"
    if ckpt["dtype"] not in WEIGHT_FORMATS:
        return f"precision.checkpoint.dtype must be one of {sorted(WEIGHT_FORMATS)}, got {ckpt['dtype']!r}"
    if "revision" in ckpt and not _REVISION.fullmatch(str(ckpt["revision"])):
        return "precision.checkpoint.revision must be a full 40-hex commit"
    if source in ("vendor", "third-party") and ckpt["repo"] == model and ckpt["revision"] != revision:
        return "precision.checkpoint names the served repo at another revision than the recipe's"
    if source == "self" and not (isinstance(ckpt["recipe"], str) and ckpt["recipe"].strip()):
        return "precision.checkpoint.recipe must name what produced the checkpoint"
    if scheme != "none" and source == "published":
        return ("a quantized scheme needs a vendor, third-party or self-made checkpoint "
                "(a published repo that ships quantized weights is source = \"vendor\" "
                "or \"third-party\")")
    return ""


def _precision_problem(engine: str, prec: object, recipe: dict[str, Any]) -> str:
    if not isinstance(prec, dict):
        return "precision must be a table"
    if missing := PRECISION_KEYS - prec.keys():
        return f"precision needs {sorted(missing)}"
    if unknown := prec.keys() - PRECISION_KEYS - PRECISION_OPTIONAL:
        return f"precision keys {sorted(unknown)} are not precision keys"
    weights, acts, scheme = prec["weights"], prec["activations"], prec["scheme"]
    if weights not in WEIGHT_FORMATS:
        return f"precision.weights must be one of {sorted(WEIGHT_FORMATS)}, got {weights!r}"
    if acts not in ACTIVATION_FORMATS:
        return f"precision.activations must be one of {sorted(ACTIVATION_FORMATS)}, got {acts!r}"
    if not (isinstance(scheme, str) and _SCHEME.fullmatch(scheme)):
        return f"precision.scheme must be a lower-case slug, got {scheme!r}"
    if scheme == "none":
        if weights not in FLOAT_FORMATS or acts != weights:
            return ('precision.scheme "none" means plain floating point: weights '
                    "must be fp32/bf16/fp16 and activations the same")
    elif weights in FLOAT_FORMATS and acts == weights:
        return f'precision.scheme {scheme!r}: weights and activations are plain {weights}, scheme must be "none"'
    tp = prec["tensor_parallel"]
    if not (isinstance(tp, int) and not isinstance(tp, bool) and tp >= 1):
        return "precision.tensor_parallel must be a positive integer"
    if engine != "vllm":
        # The NeMo server and the transformers stand-in load one checkpoint
        # unquantized onto one card; neither has a way to do anything else.
        if scheme != "none":
            return f"precision.scheme must be \"none\" on {engine}"
        if tp != 1:
            return f"precision.tensor_parallel must be 1 on {engine}"
    elif acts not in FLOAT_FORMATS:
        return "precision.activations must be fp32/bf16/fp16 on vllm (it is vLLM's --dtype)"
    if "control" in prec:
        control = prec["control"]
        if not (isinstance(control, str) and control.strip()):
            return "precision.control must name the quantized recipe this one controls"
        if scheme != "none":
            return "a control serves plain floating point: precision.scheme must be \"none\""
    if engine == "vllm":
        args = [str(a) for a in recipe.get("args", [])]
        if clash := [a for a in args if a.split("=", 1)[0] in PRECISION_FLAGS]:
            return f"args {clash} set what precision owns"
    if problem := _checkpoint_problem(prec["checkpoint"], scheme, recipe["model"], str(recipe["revision"])):
        return problem
    if engine == "vllm" and scheme != "none" and scheme not in VLLM_SCHEME_ENV:
        return (f"precision.scheme {scheme!r} names no kernel the launcher can pin "
                f"on vllm; known: {sorted(VLLM_SCHEME_ENV)}")
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
        if problem := _precision_problem(engine, recipe["precision"], recipe):
            raise RecipeError(f"recipe {rid!r}: {problem}")
        if problem := _plugins_problem(recipe.get("vllm_plugins", [])):
            raise RecipeError(f"recipe {rid!r}: {problem}")
        model = recipe["model"]
        if model in models:
            raise RecipeError(f"model {model!r} has two recipes: {models[model]!r} and {rid!r}")
        models[model] = rid
    for rid, recipe in recipes.items():
        if problem := _control_problem(recipe["precision"], recipes):
            raise RecipeError(f"recipe {rid!r}: {problem}")
    return recipes


#: vLLM plugins (entry points of the ``vllm.*_plugins`` groups) installed in the
#: vLLM environment that a recipe may load, with the package that ships each.
#: vLLM loads every installed plugin unless VLLM_PLUGINS names the ones to load;
#: the launcher always sets it to exactly the recipe's ``vllm_plugins``, so a
#: plugin installed for one model never runs in another model's engine.
#:   nemo_speechlm  NeMo Speech's SpeechLM models (nemo_toolkit, pinned in
#:                  vllm/requirements.txt); the vendor's own serving command for
#:                  Nemotron-3.5-Transcribe sets VLLM_PLUGINS=nemo_speechlm too.
VLLM_PLUGINS: dict[str, str] = {"nemo_speechlm": "nemo_toolkit"}


def _plugins_problem(plugins: object) -> str:
    """``vllm_plugins`` is a list of known plugin names, each at most once."""
    if not isinstance(plugins, list) or not all(isinstance(p, str) for p in plugins):
        return "vllm_plugins must be a list of plugin names"
    if unknown := sorted(set(plugins) - VLLM_PLUGINS.keys()):
        return f"vllm_plugins {unknown} are not installed plugins; known: {sorted(VLLM_PLUGINS)}"
    if len(set(plugins)) != len(plugins):
        return "vllm_plugins names a plugin twice"
    return ""


def _control_problem(prec: dict[str, Any], recipes: dict[str, dict[str, Any]]) -> str:
    """A control must name a quantized recipe of this file that is not a control itself."""
    if "control" not in prec:
        return ""
    target = recipes.get(prec["control"])
    if target is None:
        return f"precision.control names no recipe: {prec['control']!r}"
    if target["precision"]["scheme"] == "none":
        return f"precision.control names {prec['control']!r}, which is not quantized"
    if "control" in target["precision"]:
        return f"precision.control names {prec['control']!r}, which is itself a control"
    return ""


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


def revision_of(name: str) -> str | None:
    """The pinned model revision of ``name``'s recipe, or None."""
    try:
        recipes = load_recipes()
        return str(recipes[resolve(name, recipes)]["revision"])
    except (OSError, RecipeError, tomllib.TOMLDecodeError, KeyError):
        return None


def precision(name: str) -> dict[str, Any] | None:
    """The ``precision`` table of ``name``'s recipe, as servers report it, or None.

    Reported in every response as ``x_raven.precision`` and recorded with the
    serving block, so a result names the number format it was measured in.
    """
    try:
        recipes = load_recipes()
        prec = recipes[resolve(name, recipes)]["precision"]
    except (OSError, RecipeError, tomllib.TOMLDecodeError):
        return None
    return {**prec, "checkpoint": dict(prec["checkpoint"])}


def weights_mismatch(numel_by_dtype: dict[str, int], weights: str) -> str:
    """How a loaded model's parameters contradict the declared ``weights``, or ``""``.

    ``numel_by_dtype`` counts the floating-point parameter elements of the loaded
    model per torch dtype name (``"float32"``, ``"bfloat16"``, …). The format
    holding most of them must be the declared one: a model may keep a few small
    modules in fp32 by design (transformers' ``_keep_in_fp32_modules``), but the
    bulk of the weights is what the precision names. A server refuses to serve
    on a mismatch, so what it reports is what it computes with.
    """
    expected = FLOAT_FORMATS.get(weights)
    if expected is None:
        return f"weights {weights!r} is not a plain float format this check knows"
    if not numel_by_dtype:
        return "the loaded model has no floating-point parameters to check"
    bulk = max(numel_by_dtype, key=lambda k: numel_by_dtype[k])
    if bulk != expected:
        return (f"loaded weights are mostly {bulk} ({numel_by_dtype}), "
                f"the recipe declares {weights} ({expected})")
    return ""
