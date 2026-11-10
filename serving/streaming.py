"""Streaming recipes (``streaming.toml``): load, validate, hash, launch.

A stream recipe names an offline recipe (``base``) for the weights, precision
and engine pins, and adds the model's documented streaming mode. Its identity
``{"id", "sha256"}`` hashes the stream table, the base recipe's own hash and
every file the live path runs through, so a capacity curve names exactly what
served it — as ``recipe.recipe_sha256`` does for an offline run.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

from recipe import RecipeError, load_recipes, recipe_sha256

SERVING_DIR = Path(__file__).resolve().parent
STREAMS_FILE = SERVING_DIR / "streaming.toml"

PROTOCOLS = frozenset({"vllm-realtime", "nemo-pipeline", "moonshine-voice"})
PIPELINES = frozenset({"cache_aware", "buffered"})

#: Files each protocol runs through, besides the base recipe's (whose hash
#: already covers launch.py, recipe.py and the engine requirements).
PROTOCOL_FILES: dict[str, tuple[str, ...]] = {
    "vllm-realtime": ("streaming.py",),
    "nemo-pipeline": (
        "streaming.py", "nemo/stream_server.py", "nemo/stream_frames.py",
        "nemo/conf/cache_aware_rnnt.yaml", "nemo/conf/buffered_rnnt.yaml",
    ),
    "moonshine-voice": (
        "streaming.py", "moonshine/stream_server.py", "moonshine/requirements.txt",
    ),
}

COMMON_KEYS = frozenset({"base", "protocol", "chunk_ms", "latency", "notes"})
PROTOCOL_KEYS: dict[str, frozenset[str]] = {
    "vllm-realtime": frozenset({"gpu_mem", "max_num_seqs"}),
    "nemo-pipeline": frozenset({
        "gpu_mem", "pipeline", "lang", "batch_size", "num_slots", "matmul_precision",
        "att_context_size", "left_context_s", "chunk_s", "right_context_s",
    }),
    # The vendor runtime: its own package, model files and precision, on the
    # device it ships for. Nothing of the base recipe's engine is used; the base
    # names the checkpoint the vendor build was exported from.
    # How the sessions share it (moonshine/stream_server.py): one Transcriber
    # or one per live session, and ONNX Runtime's thread pools as the vendor
    # leaves them or single-threaded sessions.
    "moonshine-voice": frozenset({
        "runtime", "device", "model_url", "model_arch", "update_interval_s",
        "precision", "files", "transcribers", "ort_threads",
    }),
}
PROTOCOL_REQUIRED: dict[str, frozenset[str]] = {
    "vllm-realtime": frozenset({"gpu_mem", "max_num_seqs"}),
    "nemo-pipeline": frozenset({"gpu_mem"}),
    "moonshine-voice": PROTOCOL_KEYS["moonshine-voice"],
}
PIPELINE_REQUIRED: dict[str, frozenset[str]] = {
    "cache_aware": frozenset({"att_context_size", "num_slots"}),
    "buffered": frozenset({"left_context_s", "chunk_s", "right_context_s"}),
}
#: Base engine each protocol serves.
PROTOCOL_ENGINE = {"vllm-realtime": "vllm", "nemo-pipeline": "nemo",
                   "moonshine-voice": "transformers"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
MATMUL = frozenset({"highest", "high", "medium"})
MOONSHINE_TRANSCRIBERS = frozenset({"shared", "per_stream"})
MOONSHINE_ORT_THREADS = frozenset({"vendor", "single"})


def _problem(sid: str, s: dict[str, Any], recipes: dict[str, dict[str, Any]]) -> str:
    missing = COMMON_KEYS - s.keys()
    if missing:
        return f"missing {sorted(missing)}"
    proto = s["protocol"]
    if proto not in PROTOCOLS:
        return f"unknown protocol {proto!r}"
    unknown = s.keys() - COMMON_KEYS - PROTOCOL_KEYS[proto]
    if unknown:
        return f"keys {sorted(unknown)} do not apply to {proto}"
    base = recipes.get(s["base"])
    if base is None:
        return f"base {s['base']!r} is no recipe in recipes.toml"
    if base["engine"] != PROTOCOL_ENGINE[proto]:
        return f"{proto} serves {PROTOCOL_ENGINE[proto]} recipes, base is {base['engine']}"
    if not (isinstance(s["chunk_ms"], int) and s["chunk_ms"] > 0 and s["chunk_ms"] % 10 == 0):
        return "chunk_ms must be a positive multiple of 10"
    if missing := sorted(PROTOCOL_REQUIRED[proto] - s.keys()):
        return f"{proto} needs {missing}"
    if proto == "moonshine-voice":
        return _moonshine_problem(s)
    if not (0 < float(s["gpu_mem"]) <= 0.95):
        return "gpu_mem must be in (0, 0.95]"
    if proto == "vllm-realtime":
        if not (isinstance(s.get("max_num_seqs"), int) and s["max_num_seqs"] > 0):
            return "vllm-realtime needs max_num_seqs"
        return ""
    pipe = s.get("pipeline")
    if pipe not in PIPELINES:
        return f"pipeline must be one of {sorted(PIPELINES)}"
    for key in ("lang", "batch_size", "matmul_precision", *PIPELINE_REQUIRED[pipe]):
        if key not in s:
            return f"{pipe} needs {key}"
    other = set().union(*(PIPELINE_REQUIRED[p] for p in PIPELINES if p != pipe))
    if stray := sorted(other & s.keys()):
        return f"keys {stray} do not apply to the {pipe} pipeline"
    if s["matmul_precision"] not in MATMUL:
        return f"matmul_precision must be one of {sorted(MATMUL)}"
    if pipe == "cache_aware":
        acs = s["att_context_size"]
        if not (isinstance(acs, list) and len(acs) == 2 and all(isinstance(v, int) for v in acs)):
            return "att_context_size must be [left, right] in encoder frames"
        if s["num_slots"] < s["batch_size"]:
            return "num_slots must be >= batch_size"
    if base["precision"]["weights"] != "fp32":
        return "nemo-pipeline serves the base recipe's fp32 only"
    return ""


def _moonshine_problem(s: dict[str, Any]) -> str:
    """The vendor runtime is pinned exactly: package, files by sha256, precision, device."""
    if not re.fullmatch(r"moonshine-voice==\d+(\.\d+)+", str(s["runtime"])):
        return "runtime must pin moonshine-voice==<version>"
    if s["device"] not in ("cpu", "cuda"):
        return "device must be cpu or cuda"
    files = s["files"]
    if not (isinstance(files, dict) and files
            and all(isinstance(v, str) and _SHA256.fullmatch(v) for v in files.values())):
        return "files must map every model file to its sha256"
    prec = s["precision"]
    if not isinstance(prec, dict) or {"weights", "activations", "scheme", "checkpoint"} - prec.keys():
        return "precision needs weights, activations, scheme and checkpoint"
    if prec["checkpoint"].get("source") != "vendor" or prec["checkpoint"].get("url") != s["model_url"]:
        return "precision.checkpoint must name the vendor build at model_url"
    if s["transcribers"] not in MOONSHINE_TRANSCRIBERS:
        return f"transcribers must be one of {sorted(MOONSHINE_TRANSCRIBERS)}"
    if s["ort_threads"] not in MOONSHINE_ORT_THREADS:
        return f"ort_threads must be one of {sorted(MOONSHINE_ORT_THREADS)}"
    return ""


def load_streams(
    path: Path = STREAMS_FILE, recipes: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """``(streams, skipped)``: every stream recipe, validated, and every documented skip."""
    recipes = load_recipes() if recipes is None else recipes
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    streams = data.get("streams") or {}
    skipped = data.get("skipped") or {}
    for sid, s in streams.items():
        if problem := _problem(sid, s, recipes):
            raise RecipeError(f"stream recipe {sid!r}: {problem}")
    for sid, s in skipped.items():
        if not isinstance(s.get("reason"), str) or not s["reason"].strip():
            raise RecipeError(f"skipped {sid!r}: a skip needs its reason")
        if sid in streams:
            raise RecipeError(f"{sid!r} is both a stream recipe and skipped")
    return streams, skipped


def stream_sha256(
    sid: str,
    streams: dict[str, dict[str, Any]] | None = None,
    recipes: dict[str, dict[str, Any]] | None = None,
) -> str:
    """SHA-256 over the stream table, its base recipe's hash and the live-path files."""
    recipes = load_recipes() if recipes is None else recipes
    streams = load_streams(recipes=recipes)[0] if streams is None else streams
    s = streams[sid]
    h = hashlib.sha256()
    h.update(b"stream\0" + sid.encode() + b"\0")
    h.update(json.dumps(s, sort_keys=True, ensure_ascii=False).encode() + b"\0")
    h.update(b"base\0" + recipe_sha256(s["base"], recipes).encode() + b"\0")
    for rel in PROTOCOL_FILES[s["protocol"]]:
        h.update(b"file\0" + rel.encode() + b"\0")
        h.update((SERVING_DIR / rel).read_bytes() + b"\0")
    return h.hexdigest()


def stream_identity(sid: str) -> dict[str, str]:
    """``{"id", "sha256"}`` of one stream recipe."""
    return {"id": sid, "sha256": stream_sha256(sid)}


def resolve_stream(name: str, streams: dict[str, dict[str, Any]],
                   recipes: dict[str, dict[str, Any]]) -> str:
    """The stream id for ``name``: a stream id, its base recipe id or the served model id."""
    if name in streams:
        return name
    for sid, s in streams.items():
        if s["base"] == name or recipes[s["base"]]["model"] == name:
            return sid
    raise RecipeError(f"no stream recipe for {name!r} in {STREAMS_FILE.name}")


def served_recipe(sid: str, streams: dict[str, dict[str, Any]],
                  recipes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The base recipe with the stream recipe's engine overrides applied.

    vllm-realtime runs the base recipe's own ``vllm serve`` command; only the
    memory budget and the session cap change. The NeMo pipeline takes the base
    recipe's checkpoint and precision and nothing of its offline batching.
    """
    s = streams[sid]
    out = dict(recipes[s["base"]])
    if s["protocol"] == "moonshine-voice":
        # Nothing of the base engine runs: the vendor build has its own files,
        # precision and device.
        out["precision"] = s["precision"]
        out.pop("checkpoint_file", None)
        return out
    out["gpu_mem"] = s["gpu_mem"]
    if s["protocol"] == "vllm-realtime":
        out["max_num_seqs"] = s["max_num_seqs"]
    return out
