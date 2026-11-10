"""Start the server for one self-hosted model from its recipe (``recipes.toml``).

    python serving/launch.py <recipe-id | model-id>            start it (exec)
    python serving/launch.py <recipe-id | model-id> --print    the command, as JSON
    python serving/launch.py <recipe-id | model-id> --engine   vllm | nemo | transformers
    python serving/launch.py <recipe-id | model-id> --identity {"id", "sha256"}
    python serving/launch.py --list                            every recipe id and model
    python serving/launch.py <stream-id | model-id> --streaming [--print | --identity]
                                                               the model's LIVE server
                                                               (streaming.toml)

Run it with the Python of the engine's environment: the NeMo venv for ``nemo``
recipes, the vLLM venv for ``vllm`` and ``transformers`` recipes (the ``vllm``
executable is taken from beside that interpreter). The process is replaced by
the server, so a supervisor's signals reach the server directly.

What this starts is the *engine*. For ``vllm`` and ``transformers`` recipes the
harness talks to ``vllm/shim.py`` in front of it (same environment, its own
port); the NeMo server speaks the harness's protocol itself.

Environment (everything a host decides, nothing a model decides):

    CUDA_VISIBLE_DEVICES      exactly precision.tensor_parallel GPU indices (one,
                              unless the recipe splits the model) — required. The
                              process sees those cards and no other.
    RAVEN_SERVE_HOST          bind address (default 127.0.0.1)
    RAVEN_SERVE_PORT          port (default 8000)
    RAVEN_SERVE_GPU_MEM       VRAM budget fraction (default: the recipe's gpu_mem)
    RAVEN_SERVE_MAX_NUM_SEQS  vLLM --max-num-seqs (default: the recipe's). Give the
                              shim the same value: it reports it in x_raven.batching.
    RAVEN_SERVE_NEMO_MAX_BATCH, RAVEN_SERVE_NEMO_BATCH_WAIT_MS
                              the NeMo server's micro-batch size and wait (default:
                              the recipe's max_batch / batch_wait_ms); the server
                              reports both in x_raven.batching. Named load-test
                              settings of these knobs: serving/load_variants.toml.
    RAVEN_SERVE_MIN_FREE_GB   if set, refuse to start a download unless the HF
                              cache keeps this much free beyond the weights
    HF_HOME, HF_TOKEN         as usual for huggingface_hub

Set by the recipe, never by the host: VLLM_PLUGINS (the recipe's ``vllm_plugins``,
empty for all others) and, for a quantized scheme, the kernel environment and
flags (``recipe.VLLM_SCHEMES``).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streaming
from recipe import (
    FLOAT_FORMATS,
    SERVING_DIR,
    VLLM_SCHEMES,
    identity,
    load_recipes,
    resolve,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
#: vLLM's transcription endpoint rejects uploads above 25 MB by default; a
#: 10-minute 16 kHz PCM16 clip is ~19 MB. Aligned with the shim's upload cap.
VLLM_MAX_AUDIO_MB = "512"
#: vLLM refuses audio longer than VLLM_MAX_AUDIO_DECODE_DURATION_S (600 s by
#: default) before its own long-audio splitting runs, so a production-shaped
#: whole-file window cut near 600 s (up to 720 s) was answered 400. 7200 s is the
#: longest track Raven's production STT service accepts (MAX_AUDIO_DURATION).
VLLM_MAX_AUDIO_S = "7200"
#: …and refuses an upload whose decoded float32 PCM would exceed
#: VLLM_MAX_AUDIO_DECODE_BYTES (256 MiB by default, ~70 min at 16 kHz mono), so
#: the duration limit above alone still turned a 97-minute track away. Sized to
#: the same 7200 s at the rate every self-hosted endpoint is sent (16 kHz mono,
#: raven_asr.adapters.vllm_openai.INPUT_SAMPLE_RATE), 4 bytes a sample.
VLLM_MAX_AUDIO_BYTES = str(int(VLLM_MAX_AUDIO_S) * 16_000 * 4)
#: Placeholder for the snapshot path in ``--print`` (which never downloads).
SNAPSHOT_PLACEHOLDER = "<snapshot>"


class LaunchError(RuntimeError):
    """The host is not in a state this recipe may be started in."""


def _knob(env: dict[str, str], name: str, default: Any) -> str:
    value = env.get(name, "").strip()
    return value or str(default)


def build(
    recipe: dict[str, Any],
    rid: str,
    env: dict[str, str],
    *,
    python: str,
    snapshot: str | None = None,
    checkpoint: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """``(argv, extra_env)`` for one recipe; pure, so it is testable without an engine.

    ``snapshot`` is the local snapshot path for recipes that serve from one or
    mount LoRA modules out of one; ``checkpoint`` the local .nemo path (every NeMo
    recipe names its ``checkpoint_file``). Both are resolved by :func:`_fetch`
    before a real launch.
    """
    host = _knob(env, "RAVEN_SERVE_HOST", DEFAULT_HOST)
    port = _knob(env, "RAVEN_SERVE_PORT", DEFAULT_PORT)
    gpu_mem = _knob(env, "RAVEN_SERVE_GPU_MEM", recipe["gpu_mem"])
    engine = recipe["engine"]
    extra = {"RAVEN_SERVING_RECIPE": rid}

    if engine == "nemo":
        extra.update({
            "BENCH_MODEL": recipe["model"],
            # The .nemo downloaded at the recipe's revision (_fetch); the server
            # restores exactly this file and never pulls from the hub itself.
            "BENCH_MODEL_PATH": checkpoint or "",
            "BENCH_MODEL_REVISION": recipe["revision"],
            "BENCH_GPU_MEM_FRACTION": gpu_mem,
            "BENCH_LONGFORM": "true" if recipe.get("longform", True) else "false",
            "NEMO_MAX_BATCH": _knob(env, "RAVEN_SERVE_NEMO_MAX_BATCH",
                                    recipe.get("max_batch", 16)),
            "NEMO_BATCH_WAIT_MS": _knob(env, "RAVEN_SERVE_NEMO_BATCH_WAIT_MS",
                                        recipe.get("batch_wait_ms", 20)),
            "NEMO_BATCH_MAX_AUDIO_S": str(recipe.get("batch_max_audio_s", 40)),
        })
        argv = [python, "-m", "uvicorn", "server:app",
                "--app-dir", str(SERVING_DIR / "nemo"), "--host", host, "--port", port]
        return argv, extra

    if engine == "transformers":
        extra.update({
            "TRANSFORMERS_MODEL": recipe["model"],
            "TRANSFORMERS_REVISION": recipe["revision"],
            "TRANSFORMERS_CLASS": recipe["model_class"],
            "TRANSFORMERS_HOST": host,
            "TRANSFORMERS_PORT": port,
            "TRANSFORMERS_GPU_MEM_FRACTION": gpu_mem,
            # The server loads in exactly this dtype and refuses to serve when
            # the loaded parameters are mostly another one.
            "TRANSFORMERS_DTYPE": FLOAT_FORMATS[recipe["precision"]["weights"]],
        })
        return [python, str(SERVING_DIR / "vllm" / "transformers_server.py")], extra

    # vllm
    vllm = str(Path(python).parent / "vllm")
    seqs = _knob(env, "RAVEN_SERVE_MAX_NUM_SEQS", recipe.get("max_num_seqs", 16))
    snap = snapshot or SNAPSHOT_PLACEHOLDER
    if recipe.get("serve_from_snapshot"):
        # The snapshot was downloaded at the recipe's revision (_fetch).
        target = [snap, "--served-model-name", recipe["model"]]
    else:
        # --revision also pins tokenizer and processor (vLLM defaults
        # --tokenizer-revision to it).
        target = [recipe["model"], "--revision", recipe["revision"]]
    argv = [vllm, "serve", *target, "--host", host, "--port", port,
            "--gpu-memory-utilization", gpu_mem, "--max-num-seqs", seqs]
    # The computation's format; for an unquantized recipe also the weights'.
    # A quantized checkpoint's own format comes from its config (vLLM reads the
    # quantization_config); --dtype is then the activation format around it.
    precision = recipe["precision"]
    argv += ["--dtype", FLOAT_FORMATS[precision["activations"]]]
    if precision["tensor_parallel"] > 1:
        argv += ["--tensor-parallel-size", str(precision["tensor_parallel"])]
    # A quantized scheme names its kernels; its environment and flags are what
    # make vLLM pick them (recipe.VLLM_SCHEMES). The environment overrides the
    # host's value on purpose; a recipe's args may not repeat the flags.
    if (scheme := VLLM_SCHEMES.get(precision["scheme"])) is not None:
        extra.update(scheme.env)
        argv += list(scheme.args)
    if "max_model_len" in recipe:
        argv += ["--max-model-len", str(recipe["max_model_len"])]
    if recipe.get("trust_remote_code"):
        argv.append("--trust-remote-code")
    argv += list(recipe.get("args", []))
    if recipe.get("lora_modules"):
        argv.append("--lora-modules")
        argv += [f"{name}={snap}/{sub}" for name, sub in recipe["lora_modules"].items()]
    extra["VLLM_MAX_AUDIO_CLIP_FILESIZE_MB"] = env.get(
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", VLLM_MAX_AUDIO_MB
    )
    extra["VLLM_MAX_AUDIO_DECODE_DURATION_S"] = env.get(
        "VLLM_MAX_AUDIO_DECODE_DURATION_S", VLLM_MAX_AUDIO_S
    )
    extra["VLLM_MAX_AUDIO_DECODE_BYTES"] = env.get(
        "VLLM_MAX_AUDIO_DECODE_BYTES", VLLM_MAX_AUDIO_BYTES
    )
    # Exactly the plugins the recipe names (recipe.VLLM_PLUGINS); "" loads none.
    # Unset, vLLM would load every plugin installed in the environment, so a
    # plugin installed for one model would patch every other model's engine.
    # It overrides the host's value on purpose.
    extra["VLLM_PLUGINS"] = ",".join(recipe.get("vllm_plugins", []))
    return argv, extra


def _gpus(env: dict[str, str], count: int) -> None:
    """CUDA_VISIBLE_DEVICES names exactly ``count`` distinct GPU indices.

    ``count`` is the recipe's precision.tensor_parallel. A server that sees more
    cards may spread over them, one that sees fewer cannot split as recorded;
    either way its numbers would describe another setup than the one reported.
    """
    visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    indices = visible.split(",") if visible else []
    if not (
        len(indices) == count
        and all(re.fullmatch(r"\d+", i) for i in indices)
        and len(set(indices)) == count
    ):
        raise LaunchError(
            f"CUDA_VISIBLE_DEVICES={visible!r}: set exactly {count} distinct GPU "
            f"ind{'ex' if count == 1 else 'ices'} (precision.tensor_parallel = {count}). "
            "A server that sees other cards than that may spread over them or "
            "fail to split, and its numbers then describe a different setup than "
            "the one recorded."
        )


def _hf_cache(env: dict[str, str]) -> Path:
    home = env.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def _cached(model: str, env: dict[str, str], revision: str | None = None) -> bool:
    """A complete snapshot of ``model`` (at ``revision``, if given) is in the HF cache."""
    repo = _hf_cache(env) / f"models--{model.replace('/', '--')}"
    snapshots = repo / "snapshots"
    present = (snapshots / revision).is_dir() if revision else (
        snapshots.is_dir() and any(snapshots.iterdir())
    )
    return present and not any(repo.rglob("*.incomplete"))


def _disk_floor(recipe: dict[str, Any], env: dict[str, str]) -> None:
    floor = env.get("RAVEN_SERVE_MIN_FREE_GB", "").strip()
    if not floor or _cached(recipe["model"], env, recipe.get("revision")):
        return
    cache = _hf_cache(env)
    cache.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cache).free / 1024**3
    need = float(recipe["weights_gb"]) + float(floor)
    if free_gb < need:
        raise LaunchError(
            f"{free_gb:.0f} GB free under {cache}, need ~{recipe['weights_gb']} GB of "
            f"weights + {floor} GB floor (RAVEN_SERVE_MIN_FREE_GB)"
        )


def _excluded_in_snapshot(snapshot: str, patterns: list[str]) -> list[str]:
    """Weight files in ``snapshot`` that the recipe's ``snapshot_ignore`` excludes.

    snapshot_download leaves out what it is told to, but the cache keeps one
    directory per commit: a file another tool downloaded at the same commit is
    still in it. vLLM loads every ``*.safetensors`` of a directory without an
    index, so an excluded weight file that is present anyway changes what is
    served (granite-speech-5.0-2b's out_llm.safetensors fails the load). Other
    excluded files (docs, audio samples, a .bin vLLM never reads beside
    safetensors) change nothing and are left alone.
    """
    root = Path(snapshot)
    found: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() or path.is_symlink():
            rel = path.relative_to(root).as_posix()
            if rel.endswith(".safetensors") and any(fnmatch.fnmatch(rel, p) for p in patterns):
                found.append(rel)
    return sorted(found)


def _fetch(recipe: dict[str, Any]) -> tuple[str | None, str | None]:
    """Download what has to be local before the engine starts: ``(snapshot, checkpoint)``."""
    snapshot = checkpoint = None
    if recipe.get("serve_from_snapshot") or recipe.get("lora_modules"):
        from huggingface_hub import snapshot_download

        ignore = list(recipe.get("snapshot_ignore", []))
        snapshot = snapshot_download(
            recipe["model"], revision=recipe.get("revision"), ignore_patterns=ignore,
        )
        if recipe.get("serve_from_snapshot") and (
            stray := _excluded_in_snapshot(snapshot, ignore)
        ):
            raise LaunchError(
                f"the snapshot at {snapshot} holds files the recipe excludes "
                f"({', '.join(stray)}); vLLM would load them. Remove them from the "
                "snapshot (they are cache symlinks) and start again."
            )
    if recipe.get("checkpoint_file"):
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            recipe["model"], recipe["checkpoint_file"], revision=recipe["revision"]
        )
    return snapshot, checkpoint


def build_stream(
    sid: str,
    streams: dict[str, dict[str, Any]],
    recipes: dict[str, dict[str, Any]],
    env: dict[str, str],
    *,
    python: str,
    snapshot: str | None = None,
    checkpoint: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """``(argv, extra_env)`` for one stream recipe; pure, like :func:`build`."""
    s = streams[sid]
    served = streaming.served_recipe(sid, streams, recipes)
    if s["protocol"] == "vllm-realtime":
        # The base recipe's own `vllm serve`: vLLM registers /v1/realtime for a
        # model class that supports the realtime task.
        argv, extra = build(served, s["base"], env, python=python, snapshot=snapshot)
        extra["RAVEN_STREAM_RECIPE"] = sid
        return argv, extra
    host = _knob(env, "RAVEN_SERVE_HOST", DEFAULT_HOST)
    port = _knob(env, "RAVEN_SERVE_PORT", DEFAULT_PORT)
    if s["protocol"] == "moonshine-voice":
        # The server fetches and checks the vendor's files itself (streaming.toml).
        return ([python, "-m", "uvicorn", "stream_server:app", "--app-dir",
                 str(SERVING_DIR / "moonshine"), "--host", host, "--port", port],
                {"RAVEN_STREAM_RECIPE": sid})
    extra = {
        "RAVEN_SERVING_RECIPE": s["base"],
        "RAVEN_STREAM_RECIPE": sid,
        "BENCH_MODEL": served["model"],
        "BENCH_MODEL_PATH": checkpoint or "",
        "BENCH_MODEL_REVISION": served["revision"],
        "BENCH_GPU_MEM_FRACTION": _knob(env, "RAVEN_SERVE_GPU_MEM", served["gpu_mem"]),
    }
    argv = [python, "-m", "uvicorn", "stream_server:app",
            "--app-dir", str(SERVING_DIR / "nemo"), "--host", host, "--port", port]
    return argv, extra


def _main_streaming(args: argparse.Namespace, recipes: dict[str, dict[str, Any]]) -> int:
    streams, skipped = streaming.load_streams(recipes=recipes)
    if args.list:
        for sid, s in streams.items():
            print(f"{sid}\t{s['protocol']}\t{recipes[s['base']]['model']}")
        for sid, s in skipped.items():
            print(f"{sid}\tskipped\t{s['reason']}")
        return 0
    if not args.name:
        print("[launch] name a stream id or a model id", file=sys.stderr)
        return 2
    if args.name in skipped:
        print(f"[launch] {args.name} is skipped: {skipped[args.name]['reason']}", file=sys.stderr)
        return 1
    sid = streaming.resolve_stream(args.name, streams, recipes)
    s = streams[sid]
    env = dict(os.environ)
    if args.identity:
        print(json.dumps(streaming.stream_identity(sid)))
        return 0
    if args.engine:
        print(recipes[s["base"]]["engine"])
        return 0
    if args.print:
        cmd, extra = build_stream(sid, streams, recipes, env, python=sys.executable)
        print(json.dumps({"stream": streaming.stream_identity(sid), "argv": cmd, "env": extra},
                         indent=2))
        return 0
    served = streaming.served_recipe(sid, streams, recipes)
    try:
        if s.get("device", "cuda") != "cpu":
            _gpus(env, served["precision"]["tensor_parallel"])
        _disk_floor(served, env)
        snapshot, checkpoint = (None, None) if s["protocol"] == "moonshine-voice" else _fetch(served)
    except LaunchError as exc:
        print(f"[launch] refusing {sid}: {exc}", file=sys.stderr)
        return 1
    cmd, extra = build_stream(sid, streams, recipes, env, python=sys.executable,
                              snapshot=snapshot, checkpoint=checkpoint)
    env.update(extra)
    print(f"[launch] stream {sid} ({json.dumps(streaming.stream_identity(sid))}): "
          f"{' '.join(cmd)}", flush=True)
    os.execvpe(cmd[0], cmd, env)
    return 0  # pragma: no cover - exec does not return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("name", nargs="?", help="recipe id or served model id")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--print", action="store_true", help="print the command, start nothing")
    mode.add_argument("--engine", action="store_true", help="print the recipe's engine")
    mode.add_argument("--identity", action="store_true", help="print {id, sha256}")
    mode.add_argument("--list", action="store_true", help="list every recipe")
    ap.add_argument("--streaming", action="store_true",
                    help="the stream recipe (streaming.toml) instead of the offline one")
    args = ap.parse_args(argv)

    recipes = load_recipes()
    if args.streaming:
        return _main_streaming(args, recipes)
    if args.list:
        for rid, r in recipes.items():
            print(f"{rid}\t{r['engine']}\t{r['model']}")
        return 0
    if not args.name:
        ap.error("name a recipe id or a model id")
    rid = resolve(args.name, recipes)
    recipe = recipes[rid]
    env = dict(os.environ)
    if args.engine:
        print(recipe["engine"])
        return 0
    if args.identity:
        print(json.dumps(identity(rid)))
        return 0
    if args.print:
        cmd, extra = build(recipe, rid, env, python=sys.executable)
        print(json.dumps({"recipe": identity(rid), "argv": cmd, "env": extra}, indent=2))
        return 0

    try:
        _gpus(env, recipe["precision"]["tensor_parallel"])
        _disk_floor(recipe, env)
    except LaunchError as exc:
        print(f"[launch] refusing {rid}: {exc}", file=sys.stderr)
        return 1
    try:
        snapshot, checkpoint = _fetch(recipe)
    except LaunchError as exc:
        print(f"[launch] refusing {rid}: {exc}", file=sys.stderr)
        return 1
    cmd, extra = build(recipe, rid, env, python=sys.executable,
                       snapshot=snapshot, checkpoint=checkpoint)
    env.update(extra)
    print(f"[launch] {rid} ({json.dumps(identity(rid))}): {' '.join(cmd)}", flush=True)
    os.execvpe(cmd[0], cmd, env)
    return 0  # pragma: no cover - exec does not return


if __name__ == "__main__":
    sys.exit(main())
