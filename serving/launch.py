"""Start the server for one self-hosted model from its recipe (``recipes.toml``).

    python serving/launch.py <recipe-id | model-id>            start it (exec)
    python serving/launch.py <recipe-id | model-id> --print    the command, as JSON
    python serving/launch.py <recipe-id | model-id> --engine   vllm | nemo | transformers
    python serving/launch.py <recipe-id | model-id> --identity {"id", "sha256"}
    python serving/launch.py --list                            every recipe id and model

Run it with the Python of the engine's environment: the NeMo venv for ``nemo``
recipes, the vLLM venv for ``vllm`` and ``transformers`` recipes (the ``vllm``
executable is taken from beside that interpreter). The process is replaced by
the server, so a supervisor's signals reach the server directly.

What this starts is the *engine*. For ``vllm`` and ``transformers`` recipes the
harness talks to ``vllm/shim.py`` in front of it (same environment, its own
port); the NeMo server speaks the harness's protocol itself.

Environment (everything a host decides, nothing a model decides):

    CUDA_VISIBLE_DEVICES      exactly one GPU index — required. The process sees
                              that card as device 0 and no other.
    RAVEN_SERVE_HOST          bind address (default 127.0.0.1)
    RAVEN_SERVE_PORT          port (default 8000)
    RAVEN_SERVE_GPU_MEM       VRAM budget fraction (default: the recipe's gpu_mem)
    RAVEN_SERVE_MAX_NUM_SEQS  vLLM --max-num-seqs (default: the recipe's). Give the
                              shim the same value: it reports it in x_raven.batching.
    RAVEN_SERVE_MIN_FREE_GB   if set, refuse to start a download unless the HF
                              cache keeps this much free beyond the weights
    HF_HOME, HF_TOKEN         as usual for huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipe import SERVING_DIR, identity, load_recipes, resolve  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
#: vLLM's transcription endpoint rejects uploads above 25 MB by default; a
#: 10-minute 16 kHz PCM16 clip is ~19 MB. Aligned with the shim's upload cap.
VLLM_MAX_AUDIO_MB = "512"
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
    mount LoRA modules out of one; ``checkpoint`` the local .nemo path for recipes with
    ``checkpoint_file``. Both are resolved by :func:`_fetch` before a real launch.
    """
    host = _knob(env, "RAVEN_SERVE_HOST", DEFAULT_HOST)
    port = _knob(env, "RAVEN_SERVE_PORT", DEFAULT_PORT)
    gpu_mem = _knob(env, "RAVEN_SERVE_GPU_MEM", recipe["gpu_mem"])
    engine = recipe["engine"]
    extra = {"RAVEN_SERVING_RECIPE": rid}

    if engine == "nemo":
        extra.update({
            "BENCH_MODEL": recipe["model"],
            "BENCH_MODEL_PATH": checkpoint or "",
            "BENCH_GPU_MEM_FRACTION": gpu_mem,
            "BENCH_LONGFORM": "true" if recipe.get("longform", True) else "false",
            "NEMO_MAX_BATCH": str(recipe.get("max_batch", 16)),
            "NEMO_BATCH_WAIT_MS": str(recipe.get("batch_wait_ms", 20)),
            "NEMO_BATCH_MAX_AUDIO_S": str(recipe.get("batch_max_audio_s", 40)),
        })
        argv = [python, "-m", "uvicorn", "server:app",
                "--app-dir", str(SERVING_DIR / "nemo"), "--host", host, "--port", port]
        return argv, extra

    if engine == "transformers":
        extra.update({
            "MOONSHINE_MODEL": recipe["model"],
            "MOONSHINE_HOST": host,
            "MOONSHINE_PORT": port,
            "MOONSHINE_GPU_MEM_FRACTION": gpu_mem,
        })
        return [python, str(SERVING_DIR / "vllm" / "moonshine_shim.py")], extra

    # vllm
    vllm = str(Path(python).parent / "vllm")
    seqs = _knob(env, "RAVEN_SERVE_MAX_NUM_SEQS", recipe.get("max_num_seqs", 16))
    snap = snapshot or SNAPSHOT_PLACEHOLDER
    if recipe.get("serve_from_snapshot"):
        target = [snap, "--served-model-name", recipe["model"]]
    else:
        target = [recipe["model"]]
    argv = [vllm, "serve", *target, "--host", host, "--port", port,
            "--gpu-memory-utilization", gpu_mem, "--max-num-seqs", seqs]
    if "dtype" in recipe:
        argv += ["--dtype", recipe["dtype"]]
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
    return argv, extra


def _single_gpu(env: dict[str, str]) -> None:
    visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not re.fullmatch(r"\d+", visible):
        raise LaunchError(
            f"CUDA_VISIBLE_DEVICES={visible!r}: set exactly one GPU index. A server "
            "that sees several cards may spread over them, and its numbers then "
            "describe a different setup than the one recorded."
        )


def _hf_cache(env: dict[str, str]) -> Path:
    home = env.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def _cached(model: str, env: dict[str, str]) -> bool:
    """A complete snapshot of ``model`` is in the HF cache (no partial blobs)."""
    repo = _hf_cache(env) / f"models--{model.replace('/', '--')}"
    snapshots = repo / "snapshots"
    return (
        snapshots.is_dir()
        and any(snapshots.iterdir())
        and not any(repo.rglob("*.incomplete"))
    )


def _disk_floor(recipe: dict[str, Any], env: dict[str, str]) -> None:
    floor = env.get("RAVEN_SERVE_MIN_FREE_GB", "").strip()
    if not floor or _cached(recipe["model"], env):
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


def _fetch(recipe: dict[str, Any]) -> tuple[str | None, str | None]:
    """Download what has to be local before the engine starts: ``(snapshot, checkpoint)``."""
    snapshot = checkpoint = None
    if recipe.get("serve_from_snapshot") or recipe.get("lora_modules"):
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            recipe["model"], ignore_patterns=list(recipe.get("snapshot_ignore", []))
        )
    if recipe.get("checkpoint_file"):
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(recipe["model"], recipe["checkpoint_file"])
    return snapshot, checkpoint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("name", nargs="?", help="recipe id or served model id")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--print", action="store_true", help="print the command, start nothing")
    mode.add_argument("--engine", action="store_true", help="print the recipe's engine")
    mode.add_argument("--identity", action="store_true", help="print {id, sha256}")
    mode.add_argument("--list", action="store_true", help="list every recipe")
    args = ap.parse_args(argv)

    recipes = load_recipes()
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
        _single_gpu(env)
        _disk_floor(recipe, env)
    except LaunchError as exc:
        print(f"[launch] refusing {rid}: {exc}", file=sys.stderr)
        return 1
    snapshot, checkpoint = _fetch(recipe)
    cmd, extra = build(recipe, rid, env, python=sys.executable,
                       snapshot=snapshot, checkpoint=checkpoint)
    env.update(extra)
    print(f"[launch] {rid} ({json.dumps(identity(rid))}): {' '.join(cmd)}", flush=True)
    os.execvpe(cmd[0], cmd, env)
    return 0  # pragma: no cover - exec does not return


if __name__ == "__main__":
    sys.exit(main())
