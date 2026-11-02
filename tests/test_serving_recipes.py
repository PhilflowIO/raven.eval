"""The serving recipe in ``serving/``: one per self-hosted model, launched as written.

Pins what a third party relies on to stand the self-hosted models up the way
they were measured: every model in ``SELF_HOSTED_MODELS`` has exactly one recipe
on the right server type, engines are pinned exactly, the launcher builds the
command the recipe describes (host-dependent values only from env), refuses
anything but one GPU, and the recipe hash moves with the recipe and the files
that execute it — and with nothing else.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from raven_asr.config import NEMO_SERVED_MODELS, SELF_HOSTED_MODELS, VLLM_SERVED_MODELS

SERVING = Path(__file__).resolve().parents[1] / "serving"
sys.path.insert(0, str(SERVING))

import launch  # noqa: E402
import recipe  # noqa: E402

RECIPES = recipe.load_recipes()
PY = "/venv/bin/python"


def _build(name: str, **env: str) -> tuple[list[str], dict[str, str]]:
    rid = recipe.resolve(name, RECIPES)
    return launch.build(RECIPES[rid], rid, dict(env), python=PY,
                        snapshot="/snap", checkpoint="/ckpt/m.nemo")


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_one_recipe_per_self_hosted_model_on_its_server_type() -> None:
    by_model = {r["model"]: (rid, r["engine"]) for rid, r in RECIPES.items()}
    assert set(by_model) == set(SELF_HOSTED_MODELS)
    for model, (rid, engine) in by_model.items():
        # The recipe id is the result-directory label, so artifact and recipe pair up.
        assert rid == SELF_HOSTED_MODELS[model].label
        if model in NEMO_SERVED_MODELS:
            assert engine == "nemo", model
        else:
            assert model in VLLM_SERVED_MODELS and engine in {"vllm", "transformers"}, model


def test_every_recipe_says_how_long_audio_is_handled() -> None:
    for rid, r in RECIPES.items():
        assert r["chunking"].strip(), rid
        assert 0 < float(r["gpu_mem"]) < 1, rid


def test_engines_are_pinned_exactly() -> None:
    vllm_req = (SERVING / "vllm" / "requirements.txt").read_text()
    nemo_req = (SERVING / "nemo" / "requirements.txt").read_text()
    for pin in ("vllm[audio]==", "flashinfer-python==", "flashinfer-jit-cache==", "librosa=="):
        assert pin in vllm_req, pin
    assert re.search(r"^nemo_toolkit\[asr\]==\S+$", nemo_req, re.MULTILINE)


def test_vllm_command_is_the_recipe() -> None:
    argv, env = _build("mistralai/Voxtral-Mini-3B-2507")
    assert argv[:3] == ["/venv/bin/vllm", "serve", "mistralai/Voxtral-Mini-3B-2507"]
    assert _flag(argv, "--revision") == "3060fe34b35ba5d44202ce9ff3c097642914f8f3"
    assert _flag(argv, "--gpu-memory-utilization") == "0.85"
    assert _flag(argv, "--max-num-seqs") == "16"
    assert _flag(argv, "--dtype") == "bfloat16"
    assert _flag(argv, "--tokenizer-mode") == "mistral"
    assert _flag(argv, "--host") == "127.0.0.1"
    assert env["RAVEN_SERVING_RECIPE"] == "voxtral-mini-3b-2507"
    assert env["VLLM_MAX_AUDIO_CLIP_FILESIZE_MB"] == "512"


def test_every_vllm_and_transformers_recipe_pins_a_full_commit() -> None:
    for rid, r in RECIPES.items():
        if r["engine"] in recipe.REVISION_ENGINES:
            assert re.fullmatch(r"[0-9a-f]{40}", r["revision"]), rid
        else:
            assert "revision" not in r, rid


def test_granite_serves_the_checkpoint_vllm_028_can_load() -> None:
    """The 2026-10-21 re-upload (e69efe37) renamed the text-model weights to
    ``language_model.model.model.*``; vLLM 0.28.0 has no mapper for that and
    refuses to start. The pin is the checkpoint the September sweep served."""
    argv, _ = _build("ibm-granite/granite-speech-4.1-2b-plus")
    assert argv[2] == "ibm-granite/granite-speech-4.1-2b-plus"
    assert _flag(argv, "--revision") == "1454e6e1e33845ca9280ff65f52cf1141ba6e6e2"
    assert "vllm[audio]==0.28.0" in (SERVING / "vllm" / "requirements.txt").read_text()


@pytest.mark.parametrize("revision", [None, "main", "1454e6e1", "1454E6E1E33845CA9280FF65F52CF1141BA6E6E2"])
def test_a_vllm_recipe_without_a_full_commit_is_refused(
    tmp_path: Path, revision: str | None
) -> None:
    bad = tmp_path / "r.toml"
    pin = "" if revision is None else f'revision = "{revision}"\n'
    bad.write_text('[recipes."x"]\nmodel = "a/b"\nengine = "vllm"\nweights_gb = 1\n'
                   f'gpu_mem = 0.5\nchunking = "none"\n{pin}')
    with pytest.raises(recipe.RecipeError, match="revision"):
        recipe.load_recipes(bad)


def test_host_values_come_from_env_only() -> None:
    argv, _ = _build("Qwen/Qwen3-ASR-1.7B", RAVEN_SERVE_HOST="127.0.0.2",
                     RAVEN_SERVE_PORT="18273", RAVEN_SERVE_GPU_MEM="0.9",
                     RAVEN_SERVE_MAX_NUM_SEQS="24")
    assert (_flag(argv, "--host"), _flag(argv, "--port")) == ("127.0.0.2", "18273")
    assert _flag(argv, "--gpu-memory-utilization") == "0.9"
    assert _flag(argv, "--max-num-seqs") == "24"
    assert _flag(argv, "--max-model-len") == "16384"


def test_phi4_mounts_only_the_speech_lora_from_its_snapshot() -> None:
    argv, _ = _build("microsoft/Phi-4-multimodal-instruct")
    assert argv[2] == "microsoft/Phi-4-multimodal-instruct"  # served by repo id
    assert _flag(argv, "--lora-modules") == "speech=/snap/speech-lora"
    assert "--trust-remote-code" in argv and "--enable-lora" in argv
    assert _flag(argv, "--limit-mm-per-prompt") == '{"audio":1}'


def test_whisper_v3_serves_its_filtered_snapshot_under_the_model_id() -> None:
    argv, _ = _build("openai/whisper-large-v3")
    assert argv[2:5] == ["/snap", "--served-model-name", "openai/whisper-large-v3"]
    assert "--revision" not in argv  # the snapshot itself is the pinned commit
    assert _flag(argv, "--dtype") == "float16"


def test_snapshots_are_downloaded_at_the_recipe_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    calls: list[dict[str, object]] = []

    def fake_snapshot_download(repo_id: str, **kw: object) -> str:
        calls.append({"repo_id": repo_id, **kw})
        return "/snap"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    for rid in ("openai-whisper-large-v3", "phi-4-multimodal"):
        assert launch._fetch(RECIPES[rid]) == ("/snap", None)
        assert calls[-1]["repo_id"] == RECIPES[rid]["model"]
        assert calls[-1]["revision"] == RECIPES[rid]["revision"]


def test_the_disk_check_looks_for_the_pinned_snapshot(tmp_path: Path) -> None:
    repo = tmp_path / "hub" / "models--a--b" / "snapshots"
    (repo / ("0" * 40)).mkdir(parents=True)
    env = {"HF_HOME": str(tmp_path)}
    assert launch._cached("a/b", env, "0" * 40)
    assert not launch._cached("a/b", env, "1" * 40)
    assert launch._cached("a/b", env)


def test_nemo_command_and_batching_come_from_the_recipe() -> None:
    argv, env = _build("primeline/parakeet-primeline", RAVEN_SERVE_PORT="18171")
    assert argv[:4] == [PY, "-m", "uvicorn", "server:app"]
    assert _flag(argv, "--app-dir") == str(SERVING / "nemo")
    assert _flag(argv, "--port") == "18171"
    assert env["BENCH_MODEL"] == "primeline/parakeet-primeline"
    assert env["BENCH_MODEL_PATH"] == "/ckpt/m.nemo"
    assert (env["NEMO_MAX_BATCH"], env["NEMO_BATCH_WAIT_MS"], env["NEMO_BATCH_MAX_AUDIO_S"]) == (
        "16", "20", "40"
    )
    assert env["RAVEN_SERVING_RECIPE"] == "primeline-parakeet"


def test_the_transformers_stand_in_binds_where_the_engine_would() -> None:
    argv, env = _build("moonshine-ai/moonshine-streaming-small-de", RAVEN_SERVE_PORT="18273")
    assert argv == [PY, str(SERVING / "vllm" / "moonshine_shim.py")]
    assert env["MOONSHINE_PORT"] == "18273" and env["MOONSHINE_GPU_MEM_FRACTION"] == "0.4"
    assert env["MOONSHINE_REVISION"] == "56473ba32b22b484fdfc7eb03d34c393ff494197"


@pytest.mark.parametrize("visible", ["", "0,1", "all"])
def test_launch_refuses_anything_but_one_gpu(visible: str) -> None:
    with pytest.raises(launch.LaunchError):
        launch._single_gpu({"CUDA_VISIBLE_DEVICES": visible})
    launch._single_gpu({"CUDA_VISIBLE_DEVICES": "1"})


def test_print_mode_starts_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    assert launch.main(["Qwen/Qwen3-ASR-1.7B", "--print"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["recipe"]["id"] == "qwen3-asr-1.7b"
    assert out["argv"][1:3] == ["serve", "Qwen/Qwen3-ASR-1.7B"]


def test_the_hash_moves_with_the_recipe_and_its_files_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = {rid: recipe.recipe_sha256(rid, RECIPES) for rid in RECIPES}
    assert len(set(base.values())) == len(base)
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in base.values())

    edited = {rid: dict(r) for rid, r in RECIPES.items()}
    edited["qwen3-asr-1.7b"]["max_model_len"] = 8192
    assert recipe.recipe_sha256("qwen3-asr-1.7b", edited) != base["qwen3-asr-1.7b"]
    assert recipe.recipe_sha256("voxtral-mini-3b-2507", edited) == base["voxtral-mini-3b-2507"]

    # An engine file is part of its engine's recipes, not of the others'.
    copy = tmp_path / "serving"
    for rel in {f for files in recipe.ENGINE_FILES.values() for f in files}:
        (copy / rel).parent.mkdir(parents=True, exist_ok=True)
        (copy / rel).write_bytes((SERVING / rel).read_bytes())
    monkeypatch.setattr(recipe, "SERVING_DIR", copy)
    assert recipe.recipe_sha256("primeline-parakeet", RECIPES) == base["primeline-parakeet"]
    (copy / "nemo" / "batcher.py").write_text("# changed\n")
    assert recipe.recipe_sha256("primeline-parakeet", RECIPES) != base["primeline-parakeet"]
    assert recipe.recipe_sha256("qwen3-asr-1.7b", RECIPES) == base["qwen3-asr-1.7b"]


def test_malformed_recipes_are_refused(tmp_path: Path) -> None:
    bad = tmp_path / "r.toml"
    bad.write_text('[recipes."x"]\nmodel = "a/b"\nengine = "nemo"\nweights_gb = 1\n'
                   'gpu_mem = 0.5\nchunking = "none"\ndtype = "float16"\n')
    with pytest.raises(recipe.RecipeError, match="do not apply"):
        recipe.load_recipes(bad)
    # NeMo's from_pretrained takes no revision: a pin there would be a lie.
    bad.write_text('[recipes."x"]\nmodel = "a/b"\nengine = "nemo"\nweights_gb = 1\n'
                   'gpu_mem = 0.5\nchunking = "none"\nrevision = "' + "0" * 40 + '"\n')
    with pytest.raises(recipe.RecipeError, match="do not apply"):
        recipe.load_recipes(bad)


def test_nothing_private_lands_in_serving() -> None:
    """The recipe is published: no internal addresses, hosts or operator paths."""
    pattern = re.compile(
        r"100\.64\.|192\.168\.|10\.\d+\.\d+\.\d+|tailnet|philsai|ragio|stt-exo|"
        r"/mnt/|/home/|~/|infisical|flow\.raven", re.IGNORECASE,
    )
    for path in SERVING.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".toml", ".txt", ".md", ".sh"}:
            hits = pattern.findall(path.read_text(encoding="utf-8"))
            assert not hits, f"{path.relative_to(SERVING)}: {hits}"
