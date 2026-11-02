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
    for pin in ("vllm[audio]==", "flashinfer-python==", "flashinfer-cubin==",
                "flashinfer-jit-cache==", "transformers==", "librosa=="):
        assert pin in vllm_req, pin
    assert re.search(r"^nemo_toolkit\[asr\]==\S+$", nemo_req, re.MULTILINE)


def _pins(text: str) -> dict[str, str]:
    return dict(re.findall(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==(\S+)$", text, re.MULTILINE))


def test_flashinfer_kernels_are_prebuilt_for_the_pinned_flashinfer_and_cuda() -> None:
    """The host has no CUDA toolkit: a jit-cache of another flashinfer version or
    CUDA line is refused or misses, and the sampler then JIT-compiles and dies."""
    text = (SERVING / "vllm" / "requirements.txt").read_text()
    pins = _pins(text)
    version, _, cuda = pins["flashinfer-jit-cache"].partition("+")
    assert version == pins["flashinfer-python"]
    assert re.fullmatch(r"cu\d{3}", cuda), cuda
    indexes = [u.rstrip("/") for u in re.findall(r"^--extra-index-url (\S+)$", text, re.MULTILINE)]
    assert any(u.endswith(f"/whl/{cuda}") for u in indexes), indexes


def test_vllm_enables_flashinfer_without_a_cuda_toolkit() -> None:
    """vLLM 0.33 turns FlashInfer on only when flashinfer-cubin is importable (or
    nvcc and ninja exist); the jit-cache alone leaves it off. The cubin wheel is
    CUDA-independent and lives on FlashInfer's own index, not on PyPI."""
    text = (SERVING / "vllm" / "requirements.txt").read_text()
    pins = _pins(text)
    assert pins["flashinfer-cubin"] == pins["flashinfer-python"]
    indexes = [u.rstrip("/") for u in re.findall(r"^--extra-index-url (\S+)$", text, re.MULTILINE)]
    assert "https://flashinfer.ai/whl" in indexes, indexes


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


def test_every_recipe_pins_a_full_commit() -> None:
    for rid, r in RECIPES.items():
        assert re.fullmatch(r"[0-9a-f]{40}", r["revision"]), rid
        if r["engine"] == "nemo":
            # from_pretrained takes no revision: the pin only holds through a .nemo
            # file downloaded at it.
            assert r["checkpoint_file"].endswith(".nemo"), rid


def test_granite_serves_the_retrained_checkpoint_on_an_engine_that_maps_it() -> None:
    """IBM's 2026-10-21 re-upload (e69efe37) nests the text-model weights at
    ``language_model.model.model.*``; vLLM maps that only from 0.33.0 on
    (vllm-project/vllm#63050). Older engines refuse to start on it."""
    argv, _ = _build("ibm-granite/granite-speech-4.1-2b-plus")
    assert argv[2] == "ibm-granite/granite-speech-4.1-2b-plus"
    assert _flag(argv, "--revision") == "e69efe37f97b6283c61f5a8dadda2b3eb67765d6"
    vllm = _pins((SERVING / "vllm" / "requirements.txt").read_text())["vllm"]
    assert tuple(int(x) for x in vllm.split(".")[:2]) >= (0, 33), vllm


_DECODING = {
    "vllm": 'decoding = { mode = "greedy", max_tokens = 1024 }',
    "transformers": 'decoding = { mode = "greedy", max_tokens = 1024 }',
    "nemo": 'decoding = { mode = "greedy", strategy = "greedy_batch", max_symbols = 10 }',
}


@pytest.mark.parametrize("engine", ["vllm", "transformers", "nemo"])
@pytest.mark.parametrize("revision", [None, "main", "1454e6e1", "1454E6E1E33845CA9280FF65F52CF1141BA6E6E2"])
def test_a_recipe_without_a_full_commit_is_refused(
    tmp_path: Path, engine: str, revision: str | None
) -> None:
    bad = tmp_path / "r.toml"
    pin = "" if revision is None else f'revision = "{revision}"\n'
    ckpt = {"nemo": 'checkpoint_file = "m.nemo"\n',
            "transformers": 'model_class = "GraniteSpeech5ForCTC"\n'}.get(engine, "")
    bad.write_text(f'[recipes."x"]\nmodel = "a/b"\nengine = "{engine}"\nweights_gb = 1\n'
                   f'gpu_mem = 0.5\nchunking = "none"\n{_DECODING[engine]}\n{pin}{ckpt}')
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


#: The commits the 2026-11 round loaded: the bench cache's refs/main for the three
#: nvidia repos, and primeline's main (unchanged since 2026-04-24) for the fourth —
#: all four still each repo's main on 2026-11-02. The unified model: its main then.
NEMO_PINS = {
    "nvidia/parakeet-tdt-0.6b-v3": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    "nvidia/canary-1b-v2": "d455706339a6b32e1aa40f82c713a482a0c938e2",
    "nvidia/nemotron-3.5-asr-streaming-0.6b": "ea30d66debe3740a08b573244286791d423d6b3e",
    "primeline/parakeet-primeline": "3f1a9bcb611dfeda53fe74fe5f1a3d5701e8023e",
    "nvidia/nemotron-labs-unified-asr-0.6b": "023ce907267a09a308a5cb69239447cb3d0c1597",
}


@pytest.mark.parametrize("model", sorted(NEMO_PINS))
def test_nemo_checkpoints_are_downloaded_at_the_recipe_revision(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    import huggingface_hub

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_hf_hub_download(*args: object, **kw: object) -> str:
        calls.append((args, kw))
        return "/ckpt/m.nemo"

    def no_snapshot(*_: object, **__: object) -> str:
        raise AssertionError("a NeMo recipe needs its .nemo file, not a snapshot")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_snapshot)
    r = RECIPES[recipe.resolve(model, RECIPES)]
    assert r["revision"] == NEMO_PINS[model]
    assert launch._fetch(r) == (None, "/ckpt/m.nemo")
    assert calls == [((model, r["checkpoint_file"]), {"revision": NEMO_PINS[model]})]
    _, env = _build(model)
    assert env["BENCH_MODEL_PATH"] == "/ckpt/m.nemo"
    assert env["BENCH_MODEL_REVISION"] == NEMO_PINS[model]


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
    assert env["BENCH_MODEL_REVISION"] == "3f1a9bcb611dfeda53fe74fe5f1a3d5701e8023e"
    assert (env["NEMO_MAX_BATCH"], env["NEMO_BATCH_WAIT_MS"], env["NEMO_BATCH_MAX_AUDIO_S"]) == (
        "16", "20", "40"
    )
    assert env["RAVEN_SERVING_RECIPE"] == "primeline-parakeet"


def test_the_transformers_stand_in_binds_where_the_engine_would() -> None:
    argv, env = _build("moonshine-ai/moonshine-streaming-small-de", RAVEN_SERVE_PORT="18273")
    assert argv == [PY, str(SERVING / "vllm" / "transformers_server.py")]
    assert env["TRANSFORMERS_PORT"] == "18273" and env["TRANSFORMERS_GPU_MEM_FRACTION"] == "0.4"
    assert env["TRANSFORMERS_REVISION"] == "56473ba32b22b484fdfc7eb03d34c393ff494197"
    assert env["TRANSFORMERS_CLASS"] == "MoonshineStreamingForConditionalGeneration"


def test_the_ctc_model_runs_on_the_stand_in_under_its_own_class() -> None:
    argv, env = _build("ibm-granite/granite-speech-5.0-470m-turboctc")
    assert argv == [PY, str(SERVING / "vllm" / "transformers_server.py")]
    assert env["TRANSFORMERS_MODEL"] == "ibm-granite/granite-speech-5.0-470m-turboctc"
    assert env["TRANSFORMERS_CLASS"] == "GraniteSpeech5ForCTC"
    # The multilingual 2026-10-21 release, not the English-only 947f59a.
    assert env["TRANSFORMERS_REVISION"] == "1dd23f46f6d4e6eae2531982ba2e33220086dd97"


def test_every_stand_in_class_is_one_the_server_decodes() -> None:
    """The server refuses a class it has no decoder for; a recipe must not name one."""
    import ast

    tree = ast.parse((SERVING / "vllm" / "transformers_server.py").read_text())
    decoders = next(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DECODERS" for t in node.targets)
    )
    known = {k.value for k in decoders.keys}  # type: ignore[union-attr]
    for rid, r in RECIPES.items():
        if r["engine"] == "transformers":
            assert r["model_class"] in known, rid


def test_a_transformers_recipe_without_a_class_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "r.toml"
    bad.write_text('[recipes."x"]\nmodel = "a/b"\nengine = "transformers"\nweights_gb = 1\n'
                   f'gpu_mem = 0.5\nchunking = "none"\nrevision = "{"a" * 40}"\n'
                   + _DECODING["transformers"] + "\n")
    with pytest.raises(recipe.RecipeError, match="model_class"):
        recipe.load_recipes(bad)


def test_granite_5_serves_a_snapshot_without_its_stray_output_head() -> None:
    """out_llm.safetensors sits beside model.safetensors with no index; vLLM would
    load both and fail on tensors no module owns."""
    argv, _ = _build("ibm-granite/granite-speech-5.0-2b")
    assert argv[2:5] == ["/snap", "--served-model-name", "ibm-granite/granite-speech-5.0-2b"]
    assert "out_llm.safetensors" in RECIPES["granite-speech-5.0-2b"]["snapshot_ignore"]
    assert _flag(argv, "--max-model-len") == "2048"


def test_a_snapshot_still_holding_an_excluded_weight_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"")
    (snap / "multilingual_sample.wav").write_bytes(b"")  # excluded, but never loaded
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *_a, **_k: str(snap))
    r = RECIPES["granite-speech-5.0-2b"]
    assert launch._fetch(r) == (str(snap), None)
    (snap / "out_llm.safetensors").write_bytes(b"")  # left by an earlier full download
    with pytest.raises(launch.LaunchError, match="out_llm.safetensors"):
        launch._fetch(r)


def test_voxtral_realtime_is_served_offline_with_the_card_flags() -> None:
    argv, _ = _build("mistralai/Voxtral-Mini-4B-Realtime-2602")
    assert _flag(argv, "--revision") == "2769294da9567371363522aac9bbcfdd19447add"
    assert (_flag(argv, "--tokenizer-mode"), _flag(argv, "--config-format"),
            _flag(argv, "--load-format")) == ("mistral", "mistral", "mistral")
    assert json.loads(_flag(argv, "--compilation-config")) == {"cudagraph_mode": "PIECEWISE"}
    assert _flag(argv, "--max-model-len") == "16384"


def test_vibevoice_is_the_transformers_format_release_with_a_card_to_itself() -> None:
    argv, _ = _build("microsoft/VibeVoice-ASR-HF")
    assert argv[2] == "microsoft/VibeVoice-ASR-HF"
    assert _flag(argv, "--revision") == "f22241c2062b3b25272bf117397e03d73381037a"
    assert float(_flag(argv, "--gpu-memory-utilization")) >= 0.9
    assert _flag(argv, "--max-num-seqs") == "4"
    assert "--trust-remote-code" not in argv  # native in the pinned transformers


def test_every_recipe_added_since_2026_11_cites_german_evidence() -> None:
    """Issue #77: the reason a model is in a German benchmark is on record."""
    for rid in ("openai-whisper-large-v3-turbo", "granite-speech-5.0-2b",
                "granite-speech-5.0-470m-turboctc", "voxtral-mini-4b-realtime-2602",
                "vibevoice-asr", "nvidia-nemotron-labs-unified-asr-0.6b"):
        notes = RECIPES[rid]["notes"]
        assert "German" in notes and "https://huggingface.co/" in notes, rid


def test_the_unified_model_is_served_in_offline_mode() -> None:
    """longform would switch it to local attention — not the offline mode its
    card's WER tables report."""
    r = RECIPES["nvidia-nemotron-labs-unified-asr-0.6b"]
    assert r["longform"] is False
    _, env = _build("nvidia/nemotron-labs-unified-asr-0.6b")
    assert env["BENCH_LONGFORM"] == "false"


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
    nemo = ('[recipes."x"]\nmodel = "a/b"\nengine = "nemo"\nweights_gb = 1\n'
            'gpu_mem = 0.5\nchunking = "none"\nrevision = "' + "0" * 40 + '"\n'
            + _DECODING["nemo"] + "\n")
    bad.write_text(nemo + 'checkpoint_file = "m.nemo"\ndtype = "float16"\n')
    with pytest.raises(recipe.RecipeError, match="do not apply"):
        recipe.load_recipes(bad)
    # NeMo's from_pretrained takes no revision: without a .nemo file to download
    # at the pin, the server would load whatever main is.
    bad.write_text(nemo)
    with pytest.raises(recipe.RecipeError, match="checkpoint_file"):
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


def test_every_recipe_decodes_greedily_and_bounds_its_output() -> None:
    """Issue #62: decoding mode and output length are part of every recipe."""
    for rid, r in recipe.load_recipes().items():
        dec = r["decoding"]
        assert dec["mode"] == "greedy", rid
        if r["engine"] in ("vllm", "transformers"):
            assert dec["max_tokens"] == 1024, rid
        else:
            assert dec["strategy"] in ("greedy_batch", "beam"), rid
            if dec["strategy"] == "beam":
                assert dec["beam_size"] == 1, rid  # beam of one = greedy
            else:
                assert dec["max_symbols"] > 0, rid


@pytest.mark.parametrize(
    ("decoding", "match"),
    [
        ('decoding = { mode = "sample", max_tokens = 1024 }', "mode"),
        ('decoding = { mode = "greedy" }', "max_tokens"),
        ('decoding = { mode = "greedy", max_tokens = 0 }', "positive"),
        ('decoding = { mode = "greedy", max_tokens = 1024, top_k = 5 }', "do not apply"),
    ],
)
def test_a_malformed_decoding_table_is_refused(tmp_path: Path, decoding: str, match: str) -> None:
    bad = tmp_path / "r.toml"
    bad.write_text('[recipes."x"]\nmodel = "a/b"\nengine = "vllm"\nweights_gb = 1\n'
                   f'gpu_mem = 0.5\nchunking = "none"\nrevision = "{"a" * 40}"\n{decoding}\n')
    with pytest.raises(recipe.RecipeError, match=match):
        recipe.load_recipes(bad)
