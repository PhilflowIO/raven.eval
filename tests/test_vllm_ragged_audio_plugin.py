"""The vLLM plugin that fixes per-clip encoding of audio of different lengths.

vLLM and torch never enter this repo's install, so ``_select_item_kwargs`` is
mirrored from vLLM 0.33.0 (vllm/model_executor/models/transformers/multimodal.py:
1154-1165) over a stand-in tensor: what is pinned is the wrapper's contract, that
each clip's mask reaches the model as a batch of one (raven.eval#93).
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "serving" / "vllm" / "plugin"
sys.path.insert(0, str(ROOT / "serving"))
import recipe  # noqa: E402


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "raven_vllm_plugin", PLUGIN_DIR / "raven_vllm_plugin.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tensor:
    """What the wrapper may touch of a torch.Tensor: length, slicing, unsqueeze."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, item: slice) -> _Tensor:
        return _Tensor((len(range(*item.indices(self.shape[0]))), *self.shape[1:]))

    def unsqueeze(self, dim: int) -> _Tensor:
        assert dim == 0
        return _Tensor((1, *self.shape))


def _vllm_select(_self: Any, kwargs: dict[str, Any], index: int, num_items: int) -> dict[str, Any]:
    return {
        key: value[index : index + 1]
        if isinstance(value, (_Tensor, list)) and len(value) == num_items
        else value
        for key, value in kwargs.items()
    }


def test_a_ragged_clips_mask_reaches_the_model_as_a_batch_of_one() -> None:
    """Unwrapped this is the crash: a one-element list, no ``.sum(dim=-1)``."""
    masks = [_Tensor((112000,)), _Tensor((291200,)), _Tensor((409600,))]
    assert isinstance(_vllm_select(None, {"padding_mask": masks}, 1, 3)["padding_mask"], list)

    select = _load().with_batch_of_one(_vllm_select)
    for index, mask in enumerate(masks):
        out = select(None, {"padding_mask": masks}, index, 3)["padding_mask"]
        assert isinstance(out, _Tensor) and out.shape == (1, mask.shape[0])


def test_entries_that_are_not_ragged_lists_keep_vllms_behaviour() -> None:
    select = _load().with_batch_of_one(_vllm_select)
    stacked, scalar, other = _Tensor((3, 7)), _Tensor((5,)), ["a", "b", "c"]
    out = select(None, {"s": stacked, "n": scalar, "o": other, "f": 1.5}, 2, 3)
    assert out["s"].shape == (1, 7)
    assert out["n"] is scalar
    assert out["o"] == ["c"]
    assert out["f"] == 1.5


def _fake_vllm(monkeypatch: pytest.MonkeyPatch, mixin: type) -> None:
    name = "vllm.model_executor.models.transformers.multimodal"
    module = types.ModuleType(name)
    module.MultiModalMixin = mixin  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)


def test_register_patches_once_however_often_vllm_loads_it(monkeypatch: pytest.MonkeyPatch) -> None:
    class Mixin:
        _select_item_kwargs = _vllm_select

    _fake_vllm(monkeypatch, Mixin)
    plugin = _load()
    plugin.register()
    patched = Mixin._select_item_kwargs
    assert patched is not _vllm_select
    plugin.register()
    assert Mixin._select_item_kwargs is patched


def test_register_is_loud_when_vllm_no_longer_has_the_method(monkeypatch: pytest.MonkeyPatch) -> None:
    class Mixin:
        pass

    _fake_vllm(monkeypatch, Mixin)
    with pytest.raises(AttributeError):
        _load().register()


def test_the_entry_point_is_the_plugin_recipes_name() -> None:
    project = tomllib.loads((PLUGIN_DIR / "pyproject.toml").read_text())["project"]
    entries = project["entry-points"]["vllm.general_plugins"]
    assert entries == {
        "raven_ragged_audio": "raven_vllm_plugin:register",
        "raven_audio_duration": "raven_vllm_plugin:register_audio_duration",
    }
    assert recipe.VLLM_PLUGINS["raven_ragged_audio"] == project["name"]
    assert recipe.VLLM_PLUGINS["raven_audio_duration"] == project["name"]


def test_the_plugin_is_installed_from_a_commit_of_this_repository() -> None:
    requirements = (ROOT / "serving" / "vllm" / "requirements.txt").read_text()
    assert re.search(
        r"^raven-vllm-plugin @ git\+https://github\.com/PhilflowIO/raven\.eval\.git"
        r"@[0-9a-f]{40}#subdirectory=serving/vllm/plugin$",
        requirements,
        re.MULTILINE,
    )


def test_only_vibevoice_loads_it() -> None:
    recipes = recipe.load_recipes()
    for plugin in ("raven_ragged_audio", "raven_audio_duration"):
        assert [rid for rid, r in recipes.items() if plugin in r.get("vllm_plugins", [])] == [
            "vibevoice-asr"
        ]


# ── raven_audio_duration ─────────────────────────────────────────────────────
# Pinned against what vLLM 0.33.0 does: the rendered chat template arrives as
# token ids (ProcessorInputs.prompt), with the placeholder still in it. Fakes for
# the tokenizer and the mm items: what matters is the wrapper's contract.

_PROMPT = "user\nThis is a <|AUDIO_DURATION|> seconds audio.\n"
_TOKEN = "<|AUDIO_DURATION|>"


class _Tok:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join(map(chr, ids))


class _Audio:
    def __init__(self, *clips: Any) -> None:
        self.clips = clips

    def get_count(self) -> int:
        return len(self.clips)

    def get(self, index: int) -> Any:
        return self.clips[index]


def _inputs(prompt: str, *clips: Any) -> Any:
    import dataclasses

    @dataclasses.dataclass
    class Inputs:
        prompt: list[int]
        mm_data_items: dict[str, Any]
        hf_processor_mm_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    return Inputs(_Tok().encode(prompt), {"audio": _Audio(*clips)})


def _processor(token: str | None = _TOKEN) -> Any:
    hf = types.SimpleNamespace(feature_extractor=types.SimpleNamespace(sampling_rate=24000))
    if token is not None:
        hf.audio_duration_token = token
    info = types.SimpleNamespace(get_hf_processor=lambda **_: hf, get_tokenizer=_Tok)
    return types.SimpleNamespace(info=info)


def _apply(token: str | None, inputs: Any) -> list[int]:
    plugin = _load()
    wrapped = plugin.with_audio_duration(lambda _self, passed, _ctx: passed.prompt)
    return wrapped(_processor(token), inputs, object())


def test_the_prompt_states_the_clips_duration_like_the_hf_processor() -> None:
    got = _apply(_TOKEN, _inputs(_PROMPT, [0] * int(24000 * 4.63)))
    assert "".join(map(chr, got)) == "user\nThis is a 4.63 seconds audio.\n"


def test_a_prompt_without_the_placeholder_passes_through_untouched() -> None:
    prompt = "user\nNo placeholder here.\n"
    assert "".join(map(chr, _apply(_TOKEN, _inputs(prompt, [0] * 24000)))) == prompt


def test_a_processor_without_the_token_passes_through_untouched() -> None:
    assert "".join(map(chr, _apply(None, _inputs(_PROMPT, [0] * 24000)))) == _PROMPT


@pytest.mark.parametrize("clips", [(), ([0] * 24000, [0] * 12000)])
def test_placeholders_that_do_not_pair_up_with_clips_are_refused(clips: tuple) -> None:
    with pytest.raises(ValueError, match="placeholders"):
        _apply(_TOKEN, _inputs(_PROMPT, *clips))


def test_a_clip_with_no_data_is_refused() -> None:
    with pytest.raises(ValueError, match="cached-only"):
        _apply(_TOKEN, _inputs(_PROMPT, None))


def test_register_audio_duration_is_idempotent_and_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    class Proc:
        def apply(self, inputs: Any, ctx: Any) -> Any:
            return inputs

    module = types.ModuleType("vllm.model_executor.models.transformers.multimodal")
    module.MultiModalProcessor = Proc
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.models", "vllm.model_executor.models.transformers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    plugin = _load()
    plugin.register_audio_duration()
    patched = Proc.apply
    plugin.register_audio_duration()
    assert Proc.apply is patched
    del Proc.apply
    Proc2 = type("Proc2", (), {})
    module.MultiModalProcessor = Proc2
    with pytest.raises(AttributeError):
        _load().register_audio_duration()
