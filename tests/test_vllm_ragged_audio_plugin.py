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
    assert entries == {"raven_ragged_audio": "raven_vllm_plugin:register"}
    assert recipe.VLLM_PLUGINS["raven_ragged_audio"] == project["name"]


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
    assert [rid for rid, r in recipes.items() if "raven_ragged_audio" in r.get("vllm_plugins", [])] == [
        "vibevoice-asr"
    ]
