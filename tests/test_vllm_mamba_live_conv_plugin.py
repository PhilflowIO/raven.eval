"""The vLLM plugin that keeps Mamba2's conv weights on the loaded parameter.

vLLM 0.33.0's ``MambaMixer2.__init__`` stores a view of ``conv1d.weight`` as the
buffer ``conv_weights`` (vllm/model_executor/layers/mamba/mamba_mixer2.py:473-477)
and the forward reads only the buffer (:834, :952). Weight offloading swaps the
parameter's ``.data`` after the layer is built (offloader/uva.py:114-122), so the
buffer keeps memory no weight is ever loaded into (raven.eval#101). vLLM and torch
never enter this repo's install: ``nn.Module``'s buffer rules and the mixer's
constructor are mirrored over stand-ins. What is pinned is the plugin's contract:
after the offloader moves the weight, ``conv_weights`` is a view of where it went.
"""

from __future__ import annotations

import importlib.util
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

CONTROL = "nemotron-3.5-transcribe-30b-a4b-bf16-control"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location(
        "raven_vllm_plugin", PLUGIN_DIR / "raven_vllm_plugin.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _View:
    """A view: it keeps the storage it was taken of, whatever the parameter does later."""

    def __init__(self, storage: list[float], shape: tuple[int, ...]) -> None:
        self.storage, self.shape = storage, shape


class _Weight:
    """A parameter: ``data`` is the storage an offloader may replace."""

    def __init__(self, channels: int, kernel: int) -> None:
        self.shape = (channels, 1, kernel)
        self.data = [0.0] * (channels * kernel)  # torch.empty: nothing loaded yet

    def size(self, dim: int) -> int:
        return self.shape[dim]

    def view(self, *shape: int) -> _View:
        return _View(self.data, shape)


class _Module:
    """``torch.nn.Module``'s buffer rules: ``register_buffer`` refuses a name that
    already resolves to something else, and ``__getattr__`` finds buffers only
    after normal lookup (class attributes, properties) has failed."""

    def __init__(self) -> None:
        self.__dict__["_buffers"] = {}

    def register_buffer(self, name: str, value: Any, persistent: bool = True) -> None:
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        self._buffers[name] = value

    def __getattr__(self, name: str) -> Any:
        buffers = self.__dict__.get("_buffers", {})
        if name in buffers:
            return buffers[name]
        raise AttributeError(name)


def _mixer_class() -> type:
    class MambaMixer2(_Module):
        def __init__(self, channels: int = 6, kernel: int = 4) -> None:
            super().__init__()
            self.conv1d = types.SimpleNamespace(weight=_Weight(channels, kernel))
            weight = self.conv1d.weight
            self.register_buffer(
                "conv_weights", weight.view(weight.size(0), weight.size(2)), persistent=False
            )

    return MambaMixer2


def _offload_and_load(mixer: Any) -> list[float]:
    """What UVAOffloader and then the weight loader do: new storage, weights into it."""
    weight = mixer.conv1d.weight
    weight.data = list(weight.data)  # p.data = view of pinned host memory
    weight.data[:] = [float(i + 1) for i in range(len(weight.data))]
    return weight.data


def test_a_stock_mixer_convolves_with_memory_nothing_was_loaded_into() -> None:
    """The bug, pinned so that a vLLM which fixes it shows up here."""
    mixer = _mixer_class()()
    loaded = _offload_and_load(mixer)
    assert mixer.conv_weights.storage is not loaded
    assert set(mixer.conv_weights.storage) == {0.0}


def test_a_patched_mixer_reads_the_weights_where_the_offloader_put_them() -> None:
    cls = _mixer_class()
    assert _load().install_live_conv_weights(cls) is True
    mixer = cls(channels=6, kernel=4)
    loaded = _offload_and_load(mixer)
    assert mixer.conv_weights.storage is loaded
    assert mixer.conv_weights.shape == (6, 4)
    assert "conv_weights" not in mixer._buffers


def test_without_offload_the_patched_mixer_reads_what_the_buffer_held() -> None:
    """No offload, no change: the same view of the same parameter."""
    stock, patched_cls = _mixer_class()(), _mixer_class()
    _load().install_live_conv_weights(patched_cls)
    patched = patched_cls()
    for mixer in (stock, patched):
        assert mixer.conv_weights.storage is mixer.conv1d.weight.data
        assert mixer.conv_weights.shape == (6, 4)


def test_install_patches_once_however_often_vllm_loads_it() -> None:
    plugin, cls = _load(), _mixer_class()
    assert plugin.install_live_conv_weights(cls) is True
    init = cls.__init__
    assert plugin.install_live_conv_weights(cls) is False
    assert cls.__init__ is init


def test_it_is_loud_when_vllm_no_longer_registers_the_buffer() -> None:
    class MambaMixer2(_Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1d = types.SimpleNamespace(weight=_Weight(6, 4))

    _load().install_live_conv_weights(MambaMixer2)
    with pytest.raises(RuntimeError, match="no longer registers"):
        MambaMixer2()


def test_it_is_loud_when_vllm_defines_conv_weights_itself() -> None:
    class MambaMixer2(_Module):
        @property
        def conv_weights(self) -> Any:
            return None

    with pytest.raises(RuntimeError, match="defines conv_weights itself"):
        _load().install_live_conv_weights(MambaMixer2)


def test_register_patches_vllms_mixer(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "vllm.model_executor.layers.mamba.mamba_mixer2"
    module = types.ModuleType(name)
    module.MambaMixer2 = _mixer_class()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    _load().register_mamba_live_conv()
    mixer = module.MambaMixer2()
    loaded = _offload_and_load(mixer)
    assert mixer.conv_weights.storage is loaded


def test_the_entry_point_is_the_recipes_plugin_name() -> None:
    project = tomllib.loads((PLUGIN_DIR / "pyproject.toml").read_text())["project"]
    entries = project["entry-points"]["vllm.general_plugins"]
    assert entries["raven_mamba_live_conv"] == "raven_vllm_plugin:register_mamba_live_conv"
    assert recipe.VLLM_PLUGINS["raven_mamba_live_conv"] == project["name"]


def test_the_offloading_mamba_row_loads_it_and_no_other_row_does() -> None:
    """Today one row offloads a Mamba model: the Nemotron control. Without the fix
    it measures uninitialised conv weights again; every other row keeps its engine."""
    recipes = recipe.load_recipes()
    assert "--cpu-offload-gb" in recipes[CONTROL]["args"]
    loading = {
        rid for rid, r in recipes.items() if "raven_mamba_live_conv" in r.get("vllm_plugins", [])
    }
    assert loading == {CONTROL}
