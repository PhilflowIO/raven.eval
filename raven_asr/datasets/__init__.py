"""Dataset loaders yielding :class:`~raven_asr.datasets.base.Sample` records.

``WER_LOADERS`` is the registry the WER harness dispatches on. Its keys are the
dataset ids published in ``benchmark.config.yaml`` (``datasets.wer[].id``) — that
shared key space is what ``tests/test_dataset_contract.py`` enforces in both
directions, so the committed scoring contract cannot promise a dataset that has
no loader, nor hide a loader that the contract never names.

Loader classes are referenced as ``"module:ClassName"`` strings and imported on
demand: the registry must stay readable without pulling in ``numpy`` /
``soundfile`` / ``datasets``, so the contract test runs offline and without the
``asr`` extra.
"""

from __future__ import annotations

import importlib
from typing import Final

# dataset id (benchmark.config.yaml) -> "<module under this package>:<class>"
WER_LOADERS: Final[dict[str, str]] = {
    "german-mixed": "flozi_mixed_evals:FloziMixedEvalsLoader",
    "fleurs": "fleurs_de:FleursDeLoader",
    "mls-de": "mls_german:MlsGermanLoader",
    "voxpopuli-de": "voxpopuli_de:VoxPopuliDeLoader",
    # Swiss German dialect corpora — translation-shaped (Swiss German audio,
    # Standard German reference), acquired by URL + sha256 rather than through
    # a pinned HF revision. See raven_asr.config.DIALECT_DATASET_IDS.
    "spc-test": "spc_test:SpcTestLoader",
    "fhnw-all-dialects": "fhnw_all_dialects:FhnwAllDialectsLoader",
    # Dialect probe + its control spur. Two ids, one module on purpose: the
    # Bavarian number is only a dialect statement as a DELTA against the
    # Standard German recording of the same speaker on the same sentences, so
    # the two are registered and retired together, never one without the other.
    "xsid-bar": "xsid_audio:XsidBavarianLoader",
    "xsid-de-control": "xsid_audio:XsidGermanControlLoader",
}

# Modules in this package that are infrastructure, not a dataset loader. Kept
# here so the contract test can tell "unregistered loader" from "shared base".
NON_LOADER_MODULES: Final[frozenset[str]] = frozenset(
    {"__init__", "base", "hf_single_config", "local_archive"}
)


def loader_module_name(dataset_id: str) -> str:
    """Return the module name (without package prefix) backing ``dataset_id``."""
    try:
        target = WER_LOADERS[dataset_id]
    except KeyError:
        raise KeyError(
            f"no loader registered for dataset {dataset_id!r}; "
            f"registered: {sorted(WER_LOADERS)}"
        ) from None
    return target.split(":", 1)[0]


def load_loader_class(dataset_id: str) -> type:
    """Import and return the loader class registered for ``dataset_id``."""
    loader_module_name(dataset_id)  # raises a named KeyError on an unknown id
    module_name, class_name = WER_LOADERS[dataset_id].split(":", 1)
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, class_name)


__all__ = [
    "WER_LOADERS",
    "NON_LOADER_MODULES",
    "loader_module_name",
    "load_loader_class",
]
