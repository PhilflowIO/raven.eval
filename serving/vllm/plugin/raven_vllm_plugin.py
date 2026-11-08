"""vLLM plugin: hand each audio clip's own keyword tensors over as a batch of one.

vLLM 0.33.0 crashes its engine on any batch of audio clips of different lengths
when the model takes a per-clip tensor besides the audio, as VibeVoice-ASR does
with ``padding_mask``. Its Transformers backend deliberately encodes such clips
one at a time (``MultiModalMixin._process_audio_input``, so that no clip is padded
to match another), adding the batch dimension to the audio itself with
``unsqueeze(0)``. The other per-clip entries go through ``_select_item_kwargs``,
which slices them: a slice of a tensor is a batch of one, but these entries are
lists exactly when the clips differ in length, and a slice of a list is a list.
The model then calls ``padding_mask.sum(dim=-1)`` on it, and the AttributeError
kills the engine for every later request (raven.eval#93).

This plugin gives list entries the treatment the audio already gets. Encoding
stays per clip, so a clip's features do not depend on what it was batched with.

``register`` runs in every vLLM process and possibly more than once. It fails
loudly if vLLM no longer has the method: an engine bump must not silently drop
the fix. Remove the plugin once a pinned vLLM carries one.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("raven_vllm_plugin")

_MARKER = "_raven_ragged_audio"

_Select = Callable[[Any, dict[str, Any], int, int], dict[str, Any]]


def with_batch_of_one(original: _Select) -> _Select:
    """Wrap ``_select_item_kwargs`` so a narrowed list becomes a one-row tensor."""

    @functools.wraps(original)
    def select(
        self: Any, kwargs: dict[str, Any], index: int, num_items: int
    ) -> dict[str, Any]:
        selected = original(self, kwargs, index, num_items)
        return {
            key: value[0].unsqueeze(0)
            if isinstance(value, list)
            and value is not kwargs[key]
            and len(value) == 1
            and hasattr(value[0], "unsqueeze")
            else value
            for key, value in selected.items()
        }

    setattr(select, _MARKER, True)
    return select


def register() -> None:
    """vLLM's ``vllm.general_plugins`` entry point."""
    from vllm.model_executor.models.transformers.multimodal import MultiModalMixin

    original = MultiModalMixin._select_item_kwargs
    if getattr(original, _MARKER, False):
        return
    MultiModalMixin._select_item_kwargs = with_batch_of_one(original)
    logger.info("raven_ragged_audio: per-clip audio kwargs are batches of one")
