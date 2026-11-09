"""vLLM plugins for audio models that the Transformers backend serves wrongly.

``raven_ragged_audio``: hand each audio clip's own keyword tensors over as a batch
of one.

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

``raven_audio_duration``: tell the model how long the clip is.

VibeVoice-ASR's chat template writes the instruction "This is a
<|AUDIO_DURATION|> seconds audio, ..." and its HF processor fills that placeholder
with the clip's length when it tokenizes the text together with the audio. vLLM
tokenizes the rendered template on its own and calls the processor with dummy
text only (``BaseMultiModalProcessor._apply_hf_processor_main``), so the fill
lands on the dummy and the model is sent the literal placeholder. Not told how
long the clip is, the model carried on past its end, inventing ``[Noise]`` and
``[Silence]`` turns up to the output cap (raven.eval#96). This plugin does what
the processor does, on vLLM's prompt: the placeholder becomes the duration, with
the processor's own formatting, before the audio is expanded.

``register`` runs in every vLLM process and possibly more than once. It fails
loudly if vLLM no longer has the method: an engine bump must not silently drop
the fix. Remove the plugin once a pinned vLLM carries one.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger("raven_vllm_plugin")

_MARKER = "_raven_ragged_audio"

_DURATION_MARKER = "_raven_audio_duration"

_Select = Callable[[Any, dict[str, Any], int, int], dict[str, Any]]
_Apply = Callable[[Any, Any, Any], Any]


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


def fill_durations(text: str, placeholder: str, durations: Sequence[float]) -> str:
    """``text`` with each ``placeholder`` replaced by its clip's duration in seconds.

    The format is the one ``VibeVoiceAsrProcessor.__call__`` writes. Placeholders
    and clips pair up in order; any other number of them is a request the model
    would answer about the wrong clip, so it is refused rather than guessed at.
    """
    if text.count(placeholder) != len(durations):
        raise ValueError(
            f"{text.count(placeholder)} placeholders {placeholder!r} "
            f"for {len(durations)} audio clips"
        )
    seconds = iter(durations)
    return re.sub(re.escape(placeholder), lambda _: f"{next(seconds):.2f}", text)


def with_audio_duration(original: _Apply) -> _Apply:
    """Wrap ``MultiModalProcessor.apply`` so the prompt states each clip's duration."""

    @functools.wraps(original)
    def apply(self: Any, inputs: Any, timing_ctx: Any) -> Any:
        hf_processor = self.info.get_hf_processor(**inputs.hf_processor_mm_kwargs)
        placeholder = getattr(hf_processor, "audio_duration_token", None)
        if placeholder is None:
            return original(self, inputs, timing_ctx)
        tokenizer = self.info.get_tokenizer()
        text = tokenizer.decode(
            inputs.prompt, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if placeholder not in text:
            return original(self, inputs, timing_ctx)
        audios = inputs.mm_data_items.get("audio")
        sampling_rate = hf_processor.feature_extractor.sampling_rate
        count = 0 if audios is None else audios.get_count()
        clips = [audios.get(i) for i in range(count)] if audios is not None else []
        if any(clip is None for clip in clips):
            raise ValueError("the audio duration of a cached-only clip is not known")
        filled = fill_durations(text, placeholder, [len(c) / sampling_rate for c in clips])
        return original(
            self,
            dataclasses.replace(
                inputs, prompt=tokenizer.encode(filled, add_special_tokens=False)
            ),
            timing_ctx,
        )

    setattr(apply, _DURATION_MARKER, True)
    return apply


def register_audio_duration() -> None:
    """vLLM's ``vllm.general_plugins`` entry point for ``raven_audio_duration``."""
    from vllm.model_executor.models.transformers.multimodal import MultiModalProcessor

    original = MultiModalProcessor.apply
    if getattr(original, _DURATION_MARKER, False):
        return
    MultiModalProcessor.apply = with_audio_duration(original)
    logger.info("raven_audio_duration: the prompt states each clip's duration")
