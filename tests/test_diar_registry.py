"""Registry seam + word→turn aggregation tests (no GPU, no network, no NeMo).

Two things are guarded here:

1. The **dispatch is data**, not an ``if``-chain: a diarizer can be added by
   dropping a module and a spec, and the runner resolves it without being edited.
   Exercised with a fake adapter registered at runtime — the real GPU adapters
   are unrunnable here by design.
2. The dispatch stays **lazy**: importing the runner must not import torch,
   pyannote or nemo, or the Tier-1 ``make verify`` path stops being GPU-free.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from raven_diar.adapters.aggregate import (
    DEFAULT_GAP_MERGE_S,
    LabelledSpan,
    spans_to_turns,
)
from raven_diar.adapters.base import DiarizeResult
from raven_diar.config import KNOWN_DIARIZERS
from raven_diar.registry import DATASET_LOADERS, DIARIZER_ADAPTERS, LazyRegistry
from raven_diar.reproduce import _make_diarizer, _make_loader

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── registry mechanics ───────────────────────────────────────────────────────


def test_every_known_diarizer_resolves_to_a_callable_adapter():
    """The registry, not a dispatcher, is what makes a spec runnable."""
    for key, spec in KNOWN_DIARIZERS.items():
        factory = DIARIZER_ADAPTERS.resolve(spec.adapter)
        assert callable(factory), f"{key}: {spec.adapter} is not callable"


def test_registry_lists_the_shipped_adapters_and_hides_plumbing():
    names = DIARIZER_ADAPTERS.available()
    assert "pyannote_community1" in names
    assert "sortformer" in names
    assert "deepgram" in names
    # base.py (the protocol) and aggregate.py (shared folding) are not adapters.
    assert "base" not in names and "aggregate" not in names


def test_every_dataset_loader_resolves():
    from raven_diar.config import DER_DATASETS

    for ds in DER_DATASETS.values():
        assert callable(DATASET_LOADERS.resolve(ds.loader))
    assert "base" not in DATASET_LOADERS.available()


def test_unknown_key_names_the_alternatives():
    with pytest.raises(ValueError, match="unknown diarizer adapter"):
        DIARIZER_ADAPTERS.resolve("does_not_exist")


def test_module_without_the_entry_attribute_is_rejected():
    reg = LazyRegistry("raven_diar.adapters", attr="ADAPTER", kind="diarizer adapter")
    reg.register("plumbing", "raven_diar.adapters.base:ADAPTER")
    with pytest.raises(ValueError, match="module-level 'ADAPTER'"):
        reg.resolve("plumbing")


def test_override_must_be_an_import_path():
    reg = LazyRegistry("raven_diar.adapters", attr="ADAPTER", kind="diarizer adapter")
    with pytest.raises(ValueError, match="module.path:ATTR"):
        reg.register("x", "raven_diar.adapters.base")


# ── the seam: a new diarizer needs no dispatch change ────────────────────────


class _FakeDiarizer:
    """A diarizer added the way a real one is: a factory + a spec entry."""

    def __init__(
        self,
        provider_id: str,
        model_id: str,
        revision: str | None,
        language: str | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.revision = revision
        self.language = language

    def diarize(self, audio_path: Path) -> DiarizeResult:
        return DiarizeResult(segments=[(0.0, 1.0, "spk_0")], latency_s=0.0)


def test_a_new_adapter_is_dispatched_without_touching_reproduce(monkeypatch):
    import raven_diar.adapters.base as base_mod
    from raven_diar.config import DiarizerSpec

    monkeypatch.setattr(base_mod, "ADAPTER", _FakeDiarizer, raising=False)
    monkeypatch.setitem(
        KNOWN_DIARIZERS,
        "fake-diarizer",
        DiarizerSpec(
            model_id="fake/model",
            adapter="fake_adapter",
            label="fake-diarizer",
            revision="f" * 40,
        ),
    )
    DIARIZER_ADAPTERS.register("fake_adapter", "raven_diar.adapters.base:ADAPTER")
    try:
        diarizer = _make_diarizer("fake-diarizer", revision=None, language="en")
    finally:
        DIARIZER_ADAPTERS._overrides.pop("fake_adapter", None)

    assert isinstance(diarizer, _FakeDiarizer)
    # The spec's fields reach the adapter unchanged — that is the whole contract.
    assert diarizer.provider_id == "fake-diarizer"
    assert diarizer.model_id == "fake/model"
    assert diarizer.revision == "f" * 40
    assert diarizer.language == "en"
    assert diarizer.diarize(Path("/dev/null")).segments == [(0.0, 1.0, "spk_0")]


def test_make_loader_passes_the_split_through(monkeypatch):
    seen: dict[str, object] = {}

    class _FakeLoader:
        def __init__(self, split: str | None = None) -> None:
            seen["split"] = split

    monkeypatch.setattr(DATASET_LOADERS, "resolve", lambda name: _FakeLoader)
    _make_loader("whatever", split="test")
    assert seen == {"split": "test"}


def test_importing_the_runner_pulls_no_heavy_backend():
    """Tier-1 verify must stay torch/pyannote.audio/nemo-free; a subprocess proves it.

    ``pyannote.metrics`` is deliberately NOT in this list: it IS the light Tier-1
    scorer (raven_eval_core.der) and is always imported. What must never appear is
    a *backend*: torch, pyannote.audio, nemo.
    """
    probe = (
        "import sys, raven_diar.reproduce, raven_diar.registry;"
        "heavy=[m for m in ('torch','pyannote.audio','nemo') if m in sys.modules];"
        "print(heavy)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_listing_adapters_imports_no_adapter_module():
    probe = (
        "import sys; from raven_diar.registry import DIARIZER_ADAPTERS;"
        "names=DIARIZER_ADAPTERS.available();"
        "assert 'sortformer' in names, names;"
        "print('raven_diar.adapters.sortformer' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False", out.stdout


# ── sortformer adapter (segment parsing only — the model needs a GPU) ────────


def test_sortformer_is_registered_with_a_pinned_public_model():
    spec = KNOWN_DIARIZERS["sortformer-4spk-v1"]
    assert spec.model_id == "nvidia/diar_sortformer_4spk-v1"
    assert spec.adapter == "sortformer"


def test_streaming_sortformer_v2_is_registered_shippable_and_pinned():
    """The shippable Sortformer: CC-BY-4.0 weights, so its row may compete."""
    spec = KNOWN_DIARIZERS["sortformer-streaming-4spk-v2"]
    assert spec.model_id == "nvidia/diar_streaming_sortformer_4spk-v2"
    assert spec.adapter == "sortformer"  # one adapter, two checkpoints
    assert spec.license == "CC-BY-4.0"
    assert spec.shippable is True
    assert spec.hosted is False
    assert spec.revision_kind == "commit"


def test_sortformer_checkpoint_filename_follows_the_repo_name():
    from raven_diar.adapters.sortformer import _checkpoint_filename

    assert (
        _checkpoint_filename("nvidia/diar_sortformer_4spk-v1")
        == "diar_sortformer_4spk-v1.nemo"
    )
    assert (
        _checkpoint_filename("nvidia/diar_streaming_sortformer_4spk-v2")
        == "diar_streaming_sortformer_4spk-v2.nemo"
    )


def test_only_the_streaming_checkpoint_gets_a_streaming_config():
    """v1's published rows must not shift because v2 was added.

    The streaming knobs are keyed by model id, so building the offline adapter
    resolves to no config at all and `_apply_streaming_config` is a no-op.
    """
    from raven_diar.adapters.sortformer import SortformerDiarizer

    offline = SortformerDiarizer(
        model_id="nvidia/diar_sortformer_4spk-v1", revision="a" * 40
    )
    assert offline._streaming_config == {}

    streaming = SortformerDiarizer(
        model_id="nvidia/diar_streaming_sortformer_4spk-v2", revision="b" * 40
    )
    # The model card's "very high latency" row, verbatim, in 80 ms frames.
    assert streaming._streaming_config == {
        "chunk_len": 340,
        "chunk_right_context": 40,
        "fifo_len": 40,
        "spkcache_update_period": 300,
        "spkcache_len": 188,
    }


def test_latency_presets_match_the_model_card_table():
    """The four presets are a transcription of NVIDIA's table, not a guess.

    Values verbatim from the ``diar_streaming_sortformer_4spk-v2`` model card,
    section "Setting up Streaming Configuration" (fetched 2026-09-04), in 80 ms
    frames. A preset that drifts from this table silently changes what a
    diagnostic run means, so the table is pinned here.
    """
    from raven_diar.adapters.sortformer import LATENCY_PRESETS

    assert LATENCY_PRESETS == {
        "very-high-latency": {
            "chunk_len": 340, "chunk_right_context": 40, "fifo_len": 40,
            "spkcache_update_period": 300, "spkcache_len": 188,
        },
        "high-latency": {
            "chunk_len": 124, "chunk_right_context": 1, "fifo_len": 124,
            "spkcache_update_period": 124, "spkcache_len": 188,
        },
        "low-latency": {
            "chunk_len": 6, "chunk_right_context": 7, "fifo_len": 188,
            "spkcache_update_period": 144, "spkcache_len": 188,
        },
        "ultra-low-latency": {
            "chunk_len": 3, "chunk_right_context": 1, "fifo_len": 188,
            "spkcache_update_period": 144, "spkcache_len": 188,
        },
    }
    # Input-buffer latency is chunk + right context; the ordering the names
    # claim has to be the ordering the numbers produce.
    latency = {
        name: cfg["chunk_len"] + cfg["chunk_right_context"]
        for name, cfg in LATENCY_PRESETS.items()
    }
    assert (latency["very-high-latency"] > latency["high-latency"]
            > latency["low-latency"] > latency["ultra-low-latency"])


def test_the_published_preset_is_the_default_and_is_named_in_run_config():
    """Every committed v2 row was measured at "very high latency"."""
    from raven_diar.adapters.sortformer import SortformerDiarizer

    streaming = SortformerDiarizer(
        model_id="nvidia/diar_streaming_sortformer_4spk-v2", revision="b" * 40
    )
    assert streaming.latency_preset == "very-high-latency"
    assert streaming.run_config["latency_preset"] == "very-high-latency"
    assert streaming.run_config["streaming_config"]["chunk_len"] == 340

    offline = SortformerDiarizer(
        model_id="nvidia/diar_sortformer_4spk-v1", revision="a" * 40
    )
    assert offline.latency_preset is None
    assert offline.run_config["streaming_config"] is None


def test_env_selects_a_preset_and_the_choice_travels_with_the_run(monkeypatch):
    """A diagnostic run must be self-describing, or it can be promoted by mistake."""
    from raven_diar.adapters.sortformer import PRESET_ENV_VAR, SortformerDiarizer

    monkeypatch.setenv(PRESET_ENV_VAR, "low-latency")
    diarizer = SortformerDiarizer(
        model_id="nvidia/diar_streaming_sortformer_4spk-v2", revision="b" * 40
    )
    assert diarizer.latency_preset == "low-latency"
    assert diarizer._streaming_config["chunk_len"] == 6
    assert diarizer.run_config == {
        "latency_preset": "low-latency",
        "streaming_config": {
            "chunk_len": 6, "chunk_right_context": 7, "fifo_len": 188,
            "spkcache_update_period": 144, "spkcache_len": 188,
        },
    }
    # An explicit argument still wins over the environment.
    assert SortformerDiarizer(
        model_id="nvidia/diar_streaming_sortformer_4spk-v2",
        revision="b" * 40,
        latency_preset="high-latency",
    ).latency_preset == "high-latency"


def test_a_bad_preset_fails_at_construction_not_as_a_different_der():
    from raven_diar.adapters.sortformer import SortformerDiarizer

    with pytest.raises(ValueError, match="unknown latency preset"):
        SortformerDiarizer(
            model_id="nvidia/diar_streaming_sortformer_4spk-v2",
            revision="b" * 40,
            latency_preset="medium-latency",
        )
    # v1 has no streaming mode at all — asking for a preset is a mistake, and a
    # silent no-op would publish a number under a configuration it never had.
    with pytest.raises(ValueError, match="not a streaming checkpoint"):
        SortformerDiarizer(
            model_id="nvidia/diar_sortformer_4spk-v1",
            revision="a" * 40,
            latency_preset="low-latency",
        )


def test_streaming_config_is_pushed_into_nemo_and_validated_by_nemo():
    """Prove the knobs land on the model and that NeMo's own check is called."""
    from raven_diar.adapters.sortformer import SortformerDiarizer

    class _Modules:
        chunk_len = 0
        chunk_right_context = 0
        fifo_len = 0
        spkcache_update_period = 0
        spkcache_len = 0

        def __init__(self) -> None:
            self.checked = False

        def _check_streaming_parameters(self) -> None:
            self.checked = True

    class _Model:
        def __init__(self) -> None:
            self.sortformer_modules = _Modules()

    model = _Model()
    SortformerDiarizer(
        model_id="nvidia/diar_streaming_sortformer_4spk-v2", revision="b" * 40
    )._apply_streaming_config(model)
    assert model.sortformer_modules.chunk_len == 340
    assert model.sortformer_modules.spkcache_len == 188
    assert model.sortformer_modules.checked, (
        "NeMo's own parameter check must run — it is what turns a typo'd "
        "streaming config into a loud failure instead of a different number"
    )

    # An unknown knob is a NeMo-version mismatch, and must not pass silently.
    class _OldModel:
        class sortformer_modules:  # mirrors NeMo's attribute name
            chunk_len = 0

    with pytest.raises(ValueError, match="streaming config"):
        SortformerDiarizer(
            model_id="nvidia/diar_streaming_sortformer_4spk-v2", revision="b" * 40
        )._apply_streaming_config(_OldModel())


def test_sortformer_parses_both_nemo_output_shapes():
    from raven_diar.adapters.sortformer import _parse_segment

    assert _parse_segment("0.03 1.28 speaker_0") == (0.03, 1.28, "speaker_0")
    assert _parse_segment((2.0, 3.5, "speaker_1")) == (2.0, 3.5, "speaker_1")
    assert _parse_segment("4.0 4.0 speaker_2") is None  # zero-length → dropped
    with pytest.raises(ValueError):
        _parse_segment("0.03 1.28")
    with pytest.raises(ValueError):
        _parse_segment(object())


def test_sortformer_result_is_the_shape_the_scorer_already_accepts(monkeypatch):
    """No change to DiarizeResult / score.py: prove the adapter feeds them as-is."""
    from raven_diar.adapters.sortformer import SortformerDiarizer
    from raven_diar.score import score_segment_pairs
    from raven_eval_core.der import to_rttm

    class _FakeNemoModel:
        def diarize(self, audio, batch_size):  # noqa: ARG002 - NeMo's signature
            return [["5.0 6.0 speaker_1", "0.0 4.0 speaker_0", "6.0 6.0 speaker_0"]]

    diarizer = SortformerDiarizer(revision="a" * 40)
    monkeypatch.setattr(diarizer, "_ensure_model", lambda: _FakeNemoModel())
    result = diarizer.diarize(Path("/dev/null"))

    assert result.segments == [(0.0, 4.0, "speaker_0"), (5.0, 6.0, "speaker_1")]
    assert result.raw["revision"] == "a" * 40
    # Serialises to RTTM and scores through the untouched scorer.
    assert to_rttm(result.segments, file_id="f1").count("SPEAKER") == 2
    gold = [(0.0, 4.0, "A"), (5.0, 6.0, "B")]
    assert score_segment_pairs("demo", [(gold, result.segments)]).der_full == 0.0


# ── shared word/utterance → turn aggregation ─────────────────────────────────


def test_consecutive_same_speaker_words_merge_into_one_turn():
    words = [
        LabelledSpan(0.00, 0.30, "A"),
        LabelledSpan(0.35, 0.60, "A"),
        LabelledSpan(0.70, 1.00, "A"),  # all gaps well under the threshold
    ]
    assert spans_to_turns(words) == [(0.0, 1.0, "A")]


def test_speaker_change_starts_a_new_turn_even_with_no_gap():
    words = [(0.0, 1.0, "A"), (1.0, 2.0, "B")]
    assert spans_to_turns(words) == [(0.0, 1.0, "A"), (1.0, 2.0, "B")]


def test_gap_larger_than_the_threshold_splits_one_speakers_turn():
    words = [(0.0, 1.0, "A"), (1.0 + DEFAULT_GAP_MERGE_S + 0.01, 2.0, "A")]
    turns = spans_to_turns(words)
    assert len(turns) == 2 and turns[0] == (0.0, 1.0, "A")
    # …and the same input with a wider threshold is ONE turn: the parameter is
    # what decides, not the caller's own folding.
    assert spans_to_turns(words, gap_merge_s=2.0) == [(0.0, 2.0, "A")]


def test_gap_exactly_at_the_threshold_merges():
    words = [(0.0, 1.0, "A"), (1.0 + DEFAULT_GAP_MERGE_S, 2.0, "A")]
    assert spans_to_turns(words) == [(0.0, 2.0, "A")]


def test_interleaved_and_overlapping_spans_merge_per_speaker():
    # B interjects inside A's stretch, and two A words overlap slightly. A must
    # stay one turn: merging on the globally sorted stream would chop it in two.
    words = [
        (0.0, 0.5, "A"),
        (0.4, 0.9, "B"),   # overlaps A
        (0.45, 1.2, "A"),  # overlaps the previous A word
        (1.3, 1.6, "A"),
        (3.0, 3.5, "B"),   # far from B's first span → separate B turn
    ]
    assert spans_to_turns(words) == [
        (0.0, 1.6, "A"),
        (0.4, 0.9, "B"),
        (3.0, 3.5, "B"),
    ]


def test_zero_length_and_inverted_spans_are_dropped():
    assert spans_to_turns([(1.0, 1.0, "A"), (3.0, 2.0, "A"), (5.0, 6.0, "A")]) == [
        (5.0, 6.0, "A")
    ]
    assert spans_to_turns([]) == []


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError, match="gap_merge_s"):
        spans_to_turns([(0.0, 1.0, "A")], gap_merge_s=-0.1)


def test_aggregated_turns_score_through_the_untouched_scorer():
    """The helper's output is a DiarizeResult segment list, nothing new."""
    from raven_diar.score import score_segment_pairs

    hyp = spans_to_turns([(0.0, 0.4, "A"), (0.5, 1.0, "A"), (2.0, 3.0, "B")])
    gold = [(0.0, 1.0, "A"), (2.0, 3.0, "B")]
    assert score_segment_pairs("demo", [(gold, hyp)]).der_full == 0.0


# ── deepgram adapter (the first HOSTED diarizer; no key, no network here) ─────


def test_deepgram_is_registered_with_both_halves_pinned():
    """A hosted row pins TWO models: the ASR one and the diarizer version."""
    spec = KNOWN_DIARIZERS["deepgram-nova-3"]
    assert spec.adapter == "deepgram"
    # Not the family alias "nova-3": the vendor may repoint an alias silently.
    assert spec.model_id == "nova-3-general"
    # `revision` carries the `diarize_model` version, never the floating "latest".
    assert spec.revision == "v2"
    assert spec.revision != "latest"


def _deepgram_body(words, *, diarized=True):
    metadata = {"request_id": "req-1", "model_info": {}}
    if diarized:
        metadata["diarize_info"] = {"model_uuid": "u-1", "arch": "v2"}
    return {
        "metadata": metadata,
        "results": {"channels": [{"alternatives": [{"words": words}]}]},
    }


def test_deepgram_words_become_labelled_spans():
    from raven_diar.adapters.deepgram import words_to_spans

    body = _deepgram_body([
        {"word": "hallo", "start": 0.1, "end": 0.4, "speaker": 0},
        {"word": "ja", "start": 0.5, "end": 0.7, "speaker": 1},
    ])
    assert words_to_spans(body) == [
        LabelledSpan(0.1, 0.4, "speaker_0"),
        LabelledSpan(0.5, 0.7, "speaker_1"),
    ]


def test_deepgram_words_without_a_speaker_are_dropped_not_invented():
    from raven_diar.adapters.deepgram import words_to_spans

    body = _deepgram_body([
        {"word": "hallo", "start": 0.1, "end": 0.4, "speaker": 0},
        {"word": "hm", "start": 0.5, "end": 0.7},          # no speaker label
        {"word": "ja", "start": 0.8, "end": 0.9, "speaker": 0},
    ])
    assert [s.speaker for s in words_to_spans(body)] == ["speaker_0", "speaker_0"]


def test_deepgram_refuses_a_response_where_the_diarizer_did_not_run():
    """Words but no speaker anywhere = no diarizer ran; never score that as 1 speaker."""
    from raven_diar.adapters.deepgram import words_to_spans

    body = _deepgram_body([
        {"word": "hallo", "start": 0.0, "end": 0.4},
        {"word": "ja", "start": 0.5, "end": 0.7},
    ], diarized=False)
    with pytest.raises(ValueError, match="no speaker label"):
        words_to_spans(body)


def test_deepgram_does_not_gate_on_the_vendor_metadata_block():
    """The pin is evidenced by diarize_info, but the run is not hostage to it."""
    from raven_diar.adapters.deepgram import words_to_spans

    body = _deepgram_body(
        [{"word": "hallo", "start": 0.0, "end": 0.4, "speaker": 0}], diarized=False
    )
    assert words_to_spans(body) == [LabelledSpan(0.0, 0.4, "speaker_0")]


def test_deepgram_requires_its_key_and_names_the_env_var(monkeypatch):
    from raven_diar.adapters.deepgram import DeepgramDiarizer

    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        DeepgramDiarizer(language="de")


def test_deepgram_folds_through_the_shared_aggregator_not_its_own(monkeypatch):
    """The anti-fork guard: a second folding path would confound provider DERs.

    Asserted by observation, not by reading: the module-level ``spans_to_turns``
    is replaced with a sentinel, and the adapter's output must be the sentinel's.
    """
    import raven_diar.adapters.deepgram as dg

    monkeypatch.setenv("DEEPGRAM_API_KEY", "not-a-real-key")
    calls: dict[str, object] = {}

    def _sentinel(spans, *, gap_merge_s):
        calls["spans"] = list(spans)
        calls["gap_merge_s"] = gap_merge_s
        return [(0.0, 9.0, "SENTINEL")]

    monkeypatch.setattr(dg, "spans_to_turns", _sentinel)
    diarizer = dg.DeepgramDiarizer(revision="v2", language="de")
    monkeypatch.setattr(
        diarizer,
        "_alisten",
        lambda audio_bytes: _async_value(  # noqa: ARG005 - signature parity
            _deepgram_body([
                {"word": "a", "start": 0.0, "end": 0.2, "speaker": 0},
                {"word": "b", "start": 0.25, "end": 0.5, "speaker": 0},
            ])
        ),
    )
    result = diarizer.diarize(Path("/dev/null"))

    assert result.segments == [(0.0, 9.0, "SENTINEL")]
    assert calls["gap_merge_s"] == DEFAULT_GAP_MERGE_S
    assert len(calls["spans"]) == 2
    assert result.raw["revision"] == "v2"
    assert result.raw["diarize_info"] == {"model_uuid": "u-1", "arch": "v2"}


async def _async_value(value):
    return value


def test_deepgram_gap_threshold_is_overridable_only_explicitly(monkeypatch):
    """The sweep knob exists, but the DEFAULT is the shared constant."""
    import raven_diar.adapters.deepgram as dg

    monkeypatch.setenv("DEEPGRAM_API_KEY", "not-a-real-key")
    monkeypatch.delenv(dg.GAP_MERGE_ENV, raising=False)
    assert dg.DeepgramDiarizer(language="de")._gap_merge_s == DEFAULT_GAP_MERGE_S
    monkeypatch.setenv(dg.GAP_MERGE_ENV, "1.0")
    assert dg.DeepgramDiarizer(language="de")._gap_merge_s == 1.0
    assert dg.DeepgramDiarizer(gap_merge_s=0.25, language="de")._gap_merge_s == 0.25


# ── publication eligibility is data, not a comment ───────────────────────────


def test_every_diarizer_states_its_licence() -> None:
    """"unknown" is the default, and it must never survive into a spec.

    Whether a measured row may carry a winner mark hangs entirely on this one
    fact. Leaving it at the default would publish a ranking whose rules nobody
    recorded.
    """
    from raven_diar.config import KNOWN_DIARIZERS

    unstated = sorted(
        key for key, spec in KNOWN_DIARIZERS.items() if spec.license == "unknown"
    )
    assert not unstated, (
        f"diarizer(s) {unstated} carry no licence — under ADR-app-0036 the "
        f"licence decides whether a row may win, so it cannot be left implicit."
    )


def test_non_commercial_weights_are_never_shippable() -> None:
    """A model we could not ship must not be able to win a comparison.

    Sortformer is the live case: it currently beats the shipped default on
    German telephone speech, which is exactly when the temptation to award it
    is strongest and exactly when doing so would mislead.
    """
    from raven_diar.config import KNOWN_DIARIZERS

    for key, spec in KNOWN_DIARIZERS.items():
        if "-NC-" in spec.license or "NonCommercial" in spec.license:
            assert not spec.shippable, (
                f"{key}: non-commercial weights ({spec.license}) — a reference "
                f"row, never an award. Measured and shown, never prized."
            )


# ── the corpus decides the language (the AMI 99.99 % lesson) ─────────────────


def test_every_dataset_declares_a_language():
    """A corpus without a language is a hosted run that silently scores ~100 %.

    The Deepgram adapter defaulted to German. On the English AMI smoke it came
    back with zero turns for every file and the harness dutifully reported
    99.99 % DER — a number that looks like a terrible diarizer and is in fact a
    wrong ASR request. The language belongs to the corpus, so the corpus must
    state it.
    """
    from raven_diar.config import DER_DATASETS

    for key, ds in DER_DATASETS.items():
        assert ds.language, f"{key} declares no language"
        assert ds.language in {"de", "en"}, f"{key}: unknown language {ds.language!r}"


def test_the_dataset_language_reaches_the_adapter(monkeypatch):
    """The runner passes the CORPUS language, never the adapter's idea of one."""
    import raven_diar.adapters.base as base_mod
    from raven_diar.config import DiarizerSpec

    monkeypatch.setattr(base_mod, "ADAPTER", _FakeDiarizer, raising=False)
    monkeypatch.setitem(
        KNOWN_DIARIZERS,
        "fake-hosted",
        DiarizerSpec(
            model_id="fake/hosted",
            adapter="fake_adapter",
            label="fake-hosted",
            revision="vendor-alias",
            hosted=True,
            revision_kind="vendor-alias",
        ),
    )
    DIARIZER_ADAPTERS.register("fake_adapter", "raven_diar.adapters.base:ADAPTER")
    try:
        assert _make_diarizer("fake-hosted", revision=None, language="de").language == "de"
        assert _make_diarizer("fake-hosted", revision=None, language="en").language == "en"
    finally:
        DIARIZER_ADAPTERS._overrides.pop("fake_adapter", None)


def test_hosted_adapters_refuse_to_be_built_without_a_language():
    """No default: leaving it out must be a TypeError, not a wrong number."""
    import inspect

    from raven_diar.adapters import assemblyai, deepgram

    for module in (deepgram, assemblyai):
        param = inspect.signature(module.ADAPTER.__init__).parameters["language"]
        assert param.default is inspect.Parameter.empty, (
            f"{module.__name__}: language has a default again — that is the bug"
        )
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


# ── diarizen adapter (the pipeline needs a GPU; the pinning does not) ─────────


def test_diarizen_is_registered_non_shippable_and_pinned():
    """The ETH benchmark's best open-source candidate, measured on our terms.

    CC-BY-NC-4.0, so it is a reference row: shown, never awarded. That pairing is
    also asserted generically by `test_non_commercial_weights_are_never_shippable`
    — this test pins the specific facts that make the row meaningful.
    """
    spec = KNOWN_DIARIZERS["diarizen-wavlm-large-s80-md-v2"]
    assert spec.model_id == "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    assert spec.adapter == "diarizen"
    assert spec.license == "CC-BY-NC-4.0"
    assert spec.shippable is False
    assert not spec.hosted and spec.revision_kind == "commit"
    assert "diarizen" in DIARIZER_ADAPTERS.available()


def test_every_weights_revision_an_adapter_pins_itself_is_a_commit():
    """The registry is not the only place a published number hangs on weights.

    DiariZen clusters with a SECOND model from a different HF repo, which its own
    library downloads without a revision. Pinning it in the adapter is correct —
    it is a property of the library, not a choice per row — but a pin outside
    `KNOWN_DIARIZERS` is a pin the spec sweep never sees. So the invariant is
    extended rather than duplicated: every `*_REVISION` an adapter module
    declares must be a full commit hash, for every adapter, forever.
    """
    import importlib
    import re

    checked = 0
    for name in DIARIZER_ADAPTERS.available():
        module = importlib.import_module(f"raven_diar.adapters.{name}")
        for attr in dir(module):
            if not attr.endswith("_REVISION"):
                continue
            value = getattr(module, attr)
            if not isinstance(value, str):
                continue
            checked += 1
            assert re.fullmatch(r"[0-9a-f]{40}", value), (
                f"raven_diar.adapters.{name}.{attr} = {value!r} is not a full "
                f"commit hash — a published number would follow a branch"
            )
    assert checked, "the sweep found no adapter-level revision pin to check"


def test_diarizen_pins_both_weight_sets_it_downloads(monkeypatch):
    """`DiariZenPipeline.from_pretrained` passes no revision to either download.

    Using it would leave a published DER hanging on two HuggingFace branches at
    once — the checkpoint and the speaker-embedding model — so the adapter does
    both downloads itself. This drives the real code path with the library
    faked out, so the guarantee survives a refactor of the adapter.
    """
    import sys
    import types

    from raven_diar.adapters import diarizen as dz

    calls: dict[str, dict] = {}

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kw: calls.setdefault("snapshot", kw) and "/hub"
    hub.hf_hub_download = lambda **kw: calls.setdefault("file", kw) and "/emb.bin"
    # `and` short-circuits on the falsy dict return, so return the paths plainly:
    hub.snapshot_download = lambda **kw: (calls.__setitem__("snapshot", kw), "/hub")[1]
    hub.hf_hub_download = lambda **kw: (calls.__setitem__("file", kw), "/emb.bin")[1]

    inference = types.ModuleType("diarizen.pipelines.inference")
    inference.DiariZenPipeline = lambda **kw: calls.setdefault("pipeline", kw)
    pipelines = types.ModuleType("diarizen.pipelines")
    package = types.ModuleType("diarizen")

    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "diarizen", package)
    monkeypatch.setitem(sys.modules, "diarizen.pipelines", pipelines)
    monkeypatch.setitem(sys.modules, "diarizen.pipelines.inference", inference)

    dz.ADAPTER(revision="c" * 40)._ensure_pipeline()

    assert calls["snapshot"]["repo_id"] == "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    assert calls["snapshot"]["revision"] == "c" * 40
    assert calls["file"]["repo_id"] == "pyannote/wespeaker-voxceleb-resnet34-LM"
    assert calls["file"]["revision"] == dz.EMBEDDING_REVISION
    # The harness owns the RTTM layout; the library must not write its own.
    assert calls["pipeline"]["rttm_out_dir"] is None


def test_diarizen_names_the_embedding_pin_in_the_run_config():
    """A summary naming only the checkpoint would under-describe the run."""
    from raven_diar.adapters import diarizen as dz

    assert dz.ADAPTER(revision="c" * 40).run_config == {
        "embedding_model_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "embedding_revision": dz.EMBEDDING_REVISION,
    }


def test_diarizen_turns_a_pyannote_annotation_into_scoreable_segments():
    """The pipeline returns an Annotation; the harness wants sorted triples."""
    from raven_diar.adapters.diarizen import DiariZenDiarizer

    class _Turn:
        def __init__(self, start, end):
            self.start, self.end = start, end

    class _Annotation:
        def itertracks(self, yield_label=False):  # noqa: ARG002 - signature parity
            # Deliberately unsorted, with one zero-length turn: pyannote does not
            # promise an order, and a zero-length turn is not scoreable.
            yield _Turn(10.0, 20.0), "_", "B"
            yield _Turn(0.0, 5.0), "_", "A"
            yield _Turn(7.0, 7.0), "_", "A"

    diarizer = DiariZenDiarizer(revision="c" * 40)
    diarizer._pipeline = lambda path: _Annotation()  # skip the GPU pipeline
    result = diarizer.diarize(Path("unused.wav"))
    assert result.segments == [(0.0, 5.0, "A"), (10.0, 20.0, "B")]
    assert result.raw["n_speakers"] == 2
    assert result.raw["revision"] == "c" * 40
