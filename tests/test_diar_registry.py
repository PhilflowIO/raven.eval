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

    def __init__(self, provider_id: str, model_id: str, revision: str | None) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.revision = revision

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
        diarizer = _make_diarizer("fake-diarizer", revision=None)
    finally:
        DIARIZER_ADAPTERS._overrides.pop("fake_adapter", None)

    assert isinstance(diarizer, _FakeDiarizer)
    # The spec's fields reach the adapter unchanged — that is the whole contract.
    assert diarizer.provider_id == "fake-diarizer"
    assert diarizer.model_id == "fake/model"
    assert diarizer.revision == "f" * 40
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
