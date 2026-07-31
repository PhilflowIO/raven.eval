"""Tests for the flozi-strict WER core (raven_eval_core.flozi_wer).

This module is the single source of truth for published German ASR-WER numbers,
imported by both the Tier-2 runner and the Tier-1 re-scorer. The last test
pins that SSOT: scripts/verify.py must use *these* functions, not a copy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from raven_eval_core.flozi_wer import (
    corpus_cer_pct,
    corpus_wer_filler_tolerant_pct,
    corpus_wer_pct,
    evaluate,
    normalize_flozi,
)


class TestCorpusWer:
    def test_perfect_match_is_zero(self) -> None:
        assert corpus_wer_pct(["hallo welt"], ["hallo welt"]) == 0.0

    def test_one_substitution_in_two_words(self) -> None:
        assert corpus_wer_pct(["hallo welt"], ["hallo erde"]) == pytest.approx(
            50.0, rel=0.01
        )

    def test_empty_inputs_return_zero(self) -> None:
        assert corpus_wer_pct([], []) == 0.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            corpus_wer_pct(["a b"], ["a", "b"])

    def test_corpus_aggregation_not_mean_of_per_utterance(self) -> None:
        # Utterance A: 1 error / 1 word = 100%. Utterance B: 0 / 3 = 0%.
        # Mean-of-WERs would be 50%. Corpus WER = 1 edit / 4 ref words = 25%.
        w = corpus_wer_pct(["a", "x y z"], ["b", "x y z"])
        assert w == pytest.approx(25.0, abs=0.01)


class TestCer:
    def test_perfect_match_is_zero(self) -> None:
        assert corpus_cer_pct(["hallo"], ["hallo"]) == 0.0

    def test_one_char_off(self) -> None:
        assert corpus_cer_pct(["hallo"], ["hallx"]) == pytest.approx(20.0, rel=0.01)


class TestFloziNormalization:
    def test_umlauts_preserved_in_metric(self) -> None:
        # If umlauts were ASCIIfied, "Müller" vs "Muller" would tie at 0%.
        assert corpus_wer_pct(["Müller"], ["Muller"]) > 0.0

    def test_number_words_collapse_to_digits(self) -> None:
        assert corpus_wer_pct(["zwanzig Minuten"], ["20 Minuten"]) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_case_does_not_affect_wer(self) -> None:
        assert corpus_wer_pct(["Hallo Welt"], ["hallo welt"]) == 0.0

    def test_ss_folding_both_sides(self) -> None:
        assert corpus_wer_pct(["die strasse"], ["die straße"]) == 0.0

    def test_normalize_flozi_preserves_case_and_umlaut(self) -> None:
        assert normalize_flozi("Müller, Straße!") == "Müller Strasse"


class TestFillerTolerant:
    def test_filler_only_difference_collapses(self) -> None:
        strict = corpus_wer_pct(["ich meine das"], ["ich äh meine also das"])
        tolerant = corpus_wer_filler_tolerant_pct(
            ["ich meine das"], ["ich äh meine also das"]
        )
        assert strict > 0.0
        assert tolerant == pytest.approx(0.0, abs=1e-6)


class TestEvaluate:
    def test_returns_all_metrics_and_count(self) -> None:
        r = evaluate(["hallo welt", "guten tag"], ["hallo welt", "guten tag"])
        assert r.wer_pct == 0.0
        assert r.cer_pct == 0.0
        assert r.n_samples == 2
        assert r.wer_filler_tolerant_pct == 0.0


def _load_verify():
    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "verify", repo / "scripts" / "verify.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_verify_uses_the_core_not_a_copy() -> None:
    """SSOT guard: the Tier-1 re-scorer must import the core flozi functions,
    not re-implement them — otherwise the two paths can silently diverge."""
    verify = _load_verify()
    assert verify.corpus_wer_pct is corpus_wer_pct
    assert verify.corpus_cer_pct is corpus_cer_pct
    assert verify.normalize_text is normalize_flozi
