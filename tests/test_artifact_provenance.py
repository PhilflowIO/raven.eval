"""Every committed artifact names what it was measured on and under which settings.

`make verify` proves a committed number re-scores from its committed data. What
it cannot prove is where that data came from or how it was produced — which
references, which model, which operating point. These guards make every artifact
carry that, so a disputed row can be re-derived from upstream rather than
trusted. The DER half of the revision check lives beside the published table
(`test_every_committed_artifact_is_traceable_to_a_pinned_revision`).
"""

from __future__ import annotations

import json
from pathlib import Path

from test_published_table import (
    ARTIFACTS,
    FIXTURE_PREFIXES,
    REPO_ROOT,
    committed_der_artifacts,
)


def test_every_configurable_diarizer_artifact_names_its_operating_point():
    """Two runs of one checkpoint at different settings are different numbers.

    An adapter that exposes ``run_config`` has settings that change what was
    measured without being a model or dataset pin — Sortformer's streaming
    latency preset, DiariZen's embedding model. Its artifact must say which ones
    it ran under, or a diagnostic run could be published as the shipped one.
    """
    from raven_diar.registry import DIARIZER_ADAPTERS

    offenders = []
    for artifact in committed_der_artifacts():
        summary = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
        adapter_cls = DIARIZER_ADAPTERS.resolve(summary["adapter"])
        if hasattr(adapter_cls, "run_config") and "diarizer_config" not in summary:
            offenders.append(str(artifact.relative_to(REPO_ROOT)))
    assert not offenders, (
        "artifacts of a configurable diarizer that do not record the "
        f"configuration they were measured under: {offenders}"
    )


def committed_wer_artifacts() -> list[Path]:
    """Every committed WER artifact that is a product number, not a fixture."""
    return sorted({
        p.parent
        for p in ARTIFACTS.rglob("predictions_*.jsonl")
        if not p.relative_to(ARTIFACTS).parts[0].startswith(FIXTURE_PREFIXES)
    })


#: Adapters that talk to an endpoint someone else loaded weights into. The client
#: sees a model name, not which weights answer, so a ``model_revision`` here
#: would be an assertion nobody checked; it is recorded as null instead. What
#: the endpoint reports about itself belongs in the run manifest (#27).
_UNATTESTABLE_MODEL_ADAPTERS = frozenset({"vllm_openai", "modal_app"})

_FLOATING = {"", "none", "latest", "main", "master", "head"}


def test_every_committed_wer_artifact_is_traceable_to_its_references():
    """The WER twin of the DER provenance guard above.

    Every result names the reference set it was scored against — a revision, or
    the archive digest where upstream has no version history — and every hosted
    model names the versioned model it was asked for.
    """
    offenders = []
    for artifact in committed_wer_artifacts():
        rel = artifact.relative_to(REPO_ROOT)
        summary_path = artifact / "summary.json"
        if not summary_path.exists():
            offenders.append(f"{rel}: no summary.json")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "model_revision" not in summary:
            offenders.append(f"{rel}: no model_revision field")
        elif summary["adapter"] not in _UNATTESTABLE_MODEL_ADAPTERS and (
            str(summary["model_revision"]).lower() in _FLOATING
            or str(summary["model_revision"]).endswith("-latest")
        ):
            offenders.append(f"{rel}: model_revision={summary['model_revision']!r}")
        for r in summary["results"]:
            rev = r.get("dataset_revision")
            pinned = rev and str(rev).lower() not in _FLOATING
            if not (pinned or r.get("dataset_sha256")):
                offenders.append(f"{rel} [{r['subset']}]: dataset_revision={rev!r}")
    assert committed_wer_artifacts(), "a guard that matches nothing guards nothing"
    assert not offenders, (
        "committed WER artifacts whose provenance is not pinned: " + str(offenders)
    )
