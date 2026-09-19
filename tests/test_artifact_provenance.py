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

from test_published_table import REPO_ROOT, committed_der_artifacts


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

