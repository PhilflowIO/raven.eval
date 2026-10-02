"""Run manifest + ``schema_version`` of a Tier-2 ``summary.json``.

A committed number re-scores from its committed data (``make verify``), and its
summary names the model and dataset revisions it was measured at. What neither
says is *what produced it*: which commit of this repository, which resolved
dependency set, on what hardware, under which contract, from which command. The
run manifest is that record, written once per invocation by both harnesses
(``raven_asr.runner``, ``raven_diar.reproduce``) through this one module so the
WER and the DER summary cannot describe a run differently.

``schema_version`` names the shape of the summary around it. A reader that meets
a version it does not know must stop rather than guess, which is why
``scripts/verify.py`` fails on one. The versions are listed in ``CHANGELOG.md``
("summary.json schema versions"):

    1  every summary written before the field existed. It is identified by the
       field being ABSENT; those artifacts are not rewritten, because a manifest
       back-filled today would describe an environment nobody recorded.
    2  ``schema_version`` + ``run_manifest`` (this module).

The manifest never holds a secret. It records the command line, never an
environment variable's value and never an endpoint URL — the adapters read both
from the environment precisely so they stay out of this repository
(docs/TIER2-KEYS.md).

Standard library only: the Tier-1 re-scorer imports this module and must not
grow a dependency to do so.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

#: The version a summary written by this code carries.
SCHEMA_VERSION: Final[int] = 2
#: The version of a summary that has no ``schema_version`` field at all.
LEGACY_SCHEMA_VERSION: Final[int] = 1
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset(
    {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}
)

#: Exactly the keys of a version-2 ``run_manifest``. A key outside this set is as
#: much a format this reader does not know as an unknown version is.
MANIFEST_FIELDS: Final[tuple[str, ...]] = (
    "git", "uv_lock_sha256", "python", "platform", "libraries", "gpu",
    "config_sha256", "argv",
)

#: Libraries whose version can move a score or the bytes a score is computed
#: from, shared by both harnesses. Each harness adds its own inference stack.
CORE_LIBRARIES: Final[tuple[str, ...]] = (
    "raven-eval-core", "jiwer", "text2num", "unidecode", "num2words", "sacrebleu",
    "pyannote.metrics", "pyannote.core", "numpy",
)

# The checkout this module is imported from. Provenance is a statement about the
# code that ran, so it is anchored here and not at the caller's working directory.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


class UnknownSchemaVersionError(ValueError):
    """A summary declares a ``schema_version`` this code does not understand."""


def summary_schema_version(summary: dict[str, Any]) -> int:
    """The schema version of one parsed ``summary.json``.

    An absent field is :data:`LEGACY_SCHEMA_VERSION`. Anything that is not a
    supported integer raises — including ``true`` and ``"2"``, which Python
    would otherwise happily compare equal to a known version.
    """
    if "schema_version" not in summary:
        return LEGACY_SCHEMA_VERSION
    version = summary["schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in SUPPORTED_SCHEMA_VERSIONS
    ):
        raise UnknownSchemaVersionError(
            f"unknown schema_version {version!r}; this code reads "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)} (see CHANGELOG.md)"
        )
    return version


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _manifest_problem(manifest: object) -> str:
    if not isinstance(manifest, dict):
        return "run_manifest is not an object"
    if set(manifest) != set(MANIFEST_FIELDS):
        missing = sorted(set(MANIFEST_FIELDS) - set(manifest))
        unknown = sorted(set(manifest) - set(MANIFEST_FIELDS))
        return f"run_manifest fields: missing {missing}, unknown {unknown}"
    git = manifest["git"]
    if not isinstance(git, dict) or set(git) != {"sha", "dirty"}:
        return "run_manifest.git must hold exactly sha and dirty"
    if not _is_sha256(manifest["config_sha256"]):
        return "run_manifest.config_sha256 is not a sha256 hex digest"
    lock = manifest["uv_lock_sha256"]
    if lock is not None and not _is_sha256(lock):
        return "run_manifest.uv_lock_sha256 is not a sha256 hex digest"
    argv = manifest["argv"]
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        return "run_manifest.argv is not a list of strings"
    if not isinstance(manifest["libraries"], dict):
        return "run_manifest.libraries is not an object"
    if manifest["gpu"] is not None and not isinstance(manifest["gpu"], dict):
        return "run_manifest.gpu is neither null nor an object"
    return ""


def summary_problem(summary: object) -> str:
    """Why this ``summary.json`` is not in a format this code reads, or ``""``.

    A version-2 summary must carry a well-formed ``run_manifest``; a legacy one
    must not carry one, since a manifest beside an absent version would be read
    by nobody.
    """
    if not isinstance(summary, dict):
        return "summary.json is not an object"
    try:
        version = summary_schema_version(summary)
    except UnknownSchemaVersionError as exc:
        return str(exc)
    if version == LEGACY_SCHEMA_VERSION:
        if "run_manifest" in summary:
            return (
                "run_manifest without schema_version — a summary that carries a "
                f"manifest must declare schema_version {SCHEMA_VERSION}"
            )
        return ""
    if "run_manifest" not in summary:
        return f"schema_version {version} requires a run_manifest"
    return _manifest_problem(summary["run_manifest"])


def config_sha256(resolved_config: dict[str, Any]) -> str:
    """sha256 of a resolved config, over its canonical JSON form.

    Canonical means sorted keys and no insignificant whitespace, so two runs
    under the same settings hash alike however the dict was assembled.
    """
    canonical = json.dumps(
        resolved_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def library_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    """Installed version per distribution; ``None`` where it is not installed.

    A null is kept rather than dropped: "this lane ran without torch" is part of
    what the run was, and an absent key would not say it.
    """
    versions: dict[str, str | None] = {}
    for name in sorted(set(names)):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_state() -> dict[str, str | bool | None]:
    """``{"sha", "dirty"}`` of the checkout this code runs from.

    Both are ``None`` when the code does not run from its own git checkout (an
    installed wheel, an exported tarball): there is then no commit to name, and
    the SHA of whatever repository happens to enclose the install would be a
    false one. ``dirty`` counts untracked files too — an uncommitted module is
    code that ran and that no SHA describes.
    """
    top = _git("rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != _REPO_ROOT:
        return {"sha": None, "dirty": None}
    status = _git("status", "--porcelain")
    return {
        "sha": _git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def uv_lock_sha256() -> str | None:
    """sha256 of the ``uv.lock`` beside this checkout; ``None`` without one."""
    lock = _REPO_ROOT / "uv.lock"
    if not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def local_gpu() -> dict[str, Any]:
    """The accelerator a LOCAL inference run had available.

    Always an object, so that ``"gpu": null`` in a manifest keeps one meaning:
    inference did not run on this machine (a hosted API, a remote endpoint). A
    local run on CPU reads ``cuda_available: false`` instead.
    """
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "cuda": None, "devices": []}
    available = bool(torch.cuda.is_available())
    return {
        "cuda_available": available,
        "cuda": torch.version.cuda,
        "devices": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if available else []
        ),
    }


def cli_invocation() -> list[str]:
    """The command line of this process, interpreter flags included.

    ``sys.orig_argv`` rather than ``sys.argv``: under ``python -m`` the latter
    replaces the module name with a file path and drops ``-m``, so it is not the
    command anyone typed. The interpreter is reduced to its name — its absolute
    path names a home directory and says nothing about the run.
    """
    argv = list(sys.orig_argv)
    return [Path(argv[0]).name, *argv[1:]] if argv else []


def build_run_manifest(
    *,
    resolved_config: dict[str, Any],
    libraries: tuple[str, ...],
    gpu: dict[str, Any] | None,
) -> dict[str, Any]:
    """The ``run_manifest`` block of a version-2 summary.

    ``libraries`` are the harness's own inference-stack distributions, recorded
    in addition to :data:`CORE_LIBRARIES`. ``gpu`` is :func:`local_gpu` for a
    run that infers on this machine and ``None`` for one that calls out.
    """
    return {
        "git": git_state(),
        "uv_lock_sha256": uv_lock_sha256(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": library_versions((*CORE_LIBRARIES, *libraries)),
        "gpu": gpu,
        "config_sha256": config_sha256(resolved_config),
        "argv": cli_invocation(),
    }


def same_run_conditions(a: object, b: object) -> bool:
    """Whether two manifests describe runs whose results may share one summary.

    Everything but the command line has to match: a resumed invocation is typed
    differently by nature, but code, dependencies, hardware and config must be
    the ones the earlier results were measured under.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return all(a.get(f) == b.get(f) for f in MANIFEST_FIELDS if f != "argv")
