#!/usr/bin/env python3
"""Seal artifacts/ with a sha256 manifest, and check a tree against it.

``make verify`` proves that every committed number re-scores from the committed
bytes. It cannot say *which* bytes a citation meant: a number quoted in March and
re-checked in June re-scores green against whatever artifacts/ holds in June. The
manifest closes that gap. ``artifacts/SHA256SUMS`` lists every file under
artifacts/ with its sha256, and a ``results-YYYY-MM-DD`` tag fixes one version of
that list — so "this number, from that release" names exact bytes, including the
files no scorer reads (``summary.json`` provenance, the artifact README).

The format is plain ``sha256sum`` output, so nobody needs this repo to check it:

    cd artifacts && sha256sum -c SHA256SUMS

Paths are POSIX, relative to the artifacts dir, sorted, one file per line. The
manifest does not list itself.

Usage:

    uv run python scripts/manifest.py            # (re)write artifacts/SHA256SUMS
    uv run python scripts/manifest.py --check    # exit 1 on any drift

``scripts/verify.py`` runs the same check before it re-scores anything, so
``make verify`` and CI fail on a modified, missing or unlisted artifact file.
Regenerating is deliberate and one command: ``make manifest``, then commit the
result together with the artifact change that caused it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAME = "SHA256SUMS"

# How many paths of one kind a failure report prints before it summarises. A
# regenerated campaign can touch thousands of files; the first screenful says
# what happened, the count says how much.
_REPORT_LIMIT = 20

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_files(artifacts_dir: Path) -> list[str]:
    """Every regular file under ``artifacts_dir`` except the manifest, sorted.

    Nothing else is skipped. A stray editor backup or ``.DS_Store`` under
    artifacts/ is a file a release would ship, so it fails the check like any
    other unlisted file rather than being waved through by a pattern.
    """
    paths = []
    for p in artifacts_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(artifacts_dir).as_posix()
        if rel == MANIFEST_NAME:
            continue
        paths.append(rel)
    return sorted(paths)


def build_manifest(artifacts_dir: Path) -> dict[str, str]:
    return {rel: sha256_file(artifacts_dir / rel) for rel in artifact_files(artifacts_dir)}


def render(manifest: dict[str, str]) -> str:
    # Two spaces: sha256sum's text-mode separator, so `sha256sum -c` reads it.
    return "".join(f"{digest}  {rel}\n" for rel, digest in sorted(manifest.items()))


def parse(text: str, source: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or not rel:
            raise ValueError(f"{source}:{lineno}: not a sha256sum line: {line!r}")
        if rel in manifest:
            raise ValueError(f"{source}:{lineno}: {rel} is listed twice")
        manifest[rel] = digest
    return manifest


def write_manifest(artifacts_dir: Path) -> Path:
    out = artifacts_dir / MANIFEST_NAME
    out.write_text(render(build_manifest(artifacts_dir)), encoding="utf-8")
    return out


@dataclass
class ManifestDrift:
    """What differs between artifacts/ on disk and the manifest that seals it."""

    manifest_missing: bool = False
    modified: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)    # listed, not on disk
    unlisted: list[str] = field(default_factory=list)   # on disk, not listed

    @property
    def ok(self) -> bool:
        return not (self.manifest_missing or self.modified or self.missing
                    or self.unlisted)

    def report(self, artifacts_dir: Path) -> str:
        manifest = artifacts_dir / MANIFEST_NAME
        if self.manifest_missing:
            return (f"FAIL: no manifest at {manifest} — artifacts/ is not sealed. "
                    "Run `make manifest` and commit the result.")
        lines = [f"FAIL: artifacts differ from {manifest}:"]
        for label, paths in (
            ("modified (sha256 differs)", self.modified),
            ("missing (listed, not on disk)", self.missing),
            ("unlisted (on disk, not in the manifest)", self.unlisted),
        ):
            if not paths:
                continue
            lines.append(f"  {label}: {len(paths)}")
            lines.extend(f"    {p}" for p in paths[:_REPORT_LIMIT])
            if len(paths) > _REPORT_LIMIT:
                lines.append(f"    … and {len(paths) - _REPORT_LIMIT} more")
        lines.append(
            "If the change is intended, run `make manifest` and commit the new "
            f"{MANIFEST_NAME} with it; a published release keeps the old one."
        )
        return "\n".join(lines)


def check_manifest(artifacts_dir: Path) -> ManifestDrift:
    path = artifacts_dir / MANIFEST_NAME
    if not path.is_file():
        return ManifestDrift(manifest_missing=True)
    listed = parse(path.read_text(encoding="utf-8"), path)
    on_disk = set(artifact_files(artifacts_dir))
    drift = ManifestDrift()
    for rel in sorted(listed):
        if rel not in on_disk:
            drift.missing.append(rel)
        elif sha256_file(artifacts_dir / rel) != listed[rel]:
            drift.modified.append(rel)
    drift.unlisted = sorted(on_disk - listed.keys())
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="directory to seal or check (default: artifacts/)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="compare against the existing manifest instead of rewriting it",
    )
    args = parser.parse_args(argv)

    if not args.artifacts_dir.is_dir():
        print(f"FAIL: artifacts dir does not exist: {args.artifacts_dir}",
              file=sys.stderr)
        return 2

    if args.check:
        drift = check_manifest(args.artifacts_dir)
        if not drift.ok:
            print(drift.report(args.artifacts_dir), file=sys.stderr)
            return 1
        n = len(artifact_files(args.artifacts_dir))
        print(f"OK: {n} artifact file(s) match {args.artifacts_dir / MANIFEST_NAME}.")
        return 0

    out = write_manifest(args.artifacts_dir)
    n = len(artifact_files(args.artifacts_dir))
    print(f"wrote {out} ({n} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
