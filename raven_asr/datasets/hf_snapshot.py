"""Pinned local snapshots of the HF-hosted WER corpora.

Streaming a test split from the Hub for the length of a full run is the
failure mode this module removes: parquet streaming from huggingface.co has
been observed to time out on jobs longer than about five hours, and a corpus of
ten thousand utterances against a slow model is such a job. Instead, the exact
files a split consists of at the pinned revision are fetched once into the HF
Hub cache (``HF_HOME`` / ``HF_HUB_CACHE``, resolved by ``huggingface_hub``) and
then read from local disk.

Each loader names its split's files explicitly (``hf_data_files``): repo-relative
paths at the pinned revision, the same files ``datasets`` resolves for that
config and split when it streams. Naming them is what makes the snapshot
checkable offline — a snapshot is complete when every named file is on disk, not
when a directory of the right name exists — and it keeps a test-split fetch from
pulling a multi-hundred-GB train split along with it.

The files are read lazily (``streaming=True`` over local parquet), so a run
never materialises a split in memory or converts it to a second on-disk copy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("raven_asr.datasets")


def snapshot_paths(
    repo_id: str,
    revision: str | None,
    files: Sequence[str],
    *,
    download: bool,
) -> list[Path] | None:
    """Local paths of ``files`` from ``repo_id`` at ``revision``, in the given order.

    With ``download=False`` this never touches the network and returns None
    unless every file is already in the cache. With ``download=True`` the
    missing files are fetched first; an incomplete result then raises.
    """
    if revision is None:
        raise ValueError(
            f"{repo_id}: a local snapshot needs a pinned revision; pass the "
            f"registry pin, or stream with --streaming"
        )
    if not files:
        raise ValueError(f"{repo_id}: the loader names no data files to snapshot")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    if download:
        logger.info(
            "fetching %d pinned file(s) of %s@%s into the HF cache",
            len(files), repo_id, revision[:12],
        )
    try:
        root = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=list(files),
                local_files_only=not download,
            )
        )
    except LocalEntryNotFoundError:
        if download:
            raise
        return None
    paths = [root / f for f in files]
    missing = [f for f, p in zip(files, paths, strict=True) if not p.is_file()]
    if not missing:
        return paths
    if not download:
        return None
    raise RuntimeError(
        f"{repo_id}@{revision}: the pinned files {missing} are not in the repo at "
        f"that revision. A --dataset-revision other than the registry pin needs "
        f"its own file list; stream it with --streaming instead."
    )


def local_split(
    repo_id: str, revision: str | None, files: Sequence[str], split: str
) -> Any:
    """The split as a lazily-read local dataset, fetching the snapshot if absent."""
    paths = snapshot_paths(repo_id, revision, files, download=False)
    if paths is None:
        paths = snapshot_paths(repo_id, revision, files, download=True)
        assert paths is not None
    from datasets import load_dataset

    return load_dataset(
        "parquet",
        data_files={split: [str(p) for p in paths]},
        split=split,
        streaming=True,
    )


__all__ = ["local_split", "snapshot_paths"]
