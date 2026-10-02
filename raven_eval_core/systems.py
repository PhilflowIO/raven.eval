"""Training-data disclosure per benchmarked system — the ``systems:`` block.

``benchmark.config.yaml`` → ``systems`` records, for every system with a
published row, what its vendor documents about training data and whether that
overlaps the corpora it is scored on here. This module is the one reader of that
block: it loads it, validates it, and answers the two questions the result
tables print.

What the two published values mean — and do not mean:

``training_data.status``
    ``offengelegt``        the vendor names the training corpora.
    ``teilweise``          part of the lineage is documented and part is not
                           (a documented base model under an undocumented
                           fine-tune).
    ``nicht offengelegt``  no statement — a closed API, or a card that names
                           nothing.

``test_corpus_seen[<dataset>].seen``
    ``ja``         documented training on that corpus: its train split, or the
                   same source. It is NOT a claim that the test recordings
                   themselves were trained on.
    ``nein``       the vendor states the training set as a closed list and the
                   corpus is not in it.
    ``unbekannt``  nothing documented either way. Never a guess: an open-ended
                   list ("including …") cannot support ``nein``.

Open weights settle none of this. Weights ship without their training data and
the data cannot be read back out of them, so the only admissible sources are
model cards, papers and dataset documentation — which is why every ``ja`` and
``nein`` must carry a source URL and a verbatim quote.

There is deliberately no model list here. System keys are the registry keys of
``raven_asr.config.KNOWN_MODELS`` and ``raven_diar.config.KNOWN_DIARIZERS``, and
corpus keys are the dataset selectors those harnesses accept. Both are passed in
by the caller, because this package is the dependency-light metric core and does
not import the harnesses that depend on it.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: The contract file this block lives in, next to the packages.
CONFIG_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "benchmark.config.yaml"

TRAINING_DATA_STATUSES: Final[tuple[str, ...]] = (
    "offengelegt", "teilweise", "nicht offengelegt",
)
CORPUS_SEEN_VALUES: Final[tuple[str, ...]] = ("ja", "nein", "unbekannt")
#: The values that assert something and therefore need evidence.
EVIDENCED_SEEN_VALUES: Final[frozenset[str]] = frozenset({"ja", "nein"})
#: The metric families of ``datasets:`` — a system belongs to exactly one.
FAMILIES: Final[tuple[str, ...]] = ("wer", "der")


class SystemsContractError(ValueError):
    """The ``systems:`` block breaks its own rules. Carries every violation."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(
            "benchmark.config.yaml `systems:` is invalid:\n  " + "\n  ".join(problems)
        )


@dataclass(frozen=True)
class Source:
    """One page that was read, and what on it supports the claim."""

    url: str
    #: Verbatim from the page. None only where the finding is an absence — a
    #: page that says nothing about training data has nothing to quote.
    quote: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class CorpusSeen:
    """Whether a system's documented training data includes one test corpus."""

    seen: str
    source: Source | None = None
    note: str | None = None


@dataclass(frozen=True)
class SystemDisclosure:
    """Everything the contract records about one system's training data."""

    family: str
    key: str
    status: str
    checked: _dt.date
    sources: tuple[Source, ...]
    note: str | None
    test_corpus_seen: Mapping[str, CorpusSeen]
    #: Set when this registry key serves the same checkpoint as another key and
    #: therefore shares its entry; the fields above are that entry's.
    same_as: str | None = None

    def corpus_seen(self, dataset: str) -> CorpusSeen:
        """The recorded value for ``dataset`` — never a default.

        A missing entry is an error rather than ``unbekannt``: "nobody looked"
        and "we looked and the vendor says nothing" are different facts, and
        only the second may be published.
        """
        try:
            return self.test_corpus_seen[dataset]
        except KeyError:
            raise KeyError(
                f"systems.{self.family}.{self.key} has no test_corpus_seen entry "
                f"for {dataset!r} (has: {sorted(self.test_corpus_seen)})"
            ) from None


def _source(raw: Any, where: str, problems: list[str]) -> Source | None:
    if not isinstance(raw, Mapping) or not str(raw.get("url") or "").startswith("https://"):
        problems.append(f"{where}: a source needs an https `url`, got {raw!r}")
        return None
    return Source(url=raw["url"], quote=raw.get("quote"), note=raw.get("note"))


def parse_systems(
    block: Any,
    *,
    known_systems: Mapping[str, Collection[str]],
    known_corpora: Mapping[str, Collection[str]],
    same_checkpoint: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, dict[str, SystemDisclosure]]:
    """Validate the raw ``systems:`` block and return it typed.

    ``known_systems`` and ``known_corpora`` map each family (``wer`` / ``der``)
    to the registry keys and dataset selectors that exist in code; an entry
    naming anything else is rejected, so this block cannot grow into a second
    model list. ``same_checkpoint`` maps each family's registry keys to the
    checkpoint they serve and is what a ``same_as`` alias is checked against.

    Raises :class:`SystemsContractError` listing every violation at once.
    """
    problems: list[str] = []
    out: dict[str, dict[str, SystemDisclosure]] = {}
    if not isinstance(block, Mapping):
        raise SystemsContractError(["`systems` must be a mapping of wer / der"])
    for family in block:
        if family not in FAMILIES:
            problems.append(f"systems.{family}: unknown family, expected one of {FAMILIES}")

    aliases: list[tuple[str, str, str]] = []
    for family in FAMILIES:
        out[family] = {}
        for key, raw in (block.get(family) or {}).items():
            where = f"systems.{family}.{key}"
            if key not in known_systems[family]:
                problems.append(
                    f"{where}: not a registry key — add the system to the "
                    f"{family.upper()} registry first, this block lists no models "
                    f"of its own"
                )
                continue
            if not isinstance(raw, Mapping):
                problems.append(f"{where}: must be a mapping")
                continue
            if "same_as" in raw:
                if set(raw) != {"same_as"}:
                    problems.append(f"{where}: `same_as` stands alone, drop {sorted(set(raw) - {'same_as'})}")
                aliases.append((family, key, raw["same_as"]))
                continue

            training = raw.get("training_data")
            if not isinstance(training, Mapping):
                problems.append(f"{where}: missing `training_data`")
                continue
            status = training.get("status")
            if status not in TRAINING_DATA_STATUSES:
                problems.append(
                    f"{where}.training_data.status={status!r}, expected one of "
                    f"{TRAINING_DATA_STATUSES}"
                )
            checked = training.get("checked")
            if not isinstance(checked, _dt.date):
                problems.append(f"{where}.training_data.checked={checked!r} is not a date")
            sources = tuple(
                s for s in (
                    _source(r, f"{where}.training_data.sources", problems)
                    for r in training.get("sources") or []
                ) if s
            )
            if not sources:
                problems.append(
                    f"{where}.training_data: no source — even `nicht offengelegt` "
                    f"names the page that was checked"
                )
            elif status != "nicht offengelegt" and not any(s.quote for s in sources):
                problems.append(
                    f"{where}.training_data: status {status!r} asserts a disclosure "
                    f"but no source carries a verbatim quote"
                )

            seen_map: dict[str, CorpusSeen] = {}
            for dataset, entry in (raw.get("test_corpus_seen") or {}).items():
                at = f"{where}.test_corpus_seen.{dataset}"
                if dataset not in known_corpora[family]:
                    problems.append(f"{at}: not a {family.upper()} dataset of this repo")
                    continue
                if not isinstance(entry, Mapping) or entry.get("seen") not in CORPUS_SEEN_VALUES:
                    problems.append(f"{at}: `seen` must be one of {CORPUS_SEEN_VALUES}, got {entry!r}")
                    continue
                source = None
                if "source" in entry:
                    source = _source(
                        {"url": entry["source"], "quote": entry.get("quote")}, at, problems
                    )
                if entry["seen"] in EVIDENCED_SEEN_VALUES and not (source and source.quote):
                    problems.append(
                        f"{at}: `{entry['seen']}` is a claim and needs a `source` URL "
                        f"plus a verbatim `quote`; without one it is `unbekannt`"
                    )
                seen_map[dataset] = CorpusSeen(entry["seen"], source, entry.get("note"))
            if not seen_map:
                problems.append(f"{where}: `test_corpus_seen` is empty")

            if status in TRAINING_DATA_STATUSES and isinstance(checked, _dt.date):
                out[family][key] = SystemDisclosure(
                    family=family, key=key, status=status, checked=checked,
                    sources=sources, note=training.get("note"),
                    test_corpus_seen=seen_map,
                )

    for family, key, target in aliases:
        where = f"systems.{family}.{key}"
        resolved = out[family].get(target)
        if resolved is None:
            problems.append(f"{where}: same_as {target!r} is not a full entry of systems.{family}")
            continue
        checkpoints = (same_checkpoint or {}).get(family, {})
        if checkpoints.get(key) is None or checkpoints.get(key) != checkpoints.get(target):
            problems.append(
                f"{where}: same_as {target!r}, but the registry does not bind both "
                f"keys to one checkpoint ({checkpoints.get(key)!r} vs "
                f"{checkpoints.get(target)!r})"
            )
            continue
        out[family][key] = SystemDisclosure(
            family=family, key=key, status=resolved.status, checked=resolved.checked,
            sources=resolved.sources, note=resolved.note,
            test_corpus_seen=resolved.test_corpus_seen, same_as=target,
        )

    if problems:
        raise SystemsContractError(problems)
    return out


def load_systems(
    *,
    known_systems: Mapping[str, Collection[str]],
    known_corpora: Mapping[str, Collection[str]],
    same_checkpoint: Mapping[str, Mapping[str, str]] | None = None,
    config_path: Path = CONFIG_PATH,
) -> dict[str, dict[str, SystemDisclosure]]:
    """Read and validate ``systems:`` from the committed contract file."""
    import yaml  # the `asr` extra; the Tier-1 re-score path never needs this block

    doc = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if "systems" not in doc:
        raise SystemsContractError([f"{config_path} has no `systems:` block"])
    return parse_systems(
        doc["systems"],
        known_systems=known_systems,
        known_corpora=known_corpora,
        same_checkpoint=same_checkpoint,
    )
