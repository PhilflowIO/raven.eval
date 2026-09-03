"""Lazy, convention-based registries for diarizer adapters and dataset loaders.

Why this exists: the runner used to dispatch through a literal ``if``-chain, so
adding a diarizer meant editing ``reproduce.py``. That makes the seam the toolkit
promises ("re-run any number we publish, with any diarizer") a lie by
construction — every new model touches the dispatcher. Here the dispatch is data:
a name resolves to a module under a package, and the module publishes ONE
attribute (a zero-or-more-kwarg factory). Adding a diarizer is therefore a module
plus a :class:`~raven_diar.config.DiarizerSpec` entry — no dispatch code touched.

**Laziness is a hard requirement, not a nicety.** The old ``if``-chain imported
inside its branches on purpose: ``pyannote_community1`` pulls torch, and the
Tier-1 re-score path (``scripts/verify.py``) must stay GPU-free and torch-free.
So this registry stores *names*, never imported objects: nothing is imported
until :meth:`LazyRegistry.resolve` is called for that one name. Listing the
available names (:meth:`LazyRegistry.available`) walks the package directory with
``pkgutil`` and imports nothing at all.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

# Modules that live in an adapter/loader package but are shared plumbing, not
# entries: ``base`` (the protocol) and ``aggregate`` (the word/utterance folding
# helper). Listed per registry so a new shared module is one word, not a rule.


class LazyRegistry:
    """Resolve a registry key to a factory, importing at most one module.

    By convention a key maps to ``<package>.<key>`` and the module exposes
    ``<attr>`` (a callable factory — normally the adapter/loader class itself).
    Keys that cannot follow that convention can be pointed elsewhere with
    :meth:`register`, which stores an import path *string*, so an override is
    just as lazy as the convention.
    """

    def __init__(
        self,
        package: str,
        attr: str,
        kind: str,
        shared_modules: frozenset[str] = frozenset({"base"}),
    ) -> None:
        self._package = package
        self._attr = attr
        self._kind = kind
        self._shared = shared_modules
        self._overrides: dict[str, str] = {}

    def register(self, key: str, target: str) -> None:
        """Point ``key`` at ``"module.path:ATTRIBUTE"`` instead of the convention.

        The target is kept as a string and imported only on ``resolve(key)``.
        """
        if ":" not in target:
            raise ValueError(
                f"{self._kind} override for {key!r} must be 'module.path:ATTR', "
                f"got {target!r}"
            )
        self._overrides[key] = target

    def available(self) -> list[str]:
        """Keys this registry can resolve — a directory walk, zero imports."""
        package = importlib.import_module(self._package)
        found = {
            info.name
            for info in pkgutil.iter_modules(package.__path__)
            if not info.name.startswith("_") and info.name not in self._shared
        }
        return sorted(found | set(self._overrides))

    def resolve(self, key: str) -> Callable[..., Any]:
        """Import the module backing ``key`` and return its factory.

        Raises ``ValueError`` with the resolvable keys when ``key`` is unknown, and
        lets an ImportError from the module itself propagate untouched (a missing
        heavy dependency must not be reported as "unknown adapter").
        """
        module_path, _, attr = self._overrides.get(
            key, f"{self._package}.{key}:{self._attr}"
        ).partition(":")
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            # Only "the entry module itself is missing" means an unknown key; a
            # missing *transitive* dependency (torch, nemo) must surface as-is.
            if exc.name != module_path:
                raise
            raise ValueError(
                f"unknown {self._kind}: {key!r} — expected a module "
                f"{module_path!r}; known: {', '.join(self.available())}"
            ) from exc
        try:
            factory = getattr(module, attr)
        except AttributeError as exc:
            raise ValueError(
                f"{module_path!r} is not a {self._kind}: it must expose "
                f"a module-level {attr!r} factory"
            ) from exc
        if not callable(factory):
            raise ValueError(
                f"{module_path}.{attr} is not callable — a {self._kind} entry "
                "must be a class or a factory function"
            )
        return factory


# The two registries the DER runner dispatches through. Values are resolved
# lazily, so importing this module (or ``raven_diar.reproduce``) never imports
# torch, pyannote.audio or nemo.
DIARIZER_ADAPTERS = LazyRegistry(
    "raven_diar.adapters",
    attr="ADAPTER",
    kind="diarizer adapter",
    shared_modules=frozenset({"base", "aggregate"}),
)
DATASET_LOADERS = LazyRegistry(
    "raven_diar.datasets", attr="LOADER", kind="dataset loader"
)
