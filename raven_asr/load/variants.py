"""Named load-test server settings (serving/load_variants.toml), validated.

A variant is a published recipe plus host knobs of serving/launch.py and the
reason production would set them. The harness records the whole entry next to
every result; ``python -m raven_asr.load variant-env <id>`` prints the knobs as
``KEY=VALUE`` lines for whoever starts the server.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

VARIANTS_FILE = Path(__file__).resolve().parents[2] / "serving" / "load_variants.toml"
RECIPES_FILE = VARIANTS_FILE.with_name("recipes.toml")

#: Host knobs per engine (serving/launch.py's environment list).
KNOBS = {
    "nemo": {"RAVEN_SERVE_GPU_MEM", "RAVEN_SERVE_NEMO_MAX_BATCH", "RAVEN_SERVE_NEMO_BATCH_WAIT_MS"},
    "vllm": {"RAVEN_SERVE_GPU_MEM", "RAVEN_SERVE_MAX_NUM_SEQS"},
    "transformers": {"RAVEN_SERVE_GPU_MEM"},
}


class VariantError(ValueError):
    pass


def load_variants(path: Path = VARIANTS_FILE, recipes_path: Path = RECIPES_FILE) -> dict[str, dict]:
    variants = tomllib.loads(path.read_text(encoding="utf-8")).get("variants", {})
    recipes = tomllib.loads(recipes_path.read_text(encoding="utf-8"))["recipes"]
    for vid, v in variants.items():
        base = v.get("base")
        if base not in recipes:
            raise VariantError(f"variant {vid!r}: base {base!r} is not a recipe")
        if not vid.startswith(f"{base}@"):
            raise VariantError(f"variant {vid!r}: the id must be '<base>@<name>'")
        env = v.get("env")
        if not isinstance(env, dict) or not env:
            raise VariantError(f"variant {vid!r}: env must name at least one knob")
        allowed = KNOBS[recipes[base]["engine"]]
        if bad := sorted(set(env) - allowed):
            raise VariantError(f"variant {vid!r}: {bad} are not {recipes[base]['engine']} "
                               f"host knobs; allowed: {sorted(allowed)}")
        if not all(isinstance(x, str) and x.strip() for x in env.values()):
            raise VariantError(f"variant {vid!r}: knob values are non-empty strings")
        if not str(v.get("reason", "")).strip():
            raise VariantError(f"variant {vid!r}: say why production would set this")
    return variants


def record(vid: str | None, recipe: str | None) -> dict | None:
    """The variant entry recorded with a result; checks it serves ``recipe``."""
    if not vid:
        return None
    v = load_variants().get(vid)
    if v is None:
        raise VariantError(f"no variant {vid!r} in {VARIANTS_FILE.name}")
    if recipe and v["base"] != recipe:
        raise VariantError(f"variant {vid!r} serves {v['base']!r}, not {recipe!r}")
    return {"id": vid, **v}
