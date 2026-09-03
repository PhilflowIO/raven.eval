# raven.eval — one entrypoint per verification level.
# See README.md for the three-tier verifiability model.

.PHONY: install test verify reproduce promote dscore-check clean

install:            ## Install the pinned environment (uv, fails on lockfile drift).
	uv sync --locked --extra dev --extra asr

test:               ## Run the scorer + harness tests (the regression guard).
	uv run --extra dev --extra asr pytest -q

# --- Tier 1: zero-setup verification (no GPU, no API keys) -------------------
# Re-scores BOTH the committed WER predictions_*.jsonl AND the committed DER
# gold/hyp RTTMs with our own scorer, and asserts each matches its expected.json.
# This is how anyone verifies our numbers in seconds. Exits nonzero on any
# mismatch or on an artifacts dir with no WER *and* no DER artifacts.
verify:             ## Tier-1: re-score committed WER + DER artifacts → reproduce published numbers.
	uv run python scripts/verify.py

# --- Tier 2: full re-run on public datasets (your own keys / GPU) ------------
# Downloads the public dataset, runs inference with the pinned model, scores.
# METRIC dispatches the harness: `wer` → raven_asr (`asr` extra, adapter keys),
# `der` → raven_diar (`diar` extra: torch + pyannote.audio + GPU + gated model).
#   make reproduce METRIC=wer DATASET=common_voice_19_0 MODEL=modal/parakeet
#   make reproduce METRIC=der DATASET=voxconverse       MODEL=pyannote-community-1
#   make reproduce METRIC=der DATASET=callhome-de       MODEL=pyannote-community-1 LIMIT=5
# EXTRA overrides the dependency extra, because a diarizer's backend is its own
# (pyannote -> `diar`, sortformer -> `sortformer`); nobody installs both to run one:
#   make reproduce METRIC=der DATASET=ami MODEL=sortformer-4spk-v1 EXTRA=sortformer
# Keys/GPU/gating: docs/TIER2-KEYS.md (WER) + docs/TIER2-DER-KEYS.md (DER).
EXTRA ?= $(if $(filter der,$(METRIC)),diar,asr)

reproduce:          ## Tier-2: download public data → infer/diarize → score → table.
	uv run --extra $(EXTRA) \
		python -m $(if $(filter der,$(METRIC)),raven_diar,raven_asr).reproduce \
		--metric $(METRIC) --dataset $(DATASET) --model $(MODEL) \
		$(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(MODEL_REV),--model-revision $(MODEL_REV),) \
		$(if $(DATASET_REV),--dataset-revision $(DATASET_REV),)

# Turn a completed Tier-2 run into a committable Tier-1 artifact under artifacts/,
# which `make verify` then re-scores. METRIC selects the harness (default wer).
#   make promote METRIC=wer RESULTS=results/reproduce/modal-parakeet RUN=2026-07-30
#   make promote METRIC=der RESULTS=results/reproduce-der/pyannote-community-1 RUN=2026-07-30
promote:            ## Bridge: Tier-2 run → committable Tier-1 artifact.
	uv run --extra $(if $(filter der,$(METRIC)),diar,asr) \
		python -m $(if $(filter der,$(METRIC)),raven_diar,raven_asr).promote \
		--results-dir $(RESULTS) $(if $(RUN),--run-name $(RUN),)

# Optional DER cross-check: assert nryant/dscore agrees with pyannote.metrics on
# the same RTTMs. Needs a pinned dscore checkout (dscore is not pip-installable):
#   git clone https://github.com/nryant/dscore ~/dscore && export DSCORE_DIR=~/dscore
# Skips cleanly (exit 0) when DSCORE_DIR is unset — see raven_diar/dscore_check.py.
#   make dscore-check GOLD=path/to/gold.rttm HYP=path/to/hyp.rttm
dscore-check:       ## Optional: cross-check DER vs nryant/dscore (needs DSCORE_DIR).
	uv run python -m raven_diar.dscore_check --gold $(GOLD) --hyp $(HYP)

clean:
	rm -rf .venv .pytest_cache **/__pycache__

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'
