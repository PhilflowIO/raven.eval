# Tier-2 DER reproduce — what each diarizer costs you (token, GPU, licence)

Tier-1 (`make verify`) re-scores committed RTTMs and needs **nothing** — no GPU,
no token. Tier-2 (`make reproduce METRIC=der`) actually runs a diarizer over
public audio, so it costs something. That cost differs per diarizer, and this
page states it per diarizer rather than hiding it.

## Which diarizer costs what

| `MODEL=` | Extra | HF token | Gated licence | GPU | Notes |
|---|---|---|---|---|---|
| `pyannote-community-1` | `diar` | **yes** | **yes** (accept on HF) | strongly recommended | needs system FFmpeg *shared* libs (see below) |
| `sortformer-4spk-v1` | `sortformer` | no | no (weights are public) | strongly recommended | CC-BY-NC-4.0 → **non-commercial**; hard cap of **4 speakers** |
| `assemblyai-universal-3-5-pro` | `assemblyai` | no | no | **none** (hosted) | needs `ASSEMBLYAI_API_KEY`; **costs money per hour of audio** — see below |

The two local models need no API key; the hosted one needs no GPU. Adding a third diarizer is a module
under `raven_diar/adapters/` exposing an `ADAPTER` factory plus a `DiarizerSpec`
entry in `raven_diar/config.py` — the runner dispatches through
`raven_diar/registry.py` and is not edited (see that module's docstring).

## The pyannote community-1 requirements

1. **A Hugging Face token.** Export one of `HF_TOKEN` / `HUGGINGFACE_TOKEN`
   (create at <https://huggingface.co/settings/tokens>).
2. **Accept the gated model license.** Visit
   <https://huggingface.co/pyannote/speaker-diarization-community-1> while logged
   in and accept the conditions. Without this the pipeline load returns `None`
   and the adapter raises a clear error.
3. **A GPU.** CPU technically works but is impractically slow for meeting-length
   audio. The adapter auto-selects CUDA when available (override with
   `PyannoteCommunity1Diarizer(device=...)`).
4. **System FFmpeg *shared* libraries.** pyannote.audio 4 decodes audio through
   `torchcodec`, which dlopens `libav*` at runtime — a *static* ffmpeg binary is
   not enough. Without the shared `libavutil`/`libavcodec`/… `.so`s on the loader
   path, decoding fails with `torchcodec is not available. Cannot read audio file`.
   On a box without system libav, extract a shared FFmpeg build (e.g. a BtbN
   `*-shared*` FFmpeg 7.x) and point `LD_LIBRARY_PATH` at its `lib/`.

## The Sortformer requirements

`nvidia/diar_sortformer_4spk-v1` is an end-to-end neural diarizer run locally
through NVIDIA NeMo. What it does **not** need: an HF token, a gated-licence
acceptance, an API key. What it does need:

1. **The `sortformer` extra** (`nemo_toolkit[asr]` + torch + huggingface-hub).
   It is its own extra because NeMo is a large dependency tree of its own —
   nobody should have to install it to score DER or to run the pyannote lane.
   The adapter imports NeMo lazily, so the repo installs and its tests pass
   without the extra.
2. **A GPU.** CPU runs but is impractically slow for meeting-length audio.
   Override the auto-selected device with `SortformerDiarizer(device=...)`.
3. **Awareness of two model properties, not bugs:**
   * The weights are **CC-BY-NC-4.0 — non-commercial**. Read it before you ship
     a number derived from it into a commercial context.
   * The checkpoint is **`4spk`**: a hard cap of four speakers. Audio with more
     speakers gets folded into four tracks, which surfaces as speaker confusion.
     That is why AMI (4 speakers) and CALLHOME-de (2) are in scope for it.

The revision is pinned via `hf_hub_download` rather than NeMo's
`from_pretrained`, which accepts no `revision` and would silently track the HF
branch — a published DER must not be able to drift under a re-upload.

## The AssemblyAI requirements

`assemblyai-universal-3-5-pro` is the first **hosted** diarizer here: no GPU, no
weights, no licence to accept — and the first one that costs money per hour of
audio. Three things about it are benchmark-relevant, not trivia:

1. **There is no diarization-only endpoint.** Diarization is the
   `speaker_labels` flag on a normal transcription request, so every DER file
   also buys a German transcript you do not score. That is why the rate below
   includes the transcription base price.
2. **Price (verified on assemblyai.com/pricing, 2026-09-03).** Universal-3.5 Pro
   is **0.21 $/h**, the Speaker Diarization add-on is **0.02 $/h** → **0.23 $/h**
   all-in. German sits on the same tier as English. Billing is per second, so a
   corpus of many short files carries no rounding premium. CALLHOME-de is 120
   files / ~18.4 h ≈ **4.2 $** for one full sweep. Budget before you run.
3. **The pin is an alias, and that is the vendor's limit, not ours.** AssemblyAI
   publishes no immutable model version. The only selector is
   `speech_models`, and its *default* is a fallback chain
   `["universal-3-5-pro", "universal-2"]` — two different models, so an unpinned
   published DER could silently come from either. The adapter therefore sends a
   **single-element** list and asserts the response's `speech_model_used` is that
   same alias, failing the file otherwise. If AssemblyAI re-trains behind the
   alias, the number moves and nothing in the API says so. Stated plainly here
   rather than dressed up as a version pin.

```bash
export ASSEMBLYAI_API_KEY=...          # your key; this repo ships names, never values
make reproduce METRIC=der DATASET=callhome-de MODEL=assemblyai-universal-3-5-pro \
  EXTRA=assemblyai LIMIT=3             # smoke first: 3 files ≈ 0.10 $
```

Word-level speaker labels are folded into turns by the **shared**
`raven_diar/adapters/aggregate.py` at the shared threshold — never by
AssemblyAI's own `utterances` grouping, which would make a DER difference
against another provider partly a difference of two vendors' folding rules.

## Install

```bash
uv sync --extra dev --extra diar        # pyannote lane: torch + pyannote.audio
uv sync --extra dev --extra sortformer  # sortformer lane: nemo_toolkit[asr] + torch
uv sync --extra dev --extra assemblyai  # hosted lane: httpx only (no torch, no GPU)
```

Both extras are heavy (torch) and deliberately isolated: Tier-1 verify never
imports either, so `make verify` stays light and GPU-free. `make reproduce`
defaults to `diar` for `METRIC=der`; select the other lane with
`EXTRA=sortformer`.

## Pin the model & dataset revisions (reproducibility)

Pin **both** via the Makefile for a byte-reproducible run. The committed VoxConverse
numbers used exactly these commits:

```bash
make reproduce METRIC=der DATASET=voxconverse-test MODEL=pyannote-community-1 \
  MODEL_REV=3533c8cf8e369892e6b79ff1bf80f7b0286a54ee \
  DATASET_REV=24bf60be297701cd7e4ef18550c6d390c1b87365
```

The Sortformer lane pins the same way (its own extra, no token needed):

```bash
make reproduce METRIC=der DATASET=ami MODEL=sortformer-4spk-v1 EXTRA=sortformer \
  MODEL_REV=9f17b10df44c0a4c8f3c86fbddc9ee2d6ab9ac08
```

`MODEL_REV` → the accepted-license `pyannote/speaker-diarization-community-1` commit;
`DATASET_REV` → the `joonson/voxconverse` gold commit. VoxConverse's `v0.3` is a
*dataset-version label*, not a git tag (the repo has none) — it **is** the `master`
branch, so always pin the commit, never `v0.3`. Without a pin the run records a
floating revision, which a published artifact must not carry.

## Datasets (audio is caller-fetched, never redistributed here)

| `DATASET=`     | Gold source                                   | Audio you fetch |
|----------------|-----------------------------------------------|-----------------|
| `voxconverse`  | RTTMs shipped in `joonson/voxconverse` (pinned tag) — **easiest, fastest smoke** | `voxconverse_dev_wav.zip` / `_test_wav.zip` → `data/diar/voxconverse/audio/` |
| `callhome-de`  | `talkbank/callhome` config `deu`, converted to RTTM by the loader — **the German anchor** | materialised from the HF dataset by `prepare()` |
| `ami`          | prepared `only_words` RTTMs from `pyannote/AMI-diarization-setup` (pinned commit; an existing clone is verified against the pin) | fetched by `prepare()` from the Edinburgh AMI mirror — test split only (16 Mix-Headset wavs, ~1.1 GB) → `data/diar/ami/audio/` |

`prepare()` downloads the gold labels and, where the license allows it (AMI,
CC-BY-4.0), the audio too; otherwise it prints the exact audio step. A file
whose audio is missing is **skipped loudly** (never silently scored).

## Recommended run order

1. **VoxConverse** first — gold RTTMs need no conversion, so it's the quickest
   end-to-end DER and the fastest way to confirm the pipeline + token + GPU work:
   ```bash
   make reproduce METRIC=der DATASET=voxconverse MODEL=pyannote-community-1 LIMIT=3
   ```
2. **CALLHOME-de** next — the strategically important German number:
   ```bash
   make reproduce METRIC=der DATASET=callhome-de MODEL=pyannote-community-1
   ```
3. **AMI** for the 4-speaker meeting regime.

## From run to committed proof

```bash
make reproduce METRIC=der DATASET=voxconverse MODEL=pyannote-community-1
make promote   METRIC=der RESULTS=results/reproduce-der/pyannote-community-1 RUN=$(date +%F)
make verify     # re-scores the promoted RTTMs with no GPU / no gated model
```

`promote` copies the `gold/` + `hyp/` RTTM trees and derives `expected.json` from
the run's `summary.json` (never hand-typed), so a committed DER can only ever
equal what the scorer produced.

## Optional: dscore cross-check

To independently confirm our DER against `nryant/dscore`:

```bash
git clone https://github.com/nryant/dscore ~/dscore
git -C ~/dscore checkout <pinned commit; see raven_diar/dscore_check.py>
export DSCORE_DIR=~/dscore
make dscore-check GOLD=data/diar/voxconverse/labels/dev/<id>.rttm \
                  HYP=results/reproduce-der/pyannote-community-1/hyp/voxconverse/<id>.rttm
```

dscore is a script repo (not pip-installable), so this is optional and gated on
`DSCORE_DIR`; it skips cleanly when unset and is not part of CI.

## Latency is NOT a Tier-2 number

The adapter records a per-file diarization latency, but it depends on your GPU and
audio length — treat it as Tier-3 transparency, never a published comparable
number. Only DER crosses the Tier-2 line as reproducible.
