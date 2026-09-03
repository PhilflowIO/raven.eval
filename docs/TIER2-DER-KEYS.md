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
| `deepgram-nova-3` | `diar-hosted` | no | no | **none** | hosted API — needs `DEEPGRAM_API_KEY` and **costs money per hour of audio** |

The first two run locally and need no API key; the Deepgram lane is the inverse —
no GPU at all, but a metered vendor bill. Adding a fourth diarizer is a module
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
   * **It is an offline model, and its memory grows with the square of the
     recording length.** Measured on one RTX 3090 (peak CUDA allocation):
     1.02 GB @ 2 min, 2.51 @ 4, 4.95 @ 6, 8.32 @ 8, 12.65 @ 10, OOM @ 12
     (≈3.5e-5 GB/s²). So a 24 GB card tops out near 13 minutes of audio and an
     80 GB card near 25. CALLHOME-de (≈9 min mean) fits; **AMI does not** — its
     shortest test meeting is 14 min (≈25 GB), its longest 50 min (≈310 GB), so
     `DATASET=ami MODEL=sortformer-4spk-v1` will OOM on any single GPU. For
     long-form audio NVIDIA ships a different checkpoint,
     `nvidia/diar_streaming_sortformer_4spk-v2`; that is a new
     `DiarizerSpec` entry, not a flag on this one. Switching v1 into NeMo's
     streaming path does run in ~1 GB, but it is a different regime and it is
     worse: 23.13 % / 16.42 % DER on ten CALLHOME files where the offline model
     reads 18.94 % / 12.77 %.

The revision is pinned via `hf_hub_download` rather than NeMo's
`from_pretrained`, which accepts no `revision` and would silently track the HF
branch — a published DER must not be able to drift under a re-upload.

## The Deepgram requirements

`deepgram-nova-3` is the first **hosted** diarizer in the harness. It needs no
GPU, no weights and no licence acceptance — it needs an account and a budget.

1. **`DEEPGRAM_API_KEY`** in your environment. This repo ships env-var *names*
   only, never values or URLs.
2. **The `diar-hosted` extra** (`httpx` + the dataset loader). Deliberately not
   `diar`: a hosted diarizer must not drag in torch.
3. **Money.** There is no free lane here. See the cost note below.

### It costs per hour of audio, and there is no diarization-only endpoint

Deepgram does not expose diarization on its own. Diarization is a parameter on
the **pre-recorded transcription** request (`POST /v1/listen`), so every scored
file also pays for a transcription — and consequently there is no separate
diarization price line. Diarization is included in the pre-recorded base rate;
only the *streaming* tab adds a per-minute diarization surcharge, which is why
this adapter uses the pre-recorded endpoint exclusively. At the rate this
benchmark was budgeted against (0.258 $/h, German on the same tier as English),
the 120-file / ~18.4 h CALLHOME-de set is roughly **4.75 $** per full sweep.
Billing is per second, so a corpus of many short files pays no rounding premium.
Check <https://deepgram.com/pricing> before you rely on that figure — a vendor
price is not a reproducible constant.

### Two models run, so two things are pinned

`model` is the ASR model whose word timings the turns are folded from;
`diarize_model` is the diarizer itself. Deepgram versions its diarizers
explicitly (`v1` / `v2`, with `latest` resolving to the newest GA batch model),
so `DiarizerSpec.revision` carries the `diarize_model` **version** — that is this
lane's analogue of an HF revision hash. Neither is set to a floating alias:
`nova-3-general` rather than `nova-3`, `v2` rather than `latest`. The vendor
echoes `metadata.diarize_info` (`model_uuid` + `arch`) on every request where a
diarizer actually ran; the adapter keeps it in `DiarizeResult.raw`, so the pin is
evidenced rather than asserted, and a response *without* that block is raised on
rather than scored as a one-speaker file.

### It is also the first caller of the shared word→turn aggregator

Deepgram returns speaker-labelled **words**, not turns. The folding into turns
lives once, in `raven_diar/adapters/aggregate.py`, and every hosted adapter calls
it with the shared `DEFAULT_GAP_MERGE_S`. That is not tidiness: if each provider
folded its own way, a DER *difference* between two providers would partly measure
our two folding rules instead of the two models.

## Install

```bash
uv sync --extra dev --extra diar        # pyannote lane: torch + pyannote.audio
uv sync --extra dev --extra sortformer  # sortformer lane: nemo_toolkit[asr] + torch
uv sync --extra dev --extra diar-hosted # hosted lane: httpx + loader, no torch
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

The hosted lane runs the same way, with its own extra and no GPU — but smoke it
on a handful of files first, because every file is billed:

```bash
make reproduce METRIC=der DATASET=callhome-de MODEL=deepgram-nova-3 \
  EXTRA=diar-hosted LIMIT=3
```

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
