# Tier-2 DER reproduce — HF token, gated model & GPU (the price of a real DER)

Tier-1 (`make verify`) re-scores committed RTTMs and needs **nothing** — no GPU,
no token. Tier-2 (`make reproduce METRIC=der`) actually runs a diarizer over
public audio, so it needs three things. This is the honest cost of a from-audio
DER reproduction — documented here, not hidden.

## The three requirements

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

## Install

```bash
uv sync --extra dev --extra diar     # torch + pyannote.audio + datasets + soundfile
```

The `diar` extra is heavy (torch) and deliberately isolated: Tier-1 verify never
imports it, so `make verify` stays light and GPU-free.

## Pin the model & dataset revisions (reproducibility)

Pin **both** via the Makefile for a byte-reproducible run. The committed VoxConverse
numbers used exactly these commits:

```bash
make reproduce METRIC=der DATASET=voxconverse-test MODEL=pyannote-community-1 \
  MODEL_REV=3533c8cf8e369892e6b79ff1bf80f7b0286a54ee \
  DATASET_REV=24bf60be297701cd7e4ef18550c6d390c1b87365
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
| `ami`          | prepared RTTMs from `pyannote/AMI-diarization-setup` (pinned commit) | that repo's `download_ami.sh` (Mix-Headset) → `data/diar/ami/audio/` |

`prepare()` downloads the gold labels and prints the exact audio step; a file
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
