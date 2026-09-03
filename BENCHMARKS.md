# Benchmarks

> Version-stamped results. Each table states the exact commit + scoring contract
> that produced it, and links the command to reproduce it. Empty until the dataset
> runners (Etappe 4/5) land — the structure is fixed now so numbers only ever get
> *added* under a known contract, never asserted without one.

## Scoring contract

See [`benchmark.config.yaml`](./benchmark.config.yaml). DER reported at both
`collar=0.0, skip_overlap=False` (pyannote/DIHARD "full") and `collar=0.25`
(NIST/CALLHOME classic). Automatic VAD, Hungarian speaker mapping. Primary scorer
`pyannote.metrics`. A cross-check against `dscore` (nryant/dscore) is wired as an
**optional** `make dscore-check` (see `raven_diar/dscore_check.py`): it asserts
dscore's DER agrees with `pyannote.metrics` on the same RTTMs within 0.5 pp. It is
gated on a local dscore checkout (dscore is a script repo, not a pip package, so
it can't be pinned as a normal dependency) and therefore is **not** in CI — it
skips cleanly when `DSCORE_DIR` is unset.

## Tier-1 re-score (how a WER row becomes reproducible)

Every published WER number is backed by the exact per-utterance model output it
was computed from, committed under `artifacts/<run>/<model>/`:

- `predictions_<subset>.jsonl` — one line per utterance:
  `{"reference","prediction","latency_s"}` (the format Raven's eval runner
  writes, `german_asr/runner.py`).
- `expected.json` — `{"<subset>": {"wer_pct": <float>, "cer_pct": <float>}}`.

`make verify` (→ `scripts/verify.py`) re-scores every `predictions_*.jsonl`
with our own scorer and asserts each subset matches `expected.json` within
**±0.05 pp**. CI runs it on every push, so a table row cannot drift from its
committed data. The scorer mirrors german-asr's flozi-strict pipeline exactly
(`unidecode` + `alpha2digit` + jiwer `wer_standardize_contiguous`, **corpus**
aggregation = total word edits / total reference words, raw-text CER) — so the
same predictions in reproduce the same published `wer_pct` out. Reconciliation
notes (why we do **not** call `raven_eval_core.normalize_strict_de` here) are in
the header of `scripts/verify.py`.

A committed `artifacts/_demo/` dir proves the mechanism end-to-end; it is a
self-consistent fixture, **not** a Raven product number.

## WER — German ASR (public datasets)

**Harness wired (Etappe 4).** Datasets, all of them loadable from this repo
(`raven_asr/datasets/`, registered in `raven_asr.config.WER_DATASETS`):

| id | source | config / split | license |
|----|--------|----------------|---------|
| `german-mixed` | `flozi00/asr-german-mixed-evals` | subsets `Tuda-De`, `common_voice_19_0`, `multilingual_librispeech` | CC-BY-4.0 / CC0 per subset (attribution in `/NOTICE`) |
| `fleurs` | `google/fleurs` | `de_de` / `test` | CC-BY-4.0 |
| `mls-de` | `facebook/multilingual_librispeech` | `german` / `test` | CC-BY-4.0 |
| `voxpopuli-de` | `facebook/voxpopuli` | `de` / `test` | CC0-1.0 |

Revisions are pinned in `raven_asr.config.WER_DATASETS`; the harness fetches the
audio itself. Numbers appear here once a model run is **promoted**
into `artifacts/` and committed — at which point `make verify` and CI re-score it
on every push. Full re-run from raw audio (your keys/GPU,
[`docs/TIER2-KEYS.md`](./docs/TIER2-KEYS.md)):

```bash
make reproduce METRIC=wer DATASET=common_voice_19_0 MODEL=modal/parakeet
make promote   RESULTS=results/reproduce/modal-parakeet RUN=$(date +%F)
make verify
```

WER is the **flozi-strict corpus** WER (`raven_eval_core.flozi_wer`): total word
edits / total reference words after flozi-canonical normalization — comparable to
the `flozi00/asr-german-mixed-evals` published table, not to a mean-of-utterance
WER. CER is on raw text.

| model | dataset | WER strict % | CER % | n | run | flozi ref WER |
|-------|---------|------------:|------:|--:|-----|--------------:|
| primeline/parakeet-primeline | Tuda-De | 4.02 | 2.75 | 414 | [2026-07-30](./artifacts/2026-07-30-parakeet-primeline/primeline-parakeet/) | 4.11 |
| primeline/parakeet-primeline | multilingual_librispeech | 3.04 | 1.90 | 3996 | [2026-07-30](./artifacts/2026-07-30-parakeet-primeline/primeline-parakeet/) | 2.60 |
| primeline/parakeet-primeline | common_voice_19_0 | 2.58 | 0.85 | 5389 | [2026-07-30](./artifacts/2026-07-30-parakeet-primeline/primeline-parakeet/) | 3.03 |

Re-score any row with `make verify` (Tier-1, no GPU) — it recomputes these from
the committed `predictions_*.jsonl` and asserts they match within ±0.05 pp.
`flozi ref WER` is the published anchor from the dataset card (sanity guard, not
a re-scored number). The run's `predictions_*.jsonl` were produced by the
`primeline/parakeet-primeline` model (NeMo, `2_95_WER.nemo`, fp16) served
locally on a dual-RTX-3090 box via its OpenAI-compatible `/v1/audio/transcriptions`
endpoint; the harness pulled the public audio from Hugging Face and scored on the
host. Latency is therefore environment-specific (Tier-3) and deliberately not
tabled here.



> Raven's internal private-meeting WER/DER numbers are measured on a corpus that
> cannot be published (consent) and are reported separately (Tier 3) — the rows
> here are the public-dataset numbers anyone can re-score. The Tier-1 DER
> artifacts (committed RTTMs) land with Etappe 5.

## Tier-1 DER re-score (how a DER row becomes reproducible)

Every published DER number is backed by the exact RTTMs it was computed from,
committed under `artifacts/<run>/<model>/`:

- `gold/<dataset>/<file>.rttm` — the reference (gold) diarization.
- `hyp/<dataset>/<file>.rttm` — the diarizer's hypothesis for the same file.
- `expected.json` — `{"<dataset>": {"der_full", "der_classic", "miss", "fa", "conf"}}`
  (percent; `miss + fa + conf == der_full` by construction).

`make verify` re-loads gold+hyp and recomputes corpus DER at **both** collars with
`raven_eval_core.der` (pyannote.metrics — no torch, no GPU, no gated model),
asserting each field matches `expected.json` within **±0.05 pp**. Corpus DER is
the NIST-correct `Σ(miss+fa+conf) / Σ(total)` over files, never a mean of per-file
DERs. A committed `artifacts/_demo_der/` dir proves the mechanism end-to-end; it is
a self-consistent fixture, **not** a Raven product number.

## DER — speaker diarization (public datasets)

**Reproduced (Etappe 5).** `pyannote-community-1` was run on VoxConverse, CALLHOME-de and AMI; the
per-file gold + hypothesis RTTMs and `expected.json` are committed under
`artifacts/`, so `make verify` (and CI) re-score every number below on every push
with **no GPU and no gated model**. The from-audio run needs an HF token, the
gated model license, and a GPU ([`docs/TIER2-DER-KEYS.md`](./docs/TIER2-DER-KEYS.md));
the re-score does not.

**Independent reproduction of the vendor benchmark.** On VoxConverse **test** at
`collar=0.0, skip_overlap=False` — the exact metric pyannote's model card reports
("no forgiveness collar, nor skipping overlapping speech") — we measure
**11.15 % DER** against pyannote's own published **11.2 %**: a **0.05 pp** delta,
computed here with our scorer on committed RTTMs. The `v0.3` labels pyannote
benchmarked are the VoxConverse repo's `master` branch (the project has no git
tags — `v0.3` is a dataset-version label), pinned here at commit `24bf60be`.
Model `pyannote/speaker-diarization-community-1` @ `3533c8cf`. Determinism: the
full set was run twice, DER identical to 15 significant digits.

**The German anchor (CALLHOME-de).** On the German telephone set
(`talkbank/callhome` config `deu`, n=120) community-1 scores **16.08 % DER** at
`collar=0.25, skip_overlap=False` — the protocol the ETH diarization benchmark
(arXiv 2509.26177) uses for its per-language German column. There it sits
**between pyannote 3.1 (19.0 %) and the commercial pyannoteAI/precision-2
(8.3 %)** — exactly where an open model belongs. Unlike VoxConverse there is no
published community-1-on-German-CALLHOME number to match exactly, so this is a
*sanity-banded* reproducible measurement, corroborated by the reference overlap
fraction (10.8 %, paper ≈ 12.6 %) and the miss-dominated error the literature
predicts. **Protocol matters:** the same run reads 20.58 % at collar 0.0 vs
16.08 % at 0.25 — 4.5 pp on convention alone, so a CALLHOME DER without a stated
collar + overlap rule is noise.

**The first hosted diarizer (AssemblyAI).** On the same German telephone set,
AssemblyAI's Universal-3.5 Pro with `speaker_labels` scores **21.74 % DER** at
`collar=0.25` against community-1's **16.08 %** under the identical protocol,
scorer and gold RTTMs — 5.7 pp behind the open local model on German telephone
speech. The error is **miss-dominated** (21.59 miss vs 3.28 false alarm): the
hypothesis is derived from a *transcript*, so speech the ASR does not transcribe
— backchannels, crosstalk, laughter — produces no turn at all and is scored as
missed speech. That is a structural property of diarization-as-a-transcription-
flag, not a tuning gap, and it is the reason both collars are published here:
the same run reads 28.69 % at collar 0.0.

Two things make this row comparable rather than merely adjacent. Both providers'
turns come from the **same shared aggregator**
(`raven_diar/adapters/aggregate.py`, 0.5 s gap merge), so the difference measures
the models and not two vendors' folding rules. And the model is **pinned by
alias**: AssemblyAI publishes no immutable version, its default is a two-model
fallback chain, so the adapter sends `speech_models: ["universal-3-5-pro"]` alone
and fails the file if the response's `speech_model_used` names anything else. An
alias can still move under a re-train — that limitation is the vendor's, and it
is stated rather than hidden. Cost of this row: 18.43 h of audio at 0.23 $/h
(0.21 model + 0.02 diarization add-on) ≈ **4.24 $**.

**The meeting regime (AMI).** On the AMI test split (16 four-speaker meetings,
Mix-Headset = the IHM condition, gold = the `only_words` RTTMs of
`pyannote/AMI-diarization-setup` @ `67c2d539`) community-1 scores **17.05 % DER**
at `collar=0.0, skip_overlap=False` — against pyannote's own published
**17.0 %** for "AMI (IHM)" under the same protocol: a **0.05 pp** delta, the
second vendor number this scorer reproduces exactly. At collar 0.25 the same
run reads 13.10 %. Audio for this set is fetched by `prepare()` from the
Edinburgh AMI mirror (CC-BY-4.0), so the run below is fully self-contained.

Reproduce (your HF token + gated license + GPU + shared FFmpeg libs):

```bash
make reproduce METRIC=der DATASET=voxconverse-test MODEL=pyannote-community-1 \
  MODEL_REV=3533c8cf8e369892e6b79ff1bf80f7b0286a54ee \
  DATASET_REV=24bf60be297701cd7e4ef18550c6d390c1b87365
make reproduce METRIC=der DATASET=ami MODEL=pyannote-community-1   # revisions pinned in raven_diar/config.py
make promote   METRIC=der RESULTS=results/reproduce-der/pyannote-community-1 RUN=$(date +%F)
make verify
```

| model | dataset | DER (collar 0.0) | DER (collar 0.25) | miss | FA | conf | n | run |
|-------|---------|-----------------:|------------------:|-----:|---:|-----:|--:|-----|
| pyannote-community-1 | voxconverse (dev) | 7.17 | 5.00 | 2.33 | 2.26 | 2.58 | 216 | [2026-07-30](./artifacts/2026-07-30/pyannote-community-1/) |
| pyannote-community-1 | voxconverse (**test**) | **11.15** | 8.41 | 3.38 | 4.08 | 3.68 | 232 | [2026-07-31](./artifacts/2026-07-31-voxconverse-test/pyannote-community-1/) |
| pyannote-community-1 | callhome-de (German, telephone) | 20.58 | **16.08** | 13.46 | 3.54 | 3.57 | 120 | [2026-07-31](./artifacts/2026-07-31-callhome-de/pyannote-community-1/) |
| pyannote-community-1 | ami (test, 4-speaker meetings, IHM) | **17.05** | 13.10 | 9.52 | 3.58 | 3.95 | 16 | [2026-09-02](./artifacts/2026-09-02-ami/pyannote-community-1/) |
| assemblyai-universal-3-5-pro | callhome-de (German, telephone) | 28.69 | **21.74** | 21.59 | 3.28 | 3.82 | 120 | [2026-09-03](./artifacts/2026-09-03-callhome-de-assemblyai/assemblyai-universal-3-5-pro/) |

> VoxConverse **test** DER@0.0 = 11.15 % vs pyannote's published 11.2 % (Δ 0.05 pp)
> — a direct, un-caveated reproduction. Dev (7.17 %) is the easier split. For
> CALLHOME-de the comparable column is **collar 0.25** (16.08 %, the ETH-paper
> protocol), which lands between pyannote 3.1 and pyannoteAI (see above). AMI
> **test** DER@0.0 = 17.05 % vs pyannote's published 17.0 % for AMI (IHM)
> (Δ 0.05 pp) — the second direct reproduction. On CALLHOME-de the hosted
> AssemblyAI row (21.74 % at collar 0.25) sits 5.7 pp behind community-1
> (16.08 %) under the same protocol — a measurement on one public set, not a
> verdict on either product.

> Public-dataset DER is **not** Raven's private-meeting DER — it is the externally
> checkable proxy anyone can reproduce, and it will differ from the internal number.

> **Note on Raven's internal DER numbers.** The DER values Raven cites internally
> (e.g. community-1 ~8% vs cloud ~15%) were measured on Raven's *private* meeting
> corpus, which cannot be published (consent). Those are a real-data datapoint, not
> an externally checkable claim. The numbers in *this* table are measured on the
> public datasets above and will differ — these are the ones anyone can reproduce.
