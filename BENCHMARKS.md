# Benchmarks

> Version-stamped results. Each table states the exact commit + scoring contract
> that produced it, and links the command to reproduce it. Empty until the dataset
> runners (Etappe 4/5) land — the structure is fixed now so numbers only ever get
> *added* under a known contract, never asserted without one.

## Scoring contract

See [`benchmark.config.yaml`](./benchmark.config.yaml). Every metric declared
there has an implementation in `raven_eval_core` and vice versa —
`tests/test_metric_contract.py` fails the build in both directions, the metric-side
twin of the dataset-list guard in `tests/test_dataset_contract.py`. DER reported at both
`collar=0.0, skip_overlap=False` (pyannote/DIHARD "full") and `collar=0.25`
(NIST/CALLHOME classic). Automatic VAD, Hungarian speaker mapping. Primary scorer
`pyannote.metrics`. A cross-check against `dscore` (nryant/dscore) is wired as an
**optional** `make dscore-check` (see `raven_diar/dscore_check.py`): it asserts
dscore's DER agrees with `pyannote.metrics` on the same RTTMs within 0.5 pp. It is
gated on a local dscore checkout (dscore is a script repo, not a pip package, so
it can't be pinned as a normal dependency) and therefore is **not** in CI — it
skips cleanly when `DSCORE_DIR` is unset.

## BLEU — translation-shaped corpora

Reported for datasets whose reference is a *translation*, not a transcript: Swiss
German is spoken in dialect and written down in standard German, so WER charges
every legitimate lexical or word-order choice as an error. Those corpora are
scored **bleu+wer**, with BLEU as the headline of the pair. Scale is 0–100 and
**higher is better** — the opposite direction from WER and DER, so never put them
in one column.

"BLEU" alone is not a number. Every convention that moves it is pinned in
[`benchmark.config.yaml`](./benchmark.config.yaml) → `bleu.variants`, mirrored by
the constants in `raven_eval_core/bleu.py`, and asserted equal by
`tests/test_metric_contract.py`:

| convention | value | why |
|---|---|---|
| scorer | `sacrebleu` | pins tokenization and emits a signature; `nltk` scores whatever tokens the caller hands it |
| tokenizer | `13a` | mteval-v13a, the WMT default |
| case | sensitive (`case:mixed`) | German capitalizes nouns |
| smoothing | `exp` | pinned, not inherited from a library default |
| n-gram order | 4, `effective_order=false` | the standard corpus convention |
| aggregation | **corpus** | Σ n-gram stats over the subset, never a mean of per-sentence BLEUs |

Every published BLEU carries its sacrebleu signature, e.g.
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`. That string, not the
word "BLEU", is what makes the number comparable to somebody else's (Post 2018,
*A Call for Clarity in Reporting BLEU Scores*). A per-sentence BLEU exists in
`raven_eval_core.bleu.sentence_bleu_diagnostic` — computed under
`effective_order=True`, i.e. a *different* convention — and is a diagnostic for
finding bad rows, never a published number.

**Rows: none yet.** The metric and its Tier-1 re-score path are wired and covered
by CI; the dialect corpora land separately. `artifacts/_demo_bleu/` proves the
mechanism today and is **not** a Raven product number — on its eight rows the same
output scores 14.29 % WER and 72.24 BLEU, which is precisely the mismeasurement
this metric exists to avoid.

## Tier-1 re-score (how a WER row becomes reproducible)

Every published WER number is backed by the exact per-utterance model output it
was computed from, committed under `artifacts/<run>/<model>/`:

- `predictions_<subset>.jsonl` — one line per utterance:
  `{"reference","prediction","latency_s"}` (the format Raven's eval runner
  writes, `german_asr/runner.py`).
- `expected.json` — `{"<subset>": {"wer_pct": <float>, "cer_pct": <float>}}`,
  plus an optional `"bleu"` + `"bleu_signature"` on translation-shaped subsets.

`make verify` (→ `scripts/verify.py`) re-scores every `predictions_*.jsonl`
with our own scorer and asserts each subset matches `expected.json` within
**±0.05 pp**. CI runs it on every push, so a table row cannot drift from its
committed data. The scorer mirrors german-asr's flozi-strict pipeline exactly
(`unidecode` + `alpha2digit` + jiwer `wer_standardize_contiguous`, **corpus**
aggregation = total word edits / total reference words, raw-text CER) — so the
same predictions in reproduce the same published `wer_pct` out. Reconciliation
notes (why we do **not** call `raven_eval_core.normalize_strict_de` here) are in
the header of `scripts/verify.py`.

BLEU uses the same file and the same ±0.05 tolerance: a subset whose expected
entry carries a `"bleu"` key is additionally re-scored with
`raven_eval_core.bleu.corpus_bleu_score` on the **raw** text (BLEU's tokenizer is
the declared normalization; flozi's punctuation/case stripping would destroy what a
translation reference asks for). The key is optional, so plain transcription
subsets are unaffected. `sacrebleu` is a base dependency, not behind an extra,
exactly so this stays a no-GPU no-network path.

Committed `artifacts/_demo/` and `artifacts/_demo_bleu/` dirs prove the mechanism
end-to-end; they are self-consistent fixtures, **not** Raven product numbers.

## WER — German ASR (public datasets)

**Harness wired (Etappe 4).** Datasets, all of them loadable from this repo
(`raven_asr/datasets/`, registered in `raven_asr.config.WER_DATASETS`):

| id | source | config / split | license |
|----|--------|----------------|---------|
| `german-mixed` | `flozi00/asr-german-mixed-evals` | subsets `Tuda-De`, `common_voice_19_0`, `multilingual_librispeech` | CC-BY-4.0 / CC0 per subset (attribution in `/NOTICE`) |
| `fleurs` | `google/fleurs` | `de_de` / `test` | CC-BY-4.0 |
| `mls-de` | `facebook/multilingual_librispeech` | `german` / `test` | CC-BY-4.0 |
| `voxpopuli-de` | `facebook/voxpopuli` | `de` / `test` | CC0-1.0 |
| `spc-test` | `i4ds/SPC_test` | `test` | MIT — *inferred* from upstream FHNW SPC; the HF card carries no license tag (`/NOTICE`) |
| `fhnw-all-dialects` | FHNW i4ds, SwissText 2021 task 3 ([Dropbox distribution](https://www.dropbox.com/s/rfmjqkdjox7xstq/clickworker_collection_1.zip?dl=1)) | `test` (public half) | MIT (`/NOTICE`) |

The last two are Swiss German dialect corpora and are **translation-shaped** —
Swiss German spoken, Standard German written — so they declare
`metric = "bleu+wer"`. Each is reported on its own: there is no aggregate
spanning dialects, and no dialect number feeds any overall average. BLEU itself
is not implemented here yet; until it lands these two carry a WER number only,
which is a floor, not the figure to compare on.

They are also the two datasets that are *not* acquired through a pinned Hugging
Face revision. They are loose files behind a URL, so the pin is an explicit
sha256 verified on every acquisition
(`raven_asr/datasets/local_archive.py`). `fhnw-all-dialects` is distributed via a
Dropbox share link: citable and checksummed, but with no version history and no
institutional commitment, so it is ranked `durability = "vendor"` and carried as
a tracked liability — a durable mirror (self-hosting, or a Zenodo deposit with a
DOI) is **outstanding**. See `/NOTICE` for the full statement.

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

> VoxConverse **test** DER@0.0 = 11.15 % vs pyannote's published 11.2 % (Δ 0.05 pp)
> — a direct, un-caveated reproduction. Dev (7.17 %) is the easier split. For
> CALLHOME-de the comparable column is **collar 0.25** (16.08 %, the ETH-paper
> protocol), which lands between pyannote 3.1 and pyannoteAI (see above). AMI
> **test** DER@0.0 = 17.05 % vs pyannote's published 17.0 % for AMI (IHM)
> (Δ 0.05 pp) — the second direct reproduction.

> Public-dataset DER is **not** Raven's private-meeting DER — it is the externally
> checkable proxy anyone can reproduce, and it will differ from the internal number.

> **Note on Raven's internal DER numbers.** The DER values Raven cites internally
> (e.g. community-1 ~8% vs cloud ~15%) were measured on Raven's *private* meeting
> corpus, which cannot be published (consent). Those are a real-data datapoint, not
> an externally checkable claim. The numbers in *this* table are measured on the
> public datasets above and will differ — these are the ones anyone can reproduce.
