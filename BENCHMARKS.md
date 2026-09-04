# Benchmarks

> Version-stamped results. Each table states the exact commit + scoring contract
> that produced it, and links the command to reproduce it. Numbers only ever get
> *added* under a known contract, never asserted without one — and every row on
> this page resolves to a committed artifact that `make verify` re-scores on
> every push, in both directions (`tests/test_published_table.py`).

## Scoring contract

See [`benchmark.config.yaml`](./benchmark.config.yaml). Every metric declared
there has an implementation in `raven_eval_core` and vice versa —
`tests/test_metric_contract.py` fails the build in both directions, the metric-side
twin of the dataset-list guard in `tests/test_dataset_contract.py`. DER reported at both
`collar=0.0, skip_overlap=False` (pyannote/DIHARD "full") and `collar=0.25`
(NIST/CALLHOME classic), each with its **own** miss/FA/confusion decomposition,
aggregated **corpus-wide** (`Σerr/Σtotal`) with the unweighted file-mean reported
alongside. Uncertainty on every published DER is a seeded 10 000-resample
bootstrap over files. Automatic VAD, Hungarian speaker mapping. Primary scorer
`pyannote.metrics`. The collars, the folding threshold, the aggregation and the
bootstrap settings are all read from the contract file by
`tests/test_diar_harness.py` — a test that restated them as literals would pass
just as happily against a contract that had changed underneath it.

**Cross-checked against a second implementation.** `dscore` (nryant/dscore) wraps
NIST's `md-eval`, the reference DER. On **2026-09-04 that cross-check was
executed** for the first time: 15 (gold, hypothesis) pairs drawn from four
diarizers and three datasets, both collars, 30 comparisons, worst disagreement
**0.030 pp** — inside even the ±0.05 pp tolerance this repo reproduces its own
numbers to. Two implementations, one answer.

Worth naming what that run cost, because the check had been advertised as wired
for weeks without ever having been executed, and each of these alone would have
failed the first attempt: it passed no collar flag at all (comparing our
collar-0.25 DER against dscore's collar-0.0 default), it named a flag
(`--score_overlaps`) dscore does not have, it pinned a commit (`f2d33d3`) that
does not exist in that repository, and it handed relative paths to a subprocess
run from another directory. A cross-check that has not been run is not a
cross-check. The collar conversion the run established — `dscore --collar X`
equals `pyannote collar 2X`, because md-eval applies the collar per side while
pyannote centres a window of that total width — is now a named constant with a
test that needs no checkout.

`make dscore-check` stays **optional** and outside CI: dscore is a script
repository, not a pip package, so it cannot be pinned as a normal dependency. It
is gated on a local checkout and skips cleanly when `DSCORE_DIR` is unset.

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
spanning dialects, and no dialect number feeds any overall average. BLEU is
implemented (`raven_eval_core/bleu.py`, signature pinned in
`benchmark.config.yaml`), so where a run reports both, BLEU is the figure to
compare on and the WER beside it is a floor.

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
> here are the public-dataset numbers anyone can re-score.

## Tier-1 DER re-score (how a DER row becomes reproducible)

Every published DER number is backed by the exact RTTMs it was computed from,
committed under `artifacts/<run>/<model>/`:

- `gold/<dataset>/<file>.rttm` — the reference (gold) diarization.
- `hyp/<dataset>/<file>.rttm` — the diarizer's hypothesis for the same file.
- `expected.json` — every published scalar per dataset: `der_full` /
  `der_classic` with **each collar's own** `miss`/`fa`/`conf` decomposition, and
  `der_full_filemean` / `der_classic_filemean`, the same files under the other
  aggregation convention.

`make verify` re-loads gold+hyp and recomputes all of them with
`raven_diar.score` (→ `raven_eval_core.der` → pyannote.metrics — no torch, no
GPU, no gated model), asserting each field matches `expected.json` within
**±0.05 pp**. A committed `artifacts/_demo_der/` dir proves the mechanism
end-to-end; it is a self-consistent fixture, **not** a Raven product number.

**Two conventions, both published.** A DER is not a number until its
*aggregation* is named, the same way it is not a number until its collar is
named:

| aggregation | definition | who reports it |
|---|---|---|
| **corpus** (primary) | `Σ(miss+fa+conf) / Σ(scored reference speech)` over all files — a 50-minute meeting outweighs a 30-second clip | CALLHOME, DIHARD, the pyannote model cards |
| **file-mean** | unweighted mean of per-file DERs — every file counts once | the ETH diarization benchmark ([arXiv 2509.26177](https://arxiv.org/abs/2509.26177), Table 3 caption) |

The two are 0.334 pp apart on our German CALLHOME row and 2.4 pp apart on
VoxConverse dev, where short clips dominate the count but not the speech. Both
are pinned in [`benchmark.config.yaml`](./benchmark.config.yaml) →
`der.aggregation`, mirrored by `raven_diar.config` and asserted equal by
`tests/test_diar_harness.py`. The headline table prints the corpus figure and
carries the file-mean beside it.

**Past the corpus scalar.** `expected.json` is an aggregate; the per-file scores
underneath it are what make a claim *about* that aggregate checkable.
`make analyse ARTIFACT=artifacts/<run>/<model>` recomputes, from the same
committed RTTMs and on a laptop, five things a single DER cannot say: a
seeded bootstrap **confidence interval** over files, DER **by reference speaker
count**, the reference **overlap fraction**, speaker-aware **boundary offsets**
with the segment-length regime the missed speech sits in, and the **turn-folding
residue** — how far this row would move if the shared hosted-adapter folding were
applied to it too. Every paragraph below that makes a claim of precision,
difficulty, error *kind* or protocol equality quotes that command's output.

## DER — speaker diarization (public datasets)

**Reproduced (Etappe 5).** `pyannote-community-1` was run on VoxConverse,
CALLHOME-de and AMI; the per-file gold + hypothesis RTTMs and `expected.json`
are committed under `artifacts/`, so `make verify` (and CI) re-score every number
below on every push with **no GPU and no gated model**. The from-audio run needs
an HF token, the gated model license, and a GPU
([`docs/TIER2-DER-KEYS.md`](./docs/TIER2-DER-KEYS.md)); the re-score does not.

**Independent reproduction of the vendor benchmark — and how precise it is.** On
VoxConverse **test** at `collar=0.0, skip_overlap=False` — the exact metric
pyannote's model card reports ("no forgiveness collar, nor skipping overlapping
speech") — we measure **11.15 % DER** against pyannote's own published
**11.2 %**. The `v0.3` labels pyannote benchmarked are the VoxConverse repo's
`master` branch (the project has no git tags — `v0.3` is a dataset-version
label), pinned here at commit `24bf60be`. Model
`pyannote/speaker-diarization-community-1` @ `3533c8cf`.

That 0.05 pp gap is **not** evidence of agreement to 0.05 pp, and this page used
to leave that impression. Bootstrapping the same 232 files gives a 95 % interval
of **[10.12, 12.23]**, a half-width of ±1.06 pp — twenty times the gap. What the
reproduction shows is that our scorer and pyannote's land in the same place on
the same data; the *precision* of either number is bounded by the corpus, not by
the agreement. The same holds on AMI: **17.05 %** against pyannote's **17.0 %**,
with an interval of **[14.96, 19.07]**, ±2.06 pp on 16 meetings. Determinism is a
separate property and holds separately: the full VoxConverse set was run twice,
DER identical to 15 significant digits.

**The German anchor (CALLHOME-de).** On the German telephone set
(`talkbank/callhome` config `deu`, n=120) community-1 scores **16.08 % DER** at
`collar=0.25, skip_overlap=False`, 95 % CI **[14.97, 17.27]** — the protocol the
ETH diarization benchmark uses for its per-language German column. There it sits
**between pyannote 3.1 (19.0 %) and the commercial pyannoteAI/precision-2
(8.3 %)** — exactly where an open model belongs. Unlike VoxConverse there is no
published community-1-on-German-CALLHOME number to match exactly, so this is a
*sanity-banded* reproducible measurement. **Protocol matters:** the same run
reads 20.58 % at collar 0.0 vs 16.08 % at 0.25 — 4.5 pp on convention alone, so a
CALLHOME DER without a stated collar + overlap rule is noise.

**What the corpus itself looks like.** The reference overlap of these 120 files
is **12.01 % of speech** (overlapped seconds / seconds where anyone speaks) and
**11.90 % ± 5.44** as a per-file mean, over 18.4344 h of recording. The ETH paper
reports 12.58 % ± 7.31 for CALLHOME — as a per-file mean, and across all five of
its languages rather than German alone, so 11.90 ± 5.44 is the figure that
compares. It also states 18.4 h for its German split, which is our 18.4344 h:
same scope, same corpus, independently arrived at.

Two things about the number itself. First, it depends entirely on the
denominator: the same overlap is 10.72 % of Σ per-speaker time and 11.09 % of
wall clock, and "the overlap fraction" names none of the three. All three are
printed by `make analyse` and none of them is an assertion. Second, this page
previously published **10.8 %**, which no denominator reproduces — the closest,
overlap-over-speaker-time, is 0.1 pp away and the earlier figure's derivation is
not recorded anywhere in the repository. It is withdrawn and replaced by the
numbers above, which any reader can recompute.

**The first hosted diarizer (AssemblyAI).** On the same German telephone set,
AssemblyAI's Universal-3.5 Pro with `speaker_labels` scores **21.74 % DER** at
`collar=0.25` (CI [20.66, 22.90]) against community-1's **16.08 %** under the
identical protocol, scorer and gold RTTMs. Paired over the 120 files the gap is
**+5.66 pp, 95 % CI [+4.52, +6.73]** — an interval clear of zero, so the ordering
is a finding and not a coin flip on this corpus.

The row is **miss-dominated**: at collar 0.25, miss 17.33 against false alarm
1.66 and confusion 2.75. This page used to explain that as the ASR skipping
backchannels, crosstalk and laughter — speech too short to be transcribed and so
never turned into a turn. **Our own artifacts do not support that explanation.**
Splitting the uncovered reference speech by the length of the reference segment
it sits in gives 1 504 s in segments shorter than 0.5 s against 15 935 s in
longer ones: **91 % of the missed speech is in long segments**, not short ones.
Whatever the hosted row is losing, it is mostly not backchannels. What the
instrument does confirm is a boundary difference: the median offset between a
reference speaker-turn boundary and the nearest boundary of the same mapped
speaker is **1 080 ms** for AssemblyAI and **92 ms** for community-1, with 28 %
against 68 % of boundaries inside 250 ms. Deepgram sits between them (755 ms,
34 %). That is the structural property of diarization-derived-from-a-transcript
worth stating, and it is measured rather than inferred.

Two things make this row comparable rather than merely adjacent, and one thing
qualifies it.

Both *hosted* providers' turns come from the **same shared aggregator**
(`raven_diar/adapters/aggregate.py`, 0.5 s gap merge), so a Deepgram-vs-AssemblyAI
difference measures the models and not two vendors' folding rules. That is what
the aggregator exists for.

It does not extend to hosted-vs-local, and this page used to imply that it did.
A hosted API returns labelled *words*; turns must be reconstructed before
anything can be scored, and the aggregator is that reconstruction. A local
diarizer already emits turns, so folding them again would be a second opinion we
impose rather than one we cannot avoid — the local adapters are deliberately
exempt. So the AssemblyAI-vs-community-1 comparison is not "identical protocol";
it is identical gold, identical scorer, identical collars, and one step that
applies to one side because only one side needs it.

`make analyse` now measures that residue rather than leaving it to the reader:
each row reports how far it would move if the shared folding were applied to it
too. The hosted rows are fixed points and read **+0.000 pp** — folding is
idempotent on already-folded turns, which is the direct evidence that the step is
reconstruction and not tuning. Local rows move by between −0.19 and −1.77 pp, in
both directions and differently per corpus, which is also why uniform folding is
not simply applied to everything: it would move six published numbers, and it
would break the two reproductions this page rests on (community-1 on VoxConverse
test goes 11.15 → 10.95 against pyannote's published 11.2). Comparability with
the field's own published numbers is worth more than a sentence that is literally
true.

The model is **pinned by alias**: AssemblyAI publishes no immutable version, its default is a two-model
fallback chain, so the adapter sends `speech_models: ["universal-3-5-pro"]` alone
and fails the file if the response's `speech_model_used` names anything else. An
alias can still move under a re-train — that limitation is the vendor's, and it
is stated rather than hidden. Cost of this row: the gold RTTMs span **18.4344 h**
of recording (recomputed from the committed references by `make analyse`, so that
half is an artifact) at AssemblyAI's list 0.23 $/h (0.21 model + 0.02
diarization add-on) ≈ **4.24 $** — a list-price calculation, not a measurement,
and the only figure on this page that is not re-derivable from the repository.

**The meeting regime (AMI).** On the AMI test split (16 four-speaker meetings,
Mix-Headset = the IHM condition, gold = the `only_words` RTTMs of
`pyannote/AMI-diarization-setup` @ `67c2d539`) community-1 scores **17.05 % DER**
at `collar=0.0, skip_overlap=False` — against pyannote's own published
**17.0 %** for "AMI (IHM)" under the same protocol, with the ±2.06 pp interval
noted above. At collar 0.25 the same run reads 13.10 %. Reference overlap on this
split is 14.58 % of speech, a fifth again more than the telephone set. Audio is
fetched by `prepare()` from the Edinburgh AMI mirror (CC-BY-4.0), so the run
below is fully self-contained.

**The second diarizer (Sortformer 4spk-v1) — a reference row, not a shippable
option.** `nvidia/diar_sortformer_4spk-v1` @ `9f17b10d` is an end-to-end neural
diarizer run locally through NeMo (no HF token, no gated licence). On CALLHOME-de
— the same 120 German telephone files, the same gold, the same scorer — it reads
**17.34 % DER at collar 0.0** and **11.41 % at collar 0.25** (CI [10.19, 12.67]).
Its weights are **CC-BY-NC-4.0 (non-commercial)**, so this row exists to bound
the field, not to be shipped: no commercial deployment may follow from it,
whatever the number says.

**This row is also the external check on our whole measurement chain.** The ETH
benchmark measures the same checkpoint on the same corpus under the same collar
and reports **11.1 %**. We report 11.41 %. Under the unweighted file-mean, our
own committed RTTMs read **11.07 %**, which rounds to their 11.1 %. It is the
same measurement; the 0.3 pp is the aggregation convention and nothing else.

That their language table uses the file-mean is not only read off the Table 3
caption ("averaging all samples … and averaging over them") — this row is
evidence for it. Corpus aggregation would have to produce 11.41 to match us, and
their column reads 11.1. Two aggregations, one of which lands on their number.

Ruled out as causes, with numbers: UEM (0.000 pp on this corpus, see below),
sample scope (n=120, 18.4344 h against their stated 18.4 h for the same German
split), and chunking (their 12-minute threshold is above our 9.2-minute mean file
length, so no CALLHOME file is chunked on either side). This is the
reconciliation that made the convention worth pinning in the contract file rather
than leaving implicit.

**The same reconciliation fails for the streaming v2 checkpoint, in our favour —
and the reason is the latency setting.** ETH report 9.6 % for
`diar_streaming_sortformer_4spk-v2` on German CALLHOME; we read **8.98 %** corpus
and **9.07 %** file-mean, 0.5–0.6 pp *better* under either convention. A gap in
our favour is not a reason to be relaxed about it, so it was measured rather than
argued.

Three candidate causes were eliminated first. *Gold preparation*: the gold RTTMs
under all five committed CALLHOME-de artifacts are byte-identical, and the same
gold reconciles the v1 row to their number exactly — a gold defect cannot be
selective about which checkpoint it breaks. *Chunking*: their paper scores two v2
variants, one on 12-minute chunks and one on full audio, and both read 9.6 on
German, because no CALLHOME file is long enough to be chunked. *Which column we
compare to*: for the same reason, the streaming variant and the chunked one are
the same run here.

That left the streaming configuration. A streaming diarizer has no single DER —
it has one per latency setting, and NVIDIA publish four recommended
configurations spanning 30.4 s down to 0.32 s of input-buffer latency. We run the
"very high latency" preset, the highest-quality point on that curve; the paper
describes its model only as "low-latency streaming" and names no configuration.
So the same 120 files were re-run at the lower presets, same checkpoint, same
gold, same scorer:

| streaming preset | input-buffer latency | DER@0.25 corpus | DER@0.25 file-mean |
|---|---:|---:|---:|
| very high latency *(published above)* | 30.4 s | **8.98** | 9.07 |
| high latency | 10.0 s | 9.26 | 9.30 |
| low latency | 1.04 s | 9.82 | 9.97 |

The curve is monotone in the expected direction and **it brackets 9.6**: the ETH
number falls between the 10-second and the 1-second preset, which is exactly
where a configuration described as "low-latency streaming" belongs. The
disagreement was never about gold, chunking or the checkpoint — it is that two
measurements of "the same model" were taken at two points on a latency/quality
trade-off, and only one of them said which.

The practical reading: our 8.98 % is not a better measurement of their number,
it is a different operating point, and it is only available to a product willing
to buffer 30 seconds of audio. Raven's is, because meeting diarization is not
live captioning — but the comparison to any other row on this page holds only
because every row here is measured at one stated configuration.

> These three rows are **Tier-3 diagnostics, not published numbers**: only the
> "very high latency" row has a committed artifact, and it is the one in the
> table. The other two exist to explain a discrepancy and are reproducible with
> `SORTFORMER_LATENCY_PRESET=<name> make reproduce METRIC=der
> DATASET=callhome-de MODEL=sortformer-streaming-4spk-v2 EXTRA=sortformer`. The
> preset a run used is recorded in its `summary.json`, so a run measured at a
> non-default setting cannot be promoted as if it were the shipped one.

**What the German column shows.** Five diarizers on the same 120 CALLHOME-de
files, same gold, same scorer, same two collars. At the classic collar:
pyannote-community-1 **16.08**, deepgram-nova-3 19.31, assemblyai-universal-3-5-pro
21.74, sortformer-4spk-v1 11.41 — and sortformer-streaming-4spk-v2 at **8.98**,
the lowest DER any row on this page reaches on *this dataset* and, unlike v1, one
we could ship: its weights are CC-BY-4.0, not CC-BY-NC-4.0. Paired over the same
files, the streaming checkpoint is **7.10 pp ahead of community-1, CI [+5.95,
+8.23]**, and **2.42 pp ahead of v1, CI [+1.11, +3.74]**; both intervals exclude
zero. (Numbers on different datasets are not comparable, which is why "best on
the page" is the wrong phrase: VoxConverse dev reads 5.00 % for community-1 and
means nothing next to a telephone number.)

**Where each model breaks down.** The corpus number hides the axis the field
actually degrades on. Split by *reference* speaker count, at collar 0.25
(n = 104 / 12 / 4 files at 2 / 3 / 4 speakers):

| model | 2 spk | 3 spk | 4 spk |
|---|---:|---:|---:|
| sortformer-streaming-4spk-v2 | 8.87 | 9.83 | **9.26** |
| sortformer-4spk-v1 | 10.95 | 13.75 | 15.91 |
| pyannote-community-1 | 15.59 | 17.78 | 23.38 |
| deepgram-nova-3 | 18.81 | 20.47 | **28.42** |
| assemblyai-universal-3-5-pro | 21.80 | 20.98 | 22.36 |

The four-speaker column rests on four files and carries no interval worth
printing; read it as a direction, not a measurement. The direction is consistent:
every model except the streaming checkpoint gets worse as speakers are added, and
Deepgram worst — +9.6 pp from two speakers to four. On VoxConverse **test**,
where community-1 has 146 files with five or more speakers, the same split reads
4.56 / 8.11 / 9.29 / 6.41 / **8.76** for 1 / 2 / 3 / 4 / 5+ speakers: the model
Raven runs does **not** fall off a cliff in the many-speaker regime, which is the
regime a meeting product lives in. That matters against the published cap
behaviour of the Sortformer family — NVIDIA report 13.24 % DER at ≤4 speakers
against 42.56 % at ≥5 on DIHARD-III Eval, and the ETH benchmark measures the same
jump — and no row on this page says anything about Sortformer above four
speakers, because the checkpoint is hard-capped there and no such file was
scored.

No winner mark is awarded here. Under ADR-app-0036 a star needs at least two
rows among *shippable* models on the same set, and it may never go to a
non-commercial one. CALLHOME-de now carries four shippable rows, so the bar is
met on that set — but a mark is a product decision and not this document's to
make, and the AMI column below is the reason to hesitate: the same checkpoint
that wins German telephone by 7.1 pp loses meetings by 9.9 pp.

**The offline Sortformer still has no AMI row; the streaming one does.** The
offline `4spk-v1` model attends over the whole recording, so activation memory
grows with the *square* of the duration: measured on one 3090, peak allocation is
1.02 GB at 2 min, 4.95 GB at 6 min, 12.65 GB at 10 min, and OOM at 12 min
(fit: ≈3.5e-5 GB/s²). The **shortest** AMI test meeting is 14 min (≈25 GB) and the
longest is 50 min (≈310 GB) — so the AMI split is out of reach of a 24 GB GPU by
construction, not by configuration, and an 80 GB card would still stop at ~25 min.
Forcing v1's streaming path instead is measurably a different (worse) regime — on
ten CALLHOME files it reads 23.13 % / 16.42 % where the offline model reads
18.94 % / 12.77 % — so no AMI number is published for v1 rather than one published
under a silently different protocol.

The `diar_streaming_sortformer_4spk-v2` checkpoint is NVIDIA's own answer to
exactly that, and it holds: all 16 AMI test meetings completed on the same 24 GB
3090, the 49.5-minute one in 8.2 s of wall clock. A bounded speaker cache
replaces quadratic attention, so GPU memory does not track duration — sampled at
~1.1 GB above idle during a 30-minute meeting, the same order as a 10-minute
telephone call. Meeting-length audio is reachable.

**Reachable is not the same as good.** On AMI the streaming checkpoint reads
25.97 % / 23.03 % against community-1's 17.05 % / 13.10 %. Paired over the same
16 meetings that is **+9.93 pp at collar 0.25, CI [+7.31, +12.60]** — a gap that
clears zero comfortably even on 16 files, on the corpus shape that matters most
to Raven, and this from the model that wins German telephone by 7.1 pp. The error
columns say why: at collar 0.25, miss 18.99 % against community-1's 7.78 %, with
false alarm and confusion both *lower* (1.72 / 2.33 versus 2.35 / 2.97). The
streaming model hears less rather than mishearing — it is more conservative on
meeting speech across every one of these meetings, and the paired interval is
what licenses saying so. Its boundaries are not the problem: median offset 80 ms
against community-1's 207 ms on the same split. That is consistent with NVIDIA's
own note that a newer `4spk-v2.1` checkpoint exists "providing greater robustness
for meeting speech"; measuring that one is the obvious next row, and it is not
measured here.

Both AMI recordings are 4-speaker, inside the checkpoint's hard cap. The cap
still bounds where the model may be used at all: a five-person meeting is outside
what this row says anything about.

**No UEM is declared, and on the German column that costs nothing.** Scoring runs
without an explicit un-partitioned evaluation map, so `pyannote.metrics`
approximates one by the union of the reference and hypothesis extents. Scored
against an explicit reference-extent UEM instead, CALLHOME-de moves by **0.002 pp**
for community-1 and **0.000 pp** for sortformer-v1 — the German column, and with
it the whole ETH comparison, is unaffected. AMI and VoxConverse are not: at collar
0.0 the AMI row reads 16.92 % under a reference UEM against 17.05 % as published,
which turns the 0.05 pp agreement with pyannote's 17.0 % into a 0.08 pp
disagreement *with the sign reversed*. Both sit far inside the ±2.06 pp interval,
so nothing about the reproduction claim changes — but the exact-agreement framing
is convention-dependent and is not repeated above. A UEM spanning the full audio
would be mathematically identical to the current state; only a restrictive
upstream UEM moves anything, and none is present in this repository or in the
pinned upstream revisions.

Reproduce (your HF token + gated license + GPU + shared FFmpeg libs):

```bash
make reproduce METRIC=der DATASET=voxconverse-test MODEL=pyannote-community-1 \
  MODEL_REV=3533c8cf8e369892e6b79ff1bf80f7b0286a54ee \
  DATASET_REV=24bf60be297701cd7e4ef18550c6d390c1b87365
make reproduce METRIC=der DATASET=ami MODEL=pyannote-community-1   # revisions pinned in raven_diar/config.py
make promote   METRIC=der RESULTS=results/reproduce-der/pyannote-community-1 RUN=$(date +%F)
make verify
```

Sortformer needs neither a token nor the gated licence — only its own extra:

```bash
make reproduce METRIC=der DATASET=callhome-de MODEL=sortformer-4spk-v1 EXTRA=sortformer \
  MODEL_REV=9f17b10df44c0a4c8f3c86fbddc9ee2d6ab9ac08
make promote   METRIC=der RESULTS=results/reproduce-der/sortformer-4spk-v1 RUN=$(date +%F)-callhome-de-sortformer
make verify
```

The streaming checkpoint is the same extra and the same two commands; it is the
one that also runs AMI:

```bash
make reproduce METRIC=der DATASET=callhome-de MODEL=sortformer-streaming-4spk-v2 EXTRA=sortformer \
  MODEL_REV=5240a64075176943f677d30fa2171c780229f341
make promote   METRIC=der RESULTS=results/reproduce-der/sortformer-streaming-4spk-v2 RUN=$(date +%F)-callhome-de-sortformer-v2
make reproduce METRIC=der DATASET=ami MODEL=sortformer-streaming-4spk-v2 EXTRA=sortformer \
  MODEL_REV=5240a64075176943f677d30fa2171c780229f341
make promote   METRIC=der RESULTS=results/reproduce-der/sortformer-streaming-4spk-v2 RUN=$(date +%F)-ami-sortformer-v2
make verify
```

Read the table with its column groups: `miss`/`FA`/`conf` sum to the DER **in the
same group**, never across groups. The collar-0.0 decomposition does not add up
to the collar-0.25 DER and is not a breakdown of it.

| model | dataset | DER@0.0 | miss@0.0 | FA@0.0 | conf@0.0 | DER@0.25 | miss@0.25 | FA@0.25 | conf@0.25 | file-mean@0.25 | 95 % CI @0.25 | n | run |
|-------|---------|--------:|---------:|-------:|---------:|---------:|----------:|--------:|----------:|---------------:|---------------|--:|-----|
| pyannote-community-1 | voxconverse (dev) | 7.17 | 2.33 | 2.26 | 2.58 | 5.00 | 1.64 | 1.14 | 2.22 | 7.39 | [4.37, 5.70] | 216 | [2026-07-30](./artifacts/2026-07-30/pyannote-community-1/) |
| pyannote-community-1 | voxconverse (**test**) | **11.15** | 3.38 | 4.08 | 3.68 | 8.41 | 2.64 | 2.60 | 3.17 | 8.89 | [7.59, 9.25] | 232 | [2026-07-31](./artifacts/2026-07-31-voxconverse-test/pyannote-community-1/) |
| pyannote-community-1 | callhome-de (German, telephone) | 20.58 | 13.46 | 3.54 | 3.57 | **16.08** | 11.35 | 1.65 | 3.08 | 15.99 | [14.97, 17.27] | 120 | [2026-07-31](./artifacts/2026-07-31-callhome-de/pyannote-community-1/) |
| pyannote-community-1 | ami (test, 4-speaker meetings, IHM) | **17.05** | 9.52 | 3.58 | 3.95 | 13.10 | 7.78 | 2.35 | 2.97 | 12.94 | [10.96, 15.18] | 16 | [2026-09-02](./artifacts/2026-09-02-ami/pyannote-community-1/) |
| sortformer-4spk-v1 (CC-BY-NC, non-commercial) | callhome-de (German, telephone) | 17.34 | 8.27 | 6.43 | 2.64 | 11.41 | 6.38 | 2.88 | 2.15 | 11.07 | [10.19, 12.67] | 120 | [2026-09-03](./artifacts/2026-09-03-callhome-de-sortformer/sortformer-4spk-v1/) |
| assemblyai-universal-3-5-pro | callhome-de (German, telephone) | 28.69 | 21.59 | 3.28 | 3.82 | **21.74** | 17.33 | 1.66 | 2.75 | 21.47 | [20.66, 22.90] | 120 | [2026-09-03](./artifacts/2026-09-03-callhome-de-assemblyai/assemblyai-universal-3-5-pro/) |
| deepgram-nova-3 | callhome-de (German, telephone) | 26.12 | 17.27 | 4.16 | 4.69 | **19.31** | 14.05 | 1.81 | 3.45 | 19.16 | [18.17, 20.53] | 120 | [2026-09-03](./artifacts/2026-09-03-callhome-de-deepgram/deepgram-nova-3/) |
| sortformer-streaming-4spk-v2 | callhome-de (German, telephone) | 14.82 | 5.94 | 7.20 | 1.68 | **8.98** | 4.77 | 2.93 | 1.28 | 9.07 | [8.11, 9.96] | 120 | [2026-09-03](./artifacts/2026-09-03-callhome-de-sortformer-v2/sortformer-streaming-4spk-v2/) |
| sortformer-streaming-4spk-v2 | ami (test, 4-speaker meetings, IHM) | **25.97** | 20.52 | 2.82 | 2.63 | 23.03 | 18.99 | 1.72 | 2.33 | 23.07 | [19.84, 25.80] | 16 | [2026-09-03](./artifacts/2026-09-03-ami-sortformer-v2/sortformer-streaming-4spk-v2/) |

Every interval is a 10 000-resample percentile bootstrap over **files**, seed
`20260903`, pinned in [`benchmark.config.yaml`](./benchmark.config.yaml) →
`der.uncertainty` — so it is reproducible to the digit, not merely to the
concept. Recompute any row's interval, speaker split, overlap fraction and
boundary distribution with `make analyse ARTIFACT=<the run link>`.

> VoxConverse **test** DER@0.0 = 11.15 % vs pyannote's published 11.2 % and AMI
> **test** DER@0.0 = 17.05 % vs their 17.0 % — two direct reproductions on the
> exact protocol they state. Read both as "the same place on the same data", not
> as agreement to a tenth: the corpora themselves carry ±1.06 pp and ±2.06 pp.
> Dev (7.17 %) is the easier split. For CALLHOME-de the comparable column is
> **collar 0.25** (16.08 %, the ETH-paper protocol), which lands between pyannote
> 3.1 and pyannoteAI. The hosted AssemblyAI row sits 5.66 pp behind community-1
> under the same protocol, CI [+4.52, +6.73] — a measurement on one public set,
> not a verdict on either product.

> The `sortformer-4spk-v1` row is a **non-commercial reference measurement**
> (CC-BY-NC-4.0 weights). It is listed to bound the field on German telephone
> speech; it is not a candidate for anything Raven ships, and no ranking here
> implies otherwise. It carries no AMI entry for the reason stated above.

> The two `sortformer-streaming-4spk-v2` rows are **CC-BY-4.0** and therefore
> shippable — the whole reason to measure this checkpoint rather than only v1.
> Both were produced with one streaming configuration (NVIDIA's "very high
> latency" row: chunk 340, right context 40, FIFO 40, update period 300, speaker
> cache 188, all in 80 ms frames — `very-high-latency` in the adapter, and named
> in each run's `summary.json`), automatic VAD, no oracle speaker count and no
> per-dataset post-processing, so the two numbers are comparable to each other
> and to every other row on this page. Which preset a streaming row was measured
> at is not a detail: see the latency curve above, where it moves German
> CALLHOME by 0.84 pp. NVIDIA's own CALLHOME and DIHARD
> post-processing YAMLs are per-corpus tuning and are deliberately not applied.
> The checkpoint is capped at **4 speakers**; both sets are inside that cap, and
> nothing here says anything about a five-person meeting.

> The v1 memory curve and the forced-streaming comparison above are **Tier-3
> diagnostics, not published numbers**: they were measured on ten files and on one
> particular GPU, and no artifact backs them, so by this repo's own rule they
> cannot be cited or compared. They are recorded because they are the evidence for
> a *decision* — why the v1 AMI cell is empty — not as results. Both
> `sortformer-streaming-4spk-v2` figures are published rows and have artifacts.

> Public-dataset DER is **not** Raven's private-meeting DER — it is the externally
> checkable proxy anyone can reproduce, and it will differ from the internal number.

> **Note on Raven's internal DER numbers.** The DER values Raven cites internally
> (e.g. community-1 ~8% vs cloud ~15%) were measured on Raven's *private* meeting
> corpus, which cannot be published (consent). Those are a real-data datapoint, not
> an externally checkable claim. The numbers in *this* table are measured on the
> public datasets above and will differ — these are the ones anyone can reproduce.
