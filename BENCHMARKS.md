# Benchmarks

> Version-stamped results. Each table states the exact commit + scoring contract
> that produced it, and links the command to reproduce it. Numbers only ever get
> *added* under a known contract, never asserted without one — and every row on
> this page resolves to a committed artifact that `make verify` re-scores on
> every push, in both directions (`tests/test_published_table.py`).

## Who is measuring, and what Raven has at stake

Raven sells meeting transcription, and this page ranks models Raven's own product
runs. `primeline/parakeet-primeline` transcribes every standard tier (the
`primeline/parakeet-primeline` rows of the WER table), and
`pyannote/speaker-diarization-community-1` is the diarizer built into the product
(the `pyannote-community-1` rows of the DER tables). Raven's sovereign tier uses a
third model, Voxtral Mini 3B, served by Privatemode; it has no row here. Every
other row is a system Raven does not run.

That interest is bounded by rules the repository enforces rather than by our
word. The scoring conventions and bootstrap seeds are pinned in
[`benchmark.config.yaml`](./benchmark.config.yaml), every model and dataset is
pinned to a revision that each artifact records, and a test fails any artifact
that does not. A failed request or a missing hypothesis fails `make verify`
instead of quietly shrinking the corpus a number stands for. Training-data
overlap is named beside the rows it affects, not in a footnote. No row carries a
winner mark. And every number below re-scores from committed model outputs with
`make verify`, on a laptop, by anyone who doubts it.

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
  plus an optional `"bleu"` + `"bleu_signature"` on translation-shaped subsets,
  and the [numeric entity score](#numbers--the-error-wer-hides-numeric-entity-hit-rate)
  (`entity_hit_rate_pct`, its interval, `n_entities`, `n_utterances_with_entities`),
  and the runaway count (`n_runaway`, `runaway_pct`, `runaway_edit_share_pct`).

A model whose server wraps the transcript in markup of its own —
MOSS-Transcribe-Diarize interleaves `[4.18]` timestamps and `[S01]` speaker
tags with the words — is parsed where its output is read, not in the scoring
contract (`raven_asr.output_formats`, bound per model in
`raven_asr.config.OUTPUT_FORMATS`). Its lines then carry `prediction` (the spoken
text, what every number here is computed on), `raw_prediction` (the server's
output, byte for byte) and `segments` (the speaker turns the markup carried).
Scored raw, every tag counts as an inserted word: on the 2026-11 sweep that put
MOSS at ~100 % strict-de WER on read speech it transcribes almost perfectly. A
run recorded before its model had a parser is re-derived from that raw text
with `make reparse RESULTS=…` — no inference, every score recomputed, the step
named under `post_hoc` in `summary.json` (the same step applies the recipe's
output-length cap, [below](#wer--german-asr-public-datasets)).

`make verify` (→ `scripts/verify.py`) re-scores every `predictions_*.jsonl`
with our own scorer and asserts each subset matches `expected.json` within
**±0.05 pp**. CI runs it on every push, so a table row cannot drift from its
committed data. The scorer uses german-asr's flozi-strict normalization
(`unidecode` + `alpha2digit`), aligns **each utterance on its own**, and
aggregates over the **corpus** (total word edits / total reference words), with
CER on raw text — so the same predictions in reproduce the same published
`wer_pct` out. The per-utterance alignment is the one deliberate divergence from
flozi's own code, which concatenates a subset into one sentence first; see
[how WER is counted](#how-wer-is-counted-per-utterance-since-2026-09-19). Reconciliation
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
| `avemio-german-mixed-test` | `avemio/ASR-GERMAN-MIXED-TEST` | `test` (Common Voice + MLS German, one split) | CC-BY-4.0, strictest component (attribution in `/NOTICE`) |
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

Every dataset entry in `benchmark.config.yaml` also states what is spoken, in
three fields: a BCP 47 `language_tag` (`gsw-CH` for both Swiss sets, `bar` for
the Bavarian xSID probe, `de` or `en` elsewhere), a `variety_label` in words, and
a `locality` as the source documents it, or null. A region subtag is always a
country — `de-BY` would be German as spoken in Belarus — and the contract test
rejects a state or canton in that slot.

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

**Loops are counted, and named.** A model that loops — "sehr, sehr, sehr, …" for
12,000 characters — on a few clips can out-insert every other error in a corpus:
on the 2026-11 sweep Phi-4-multimodal scored 70.3 % on the first 50 `spc-test`
utterances and 110.5 % on all 3,332, and 9 looping answers (0.27 % of the
utterances) made 40.7 % of its word edits. The WER keeps them; it is the headline,
loops included. Beside it, `loops` counts the utterances whose hypothesis has
**more than 3× the reference's words** (flozi-strict words; `wer.runaway` in
`benchmark.config.yaml`), and `loop edits %` is their share of the corpus word
edits — the part of the WER they are. Both are committed in `expected.json`
(`n_runaway`, `runaway_pct`, `runaway_edit_share_pct`) and re-derived by
`make verify`.

How much a loop costs depends on how long the server lets it run, so that is part
of the serving recipe too. Every self-hosted model decodes **greedily** (the shim
sends temperature 0 and repetition penalty 1.0 with every request; NeMo models
are checked against their checkpoint's decoding config), and every vLLM and
transformers recipe caps output at **`max_tokens` = 1024**: the longest reference
in any dataset here is 686 UTF-8 bytes, no tokenizer these models use spends less
than a byte per token, so no faithful transcript comes near it (that reference is
145 Phi-4 tokens). Before the cap, the engine default decided — 4,096 tokens for
Phi-4, the context window or a generation_config limit elsewhere — and a third party serving the same model with
another default got another WER. Each run records the decoding in its `serving`
block (`decoding`). A run measured before the cap is cut to it after the fact by
`make reparse` (no inference), marked `capped_post_hoc` with the cap and the
tokenizer revision under `post_hoc` in `summary.json`, and flagged
`"exact": false`: the cut is made on the re-tokenized text, not the generated
token ids (on the 2026-11 Phi-4 run, every loop re-tokenized to the 4,096 tokens
the server had capped it at, once 4,095), and audio vLLM splits is capped per
piece by a server. The next run with the cap server-side is the exact number.

**Precision is part of every self-hosted row.** Two runs of one model at different
weight precision measure different things, so the `precision` column names the
format the weights were served in and the computation ran in (`bf16`, `fp16`,
`fp32`; for a quantized model the scheme, e.g. `fp8 W8A16`, and the GPUs it was
split over). It comes from the run's own `serving.precision` block, which every
server reports and every run records (`docs/TIER2-KEYS.md`, "Precision"); a
change of precision mid-subset aborts the run. `hosted` rows ran on someone
else's servers, whose precision nobody outside can see. A **quantized** row is
published with its **control**: the same model in bf16 (with CPU offload where
it does not fit the cards), measured on `fleurs`, and the paired WER difference
quantized − bf16 with its 95 % interval, directly under the quantized row.

¹ The parakeet-primeline rows predate the published serving code (2026-07-30
through a server whose source is not in this repo, 2026-10-02 through a Modal
app); neither recorded its precision. The checkpoint stores fp32, but whether
those servers cast it is not on record, so the column does not claim it.

| model | precision | dataset | WER strict % | 95 % CI | CER % | n | numbers hit % | 95 % CI | numbers | loops | loop edits % | run | flozi ref WER |
|-------|-----------|---------|------------:|--------:|------:|--:|--------------:|--------:|--------:|------:|-------------:|-----|--------------:|
| primeline/parakeet-primeline | not recorded¹ | Tuda-De | 4.02 | [2.76, 5.55] | 2.75 | 414 | 94.51 | [87.50, 100.00] | 91 | 0 | 0.00 | [2026-07-30](./artifacts/2026-07-30-parakeet-primeline/primeline-parakeet/) | 4.11 |
| primeline/parakeet-primeline | not recorded¹ | multilingual_librispeech | 3.04 | [2.91, 3.18] | 1.90 | 3996 | 97.05 | [95.10, 98.60] | 339 | 0 | 0.00 | [2026-07-30](./artifacts/2026-07-30-parakeet-primeline/primeline-parakeet/) | 2.60 |
| primeline/parakeet-primeline | not recorded¹ | common_voice_19_0 | 2.58 | [2.41, 2.75] | 0.85 | 5389 | 98.31 | [96.62, 99.66] | 295 | 0 | 0.00 | [2026-07-30](./artifacts/2026-07-30-parakeet-primeline/primeline-parakeet/) | 3.03 |
| primeline/parakeet-primeline | not recorded¹ | fhnw-all-dialects | 31.04 | [30.24, 31.86] | 16.42 | 2875 | 93.28 | [88.50, 97.48] | 119 | 0 | 0.00 | [2026-10-02](./artifacts/2026-10-02-fhnw-all-dialects-parakeet-primeline/modal-parakeet/) | — |
| xai/grok-voice-transcribe-2.0 | hosted | avemio-german-mixed-test | 6.87 | [5.24, 8.76] | 3.85 | 100 | 100.00 | [100.00, 100.00] | 3 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-avemio-german-mixed-test-grok-voice-transcribe-2-0/xai-grok-voice-transcribe-2.0/) | — |
| xai/grok-voice-transcribe-2.0 | hosted | fleurs | 4.59 | [3.43, 5.90] | 2.57 | 100 | 93.18 | [83.78, 100.00] | 44 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-fleurs-grok-voice-transcribe-2-0/xai-grok-voice-transcribe-2.0/) | — |
| xai/grok-voice-transcribe-2.0 | hosted | mls-de | 8.37 | [6.87, 9.95] | 10.53 | 100 | 90.00 | [66.67, 100.00] | 10 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-mls-de-grok-voice-transcribe-2-0/xai-grok-voice-transcribe-2.0/) | — |
| xai/grok-voice-transcribe-2.0 | hosted | voxpopuli-de | 12.99 | [10.88, 15.32] | 8.39 | 100 | 75.00 | [50.00, 94.44] | 16 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-voxpopuli-de-grok-voice-transcribe-2-0/xai-grok-voice-transcribe-2.0/) | — |
| xai/grok-voice-transcribe-2.0 | hosted | spc-test | 41.77 | [35.69, 47.99] | 23.85 | 50 | 60.00 | [20.00, 100.00] | 5 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-spc-test-grok-voice-transcribe-2-0/xai-grok-voice-transcribe-2.0/) | — |
| xai/grok-voice-transcribe-2.0 | hosted | fhnw-all-dialects | 43.92 | [35.98, 52.24] | 22.70 | 50 | 75.00 | [50.00, 100.00] | 4 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-fhnw-all-dialects-grok-voice-transcribe-2-0/xai-grok-voice-transcribe-2.0/) | — |
| xai/grok-voice-transcribe-1.0 | hosted | avemio-german-mixed-test | 7.21 | [5.45, 9.29] | 4.41 | 100 | 66.67 | [0.00, 100.00] | 3 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-avemio-german-mixed-test-grok-voice-transcribe-1-0/xai-grok-voice-transcribe-1.0/) | — |
| xai/grok-voice-transcribe-1.0 | hosted | fleurs | 6.75 | [4.74, 9.56] | 9.49 | 100 | 90.91 | [80.49, 98.04] | 44 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-fleurs-grok-voice-transcribe-1-0/xai-grok-voice-transcribe-1.0/) | — |
| xai/grok-voice-transcribe-1.0 | hosted | mls-de | 8.22 | [7.04, 9.47] | 10.45 | 100 | 90.00 | [66.67, 100.00] | 10 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-mls-de-grok-voice-transcribe-1-0/xai-grok-voice-transcribe-1.0/) | — |
| xai/grok-voice-transcribe-1.0 | hosted | voxpopuli-de | 11.27 | [9.43, 13.21] | 7.14 | 100 | 75.00 | [50.00, 94.44] | 16 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-voxpopuli-de-grok-voice-transcribe-1-0/xai-grok-voice-transcribe-1.0/) | — |
| xai/grok-voice-transcribe-1.0 | hosted | spc-test | 50.07 | [43.82, 55.98] | 27.01 | 50 | 80.00 | [40.00, 100.00] | 5 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-spc-test-grok-voice-transcribe-1-0/xai-grok-voice-transcribe-1.0/) | — |
| xai/grok-voice-transcribe-1.0 | hosted | fhnw-all-dialects | 39.66 | [32.79, 46.94] | 22.16 | 50 | 75.00 | [50.00, 100.00] | 4 | 0 | 0.00 | [2026-09-18](./artifacts/2026-09-18-fhnw-all-dialects-grok-voice-transcribe-1-0/xai-grok-voice-transcribe-1.0/) | — |

**xAI Grok Voice Transcribe 2.0** (hosted, us-east-1, `language=de`, `format=true`,
2026-09-18) was run on exactly the samples the Raven benchmark page compares every
provider on: the first 100 rows of `avemio-german-mixed-test`, `fleurs`, `mls-de`
and `voxpopuli-de`, the first 50 of `spc-test` and `fhnw-all-dialects`. That the
loaders here yield those same rows was checked reference by reference before the
run. 500 of 500 utterances returned; 1.50 h of audio, USD 0.15 at the listed
USD 0.10 per hour. Each artifact also commits `wer_strict_de_pct` — the page's
strict-de, length-weighted lens (`raven_eval_core.corpus_wer_strict_de_pct`),
re-scored by `make verify` like the published number — so the page's value is
computed here and not only next to it: 6.62 / 5.19 / 8.44 / 12.36 % on the four
read-aloud sets. The two Swiss sets are translation-shaped, so their headline is
BLEU (43.42 on `spc-test`, 40.65 on `fhnw-all-dialects`, signature in each
`expected.json`) and the WER beside it is a floor. On `mls-de` the model sometimes
appends the audiobook's spoken section marker ("Ende von Abschnitt eins"), which
the reference omits; those are counted as insertions, which is why CER exceeds WER
there.

Its predecessor **Grok Voice Transcribe 1.0** ran on the same samples with the
same settings (500 of 500, 1.50 h, USD 0.15): strict-de 6.84 / 7.51 / 8.55 /
11.16 %, BLEU 34.13 (`spc-test`) and 37.10 (`fhnw-all-dialects`). 1.0 returns
mostly lower-case text without punctuation, which is why its raw-text CER sits
above its WER on `fleurs` and `mls-de`; the normalised lenses are unaffected.

**Is the generation step real?** Both versions ran on the same utterances, so
they can be compared *paired*: every resample draws one set of utterances and
scores both models on it, which cancels utterance difficulty (published WER,
95 %, 10 000 resamples, `make analyse ARTIFACT=<2.0 dir> COMPARE=<1.0 dir>`):

| dataset | WER 2.0 − 1.0 (pp) | 95 % CI | reads as |
|---------|-------------------:|--------:|----------|
| `fleurs` | −2.17 | [−5.02, −0.13] | 2.0 ahead |
| `spc-test` | −8.30 | [−12.30, −3.87] | 2.0 ahead |
| `voxpopuli-de` | +1.72 | [+0.49, +3.13] | 1.0 ahead |
| `avemio-german-mixed-test` | −0.34 | [−1.37, +0.68] | no ranking |
| `mls-de` | +0.15 | [−0.87, +1.19] | no ranking |
| `fhnw-all-dialects` | +4.26 | [−0.68, +9.13] | no ranking |

So the step is not uniform, and on half of these sets 100 or 50 utterances do
not rank the two versions at all. The strict-de lens agrees everywhere except
VoxPopuli, where its interval ends at exactly zero ([+0.00, +2.52]) — a lead
for 1.0 there, not a settled one. BLEU carries no interval yet, so the Swiss
BLEU gap (43.42 against 34.13 on `spc-test`, 40.65 against 37.10 on
`fhnw-all-dialects`) is a point comparison and ranks nothing on its own.

**Per dialect region.** "Swiss German" is not one evaluation category —
KugelAudio's open dialect TTS benchmark protocol reports per locality before
aggregating, and this page follows it. `fhnw-all-dialects` writes each
speaker's canton into the sample id, and `make analyse` splits the corpus by it:
n, WER with its interval, and BLEU with n only (no interval, as above). A canton
with fewer than 30 utterances (`benchmark.config.yaml` →
`dialect_region_breakdown.min_n`) shows its n and a note, not a number. The 50
utterances behind both rows above spread over 14 cantons, and none reaches the
floor:

| canton | ZH | SG | AG | BL¹ | BE | SO | TG | ZG | GL | GR | LU | SZ | UR | VS |
|--------|---:|---:|---:|----:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| n | 9 | 7 | 6 | 6 | 5 | 4 | 4 | 3 | 1 | 1 | 1 | 1 | 1 | 1 |

¹ Basel-Landschaft and Basel-Stadt: the corpus README labels BS speakers BL.

So these runs say nothing about any single region.

**Parakeet-primeline on the whole public half** (2026-10-02) covers all 2,875
utterances of `fhnw-all-dialects`; every row of the corpus's `public.tsv` yielded
a clip (`n_dropped` 0). It ran on the same weights served from Modal instead of
the AI box; on `Tuda-De` that deployment scores 4.00 % against the 4.02 %
published above, on the same 414 utterances. Corpus BLEU 48.73, strict-de WER
31.02 %. Thirteen of the 17 cantons reach the floor (`make analyse
ARTIFACT=artifacts/2026-10-02-fhnw-all-dialects-parakeet-primeline/modal-parakeet`):

| canton | n | WER strict % | 95 % CI | BLEU |
|--------|--:|------------:|--------:|-----:|
| BE | 684 | 32.48 | [30.77, 34.25] | 47.48 |
| ZH | 480 | 29.96 | [28.01, 31.87] | 49.65 |
| SG | 361 | 31.64 | [29.29, 34.05] | 48.10 |
| AG | 353 | 29.35 | [27.32, 31.42] | 50.11 |
| BL¹ | 188 | 29.14 | [26.35, 32.00] | 49.95 |
| LU | 170 | 30.88 | [27.75, 34.13] | 50.16 |
| TG | 170 | 30.99 | [27.93, 34.14] | 47.37 |
| SO | 120 | 34.92 | [30.24, 39.91] | 45.19 |
| ZG | 97 | 31.22 | [26.15, 36.64] | 49.06 |
| UR | 56 | 29.08 | [24.58, 33.59] | 51.63 |
| VS | 56 | 35.17 | [28.60, 42.03] | 44.93 |
| GR | 42 | 26.07 | [20.96, 31.39] | 52.35 |
| SZ | 31 | 26.57 | [19.87, 33.92] | 51.47 |
| GL | 25 | — | — | — |
| FR | 24 | — | — | — |
| SH | 11 | — | — | — |
| NW | 7 | — | — | — |

GL, FR, SH and NW stay below 30 utterances in the public half, so the corpus
cannot speak for them at any run size. Most intervals overlap; the table
describes where the model stands per region and ranks no region against
another. Region rows never carry a winner mark and are never averaged — the
corpus row pools utterances, it is not a mean of regions. Canton names follow ISO 3166-2:CH; a code the corpus
README does not explain stays a code.

Re-score any row with `make verify` (Tier-1, no GPU) — it recomputes these from
the committed `predictions_*.jsonl` and asserts they match within ±0.05 pp,
intervals and the `numbers` columns included. It also fails a row whose predictions file records a failed
request or holds a different number of lines than `expected.json` commits to: a
WER over only the clips a model finished is not the published quantity.

The **95 % CI** is a percentile bootstrap over utterances (10 000 resamples, seed
and settings in `benchmark.config.yaml` → `wer.uncertainty`), each resample
re-aggregating total edits / total reference words — an interval around the
printed number, not a cousin of it. On 50–100 utterances it is several points
wide; two rows whose intervals overlap are not ranked by this table. Whether a
gap between two models is real is a *paired* question — `make analyse` with
`COMPARE=` answers it.

`flozi ref WER` is the published anchor from the dataset card (sanity guard, not
a re-scored number). The run's `predictions_*.jsonl` were produced by the
`primeline/parakeet-primeline` model (NeMo, `2_95_WER.nemo`, fp16) served
locally on a dual-RTX-3090 box via its OpenAI-compatible `/v1/audio/transcriptions`
endpoint; the harness pulled the public audio from Hugging Face and scored on the
host. Latency is therefore environment-specific (Tier-3) and deliberately not
tabled here.

**Serving.** Every self-hosted row is served from the recipe in `serving/`
(engine pins, launch flags, the NeMo batching server, the shim), and runs
record which recipe, engine and batching answered (`summary.json` → `serving`).
All vLLM-served models share one engine, kept at the newest vLLM release, and
each of their recipes pins the newest Hugging Face commit of its model; an engine
bump is followed by re-measuring every vLLM model, so a row's `serving` block names
the version it was measured on.
How to stand the servers up: `docs/TIER2-KEYS.md`, "Serving self-hosted
models". Self-hosted rows reproduce to character-level batching tolerance, not
bit-exactly: server-side batching changed 2 of 862 FLEURS transcripts of
Parakeet-TDT-0.6B-v3 by one character (WER unchanged, CER +0.0008 pp).



> Raven's internal private-meeting WER/DER numbers are measured on a corpus that
> cannot be published (consent) and are reported separately (Tier 3) — the rows
> here are the public-dataset numbers anyone can re-score.

### The 2026-11 campaign: 22 self-hosted models on full test splits

**In one paragraph.** Between 2026-11-02 and 2026-11-09, 22 open models were
served from the recipes in `serving/` and run over the **whole** test split of
eight German sets: the four read-aloud sets `fleurs` (862), `avemio-german-mixed-test`
(9,799), `mls-de` (3,394) and `voxpopuli-de` (1,968), and the dialect and probe
sets `spc-test` (3,332), `fhnw-all-dialects` (2,875), `xsid-bar` (800) and
`xsid-de-control` (800). That is 23,830 utterances per model and 524,260 in all,
with zero failed requests. The runs came from five raven.eval commits (each row's
commit is in the serving table below) on two machines with the same card type,
and two of the models are quantized and carry a bf16 control. Every row links
its artifact; `make verify` re-scores all of them.

The tables are per dataset and sorted by WER. Sorting is for reading, not a
ranking: rows whose intervals overlap are not ordered by this page, and no row
carries a winner mark. Read-aloud sets first, then the dialect and probe sets,
where the Swiss sets are translation-shaped and BLEU is the headline (listed
above each of those tables).

² Nine models (Cohere Transcribe, Granite Speech 4.1, Phi-4-multimodal,
Voxtral-Mini-3B, Moonshine Streaming, Whisper large-v3, MOSS Transcribe-Diarize,
Primeline Whisper turbo German, Qwen3-ASR) ran on raven.eval `f25c72ed`, before
runs recorded their precision (#81) and before each subset kept its own run
manifest. Their `precision` cell therefore says `not recorded`; the format they
were served in is the one their recipe in `serving/recipes.toml` pins (bf16;
fp16 for Whisper large-v3; fp32 for Moonshine), and the Raven page shows it as
taken from the recipe. Their summaries were completed by
`raven_asr.resummarize --adopt-unrecorded`, which says so in `post_hoc`.

#### `fleurs`

| model | precision | dataset | WER strict % | 95 % CI | CER % | n | numbers hit % | 95 % CI | numbers | loops | loop edits % | run | flozi ref WER |
|-------|-----------|---------|------------:|--------:|------:|--:|--------------:|--------:|--------:|------:|-------------:|-----|--------------:|
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | fp8-e4m3 fp8-w8a16-marlin, TP 2 | fleurs | 2.43 | [2.15, 2.71] | 1.47 | 862 | 94.30 | [90.67, 97.35] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-small-24b-2507-fp8/voxtral-small-24b-2507-fp8/) | — |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | int8 int8-w8a16-marlin, TP 2 | fleurs | 3.80 | [3.34, 4.30] | 4.18 | 862 | 79.19 | [73.54, 84.56] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nemotron-3.5-transcribe-30b-a4b-w8a16/nemotron-3.5-transcribe-30b-a4b-w8a16/) | — |
| mistralai/Voxtral-Mini-3B-2507 | not recorded² | fleurs | 3.82 | [3.45, 4.20] | 2.09 | 862 | 88.59 | [81.58, 94.74] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-3b-2507/voxtral-mini-3b-2507/) | — |
| openai/whisper-large-v3 | not recorded² | fleurs | 3.85 | [3.49, 4.24] | 2.37 | 862 | 86.91 | [80.39, 92.66] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3/openai-whisper-large-v3/) | — |
| CohereLabs/cohere-transcribe-03-2026 | not recorded² | fleurs | 3.90 | [3.52, 4.30] | 2.19 | 862 | 86.24 | [79.56, 92.17] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-cohere-transcribe-03-2026/cohere-transcribe-03-2026/) | — |
| Qwen/Qwen3-ASR-1.7B | not recorded² | fleurs | 4.17 | [3.73, 4.62] | 2.25 | 862 | 87.92 | [80.88, 94.04] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-qwen3-asr-1.7b/qwen3-asr-1.7b/) | — |
| primeline/whisper-large-v3-turbo-german | not recorded² | fleurs | 4.18 | [3.81, 4.56] | 2.92 | 862 | 82.21 | [75.68, 88.32] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-turbo-german/primeline-whisper-large-v3-turbo-german/) | — |
| nvidia/canary-1b-v2 | fp32 | fleurs | 4.30 | [3.89, 4.72] | 2.26 | 862 | 85.23 | [78.34, 91.38] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-canary-1b-v2/nvidia-canary-1b-v2/) | — |
| primeline/whisper-large-v3-german | bf16 | fleurs | 4.36 | [3.93, 4.80] | 4.52 | 862 | 69.46 | [63.12, 75.48] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-german/primeline-whisper-large-v3-german/) | — |
| openai/whisper-large-v3-turbo | fp16 | fleurs | 4.58 | [4.13, 5.04] | 2.45 | 862 | 89.93 | [84.78, 94.31] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3-turbo/openai-whisper-large-v3-turbo/) | — |
| primeline/parakeet-primeline | fp32 | fleurs | 4.58 | [4.16, 5.00] | 2.84 | 862 | 83.56 | [77.32, 89.12] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-parakeet/primeline-parakeet/) | — |
| nvidia/parakeet-tdt-0.6b-v3 | fp32 | fleurs | 4.83 | [4.41, 5.26] | 2.30 | 862 | 84.23 | [77.63, 90.41] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-parakeet-tdt-0.6b-v3/nvidia-parakeet-tdt-0.6b-v3/) | — |
| nvidia/nemotron-labs-unified-asr-0.6b | fp32 | fleurs | 4.89 | [4.44, 5.35] | 2.75 | 862 | 85.23 | [79.73, 90.32] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-labs-unified-asr-0.6b/nvidia-nemotron-labs-unified-asr-0.6b/) | — |
| ibm-granite/granite-speech-5.0-470m-turboctc | bf16 | fleurs | 5.02 | [4.59, 5.46] | 3.43 | 862 | 68.79 | [62.50, 75.00] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc/) | — |
| ibm-granite/granite-speech-5.0-2b | bf16 | fleurs | 5.24 | [4.76, 5.74] | 9.89 | 862 | 72.15 | [66.12, 78.14] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-2b/granite-speech-5.0-2b/) | — |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | bf16 | fleurs | 5.59 | [4.93, 6.31] | 3.23 | 862 | 85.57 | [79.55, 90.97] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-4b-realtime-2602/voxtral-mini-4b-realtime-2602/) | — |
| microsoft/Phi-4-multimodal-instruct | not recorded² | fleurs | 5.62 | [5.17, 6.10] | 3.66 | 862 | 89.93 | [84.57, 94.43] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-phi-4-multimodal/phi-4-multimodal/) | — |
| ibm-granite/granite-speech-4.1-2b-plus | not recorded² | fleurs | 6.61 | [6.08, 7.16] | 10.07 | 862 | 69.46 | [63.42, 75.24] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus/) | — |
| moonshine-ai/moonshine-streaming-small-de | not recorded² | fleurs | 6.69 | [6.19, 7.21] | 4.13 | 862 | 79.87 | [72.79, 86.38] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moonshine-streaming-small-de/moonshine-streaming-small-de/) | — |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | not recorded² | fleurs | 7.42 | [6.88, 7.97] | 4.03 | 862 | 88.93 | [84.38, 92.99] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moss-transcribe-diarize/moss-transcribe-diarize/) | — |
| microsoft/VibeVoice-ASR-HF | bf16 | fleurs | 7.69 | [5.99, 10.57] | 6.88 | 862 | 84.56 | [79.08, 89.74] | 298 | 1 | 15.81 | [2026-11](./artifacts/2026-11-campaign-vibevoice-asr/vibevoice-asr/) | — |
| nvidia/nemotron-3.5-asr-streaming-0.6b | fp32 | fleurs | 11.10 | [10.32, 11.89] | 7.38 | 862 | 67.45 | [61.17, 73.72] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-3.5-asr-streaming-0.6b/nvidia-nemotron-3.5-asr-streaming-0.6b/) | — |
| nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B | bf16, TP 2, control | fleurs | 3.76 | [3.30, 4.26] | 4.14 | 862 | 79.19 | [73.52, 84.54] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nemotron-3.5-transcribe-30b-a4b-bf16-control/nemotron-3.5-transcribe-30b-a4b-bf16-control/) | — |
| mistralai/Voxtral-Small-24B-2507 | bf16, TP 2, control | fleurs | 2.43 | [2.15, 2.71] | 1.48 | 862 | 94.30 | [90.67, 97.35] | 298 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-small-24b-2507-bf16-control/voxtral-small-24b-2507-bf16-control/) | — |

#### `avemio-german-mixed-test`

| model | precision | dataset | WER strict % | 95 % CI | CER % | n | numbers hit % | 95 % CI | numbers | loops | loop edits % | run | flozi ref WER |
|-------|-----------|---------|------------:|--------:|------:|--:|--------------:|--------:|--------:|------:|-------------:|-----|--------------:|
| primeline/whisper-large-v3-turbo-german | not recorded² | avemio-german-mixed-test | 2.60 | [2.49, 2.72] | 1.40 | 9799 | 97.52 | [96.35, 98.60] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-turbo-german/primeline-whisper-large-v3-turbo-german/) | — |
| primeline/whisper-large-v3-german | bf16 | avemio-german-mixed-test | 2.83 | [2.68, 3.00] | 4.16 | 9799 | 96.28 | [94.61, 97.78] | 725 | 4 | 2.47 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-german/primeline-whisper-large-v3-german/) | — |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | int8 int8-w8a16-marlin, TP 2 | avemio-german-mixed-test | 2.84 | [2.72, 2.96] | 2.27 | 9799 | 97.38 | [95.91, 98.65] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nemotron-3.5-transcribe-30b-a4b-w8a16/nemotron-3.5-transcribe-30b-a4b-w8a16/) | — |
| CohereLabs/cohere-transcribe-03-2026 | not recorded² | avemio-german-mixed-test | 2.94 | [2.82, 3.06] | 5.05 | 9799 | 94.48 | [92.63, 96.24] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-cohere-transcribe-03-2026/cohere-transcribe-03-2026/) | — |
| primeline/parakeet-primeline | fp32 | avemio-german-mixed-test | 2.94 | [2.83, 3.07] | 1.60 | 9799 | 96.69 | [95.18, 98.04] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-parakeet/primeline-parakeet/) | — |
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | fp8-e4m3 fp8-w8a16-marlin, TP 2 | avemio-german-mixed-test | 2.97 | [2.85, 3.10] | 4.12 | 9799 | 96.28 | [94.77, 97.60] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-small-24b-2507-fp8/voxtral-small-24b-2507-fp8/) | — |
| ibm-granite/granite-speech-5.0-2b | bf16 | avemio-german-mixed-test | 3.07 | [2.95, 3.19] | 7.21 | 9799 | 94.90 | [92.73, 96.79] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-2b/granite-speech-5.0-2b/) | — |
| ibm-granite/granite-speech-5.0-470m-turboctc | bf16 | avemio-german-mixed-test | 3.12 | [3.00, 3.25] | 2.59 | 9799 | 93.38 | [91.33, 95.32] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc/) | — |
| openai/whisper-large-v3 | not recorded² | avemio-german-mixed-test | 3.27 | [3.14, 3.40] | 5.30 | 9799 | 94.07 | [92.20, 95.82] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3/openai-whisper-large-v3/) | — |
| openai/whisper-large-v3-turbo | fp16 | avemio-german-mixed-test | 3.61 | [3.47, 3.74] | 5.39 | 9799 | 92.69 | [90.66, 94.59] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3-turbo/openai-whisper-large-v3-turbo/) | — |
| nvidia/parakeet-tdt-0.6b-v3 | fp32 | avemio-german-mixed-test | 3.63 | [3.50, 3.77] | 2.43 | 9799 | 97.10 | [95.68, 98.40] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-parakeet-tdt-0.6b-v3/nvidia-parakeet-tdt-0.6b-v3/) | — |
| nvidia/canary-1b-v2 | fp32 | avemio-german-mixed-test | 3.64 | [3.51, 3.77] | 2.36 | 9799 | 96.69 | [95.21, 98.03] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-canary-1b-v2/nvidia-canary-1b-v2/) | — |
| nvidia/nemotron-labs-unified-asr-0.6b | fp32 | avemio-german-mixed-test | 3.64 | [3.50, 3.78] | 2.45 | 9799 | 97.66 | [96.43, 98.74] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-labs-unified-asr-0.6b/nvidia-nemotron-labs-unified-asr-0.6b/) | — |
| ibm-granite/granite-speech-4.1-2b-plus | not recorded² | avemio-german-mixed-test | 3.81 | [3.67, 3.95] | 7.53 | 9799 | 92.14 | [89.64, 94.44] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus/) | — |
| Qwen/Qwen3-ASR-1.7B | not recorded² | avemio-german-mixed-test | 3.99 | [3.86, 4.14] | 2.70 | 9799 | 97.10 | [95.59, 98.44] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-qwen3-asr-1.7b/qwen3-asr-1.7b/) | — |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | bf16 | avemio-german-mixed-test | 4.37 | [4.18, 4.57] | 3.24 | 9799 | 95.86 | [94.10, 97.48] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-4b-realtime-2602/voxtral-mini-4b-realtime-2602/) | — |
| moonshine-ai/moonshine-streaming-small-de | not recorded² | avemio-german-mixed-test | 4.48 | [4.32, 4.65] | 5.68 | 9799 | 94.21 | [92.20, 95.99] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moonshine-streaming-small-de/moonshine-streaming-small-de/) | — |
| mistralai/Voxtral-Mini-3B-2507 | not recorded² | avemio-german-mixed-test | 4.50 | [4.36, 4.66] | 5.13 | 9799 | 95.17 | [93.42, 96.75] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-3b-2507/voxtral-mini-3b-2507/) | — |
| microsoft/Phi-4-multimodal-instruct | not recorded² | avemio-german-mixed-test | 4.87 | [4.72, 5.03] | 3.95 | 9799 | 94.21 | [92.23, 96.03] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-phi-4-multimodal/phi-4-multimodal/) | — |
| nvidia/nemotron-3.5-asr-streaming-0.6b | fp32 | avemio-german-mixed-test | 6.24 | [6.05, 6.44] | 3.84 | 9799 | 93.10 | [90.66, 95.37] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-3.5-asr-streaming-0.6b/nvidia-nemotron-3.5-asr-streaming-0.6b/) | — |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | not recorded² | avemio-german-mixed-test | 6.73 | [6.54, 6.92] | 3.81 | 9799 | 96.28 | [94.64, 97.70] | 725 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moss-transcribe-diarize/moss-transcribe-diarize/) | — |
| microsoft/VibeVoice-ASR-HF | bf16 | avemio-german-mixed-test | 8.25 | [7.51, 9.31] | 5.91 | 9799 | 93.52 | [91.21, 95.57] | 725 | 4 | 8.59 | [2026-11](./artifacts/2026-11-campaign-vibevoice-asr/vibevoice-asr/) | — |

#### `mls-de`

| model | precision | dataset | WER strict % | 95 % CI | CER % | n | numbers hit % | 95 % CI | numbers | loops | loop edits % | run | flozi ref WER |
|-------|-----------|---------|------------:|--------:|------:|--:|--------------:|--------:|--------:|------:|-------------:|-----|--------------:|
| CohereLabs/cohere-transcribe-03-2026 | not recorded² | mls-de | 3.46 | [3.29, 3.63] | 2.19 | 3394 | 79.62 | [74.74, 84.23] | 471 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-cohere-transcribe-03-2026/cohere-transcribe-03-2026/) | — |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | int8 int8-w8a16-marlin, TP 2 | mls-de | 3.72 | [3.53, 3.91] | 8.41 | 3394 | 97.45 | [95.37, 99.15] | 471 | 3 | 0.60 | [2026-11](./artifacts/2026-11-campaign-nemotron-3.5-transcribe-30b-a4b-w8a16/nemotron-3.5-transcribe-30b-a4b-w8a16/) | — |
| ibm-granite/granite-speech-5.0-470m-turboctc | bf16 | mls-de | 3.95 | [3.76, 4.15] | 8.66 | 3394 | 79.41 | [74.49, 84.19] | 471 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc/) | — |
| ibm-granite/granite-speech-5.0-2b | bf16 | mls-de | 4.11 | [3.94, 4.29] | 2.04 | 3394 | 83.01 | [78.80, 87.09] | 471 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-2b/granite-speech-5.0-2b/) | — |
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | fp8-e4m3 fp8-w8a16-marlin, TP 2 | mls-de | 4.29 | [4.12, 4.47] | 7.08 | 3394 | 90.66 | [87.42, 93.63] | 471 | 2 | 0.27 | [2026-11](./artifacts/2026-11-campaign-voxtral-small-24b-2507-fp8/voxtral-small-24b-2507-fp8/) | — |
| openai/whisper-large-v3 | not recorded² | mls-de | 4.41 | [4.22, 4.60] | 2.89 | 3394 | 77.71 | [72.78, 82.60] | 471 | 2 | 0.26 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3/openai-whisper-large-v3/) | — |
| nvidia/canary-1b-v2 | fp32 | mls-de | 4.52 | [4.33, 4.73] | 8.48 | 3394 | 88.53 | [84.89, 91.95] | 471 | 2 | 0.25 | [2026-11](./artifacts/2026-11-campaign-nvidia-canary-1b-v2/nvidia-canary-1b-v2/) | — |
| primeline/whisper-large-v3-turbo-german | not recorded² | mls-de | 4.59 | [4.41, 4.77] | 8.61 | 3394 | 90.66 | [87.45, 93.74] | 471 | 2 | 0.25 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-turbo-german/primeline-whisper-large-v3-turbo-german/) | — |
| primeline/parakeet-primeline | fp32 | mls-de | 5.25 | [5.06, 5.44] | 8.68 | 3394 | 90.87 | [87.63, 93.83] | 471 | 2 | 0.28 | [2026-11](./artifacts/2026-11-campaign-primeline-parakeet/primeline-parakeet/) | — |
| openai/whisper-large-v3-turbo | fp16 | mls-de | 5.42 | [5.07, 5.81] | 4.73 | 3394 | 78.77 | [73.77, 83.44] | 471 | 1 | 0.11 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3-turbo/openai-whisper-large-v3-turbo/) | — |
| primeline/whisper-large-v3-german | bf16 | mls-de | 5.44 | [5.25, 5.64] | 6.71 | 3394 | 92.36 | [89.75, 94.74] | 471 | 1 | 0.11 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-german/primeline-whisper-large-v3-german/) | — |
| ibm-granite/granite-speech-4.1-2b-plus | not recorded² | mls-de | 5.63 | [5.41, 5.86] | 2.91 | 3394 | 75.80 | [70.85, 80.58] | 471 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus/) | — |
| Qwen/Qwen3-ASR-1.7B | not recorded² | mls-de | 5.73 | [5.49, 5.99] | 9.06 | 3394 | 94.06 | [91.26, 96.41] | 471 | 3 | 0.39 | [2026-11](./artifacts/2026-11-campaign-qwen3-asr-1.7b/qwen3-asr-1.7b/) | — |
| nvidia/nemotron-labs-unified-asr-0.6b | fp32 | mls-de | 5.86 | [5.60, 6.13] | 8.49 | 3394 | 90.02 | [86.27, 93.41] | 471 | 3 | 0.38 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-labs-unified-asr-0.6b/nvidia-nemotron-labs-unified-asr-0.6b/) | — |
| nvidia/parakeet-tdt-0.6b-v3 | fp32 | mls-de | 6.29 | [6.01, 6.59] | 9.06 | 3394 | 89.60 | [85.80, 93.04] | 471 | 3 | 0.35 | [2026-11](./artifacts/2026-11-campaign-nvidia-parakeet-tdt-0.6b-v3/nvidia-parakeet-tdt-0.6b-v3/) | — |
| moonshine-ai/moonshine-streaming-small-de | not recorded² | mls-de | 6.36 | [6.11, 6.61] | 3.32 | 3394 | 79.41 | [74.89, 83.73] | 471 | 1 | 0.09 | [2026-11](./artifacts/2026-11-campaign-moonshine-streaming-small-de/moonshine-streaming-small-de/) | — |
| mistralai/Voxtral-Mini-3B-2507 | not recorded² | mls-de | 6.43 | [6.20, 6.66] | 6.39 | 3394 | 87.69 | [83.71, 91.33] | 471 | 3 | 0.35 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-3b-2507/voxtral-mini-3b-2507/) | — |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | bf16 | mls-de | 6.54 | [6.23, 6.85] | 9.63 | 3394 | 90.23 | [86.64, 93.53] | 471 | 3 | 0.34 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-4b-realtime-2602/voxtral-mini-4b-realtime-2602/) | — |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | not recorded² | mls-de | 7.65 | [7.42, 7.89] | 9.51 | 3394 | 96.18 | [93.70, 98.13] | 471 | 3 | 0.29 | [2026-11](./artifacts/2026-11-campaign-moss-transcribe-diarize/moss-transcribe-diarize/) | — |
| microsoft/VibeVoice-ASR-HF | bf16 | mls-de | 8.43 | [6.90, 10.56] | 11.00 | 3394 | 95.75 | [93.15, 97.89] | 471 | 5 | 18.43 | [2026-11](./artifacts/2026-11-campaign-vibevoice-asr/vibevoice-asr/) | — |
| microsoft/Phi-4-multimodal-instruct | not recorded² | mls-de | 8.84 | [7.44, 10.88] | 11.39 | 3394 | 81.53 | [77.12, 85.66] | 471 | 3 | 13.31 | [2026-11](./artifacts/2026-11-campaign-phi-4-multimodal/phi-4-multimodal/) | — |
| nvidia/nemotron-3.5-asr-streaming-0.6b | fp32 | mls-de | 9.74 | [9.44, 10.04] | 9.26 | 3394 | 84.08 | [79.87, 87.93] | 471 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-3.5-asr-streaming-0.6b/nvidia-nemotron-3.5-asr-streaming-0.6b/) | — |

#### `voxpopuli-de`

| model | precision | dataset | WER strict % | 95 % CI | CER % | n | numbers hit % | 95 % CI | numbers | loops | loop edits % | run | flozi ref WER |
|-------|-----------|---------|------------:|--------:|------:|--:|--------------:|--------:|--------:|------:|-------------:|-----|--------------:|
| CohereLabs/cohere-transcribe-03-2026 | not recorded² | voxpopuli-de | 7.51 | [6.87, 8.28] | 4.88 | 1968 | 89.72 | [85.84, 93.24] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-cohere-transcribe-03-2026/cohere-transcribe-03-2026/) | — |
| ibm-granite/granite-speech-5.0-470m-turboctc | bf16 | voxpopuli-de | 8.43 | [7.82, 9.16] | 5.68 | 1968 | 86.67 | [82.45, 90.54] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc/) | — |
| nvidia/parakeet-tdt-0.6b-v3 | fp32 | voxpopuli-de | 8.88 | [8.25, 9.62] | 6.63 | 1968 | 76.39 | [71.27, 81.40] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-parakeet-tdt-0.6b-v3/nvidia-parakeet-tdt-0.6b-v3/) | — |
| nvidia/canary-1b-v2 | fp32 | voxpopuli-de | 8.89 | [8.29, 9.59] | 6.50 | 1968 | 73.06 | [67.55, 78.63] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-canary-1b-v2/nvidia-canary-1b-v2/) | — |
| nvidia/nemotron-labs-unified-asr-0.6b | fp32 | voxpopuli-de | 9.14 | [8.50, 9.89] | 6.93 | 1968 | 76.94 | [71.74, 81.96] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-labs-unified-asr-0.6b/nvidia-nemotron-labs-unified-asr-0.6b/) | — |
| primeline/whisper-large-v3-german | bf16 | voxpopuli-de | 9.28 | [8.43, 10.40] | 6.19 | 1968 | 88.06 | [84.06, 91.83] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-german/primeline-whisper-large-v3-german/) | — |
| ibm-granite/granite-speech-5.0-2b | bf16 | voxpopuli-de | 9.37 | [8.74, 10.15] | 11.37 | 1968 | 72.50 | [67.04, 78.00] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-2b/granite-speech-5.0-2b/) | — |
| primeline/parakeet-primeline | fp32 | voxpopuli-de | 10.52 | [9.85, 11.27] | 7.00 | 1968 | 83.89 | [79.59, 87.96] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-parakeet/primeline-parakeet/) | — |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | int8 int8-w8a16-marlin, TP 2 | voxpopuli-de | 10.64 | [9.98, 11.40] | 8.13 | 1968 | 73.89 | [68.44, 79.18] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nemotron-3.5-transcribe-30b-a4b-w8a16/nemotron-3.5-transcribe-30b-a4b-w8a16/) | — |
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | fp8-e4m3 fp8-w8a16-marlin, TP 2 | voxpopuli-de | 11.03 | [10.39, 11.75] | 7.40 | 1968 | 88.33 | [84.43, 91.83] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-small-24b-2507-fp8/voxtral-small-24b-2507-fp8/) | — |
| moonshine-ai/moonshine-streaming-small-de | not recorded² | voxpopuli-de | 11.43 | [10.51, 12.66] | 12.21 | 1968 | 71.11 | [65.55, 76.84] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moonshine-streaming-small-de/moonshine-streaming-small-de/) | — |
| mistralai/Voxtral-Mini-3B-2507 | not recorded² | voxpopuli-de | 11.68 | [11.03, 12.42] | 7.86 | 1968 | 87.22 | [83.16, 90.88] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-3b-2507/voxtral-mini-3b-2507/) | — |
| ibm-granite/granite-speech-4.1-2b-plus | not recorded² | voxpopuli-de | 12.15 | [10.09, 15.76] | 12.95 | 1968 | 70.56 | [64.99, 76.26] | 360 | 1 | 13.40 | [2026-11](./artifacts/2026-11-campaign-granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus/) | — |
| microsoft/Phi-4-multimodal-instruct | not recorded² | voxpopuli-de | 12.72 | [12.02, 13.52] | 8.89 | 1968 | 87.22 | [83.00, 90.96] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-phi-4-multimodal/phi-4-multimodal/) | — |
| primeline/whisper-large-v3-turbo-german | not recorded² | voxpopuli-de | 12.85 | [10.74, 15.68] | 8.93 | 1968 | 85.00 | [80.62, 89.02] | 360 | 2 | 13.03 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-turbo-german/primeline-whisper-large-v3-turbo-german/) | — |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | bf16 | voxpopuli-de | 13.74 | [12.89, 14.68] | 9.53 | 1968 | 84.72 | [80.18, 88.89] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-4b-realtime-2602/voxtral-mini-4b-realtime-2602/) | — |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | not recorded² | voxpopuli-de | 14.66 | [13.33, 16.32] | 10.68 | 1968 | 81.39 | [76.45, 86.04] | 360 | 2 | 6.23 | [2026-11](./artifacts/2026-11-campaign-moss-transcribe-diarize/moss-transcribe-diarize/) | — |
| Qwen/Qwen3-ASR-1.7B | not recorded² | voxpopuli-de | 15.85 | [11.36, 22.77] | 11.03 | 1968 | 87.22 | [83.33, 90.86] | 360 | 3 | 27.32 | [2026-11](./artifacts/2026-11-campaign-qwen3-asr-1.7b/qwen3-asr-1.7b/) | — |
| microsoft/VibeVoice-ASR-HF | bf16 | voxpopuli-de | 16.62 | [14.61, 19.76] | 11.19 | 1968 | 84.72 | [80.48, 88.60] | 360 | 3 | 10.92 | [2026-11](./artifacts/2026-11-campaign-vibevoice-asr/vibevoice-asr/) | — |
| nvidia/nemotron-3.5-asr-streaming-0.6b | fp32 | voxpopuli-de | 17.71 | [16.72, 18.73] | 12.36 | 1968 | 62.50 | [56.70, 68.66] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-3.5-asr-streaming-0.6b/nvidia-nemotron-3.5-asr-streaming-0.6b/) | — |
| openai/whisper-large-v3 | not recorded² | voxpopuli-de | 17.96 | [16.64, 19.34] | 14.08 | 1968 | 81.94 | [76.78, 86.72] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3/openai-whisper-large-v3/) | — |
| openai/whisper-large-v3-turbo | fp16 | voxpopuli-de | 20.84 | [19.30, 22.46] | 16.69 | 1968 | 79.72 | [74.21, 84.81] | 360 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3-turbo/openai-whisper-large-v3-turbo/) | — |

#### `spc-test`

BLEU (headline for translation-shaped sets; WER beside it is a floor): openai/whisper-large-v3 55.43, openai/whisper-large-v3-turbo 54.00, RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic 53.38, CohereLabs/cohere-transcribe-03-2026 53.37, primeline/whisper-large-v3-german 53.32, primeline/whisper-large-v3-turbo-german 50.94, mistralai/Voxtral-Mini-3B-2507 50.27, moonshine-ai/moonshine-streaming-small-de 49.74, primeline/parakeet-primeline 47.49, nvidia/parakeet-tdt-0.6b-v3 44.70, nvidia/nemotron-labs-unified-asr-0.6b 39.63, Qwen/Qwen3-ASR-1.7B 39.32, PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 35.85, mistralai/Voxtral-Mini-4B-Realtime-2602 32.38, nvidia/canary-1b-v2 30.39, OpenMOSS-Team/MOSS-Transcribe-Diarize 30.31, microsoft/VibeVoice-ASR-HF 25.78, microsoft/Phi-4-multimodal-instruct 20.18, ibm-granite/granite-speech-5.0-470m-turboctc 17.22, nvidia/nemotron-3.5-asr-streaming-0.6b 11.00, ibm-granite/granite-speech-5.0-2b 7.03, ibm-granite/granite-speech-4.1-2b-plus 2.05.

| model | precision | dataset | WER strict % | 95 % CI | CER % | n | numbers hit % | 95 % CI | numbers | loops | loop edits % | run | flozi ref WER |
|-------|-----------|---------|------------:|--------:|------:|--:|--------------:|--------:|--------:|------:|-------------:|-----|--------------:|
| openai/whisper-large-v3 | not recorded² | spc-test | 29.30 | [28.54, 30.03] | 18.21 | 3332 | 88.96 | [85.99, 91.77] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3/openai-whisper-large-v3/) | — |
| openai/whisper-large-v3-turbo | fp16 | spc-test | 30.00 | [29.27, 30.74] | 18.34 | 3332 | 86.45 | [83.17, 89.61] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3-turbo/openai-whisper-large-v3-turbo/) | — |
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | fp8-e4m3 fp8-w8a16-marlin, TP 2 | spc-test | 30.81 | [30.04, 31.56] | 18.94 | 3332 | 88.80 | [85.85, 91.65] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-small-24b-2507-fp8/voxtral-small-24b-2507-fp8/) | — |
| CohereLabs/cohere-transcribe-03-2026 | not recorded² | spc-test | 31.04 | [29.42, 33.54] | 18.89 | 3332 | 87.12 | [83.99, 90.19] | 598 | 1 | 3.37 | [2026-11](./artifacts/2026-11-campaign-cohere-transcribe-03-2026/cohere-transcribe-03-2026/) | — |
| primeline/whisper-large-v3-german | bf16 | spc-test | 32.56 | [31.77, 33.35] | 19.46 | 3332 | 81.94 | [78.44, 85.29] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-german/primeline-whisper-large-v3-german/) | — |
| mistralai/Voxtral-Mini-3B-2507 | not recorded² | spc-test | 33.32 | [32.55, 34.10] | 20.67 | 3332 | 84.45 | [81.07, 87.75] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-3b-2507/voxtral-mini-3b-2507/) | — |
| primeline/whisper-large-v3-turbo-german | not recorded² | spc-test | 33.54 | [32.49, 34.82] | 19.74 | 3332 | 78.60 | [74.87, 82.26] | 598 | 1 | 1.36 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-turbo-german/primeline-whisper-large-v3-turbo-german/) | — |
| moonshine-ai/moonshine-streaming-small-de | not recorded² | spc-test | 34.46 | [33.68, 35.26] | 20.55 | 3332 | 80.43 | [76.68, 84.07] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moonshine-streaming-small-de/moonshine-streaming-small-de/) | — |
| primeline/parakeet-primeline | fp32 | spc-test | 35.93 | [35.17, 36.67] | 20.79 | 3332 | 78.93 | [75.21, 82.62] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-parakeet/primeline-parakeet/) | — |
| nvidia/parakeet-tdt-0.6b-v3 | fp32 | spc-test | 37.24 | [36.47, 38.01] | 21.99 | 3332 | 66.56 | [62.25, 70.77] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-parakeet-tdt-0.6b-v3/nvidia-parakeet-tdt-0.6b-v3/) | — |
| nvidia/nemotron-labs-unified-asr-0.6b | fp32 | spc-test | 40.32 | [39.54, 41.10] | 24.05 | 3332 | 67.73 | [63.51, 71.93] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-labs-unified-asr-0.6b/nvidia-nemotron-labs-unified-asr-0.6b/) | — |
| Qwen/Qwen3-ASR-1.7B | not recorded² | spc-test | 42.78 | [41.94, 43.59] | 24.47 | 3332 | 69.90 | [65.92, 73.90] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-qwen3-asr-1.7b/qwen3-asr-1.7b/) | — |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | int8 int8-w8a16-marlin, TP 2 | spc-test | 48.72 | [47.71, 49.69] | 27.17 | 3332 | 57.86 | [53.48, 62.30] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nemotron-3.5-transcribe-30b-a4b-w8a16/nemotron-3.5-transcribe-30b-a4b-w8a16/) | — |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | not recorded² | spc-test | 52.50 | [51.61, 53.38] | 29.22 | 3332 | 66.72 | [62.44, 71.00] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moss-transcribe-diarize/moss-transcribe-diarize/) | — |
| nvidia/canary-1b-v2 | fp32 | spc-test | 54.70 | [53.71, 55.69] | 32.04 | 3332 | 61.54 | [57.10, 65.85] | 598 | 2 | 0.49 | [2026-11](./artifacts/2026-11-campaign-nvidia-canary-1b-v2/nvidia-canary-1b-v2/) | — |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | bf16 | spc-test | 55.42 | [54.31, 56.54] | 35.18 | 3332 | 70.57 | [66.10, 74.96] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-4b-realtime-2602/voxtral-mini-4b-realtime-2602/) | — |
| microsoft/VibeVoice-ASR-HF | bf16 | spc-test | 66.11 | [61.57, 71.94] | 38.14 | 3332 | 62.04 | [57.67, 66.43] | 598 | 10 | 10.31 | [2026-11](./artifacts/2026-11-campaign-vibevoice-asr/vibevoice-asr/) | — |
| ibm-granite/granite-speech-5.0-470m-turboctc | bf16 | spc-test | 67.50 | [66.60, 68.40] | 40.56 | 3332 | 39.13 | [34.99, 43.37] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc/) | — |
| ibm-granite/granite-speech-5.0-2b | bf16 | spc-test | 68.44 | [66.51, 71.03] | 42.40 | 3332 | 34.95 | [30.92, 39.15] | 598 | 2 | 2.25 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-2b/granite-speech-5.0-2b/) | — |
| nvidia/nemotron-3.5-asr-streaming-0.6b | fp32 | spc-test | 74.70 | [73.84, 75.55] | 52.16 | 3332 | 23.41 | [19.90, 27.09] | 598 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-3.5-asr-streaming-0.6b/nvidia-nemotron-3.5-asr-streaming-0.6b/) | — |
| microsoft/Phi-4-multimodal-instruct | not recorded² | spc-test | 75.34 | [68.83, 83.52] | 51.11 | 3332 | 40.47 | [36.27, 44.85] | 598 | 8 | 12.89 | [2026-11](./artifacts/2026-11-campaign-phi-4-multimodal/phi-4-multimodal/) | — |
| ibm-granite/granite-speech-4.1-2b-plus | not recorded² | spc-test | 232.92 | [205.25, 262.47] | 158.11 | 3332 | 24.25 | [20.66, 27.91] | 598 | 119 | 67.88 | [2026-11](./artifacts/2026-11-campaign-granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus/) | — |

#### `fhnw-all-dialects`

BLEU (headline for translation-shaped sets; WER beside it is a floor): RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic 58.70, openai/whisper-large-v3 58.18, mistralai/Voxtral-Mini-3B-2507 55.89, openai/whisper-large-v3-turbo 54.65, primeline/whisper-large-v3-german 51.61, CohereLabs/cohere-transcribe-03-2026 51.07, primeline/whisper-large-v3-turbo-german 48.81, primeline/parakeet-primeline 48.73, moonshine-ai/moonshine-streaming-small-de 48.65, Qwen/Qwen3-ASR-1.7B 44.29, nvidia/parakeet-tdt-0.6b-v3 43.45, nvidia/nemotron-labs-unified-asr-0.6b 41.87, PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 34.82, mistralai/Voxtral-Mini-4B-Realtime-2602 32.58, OpenMOSS-Team/MOSS-Transcribe-Diarize 30.06, nvidia/canary-1b-v2 29.77, microsoft/VibeVoice-ASR-HF 24.63, microsoft/Phi-4-multimodal-instruct 24.28, ibm-granite/granite-speech-5.0-470m-turboctc 21.36, nvidia/nemotron-3.5-asr-streaming-0.6b 8.73, ibm-granite/granite-speech-5.0-2b 7.84, ibm-granite/granite-speech-4.1-2b-plus 6.50.

| model | precision | dataset | WER strict % | 95 % CI | CER % | n | numbers hit % | 95 % CI | numbers | loops | loop edits % | run | flozi ref WER |
|-------|-----------|---------|------------:|--------:|------:|--:|--------------:|--------:|--------:|------:|-------------:|-----|--------------:|
| openai/whisper-large-v3 | not recorded² | fhnw-all-dialects | 23.77 | [23.01, 24.56] | 12.97 | 2875 | 95.80 | [91.74, 99.15] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3/openai-whisper-large-v3/) | — |
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | fp8-e4m3 fp8-w8a16-marlin, TP 2 | fhnw-all-dialects | 24.77 | [23.97, 25.59] | 14.54 | 2875 | 100.00 | [100.00, 100.00] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-small-24b-2507-fp8/voxtral-small-24b-2507-fp8/) | — |
| mistralai/Voxtral-Mini-3B-2507 | not recorded² | fhnw-all-dialects | 26.53 | [25.73, 27.33] | 15.35 | 2875 | 98.32 | [95.69, 100.00] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-3b-2507/voxtral-mini-3b-2507/) | — |
| openai/whisper-large-v3-turbo | fp16 | fhnw-all-dialects | 26.60 | [25.79, 27.42] | 14.02 | 2875 | 95.80 | [91.74, 99.16] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3-turbo/openai-whisper-large-v3-turbo/) | — |
| primeline/whisper-large-v3-german | bf16 | fhnw-all-dialects | 29.14 | [28.32, 29.96] | 15.89 | 2875 | 94.12 | [89.47, 98.23] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-german/primeline-whisper-large-v3-german/) | — |
| CohereLabs/cohere-transcribe-03-2026 | not recorded² | fhnw-all-dialects | 30.39 | [29.58, 31.25] | 16.73 | 2875 | 94.12 | [89.47, 98.23] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-cohere-transcribe-03-2026/cohere-transcribe-03-2026/) | — |
| primeline/parakeet-primeline | fp32 | fhnw-all-dialects | 31.05 | [30.25, 31.87] | 16.43 | 2875 | 93.28 | [88.50, 97.48] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-parakeet/primeline-parakeet/) | — |
| primeline/whisper-large-v3-turbo-german | not recorded² | fhnw-all-dialects | 31.29 | [30.45, 32.14] | 16.35 | 2875 | 92.44 | [87.39, 96.69] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-turbo-german/primeline-whisper-large-v3-turbo-german/) | — |
| moonshine-ai/moonshine-streaming-small-de | not recorded² | fhnw-all-dialects | 32.11 | [31.25, 32.98] | 17.16 | 2875 | 90.76 | [85.25, 95.69] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moonshine-streaming-small-de/moonshine-streaming-small-de/) | — |
| Qwen/Qwen3-ASR-1.7B | not recorded² | fhnw-all-dialects | 36.01 | [35.14, 36.91] | 18.96 | 2875 | 93.28 | [88.33, 97.48] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-qwen3-asr-1.7b/qwen3-asr-1.7b/) | — |
| nvidia/parakeet-tdt-0.6b-v3 | fp32 | fhnw-all-dialects | 36.39 | [35.49, 37.28] | 18.60 | 2875 | 85.71 | [79.13, 91.67] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-parakeet-tdt-0.6b-v3/nvidia-parakeet-tdt-0.6b-v3/) | — |
| nvidia/nemotron-labs-unified-asr-0.6b | fp32 | fhnw-all-dialects | 36.58 | [35.74, 37.42] | 18.56 | 2875 | 87.39 | [81.36, 93.04] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-labs-unified-asr-0.6b/nvidia-nemotron-labs-unified-asr-0.6b/) | — |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | int8 int8-w8a16-marlin, TP 2 | fhnw-all-dialects | 48.13 | [47.00, 49.29] | 22.62 | 2875 | 74.79 | [66.39, 82.64] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nemotron-3.5-transcribe-30b-a4b-w8a16/nemotron-3.5-transcribe-30b-a4b-w8a16/) | — |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | not recorded² | fhnw-all-dialects | 50.37 | [49.36, 51.36] | 24.51 | 2875 | 84.03 | [77.39, 90.24] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moss-transcribe-diarize/moss-transcribe-diarize/) | — |
| nvidia/canary-1b-v2 | fp32 | fhnw-all-dialects | 52.08 | [51.09, 53.09] | 27.03 | 2875 | 68.07 | [59.65, 76.27] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-canary-1b-v2/nvidia-canary-1b-v2/) | — |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | bf16 | fhnw-all-dialects | 52.84 | [51.75, 53.97] | 28.86 | 2875 | 78.99 | [71.54, 86.07] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-4b-realtime-2602/voxtral-mini-4b-realtime-2602/) | — |
| ibm-granite/granite-speech-5.0-2b | bf16 | fhnw-all-dialects | 57.56 | [56.61, 58.52] | 32.18 | 2875 | 60.50 | [52.10, 69.03] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-2b/granite-speech-5.0-2b/) | — |
| ibm-granite/granite-speech-5.0-470m-turboctc | bf16 | fhnw-all-dialects | 58.83 | [57.90, 59.76] | 28.10 | 2875 | 47.90 | [38.71, 57.14] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc/) | — |
| microsoft/Phi-4-multimodal-instruct | not recorded² | fhnw-all-dialects | 62.86 | [57.99, 69.56] | 39.72 | 2875 | 54.62 | [45.00, 63.93] | 119 | 3 | 7.84 | [2026-11](./artifacts/2026-11-campaign-phi-4-multimodal/phi-4-multimodal/) | — |
| ibm-granite/granite-speech-4.1-2b-plus | not recorded² | fhnw-all-dialects | 67.12 | [62.68, 75.05] | 36.89 | 2875 | 51.26 | [42.37, 60.17] | 119 | 1 | 5.47 | [2026-11](./artifacts/2026-11-campaign-granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus/) | — |
| microsoft/VibeVoice-ASR-HF | bf16 | fhnw-all-dialects | 68.15 | [61.41, 76.87] | 37.61 | 2875 | 73.11 | [64.91, 80.67] | 119 | 5 | 12.38 | [2026-11](./artifacts/2026-11-campaign-vibevoice-asr/vibevoice-asr/) | — |
| nvidia/nemotron-3.5-asr-streaming-0.6b | fp32 | fhnw-all-dialects | 76.88 | [76.04, 77.73] | 46.67 | 2875 | 36.13 | [27.59, 44.92] | 119 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-3.5-asr-streaming-0.6b/nvidia-nemotron-3.5-asr-streaming-0.6b/) | — |

#### `xsid-bar`

BLEU (headline for translation-shaped sets; WER beside it is a floor): RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic 20.94, openai/whisper-large-v3 20.55, Qwen/Qwen3-ASR-1.7B 19.40, openai/whisper-large-v3-turbo 18.82, mistralai/Voxtral-Mini-3B-2507 18.54, mistralai/Voxtral-Mini-4B-Realtime-2602 18.23, nvidia/canary-1b-v2 18.17, primeline/parakeet-primeline 17.70, microsoft/Phi-4-multimodal-instruct 17.55, PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 17.40, nvidia/nemotron-labs-unified-asr-0.6b 17.08, primeline/whisper-large-v3-turbo-german 17.03, microsoft/VibeVoice-ASR-HF 16.79, nvidia/parakeet-tdt-0.6b-v3 16.61, primeline/whisper-large-v3-german 16.34, CohereLabs/cohere-transcribe-03-2026 16.28, moonshine-ai/moonshine-streaming-small-de 16.14, ibm-granite/granite-speech-5.0-470m-turboctc 14.22, OpenMOSS-Team/MOSS-Transcribe-Diarize 13.00, nvidia/nemotron-3.5-asr-streaming-0.6b 10.87, ibm-granite/granite-speech-5.0-2b 2.73, ibm-granite/granite-speech-4.1-2b-plus 2.57.

| model | precision | dataset | WER strict % | 95 % CI | CER % | n | numbers hit % | 95 % CI | numbers | loops | loop edits % | run | flozi ref WER |
|-------|-----------|---------|------------:|--------:|------:|--:|--------------:|--------:|--------:|------:|-------------:|-----|--------------:|
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | fp8-e4m3 fp8-w8a16-marlin, TP 2 | xsid-bar | 59.14 | [57.23, 61.07] | 40.94 | 800 | 68.80 | [62.81, 74.58] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-small-24b-2507-fp8/voxtral-small-24b-2507-fp8/) | — |
| Qwen/Qwen3-ASR-1.7B | not recorded² | xsid-bar | 59.39 | [57.48, 61.28] | 41.92 | 800 | 60.80 | [54.72, 66.67] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-qwen3-asr-1.7b/qwen3-asr-1.7b/) | — |
| openai/whisper-large-v3 | not recorded² | xsid-bar | 60.82 | [58.95, 62.68] | 41.10 | 800 | 70.40 | [64.63, 75.98] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3/openai-whisper-large-v3/) | — |
| mistralai/Voxtral-Mini-3B-2507 | not recorded² | xsid-bar | 60.89 | [58.99, 62.79] | 42.28 | 800 | 68.00 | [62.10, 73.68] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-3b-2507/voxtral-mini-3b-2507/) | — |
| microsoft/Phi-4-multimodal-instruct | not recorded² | xsid-bar | 61.55 | [59.74, 63.36] | 42.82 | 800 | 56.40 | [50.20, 62.65] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-phi-4-multimodal/phi-4-multimodal/) | — |
| openai/whisper-large-v3-turbo | fp16 | xsid-bar | 61.99 | [60.14, 63.81] | 41.69 | 800 | 69.20 | [63.10, 75.10] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3-turbo/openai-whisper-large-v3-turbo/) | — |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | int8 int8-w8a16-marlin, TP 2 | xsid-bar | 62.25 | [60.27, 64.21] | 44.76 | 800 | 57.20 | [51.02, 63.35] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nemotron-3.5-transcribe-30b-a4b-w8a16/nemotron-3.5-transcribe-30b-a4b-w8a16/) | — |
| CohereLabs/cohere-transcribe-03-2026 | not recorded² | xsid-bar | 63.37 | [61.58, 65.22] | 43.27 | 800 | 52.00 | [45.56, 58.54] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-cohere-transcribe-03-2026/cohere-transcribe-03-2026/) | — |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | bf16 | xsid-bar | 63.90 | [62.04, 65.76] | 44.24 | 800 | 62.40 | [56.05, 68.63] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-4b-realtime-2602/voxtral-mini-4b-realtime-2602/) | — |
| primeline/parakeet-primeline | fp32 | xsid-bar | 63.93 | [62.15, 65.66] | 42.83 | 800 | 52.80 | [46.75, 58.94] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-parakeet/primeline-parakeet/) | — |
| primeline/whisper-large-v3-german | bf16 | xsid-bar | 64.02 | [62.16, 65.86] | 44.37 | 800 | 50.40 | [44.22, 56.79] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-german/primeline-whisper-large-v3-german/) | — |
| primeline/whisper-large-v3-turbo-german | not recorded² | xsid-bar | 64.17 | [62.38, 65.99] | 43.39 | 800 | 50.00 | [43.67, 56.35] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-turbo-german/primeline-whisper-large-v3-turbo-german/) | — |
| nvidia/parakeet-tdt-0.6b-v3 | fp32 | xsid-bar | 64.82 | [62.98, 66.63] | 43.30 | 800 | 50.00 | [43.87, 56.20] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-parakeet-tdt-0.6b-v3/nvidia-parakeet-tdt-0.6b-v3/) | — |
| nvidia/nemotron-labs-unified-asr-0.6b | fp32 | xsid-bar | 65.00 | [63.21, 66.80] | 42.23 | 800 | 61.20 | [55.12, 67.46] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-labs-unified-asr-0.6b/nvidia-nemotron-labs-unified-asr-0.6b/) | — |
| moonshine-ai/moonshine-streaming-small-de | not recorded² | xsid-bar | 65.12 | [63.35, 66.89] | 44.04 | 800 | 60.00 | [54.10, 65.88] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moonshine-streaming-small-de/moonshine-streaming-small-de/) | — |
| nvidia/canary-1b-v2 | fp32 | xsid-bar | 65.68 | [63.76, 67.58] | 44.95 | 800 | 60.00 | [53.67, 66.14] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-canary-1b-v2/nvidia-canary-1b-v2/) | — |
| ibm-granite/granite-speech-5.0-470m-turboctc | bf16 | xsid-bar | 65.83 | [64.04, 67.65] | 45.09 | 800 | 48.40 | [42.17, 54.69] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc/) | — |
| ibm-granite/granite-speech-5.0-2b | bf16 | xsid-bar | 66.06 | [64.27, 67.88] | 49.47 | 800 | 49.20 | [42.91, 55.47] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-2b/granite-speech-5.0-2b/) | — |
| microsoft/VibeVoice-ASR-HF | bf16 | xsid-bar | 66.32 | [62.92, 71.18] | 46.26 | 800 | 64.00 | [57.96, 70.04] | 250 | 1 | 2.98 | [2026-11](./artifacts/2026-11-campaign-vibevoice-asr/vibevoice-asr/) | — |
| ibm-granite/granite-speech-4.1-2b-plus | not recorded² | xsid-bar | 68.63 | [66.78, 70.50] | 51.33 | 800 | 50.00 | [43.70, 56.38] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus/) | — |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | not recorded² | xsid-bar | 69.07 | [67.26, 70.87] | 46.21 | 800 | 53.60 | [47.41, 59.84] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moss-transcribe-diarize/moss-transcribe-diarize/) | — |
| nvidia/nemotron-3.5-asr-streaming-0.6b | fp32 | xsid-bar | 71.35 | [69.65, 73.07] | 48.62 | 800 | 49.60 | [43.43, 55.93] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-3.5-asr-streaming-0.6b/nvidia-nemotron-3.5-asr-streaming-0.6b/) | — |

#### `xsid-de-control`

| model | precision | dataset | WER strict % | 95 % CI | CER % | n | numbers hit % | 95 % CI | numbers | loops | loop edits % | run | flozi ref WER |
|-------|-----------|---------|------------:|--------:|------:|--:|--------------:|--------:|--------:|------:|-------------:|-----|--------------:|
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | fp8-e4m3 fp8-w8a16-marlin, TP 2 | xsid-de-control | 8.23 | [7.15, 9.37] | 6.83 | 800 | 91.60 | [88.07, 94.84] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-small-24b-2507-fp8/voxtral-small-24b-2507-fp8/) | — |
| openai/whisper-large-v3 | not recorded² | xsid-de-control | 8.47 | [7.37, 9.62] | 6.63 | 800 | 89.60 | [85.66, 93.18] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3/openai-whisper-large-v3/) | — |
| CohereLabs/cohere-transcribe-03-2026 | not recorded² | xsid-de-control | 8.89 | [7.72, 10.10] | 8.56 | 800 | 77.20 | [72.09, 82.00] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-cohere-transcribe-03-2026/cohere-transcribe-03-2026/) | — |
| openai/whisper-large-v3-turbo | fp16 | xsid-de-control | 9.45 | [8.30, 10.61] | 6.94 | 800 | 89.60 | [85.60, 93.23] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-openai-whisper-large-v3-turbo/openai-whisper-large-v3-turbo/) | — |
| primeline/parakeet-primeline | fp32 | xsid-de-control | 9.87 | [8.73, 11.07] | 8.55 | 800 | 75.20 | [69.80, 80.41] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-parakeet/primeline-parakeet/) | — |
| Qwen/Qwen3-ASR-1.7B | not recorded² | xsid-de-control | 9.97 | [8.80, 11.21] | 9.46 | 800 | 76.80 | [71.72, 81.89] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-qwen3-asr-1.7b/qwen3-asr-1.7b/) | — |
| ibm-granite/granite-speech-5.0-470m-turboctc | bf16 | xsid-de-control | 10.11 | [8.96, 11.29] | 10.39 | 800 | 74.00 | [68.75, 79.12] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-470m-turboctc/granite-speech-5.0-470m-turboctc/) | — |
| primeline/whisper-large-v3-turbo-german | not recorded² | xsid-de-control | 10.31 | [9.18, 11.46] | 8.31 | 800 | 76.80 | [71.60, 81.78] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-turbo-german/primeline-whisper-large-v3-turbo-german/) | — |
| primeline/whisper-large-v3-german | bf16 | xsid-de-control | 10.41 | [9.20, 11.62] | 10.44 | 800 | 72.80 | [67.47, 78.04] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-primeline-whisper-large-v3-german/primeline-whisper-large-v3-german/) | — |
| nvidia/parakeet-tdt-0.6b-v3 | fp32 | xsid-de-control | 10.52 | [9.32, 11.72] | 8.77 | 800 | 75.20 | [69.77, 80.33] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-parakeet-tdt-0.6b-v3/nvidia-parakeet-tdt-0.6b-v3/) | — |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | int8 int8-w8a16-marlin, TP 2 | xsid-de-control | 10.88 | [9.60, 12.17] | 10.89 | 800 | 72.80 | [67.49, 77.96] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nemotron-3.5-transcribe-30b-a4b-w8a16/nemotron-3.5-transcribe-30b-a4b-w8a16/) | — |
| mistralai/Voxtral-Mini-3B-2507 | not recorded² | xsid-de-control | 10.97 | [9.71, 12.25] | 8.47 | 800 | 90.40 | [86.61, 93.88] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-3b-2507/voxtral-mini-3b-2507/) | — |
| microsoft/Phi-4-multimodal-instruct | not recorded² | xsid-de-control | 11.00 | [9.82, 12.21] | 8.95 | 800 | 62.00 | [55.78, 68.09] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-phi-4-multimodal/phi-4-multimodal/) | — |
| ibm-granite/granite-speech-5.0-2b | bf16 | xsid-de-control | 11.13 | [9.87, 12.42] | 17.20 | 800 | 74.80 | [69.53, 79.92] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-5.0-2b/granite-speech-5.0-2b/) | — |
| nvidia/nemotron-labs-unified-asr-0.6b | fp32 | xsid-de-control | 11.13 | [9.92, 12.39] | 7.43 | 800 | 87.20 | [82.72, 91.27] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-labs-unified-asr-0.6b/nvidia-nemotron-labs-unified-asr-0.6b/) | — |
| nvidia/canary-1b-v2 | fp32 | xsid-de-control | 11.74 | [10.34, 13.15] | 9.00 | 800 | 85.60 | [81.30, 89.75] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-canary-1b-v2/nvidia-canary-1b-v2/) | — |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | bf16 | xsid-de-control | 11.98 | [10.62, 13.39] | 8.23 | 800 | 87.20 | [83.00, 91.16] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-voxtral-mini-4b-realtime-2602/voxtral-mini-4b-realtime-2602/) | — |
| microsoft/VibeVoice-ASR-HF | bf16 | xsid-de-control | 13.35 | [11.91, 14.80] | 10.61 | 800 | 81.20 | [76.47, 85.77] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-vibevoice-asr/vibevoice-asr/) | — |
| moonshine-ai/moonshine-streaming-small-de | not recorded² | xsid-de-control | 13.43 | [12.07, 14.79] | 9.68 | 800 | 82.00 | [77.20, 86.48] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moonshine-streaming-small-de/moonshine-streaming-small-de/) | — |
| ibm-granite/granite-speech-4.1-2b-plus | not recorded² | xsid-de-control | 13.87 | [12.54, 15.24] | 18.27 | 800 | 73.60 | [68.29, 78.80] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-granite-speech-4.1-2b-plus/granite-speech-4.1-2b-plus/) | — |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | not recorded² | xsid-de-control | 14.62 | [13.23, 16.03] | 10.56 | 800 | 80.00 | [75.10, 84.88] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-moss-transcribe-diarize/moss-transcribe-diarize/) | — |
| nvidia/nemotron-3.5-asr-streaming-0.6b | fp32 | xsid-de-control | 15.32 | [13.86, 16.82] | 11.83 | 800 | 71.60 | [66.28, 76.83] | 250 | 0 | 0.00 | [2026-11](./artifacts/2026-11-campaign-nvidia-nemotron-3.5-asr-streaming-0.6b/nvidia-nemotron-3.5-asr-streaming-0.6b/) | — |

#### Throughput (× real time, whole set, parallel requests)

| model | parallel | hardware | fleurs | avemio-german-mixed-test | mls-de | voxpopuli-de | spc-test | fhnw-all-dialects | xsid-bar | xsid-de-control |
|-------|---------:|----------|------:|------:|------:|------:|------:|------:|------:|------:|
| CohereLabs/cohere-transcribe-03-2026 | 8 | 1x RTX 3090, driver 580.178.04 | 277.3 | 244.5 | 242.2 | 215.8 | 191.5 | 244.2 | 123.9 | 126.0 |
| ibm-granite/granite-speech-4.1-2b-plus | 8 | 1x RTX 3090, driver 580.178.04 | 166.7 | 151.6 | 157.4 | 130.5 | 69.5 | 157.8 | 93.8 | 91.7 |
| ibm-granite/granite-speech-5.0-2b | 8 | 1x RTX 3090, driver 615.71.09 | 173.2 | 183.4 | 152.9 | 83.1 | 73.8 | 100.7 | 70.1 | 68.6 |
| ibm-granite/granite-speech-5.0-470m-turboctc | 4 | 1x RTX 3090, driver 615.71.09 | 913.2 | 520.4 | 588.4 | 648.4 | 403.0 | 255.5 | 161.3 | 193.6 |
| microsoft/Phi-4-multimodal-instruct | 8 | 1x RTX 3090, driver 580.178.04 | 92.9 | 79.1 | 79.6 | 71.5 | 60.1 | 84.7 | 46.1 | 45.7 |
| microsoft/VibeVoice-ASR-HF | 4 | 1x RTX 3090, driver 615.71.09 | 25.3 | 25.5 | 25.6 | 22.6 | 19.2 | 21.8 | 14.1 | 14.1 |
| mistralai/Voxtral-Mini-3B-2507 | 8 | 1x RTX 3090, driver 580.178.04 | 84.7 | 69.3 | 81.7 | 61.0 | 48.8 | 65.8 | 29.7 | 29.9 |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | 8 | 1x RTX 3090, driver 615.71.09 | 33.0 | 32.6 | 33.2 | 32.2 | 31.4 | 32.3 | 27.8 | 27.9 |
| moonshine-ai/moonshine-streaming-small-de | 4 | 1x RTX 3090, driver 580.178.04 | 36.8 | 34.0 | 31.5 | 30.0 | 27.9 | 41.9 | 21.5 | 22.4 |
| nvidia/canary-1b-v2 | 8 | 1x RTX 3090, driver 615.71.09 | 85.3 | 55.9 | 77.8 | 55.9 | 58.3 | 119.4 | 83.3 | 83.6 |
| nvidia/nemotron-3.5-asr-streaming-0.6b | 8 | 1x RTX 3090, driver 580.178.04 | 73.8 | 58.2 | 84.5 | 60.2 | 41.1 | 48.9 | 21.7 | 21.9 |
| nvidia/nemotron-labs-unified-asr-0.6b | 8 | 1x RTX 3090, driver 615.71.09 | 144.4 | 130.6 | 149.2 | 137.3 | 108.3 | 144.9 | 74.8 | 70.2 |
| nvidia/parakeet-tdt-0.6b-v3 | 8 | 1x RTX 3090, driver 615.71.09 | 263.0 | 210.6 | 302.1 | 201.5 | 162.4 | 204.0 | 109.7 | 110.7 |
| openai/whisper-large-v3 | 8 | 1x RTX 3090, driver 580.178.04 | 127.4 | 105.1 | 124.7 | 95.1 | 75.0 | 99.7 | 45.6 | 46.3 |
| openai/whisper-large-v3-turbo | 8 | 1x RTX 3090, driver 615.71.09 | 119.4 | 93.0 | 132.8 | 84.8 | 63.2 | 75.1 | 33.2 | 35.6 |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | 8 | 1x RTX 3090, driver 580.178.04 | 191.9 | 159.2 | 165.6 | 139.7 | 117.1 | 159.9 | 75.4 | 76.7 |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | 8 | 2x RTX 3090 TP 2, driver 580.178.04 | 105.3 | 85.3 | 103.2 | 74.7 | 62.9 | 92.4 | 46.2 | 48.3 |
| primeline/parakeet-primeline | 8 | 1x RTX 3090, driver 615.71.09 | 265.8 | 188.9 | 307.6 | 182.0 | 137.1 | 187.9 | 79.1 | 86.7 |
| primeline/whisper-large-v3-german | 8 | 1x RTX 3090, driver 615.71.09 | 137.2 | 113.8 | 131.9 | 101.5 | 82.3 | 108.5 | 50.3 | 50.8 |
| primeline/whisper-large-v3-turbo-german | 8 | 1x RTX 3090, driver 580.178.04 | 213.0 | 164.8 | 230.7 | 151.3 | 114.4 | 141.9 | 62.0 | 62.9 |
| Qwen/Qwen3-ASR-1.7B | 8 | 1x RTX 3090, driver 580.178.04 | 220.2 | 197.5 | 175.5 | 168.4 | 157.1 | 221.0 | 128.2 | 127.1 |
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | 8 | 2x RTX 3090 TP 2, driver 580.178.04 | 46.9 | 37.8 | 46.3 | 33.6 | 26.4 | 35.0 | 15.5 | 15.7 |

#### Serving recipe per model

| model | raven.eval commit | engine | recipe sha256 (first 10) | precision source |
|-------|-------------------|--------|--------------------------|------------------|
| CohereLabs/cohere-transcribe-03-2026 | `f25c72ed` | vllm 0.33.0 | `8403015c83` | recipe (`serving/recipes.toml`)² |
| ibm-granite/granite-speech-4.1-2b-plus | `f25c72ed` | vllm 0.33.0 | `512bf402d0` | recipe (`serving/recipes.toml`)² |
| ibm-granite/granite-speech-5.0-2b | `b0f6b6f7` | vllm 0.33.0 | `8c27e5884b` | recorded |
| ibm-granite/granite-speech-5.0-470m-turboctc | `b0f6b6f7` | transformers 5.20.2 (torch 2.14.0+cu130) | `d8ce3611c0` | recorded |
| microsoft/Phi-4-multimodal-instruct | `f25c72ed` | vllm 0.33.0 | `4a20920ac3` | recipe (`serving/recipes.toml`)² |
| microsoft/VibeVoice-ASR-HF | `edd9cba0` | vllm 0.33.0 | `6cda85b39d` | recorded |
| mistralai/Voxtral-Mini-3B-2507 | `f25c72ed` | vllm 0.33.0 | `3b2d3e5727` | recipe (`serving/recipes.toml`)² |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | `3055509b` | vllm 0.33.0 | `f1d189dd8e` | recorded |
| mistralai/Voxtral-Small-24B-2507 (bf16 control) | `dfb0247e` | vllm 0.33.0 | `5bdd2ff433` | recorded |
| moonshine-ai/moonshine-streaming-small-de | `f25c72ed` | transformers 5.20.2 (torch 2.14.0+cu130) | `023fdf3e52` | recipe (`serving/recipes.toml`)² |
| nvidia/canary-1b-v2 | `b0f6b6f7` | nemo 3.1.0+b19b1a79d (torch 2.14.1+cu130) | `20af60ed56` | recorded |
| nvidia/nemotron-3.5-asr-streaming-0.6b | `b0f6b6f7` | nemo 3.1.0+b19b1a79d (torch 2.14.1+cu130) | `5d37c6fbef` | recorded |
| nvidia/nemotron-labs-unified-asr-0.6b | `b0f6b6f7` | nemo 3.1.0+b19b1a79d (torch 2.14.1+cu130) | `c0c6e7de7b` | recorded |
| nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (bf16 control) | `b3f06027` | vllm 0.33.0 | `ff0e218084` | recorded |
| nvidia/parakeet-tdt-0.6b-v3 | `b0f6b6f7` | nemo 3.1.0+b19b1a79d (torch 2.14.1+cu130) | `a3be3abdd6` | recorded |
| openai/whisper-large-v3 | `f25c72ed` | vllm 0.33.0 | `a3fe32dfbb` | recipe (`serving/recipes.toml`)² |
| openai/whisper-large-v3-turbo | `b0f6b6f7` | vllm 0.33.0 | `baace10895` | recorded |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | `f25c72ed` | vllm 0.33.0 | `e2307b6810` | recipe (`serving/recipes.toml`)² |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | `dfb0247e` | vllm 0.33.0 | `33991fd0a8` | recorded |
| primeline/parakeet-primeline | `b0f6b6f7` | nemo 3.1.0+b19b1a79d (torch 2.14.1+cu130) | `3dad374b76` | recorded |
| primeline/whisper-large-v3-german | `b0f6b6f7` | vllm 0.33.0 | `caf8f9c723` | recorded |
| primeline/whisper-large-v3-turbo-german | `f25c72ed` | vllm 0.33.0 | `3cb230d5df` | recipe (`serving/recipes.toml`)² |
| Qwen/Qwen3-ASR-1.7B | `f25c72ed` | vllm 0.33.0 | `2fbf998c60` | recipe (`serving/recipes.toml`)² |
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | `b0f6b6f7` | vllm 0.33.0 | `53c9bb99b9` | recorded |

#### Loops per model and set

| model | fleurs | avemio-german-mixed-test | mls-de | voxpopuli-de | spc-test | fhnw-all-dialects | xsid-bar | xsid-de-control | total |
|-------|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| CohereLabs/cohere-transcribe-03-2026 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| ibm-granite/granite-speech-4.1-2b-plus | 0 | 0 | 0 | 1 | 119 | 1 | 0 | 0 | 121 |
| ibm-granite/granite-speech-5.0-2b | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 |
| ibm-granite/granite-speech-5.0-470m-turboctc | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| microsoft/Phi-4-multimodal-instruct | 0 | 0 | 3 | 0 | 8 | 3 | 0 | 0 | 14 |
| microsoft/VibeVoice-ASR-HF | 1 | 4 | 5 | 3 | 10 | 5 | 1 | 0 | 29 |
| mistralai/Voxtral-Mini-3B-2507 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| moonshine-ai/moonshine-streaming-small-de | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| nvidia/canary-1b-v2 | 0 | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 4 |
| nvidia/nemotron-3.5-asr-streaming-0.6b | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| nvidia/nemotron-labs-unified-asr-0.6b | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| nvidia/parakeet-tdt-0.6b-v3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| openai/whisper-large-v3 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| openai/whisper-large-v3-turbo | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | 0 | 0 | 3 | 2 | 0 | 0 | 0 | 0 | 5 |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| primeline/parakeet-primeline | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| primeline/whisper-large-v3-german | 0 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 5 |
| primeline/whisper-large-v3-turbo-german | 0 | 0 | 2 | 2 | 1 | 0 | 0 | 0 | 5 |
| Qwen/Qwen3-ASR-1.7B | 0 | 0 | 3 | 3 | 0 | 0 | 0 | 0 | 6 |
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |

**Throughput and hardware.** The speed is a throughput: audio seconds of the whole
set over wall-clock seconds, with the stated number of requests in flight against
a batching server. It is not a per-request latency and is not comparable to one.
Two machines did the measuring, each with RTX 3090 cards; the driver version in
the hardware column tells them apart (580 and 615). On the 615 machine a small
text-embedding service was found sharing the card on 2026-11-09, the day of the
VibeVoice-ASR run; WER is untouched by it, throughput there may be slightly low.
Whether it also ran during that machine's earlier runs (2026-11-04 to 11-08) can
no longer be established. The two `TP 2` rows split one
model over two NVLinked cards and are not comparable to the one-card rows.

**Batching drift.** The full sets ran batched. The single-request cross-check on
`fleurs` (Parakeet-TDT-0.6B-v3, see "Serving" above) changed 2 of 862 transcripts
by one character, WER unchanged; the raw transcripts of that cross-check are not
among the artifacts.

**Quantization controls.** Both quantized models were also run in bf16 on
`fleurs` (two cards, CPU offload for the weights that do not fit), on the same 862
utterances. Paired difference quantized − bf16 in this page's lens (`make analyse
ARTIFACT=<quantized> COMPARE=<control>`, 10,000 resamples):

| quantized model | precision | WER quantized | WER bf16 | quantized − bf16 (pp) | 95 % CI | reads as |
|-----------------|-----------|--------------:|---------:|----------------------:|--------:|----------|
| RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic | fp8-e4m3 fp8-w8a16-marlin, TP 2 | 2.43 | 2.43 | +0.00 | [−0.05, +0.05] | no measurable cost |
| PhilflowIO/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-W8A16 | int8 int8-w8a16-marlin, TP 2 | 3.80 | 3.76 | +0.04 | [−0.08, +0.16] | no measurable cost |

The Raven benchmark page scores with its own strict-de lens and shows the same
comparison as +0.01 pp [−0.05, +0.07] and +0.07 pp [−0.04, +0.20]; both lenses
say the same thing. The Nemotron W8A16 checkpoint is our own quantization. A
first bf16 control of Nemotron returned 110.7 % WER because offloaded Mamba
layers convolved with unloaded memory (#101, fixed in #103); it was discarded and
re-measured, and is not among the artifacts.

**Loops.** 215 of 524,260 utterances ran into a repetition loop (more than 3× the
reference's words), counted per model and set in the table above and included in
every WER. Granite Speech 4.1 on `spc-test` accounts for 119 of them.

**Model-specific notes.**

- **Nemotron 3.5 ASR Streaming** writes locale tags such as `<de-DE>` into the
  middle of a transcript, not only at its end; 2,320 of 3,394 `mls-de` answers
  carried one. The server stripped only a trailing tag; #98 strips all of them
  (`nemotron-locale-tags` output format), and the recorded runs were corrected by
  `make reparse` (raw text kept in `raw_prediction`, the step recorded in
  `post_hoc`). `mls-de` moves from 12.82 % to the 9.74 % shown here.
- **VibeVoice-ASR** has no language parameter: neither its model card, nor
  transformers, nor Microsoft's own code offers one. Its only input besides audio
  is a context text ("This is a N seconds audio, with extra info: …"), which
  Microsoft documents for hotwords and metadata. Since #104 the harness sends
  "The audio is in German." there when a run asks for German. That is a hint the
  model learned to read, not a hard switch like Canary's language token. Without
  it `xsid-de-control` scored 24.73 %, 600 of its 1,379 word errors from a single
  clip that looped in Spanish; with it 13.34 %, and answers in a wrong language
  dropped on every set (e.g. `spc-test` 287 → 105). On `mls-de` and
  `voxpopuli-de` the new run is worse only through a few greedy repetition loops
  ("die Rose, die Rose, …"); without the ten clips that changed most it is 6.83 %
  and 14.78 %. The run without the hint is not published.
- **Empty answers** count as 100 % WER for the utterance and are model behaviour,
  not failures: Nemotron 3.5 ASR Streaming 1,293 (`spc-test` 706,
  `fhnw-all-dialects` 477, identical over two measurements), Voxtral-Mini-4B
  Realtime 374 (`spc-test` 267), VibeVoice-ASR 307 (`avemio-german-mixed-test`
  181, `fhnw-all-dialects` 112), Nemotron Labs Unified 53, Parakeet-TDT-0.6B-v3
  37; every other model has 11 or fewer.
- **Host label.** The hardware label of the runs on the 615 machine named the
  host after the driver version. A public artifact names hardware, not
  infrastructure (`raven_asr/throughput.py`), so it was removed from those
  artifacts; each one records the step in `post_hoc`.

### Numbers — the error WER hides (numeric entity hit rate)

A wrong digit in a date, an amount or a count is the costliest error a meeting
transcript can make, and to WER it is one substitution among hundreds of words.
So the table scores the numbers on their own, the way KugelAudio's open German
TTS benchmark protocol separates entity correctness from CER: **numbers hit %**
is the share of the reference's numbers the transcript reproduces exactly, and
**numbers** is how many there are. Definition in `raven_eval_core/entities.py`,
rules in `benchmark.config.yaml` → `entity`.

- An entity is a digit run with its decimal comma, date dots or time colon kept
  inside it ("3,5", "12.03.2024", "10:30"). Number words count once flozi's
  `alpha2digit` has made them digits ("dreiundzwanzig" → "23"). It is read
  *before* flozi's final punctuation strip, which would fuse "3,5" into "35" and
  so credit a transcript that said thirty-five.
- Reference and transcript are normalized independently, and a hit is exact
  equality, matched as a multiset per utterance. A contract test pins that every
  German number word from 3 to 2999 maps onto its own digits and no other, so a
  misrecognized number cannot score.
- It is a floor. A formatting difference can cost a hit (flozi leaves "eins" and
  "zwei" as words; "20er Jahre" against "zwanziger Jahre" is a miss). A wrong
  number can never earn one.

**Read the `numbers` column before the percentage.** On the three flozi subsets
the rate rests on 91–339 numbers and its interval is a few points wide: 94.51 /
97.05 / 98.31 % on Tuda-De / MLS / Common Voice. On the Grok samples it rests on
3 to 44, and the intervals say what that is worth: avemio's 100.00 (2.0) and
66.67 (1.0) are three numbers each, and 1.0's interval is [0.00, 100.00]. No set
ranks Grok 2.0 against 1.0 on numbers; every paired interval spans or touches
zero (FLEURS +2.27 pp [−5.26, +10.64] over the 30 utterances that hold one,
`make analyse ARTIFACT=<2.0 dir> COMPARE=<1.0 dir>`). Below about 50 numbers,
treat a row as a list of cases, not as a rate.

The cases are what the metric is for. 2.0 writes the dam's "1963" as "163" and
"GM 35" as "530". Both versions write FLEURS's "2007" as "2017", the "175 Meter"
of `spc-test` as "275 m", and VoxPopuli's "60 Soldaten" as "60.000". Each of
these is one or two word errors to WER. Two more VoxPopuli misses, shared by
both versions, are references that open on a number ("21. Februar", "19.
Mitglied") the transcript does not contain: the clip edge, where the WER count
below finds the same corpus effect.

### How WER is counted (per utterance, since 2026-09-19)

Every utterance is aligned against its reference **on its own**, and the corpus
WER is total edits over total reference words across those alignments. Until
2026-09-19 the scorer followed flozi's code, which concatenates a whole subset
into one sentence before aligning. That lets an error at the end of one clip
cancel against an error at the start of the next, although the two are separate
recordings: reference "wir gehen heute" / "morgen regnet es", transcript "wir
gehen" / "heute morgen regnet es" is two errors, and zero once concatenated.

The effect depends on the corpus, not the model. It needs errors at clip edges,
which is where dialect-to-Standard-German references and clips cut out of running
parliamentary speech carry them. Re-scoring every committed artifact:

| rows | concatenated → per utterance |
|------|------------------------------|
| `fhnw-all-dialects` 1.0 / 2.0 | 38.81 → 39.66 / 43.50 → 43.92 |
| `spc-test` 1.0 / 2.0 | 49.38 → 50.07 / 41.08 → 41.77 |
| `voxpopuli-de` 1.0 / 2.0 | 11.01 → 11.27 / 12.68 → 12.99 |
| everything else, incl. flozi's Tuda-De / MLS / Common Voice | unchanged at the printed precision (largest move +0.001 pp) |

So the old count flattered every model on exactly the hard sets. The `flozi ref
WER` anchors stay comparable: on flozi's own subsets the two counts agree. The
normalization is still flozi's, verbatim; only the alignment changed
(`raven_eval_core/flozi_wer.py`).

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
under all six committed CALLHOME-de artifacts are byte-identical, and the same
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
| [high latency](./diagnostics/sortformer-v2-latency/high-latency.json) | 10.0 s | 9.26 | 9.30 |
| [low latency](./diagnostics/sortformer-v2-latency/low-latency.json) | 1.04 s | 9.82 | 9.97 |

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
> "very high latency" row has a committed *artifact* (gold + hyp RTTMs that
> `make verify` re-scores), and it is the one in the table. The other two are
> linked to their run summaries above — same 120 files, same checkpoint
> revision, same scorer, but no committed RTTMs, so they explain a discrepancy
> and may not be cited. They are reproducible with
> `SORTFORMER_LATENCY_PRESET=<name> make reproduce METRIC=der
> DATASET=callhome-de MODEL=sortformer-streaming-4spk-v2 EXTRA=sortformer`. The
> preset a run used is recorded in its `summary.json`, so a run measured at a
> non-default setting cannot be promoted as if it were the shipped one.

**What the German column shows.** Six diarizers on the same 120 CALLHOME-de
files, same gold, same scorer, same two collars. At the classic collar:
pyannote-community-1 **16.08**, deepgram-nova-3 19.31, assemblyai-universal-3-5-pro
21.74, diarizen-wavlm-large-s80-md-v2 12.07, sortformer-4spk-v1 11.41 — and
sortformer-streaming-4spk-v2 at **8.98**,
the lowest DER any row on this page reaches on *this dataset* and, unlike v1, one
we could ship: its weights are CC-BY-4.0, not CC-BY-NC-4.0. Paired over the same
files, the streaming checkpoint is **7.10 pp ahead of community-1, CI [+5.95,
+8.23]**, and **2.42 pp ahead of v1, CI [+1.11, +3.74]**; both intervals exclude
zero. (Numbers on different datasets are not comparable, which is why "best on
the page" is the wrong phrase: VoxConverse dev reads 5.00 % for community-1 and
means nothing next to a telephone number.)

**The open-source favourite, measured.** The ETH benchmark names DiariZen the
strongest open-source system on its German column and reads **11.6 %** there.
Until now this page cited that claim without testing it. On the same 120 files,
same gold, same scorer, DiariZen reads **12.07 %** corpus and **11.84 %**
file-mean — and the file-mean is the convention that column uses, so 11.84
against 11.6 is a 0.24 pp agreement. That is the second independent checkpoint
reconciled to their number on this corpus, and it is the strongest evidence yet
that our German harness measures what theirs does.

It also settles a ranking the framing had left open. Paired over the same files,
DiariZen is **3.09 pp behind sortformer-streaming-4spk-v2, CI [+2.16, +3.91]** —
an interval that excludes zero, so the shippable checkpoint really is ahead of
the open-source favourite here. Against the offline v1 the same comparison reads
**+0.66 pp, CI [-0.56, +1.84]**: that interval spans zero, so on this corpus the
two are indistinguishable and neither is "better" however the decimals fall.
Against community-1 the sign flips and the interval again excludes zero:
**-4.01 pp, CI [-4.92, -3.28]**, DiariZen ahead. Its weights are CC-BY-NC-4.0,
so under ADR-app-0036 it is a reference row like v1 — measured and shown, never
awarded.

**Where each model breaks down.** The corpus number hides the axis the field
actually degrades on. Split by *reference* speaker count, at collar 0.25
(n = 104 / 12 / 4 files at 2 / 3 / 4 speakers):

| model | 2 spk | 3 spk | 4 spk |
|---|---:|---:|---:|
| sortformer-streaming-4spk-v2 | 8.87 | 9.83 | **9.26** |
| sortformer-4spk-v1 | 10.95 | 13.75 | 15.91 |
| diarizen-wavlm-large-s80-md-v2 | 12.01 | 12.27 | 12.89 |
| pyannote-community-1 | 15.59 | 17.78 | 23.38 |
| deepgram-nova-3 | 18.81 | 20.47 | **28.42** |
| assemblyai-universal-3-5-pro | 21.80 | 20.98 | 22.36 |

The four-speaker column rests on four files and carries no interval worth
printing; read it as a direction, not a measurement. The direction separates the
table into two groups. Three models degrade steeply from two speakers to four —
community-1 +7.8 pp, sortformer-4spk-v1 +5.0 pp, and Deepgram worst at +9.6 pp —
while three stay within a point of flat: the streaming checkpoint +0.4 pp,
DiariZen +0.9 pp, AssemblyAI +0.6 pp. Flat here is not yet evidence of holding up
under many speakers, because **this corpus never has many**: CALLHOME-de tops out
at four, which is exactly where the Sortformer checkpoints are capped, so the one
regime that would separate a capped model from an uncapped one is the one regime
these 120 files cannot test.

**The regime CALLHOME cannot reach.** VoxConverse is where that question can be
asked: 146 of its 232 test files carry five or more speakers. DiariZen reads
**6.83 %** on test and **2.75 %** on dev, and split by reference speaker count on
test it goes 5.64 / 6.94 / 5.95 / 4.92 / **7.09** for 1 / 2 / 3 / 4 / 5+ — flat,
with the many-speaker bucket no worse than the two-speaker one. Community-1 on
the same 232 files reads 4.56 / 8.11 / 9.29 / 6.41 / **8.76**; paired over those
files DiariZen is **1.58 pp ahead, CI [-1.96, -1.19]**, an interval that excludes
zero. It is also the lowest of the four rows this set now carries — 6.83 against
community-1's 8.41, AssemblyAI's 18.13 and Deepgram's 37.81 — and the two hosted
rows are the sharper reminder of what this corpus asks: both are within a few
points of their German numbers on telephone speech and neither survives
in-the-wild audio with many speakers. The Sortformer column is empty here and stays empty on purpose: the
checkpoints are hard-capped at four speakers, so a 5+ number for them would be a
statement about folding, not about diarization.

**DiariZen was trained on two of these three corpora, and that changes how two
of its rows may be read.** Its model card states the training data verbatim:
"trained on far-field, single-channel audio from a diverse set of public
datasets, including **AMI**, AISHELL-4, AliMeeting, NOTSOFAR-1, MSDWild, DIHARD3,
RAMC, and **VoxConverse**." So its VoxConverse and AMI numbers are in-domain —
possibly on the very recordings it saw — while every other model on those rows is
being tested out-of-domain. That is not a defect of the model and not a mistake
in the measurement; it is a property of the comparison, and it has to be stated
where the comparison is made:

- The **VoxConverse** rows above, including the 1.58 pp lead over community-1 and
  the flat speaker-count profile, compare a model on its training distribution
  against models that never saw it. The lead is real as a number and is **not**
  evidence of better generalisation.
- The **AMI** row below is the same situation.
- **CALLHOME is not in that list**, so the German row — the one this page's main
  ranking rides on, and the one that reconciles with the ETH benchmark to 0.24 pp
  — is a clean out-of-domain measurement for every model in it.

A benchmark that quietly ranks a model on its own training data is the failure
this page exists to avoid, so the training overlap is named beside the rows
rather than in a footnote.

**On meetings DiariZen does not rescue the picture.** AMI test, all 16 meetings,
same gold and same scorer: **23.14 %** at collar 0.25 against the streaming
Sortformer's 23.03 and community-1's 13.10. It ties the capped checkpoint and
loses to community-1 by 10 pp — while holding the in-domain advantage described
above. Whatever the many-speaker argument is worth on VoxConverse, it does not
transfer to meeting audio.

So the two corpora answer different questions and neither answers both. On
German telephone speech — the one set nobody here trained on — the shippable
streaming checkpoint is genuinely ahead of DiariZen. On in-the-wild audio with
many speakers DiariZen is the only entrant measured at all in that regime, and
it is measured there in-domain. A meeting product needs the second question
answered, and this page can now say that it has not been answered for the model
Raven runs — not because the model failed it, but because the architecture
forbids the measurement.

**The third reconciliation, and the one that needed the splits.** The ETH
benchmark reads **5.2 %** for DiariZen on VoxConverse and states that it evaluated
*both* the dev and test splits. Our two rows are therefore not individually
comparable to it — they are 2.75 and 6.83, and the split alone moves the number
by 4 pp. Pooling all 448 files as one set under the file-mean their Table 3
caption describes gives **5.14 %**, against their 5.2. That inference about how
they pooled is ours, not theirs, so it is weaker evidence than the German row —
but it is the third of their numbers our gold and our scorer land on.

One number we deliberately do not claim to have matched: the paper reports 7.1 %
for DiariZen at 5+ speakers, and our VoxConverse-test 5+ bucket reads 7.09. That
is a coincidence. Their Table 3 aggregates speaker counts across all four of
their datasets, not VoxConverse alone; pooling our two VoxConverse splits at 5+
gives 5.65, not 7.1. The agreement of the two-decimal figures is the kind of
thing that reads as confirmation and is not one. On VoxConverse **test**,
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
grows with the *square* of the duration. That is not asserted here — it is
measured by [`scripts/measure_sortformer_memory.py`](./scripts/measure_sortformer_memory.py),
which cuts one AMI recording into a duration ladder, records peak allocation for
each, fits **both** a linear and a quadratic law through the origin, and names
whichever fits better. The committed curves are
[`diagnostics/sortformer-4spk-v1-memory.json`](./diagnostics/sortformer-4spk-v1-memory.json)
and [`diagnostics/sortformer-streaming-4spk-v2-memory.json`](./diagnostics/sortformer-streaming-4spk-v2-memory.json).

| duration | `4spk-v1` peak | `streaming-4spk-v2` peak |
|---:|---:|---:|
| 2 min | 1.02 GB | 0.66 GB |
| 6 min | 4.95 GB | 0.67 GB |
| 10 min | 12.65 GB | 0.81 GB |
| 12 min | **OOM** (16.0 GB free) | 0.88 GB |
| 20 min | — | 1.16 GB |
| 40 min | — | 1.85 GB |

The two checkpoints do not merely differ in size, they differ in *regime*, and
the fit says so rather than the prose: for v1 the quadratic law wins (a =
3.58e-5 GB/s², residual 0.35 GB against the linear law's 1.44 GB); for v2 the
linear law wins (a = 9.06e-4 GB/s, residual 0.32 GB against the quadratic law's
0.60 GB). A bounded speaker cache replaces quadratic attention, which is exactly
what NVIDIA claim for it — v2 uses less memory at 40 minutes than v1 does at 3.

Extrapolating v1's fitted coefficient: the **shortest** AMI test meeting is 14
min (≈25 GB) and the longest is 50 min (≈322 GB). So the AMI split is out of
reach of a 24 GB GPU by construction, not by configuration, and an 80 GB card
would still stop at ~25 min. Forcing v1's streaming path instead is measurably a
different (worse) regime — on ten CALLHOME files it reads 23.13 % / 16.42 % where
the offline model reads 18.94 % / 12.77 % — so no AMI number is published for v1
rather than one published under a silently different protocol.

The `diar_streaming_sortformer_4spk-v2` checkpoint is NVIDIA's own answer to
exactly that, and it holds: all 16 AMI test meetings completed on the same 24 GB
3090, the 49.5-minute one in 8.2 s of wall clock. Meeting-length audio is
reachable.

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
| assemblyai-universal-3-5-pro | voxconverse (**test**) | 20.64 | 12.86 | 2.64 | 5.14 | 18.13 | 11.24 | 2.01 | 4.88 | 18.25 | [16.82, 19.49] | 232 | [2026-09-04](./artifacts/2026-09-04-voxconverse-test-assemblyai-universal-3-5-pro/assemblyai-universal-3-5-pro/) |
| deepgram-nova-3 | voxconverse (**test**) | 40.33 | 28.46 | 2.51 | 9.36 | 37.81 | 26.99 | 1.49 | 9.33 | 32.18 | [32.22, 43.48] | 232 | [2026-09-04](./artifacts/2026-09-04-voxconverse-test-deepgram-nova-3/deepgram-nova-3/) |
| deepgram-nova-3 | ami (test, 4-speaker meetings, IHM) | 34.07 | 25.19 | 2.88 | 5.99 | 29.37 | 22.56 | 1.70 | 5.11 | 28.93 | [25.11, 33.50] | 16 | [2026-09-04](./artifacts/2026-09-04-ami-deepgram-nova-3/deepgram-nova-3/) |
| assemblyai-universal-3-5-pro | ami (test, 4-speaker meetings, IHM) | 39.94 | 36.46 | 1.98 | 1.50 | 36.04 | 33.08 | 1.83 | 1.13 | 35.85 | [32.46, 39.32] | 16 | [2026-09-04](./artifacts/2026-09-04-ami-assemblyai-universal-3-5-pro/assemblyai-universal-3-5-pro/) |
| diarizen-wavlm-large-s80-md-v2 (CC-BY-NC, non-commercial) | callhome-de (German, telephone) | 15.70 | 11.95 | 2.95 | 0.80 | **12.07** | 10.44 | 1.08 | 0.55 | 11.84 | [11.19, 13.01] | 120 | [2026-09-04](./artifacts/2026-09-04-callhome-de-diarizen/diarizen-wavlm-large-s80-md-v2/) |
| diarizen-wavlm-large-s80-md-v2 (CC-BY-NC, non-commercial) | voxconverse (dev) | 4.52 | 1.51 | 1.67 | 1.34 | **2.75** | 0.86 | 0.72 | 1.17 | 3.31 | [2.32, 3.26] | 216 | [2026-09-04](./artifacts/2026-09-04-voxconverse-dev-diarizen/diarizen-wavlm-large-s80-md-v2/) |
| diarizen-wavlm-large-s80-md-v2 (CC-BY-NC, non-commercial) | voxconverse (**test**) | 9.17 | 3.55 | 3.42 | 2.21 | **6.83** | 2.86 | 2.07 | 1.90 | 6.85 | [6.12, 7.59] | 232 | [2026-09-04](./artifacts/2026-09-04-voxconverse-test-diarizen/diarizen-wavlm-large-s80-md-v2/) |
| diarizen-wavlm-large-s80-md-v2 (CC-BY-NC, non-commercial) | ami (meetings) | 25.69 | 22.68 | 1.14 | 1.87 | **23.14** | 21.38 | 0.51 | 1.25 | 22.49 | [20.16, 26.11] | 16 | [2026-09-05](./artifacts/2026-09-05-ami-diarizen/diarizen-wavlm-large-s80-md-v2/) |

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

> The memory curves and the forced-streaming comparison above are **Tier-3
> diagnostics, not published numbers**: they were measured on ten files and on one
> particular GPU, so by this repo's own rule they cannot be cited or compared.
> They are recorded because they are the evidence for a *decision* — why the v1
> AMI cell is empty — not as results. The curves now have committed JSON and a
> script that reproduces them, which makes the reasoning checkable without making
> the numbers portable: each file records the device, driver, torch and NeMo
> build it belongs to. Note in particular that the 12-minute OOM is against the
> **16.0 GB this shared box actually had free** (a resident speech service holds
> the rest), not against the card's nominal 24 GB — the transferable finding is
> the growth law, not the duration at which one machine ran out. Both
> `sortformer-streaming-4spk-v2` figures are published rows and have artifacts.

> Public-dataset DER is **not** Raven's private-meeting DER — it is the externally
> checkable proxy anyone can reproduce, and it will differ from the internal number.

> **Note on Raven's internal DER numbers.** The DER values Raven cites internally
> (e.g. community-1 ~8% vs cloud ~15%) were measured on Raven's *private* meeting
> corpus, which cannot be published (consent). Those are a real-data datapoint, not
> an externally checkable claim. The numbers in *this* table are measured on the
> public datasets above and will differ — these are the ones anyone can reproduce.
