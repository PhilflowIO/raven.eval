# artifacts/ — Tier-1 committed model outputs

Two artifact shapes live here — **WER/BLEU** (per-utterance transcripts) and
**DER** (gold+hyp RTTMs). `scripts/verify.py` (via `make verify`) re-scores both
and asserts each matches its committed `expected.json` within ±0.05 pp. CI runs it
on every push, so a committed number can never silently drift from the committed
data.

## WER artifacts

Each `<run>/<model>/` directory holds the per-utterance model outputs Raven's
eval runner produced, plus the WER/CER it published for them:

```
artifacts/<run>/<model>/
  predictions_<subset>.jsonl   # one line per utterance: {"reference","prediction","latency_s"}
  expected.json                # {"<subset>": {"wer_pct": <float>, "cer_pct": <float>}, ...}
```

`make verify` re-scores every `predictions_<subset>.jsonl` and asserts the
recomputed corpus WER + CER match `expected.json` within ±0.05 pp.

### BLEU rides the same artifact

A translation-shaped subset (dialect spoken, standard German transcribed) adds
**one optional key** to the same `expected.json` entry — there is no second
artifact shape:

```
"<subset>": {"wer_pct": …, "cer_pct": …, "bleu": <float>, "bleu_signature": "<sacrebleu signature>"}
```

`bleu` is corpus BLEU on the **raw** reference/prediction text (BLEU's own
tokenizer is the declared normalization — flozi's punctuation- and case-stripping
would destroy exactly what a translation reference is asking for). Present only
where the corpus warrants it; a plain transcription subset has no `bleu` key and
is not scored for one. `bleu_signature` is the string that makes the number
comparable to somebody else's — commit it next to the number, always.

## DER artifacts (Etappe 5)

Each `<run>/<model>/` directory holds the gold + hypothesis RTTMs the diarizer
produced, plus the DER it published for them:

```
artifacts/<run>/<model>/
  gold/<dataset>/<file>.rttm   # reference diarization
  hyp/<dataset>/<file>.rttm    # diarizer hypothesis for the same file
  expected.json                # {"<dataset>": {ten scalars, percent}} — see below
```

`expected.json` carries, per dataset:

| field | meaning |
|---|---|
| `der_full`, `miss`, `fa`, `conf` | corpus DER at collar 0.0 and **its** decomposition (`miss + fa + conf == der_full`) |
| `der_classic`, `miss_classic`, `fa_classic`, `conf_classic` | corpus DER at collar 0.25 and **its** decomposition (`… == der_classic`) |
| `der_full_filemean`, `der_classic_filemean` | the same files under the unweighted file-mean aggregation |

Two decompositions rather than one because a `miss`/`fa`/`conf` triple printed
beside a DER computed under a *different* collar does not sum to it and is not a
breakdown of it. Two aggregations because a DER is not comparable until its
aggregation is named — corpus (`Σerr/Σtotal`) and file-mean differ by 0.334 pp on
the German CALLHOME row, seven times the reproduction tolerance.

`make verify` re-loads gold+hyp, recomputes all ten with `raven_diar.score` (the
same module the Tier-2 runner uses → `raven_eval_core.der` → pyannote.metrics, no
torch/GPU), and asserts each matches within ±0.05 pp.

Artifacts committed before a scalar existed are brought up to the current field
set by `make rescore` (`raven_diar.rescore`), which recomputes from the
artifact's own RTTMs and writes **only keys that are absent** — it refuses to
touch a file whose committed value disagrees with the recomputation, because that
disagreement is precisely what `make verify` exists to surface.

`make analyse ARTIFACT=artifacts/<run>/<model>` reads the same RTTMs past the
aggregate: a seeded bootstrap confidence interval over files, DER by reference
speaker count, the reference overlap fraction, and speaker-aware boundary
offsets. Those are the numbers behind every precision, difficulty and
error-kind claim in `BENCHMARKS.md`; none of them is stored, all of them are
recomputed on demand from what is committed here.

The scorer is `raven_eval_core.flozi_wer` — the flozi-strict pipeline (unidecode +
`alpha2digit` + `wer_standardize_contiguous`, corpus aggregation, raw-text CER).
The Tier-2 runner and this Tier-1 re-scorer import the *same* module, so the
same predictions reproduce the same published `wer_pct` by construction. See the
header of `scripts/verify.py` for why this is not `normalize_strict_de`.

## `_demo/`, `_demo_bleu/` and `_demo_der/` are NOT Raven product numbers

`_demo/demo-model/` (WER), `_demo_bleu/demo-model/` (BLEU) and
`_demo_der/demo-diarizer/` (DER) are hand-built,
self-consistent fixtures that exist only to exercise the verification mechanism
(and to keep `make verify` from ever green-lighting an empty run). Their
`expected.json` is whatever *our own scorer* computed over the committed
predictions/RTTMs — a proof that the machine round-trips, not a claim about
Raven's ASR/diarization quality on any real dataset. The `_demo_der` fixture uses
three tiny synthetic recordings chosen so its DER decomposes cleanly
(confusion 10s + miss 2s + false-alarm 5s over 40s → 42.5% DER at collar 0). The
`_demo_bleu` fixture is written to make the metric's reason visible: the same
eight rows score **14.29 % WER** — which reads as a badly broken transcript — and
**72.24 BLEU**, which reads as a largely correct translation. That gap is the
whole argument for scoring dialect corpora with BLEU.

## How a real artifact gets here (Etappe 4 wired)

The WER harness is `raven_asr` (Tier-2). One run + promote produces a real
committable artifact:

```
make reproduce METRIC=wer DATASET=common_voice_19_0 MODEL=modal/parakeet   # GPU/keys
make promote   RESULTS=results/reproduce/modal-parakeet RUN=2026-07-30      # → artifacts/
make verify                                                                # re-score, no GPU
```

`promote` copies `predictions_*.jsonl` and derives `expected.json` **from the
run's `summary.json`** (never hand-typed), so a committed number can only ever
equal what the scorer produced. The public subsets are `Tuda-De`,
`multilingual_librispeech`, `common_voice_19_0` (from `flozi00/asr-german-mixed-evals`).

## Dataset attribution (reference transcripts)

The committed reference transcripts belong to their datasets, not to Raven:

- `Tuda-De` — **CC-BY-4.0**, attribution required — see `/NOTICE`.
- `common_voice_19_0` — CC0
- `multilingual_librispeech` — **CC-BY-4.0**, attribution required — see `/NOTICE`.

Audio is never redistributed here; a reproducer pulls it from Hugging Face
themselves. Only the reference/hypothesis text labels are committed.
