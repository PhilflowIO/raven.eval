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
  expected.json                # {"<dataset>": {"der_full","der_classic","miss","fa","conf"}}  (percent)
```

`make verify` re-loads gold+hyp, recomputes corpus DER at both collars (0.0 full /
0.25 classic) with `raven_eval_core.der` (pyannote.metrics — no torch/GPU), and
asserts each field matches within ±0.05 pp. `miss + fa + conf == der_full` by
construction (collar-0 decomposition). A committed real DER row is **pending the
Etappe-5 model run** (needs a GPU + the gated pyannote model); `_demo_der/` proves
the mechanism today.

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
