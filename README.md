<p align="center">
  <img src="assets/image_nano-banana-2_20260731_142913_hero-wordmark-1600w.png" alt="raven.eval" width="100%">
</p>

# raven.eval — reproducible German ASR (WER) and speaker diarization (DER) benchmarks

<p>
  <a href="https://github.com/PhilflowIO/raven.eval/actions/workflows/ci.yml"><img src="https://github.com/PhilflowIO/raven.eval/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
</p>

**Reproduce German speech-recognition (WER) and speaker-diarization (DER) numbers on public data. One command, no GPU, no API keys.**

Our scorer measures **11.15 % DER** on VoxConverse test with
`pyannote/speaker-diarization-community-1`. pyannote's own model card says
**11.2 %**. Run `make verify` and check that yourself in seconds, from committed
data, on a laptop.

```bash
make install      # pinned env via uv (uv.lock)
make verify       # re-scores every published number. no GPU, no keys, no HF token.
```

That command re-computes every row below from committed model outputs. It does not
download a dataset, load a model, or need a license.

## Numbers you can re-score right now

| metric | dataset | number | comparison |
|--------|---------|--------|------------|
| **DER** | VoxConverse test (EN, in-the-wild) | **11.15 %** @ collar 0.0 | pyannote card **11.2 %** ✓ (Δ 0.05 pp) |
| DER | VoxConverse dev (EN) | 7.17 % @ collar 0.0 | easier split |
| **DER** | CALLHOME-de (DE, telephone) | **16.08 %** @ collar 0.25 | between pyannote 3.1 (19.0) & pyannoteAI (8.3) |
| **WER** | Tuda-De / CommonVoice / MLS (DE) | 2.6–4.0 % | flozi dataset-card anchors |

Every row is re-scored on every push by CI. Full tables with per-file provenance
and pinned model/dataset commits: [**`BENCHMARKS.md`**](./BENCHMARKS.md).
Cost is not modelled here; latency is Tier-3 (hardware/network-dependent, not
portably reproducible).

## Why this repository exists

Raven is a German meeting-transcription product that publishes measured WER and DER
numbers. This repository is how you check them without asking us for anything:
public datasets (DER: VoxConverse, CALLHOME-de, AMI — WER: Tuda-De, Common Voice,
MLS, FLEURS, VoxPopuli-de, xSID-audio), pinned model and dataset commits, and the
scoring code Raven runs internally, lifted out and stripped of anything private.

It is packaged so a stranger can check our work. That includes you.

### The dialect rows come with rules

`xsid-bar` (Bavarian) and `xsid-de-control` (Standard German) are a matched pair
from Zenodo record [21605015](https://zenodo.org/records/21605015), DOI
`10.5281/zenodo.21605015` v0.2. **One** person recorded both varieties, reading
the same sentences aloud. That makes the pair a probe — does a model collapse on
Bavarian at all — and not a Bavarian benchmark: only the *delta* between the two
ids is a statement about dialect; either number alone describes one voice.

So: no winner mark on a dialect row, no aggregate spanning two dialect areas, and
dialect ids stay out of any cross-dataset average. The rules are written out in
`benchmark.config.yaml` under `dialect_publication_rules` and carried in code by
`WerDatasetSpec.eligible_for_aggregate`. The corpus also carries an authors'
condition that is **not** part of its CC BY-SA licence and therefore does not
travel on its own — it is reproduced verbatim in [`NOTICE`](./NOTICE).

## How do you know the scorer itself is correct?

You point it at a benchmark where the answer is already public. raven.eval
reproduces pyannote's published VoxConverse test DER of 11.2 % to within 0.05 pp
(measured: 11.15 % at `collar=0.0`, overlapping speech included, no GPU required).
The delta is computed from committed RTTMs by `make verify`, with no gated model.

An instrument that reproduces the vendor's own number on the vendor's own benchmark
is an instrument you can hold us to on the datasets nobody publishes. CALLHOME-de
is one of those: a German-language telephone diarization benchmark, where
`pyannote/speaker-diarization-community-1` scores **16.08 % DER** at collar 0.25.

## Tier 1, 2, 3: what you can re-run, and what you can't

Not every number is portably reproducible. We are explicit about which is which,
and we make the reproducible ones *actually* reproducible:

- **Tier 1 — verify in seconds, no GPU, no API keys.**
  We commit the model outputs (per-utterance transcripts for WER; gold + hypothesis
  RTTMs for DER) next to the gold references. `make verify` re-scores them and
  reproduces the published tables — WER **and** DER (both collars, each with its
  own miss/FA/conf, plus both aggregation conventions) — using only
  `pyannote.metrics` and `jiwer`, no torch, no GPU. CI runs it on every push, so a
  number can't silently drift from its committed data. `make analyse
  ARTIFACT=artifacts/<run>/<model>` goes past the aggregate on the same data:
  bootstrap confidence intervals over files, DER by reference speaker count, the
  reference overlap fraction, and speaker-aware boundary offsets.

- **Tier 2 — full re-run on public data (your own keys / GPU).**
  `make reproduce` downloads a public dataset, runs the pinned model, and scores it.
  WER on the `flozi00/asr-german-mixed-evals` subsets plus FLEURS (`de_de`), MLS
  German and VoxPopuli German — every id in `benchmark.config.yaml` resolves to a
  loader here, and a test fails the build if that ever stops being true. DER
  (`make reproduce METRIC=der`) with `pyannote/speaker-diarization-community-1` on
  VoxConverse / CALLHOME-de / AMI (needs an HF token, the gated model license, a GPU,
  and shared FFmpeg libs — see [`docs/TIER2-DER-KEYS.md`](./docs/TIER2-DER-KEYS.md)).
  If it doesn't match what we publish, open an issue.

- **Tier 3 — transparency only (not portably reproducible).**
  Latency / TTFT and any private-fixture result are documented with methodology and an
  explicit reason they can't be re-run elsewhere. No hidden numbers, no pretense.

## Every number states the rules it was computed under

The exact scoring rules live in [`benchmark.config.yaml`](./benchmark.config.yaml).
Every metric declared there has an implementation in `raven_eval_core`, and every
implementation is declared there — `tests/test_metric_contract.py` fails the build
in both directions, and for BLEU it compares the pinned conventions field by field.
DER is reported at **both** `collar=0.0` and `collar=0.25` (the two conventions are
not comparable), so our numbers line up with the pyannote model cards and the ETH
benchmark paper (arXiv 2509.26177). DER is always dataset-relative: the same model
reads anywhere from ~7 % to ~27 % depending on the dataset. A bare "DER" without a
stated dataset, collar and overlap rule is noise.

## What is word error rate (WER)?

Word error rate (WER) is the standard metric for speech-recognition accuracy: the
number of substitutions, deletions and insertions needed to turn the transcript into
the reference, divided by the number of reference words. Lower is better; text
normalization changes the result, which is why this repo uses one shared normalizer
(`flozi_wer.py`) for the runner and the re-scorer. On the German public sets here,
the measured range is 2.6–4.0 % WER.

**What is a good word error rate?** On clean read speech, strong models score under
5 %; on spontaneous, noisy or telephone speech, 10–20 % is common. Compare WER only
under the same normalization and dataset.

**Can WER be above 100 %?** Yes — insertions count as errors, so a hypothesis much
longer than the reference can push WER past 100 %.

## What is BLEU, and why does a speech benchmark report it?

BLEU (bilingual evaluation understudy) scores how many n-grams — word sequences of
length 1 to 4 — a candidate text shares with a reference, with a brevity penalty
for output that is too short. It runs 0–100; **higher is better**, the opposite
direction from WER and DER.

It is here because not every corpus is transcription-shaped. Swiss-German dialect
speech is *spoken* in dialect and *transcribed* in standard German: the reference
is a translation of what was said, not a record of it. A correct output can
legitimately differ in word choice and order ("gäll" → "nicht wahr", "lueg" →
"schau"), and WER charges every one of those as an error. Such a dataset is scored
**bleu+wer**, with BLEU as the headline of the pair.

The committed `artifacts/_demo_bleu/` fixture shows the gap on eight rows: the
same output reads as **14.29 % WER** (a badly broken transcript) and **72.24
BLEU** (a largely correct translation).

**Which BLEU?** The word alone is not a number — tokenizer, case handling and
smoothing each move it by whole points, which is why cross-paper BLEU comparisons
were unreliable for years (Post 2018, *A Call for Clarity in Reporting BLEU
Scores*). We score with **sacrebleu** under conventions pinned in
[`benchmark.config.yaml`](./benchmark.config.yaml) — `13a` tokenizer,
case-sensitive, `exp` smoothing, corpus-level — and publish sacrebleu's signature
(`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:…`) next to every number.
That string, not the word "BLEU", is what makes the number comparable. Per-sentence
BLEU exists in `raven_eval_core.bleu` as a diagnostic and is never published.

## What is diarization error rate (DER)?

Diarization error rate (DER) measures how accurately a system answers "who spoke
when". It is the sum of three components, as a fraction of total speech time:
**missed speech**, **false alarm speech**, and **speaker confusion** (after
Hungarian mapping of speaker labels). Lower is better.

**What is a good DER?** On clean benchmark data, current models reach 5–10 %; on
real-world telephone or meeting audio, 15–25 % is typical. The collar (a forgiveness
window around speaker boundaries, usually 0.0 or 0.25 s) and the overlap rule change
the number substantially — never compare DER across different rules.

## Repository layout

- **`raven_eval_core/`** — the standalone, secret-free metric core. `der.py`
  (Diarization Error Rate via `pyannote.metrics`, collar + overlap parametrized,
  Hungarian mapping, RTTM I/O), `flozi_wer.py` (the single source of truth for
  published German WER — same normalization for the runner and the re-scorer, so they
  can't drift) and `bleu.py` (corpus BLEU via `sacrebleu` under pinned conventions,
  for translation-shaped corpora). No hardcoded names, no private data.
- **`raven_asr/`** — Tier-2 WER harness: pull a public HF subset → run a model
  (Modal / OpenAI / Deepgram / Mistral / vLLM adapter) → score → `predictions_*.jsonl`.
- **`raven_diar/`** — Tier-2 DER harness: prepare a public diarization dataset → run
  the pinned `pyannote-community-1` diarizer → score DER at both collars → gold/hyp
  RTTMs. Heavy deps behind the `diar` extra. Optional `make dscore-check` cross-checks
  against `nryant/dscore`.
- **`artifacts/`** — the committed proof: per-file references + hypotheses +
  `expected.json` for every published number, re-scored by `make verify`.
- **`scripts/verify.py`** — the Tier-1 re-scorer.

## What these numbers are not

- **Public-dataset numbers are not Raven's private-meeting numbers.** Raven's internal
  eval runs on real customer meetings whose audio can't be published (consent); those
  are a separate, non-public datapoint. The numbers *here* are on public datasets and
  will differ — that's expected, and we don't blur them.
- **The demo fixtures** (`artifacts/_demo/`, `artifacts/_demo_bleu/`,
  `artifacts/_demo_der/`) prove the
  re-score mechanism end-to-end; they are **not** Raven product numbers.

## License and data attribution

Code is **MIT**. Dataset licenses are separate and belong to their owners — see
[`NOTICE`](./NOTICE) for per-dataset attribution (Tuda-De / MLS / FLEURS = CC-BY,
Common Voice / VoxPopuli = CC0, VoxConverse labels = CC-BY, xSID-audio = CC-BY-SA
**plus a no-speech-synthesis condition the licence does not carry**, …). **We never redistribute restricted audio** —
only the tooling and, where the license permits, the reference/hypothesis label files.
