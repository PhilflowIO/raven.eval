"""Speech-detector A/B: which VAD should replace Raven's fixed −40 dB silencedetect.

Raven cuts every speaker track with ffmpeg ``silencedetect`` at −40 dB before it
transcribes (flow.raven ``api/app/providers/_vad.py``). A far-mic speaker at
−50 dBFS falls under that floor and is never sent; a loud room on a listener's
track is sent as speech. This package measures candidate detectors on the same
island planner production uses (:mod:`raven_asr.load.vad`): every arm only
decides, frame by frame, what is speech; pad, merge, minimum island and blocks
are production's.

Arms (flow.raven#11878, research report 2026-11-10 §4):

0. ffmpeg silencedetect at −40 dB — today.
1. Raven's per-track ``adaptive_floor`` (the enclave path's floor) on the
   Parakeet path.
2. Silero VAD v6.2, threshold 0.5 / off 0.35.
3. Silero VAD v6.2, threshold 0.3 / off 0.15, on a copy of the track normalised
   to a fixed level (decision only; the audio sent is untouched).
4. FireRedVAD, offline (non-streaming) model.
5. NVIDIA Frame-VAD Multilingual MarbleNet v2.0.
6. (optional) pyannote segmentation-3.0, any-speaker activity.

Arms are compared at equal false-alarm rate and by ROC-AUC, never at equal
threshold — the detectors' scores live on different scales.

Modules: :mod:`.frames` (the 10 ms grid, hysteresis, metrics, AUC),
:mod:`.arms` (the detectors; heavy engines imported lazily),
:mod:`.testset_a` (synthetic German two-person meetings), :mod:`.crosstalk`
(the cross-track flag), :mod:`.report` (tables), ``__main__`` (CLI).
The engines are not part of this repository's install: the measuring host gets
them from ``raven_asr/vad_ab/requirements*.txt``.
"""
