"""Offline load test: production-shaped STT requests from long recordings.

Raven transcribes a meeting after it ends, per speaker track, as compacted VAD
blocks (sparse tracks) or ~600 s windows (dense tracks), four requests in flight
per worker, each under a read budget of min(300, 60 + audio s). This package
reproduces those request shapes from long public recordings
(:mod:`raven_asr.load.vad`, :mod:`raven_asr.load.workload`), sweeps requests in
flight against one server (:mod:`raven_asr.load.sweep`), and scores the
stitched transcripts long-form (:mod:`raven_asr.load.score`,
:mod:`raven_asr.load.timing`). Entry point: ``python -m raven_asr.load``.
"""
