"""Speaker-diarization (DER) evaluation harness — the Tier-2 mirror of ``raven_asr``.

Pulls a public diarization dataset (VoxConverse / CALLHOME-de / AMI), runs a
pinned diarizer (pyannote community-1) to a hypothesis RTTM, and scores DER
against the gold RTTM at both collars via ``raven_eval_core.der``.

The heavy diarizer deps (``pyannote.audio`` + ``torch``) live behind the ``diar``
optional extra and are imported lazily inside the adapter — importing this
package, ``raven_diar.score``, or the Tier-1 re-scorer stays GPU-free and needs
only ``raven_eval_core`` (pyannote.metrics).
"""

__version__ = "0.1.0"
