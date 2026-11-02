# Micro-batcher for the NeMo server (server.py).
#
# WHY THIS EXISTS: the server used to run every request's `model.transcribe`
# straight from the uvicorn threadpool, with batch_size=1 and no lock. Under a
# parallel client that is a race on one shared model (NeMo's transcribe swaps
# decoding/dataloader state on the module), and even when it happens not to
# corrupt anything it gives no throughput, because the GPU still sees one clip
# per forward pass.
#
# The fix is one owner for the model: a single inference thread drains a queue,
# groups up to `max_batch` items (or whatever arrived within `max_wait_s` of
# the first one) into ONE batched call, and resolves each caller's Future.
# Nothing else may touch the model — that removes the race by construction
# instead of by a lock that would also serialise the batching away.
#
# Kept free of NeMo/FastAPI imports so the contract tests can drive it with a
# mock model (tests/test_serving_nemo_batching.py).
"""Single-owner micro-batching queue: collect → one batched call → fan out."""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("raven-stt-bench.batcher")

_STOP = object()  # queue sentinel for stop(); distinct from "nothing queued" (None)


@dataclass
class _Job:
    item: Any
    solo: bool
    future: Future = field(default_factory=Future)


@dataclass
class BatchStats:
    batches: int = 0
    items: int = 0
    max_seen: int = 0
    fallbacks: int = 0

    def snapshot(self) -> dict:
        return {
            "batches": self.batches,
            "items": self.items,
            "mean_batch_size": round(self.items / self.batches, 2) if self.batches else None,
            "max_batch_size_seen": self.max_seen,
            "per_item_fallbacks": self.fallbacks,
        }


class MicroBatcher:
    """Owns all calls into `run_batch`; callers get a Future per item.

    Each Future resolves to ``(result, batch_size)`` — the size of the call
    that produced it, so the server can report how it was actually batched.

    `run_batch(items)` must return one result per item, in order. If it raises
    for a batch of more than one item, every item is retried ALONE, so one
    poisoned clip fails only its own request (its exception lands on its own
    Future) instead of taking the whole batch down with it.

    `is_solo(item)` marks items that must never share a call (long audio: the
    model's own long-form path, e.g. Canary's AED chunking, only engages for a
    single input — batching would silently change the numbers).
    """

    def __init__(
        self,
        run_batch: Callable[[Sequence[Any]], Sequence[Any]],
        *,
        max_batch: int = 16,
        max_wait_s: float = 0.02,
        is_solo: Callable[[Any], bool] | None = None,
        name: str = "nemo-infer",
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self._run_batch = run_batch
        self.max_batch = max_batch
        self.max_wait_s = max(0.0, max_wait_s)
        self._is_solo = is_solo or (lambda _item: False)
        self._queue: queue.Queue[Any] = queue.Queue()
        # Jobs pulled while collecting a batch that could not join it (solo
        # items). They run next, ahead of anything newer, so order is fair.
        self._carry: collections.deque[Any] = collections.deque()
        self._lock = threading.Lock()
        self.stats = BatchStats()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._started = False

    # ── public API ──────────────────────────────────────────────────────────
    def start(self) -> None:
        with self._lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def submit(self, item: Any) -> Future:
        if not self._started:
            self.start()
        job = _Job(item=item, solo=bool(self._is_solo(item)))
        self._queue.put(job)
        return job.future

    def run(self, item: Any, timeout: float | None = None) -> Any:
        """Blocking convenience for callers already on a worker thread."""
        return self.submit(item).result(timeout=timeout)

    def stop(self, timeout: float = 5.0) -> None:
        self._queue.put(_STOP)
        if self._started:
            self._thread.join(timeout=timeout)

    # ── worker ──────────────────────────────────────────────────────────────
    def _next(self, block: bool, timeout: float | None) -> Any:
        if self._carry:
            return self._carry.popleft()
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def _collect(self, first: _Job) -> list[_Job]:
        batch = [first]
        if first.solo or self.max_batch == 1:
            return batch
        deadline = time.monotonic() + self.max_wait_s
        while len(batch) < self.max_batch:
            remaining = deadline - time.monotonic()
            # Past the wait budget we still take whatever is ALREADY queued
            # (non-blocking) — a full queue should fill the batch, not wait.
            if remaining > 0:
                job = self._next(block=True, timeout=remaining)
            else:
                job = self._next(block=False, timeout=None)
            if job is None:
                break
            if job is _STOP:
                self._carry.append(job)  # honour the stop after this batch
                break
            if job.solo:
                # Defer, never drop: it runs right after this batch.
                self._carry.append(job)
                break
            batch.append(job)
        return batch

    def _loop(self) -> None:
        while True:
            first = self._next(block=True, timeout=None)
            if first is None:
                continue
            if first is _STOP:
                return
            batch = self._collect(first)
            self._execute(batch)

    def _execute(self, batch: list[_Job]) -> None:
        items = [j.item for j in batch]
        self.stats.batches += 1
        self.stats.items += len(batch)
        self.stats.max_seen = max(self.stats.max_seen, len(batch))
        try:
            results = list(self._run_batch(items))
            if len(results) != len(items):
                raise RuntimeError(
                    f"batched call returned {len(results)} results for {len(items)} inputs"
                )
        except Exception as exc:  # noqa: BLE001 — isolate, then re-raise per item
            if len(batch) == 1:
                batch[0].future.set_exception(exc)
                return
            logger.warning(
                "batch of %d failed (%s: %s) — retrying each item alone",
                len(batch), type(exc).__name__, exc,
            )
            self.stats.fallbacks += 1
            for job in batch:
                try:
                    (single,) = list(self._run_batch([job.item]))
                except Exception as item_exc:  # noqa: BLE001
                    job.future.set_exception(item_exc)
                else:
                    job.future.set_result((single, 1))
            return
        for job, result in zip(batch, results):
            job.future.set_result((result, len(batch)))
