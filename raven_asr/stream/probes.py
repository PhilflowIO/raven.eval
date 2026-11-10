"""What the GPU and the engine did during a step.

GPU: ``nvidia-smi`` sampled every 500 ms on the host the harness runs on (it
runs next to the server: the realtime protocol is paced in real time, and a
network hop would only add its own latency). Utilization and memory are of the
whole card, other tenants included; ``apps`` lists every process on the card at
the end of the step with its own memory, so the server's share is on record.

Engine: vLLM's Prometheus gauges (``/metrics``) when the server is vLLM —
running and waiting requests and KV-cache use. ``waiting > 0`` means sessions
queued in the scheduler: the engine, not the GPU, was the cap.
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Self

import httpx

GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage",
}


def parse_gauges(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in GAUGES:
            try:
                out[GAUGES[name]] = out.get(GAUGES[name], 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


class Sampler:
    """Samples GPU (and optionally vLLM gauges) in a background thread."""

    def __init__(self, metrics_url: str | None = None, interval_s: float = 0.5,
                 stats_url: str | None = None) -> None:
        self.metrics_url = metrics_url
        self.stats_url = stats_url
        self.cpu: list[float] = []  # whole-host CPU busy %, per interval
        self.server_rss: list[float] = []
        self.interval_s = interval_s
        self.gpu: list[tuple[float, float]] = []  # (util %, memory used MiB)
        self.engine: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        client = httpx.Client(timeout=2.0) if (self.metrics_url or self.stats_url) else None
        prev = cpu_times()
        while not self._stop.is_set():
            t = time.monotonic()
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits", "-i", "0"],
                    capture_output=True, text=True, timeout=5, check=False,
                ).stdout.strip()
                util, mem = (float(x) for x in out.split(","))
                self.gpu.append((util, mem))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            cur = cpu_times()
            if prev is not None and cur is not None and cur[1] > prev[1]:
                self.cpu.append(100.0 * (cur[0] - prev[0]) / (cur[1] - prev[1]))
            prev = cur
            if client is not None and self.metrics_url:
                try:
                    self.engine.append(parse_gauges(client.get(self.metrics_url).text))
                except httpx.HTTPError:
                    pass
            if client is not None and self.stats_url:
                try:
                    self.server_rss.append(float(client.get(self.stats_url).json()["rss_max_mib"]))
                except (httpx.HTTPError, KeyError, ValueError):
                    pass
            self._stop.wait(max(0.0, self.interval_s - (time.monotonic() - t)))

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict:
        utils = [u for u, _ in self.gpu]
        mems = [m for _, m in self.gpu]
        out: dict = {
            "samples": len(self.gpu),
            "util_mean_pct": round(sum(utils) / len(utils), 1) if utils else None,
            "util_max_pct": max(utils) if utils else None,
            "memory_used_max_mib": max(mems) if mems else None,
            "apps": gpu_apps(),
            "cpu_util_mean_pct": round(sum(self.cpu) / len(self.cpu), 1) if self.cpu else None,
            "cpu_util_max_pct": round(max(self.cpu), 1) if self.cpu else None,
        }
        if self.stats_url is not None:
            out["server_rss_max_mib"] = max(self.server_rss) if self.server_rss else None
        if self.metrics_url is not None:
            for key in GAUGES.values():
                vals = [e[key] for e in self.engine if key in e]
                out[f"engine_{key}_max"] = max(vals) if vals else None
        return out


def parse_proc_stat(text: str) -> tuple[float, float] | None:
    """``(busy, total)`` jiffies from the aggregate line of /proc/stat."""
    for line in text.splitlines():
        if line.startswith("cpu "):
            v = [float(x) for x in line.split()[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0.0)  # idle + iowait
            return sum(v) - idle, sum(v)
    return None


def cpu_times() -> tuple[float, float] | None:
    try:
        with open("/proc/stat") as fh:
            return parse_proc_stat(fh.read())
    except OSError:
        return None


def gpu_apps() -> list[dict]:
    """Every process on GPU 0 with its memory (MiB)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name,used_memory",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    apps = []
    for line in out.strip().splitlines():
        name, _, mem = line.rpartition(",")
        try:
            apps.append({"process": name.strip().rsplit("/", 1)[-1], "memory_mib": float(mem)})
        except ValueError:
            continue
    return apps


def gpu_identity() -> dict:
    """Name, memory and driver of GPU 0, for the hardware record."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        name, mem, driver = (x.strip() for x in out.split(","))
        return {"name": name, "memory_total_mib": float(mem), "driver": driver}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"name": None, "memory_total_mib": None, "driver": None}
