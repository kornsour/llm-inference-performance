"""Latency percentiles and device-aware memory probes.

On CUDA, memory is the GPU allocator's peak; on CPU it's the process RSS delta.
The headline metrics mirror what an inference team tracks: p50/p95 latency,
tokens/sec, and peak memory.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import torch


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


@dataclass
class LatencyStats:
    n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> LatencyStats:
        return cls(
            n=len(samples_ms),
            p50_ms=round(percentile(samples_ms, 50), 3),
            p95_ms=round(percentile(samples_ms, 95), 3),
            mean_ms=round(sum(samples_ms) / len(samples_ms), 3) if samples_ms else 0.0,
        )


def device_name(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def _rss_bytes() -> int:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kB, macOS reports bytes.
        return ru * 1024 if os.uname().sysname == "Linux" else ru


class PeakMemory:
    """Context manager returning peak memory used (MB) for the wrapped work."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.peak_mb = 0.0

    def __enter__(self) -> PeakMemory:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            self._start_rss = _rss_bytes()
        return self

    def __exit__(self, *exc) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            self.peak_mb = torch.cuda.max_memory_allocated(self.device) / 1e6
        else:
            self.peak_mb = max(0.0, (_rss_bytes() - self._start_rss) / 1e6)
