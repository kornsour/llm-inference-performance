"""Latency percentiles and device-aware memory probes.

The headline metrics mirror what an inference team tracks: p50/p95 latency,
tokens/sec, and peak memory.

"Peak memory" here means one specific thing on both devices: the high-water mark
of tensor bytes allocated *inside* the measured block, above whatever was already
resident (model weights, the prompt) when the block began. Subtracting that
baseline is what makes two runs in the same process comparable.
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass

import torch

try:  # pragma: no cover - import guard
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError:  # pragma: no cover - very old torch
    TorchDispatchMode = None


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


def tensor_bytes(obj) -> int:
    """Logical bytes held by every distinct tensor reachable in `obj`.

    Logical (`numel x element_size`), not storage size: a tensor that is a view
    into a larger buffer is worth what it addresses, not what its base happens
    to pin. Deduped by tensor identity so a list containing the same tensor
    twice is not double-counted.
    """
    seen: dict[int, int] = {}
    for t in _flatten(obj, []):
        seen[id(t)] = t.numel() * t.element_size()
    return sum(seen.values())


def _flatten(obj, out: list[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple, set)):
        for o in obj:
            _flatten(o, out)
    elif isinstance(obj, dict):
        for o in obj.values():
            _flatten(o, out)
    return out


def _storage_of(t: torch.Tensor):
    try:
        return t.untyped_storage()
    except Exception:  # meta / sparse / fake tensors have no dense storage
        return None


class _AllocationProbe(TorchDispatchMode if TorchDispatchMode else object):
    """Peak live bytes of tensors allocated by ops running inside the block.

    Every dispatched op is checked for outputs whose storage is not aliased by
    any of its inputs — i.e. genuinely new allocations. Each is counted once and
    decremented again by a weakref finalizer when its storage dies, so the
    counter tracks live bytes and `peak` is their high-water mark. Skipping
    aliased outputs is what keeps pre-existing tensors out of the total: a view
    of a model weight would otherwise charge the whole weight to the block.

    This walks Python per aten op, so it belongs in a dedicated untimed pass —
    never wrapped around the loop whose latency is being reported.
    """

    def __init__(self) -> None:
        super().__init__()
        self.live = 0
        self.peak = 0
        self._tracked: set[int] = set()

    def _release(self, key: int, nbytes: int) -> None:
        if key in self._tracked:
            self._tracked.discard(key)
            self.live -= nbytes

    def _note(self, t: torch.Tensor) -> None:
        st = _storage_of(t)
        if st is None:
            return
        key = id(st)  # torch preserves one Python storage object per StorageImpl
        if key in self._tracked:
            return
        nbytes = st.nbytes()
        self._tracked.add(key)
        self.live += nbytes
        self.peak = max(self.peak, self.live)
        weakref.finalize(st, self._release, key, nbytes)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        inputs = {
            st.data_ptr()
            for t in _flatten((args, kwargs), [])
            if (st := _storage_of(t)) is not None
        }
        out = func(*args, **kwargs)
        for t in _flatten(out, []):
            st = _storage_of(t)
            if st is not None and st.data_ptr() not in inputs:
                self._note(t)
        return out


class PeakMemory:
    """Context manager reporting peak memory (MB) allocated by the wrapped work.

    On CUDA this is the allocator's own high-water mark, measured as a delta
    above the bytes already allocated on entry. On CPU there is no allocator
    peak counter, so `_AllocationProbe` tracks tensor storages directly.

    Deliberately *not* a process-RSS delta, which is what this used to be: RSS is
    a high-water mark that never comes back down, so the first block measured in
    a process absorbs all the heap growth and every later block reads ~0. That
    made whichever variant ran second look almost free regardless of what it
    actually cost.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.peak_mb = 0.0
        self._probe: _AllocationProbe | None = None

    def __enter__(self) -> PeakMemory:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            self._baseline = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        elif TorchDispatchMode is not None:
            self._probe = _AllocationProbe()
            self._probe.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(self.device) - self._baseline
        elif self._probe is not None:
            self._probe.__exit__(*exc)
            peak = self._probe.peak
            self._probe = None
        else:  # pragma: no cover - no dispatch-mode support
            peak = 0
        self.peak_mb = max(0.0, peak / 1e6)
