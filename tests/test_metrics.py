import pytest
import torch

from llminf.metrics import LatencyStats, PeakMemory, percentile, tensor_bytes


def test_percentiles_interpolate():
    s = [1.0, 2.0, 3.0, 4.0]
    assert percentile(s, 50) == pytest.approx(2.5)
    assert percentile(s, 0) == 1.0
    assert percentile(s, 100) == 4.0
    assert LatencyStats.from_samples(s).n == 4


def test_tensor_bytes_counts_what_tensors_address():
    t = torch.zeros(1000, dtype=torch.float32)  # 4000 bytes
    assert tensor_bytes(t) == 4000
    assert tensor_bytes([t, t]) == 4000            # deduped by identity
    assert tensor_bytes(t[:10]) == 40              # a view is worth its own span
    assert tensor_bytes((t, torch.zeros(500, dtype=torch.float32))) == 6000


def test_peak_memory_measures_the_block_not_the_baseline():
    """A pre-existing tensor must not be charged to the block that reads it."""
    resident = torch.zeros(4_000_000, dtype=torch.float32)  # 16 MB, allocated outside
    with PeakMemory("cpu") as pm:
        transient = resident * 2  # one new 16 MB allocation
        del transient
    assert 14.0 < pm.peak_mb < 20.0


def test_peak_memory_is_order_independent():
    """The bug this replaces: an RSS delta made whichever block ran second look free.

    The heavier block must read heavier no matter which order the two run in.
    """

    def light() -> float:
        with PeakMemory("cpu") as pm:
            _ = torch.zeros(250_000, dtype=torch.float32)  # 1 MB
        return pm.peak_mb

    def heavy() -> float:
        with PeakMemory("cpu") as pm:
            _ = torch.zeros(5_000_000, dtype=torch.float32)  # 20 MB
        return pm.peak_mb

    light_first, heavy_second = light(), heavy()
    heavy_first, light_second = heavy(), light()
    assert heavy_second > light_first
    assert heavy_first > light_second
    assert heavy_first == pytest.approx(heavy_second, rel=0.05)
    assert light_first == pytest.approx(light_second, rel=0.05)
