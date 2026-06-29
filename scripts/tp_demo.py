"""Tensor-parallel demo + all-reduce micro-benchmark.

Run on CPU (gloo):
    uv run torchrun --nproc_per_node=2 scripts/tp_demo.py

Run on multi-GPU (NCCL is selected automatically):
    torchrun --nproc_per_node=2 scripts/tp_demo.py

It builds a tensor-parallel MLP, checks it matches the single-process result, and
times an all-reduce across message sizes — the communication that tensor
parallelism adds per layer (and which NCCL accelerates on real hardware).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from llminf.distributed.tensor_parallel import (  # noqa: E402
    TensorParallelMLP,
    reference_mlp,
)


def main() -> None:
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)

    n, hidden = 1024, 4096
    torch.manual_seed(0)
    fc_w, fc_b = torch.randn(hidden, n, device=device), torch.randn(hidden, device=device)
    pj_w, pj_b = torch.randn(n, hidden, device=device), torch.randn(n, device=device)
    torch.manual_seed(123)
    x = torch.randn(8, n, device=device)

    out = TensorParallelMLP(fc_w, fc_b, pj_w, pj_b)(x)
    if rank == 0:
        ref = reference_mlp(x, fc_w, fc_b, pj_w, pj_b)
        max_err = (out - ref).abs().max().item()
        print(f"[rank0] backend={backend} world={world} TP-vs-reference max_err={max_err:.2e}")

    # All-reduce latency across message sizes (the per-layer TP communication cost).
    for numel in (1 << 16, 1 << 18, 1 << 20, 1 << 22):
        t = torch.randn(numel, device=device)
        for _ in range(3):  # warmup
            dist.all_reduce(t)
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        iters = 20
        for _ in range(iters):
            dist.all_reduce(t)
        if use_cuda:
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / iters * 1000
        if rank == 0:
            mb = numel * 4 / 1e6
            print(f"[rank0] all_reduce {mb:6.2f} MB  ->  {ms:7.3f} ms  "
                  f"({mb / (ms / 1000):.1f} MB/s)")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and "RANK" in os.environ:
        print("[rank0] done")


if __name__ == "__main__":
    main()
