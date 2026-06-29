"""Tensor-parallel correctness: 2 gloo processes must match the single-process MLP.

This actually launches the distributed code path (collectives over gloo), so the
same logic that uses NCCL on multi-GPU is exercised in CI on CPU.
"""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llminf.distributed.tensor_parallel import TensorParallelMLP, reference_mlp


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(rank: int, world: int, port: int, n: int, hidden: int) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method=f"tcp://127.0.0.1:{port}")
    try:
        # Every rank builds the SAME full weights (fixed seed); each shards by rank.
        torch.manual_seed(0)
        fc_w, fc_b = torch.randn(hidden, n), torch.randn(hidden)
        pj_w, pj_b = torch.randn(n, hidden), torch.randn(n)
        torch.manual_seed(123)
        x = torch.randn(3, n)

        tp_out = TensorParallelMLP(fc_w, fc_b, pj_w, pj_b)(x)
        if rank == 0:
            ref = reference_mlp(x, fc_w, fc_b, pj_w, pj_b)
            torch.testing.assert_close(tp_out, ref, rtol=1e-4, atol=1e-4)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_tensor_parallel_matches_single_process():
    mp.spawn(_worker, args=(2, _free_port(), 64, 256), nprocs=2, join=True)
