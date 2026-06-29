"""Tensor-parallel building blocks.

The collectives here (all-reduce / all-gather) are backend-agnostic: they run on
**gloo** on CPU — so the correctness test actually executes the distributed code
path locally — and on **NCCL** on multi-GPU, which is the production backend.
"""

from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelMLP,
    get_rank,
    get_world_size,
)

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelMLP",
    "get_rank",
    "get_world_size",
]
