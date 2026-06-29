"""Megatron-style tensor parallelism for an MLP block.

A linear layer is sharded across `world_size` ranks:

    ColumnParallelLinear  shards the OUTPUT dim. Each rank produces a slice of
                          the output; no communication in forward.
    RowParallelLinear     shards the INPUT dim. Each rank computes a partial sum
                          over its input slice; an all-reduce sums them.

Composed as (column-parallel c_fc) -> GELU -> (row-parallel c_proj), the MLP
needs exactly one all-reduce per forward — the canonical pattern. On GPU the
all-reduce runs over NCCL; here it runs over gloo so the path is unit-tested.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _all_reduce(t: torch.Tensor) -> torch.Tensor:
    if get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _all_gather_last(t: torch.Tensor) -> torch.Tensor:
    world = get_world_size()
    if world == 1:
        return t
    parts = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(parts, t.contiguous())
    return torch.cat(parts, dim=-1)


class ColumnParallelLinear(nn.Module):
    """y = x @ Wᵀ + b, with W's output dim sharded across ranks."""

    def __init__(self, full_weight: torch.Tensor, full_bias: torch.Tensor | None = None,
                 gather_output: bool = False) -> None:
        super().__init__()
        out_features = full_weight.shape[0]
        world, rank = get_world_size(), get_rank()
        assert out_features % world == 0, "out_features must be divisible by world_size"
        shard = out_features // world
        sl = slice(rank * shard, (rank + 1) * shard)
        self.gather_output = gather_output
        self.weight = nn.Parameter(full_weight[sl, :].clone())
        self.bias = nn.Parameter(full_bias[sl].clone()) if full_bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        return _all_gather_last(y) if self.gather_output else y


class RowParallelLinear(nn.Module):
    """y = x @ Wᵀ + b, with W's input dim sharded; input is already sharded."""

    def __init__(self, full_weight: torch.Tensor, full_bias: torch.Tensor | None = None) -> None:
        super().__init__()
        in_features = full_weight.shape[1]
        world, rank = get_world_size(), get_rank()
        assert in_features % world == 0, "in_features must be divisible by world_size"
        shard = in_features // world
        sl = slice(rank * shard, (rank + 1) * shard)
        self.weight = nn.Parameter(full_weight[:, sl].clone())
        # Bias is added once, after the all-reduce (only rank 0 holds it).
        self.bias = nn.Parameter(full_bias.clone()) if (full_bias is not None and rank == 0) else None

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x_shard, self.weight)
        out = _all_reduce(partial)
        if self.bias is not None:
            out = out + self.bias
        return out


class TensorParallelMLP(nn.Module):
    """Column-parallel c_fc -> GELU -> row-parallel c_proj (one all-reduce)."""

    def __init__(self, fc_weight: torch.Tensor, fc_bias: torch.Tensor | None,
                 proj_weight: torch.Tensor, proj_bias: torch.Tensor | None) -> None:
        super().__init__()
        self.c_fc = ColumnParallelLinear(fc_weight, fc_bias, gather_output=False)
        self.c_proj = RowParallelLinear(proj_weight, proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


def reference_mlp(x: torch.Tensor, fc_weight: torch.Tensor, fc_bias: torch.Tensor | None,
                  proj_weight: torch.Tensor, proj_bias: torch.Tensor | None) -> torch.Tensor:
    """Single-process reference the tensor-parallel result must match."""
    return F.linear(F.gelu(F.linear(x, fc_weight, fc_bias)), proj_weight, proj_bias)
