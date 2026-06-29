"""Batched decoding and throughput scaling.

Serving throughput is dominated by how well requests are batched: one request at
a time leaves the matmul units idle. Here we batch B identical-length sequences
through the cached decoder and measure aggregate tokens/sec as B grows — the
classic batching win, measured.
"""

from __future__ import annotations

import time

import torch

from .model import GPT


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def batched_generate(model: GPT, prompt: torch.Tensor, batch_size: int,
                     max_new_tokens: int) -> torch.Tensor:
    """Replicate `prompt` (1 x T) to a batch and decode with a shared KV-cache."""
    idx = prompt.repeat(batch_size, 1)
    logits, past = model(idx, use_cache=True)
    nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
    out = [nxt]
    for _ in range(max_new_tokens - 1):
        logits, past = model(nxt, past_kvs=past, use_cache=True)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        out.append(nxt)
    return torch.cat([idx, *out], dim=1)


@torch.no_grad()
def throughput(model: GPT, prompt: torch.Tensor, batch_size: int,
               max_new_tokens: int) -> dict:
    device = prompt.device
    _sync(device)
    t0 = time.perf_counter()
    batched_generate(model, prompt, batch_size, max_new_tokens)
    _sync(device)
    elapsed = time.perf_counter() - t0
    total_tokens = batch_size * max_new_tokens
    return {
        "batch_size": batch_size,
        "elapsed_s": round(elapsed, 4),
        "tokens_per_s": round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tokens": total_tokens,
    }
