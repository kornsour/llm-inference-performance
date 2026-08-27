"""Static batching, ragged prompts, and throughput scaling.

Serving throughput is dominated by how well requests are batched: one request
at a time leaves the matmul units idle. This module batches B *independent*
sequences — different prompts, different lengths, left-padded to a common
width with a correct attention mask — through the cached decoder and measures
aggregate tokens/sec as B grows.

This is **static batching**: the whole batch is assembled up front and decoded
together to a fixed length, the way it is here in `bench_batching` — not
**continuous batching** (admitting/evicting sequences mid-flight as they finish
at different times), which is what production serving frameworks actually run.
Call it by its name so the two aren't conflated: the scaling measured below is
real, and it comes from matmul utilization at bigger batch sizes, not from a
scheduler doing anything dynamic.
"""

from __future__ import annotations

import time

import torch

from .model import GPT


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def left_pad(prompts: list[torch.Tensor], pad_id: int = 0
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ragged `(1, T_i)` prompts into a `(B, T_max)` batch, padding on the
    left so every row's *last* column is always a real (non-pad) token — the
    position the decode loop reads `logits[:, -1]` from on every step.

    Returns `(idx, attention_mask)`; `attention_mask` is `True` where `idx`
    holds a real token and `False` where it holds padding.
    """
    if not prompts:
        raise ValueError("left_pad requires at least one prompt")
    device = prompts[0].device
    dtype = prompts[0].dtype
    lengths = [p.reshape(-1).shape[0] for p in prompts]
    B, T_max = len(prompts), max(lengths)
    idx = torch.full((B, T_max), pad_id, dtype=dtype, device=device)
    mask = torch.zeros((B, T_max), dtype=torch.bool, device=device)
    for i, (p, length) in enumerate(zip(prompts, lengths, strict=True)):
        idx[i, T_max - length:] = p.reshape(-1)
        mask[i, T_max - length:] = True
    return idx, mask


def make_ragged_prompts(vocab_size: int, batch_size: int, min_len: int, max_len: int,
                        seed: int, device: torch.device | str = "cpu") -> list[torch.Tensor]:
    """`batch_size` prompts with lengths spread evenly over `[min_len, max_len]`
    (deterministic from `seed`) — a batch that actually exercises padding and
    masking, instead of `batch_size` copies of one prompt."""
    if min_len < 1 or max_len < min_len:
        raise ValueError(f"need 1 <= min_len ({min_len}) <= max_len ({max_len})")
    g = torch.Generator(device="cpu").manual_seed(seed)
    if batch_size == 1:
        lengths = [max_len]
    else:
        span = max_len - min_len
        lengths = [min_len + round(i * span / (batch_size - 1)) for i in range(batch_size)]
    return [torch.randint(0, vocab_size, (1, length), generator=g).to(device)
            for length in lengths]


@torch.no_grad()
def batched_generate(model: GPT, prompts: torch.Tensor | list[torch.Tensor],
                     batch_size: int | None = None, max_new_tokens: int = 0,
                     pad_id: int = 0) -> torch.Tensor:
    """Decode a batch to `max_new_tokens` with a shared KV-cache.

    `prompts` is either a ragged `list` of `(1, T_i)` prompts (left-padded here
    with a correct attention mask — the general, ragged-batch path) or a single
    `(1, T)` prompt replicated `batch_size` times (the degenerate, all-rows-
    identical case, kept for the KV-cache-cost and shape-only tests where the
    ragged machinery would just add noise).
    """
    if isinstance(prompts, list):
        idx, mask = left_pad(prompts, pad_id)
    else:
        if batch_size is None:
            raise ValueError("batch_size is required when prompts is a single tensor")
        idx, mask = prompts.repeat(batch_size, 1), None

    logits, past = model(idx, use_cache=True, attention_mask=mask)
    nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
    out = [nxt]
    for _ in range(max_new_tokens - 1):
        if mask is not None:
            mask = torch.cat([mask, torch.ones_like(nxt, dtype=torch.bool)], dim=1)
        logits, past = model(nxt, past_kvs=past, use_cache=True, attention_mask=mask)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        out.append(nxt)
    return torch.cat([idx, *out], dim=1)


@torch.no_grad()
def throughput(model: GPT, prompts: torch.Tensor | list[torch.Tensor],
              batch_size: int | None = None, max_new_tokens: int = 0,
              pad_id: int = 0) -> dict:
    """Aggregate tokens/sec decoding `prompts` as one static batch. See
    `batched_generate` for what `prompts` may be."""
    is_ragged = isinstance(prompts, list)
    device = (prompts[0] if is_ragged else prompts).device
    b = len(prompts) if is_ragged else batch_size

    _sync(device)
    t0 = time.perf_counter()
    batched_generate(model, prompts, batch_size=b, max_new_tokens=max_new_tokens, pad_id=pad_id)
    _sync(device)
    elapsed = time.perf_counter() - t0
    total_tokens = b * max_new_tokens
    result = {
        "batch_size": b,
        "elapsed_s": round(elapsed, 4),
        "tokens_per_s": round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tokens": total_tokens,
    }
    if is_ragged:
        result["prompt_lengths"] = [p.reshape(-1).shape[0] for p in prompts]
    return result
