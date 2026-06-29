"""Greedy decoding, with and without a KV-cache.

This is the central A/B. Without a cache, every new token re-runs attention over
the whole growing prefix — O(T^2) work. With a cache, the prefill happens once
and each new token is O(T). Both paths are greedy (argmax), so they produce
*identical* tokens — verified in tests — which makes the speedup a clean,
like-for-like measurement.
"""

from __future__ import annotations

import time

import torch

from .model import GPT


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, use_cache: bool) -> torch.Tensor:
    return _generate_cached(model, idx, max_new_tokens) if use_cache \
        else _generate_no_cache(model, idx, max_new_tokens)


@torch.no_grad()
def _generate_no_cache(model: GPT, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    block = model.cfg.block_size
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:], use_cache=False)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


@torch.no_grad()
def _generate_cached(model: GPT, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    logits, past = model(idx, use_cache=True)
    nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [nxt]
    for _ in range(max_new_tokens - 1):
        logits, past = model(nxt, past_kvs=past, use_cache=True)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(nxt)
    return torch.cat([idx, *generated], dim=1)


@torch.no_grad()
def timed_generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, use_cache: bool) -> dict:
    """Return timing breakdown: TTFT (first token / prefill) and per-token decode."""
    device = idx.device
    block = model.cfg.block_size

    _sync(device)
    t0 = time.perf_counter()
    if use_cache:
        logits, past = model(idx, use_cache=True)
    else:
        logits, _ = model(idx[:, -block:], use_cache=False)
    nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
    _sync(device)
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    seq = torch.cat([idx, nxt], dim=1)
    decode_ms: list[float] = []
    for _ in range(max_new_tokens - 1):
        _sync(device)
        t = time.perf_counter()
        if use_cache:
            logits, past = model(nxt, past_kvs=past, use_cache=True)
        else:
            logits, _ = model(seq[:, -block:], use_cache=False)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        _sync(device)
        decode_ms.append((time.perf_counter() - t) * 1000.0)

    total_ms = ttft_ms + sum(decode_ms)
    tok_per_s = (max_new_tokens / total_ms * 1000.0) if total_ms > 0 else 0.0
    return {
        "ttft_ms": round(ttft_ms, 3),
        "decode_ms": decode_ms,
        "total_ms": round(total_ms, 3),
        "tokens_per_s": round(tok_per_s, 2),
        "new_tokens": max_new_tokens,
    }
