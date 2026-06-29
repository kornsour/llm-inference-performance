"""Benchmark harness.

Produces the headline numbers an inference team tracks — p50/p95 latency,
tokens/sec, peak memory — and the before/after deltas for each optimization.
Everything is device-aware: the same harness reports GPU memory on CUDA and
process RSS on CPU.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

import torch

from .batching import throughput
from .generate import generate, timed_generate
from .metrics import LatencyStats, PeakMemory, device_name
from .model import GPT, GPTConfig
from .quantize import logit_mse, model_size_bytes, quantize_int8


@dataclass
class BenchConfig:
    prompt_len: int = 64
    new_tokens: int = 128
    repeats: int = 5
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)
    seed: int = 1234


def _make(device: torch.device, cfg: GPTConfig, seed: int) -> GPT:
    torch.manual_seed(seed)
    return GPT(cfg).to(device).eval()


def _prompt(device: torch.device, cfg: GPTConfig, length: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (1, length), generator=g).to(device)


def bench_kv_cache(model: GPT, prompt: torch.Tensor, new_tokens: int, repeats: int,
                   device: torch.device) -> dict:
    results = {}
    for use_cache in (False, True):
        generate(model, prompt, max_new_tokens=8, use_cache=use_cache)  # warmup
        totals, ttfts, tps = [], [], []
        with PeakMemory(device) as pm:
            for _ in range(repeats):
                r = timed_generate(model, prompt, new_tokens, use_cache=use_cache)
                totals.append(r["total_ms"])
                ttfts.append(r["ttft_ms"])
                tps.append(r["tokens_per_s"])
        stats = LatencyStats.from_samples(totals)
        results["cache_on" if use_cache else "cache_off"] = {
            "latency": stats.__dict__,
            "ttft_ms_p50": round(sorted(ttfts)[len(ttfts) // 2], 3),
            "tokens_per_s_mean": round(sum(tps) / len(tps), 2),
            "peak_mem_mb": round(pm.peak_mb, 2),
        }
    off = results["cache_off"]["tokens_per_s_mean"]
    on = results["cache_on"]["tokens_per_s_mean"]
    results["speedup_x"] = round(on / off, 2) if off > 0 else None
    return results


def bench_batching(model: GPT, prompt: torch.Tensor, new_tokens: int,
                   batch_sizes: tuple[int, ...]) -> dict:
    rows = []
    for b in batch_sizes:
        throughput(model, prompt, batch_size=min(b, 2), max_new_tokens=4)  # warmup
        rows.append(throughput(model, prompt, batch_size=b, max_new_tokens=new_tokens))
    base = rows[0]["tokens_per_s"]
    for r in rows:
        r["speedup_vs_b1"] = round(r["tokens_per_s"] / base, 2) if base > 0 else None
    return {"rows": rows}


def bench_quantization(model: GPT, idx: torch.Tensor, new_tokens: int) -> dict:
    qmodel = quantize_int8(model)
    fp32_size = model_size_bytes(model)
    int8_size = model_size_bytes(qmodel)

    def latency(m) -> float:
        generate(m, idx, max_new_tokens=8, use_cache=True)
        samples = []
        for _ in range(3):
            r = timed_generate(m, idx, new_tokens, use_cache=True)
            samples.append(r["total_ms"])
        return round(sorted(samples)[len(samples) // 2], 3)

    return {
        "fp32_size_mb": round(fp32_size / 1e6, 3),
        "int8_size_mb": round(int8_size / 1e6, 3),
        "size_reduction_x": round(fp32_size / int8_size, 2) if int8_size else None,
        "fp32_latency_ms_p50": latency(model),
        "int8_latency_ms_p50": latency(qmodel),
        "logit_mse": round(logit_mse(model.to("cpu"), qmodel, idx.to("cpu")), 6),
    }


def run_suite(device: str = "cpu", cfg: GPTConfig | None = None,
              bcfg: BenchConfig | None = None) -> dict:
    dev = torch.device(device)
    cfg = cfg or GPTConfig.small()
    bcfg = bcfg or BenchConfig()
    model = _make(dev, cfg, bcfg.seed)
    prompt = _prompt(dev, cfg, bcfg.prompt_len, bcfg.seed)

    return {
        "env": {
            "device": dev.type,
            "device_name": device_name(dev),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
        },
        "model": {
            "params_m": round(model.num_params() / 1e6, 2),
            "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
            "block_size": cfg.block_size, "vocab_size": cfg.vocab_size,
        },
        "config": {"prompt_len": bcfg.prompt_len, "new_tokens": bcfg.new_tokens,
                   "repeats": bcfg.repeats},
        "kv_cache": bench_kv_cache(model, prompt, bcfg.new_tokens, bcfg.repeats, dev),
        "batching": bench_batching(model, prompt, bcfg.new_tokens, bcfg.batch_sizes),
        "quantization": bench_quantization(model, prompt, bcfg.new_tokens),
    }
