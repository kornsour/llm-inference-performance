"""Benchmark harness.

Produces the headline numbers an inference team tracks — p50/p95 latency,
tokens/sec, peak memory — and the before/after deltas for each optimization.
The same harness runs on CPU and on GPU (`--device cuda`); each measurement
picks the right probe for the device, and anything that is CPU-only (int8
dynamic quantization) says so in its results rather than crashing on GPU.
"""

from __future__ import annotations

import copy
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from .batching import make_ragged_prompts, throughput
from .generate import generate, timed_generate
from .metrics import LatencyStats, PeakMemory, device_name, tensor_bytes
from .model import GPT, GPTConfig
from .quantize import logit_mse, model_size_bytes, quantize_int8
from .rmsnorm import backend as rmsnorm_backend
from .rmsnorm import rmsnorm, rmsnorm_reference

# Number of `timed_generate` samples the quantization section's latency figure
# is a median of. Kept apart from `BenchConfig.repeats` (the 5-sample count
# behind sections 1-2's `LatencyStats.p50_ms`) so the two "p50-ish" numbers in
# one report are never confused for the same statistic — see `bench_quantization`.
QUANT_LATENCY_REPEATS = 3


def _cpu_model() -> str:
    """Best-effort CPU model string, e.g. 'Apple M4 Pro' or 'AMD EPYC 7742'.

    `platform.platform()` and `platform.processor()` collapse to generic values
    on macOS (e.g. 'arm') that cannot tell an M1 apart from an M4 — roughly a 2x
    swing on exactly the numbers this harness reports. `sysctl` exposes the real
    brand string there; everywhere else `platform.processor()` is the best
    portable signal available.
    """
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            brand = out.stdout.strip()
            if brand:
                return brand
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "unknown"


def _git_sha() -> str:
    """Short git SHA of the checkout that produced this report, or 'unknown'.

    Lets a committed artifact be traced back to the exact code that produced
    it, independent of whichever branch/tag happened to be checked out.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


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


@torch.no_grad()
def resident_kv_bytes(model: GPT, prompt: torch.Tensor, new_tokens: int) -> int:
    """Bytes of K/V the cached decode holds resident at the end of a run.

    Measured from the cache tensors themselves rather than modelled. A single
    prefill over the final sequence length yields the same cache shapes the
    decode loop ends up with, so there is no need to re-run the whole loop —
    and `tensor_bytes` counts what those tensors address rather than the buffer
    they are carved from, so the prefill's fused-QKV views are not overcounted.
    """
    length = min(prompt.shape[1] + new_tokens - 1, model.cfg.block_size)
    seq = prompt[:, :1].expand(prompt.shape[0], length).contiguous()
    _, past = model(seq, use_cache=True)
    return tensor_bytes(past)


def bench_kv_cache(model: GPT, prompt: torch.Tensor, new_tokens: int, repeats: int,
                   device: torch.device) -> dict:
    results = {}
    for use_cache in (False, True):
        generate(model, prompt, max_new_tokens=8, use_cache=use_cache)  # warmup
        # Memory gets its own untimed pass: the CPU probe does per-op bookkeeping
        # in Python, so folding it into the timed loop would inflate the very
        # latencies this table reports.
        with PeakMemory(device) as pm:
            generate(model, prompt, max_new_tokens=new_tokens, use_cache=use_cache)
        totals, ttfts, tps = [], [], []
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
    # The cache is memory the no-cache path simply does not hold, so report it
    # on its own: it is the price paid for the speedup, not a saving.
    results["resident_kv_cache_mb"] = round(
        resident_kv_bytes(model, prompt, new_tokens) / 1e6, 2)
    return results


def bench_batching(model: GPT, prompt: torch.Tensor, new_tokens: int,
                   batch_sizes: tuple[int, ...], seed: int = 1234) -> dict:
    """Static batching, one row per **ragged** prompt (see `batching.py`): each
    batch of size `b` is `b` independently-seeded prompts with lengths spread
    over `[prompt.shape[1] // 2, prompt.shape[1]]`, left-padded to a common
    width with a correct attention mask — not `b` copies of `prompt`, which
    would never exercise padding at all.
    """
    device = prompt.device
    vocab_size = model.cfg.vocab_size
    max_len = prompt.shape[1]
    min_len = max(1, max_len // 2)

    rows = []
    for b in batch_sizes:
        prompts = make_ragged_prompts(vocab_size, b, min_len, max_len,
                                      seed=seed + b, device=device)
        throughput(model, prompts, max_new_tokens=4)  # warmup
        rows.append(throughput(model, prompts, max_new_tokens=new_tokens))
    base = rows[0]["tokens_per_s"]
    for r in rows:
        r["speedup_vs_b1"] = round(r["tokens_per_s"] / base, 2) if base > 0 else None
    return {"rows": rows}


def bench_quantization(model: GPT, idx: torch.Tensor, new_tokens: int) -> dict:
    """fp32 vs int8 size, latency and numerical drift.

    `torch.ao` dynamic quantization is a CPU-only path — the int8 model cannot
    consume CUDA inputs — so this comparison always runs on CPU, and reports the
    device it ran on so a `--device cuda` run is not misread as GPU latency.
    Both models are *copies*: `model.to("cpu")` would move the caller's model
    out from under the rest of the suite.

    The latency figure here is a **plain median of `QUANT_LATENCY_REPEATS` (3)**
    samples — a different statistic, from a different sample count, than
    sections 1-2's `LatencyStats.p50_ms` (an interpolated 50th percentile of
    `BenchConfig.repeats`, 5 by default). Both are called out by name
    (`*_latency_ms_median` here vs `p50_ms` there) rather than sharing a "p50"
    label that would imply they were computed the same way.
    """
    fp32 = copy.deepcopy(model).to("cpu").eval()
    qmodel = quantize_int8(fp32)
    idx_cpu = idx.to("cpu")
    fp32_size = model_size_bytes(fp32)
    int8_size = model_size_bytes(qmodel)

    def median_latency(m) -> float:
        generate(m, idx_cpu, max_new_tokens=8, use_cache=True)
        samples = []
        for _ in range(QUANT_LATENCY_REPEATS):
            r = timed_generate(m, idx_cpu, new_tokens, use_cache=True)
            samples.append(r["total_ms"])
        return round(sorted(samples)[len(samples) // 2], 3)

    return {
        "fp32_size_mb": round(fp32_size / 1e6, 3),
        "int8_size_mb": round(int8_size / 1e6, 3),
        "size_reduction_x": round(fp32_size / int8_size, 2) if int8_size else None,
        "latency_device": "cpu",
        "latency_repeats": QUANT_LATENCY_REPEATS,
        "fp32_latency_ms_median": median_latency(fp32),
        "int8_latency_ms_median": median_latency(qmodel),
        "logit_mse": round(logit_mse(fp32, qmodel, idx_cpu), 6),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _rmsnorm_shapes(cfg: GPTConfig, prompt_len: int) -> tuple[tuple[int, int], ...]:
    """Row/column shapes spanning this model's own decode and prefill, plus a
    couple of larger/wider ones to show how the bandwidth argument scales."""
    d = cfg.n_embd
    return (
        (1, d),             # single-token decode at this model's width
        (prompt_len, d),    # prefill at this model's width
        (1, 4096),          # single-token decode, a wider hidden size
        (1024, d),          # large batch decode / short prefill at this width
        (1024, 4096),       # large batch, wide hidden
    )


def bench_rmsnorm(device: torch.device, cfg: GPTConfig, prompt_len: int,
                  repeats: int = 5, inner: int = 20,
                  shapes: tuple[tuple[int, int], ...] | None = None) -> dict:
    """Fused kernel vs. the `pow -> mean -> rsqrt -> mul -> mul` reference.

    Timed standalone from the model — RMSNorm is a bandwidth-bound op, not a
    model-shaped one — on freshly allocated tensors across a range of row/
    column shapes. Each timed sample runs `inner` calls to amortize Python/
    dispatch overhead at the smaller shapes; `repeats` samples per shape,
    reporting the median. Achieved GB/s assumes one read and one write of the
    activation (the weight is O(cols), negligible next to O(rows*cols)) — the
    bandwidth argument `kernels/README.md` makes.

    On CPU (and anywhere the CUDA build did not happen) `rmsnorm()` and
    `rmsnorm_reference()` are the same code, so both columns land on the same
    numbers — that is itself the honest result, not a bug: it says the fused
    kernel is not what ran. `backend` in the returned dict says which path was
    actually measured.
    """
    shapes = shapes or _rmsnorm_shapes(cfg, prompt_len)
    used_backend = rmsnorm_backend(torch.empty(1, device=device))

    def median_ms(fn, x: torch.Tensor, w: torch.Tensor) -> float:
        samples = []
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(inner):
                fn(x, w)
            _sync(device)
            samples.append((time.perf_counter() - t0) * 1000 / inner)
        return sorted(samples)[len(samples) // 2]

    def gbps(ms: float, nbytes: int) -> float | None:
        return round(nbytes / (ms / 1000) / 1e9, 3) if ms > 0 else None

    rows_out = []
    for rows, cols in shapes:
        x = torch.randn(rows, cols, device=device)
        w = torch.randn(cols, device=device)
        rmsnorm(x, w)
        rmsnorm_reference(x, w)  # warmup both paths

        unfused_ms = median_ms(rmsnorm_reference, x, w)
        fused_ms = median_ms(rmsnorm, x, w)
        nbytes = 2 * rows * cols * x.element_size()  # one read + one write

        rows_out.append({
            "rows": rows,
            "cols": cols,
            "unfused_latency_ms_p50": round(unfused_ms, 5),
            "fused_latency_ms_p50": round(fused_ms, 5),
            "unfused_gbps": gbps(unfused_ms, nbytes),
            "fused_gbps": gbps(fused_ms, nbytes),
            "speedup_x": round(unfused_ms / fused_ms, 2) if fused_ms > 0 else None,
        })
    return {"backend": used_backend, "rows": rows_out}


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
            "cpu_model": _cpu_model(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "argv": sys.argv,
        },
        "model": {
            "params_m": round(model.num_params() / 1e6, 2),
            "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
            "block_size": cfg.block_size, "vocab_size": cfg.vocab_size,
        },
        "config": {"prompt_len": bcfg.prompt_len, "new_tokens": bcfg.new_tokens,
                   "repeats": bcfg.repeats},
        "kv_cache": bench_kv_cache(model, prompt, bcfg.new_tokens, bcfg.repeats, dev),
        "batching": bench_batching(model, prompt, bcfg.new_tokens, bcfg.batch_sizes, bcfg.seed),
        "quantization": bench_quantization(model, prompt, bcfg.new_tokens),
        "rmsnorm": bench_rmsnorm(dev, cfg, bcfg.prompt_len, bcfg.repeats),
    }
