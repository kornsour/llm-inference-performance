# Methodology

How the numbers in [`results.md`](results.md) are produced, and what they do and
don't claim.

## What is measured

The headline metrics mirror what an inference team tracks:

- **Latency** — p50 and p95 of end-to-end generation time, plus **TTFT**
  (time-to-first-token = prefill time). Both percentiles come from
  `BenchConfig.repeats` timed samples (5 by default, called out beside each
  column header as `n=`); with that few samples, p95 interpolates between the
  4th and 5th largest, so treat it as "close to the max observed" rather than a
  precise tail estimate. Raise `repeats` for a tighter p95.
- **Throughput** — tokens/sec, aggregate across the batch.
- **Memory** — the high-water mark of tensor bytes allocated *inside* a measured
  block, above whatever was already resident (model weights, prompt) on entry.
  On CUDA that is the allocator's own peak minus the entry baseline; on CPU,
  where there is no allocator peak counter, a `TorchDispatchMode` probe tracks
  tensor storages as ops create and free them.

  It is deliberately **not** a process-RSS delta. RSS is a high-water mark that
  never comes back down, so the first block measured in a process absorbs all
  the heap growth and every later block reads ~0 — which made whichever variant
  ran second look nearly free regardless of what it cost. The probe walks Python
  per aten op, so memory is measured in its own **untimed** pass; the reported
  latencies come from separate, uninstrumented runs.

## Why random weights

The model is a nanoGPT-style decoder with **randomly initialized weights** at a
realistic shape (configurable; default ~11M params, 6 layers). This is a
*performance* lab, not a quality lab — latency, throughput, and memory of the
forward/decode loop depend on tensor shapes and the execution path, not on
whether the weights were trained. Output *quality* is therefore not evaluated.
What **is** asserted in tests is that every optimization is
**numerically faithful** to the baseline:

- KV-cache on vs off produce **identical** greedy tokens.
- Tensor-parallel MLP matches the single-process MLP.
- int8 quantization's logit drift is measured (MSE) and bounded.

## The A/Bs

| Optimization | Baseline | Variant | What the delta isolates |
| --- | --- | --- | --- |
| KV-cache | recompute full prefix each step (O(T²)) | cache K/V, decode O(T) | redundant attention compute (bought with resident K/V memory) |
| Static batching | batch size 1 | batch size 2…16, ragged prompts | matmul utilization |
| int8 quant | fp32 weights | int8 dynamic | model size & memory bandwidth |
| Fused RMSNorm | `pow → mean → rsqrt → mul → mul` (separate ops) | single fused kernel (CUDA) / same reference elsewhere | achieved GB/s of a bandwidth-bound op |

Each timed section runs a **warmup** generation first (to absorb lazy
initialization / allocator warmup), then `repeats` measured runs.

## Determinism

Seeds are fixed for the model, the prompt, and any randomized input, so a re-run
reproduces the same shapes and the same greedy outputs. Absolute timings vary
with hardware and load; the **ratios** (speedup, size reduction) are the durable
signal.

## Provenance

Every report's `env` block is self-describing: the CPU model (`sysctl
machdep.cpu.brand_string` on macOS, `platform.processor()` elsewhere —
`platform.platform()` alone collapses an M1 and an M4 to the same string),
torch/Python versions, a UTC generation timestamp, the git SHA of the checkout
that produced it, and the literal `argv` the run was invoked with. That is
enough to tell, from the JSON alone, which machine, code, and command line a
given set of numbers came from — no need to cross-reference the README or trust
that it was captioned correctly.

## Honest caveats

- **The KV-cache spends memory to buy speed.** `resident_kv_cache_mb` reports the
  K/V actually held at the end of a run (3.52 MB at 192 tokens here), growing
  linearly with sequence length and batch size — in production it is the term
  that bounds concurrency. Its measured *peak* still lands below the no-cache
  path's, because recomputing the prefix every step materializes much larger
  attention intermediates; and most of the cached path's peak is the `torch.cat`
  that grows the cache one token at a time, holding the old and new buffers
  simultaneously (preallocating the cache to max sequence length is the standard
  fix, and is not done here). None of that makes the cache a memory *saving*.
- **CPU vs GPU.** This was developed on Apple Silicon (no NVIDIA GPU). The same
  harness runs on CUDA (`--device cuda`); the custom CUDA kernel and NCCL backend
  activate there. The committed numbers are CPU numbers and are labeled as such.
- **Two different "p50"s in one report.** Sections 1-2's p50/p95 come from
  `LatencyStats`, an interpolated percentile of `BenchConfig.repeats` (5)
  timed-generate samples. Section 3's quantization latency is a **plain
  median of `QUANT_LATENCY_REPEATS` (3)** samples — named
  `fp32_latency_ms_median` / `int8_latency_ms_median`, not `*_p50`, and its
  table column says "median of 3" rather than reusing the "p50" label, so the
  two are never mistaken for the same statistic computed the same way.
- **int8 latency on CPU.** Dynamic int8 quantization's headline win here is
  **model size / memory**; on CPU (qnnpack) at this model scale it does **not**
  reduce latency and can increase it — this is reported rather than hidden. The
  latency win shows up on hardware/kernels tuned for int8 GEMM and at larger
  scales. `torch.ao` dynamic quantization is also a **CPU-only** path: the int8
  model cannot consume CUDA tensors, so under `--device cuda` this section is
  measured on CPU and labels itself (`latency_device`) rather than failing.
- **The fused RMSNorm kernel is built, not stubbed, and it runs in the model.**
  `load_inline` compiles it on first use when CUDA is present. If that build
  fails the harness falls back to the PyTorch reference but **warns**, and
  `rmsnorm.load_error()` carries the compiler output — a silent fallback would
  hide a broken kernel behind numbers that still look plausible. `GPT` uses
  `RMSNorm` (not `nn.LayerNorm`) for every norm, so section 4 of
  [`results.md`](results.md) reports a real fused-vs-unfused GB/s comparison
  rather than a kernel that sits unused next to the harness; on CPU both
  columns land on the same numbers because they're the same code path, which
  is reported (`backend`) rather than hidden.
- **This is static batching, not continuous batching.** Each row of section 2 is
  `batch_size` independent prompts of *varying* length — left-padded to a
  common width, decoded together with a correct attention mask (see
  `batching.py`, `GPT._build_attn_bias`) — rather than one prompt duplicated
  `batch_size` times, which would never exercise padding. But the whole batch
  is still assembled up front and walked to a fixed length together: no
  sequence is admitted or evicted mid-flight as others finish early, which is
  what production serving's continuous batching does and what actually
  determines goodput under real traffic. `test_ragged_batch_matches_each_prompt_generated_alone`
  is the correctness check that padding and masking don't leak across rows;
  the throughput scaling itself is a static-batching number and is labeled
  as such rather than as "request batching."
- **Batching on CPU** scales sub-linearly (limited cores / memory bandwidth); the
  trend is real but a GPU shows much steeper throughput scaling.
- **Tensor parallelism** uses gloo on CPU so the collective path is testable; on
  multi-GPU the identical code uses NCCL. `make tp-demo` is a component-level
  micro-benchmark — a sharded MLP plus its all-reduce, checked for correctness
  and timed — not an end-to-end TP serving path: attention isn't sharded and
  there's no TP generate/decode loop, so its numbers describe the MLP shard and
  the collective, not serving latency or throughput under tensor parallelism.
  See [`benchmarks/results/tp_latest.json`](../benchmarks/results/tp_latest.json)
  for the committed numbers.
