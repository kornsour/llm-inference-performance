# Methodology

How the numbers in [`results.md`](results.md) are produced, and what they do and
don't claim.

## What is measured

The headline metrics mirror what an inference team tracks:

- **Latency** — p50 and p95 of end-to-end generation time, plus **TTFT**
  (time-to-first-token = prefill time).
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
| Batching | batch size 1 | batch size 2…16 | matmul utilization |
| int8 quant | fp32 weights | int8 dynamic | model size & memory bandwidth |

Each timed section runs a **warmup** generation first (to absorb lazy
initialization / allocator warmup), then `repeats` measured runs.

## Determinism

Seeds are fixed for the model, the prompt, and any randomized input, so a re-run
reproduces the same shapes and the same greedy outputs. Absolute timings vary
with hardware and load; the **ratios** (speedup, size reduction) are the durable
signal.

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
- **int8 latency on CPU.** Dynamic int8 quantization's headline win here is
  **model size / memory**; on CPU (qnnpack) at this model scale it does **not**
  reduce latency and can increase it — this is reported rather than hidden. The
  latency win shows up on hardware/kernels tuned for int8 GEMM and at larger
  scales. `torch.ao` dynamic quantization is also a **CPU-only** path: the int8
  model cannot consume CUDA tensors, so under `--device cuda` this section is
  measured on CPU and labels itself (`latency_device`) rather than failing.
- **The fused RMSNorm kernel is built, not stubbed.** `load_inline` compiles it
  on first use when CUDA is present. If that build fails the harness falls back
  to the PyTorch reference but **warns**, and `rmsnorm.load_error()` carries the
  compiler output — a silent fallback would hide a broken kernel behind numbers
  that still look plausible.
- **Batching on CPU** scales sub-linearly (limited cores / memory bandwidth); the
  trend is real but a GPU shows much steeper throughput scaling.
- **Tensor parallelism** uses gloo on CPU so the collective path is testable; on
  multi-GPU the identical code uses NCCL.
