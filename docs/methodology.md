# Methodology

How the numbers in [`results.md`](results.md) are produced, and what they do and
don't claim.

## What is measured

The headline metrics mirror what an inference team tracks:

- **Latency** — p50 and p95 of end-to-end generation time, plus **TTFT**
  (time-to-first-token = prefill time).
- **Throughput** — tokens/sec, aggregate across the batch.
- **Memory** — peak GPU allocator memory on CUDA; peak process-RSS delta on CPU.

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
| KV-cache | recompute full prefix each step (O(T²)) | cache K/V, decode O(T) | redundant attention compute |
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

- **CPU vs GPU.** This was developed on Apple Silicon (no NVIDIA GPU). The same
  harness runs on CUDA (`--device cuda`); the custom CUDA kernel and NCCL backend
  activate there. The committed numbers are CPU numbers and are labeled as such.
- **int8 latency on CPU.** Dynamic int8 quantization's headline win here is
  **model size / memory**; on CPU (qnnpack) at this model scale it does **not**
  reduce latency and can increase it — this is reported rather than hidden. The
  latency win shows up on hardware/kernels tuned for int8 GEMM and at larger
  scales.
- **Batching on CPU** scales sub-linearly (limited cores / memory bandwidth); the
  trend is real but a GPU shows much steeper throughput scaling.
- **Tensor parallelism** uses gloo on CPU so the collective path is testable; on
  multi-GPU the identical code uses NCCL.
