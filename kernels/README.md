# Custom CUDA kernel — fused RMSNorm

A hand-written CUDA kernel that fuses RMSNorm (sum-of-squares reduce → reciprocal
RMS → scale by weight) into a single pass over the activation.

## Why

RMSNorm is **memory-bandwidth-bound** — it reads the activation, does a cheap
reduction, and writes it back. Running it as separate PyTorch ops
(`pow → mean → rsqrt → mul → mul`) makes several passes over memory. A fused
kernel does one read and one write, which is the win for bandwidth-bound ops that
run once or twice per transformer layer.

## Files

- [`rmsnorm_kernel.cu`](rmsnorm_kernel.cu) — the kernel (browsable copy).
- [`../src/llminf/rmsnorm.py`](../src/llminf/rmsnorm.py) — compiles it at runtime
  via `torch.utils.cpp_extension.load_inline` **only when CUDA is available**, and
  dispatches to it; otherwise it uses a numerically identical PyTorch reference.

## Correctness

`tests/test_rmsnorm.py` verifies the PyTorch reference against the math on every
platform. On a CUDA host, the same test path compares the fused kernel against
the reference (`torch.testing.assert_close`). This repo was developed on Apple
Silicon (no CUDA), so the kernel is shipped compile-ready and is exercised on GPU
hardware; the reference path is what runs in CI.

## Design notes

- One CUDA block per row; `blockDim.x = 256` threads cooperatively reduce the
  sum of squares in shared memory (tree reduction).
- Accumulation is done in `float` regardless of input dtype for numerical
  stability, then cast back.
- `AT_DISPATCH_FLOATING_TYPES` supports fp32/fp64; extend with
  `AT_DISPATCH_FLOATING_TYPES_AND_HALF` for fp16/bf16 serving.
- Next steps for a production kernel: vectorized (`float4`) loads, warp-shuffle
  reduction instead of shared memory, and a fused residual-add variant.
