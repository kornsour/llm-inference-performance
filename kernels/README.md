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

## Building

`load_inline(functions=["rmsnorm_forward"])` **generates** the `PYBIND11_MODULE`
block itself, into a C++ translation unit separate from the CUDA one. Two
consequences, both of which will otherwise stop the extension from building at
all — and only on a machine that actually has a GPU, which is not where this was
written:

- this file must **not** declare a `PYBIND11_MODULE` of its own, or the link ends
  up with two `PyInit_llminf_rmsnorm` symbols;
- `cpp_sources` must **declare** every exported function, or the generated
  bindings reference an undeclared `rmsnorm_forward`.

`tests/test_rmsnorm.py` asserts both invariants, and that this file stays
byte-identical to the string `rmsnorm.py` actually compiles, so the browsable
copy cannot drift from the built one.

Those are cheap proxies, though — the real proof is a build. `nvcc` needs no GPU
to compile and link, so `make check-kernel`
([`scripts/check_kernel_builds.py`](../scripts/check_kernel_builds.py)) reproduces
exactly what `load_inline` emits and drives it through to a linked `.so` on any
machine with the CUDA toolkit. It stops short of importing the module, which
would need `libcuda.so.1` from the driver. CI runs it on an ordinary GPU-less
runner, with `--self-test`: the script re-introduces each defect and requires the
build to reject it, so the guard cannot pass vacuously.

If the build fails at runtime, `rmsnorm.py` falls back to the PyTorch reference
but emits a `RuntimeWarning` and keeps the compiler output in
`rmsnorm.load_error()`. A silent fallback would hide a broken kernel behind
numbers that still look plausible.

## Correctness

`tests/test_rmsnorm.py` verifies the PyTorch reference against the math on every
platform. On a CUDA host, the same test path asserts the kernel built (no
`load_error()`), that dispatch actually selects it, and that it matches the
reference (`torch.testing.assert_close`). This repo was developed on Apple
Silicon (no CUDA), so the kernel is shipped compile-ready and is exercised on GPU
hardware; the reference path is what runs in CI.

## Design notes

- One CUDA block per row; `blockDim.x = 256` threads cooperatively reduce the
  sum of squares in shared memory (tree reduction).
- Accumulation is done in `float` regardless of input dtype for numerical
  stability, then cast back.
- `AT_DISPATCH_FLOATING_TYPES` supports fp32/fp64; extend with
  `AT_DISPATCH_FLOATING_TYPES_AND_HALF` for fp16/bf16 serving.
- Includes are kept to `c10/cuda/*` rather than the `ATen/cuda/CUDAContext.h`
  umbrella. That umbrella drags in cusparse, cublas, cublasLt and cusolver
  purely to hand back the current stream, which would make this extension
  unbuildable against a CUDA install carrying nvcc and cudart but not the math
  libraries — minimal toolkit images, slim containers, CI runners.
- Next steps for a production kernel: vectorized (`float4`) loads, warp-shuffle
  reduction instead of shared memory, and a fused residual-add variant.
