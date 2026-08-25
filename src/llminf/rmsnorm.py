"""Fused RMSNorm: a custom CUDA kernel with a PyTorch reference fallback.

RMSNorm is memory-bandwidth-bound and runs once or twice per transformer layer,
so fusing the square-mean-reduce-and-scale into a single kernel avoids extra
passes over the activation. The CUDA kernel is compiled on demand via
`load_inline` *only when a GPU is present*; on CPU (and in CI) the numerically
identical PyTorch reference is used. Tests assert the reference is correct and,
when CUDA is available, that the kernel matches it.

Extension wiring, since it is easy to get wrong: `load_inline(functions=[...])`
*generates* the ``PYBIND11_MODULE`` block into its own C++ translation unit, so
the CUDA source must not declare one itself (two ``PyInit_`` symbols will not
link) and ``cpp_sources`` must carry a declaration of every exported function,
or the generated bindings will not compile.
"""

from __future__ import annotations

import warnings

import torch

# Declaration only — this is what the auto-generated pybind11 bindings compile
# against. The definition lives in the CUDA translation unit below.
_CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps);
"""

# Keep in sync with kernels/rmsnorm_kernel.cu (tests/test_rmsnorm.py asserts it).
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

// One block per row; blockDim.x threads cooperatively reduce sum of squares.
template <typename scalar_t>
__global__ void rmsnorm_kernel(const scalar_t* __restrict__ x,
                               const scalar_t* __restrict__ weight,
                               scalar_t* __restrict__ out,
                               int n_cols, float eps) {
    int row = blockIdx.x;
    const scalar_t* xr = x + (long)row * n_cols;
    scalar_t* outr = out + (long)row * n_cols;

    extern __shared__ float sdata[];
    float local = 0.f;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        float v = (float)xr[i];
        local += v * v;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / n_cols + eps);
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        outr[i] = (scalar_t)(((float)xr[i] * inv_rms) * (float)weight[i]);
    }
}

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(),
                "weight dtype must match x dtype");
    TORCH_CHECK(weight.numel() == x.size(-1),
                "weight must have x.size(-1) elements");

    const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
    // Hoist the contiguous copies into named tensors: a temporary would be
    // destroyed before the asynchronous kernel had read from it.
    auto x2 = x.contiguous().view({-1, x.size(-1)});
    auto w = weight.contiguous();
    auto out = torch::empty_like(x2);

    const int64_t rows = x2.size(0);
    const int64_t cols = x2.size(1);
    if (rows == 0) return out.view_as(x);
    TORCH_CHECK(rows <= (int64_t)std::numeric_limits<int>::max(),
                "too many rows for a one-block-per-row launch");

    const int threads = 256;
    AT_DISPATCH_FLOATING_TYPES(x2.scalar_type(), "rmsnorm_forward", [&] {
        rmsnorm_kernel<scalar_t><<<(unsigned)rows, threads, threads * sizeof(float),
                                   at::cuda::getCurrentCUDAStream()>>>(
            x2.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), (int)cols, (float)eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return out.view_as(x);
}
"""

_kernel = None
_load_error: str | None = None
_tried = False


def _load_kernel():
    global _kernel, _load_error, _tried
    if _tried:
        return _kernel
    _tried = True
    if not torch.cuda.is_available():
        return None
    try:  # pragma: no cover - requires CUDA toolchain
        from torch.utils.cpp_extension import load_inline

        _kernel = load_inline(
            name="llminf_rmsnorm",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["rmsnorm_forward"],
            with_cuda=True,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - requires CUDA toolchain
        # Falling back silently would hide a broken build behind numbers that
        # still look plausible, so say so once and keep going on the reference.
        _kernel = None
        _load_error = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "CUDA is available but the fused RMSNorm kernel failed to build; "
            f"falling back to the PyTorch reference. {_load_error}",
            RuntimeWarning,
            stacklevel=2,
        )
    return _kernel


def load_error() -> str | None:
    """The kernel's build error, if a CUDA build was attempted and failed."""
    return _load_error


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused CUDA path when available; PyTorch reference otherwise."""
    if x.is_cuda:
        k = _load_kernel()
        if k is not None:  # pragma: no cover - requires CUDA
            return k.rmsnorm_forward(x, weight, eps)
    return rmsnorm_reference(x, weight, eps)


def backend(x: torch.Tensor | None = None) -> str:
    if x is not None and x.is_cuda and _load_kernel() is not None:  # pragma: no cover
        return "cuda-fused"
    return "pytorch-reference"
