"""Fused RMSNorm: a custom CUDA kernel with a PyTorch reference fallback.

RMSNorm is memory-bandwidth-bound and runs once or twice per transformer layer,
so fusing the square-mean-reduce-and-scale into a single kernel avoids extra
passes over the activation. The CUDA kernel is compiled on demand via
`load_inline` *only when a GPU is present*; on CPU (and in CI) the numerically
identical PyTorch reference is used. Tests assert the reference is correct and,
when CUDA is available, that the kernel matches it.
"""

from __future__ import annotations

import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

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
    auto x2 = x.contiguous().view({-1, x.size(-1)});
    auto out = torch::empty_like(x2);
    int rows = x2.size(0), cols = x2.size(1);
    int threads = 256;
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rmsnorm_forward", [&] {
        rmsnorm_kernel<scalar_t><<<rows, threads, threads * sizeof(float)>>>(
            x2.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), cols, (float)eps);
    });
    return out.view_as(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "Fused RMSNorm forward (CUDA)");
}
"""

_kernel = None
_tried = False


def _load_kernel():
    global _kernel, _tried
    if _tried:
        return _kernel
    _tried = True
    if not torch.cuda.is_available():
        return None
    try:  # pragma: no cover - requires CUDA toolchain
        from torch.utils.cpp_extension import load_inline

        _kernel = load_inline(
            name="llminf_rmsnorm",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            functions=["rmsnorm_forward"],
            verbose=False,
        )
    except Exception:
        _kernel = None
    return _kernel


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
