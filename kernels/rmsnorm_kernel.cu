// Fused RMSNorm forward (CUDA).
//
// This is the same source compiled at runtime by src/llminf/rmsnorm.py via
// torch.utils.cpp_extension.load_inline when a GPU is present. It is kept here as
// a standalone, browsable artifact, and tests/test_rmsnorm.py asserts the two
// stay byte-identical below this header.
//
// Note the absence of a PYBIND11_MODULE block: load_inline generates the module
// definition itself (from its `functions=` argument) into a separate C++
// translation unit, which is also where `rmsnorm_forward` is declared. Declaring
// a second module here would leave two PyInit_ symbols to link.
//
// One block per row; blockDim.x threads cooperatively reduce the sum of squares
// in shared memory, then normalize and scale in a single pass over the activation.
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
