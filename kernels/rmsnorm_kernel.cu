// Fused RMSNorm forward (CUDA).
//
// This is the same source compiled at runtime by src/llminf/rmsnorm.py via
// torch.utils.cpp_extension.load_inline when a GPU is present. It is kept here as
// a standalone, browsable artifact. One block per row; blockDim.x threads
// cooperatively reduce the sum of squares in shared memory, then normalize and
// scale in a single pass over the activation.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

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
