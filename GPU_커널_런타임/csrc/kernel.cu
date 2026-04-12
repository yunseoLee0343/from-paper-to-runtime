#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>

#define CUDA_CHECK_ERRORS()                                                   \
  do {                                                                        \
    cudaError_t err__ = cudaGetLastError();                                   \
    TORCH_CHECK(err__ == cudaSuccess, "CUDA kernel failed: ",                 \
                cudaGetErrorString(err__));                                   \
  } while (0)

namespace {

inline int div_up(int a, int b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ float ldg_nc_f32(const float* ptr) {
  float out;
  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(out) : "l"(ptr));
  return out;
}

__global__ __launch_bounds__(128, 2) void conv1_bias_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N,
    int C,
    int H,
    int W,
    int K,
    int R,
    int S,
    int outH,
    int outW,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    bool fuse_relu) {
  const int ow = blockIdx.x * blockDim.x + threadIdx.x;
  const int oh = blockIdx.y * blockDim.y + threadIdx.y;
  const int nk = blockIdx.z;

  if (ow >= outW || oh >= outH) {
    return;
  }

  const int k = nk % K;
  const int n = nk / K;

  float acc = ldg_nc_f32(b + k);

  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        const int ih = oh * stride_h - pad_h + r;
        const int iw = ow * stride_w - pad_w + s;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          const int x_idx = ((n * C + c) * H + ih) * W + iw;
          const int w_idx = ((k * C + c) * R + r) * S + s;
          acc = fmaf(ldg_nc_f32(x + x_idx), ldg_nc_f32(w + w_idx), acc);
        }
      }
    }
  }

  if (fuse_relu && acc < 0.0f) {
    acc = 0.0f;
  }

  const int out_idx = ((n * K + k) * outH + oh) * outW + ow;
  y[out_idx] = acc;
}

__global__ __launch_bounds__(128, 4) void maxpool3x3s2_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N,
    int C,
    int H,
    int W,
    int outH,
    int outW) {
  const int ow = blockIdx.x * blockDim.x + threadIdx.x;
  const int oh = blockIdx.y * blockDim.y + threadIdx.y;
  const int nc = blockIdx.z;

  if (ow >= outW || oh >= outH) {
    return;
  }

  const int c = nc % C;
  const int n = nc / C;
  const int h0 = oh * 2;
  const int w0 = ow * 2;

  float maxv = -FLT_MAX;
  for (int r = 0; r < 3; ++r) {
    for (int s = 0; s < 3; ++s) {
      const int ih = h0 + r;
      const int iw = w0 + s;
      if (ih < H && iw < W) {
        const int in_idx = ((n * C + c) * H + ih) * W + iw;
        maxv = fmaxf(maxv, ldg_nc_f32(x + in_idx));
      }
    }
  }

  const int out_idx = ((n * C + c) * outH + oh) * outW + ow;
  y[out_idx] = maxv;
}

}  // namespace

at::Tensor alex_conv1_fused_cuda(
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& b,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    bool fuse_relu,
    bool fuse_pool) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(w.is_cuda(), "w must be CUDA");
  TORCH_CHECK(b.is_cuda(), "b must be CUDA");

  const int64_t N = x.size(0);
  const int64_t C = x.size(1);
  const int64_t H = x.size(2);
  const int64_t W = x.size(3);
  const int64_t K = w.size(0);
  const int64_t R = w.size(2);
  const int64_t S = w.size(3);

  TORCH_CHECK(C == 3, "This kernel is specialized for conv1 input channels = 3");
  TORCH_CHECK(R == 11 && S == 11, "This kernel is specialized for 11x11 filters");

  const int64_t outH = (H + 2 * pad_h - R) / stride_h + 1;
  const int64_t outW = (W + 2 * pad_w - S) / stride_w + 1;

  auto conv_out = at::empty({N, K, outH, outW}, x.options());

  dim3 block(16, 8, 1);
  dim3 grid(div_up(static_cast<int>(outW), block.x),
            div_up(static_cast<int>(outH), block.y),
            static_cast<unsigned int>(N * K));

  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  conv1_bias_relu_kernel<<<grid, block, 0, stream>>>(
      x.data_ptr<float>(),
      w.data_ptr<float>(),
      b.data_ptr<float>(),
      conv_out.data_ptr<float>(),
      static_cast<int>(N),
      static_cast<int>(C),
      static_cast<int>(H),
      static_cast<int>(W),
      static_cast<int>(K),
      static_cast<int>(R),
      static_cast<int>(S),
      static_cast<int>(outH),
      static_cast<int>(outW),
      static_cast<int>(stride_h),
      static_cast<int>(stride_w),
      static_cast<int>(pad_h),
      static_cast<int>(pad_w),
      fuse_relu);
  CUDA_CHECK_ERRORS();

  if (!fuse_pool) {
    return conv_out;
  }

  const int64_t poolH = (outH - 3) / 2 + 1;
  const int64_t poolW = (outW - 3) / 2 + 1;
  auto pool_out = at::empty({N, K, poolH, poolW}, x.options());

  dim3 pool_block(16, 8, 1);
  dim3 pool_grid(div_up(static_cast<int>(poolW), pool_block.x),
                 div_up(static_cast<int>(poolH), pool_block.y),
                 static_cast<unsigned int>(N * K));

  maxpool3x3s2_kernel<<<pool_grid, pool_block, 0, stream>>>(
      conv_out.data_ptr<float>(),
      pool_out.data_ptr<float>(),
      static_cast<int>(N),
      static_cast<int>(K),
      static_cast<int>(outH),
      static_cast<int>(outW),
      static_cast<int>(poolH),
      static_cast<int>(poolW));
  CUDA_CHECK_ERRORS();

  return pool_out;
}
