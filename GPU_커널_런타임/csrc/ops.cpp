#include <torch/extension.h>

at::Tensor alex_conv1_fused_cuda(
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& b,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    bool fuse_relu,
    bool fuse_pool);

namespace {

void check_inputs(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(w.scalar_type() == at::kFloat, "w must be float32");
  TORCH_CHECK(b.scalar_type() == at::kFloat, "b must be float32");
  TORCH_CHECK(x.dim() == 4, "x must be NCHW");
  TORCH_CHECK(w.dim() == 4, "w must be OIHW");
  TORCH_CHECK(b.dim() == 1, "b must be 1D");
  TORCH_CHECK(x.size(1) == w.size(1), "input and weight channel count must match");
  TORCH_CHECK(w.size(0) == b.size(0), "weight output channels and bias size must match");
}

at::Tensor alex_conv1_fused(
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& b,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    bool fuse_relu,
    bool fuse_pool) {
  check_inputs(x, w, b);
  return alex_conv1_fused_cuda(
      x.contiguous(),
      w.contiguous(),
      b.contiguous(),
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      fuse_relu,
      fuse_pool);
}

}  // namespace

TORCH_LIBRARY(myimg, m) {
  m.def(
      "alex_conv1_fused(Tensor x, Tensor w, Tensor b, int sh, int sw, int ph, int pw, bool fuse_relu=True, bool fuse_pool=True) -> Tensor");
}

TORCH_LIBRARY_IMPL(myimg, CUDA, m) {
  m.impl("alex_conv1_fused", alex_conv1_fused);
}
