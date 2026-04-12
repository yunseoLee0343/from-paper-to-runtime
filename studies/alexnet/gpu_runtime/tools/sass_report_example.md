# SASS Report Example

## Kernel

- Target: `myimg::alex_conv1_fused`
- CUDA kernel symbol: `conv1_bias_relu_kernel`
- Build flags: `-O3 -lineinfo --use_fast_math -Xptxas=-v`

## Inspection Checklist

- `FFMA` mix increased after switching the inner loop to fused multiply-add friendly code paths.
- Read-only operand loads should appear around the custom `ld.global.nc.f32` inline PTX sites for `x`, `w`, and `b`.
- `ptxas -v` output should be checked for register count and spill warnings.
- `__launch_bounds__(128, 2)` keeps the main conv kernel from inflating register usage and hurting occupancy.

## Example Findings

- Baseline: scalar global loads dominated the inner loop and SASS showed extra address recomputation.
- Optimized: non-coherent read-only loads reduced pressure on the generic global path and made the inner loop more FFMA-heavy.
- Residual risk: this is still a direct convolution kernel, so memory traffic remains the primary bottleneck on large GPUs such as A6000.

## Next Optimization Targets

- Tile output channels and spatial tiles into shared memory.
- Vectorize input and weight loads where alignment allows.
- Add an fp16 input/weight path with fp32 accumulation for Tensor Core friendly deployment.
