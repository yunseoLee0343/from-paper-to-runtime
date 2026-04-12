# AlexNet NPU Runtime

`studies/alexnet/npu_runtime` is the AlexNet NPU path in this repository. It is no longer just a scheduler demo: it now contains a small compiler-like middle layer plus a descriptorized runtime skeleton and simulator.

## What Is Implemented

The current NPU path follows this flow:

```text
Frontend Graph
-> NPU IR
-> StripDropoutPass
-> FuseActivationPass
-> DecomposeUnsupportedPass
-> CanonicalizePass
-> TileScheduler
-> MemoryPlanningPass
-> PipelineLowering
-> EmitDescriptorsPass
-> RuntimeSimulator
```

Implemented pieces:

- explicit graph/tensor/op IR in `ir/`
- inference-oriented graph passes in `passes/`
- conv tile scheduling and barrier validation in `intel_npu_scheduler.*`
- SRAM bank planning and reuse metadata via `MemoryPlanningPass`
- overlap-aware pipeline lowering in `intel_npu_pipeline.*`
- runtime descriptor emission in `runtime/`
- a cycle-model simulator and deterministic command trace
- CMake-based local build and C++ tests

## What Is Simulated

This code is still simulator-first.

- DMA, compute, and barrier behavior are modeled with explicit commands and cycle estimates.
- The runtime executes a descriptorized command stream inside a software simulator.
- Memory planning is compatible with the simulator's banked SRAM abstraction, not a real device driver ABI.

What this is not:

- not a production Intel NPU runtime
- not a real DirectX 12 command submission implementation
- not a claim of hardware support for Intel NPU driver `31.0.100.1688`

## Future Backend Hookup Intent

The structure is designed so a future backend can plug in at explicit boundaries without rewriting the middle layer.

Current backend modes:

- `SIMULATOR`
- `DX12_INTEL_NPU_PREVIEW`

The `DX12_INTEL_NPU_PREVIEW` mode is declarative only for now. It exists so descriptors and manifest metadata can carry realistic backend intent while the executable path remains simulator-backed.

Future hookup boundaries are intentionally left as TODOs:

- descriptor-to-driver binding boundary
- command queue submission boundary
- fence/event synchronization boundary
- fallback stage execution boundary

## Pass Pipeline

### StripDropoutPass
Treats dropout as inference-dead and rewrites it to identity.

### FuseActivationPass
Fuses conservative single-use `Conv + ReLU` and `Conv + Clamp` patterns into producer post-ops.

### DecomposeUnsupportedPass
Handles unsupported ops with explicit policy:

- `DECOMPOSE`
- `FALLBACK`
- `REJECT`

The current first unsupported target is `LRN`. The infrastructure is real; the decomposition path is intentionally honest and still resolves to a fallback-style placeholder instead of pretending to be a real hardware lowering.

### CanonicalizePass
Removes identity ops, dead ops, dead tensors, and recomputes topological order.

### MemoryPlanningPass
Assigns SRAM bank id, aligned offsets, reuse metadata, and simple lifetime ranges after tile scheduling.

## Descriptor Emission

`EmitDescriptorsPass` converts scheduled tiles and lowered commands into runtime-facing descriptors.

Current descriptor families:

- DMA descriptors
- compute descriptors
- barrier descriptors
- manifest metadata

Current artifact format:

- JSON only

This keeps the output easy to inspect and ready for future executor integration.

## How This Differs from `gpu_runtime`

The GPU path and NPU path now intentionally serve different engineering purposes.

- GPU path: executable vertical slice baseline
- NPU path: compiler-like middle layer + descriptorized runtime skeleton/simulator

That means `gpu_runtime` is closer to a directly executable backend experiment, while `npu_runtime` is structured to show graph-to-descriptor compilation stages and future backend integration boundaries.

## Build and Test

Configure and build:

```powershell
cmake -S studies/alexnet/npu_runtime -B studies/alexnet/npu_runtime/build
cmake --build studies/alexnet/npu_runtime/build
```

Run tests:

```powershell
ctest --test-dir studies/alexnet/npu_runtime/build --output-on-failure
```

Run the demo:

```powershell
studies/alexnet/npu_runtime/build/alexnet_npu_runtime_demo.exe
```

## Main Files

- [ir/npu_ir.hpp](/d:/ai_study/studies/alexnet/npu_runtime/ir/npu_ir.hpp)
- [intel_npu_scheduler.cpp](/d:/ai_study/studies/alexnet/npu_runtime/intel_npu_scheduler.cpp)
- [intel_npu_pipeline.cpp](/d:/ai_study/studies/alexnet/npu_runtime/intel_npu_pipeline.cpp)
- [runtime/descriptor_types.hpp](/d:/ai_study/studies/alexnet/npu_runtime/runtime/descriptor_types.hpp)
- [raw/alexnet_npu.cpp](/d:/ai_study/studies/alexnet/npu_runtime/raw/alexnet_npu.cpp)

## Docs

- [docs/NPU_HW_Contract.md](/d:/ai_study/studies/alexnet/npu_runtime/docs/NPU_HW_Contract.md)
- [docs/Tiling_and_Sync.md](/d:/ai_study/studies/alexnet/npu_runtime/docs/Tiling_and_Sync.md)
- [docs/Optimization_Performance.md](/d:/ai_study/studies/alexnet/npu_runtime/docs/Optimization_Performance.md)
- [docs/Compiler_Runtime_Architecture.md](/d:/ai_study/studies/alexnet/npu_runtime/docs/Compiler_Runtime_Architecture.md)
