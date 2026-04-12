# Compiler Runtime Architecture

## Overview

The AlexNet NPU path is organized as a small compiler/runtime stack rather than a direct backend implementation.

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

This design is intentionally honest.

- It is executable today through a simulator.
- It produces explicit descriptors today.
- It is structured for future backend hookup.
- It is not a production Intel NPU runtime.

## Implemented Layers

### Frontend Graph -> NPU IR

`lower_graph_desc_to_npu_ir()` turns frontend graph metadata into an inspectable graph IR with explicit tensors, ops, post-ops, and debugging dumps.

### Graph Passes

The graph pass pipeline performs inference-oriented cleanup and normalization before tiling.

- `StripDropoutPass`: rewrites dropout to identity
- `FuseActivationPass`: folds safe `Conv + ReLU` / `Conv + Clamp` into producer post-ops
- `DecomposeUnsupportedPass`: applies `DECOMPOSE` / `FALLBACK` / `REJECT` policy to unsupported ops such as `LRN`
- `CanonicalizePass`: removes identity and dead graph structure, then recomputes topo order

### Tile Scheduler

The scheduler no longer acts as the first compiler stage. It consumes canonical IR and builds a stage sequence first, then produces `ConvTileContract` objects only for tilable conv stages.

Fallback stages are kept explicit at the compiler level even if they are not executed by the simulator yet.

### Memory Planning

`MemoryPlanningPass` owns SRAM bank selection, aligned offsets, simple lifetime metadata, and producer-consumer SRAM reuse decisions.

This keeps memory assignment separate from tile formation and makes simulator/runtime contracts clearer.

### Pipeline Lowering

`intel_npu_pipeline.*` lowers scheduled tiles into explicit commands:

- DMA load
- DMA store
- compute
- barrier wait
- barrier signal

The lowering keeps the overlap model from the earlier simulator, but now consumes post-planning contracts rather than self-inventing placement.

### Descriptor Emission

`EmitDescriptorsPass` converts scheduled tiles and lowered commands into explicit runtime descriptor families.

- DMA descriptors
- compute descriptors
- barrier descriptors
- manifest metadata

Current output is JSON for readability and debugging.

### Runtime Simulator

The runtime simulator executes the command stream with a cycle model and reports overlap metrics. This provides an executable path for validation even though no real Intel NPU or DX12 submission path is implemented yet.

## Backend Orientation

The current code names two backend modes:

- `SIMULATOR`
- `DX12_INTEL_NPU_PREVIEW`

Only `SIMULATOR` is executable today.

`DX12_INTEL_NPU_PREVIEW` exists to keep descriptor and executor boundaries realistic for future work targeting:

- Intel NPU
- DirectX 12 path
- driver `31.0.100.1688`

## Future Hookup Boundaries

These boundaries are intentionally left explicit in code and docs.

- descriptor-to-driver binding
- queue submission
- fence/event synchronization
- fallback execution backend
- real unsupported-op primitive decomposition
- backend-specific layout baking

## Why This Is Different from the GPU Path

`gpu_runtime` is a vertical-slice backend experiment.

`npu_runtime` is now a compiler-style middle layer plus descriptorized runtime skeleton. The point is to make graph transformation, tiling legality, SRAM planning, descriptor emission, and future backend binding all visible and inspectable.
