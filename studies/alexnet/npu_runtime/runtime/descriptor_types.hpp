#pragma once

#include "../intel_npu_pipeline.hpp"
#include "../ir/npu_ir.hpp"

#include <string>
#include <vector>

namespace intel_npu {

enum class BackendMode {
    SIMULATOR,
    DX12_INTEL_NPU_PREVIEW,
};

struct DmaRuntimeDescriptor {
    u32 command_id = 0;
    u32 tile_id = 0;
    i32 stage_id = 0;
    u64 bank_id = 0;
    u64 bytes = 0;
    bool is_store = false;
    bool uses_reused_input = false;
    u64 local_offset = 0;
};

struct ComputeRuntimeDescriptor {
    u32 command_id = 0;
    u32 tile_id = 0;
    i32 stage_id = 0;
    u64 bank_id = 0;
    i32 pipeline_phase = 0;
    TileRegion output_region;
    PostOpKind post_op = PostOpKind::NONE;
};

struct BarrierRuntimeDescriptor {
    u32 command_id = 0;
    u32 tile_id = 0;
    u32 barrier_id = 0;
    bool is_wait = true;
};

struct ArtifactManifest {
    BackendMode backend_mode = BackendMode::SIMULATOR;
    std::string target_environment = "Intel NPU / DirectX 12 preview / driver 31.0.100.1688";
    std::size_t tile_count = 0;
    std::size_t command_count = 0;
    std::size_t dma_descriptor_count = 0;
    std::size_t compute_descriptor_count = 0;
    std::size_t barrier_descriptor_count = 0;
    std::vector<std::string> notes;
};

struct CompiledArtifact {
    ArtifactManifest manifest;
    std::vector<DmaRuntimeDescriptor> dma_descriptors;
    std::vector<ComputeRuntimeDescriptor> compute_descriptors;
    std::vector<BarrierRuntimeDescriptor> barrier_descriptors;
};

const char* to_string(BackendMode backend_mode);
std::string serialize_artifact_json(const CompiledArtifact& artifact);
CompiledArtifact run_emit_descriptors_pass(const std::vector<ConvTileContract>& tiles, const std::vector<PipelineCommand>& commands, PassLog* log);

}  // namespace intel_npu
