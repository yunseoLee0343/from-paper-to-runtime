#include "descriptor_types.hpp"

#include <sstream>

namespace intel_npu {

const char* to_string(BackendMode backend_mode) {
    switch (backend_mode) {
        case BackendMode::SIMULATOR: return "SIMULATOR";
        case BackendMode::DX12_INTEL_NPU_PREVIEW: return "DX12_INTEL_NPU_PREVIEW";
    }
    throw std::runtime_error("Unknown backend mode");
}

namespace {

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            default: out += ch; break;
        }
    }
    return out;
}

}  // namespace

std::string serialize_artifact_json(const CompiledArtifact& artifact) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"manifest\": {\n";
    oss << "    \"backend_mode\": \"" << to_string(artifact.manifest.backend_mode) << "\",\n";
    oss << "    \"target_environment\": \"" << json_escape(artifact.manifest.target_environment) << "\",\n";
    oss << "    \"tile_count\": " << artifact.manifest.tile_count << ",\n";
    oss << "    \"command_count\": " << artifact.manifest.command_count << ",\n";
    oss << "    \"dma_descriptor_count\": " << artifact.manifest.dma_descriptor_count << ",\n";
    oss << "    \"compute_descriptor_count\": " << artifact.manifest.compute_descriptor_count << ",\n";
    oss << "    \"barrier_descriptor_count\": " << artifact.manifest.barrier_descriptor_count << ",\n";
    oss << "    \"notes\": [";
    for (std::size_t i = 0; i < artifact.manifest.notes.size(); ++i) {
        oss << "\"" << json_escape(artifact.manifest.notes[i]) << "\"";
        if (i + 1 != artifact.manifest.notes.size()) {
            oss << ", ";
        }
    }
    oss << "]\n";
    oss << "  },\n";

    oss << "  \"dma_descriptors\": [\n";
    for (std::size_t i = 0; i < artifact.dma_descriptors.size(); ++i) {
        const auto& desc = artifact.dma_descriptors[i];
        oss << "    {\"command_id\": " << desc.command_id
            << ", \"tile_id\": " << desc.tile_id
            << ", \"stage_id\": " << desc.stage_id
            << ", \"bank_id\": " << desc.bank_id
            << ", \"bytes\": " << desc.bytes
            << ", \"is_store\": " << (desc.is_store ? "true" : "false")
            << ", \"uses_reused_input\": " << (desc.uses_reused_input ? "true" : "false")
            << ", \"local_offset\": " << desc.local_offset << "}";
        oss << (i + 1 == artifact.dma_descriptors.size() ? "\n" : ",\n");
    }
    oss << "  ],\n";

    oss << "  \"compute_descriptors\": [\n";
    for (std::size_t i = 0; i < artifact.compute_descriptors.size(); ++i) {
        const auto& desc = artifact.compute_descriptors[i];
        oss << "    {\"command_id\": " << desc.command_id
            << ", \"tile_id\": " << desc.tile_id
            << ", \"stage_id\": " << desc.stage_id
            << ", \"bank_id\": " << desc.bank_id
            << ", \"pipeline_phase\": " << desc.pipeline_phase
            << ", \"post_op\": \"" << to_string(desc.post_op) << "\""
            << ", \"output_region\": {"
            << "\"n0\": " << desc.output_region.n0 << ", "
            << "\"n1\": " << desc.output_region.n1 << ", "
            << "\"c0\": " << desc.output_region.c0 << ", "
            << "\"c1\": " << desc.output_region.c1 << ", "
            << "\"h0\": " << desc.output_region.h0 << ", "
            << "\"h1\": " << desc.output_region.h1 << ", "
            << "\"w0\": " << desc.output_region.w0 << ", "
            << "\"w1\": " << desc.output_region.w1 << "}}";
        oss << (i + 1 == artifact.compute_descriptors.size() ? "\n" : ",\n");
    }
    oss << "  ],\n";

    oss << "  \"barrier_descriptors\": [\n";
    for (std::size_t i = 0; i < artifact.barrier_descriptors.size(); ++i) {
        const auto& desc = artifact.barrier_descriptors[i];
        oss << "    {\"command_id\": " << desc.command_id
            << ", \"tile_id\": " << desc.tile_id
            << ", \"barrier_id\": " << desc.barrier_id
            << ", \"is_wait\": " << (desc.is_wait ? "true" : "false") << "}";
        oss << (i + 1 == artifact.barrier_descriptors.size() ? "\n" : ",\n");
    }
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

}  // namespace intel_npu
