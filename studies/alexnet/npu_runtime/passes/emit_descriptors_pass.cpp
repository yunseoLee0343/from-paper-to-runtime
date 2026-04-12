#include "../runtime/descriptor_types.hpp"

namespace intel_npu {

CompiledArtifact run_emit_descriptors_pass(
    const std::vector<ConvTileContract>& tiles,
    const std::vector<PipelineCommand>& commands,
    PassLog* log) {
    CompiledArtifact artifact;
    artifact.manifest.backend_mode = BackendMode::SIMULATOR;
    artifact.manifest.tile_count = tiles.size();
    artifact.manifest.command_count = commands.size();
    artifact.manifest.notes.push_back("Simulator-first executable path");
    artifact.manifest.notes.push_back("TODO: queue submission boundary for DX12 Intel NPU preview backend");
    artifact.manifest.notes.push_back("TODO: descriptor-to-driver binding boundary");
    artifact.manifest.notes.push_back("TODO: fence/event synchronization boundary");

    for (const auto& command : commands) {
        if (command.kind == PipelineCommandKind::DMA_LOAD || command.kind == PipelineCommandKind::DMA_STORE) {
            DmaRuntimeDescriptor desc;
            desc.command_id = command.command_id;
            desc.tile_id = command.tile_id;
            desc.stage_id = command.contract != nullptr ? command.contract->task.stage_id : -1;
            desc.bank_id = command.bank_id;
            desc.bytes = command.bytes;
            desc.is_store = command.kind == PipelineCommandKind::DMA_STORE;
            desc.uses_reused_input = command.uses_reused_input;
            if (command.contract != nullptr) {
                desc.local_offset = desc.is_store ? command.contract->mem_plan.output_tile.offset : command.contract->mem_plan.input_tile.offset;
            }
            artifact.dma_descriptors.push_back(desc);
            continue;
        }
        if (command.kind == PipelineCommandKind::COMPUTE_CONV) {
            ComputeRuntimeDescriptor desc;
            desc.command_id = command.command_id;
            desc.tile_id = command.tile_id;
            desc.stage_id = command.contract != nullptr ? command.contract->task.stage_id : -1;
            desc.bank_id = command.bank_id;
            if (command.contract != nullptr) {
                desc.pipeline_phase = command.contract->task.pipeline_phase;
                desc.output_region = command.contract->output_region;
                desc.post_op = command.contract->post_op;
            }
            artifact.compute_descriptors.push_back(desc);
            continue;
        }
        if (command.kind == PipelineCommandKind::BARRIER_WAIT || command.kind == PipelineCommandKind::BARRIER_SIGNAL) {
            BarrierRuntimeDescriptor desc;
            desc.command_id = command.command_id;
            desc.tile_id = command.tile_id;
            desc.barrier_id = command.barrier_id;
            desc.is_wait = command.kind == PipelineCommandKind::BARRIER_WAIT;
            artifact.barrier_descriptors.push_back(desc);
        }
    }

    artifact.manifest.dma_descriptor_count = artifact.dma_descriptors.size();
    artifact.manifest.compute_descriptor_count = artifact.compute_descriptors.size();
    artifact.manifest.barrier_descriptor_count = artifact.barrier_descriptors.size();

    if (log != nullptr) {
        log->append(
            "EmitDescriptorsPass",
            "emitted dma=" + std::to_string(artifact.manifest.dma_descriptor_count) +
                " compute=" + std::to_string(artifact.manifest.compute_descriptor_count) +
                " barrier=" + std::to_string(artifact.manifest.barrier_descriptor_count));
    }

    return artifact;
}

}  // namespace intel_npu
