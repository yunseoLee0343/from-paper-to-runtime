#include "intel_npu_pipeline.hpp"
#include "ir/npu_ir.hpp"
#include "runtime/descriptor_types.hpp"

#include <iostream>

using namespace intel_npu;

int main() {
    try {
        HardwareCaps hw;
        hw.local_sram_bytes = 4 * 1024 * 1024;
        hw.sram_bank_count = 2;
        hw.dma_alignment = 64;
        hw.max_tile_h = 16;
        hw.max_tile_w = 16;
        hw.max_tile_c = 64;

        PassLog log;
        const auto tensors = make_demo_tensors_p2();
        const auto graph = make_compiler_demo_graph();
        auto ir = lower_graph_desc_to_npu_ir(graph, tensors);

        run_strip_dropout_pass(ir, &log);
        run_fuse_activation_pass(ir, &log);
        run_decompose_unsupported_pass(ir, UnsupportedPolicy::FALLBACK, &log);
        run_canonicalize_pass(ir, &log);

        auto tiles = GraphTileScheduler::schedule_from_ir(ir, tensors, hw, &log);
        run_memory_planning_pass(tiles, hw, &log);
        auto commands = lower_graph_stream(tiles, tensors);
        auto artifact = run_emit_descriptors_pass(tiles, commands, &log);

        PipelineRuntime runtime(hw);
        runtime.install_barriers(tiles);
        const auto stats = runtime.run(commands);

        std::cout << "=== Pass Log ===\n" << log.str();
        std::cout << "\n=== Canonical IR ===\n" << dump_graph_ir(ir);
        std::cout << "\n=== Artifact Manifest ===\n";
        std::cout << "backend=" << to_string(artifact.manifest.backend_mode)
                  << " tiles=" << artifact.manifest.tile_count
                  << " commands=" << artifact.manifest.command_count
                  << " dma_desc=" << artifact.manifest.dma_descriptor_count
                  << " compute_desc=" << artifact.manifest.compute_descriptor_count
                  << " barrier_desc=" << artifact.manifest.barrier_descriptor_count
                  << "\n";
        std::cout << "\n=== Runtime Stats ===\n";
        std::cout << "compute_tiles=" << stats.compute_tiles << "\n";
        std::cout << "dma_load_bytes=" << stats.dma_load_bytes << "\n";
        std::cout << "dma_store_bytes=" << stats.dma_store_bytes << "\n";
        std::cout << "overlap_triplets=" << stats.overlapped_triplets << "\n";
        std::cout << "overlap_percent=" << stats.overlap_percent << "\n";
        std::cout << "\n=== Artifact JSON Preview ===\n" << serialize_artifact_json(artifact);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FATAL: " << ex.what() << "\n";
        return 1;
    }
}
