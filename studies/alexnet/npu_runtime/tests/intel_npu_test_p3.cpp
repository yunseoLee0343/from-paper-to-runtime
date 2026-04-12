#include "intel_npu_pipeline.hpp"
#include "ir/npu_ir.hpp"

#include <cassert>
#include <iomanip>
#include <iostream>

using namespace intel_npu;

int main() {
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

    auto contracts = GraphTileScheduler::schedule_from_ir(ir, tensors, hw, &log);
    run_memory_planning_pass(contracts, hw, &log);
    const auto commands = lower_graph_stream(contracts, tensors);

    assert(!commands.empty());

    bool saw_bank0_compute = false;
    bool saw_bank1_load = false;
    bool saw_triplet_pattern = false;
    for (std::size_t i = 0; i < commands.size(); ++i) {
        if (commands[i].kind == PipelineCommandKind::COMPUTE_CONV && commands[i].bank_id == 0) {
            saw_bank0_compute = true;
        }
        if (commands[i].kind == PipelineCommandKind::DMA_LOAD && commands[i].bank_id == 1) {
            saw_bank1_load = true;
        }
        if (i + 2 < commands.size() &&
            commands[i].kind == PipelineCommandKind::DMA_LOAD &&
            commands[i + 1].kind == PipelineCommandKind::COMPUTE_CONV &&
            commands[i + 2].kind == PipelineCommandKind::DMA_STORE) {
            saw_triplet_pattern = true;
        }
    }

    assert(saw_bank0_compute);
    assert(saw_bank1_load);
    assert(saw_triplet_pattern);

    PipelineRuntime runtime_a(hw);
    runtime_a.install_barriers(contracts);
    const auto stats_a = runtime_a.run(commands);
    const auto trace_a = dump_command_trace(runtime_a.trace());

    PipelineRuntime runtime_b(hw);
    runtime_b.install_barriers(contracts);
    const auto stats_b = runtime_b.run(commands);
    const auto trace_b = dump_command_trace(runtime_b.trace());

    assert(stats_a.compute_tiles > 0);
    assert(stats_a.dma_load_bytes > 0);
    assert(stats_a.total_cycles > 0);
    assert(stats_a.dma_compute_overlap_cycles > 0);
    assert(stats_a.overlap_percent > 0.0);
    assert(stats_a.overlap_percent <= 100.0);
    assert(stats_a.total_cycles == stats_b.total_cycles);
    assert(trace_a == trace_b);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Intel NPU Phase 3 test passed\n";
    std::cout << "  Total cycles      : " << stats_a.total_cycles << "\n";
    std::cout << "  Overlap triplets  : " << stats_a.overlapped_triplets << "\n";
    std::cout << "  DMA/Compute overlap % : " << stats_a.overlap_percent << "\n";
    return 0;
}
