#include "intel_npu_pipeline.hpp"

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

    const auto tensors = make_demo_tensors_p2();
    const auto graph = make_demo_graph_p2();
    const auto contracts = GraphTileScheduler::schedule_two_conv_chain(graph, tensors, hw);
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

    PipelineRuntime runtime(hw);
    runtime.install_barriers(contracts);
    const auto stats = runtime.run(commands);

    assert(stats.compute_tiles > 0);
    assert(stats.dma_load_bytes > 0);
    assert(stats.total_cycles > 0);
    assert(stats.dma_compute_overlap_cycles > 0);
    assert(stats.overlap_percent > 0.0);
    assert(stats.overlap_percent <= 100.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Intel NPU Phase 3 test passed\n";
    std::cout << "  Total cycles      : " << stats.total_cycles << "\n";
    std::cout << "  Overlap triplets  : " << stats.overlapped_triplets << "\n";
    std::cout << "  DMA/Compute overlap % : " << stats.overlap_percent << "\n";
    return 0;
}
