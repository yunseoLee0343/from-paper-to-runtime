#include "intel_npu_scheduler.hpp"

#include <cassert>
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

    assert(!contracts.empty());

    std::size_t stage0_count = 0;
    std::size_t stage1_count = 0;
    std::size_t reused_inputs = 0;
    std::size_t waited_tiles = 0;

    for (const auto& contract : contracts) {
        if (contract.task.stage_id == 0) {
            ++stage0_count;
            assert(!contract.signal_barriers.empty());
        }
        if (contract.task.stage_id == 1) {
            ++stage1_count;
            assert(!contract.wait_barriers.empty());
            ++waited_tiles;
            if (contract.mem_plan.reuses_producer_sram) {
                ++reused_inputs;
                assert(contract.mem_plan.reused_from_tile_id != 0);
                assert(contract.mem_plan.input_tile.name == "reused_input_tile");
            }
        }
    }

    assert(stage0_count > 0);
    assert(stage1_count > 0);
    assert(waited_tiles == stage1_count);
    assert(reused_inputs > 0);

    const auto validation = validate_barrier_schedule(contracts);
    assert(validation.completed);
    assert(validation.issued_waits >= stage1_count);
    assert(validation.issued_signals == contracts.size());
    assert(validation.reused_tiles == reused_inputs);

    std::cout << "Intel NPU Phase 2 test passed: "
              << "tiles=" << contracts.size()
              << " reused=" << reused_inputs
              << " waits=" << validation.issued_waits
              << "\n";
    return 0;
}
