#include "intel_npu_pipeline.hpp"
#include "ir/npu_ir.hpp"
#include "runtime/descriptor_types.hpp"

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

    PassLog log;
    const auto tensors = make_demo_tensors_p2();
    auto ir = lower_graph_desc_to_npu_ir(make_compiler_demo_graph(), tensors);
    run_strip_dropout_pass(ir, &log);
    run_fuse_activation_pass(ir, &log);
    run_decompose_unsupported_pass(ir, UnsupportedPolicy::FALLBACK, &log);
    run_canonicalize_pass(ir, &log);

    auto tiles = GraphTileScheduler::schedule_from_ir(ir, tensors, hw, &log);
    run_memory_planning_pass(tiles, hw, &log);
    auto commands = lower_graph_stream(tiles, tensors);
    const auto artifact = run_emit_descriptors_pass(tiles, commands, &log);
    const auto json = serialize_artifact_json(artifact);

    assert(artifact.manifest.backend_mode == BackendMode::SIMULATOR);
    assert(artifact.manifest.tile_count == tiles.size());
    assert(artifact.manifest.command_count == commands.size());
    assert(!artifact.dma_descriptors.empty());
    assert(!artifact.compute_descriptors.empty());
    assert(!artifact.barrier_descriptors.empty());

    bool saw_relu_post_op = false;
    for (const auto& desc : artifact.compute_descriptors) {
        if (desc.post_op == PostOpKind::RELU) {
            saw_relu_post_op = true;
        }
    }
    assert(saw_relu_post_op);
    assert(json.find("backend_mode") != std::string::npos);
    assert(json.find("compute_descriptors") != std::string::npos);
    assert(json.find("barrier_descriptors") != std::string::npos);

    std::cout << "Intel NPU descriptor emission tests passed\n";
    return 0;
}
