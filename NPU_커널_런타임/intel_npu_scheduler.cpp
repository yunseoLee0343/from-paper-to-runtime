#include "intel_npu_scheduler.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace intel_npu {

namespace {

struct ConvProblem {
    i32 n = 1;
    i32 ic = 0;
    i32 ih = 0;
    i32 iw = 0;
    i32 oc = 0;
    i32 oh = 0;
    i32 ow = 0;
    Conv2DParams params;
};

u64 tile_elements(const TileRegion& region) {
    return static_cast<u64>(region.n_extent()) *
           static_cast<u64>(region.c_extent()) *
           static_cast<u64>(region.h_extent()) *
           static_cast<u64>(region.w_extent());
}

bool overlaps_spatial(const TileRegion& a, const TileRegion& b) {
    const bool h_overlap = !(a.h1 <= b.h0 || b.h1 <= a.h0);
    const bool w_overlap = !(a.w1 <= b.w0 || b.w1 <= a.w0);
    return h_overlap && w_overlap;
}

bool overlaps_channels(const TileRegion& a, const TileRegion& b) {
    return !(a.c1 <= b.c0 || b.c1 <= a.c0);
}

ConvProblem make_problem(const GraphOpDesc& op, const TensorTable& tensors) {
    const auto& in = tensors.get(op.input_tensor_id);
    const auto& out = tensors.get(op.output_tensor_id);
    if (in.shape.size() != 4 || out.shape.size() != 4) {
        throw std::runtime_error("Expected 4D NCHW tensor shapes");
    }

    ConvProblem problem;
    problem.n = in.shape[0];
    problem.ic = in.shape[1];
    problem.ih = in.shape[2];
    problem.iw = in.shape[3];
    problem.oc = out.shape[1];
    problem.oh = out.shape[2];
    problem.ow = out.shape[3];
    problem.params = op.conv;
    return problem;
}

TileMemoryPlan make_memory_plan(
    const ConvTileContract& contract,
    const HardwareCaps& hw,
    bool reuse_output,
    u32 reused_from_tile_id) {
    const u64 bank_count = hw.sram_bank_count == 0 ? 1 : hw.sram_bank_count;
    const u64 bank_size = hw.local_sram_bytes / bank_count;
    const u64 bank_id = static_cast<u64>(contract.task.pipeline_phase % static_cast<i32>(bank_count));
    const u64 act_bytes = 2;
    const u64 weight_bytes = 2;

    TileMemoryPlan plan;
    plan.bank_id = bank_id;
    plan.reuses_producer_sram = reuse_output;
    plan.reused_from_tile_id = reused_from_tile_id;

    u64 offset = 0;
    plan.input_tile = {
        reuse_output ? "reused_input_tile" : "input_tile",
        MemorySpace::LOCAL_SRAM,
        bank_id,
        0,
        tile_elements(contract.input_region) * act_bytes,
        hw.dma_alignment,
    };

    if (reuse_output) {
        offset = align_up(plan.input_tile.bytes, hw.dma_alignment);
    } else {
        plan.input_tile.offset = align_up(offset, hw.dma_alignment);
        offset = plan.input_tile.end();
    }

    plan.weight_tile = {
        "weight_tile",
        MemorySpace::LOCAL_SRAM,
        bank_id,
        align_up(offset, hw.dma_alignment),
        tile_elements(contract.weight_region) * weight_bytes,
        hw.dma_alignment,
    };
    offset = plan.weight_tile.end();

    plan.output_tile = {
        "output_tile",
        MemorySpace::LOCAL_SRAM,
        bank_id,
        align_up(offset, hw.dma_alignment),
        tile_elements(contract.output_region) * act_bytes,
        hw.dma_alignment,
    };

    if (plan.output_tile.end() > bank_size) {
        throw std::runtime_error("Tile memory plan exceeds SRAM bank size");
    }
    return plan;
}

std::vector<ConvTileContract> schedule_conv_tiles(
    const GraphOpDesc& op,
    const ConvProblem& problem,
    const HardwareCaps& hw,
    i32 stage_id,
    u32 tile_id_base) {
    std::vector<ConvTileContract> tiles;
    const i32 tile_h = std::min(hw.max_tile_h, problem.oh);
    const i32 tile_w = std::min(hw.max_tile_w, problem.ow);
    const i32 tile_oc = std::min(hw.max_tile_c, problem.oc);

    u32 barrier_counter = tile_id_base * 1000;
    u32 next_tile_id = tile_id_base;
    i32 seq_no = 0;

    for (i32 oc0 = 0; oc0 < problem.oc; oc0 += tile_oc) {
        const i32 oc1 = std::min(oc0 + tile_oc, problem.oc);
        for (i32 oh0 = 0; oh0 < problem.oh; oh0 += tile_h) {
            const i32 oh1 = std::min(oh0 + tile_h, problem.oh);
            for (i32 ow0 = 0; ow0 < problem.ow; ow0 += tile_w) {
                const i32 ow1 = std::min(ow0 + tile_w, problem.ow);

                ConvTileContract tile;
                tile.task.global_tile_id = next_tile_id++;
                tile.task.op_id = op.op_id;
                tile.task.stage_id = stage_id;
                tile.task.seq_no = seq_no;
                tile.task.pipeline_phase = seq_no % static_cast<i32>(std::max<u64>(1, hw.sram_bank_count));
                tile.input_tensor_id = op.input_tensor_id;
                tile.weight_tensor_id = op.weight_tensor_id;
                tile.output_tensor_id = op.output_tensor_id;
                tile.params = op.conv;

                tile.output_region = {0, problem.n, oc0, oc1, oh0, oh1, ow0, ow1};

                const i32 in_h0 = std::max(0, oh0 * problem.params.stride_h - problem.params.pad_h);
                const i32 in_w0 = std::max(0, ow0 * problem.params.stride_w - problem.params.pad_w);
                const i32 in_h1 = std::min(
                    problem.ih,
                    (oh1 - 1) * problem.params.stride_h - problem.params.pad_h + problem.params.kernel_h);
                const i32 in_w1 = std::min(
                    problem.iw,
                    (ow1 - 1) * problem.params.stride_w - problem.params.pad_w + problem.params.kernel_w);

                tile.input_region = {0, problem.n, 0, problem.ic, in_h0, in_h1, in_w0, in_w1};
                tile.weight_region = {0, 1, oc0, oc1, 0, problem.ic, 0, problem.params.kernel_h * problem.params.kernel_w};
                tile.signal_barriers.push_back(barrier_counter++);
                tile.mem_plan = make_memory_plan(tile, hw, false, 0);

                tiles.push_back(tile);
                ++seq_no;
            }
        }
    }

    return tiles;
}

void build_tile_reuse_dependencies(
    std::vector<ConvTileContract>& producers,
    std::vector<ConvTileContract>& consumers,
    const HardwareCaps& hw) {
    const u64 bank_count = std::max<u64>(1, hw.sram_bank_count);

    for (auto& consumer : consumers) {
        bool reused = false;
        for (auto& producer : producers) {
            if (!overlaps_spatial(producer.output_region, consumer.input_region)) {
                continue;
            }
            if (!overlaps_channels(producer.output_region, consumer.input_region)) {
                continue;
            }

            consumer.wait_barriers.push_back(producer.signal_barriers.front());
            consumer.producer_tile_ids.push_back(producer.task.global_tile_id);
            producer.consumer_tile_ids.push_back(consumer.task.global_tile_id);

            if (!reused &&
                overlaps_channels(producer.output_region, consumer.input_region) &&
                overlaps_spatial(producer.output_region, consumer.input_region)) {
                consumer.mem_plan.input_tile = producer.mem_plan.output_tile;
                consumer.mem_plan.input_tile.name = "reused_input_tile";
                consumer.mem_plan.reuses_producer_sram = true;
                consumer.mem_plan.reused_from_tile_id = producer.task.global_tile_id;
                consumer.task.pipeline_phase = producer.task.pipeline_phase;
                consumer.mem_plan.bank_id = producer.mem_plan.bank_id;
                reused = true;
            }
        }

        if (!reused) {
            consumer.task.pipeline_phase = consumer.task.seq_no % static_cast<i32>(bank_count);
            consumer.mem_plan = make_memory_plan(consumer, hw, false, 0);
        }

        std::sort(consumer.wait_barriers.begin(), consumer.wait_barriers.end());
        consumer.wait_barriers.erase(
            std::unique(consumer.wait_barriers.begin(), consumer.wait_barriers.end()),
            consumer.wait_barriers.end());
    }
}

}  // namespace

const TensorDesc& TensorTable::get(u32 id) const {
    auto it = table.find(id);
    if (it == table.end()) {
        throw std::runtime_error("Unknown tensor id");
    }
    return it->second;
}

std::vector<ConvTileContract> GraphTileScheduler::schedule_two_conv_chain(
    const GraphDesc& graph,
    const TensorTable& tensors,
    const HardwareCaps& hw) {
    if (graph.ops.size() != 2) {
        throw std::runtime_error("schedule_two_conv_chain expects exactly two conv ops");
    }

    const auto& conv0 = graph.ops[0];
    const auto& conv1 = graph.ops[1];
    if (conv0.kind != OpKind::CONV2D || conv1.kind != OpKind::CONV2D) {
        throw std::runtime_error("Only Conv2D ops are supported");
    }

    auto problem0 = make_problem(conv0, tensors);
    auto problem1 = make_problem(conv1, tensors);

    auto stage0 = schedule_conv_tiles(conv0, problem0, hw, 0, 1);
    auto stage1 = schedule_conv_tiles(conv1, problem1, hw, 1, static_cast<u32>(stage0.size() + 1));

    build_tile_reuse_dependencies(stage0, stage1, hw);

    std::vector<ConvTileContract> all;
    all.reserve(stage0.size() + stage1.size());
    all.insert(all.end(), stage0.begin(), stage0.end());
    all.insert(all.end(), stage1.begin(), stage1.end());
    return all;
}

SchedulerValidationResult validate_barrier_schedule(const std::vector<ConvTileContract>& tiles) {
    std::map<u32, BarrierState> barriers;
    std::set<u32> completed_tiles;
    SchedulerValidationResult result;

    for (const auto& tile : tiles) {
        for (u32 id : tile.signal_barriers) {
            barriers.emplace(id, BarrierState{});
        }
    }

    bool progress = true;
    while (completed_tiles.size() < tiles.size() && progress) {
        progress = false;
        for (const auto& tile : tiles) {
            if (completed_tiles.count(tile.task.global_tile_id) != 0) {
                continue;
            }

            bool ready = true;
            for (u32 wait_id : tile.wait_barriers) {
                auto it = barriers.find(wait_id);
                if (it == barriers.end() || !it->second.signaled) {
                    ready = false;
                    break;
                }
            }
            if (!ready) {
                continue;
            }

            progress = true;
            completed_tiles.insert(tile.task.global_tile_id);
            result.issued_waits += tile.wait_barriers.size();
            result.issued_signals += tile.signal_barriers.size();
            if (tile.mem_plan.reuses_producer_sram) {
                ++result.reused_tiles;
            }
            for (u32 signal_id : tile.signal_barriers) {
                barriers[signal_id].signaled = true;
            }
        }
    }

    result.completed = completed_tiles.size() == tiles.size();
    if (!result.completed) {
        throw std::runtime_error("Deadlock detected in barrier dependency graph");
    }
    return result;
}

TensorTable make_demo_tensors_p2() {
    TensorTable table;
    table.table[1] = {1, "input0", {1, 64, 32, 32}, MemorySpace::DDR, 0x1000, static_cast<u64>(1) * 64 * 32 * 32 * 2};
    table.table[2] = {2, "weight0", {64, 64, 3, 3}, MemorySpace::DDR, 0x2000, static_cast<u64>(64) * 64 * 3 * 3 * 2};
    table.table[3] = {3, "mid0", {1, 64, 32, 32}, MemorySpace::DDR, 0x3000, static_cast<u64>(1) * 64 * 32 * 32 * 2};
    table.table[4] = {4, "weight1", {64, 64, 3, 3}, MemorySpace::DDR, 0x4000, static_cast<u64>(64) * 64 * 3 * 3 * 2};
    table.table[5] = {5, "output1", {1, 64, 32, 32}, MemorySpace::DDR, 0x5000, static_cast<u64>(1) * 64 * 32 * 32 * 2};
    return table;
}

GraphDesc make_demo_graph_p2() {
    GraphDesc graph;
    graph.ops.push_back({10, OpKind::CONV2D, "conv0", 1, 2, 3, {1, 1, 1, 1, 3, 3}});
    graph.ops.push_back({11, OpKind::CONV2D, "conv1", 3, 4, 5, {1, 1, 1, 1, 3, 3}});
    return graph;
}

}  // namespace intel_npu
