#include "intel_npu_scheduler.hpp"

#include "ir/npu_ir.hpp"

#include <algorithm>
#include <map>
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

ConvProblem make_problem(const NpuExecutableStage& stage, const TensorTable& tensors) {
    const auto& in = tensors.get(stage.input_tensor_id);
    const auto& out = tensors.get(stage.output_tensor_id);
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
    problem.params = stage.conv;
    return problem;
}

std::vector<ConvTileContract> schedule_conv_stage_tiles(
    const NpuExecutableStage& stage,
    const TensorTable& tensors,
    const HardwareCaps& hw,
    u32 tile_id_base) {
    const auto problem = make_problem(stage, tensors);
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
                tile.task.op_id = stage.source_op_id;
                tile.task.stage_id = stage.stage_id;
                tile.task.seq_no = seq_no;
                tile.task.pipeline_phase = seq_no % static_cast<i32>(std::max<u64>(1, hw.sram_bank_count));
                tile.input_tensor_id = stage.input_tensor_id;
                tile.weight_tensor_id = stage.weight_tensor_id;
                tile.output_tensor_id = stage.output_tensor_id;
                tile.params = stage.conv;
                tile.post_op = stage.post_op;
                tile.clamp_min = stage.clamp_min;
                tile.clamp_max = stage.clamp_max;

                tile.output_region = {0, problem.n, oc0, oc1, oh0, oh1, ow0, ow1};

                const i32 in_h0 = std::max(0, oh0 * problem.params.stride_h - problem.params.pad_h);
                const i32 in_w0 = std::max(0, ow0 * problem.params.stride_w - problem.params.pad_w);
                const i32 in_h1 = std::min(problem.ih, (oh1 - 1) * problem.params.stride_h - problem.params.pad_h + problem.params.kernel_h);
                const i32 in_w1 = std::min(problem.iw, (ow1 - 1) * problem.params.stride_w - problem.params.pad_w + problem.params.kernel_w);

                tile.input_region = {0, problem.n, 0, problem.ic, in_h0, in_h1, in_w0, in_w1};
                tile.weight_region = {0, 1, oc0, oc1, 0, problem.ic, 0, problem.params.kernel_h * problem.params.kernel_w};
                tile.signal_barriers.push_back(barrier_counter++);
                tiles.push_back(tile);
                ++seq_no;
            }
        }
    }

    return tiles;
}

void build_stage_dependencies(std::vector<ConvTileContract>& producer_tiles, std::vector<ConvTileContract>& consumer_tiles) {
    auto by_region = [](const ConvTileContract* lhs, const ConvTileContract* rhs) {
        return std::tie(lhs->output_region.c0, lhs->output_region.h0, lhs->output_region.w0, lhs->task.seq_no) <
               std::tie(rhs->output_region.c0, rhs->output_region.h0, rhs->output_region.w0, rhs->task.seq_no);
    };

    std::vector<ConvTileContract*> producers;
    producers.reserve(producer_tiles.size());
    for (auto& producer : producer_tiles) {
        producers.push_back(&producer);
    }
    std::sort(producers.begin(), producers.end(), by_region);

    auto consumer_key = [](const ConvTileContract& tile) {
        return std::tie(tile.input_region.c0, tile.input_region.h0, tile.input_region.w0, tile.task.seq_no);
    };
    std::sort(consumer_tiles.begin(), consumer_tiles.end(), [&](const ConvTileContract& lhs, const ConvTileContract& rhs) {
        return consumer_key(lhs) < consumer_key(rhs);
    });

    std::size_t window_begin = 0;
    for (auto& consumer : consumer_tiles) {
        while (window_begin < producers.size()) {
            const auto& region = producers[window_begin]->output_region;
            const bool definitely_before_h = region.h1 <= consumer.input_region.h0;
            const bool definitely_before_c = region.c1 <= consumer.input_region.c0;
            if (!(definitely_before_h || definitely_before_c)) {
                break;
            }
            ++window_begin;
        }

        for (std::size_t i = window_begin; i < producers.size(); ++i) {
            auto& producer = *producers[i];

            if (producer.output_region.c0 >= consumer.input_region.c1) {
                break;
            }
            if (producer.output_region.h0 >= consumer.input_region.h1) {
                break;
            }
            if (!overlaps_spatial(producer.output_region, consumer.input_region)) {
                continue;
            }
            if (!overlaps_channels(producer.output_region, consumer.input_region)) {
                continue;
            }

            consumer.wait_barriers.push_back(producer.signal_barriers.front());
            consumer.producer_tile_ids.push_back(producer.task.global_tile_id);
            producer.consumer_tile_ids.push_back(consumer.task.global_tile_id);
        }

        std::sort(consumer.wait_barriers.begin(), consumer.wait_barriers.end());
        consumer.wait_barriers.erase(std::unique(consumer.wait_barriers.begin(), consumer.wait_barriers.end()), consumer.wait_barriers.end());
    }
}

TileMemoryPlan make_tile_memory_plan(const ConvTileContract& tile, const HardwareCaps& hw) {
    const u64 bank_count = std::max<u64>(1, hw.sram_bank_count);
    const u64 bank_size = hw.local_sram_bytes / bank_count;
    const u64 bank_id = static_cast<u64>(tile.task.pipeline_phase % static_cast<i32>(bank_count));

    TileMemoryPlan plan;
    plan.bank_id = bank_id;
    plan.lifetime_begin = tile.task.seq_no;
    plan.lifetime_end = tile.task.seq_no;

    u64 offset = 0;
    plan.input_tile = {"input_tile", MemorySpace::LOCAL_SRAM, bank_id, align_up(offset, hw.dma_alignment), tile_elements(tile.input_region) * 2, hw.dma_alignment};
    offset = plan.input_tile.end();
    plan.weight_tile = {"weight_tile", MemorySpace::LOCAL_SRAM, bank_id, align_up(offset, hw.dma_alignment), tile_elements(tile.weight_region) * 2, hw.dma_alignment};
    offset = plan.weight_tile.end();
    plan.output_tile = {"output_tile", MemorySpace::LOCAL_SRAM, bank_id, align_up(offset, hw.dma_alignment), tile_elements(tile.output_region) * 2, hw.dma_alignment};

    if (plan.output_tile.end() > bank_size) {
        throw std::runtime_error("Tile memory plan exceeds SRAM bank size");
    }
    return plan;
}

}  // namespace

const TensorDesc& TensorTable::get(u32 id) const {
    auto it = table.find(id);
    if (it == table.end()) {
        throw std::runtime_error("Unknown tensor id");
    }
    return it->second;
}

std::vector<NpuExecutableStage> GraphTileScheduler::build_stage_sequence(const NpuGraphIR& graph, PassLog* log) {
    std::vector<NpuExecutableStage> stages;
    i32 stage_id = 0;

    for (const auto& op : graph.ops) {
        if (op.removed) {
            continue;
        }
        if (op.kind == NpuOpKind::CONV) {
            NpuExecutableStage stage;
            stage.stage_id = stage_id++;
            stage.source_op_id = op.id;
            stage.kind = OpKind::CONV2D;
            stage.debug_name = op.name;
            stage.input_tensor_id = op.inputs.empty() ? 0 : op.inputs.front();
            stage.weight_tensor_id = op.inputs.size() > 1 ? op.inputs[1] : 0;
            stage.output_tensor_id = op.outputs.empty() ? 0 : op.outputs.front();
            stage.conv = {op.attrs.conv.stride_h, op.attrs.conv.stride_w, op.attrs.conv.pad_h, op.attrs.conv.pad_w, op.attrs.conv.kernel_h, op.attrs.conv.kernel_w};
            stage.post_op = op.post_op;
            stage.clamp_min = op.attrs.clamp.min_value;
            stage.clamp_max = op.attrs.clamp.max_value;
            stages.push_back(stage);
            continue;
        }
        if (op.kind == NpuOpKind::MAXPOOL || op.kind == NpuOpKind::FALLBACK || op.kind == NpuOpKind::LRN) {
            NpuExecutableStage stage;
            stage.stage_id = stage_id++;
            stage.source_op_id = op.id;
            stage.kind = OpKind::FALLBACK;
            stage.debug_name = op.name;
            stage.input_tensor_id = op.inputs.empty() ? 0 : op.inputs.front();
            stage.output_tensor_id = op.outputs.empty() ? 0 : op.outputs.front();
            stage.is_fallback_stage = true;
            stages.push_back(stage);
            if (log != nullptr) {
                log->append("TileScheduler", "non-tilable op " + std::to_string(op.id) + " kept as fallback stage");
            }
        }
    }

    return stages;
}

std::vector<ConvTileContract> GraphTileScheduler::schedule_from_ir(
    const NpuGraphIR& graph,
    const TensorTable& tensors,
    const HardwareCaps& hw,
    PassLog* log) {
    const auto stages = build_stage_sequence(graph, log);
    std::vector<ConvTileContract> all_tiles;
    std::vector<std::vector<ConvTileContract>> per_stage_tiles;
    u32 tile_id_base = 1;

    for (const auto& stage : stages) {
        if (stage.is_fallback_stage) {
            continue;
        }
        auto stage_tiles = schedule_conv_stage_tiles(stage, tensors, hw, tile_id_base);
        tile_id_base += static_cast<u32>(stage_tiles.size());
        per_stage_tiles.push_back(stage_tiles);
    }

    for (std::size_t i = 1; i < per_stage_tiles.size(); ++i) {
        build_stage_dependencies(per_stage_tiles[i - 1], per_stage_tiles[i]);
    }

    for (auto& stage_tiles : per_stage_tiles) {
        all_tiles.insert(all_tiles.end(), stage_tiles.begin(), stage_tiles.end());
    }
    return all_tiles;
}

std::vector<ConvTileContract> GraphTileScheduler::schedule_two_conv_chain(
    const GraphDesc& graph,
    const TensorTable& tensors,
    const HardwareCaps& hw) {
    auto ir = lower_graph_desc_to_npu_ir(graph, tensors);
    auto tiles = schedule_from_ir(ir, tensors, hw, nullptr);
    run_memory_planning_pass(tiles, hw, nullptr);
    return tiles;
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
    table.table[6] = {6, "relu_out", {1, 64, 32, 32}, MemorySpace::DDR, 0x6000, static_cast<u64>(1) * 64 * 32 * 32 * 2};
    table.table[7] = {7, "dropout_out", {1, 64, 32, 32}, MemorySpace::DDR, 0x7000, static_cast<u64>(1) * 64 * 32 * 32 * 2};
    table.table[8] = {8, "lrn_out", {1, 64, 32, 32}, MemorySpace::DDR, 0x8000, static_cast<u64>(1) * 64 * 32 * 32 * 2};
    return table;
}

GraphDesc make_demo_graph_p2() {
    GraphDesc graph;
    graph.ops.push_back({10, OpKind::CONV2D, "conv0", 1, 2, 3, {1, 1, 1, 1, 3, 3}});
    graph.ops.push_back({11, OpKind::CONV2D, "conv1", 3, 4, 5, {1, 1, 1, 1, 3, 3}});
    return graph;
}

GraphDesc make_compiler_demo_graph() {
    GraphDesc graph;

    GraphOpDesc conv0;
    conv0.op_id = 10;
    conv0.kind = OpKind::CONV2D;
    conv0.name = "conv0";
    conv0.input_tensor_id = 1;
    conv0.weight_tensor_id = 2;
    conv0.output_tensor_id = 3;
    conv0.conv = {1, 1, 1, 1, 3, 3};

    GraphOpDesc relu0;
    relu0.op_id = 11;
    relu0.kind = OpKind::RELU;
    relu0.name = "relu0";
    relu0.input_tensor_id = 3;
    relu0.output_tensor_id = 6;

    GraphOpDesc dropout0;
    dropout0.op_id = 12;
    dropout0.kind = OpKind::DROPOUT;
    dropout0.name = "dropout0";
    dropout0.input_tensor_id = 6;
    dropout0.output_tensor_id = 7;

    GraphOpDesc conv1;
    conv1.op_id = 13;
    conv1.kind = OpKind::CONV2D;
    conv1.name = "conv1";
    conv1.input_tensor_id = 7;
    conv1.weight_tensor_id = 4;
    conv1.output_tensor_id = 5;
    conv1.conv = {1, 1, 1, 1, 3, 3};

    GraphOpDesc lrn0;
    lrn0.op_id = 14;
    lrn0.kind = OpKind::LRN;
    lrn0.name = "lrn0";
    lrn0.input_tensor_id = 5;
    lrn0.output_tensor_id = 8;

    graph.ops = {conv0, relu0, dropout0, conv1, lrn0};
    return graph;
}

}  // namespace intel_npu


