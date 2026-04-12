#include "intel_npu_pipeline.hpp"

#include <algorithm>
#include <stdexcept>

namespace intel_npu {

namespace {

PipelineCommand make_dma_command(
    u32& next_id,
    PipelineCommandKind kind,
    const ConvTileContract& tile,
    u64 bytes,
    bool uses_reused_input) {
    PipelineCommand command;
    command.command_id = next_id++;
    command.kind = kind;
    command.engine = EngineKind::DMA;
    command.tile_id = tile.task.global_tile_id;
    command.bank_id = tile.mem_plan.bank_id;
    command.bytes = bytes;
    command.estimated_cycles = 0;
    command.uses_reused_input = uses_reused_input;
    command.contract = &tile;
    return command;
}

PipelineCommand make_compute_command(u32& next_id, const ConvTileContract& tile) {
    PipelineCommand command;
    command.command_id = next_id++;
    command.kind = PipelineCommandKind::COMPUTE_CONV;
    command.engine = EngineKind::COMPUTE;
    command.tile_id = tile.task.global_tile_id;
    command.bank_id = tile.mem_plan.bank_id;
    command.contract = &tile;
    return command;
}

PipelineCommand make_barrier_command(
    u32& next_id,
    PipelineCommandKind kind,
    u32 barrier_id,
    const ConvTileContract& tile) {
    PipelineCommand command;
    command.command_id = next_id++;
    command.kind = kind;
    command.engine = EngineKind::CONTROL;
    command.tile_id = tile.task.global_tile_id;
    command.barrier_id = barrier_id;
    command.bank_id = tile.mem_plan.bank_id;
    command.contract = &tile;
    return command;
}

}  // namespace

std::vector<PipelineCommand> OverlapLowerer::lower_stage_stream(
    const std::vector<const ConvTileContract*>& tiles,
    const TensorTable&) {
    std::vector<PipelineCommand> commands;
    u32 next_id = 1;

    for (std::size_t i = 0; i < tiles.size(); ++i) {
        const ConvTileContract* prev = (i > 0) ? tiles[i - 1] : nullptr;
        const ConvTileContract* curr = tiles[i];
        const ConvTileContract* next = (i + 1 < tiles.size()) ? tiles[i + 1] : nullptr;

        for (u32 barrier_id : curr->wait_barriers) {
            commands.push_back(make_barrier_command(next_id, PipelineCommandKind::BARRIER_WAIT, barrier_id, *curr));
        }

        if (i == 0) {
            if (!curr->mem_plan.reuses_producer_sram) {
                commands.push_back(make_dma_command(
                    next_id, PipelineCommandKind::DMA_LOAD, *curr, curr->mem_plan.input_tile.bytes, false));
            }
            commands.push_back(make_dma_command(
                next_id, PipelineCommandKind::DMA_LOAD, *curr, curr->mem_plan.weight_tile.bytes, curr->mem_plan.reuses_producer_sram));
        }

        if (next != nullptr) {
            if (!next->mem_plan.reuses_producer_sram) {
                commands.push_back(make_dma_command(
                    next_id, PipelineCommandKind::DMA_LOAD, *next, next->mem_plan.input_tile.bytes, false));
            }
            commands.push_back(make_dma_command(
                next_id, PipelineCommandKind::DMA_LOAD, *next, next->mem_plan.weight_tile.bytes, next->mem_plan.reuses_producer_sram));
        }

        commands.push_back(make_compute_command(next_id, *curr));

        if (prev != nullptr && !prev->mem_plan.reuses_producer_sram) {
            commands.push_back(make_dma_command(
                next_id, PipelineCommandKind::DMA_STORE, *prev, prev->mem_plan.output_tile.bytes, false));
        }

        for (u32 barrier_id : curr->signal_barriers) {
            commands.push_back(make_barrier_command(next_id, PipelineCommandKind::BARRIER_SIGNAL, barrier_id, *curr));
        }
    }

    if (!tiles.empty()) {
        const auto* last = tiles.back();
        if (!last->mem_plan.reuses_producer_sram) {
            commands.push_back(make_dma_command(
                next_id, PipelineCommandKind::DMA_STORE, *last, last->mem_plan.output_tile.bytes, false));
        }
    }

    return commands;
}

PipelineRuntime::PipelineRuntime(HardwareCaps hw) : hw_(hw) {}

void PipelineRuntime::install_barriers(const std::vector<ConvTileContract>& tiles) {
    barrier_ready_cycle_.clear();
    for (const auto& tile : tiles) {
        for (u32 barrier_id : tile.signal_barriers) {
            barrier_ready_cycle_[barrier_id] = 0;
        }
    }
}

PipelineStats PipelineRuntime::run(const std::vector<PipelineCommand>& commands) {
    PipelineStats stats;
    trace_.clear();
    tile_compute_end_cycle_.clear();
    tile_store_end_cycle_.clear();

    u64 dma_ready = 0;
    u64 compute_ready = 0;
    u64 control_ready = 0;
    PipelineCommandKind prev2 = PipelineCommandKind::BARRIER_SIGNAL;
    PipelineCommandKind prev1 = PipelineCommandKind::BARRIER_SIGNAL;

    std::vector<std::pair<u64, u64>> dma_intervals;
    std::vector<std::pair<u64, u64>> compute_intervals;

    for (const auto& command : commands) {
        u64 dependency_ready = 0;
        if (command.contract != nullptr) {
            for (u32 barrier_id : command.contract->wait_barriers) {
                auto it = barrier_ready_cycle_.find(barrier_id);
                if (it == barrier_ready_cycle_.end()) {
                    throw std::runtime_error("Unknown barrier in pipeline wait");
                }
                dependency_ready = std::max(dependency_ready, it->second);
            }
        }

        CommandExecutionRecord record;
        record.command = command;

        switch (command.kind) {
            case PipelineCommandKind::BARRIER_WAIT:
                record.start_cycle = std::max(control_ready, barrier_ready_cycle_.at(command.barrier_id));
                record.end_cycle = record.start_cycle;
                control_ready = record.end_cycle;
                stats.barrier_waits++;
                break;
            case PipelineCommandKind::BARRIER_SIGNAL: {
                const u64 tile_ready = tile_compute_end_cycle_.at(command.tile_id);
                record.start_cycle = std::max(control_ready, tile_ready);
                record.end_cycle = record.start_cycle;
                control_ready = record.end_cycle;
                barrier_ready_cycle_[command.barrier_id] = record.end_cycle;
                stats.barrier_signals++;
                break;
            }
            case PipelineCommandKind::DMA_LOAD:
            case PipelineCommandKind::DMA_STORE: {
                const u64 duration = dma_cycles(command);
                record.start_cycle = std::max(dma_ready, dependency_ready);
                record.end_cycle = record.start_cycle + duration;
                dma_ready = record.end_cycle;
                dma_intervals.push_back({record.start_cycle, record.end_cycle});
                if (command.kind == PipelineCommandKind::DMA_LOAD) {
                    stats.dma_load_bytes += command.bytes;
                } else {
                    stats.dma_store_bytes += command.bytes;
                    tile_store_end_cycle_[command.tile_id] = record.end_cycle;
                }
                break;
            }
            case PipelineCommandKind::COMPUTE_CONV: {
                const u64 duration = compute_cycles(command);
                record.start_cycle = std::max(compute_ready, dependency_ready);
                record.end_cycle = record.start_cycle + duration;
                compute_ready = record.end_cycle;
                compute_intervals.push_back({record.start_cycle, record.end_cycle});
                tile_compute_end_cycle_[command.tile_id] = record.end_cycle;
                stats.compute_tiles++;
                break;
            }
        }

        if (prev2 == PipelineCommandKind::DMA_LOAD &&
            prev1 == PipelineCommandKind::COMPUTE_CONV &&
            command.kind == PipelineCommandKind::DMA_STORE) {
            stats.overlapped_triplets++;
        }

        stats.total_cycles = std::max(stats.total_cycles, record.end_cycle);
        trace_.push_back(record);
        prev2 = prev1;
        prev1 = command.kind;
    }

    stats.dma_busy_cycles = merge_interval_length(dma_intervals);
    stats.compute_busy_cycles = merge_interval_length(compute_intervals);
    stats.dma_compute_overlap_cycles = overlap_interval_length(dma_intervals, compute_intervals);
    if (stats.total_cycles != 0) {
        stats.overlap_percent =
            (100.0 * static_cast<double>(stats.dma_compute_overlap_cycles)) / static_cast<double>(stats.total_cycles);
    }

    return stats;
}

u64 PipelineRuntime::dma_cycles(const PipelineCommand& command) const {
    return std::max<u64>(8, align_up(command.bytes, hw_.dma_alignment) / hw_.dma_alignment);
}

u64 PipelineRuntime::compute_cycles(const PipelineCommand& command) const {
    const auto* contract = command.contract;
    if (contract == nullptr) {
        throw std::runtime_error("Compute command missing tile contract");
    }

    const u64 work =
        static_cast<u64>(contract->output_region.c_extent()) *
        static_cast<u64>(contract->output_region.h_extent()) *
        static_cast<u64>(contract->output_region.w_extent()) *
        static_cast<u64>(contract->params.kernel_h) *
        static_cast<u64>(contract->params.kernel_w);
    return std::max<u64>(16, work / std::max<u32>(1, hw_.max_tile_c));
}

u64 PipelineRuntime::merge_interval_length(std::vector<std::pair<u64, u64>> intervals) {
    if (intervals.empty()) {
        return 0;
    }

    std::sort(intervals.begin(), intervals.end());
    u64 total = 0;
    u64 start = intervals.front().first;
    u64 end = intervals.front().second;

    for (std::size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].first <= end) {
            end = std::max(end, intervals[i].second);
        } else {
            total += end - start;
            start = intervals[i].first;
            end = intervals[i].second;
        }
    }
    total += end - start;
    return total;
}

u64 PipelineRuntime::overlap_interval_length(
    std::vector<std::pair<u64, u64>> lhs,
    std::vector<std::pair<u64, u64>> rhs) {
    std::sort(lhs.begin(), lhs.end());
    std::sort(rhs.begin(), rhs.end());

    std::size_t i = 0;
    std::size_t j = 0;
    u64 overlap = 0;

    while (i < lhs.size() && j < rhs.size()) {
        const u64 start = std::max(lhs[i].first, rhs[j].first);
        const u64 end = std::min(lhs[i].second, rhs[j].second);
        if (start < end) {
            overlap += end - start;
        }

        if (lhs[i].second < rhs[j].second) {
            ++i;
        } else {
            ++j;
        }
    }
    return overlap;
}

std::vector<const ConvTileContract*> collect_stage_tiles(
    const std::vector<ConvTileContract>& all_tiles,
    i32 stage_id) {
    std::vector<const ConvTileContract*> out;
    for (const auto& tile : all_tiles) {
        if (tile.task.stage_id == stage_id) {
            out.push_back(&tile);
        }
    }
    std::sort(out.begin(), out.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->task.seq_no < rhs->task.seq_no;
    });
    return out;
}

std::vector<PipelineCommand> lower_graph_stream(
    const std::vector<ConvTileContract>& all_tiles,
    const TensorTable& tensors) {
    std::vector<PipelineCommand> commands;
    const auto stage0 = collect_stage_tiles(all_tiles, 0);
    const auto stage1 = collect_stage_tiles(all_tiles, 1);

    const auto stage0_commands = OverlapLowerer::lower_stage_stream(stage0, tensors);
    commands.insert(commands.end(), stage0_commands.begin(), stage0_commands.end());

    const auto stage1_commands = OverlapLowerer::lower_stage_stream(stage1, tensors);
    commands.insert(commands.end(), stage1_commands.begin(), stage1_commands.end());
    return commands;
}

}  // namespace intel_npu
