#pragma once

#include "intel_npu_scheduler.hpp"

#include <map>
#include <vector>

namespace intel_npu {

enum class PipelineCommandKind {
    DMA_LOAD,
    DMA_STORE,
    COMPUTE_CONV,
    BARRIER_WAIT,
    BARRIER_SIGNAL,
};

enum class EngineKind {
    DMA,
    COMPUTE,
    CONTROL,
};

struct PipelineCommand {
    u32 command_id = 0;
    PipelineCommandKind kind = PipelineCommandKind::DMA_LOAD;
    EngineKind engine = EngineKind::CONTROL;
    u32 tile_id = 0;
    u32 barrier_id = 0;
    u64 bank_id = 0;
    u64 bytes = 0;
    u64 estimated_cycles = 0;
    bool uses_reused_input = false;
    const ConvTileContract* contract = nullptr;
};

struct CommandExecutionRecord {
    PipelineCommand command;
    u64 start_cycle = 0;
    u64 end_cycle = 0;
};

struct PipelineStats {
    u64 dma_load_bytes = 0;
    u64 dma_store_bytes = 0;
    u64 compute_tiles = 0;
    u64 barrier_waits = 0;
    u64 barrier_signals = 0;
    u64 overlapped_triplets = 0;
    u64 total_cycles = 0;
    u64 dma_busy_cycles = 0;
    u64 compute_busy_cycles = 0;
    u64 dma_compute_overlap_cycles = 0;
    double overlap_percent = 0.0;
};

class OverlapLowerer {
public:
    static std::vector<PipelineCommand> lower_stage_stream(
        const std::vector<const ConvTileContract*>& tiles,
        const TensorTable& tensors);
};

class PipelineRuntime {
public:
    explicit PipelineRuntime(HardwareCaps hw);

    void install_barriers(const std::vector<ConvTileContract>& tiles);
    PipelineStats run(const std::vector<PipelineCommand>& commands);
    const std::vector<CommandExecutionRecord>& trace() const { return trace_; }

private:
    u64 dma_cycles(const PipelineCommand& command) const;
    u64 compute_cycles(const PipelineCommand& command) const;
    static u64 merge_interval_length(std::vector<std::pair<u64, u64>> intervals);
    static u64 overlap_interval_length(
        std::vector<std::pair<u64, u64>> lhs,
        std::vector<std::pair<u64, u64>> rhs);

    HardwareCaps hw_;
    std::map<u32, u64> barrier_ready_cycle_;
    std::map<u32, u64> tile_compute_end_cycle_;
    std::map<u32, u64> tile_store_end_cycle_;
    std::vector<CommandExecutionRecord> trace_;
};

std::vector<const ConvTileContract*> collect_stage_tiles(
    const std::vector<ConvTileContract>& all_tiles,
    i32 stage_id);

std::vector<PipelineCommand> lower_graph_stream(
    const std::vector<ConvTileContract>& all_tiles,
    const TensorTable& tensors);

}  // namespace intel_npu
