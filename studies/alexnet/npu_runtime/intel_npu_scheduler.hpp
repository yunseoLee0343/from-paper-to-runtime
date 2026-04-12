#pragma once

#include "intel_npu_defs.hpp"

#include <map>
#include <string>
#include <vector>

namespace intel_npu {

struct NpuGraphIR;
struct PassLog;

struct Conv2DParams {
    i32 stride_h = 1;
    i32 stride_w = 1;
    i32 pad_h = 0;
    i32 pad_w = 0;
    i32 kernel_h = 3;
    i32 kernel_w = 3;
};

enum class OpKind {
    CONV2D,
    RELU,
    CLAMP,
    MAXPOOL,
    DROPOUT,
    IDENTITY,
    FALLBACK,
    LRN,
};

struct TensorDesc {
    u32 id = 0;
    std::string name;
    std::vector<i32> shape;
    MemorySpace mem_space = MemorySpace::DDR;
    u64 base_addr = 0;
    u64 byte_size = 0;
};

struct TensorTable {
    std::map<u32, TensorDesc> table;

    const TensorDesc& get(u32 id) const;
};

struct GraphOpDesc {
    u32 op_id = 0;
    OpKind kind = OpKind::CONV2D;
    std::string name;
    u32 input_tensor_id = 0;
    u32 weight_tensor_id = 0;
    u32 output_tensor_id = 0;
    Conv2DParams conv;
    float clamp_min = 0.0f;
    float clamp_max = 0.0f;
};

struct GraphDesc {
    std::vector<GraphOpDesc> ops;
};

struct TileRegion {
    i32 n0 = 0;
    i32 n1 = 0;
    i32 c0 = 0;
    i32 c1 = 0;
    i32 h0 = 0;
    i32 h1 = 0;
    i32 w0 = 0;
    i32 w1 = 0;

    i32 n_extent() const { return n1 - n0; }
    i32 c_extent() const { return c1 - c0; }
    i32 h_extent() const { return h1 - h0; }
    i32 w_extent() const { return w1 - w0; }
};

struct BarrierState {
    bool signaled = false;
};

struct TileTask {
    u32 global_tile_id = 0;
    u32 op_id = 0;
    i32 stage_id = 0;
    i32 seq_no = 0;
    i32 pipeline_phase = 0;
};

struct TileMemoryPlan {
    u64 bank_id = 0;
    LocalAlloc input_tile;
    LocalAlloc weight_tile;
    LocalAlloc output_tile;
    bool reuses_producer_sram = false;
    u32 reused_from_tile_id = 0;
    i32 lifetime_begin = 0;
    i32 lifetime_end = 0;
};

struct ConvTileContract {
    TileTask task;
    u32 input_tensor_id = 0;
    u32 weight_tensor_id = 0;
    u32 output_tensor_id = 0;
    Conv2DParams params;
    PostOpKind post_op = PostOpKind::NONE;
    float clamp_min = 0.0f;
    float clamp_max = 0.0f;
    TileRegion input_region;
    TileRegion weight_region;
    TileRegion output_region;
    TileMemoryPlan mem_plan;
    std::vector<u32> wait_barriers;
    std::vector<u32> signal_barriers;
    std::vector<u32> producer_tile_ids;
    std::vector<u32> consumer_tile_ids;
};

struct NpuExecutableStage {
    i32 stage_id = 0;
    u32 source_op_id = 0;
    OpKind kind = OpKind::CONV2D;
    u32 input_tensor_id = 0;
    u32 weight_tensor_id = 0;
    u32 output_tensor_id = 0;
    Conv2DParams conv;
    PostOpKind post_op = PostOpKind::NONE;
    float clamp_min = 0.0f;
    float clamp_max = 0.0f;
    bool is_fallback_stage = false;
    std::string debug_name;
};

struct SchedulerValidationResult {
    bool completed = false;
    std::size_t issued_waits = 0;
    std::size_t issued_signals = 0;
    std::size_t reused_tiles = 0;
};

class GraphTileScheduler {
public:
    static std::vector<NpuExecutableStage> build_stage_sequence(const NpuGraphIR& graph, PassLog* log = nullptr);

    static std::vector<ConvTileContract> schedule_from_ir(
        const NpuGraphIR& graph,
        const TensorTable& tensors,
        const HardwareCaps& hw,
        PassLog* log = nullptr);

    static std::vector<ConvTileContract> schedule_two_conv_chain(
        const GraphDesc& graph,
        const TensorTable& tensors,
        const HardwareCaps& hw);
};

bool run_memory_planning_pass(std::vector<ConvTileContract>& tiles, const HardwareCaps& hw, PassLog* log);
SchedulerValidationResult validate_barrier_schedule(const std::vector<ConvTileContract>& tiles);

TensorTable make_demo_tensors_p2();
GraphDesc make_demo_graph_p2();
GraphDesc make_compiler_demo_graph();

}  // namespace intel_npu
