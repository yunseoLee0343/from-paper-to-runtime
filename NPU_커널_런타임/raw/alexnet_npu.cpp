#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// ============================================================================
// Raw NPU Runtime Baseline - Graph-level Extension
// ----------------------------------------------------------------------------
// 목표:
// - 이전 single-op / single-tile baseline 위에
//   "graph-level multi-op tile scheduler + double buffering + DMA/compute overlap"
//   를 한 파일로 확장
//
// 설계 철학:
// 1) Graph는 runtime이 해석하지 않는다.
//    -> compiler/middle-layer가 "Tile Contract Graph"를 만든다.
// 2) runtime은 contract를 검증하고, descriptor queue에 낮추고, 실행한다.
// 3) overlap은 명시적이다.
//    -> DMA-load(next tile) / Compute(curr tile) / DMA-store(prev tile)
// 4) double buffering은 compiler contract로 고정된다.
//    -> SRAM bank A/B, phase parity 로 선택
// 5) barrier graph는 data dependency를 보장한다.
// 6) 지금 구현은 simulator/executor 이지만,
//    exec_dma() / exec_conv()만 실제 driver ABI로 교체하면 된다.
//
// 핵심 개념:
// - GraphOpDesc         : graph-level op metadata
// - TileTask            : op 하나의 tile instance
// - TileContract        : compiler가 생성한 tile contract
// - TileScheduler       : graph-level tile ordering
// - OverlapLowerer      : double-buffer + DMA/compute overlap descriptor lowering
// - RawNPURuntime       : queue executor / barrier / SRAM lifetime simulation
//
// 지원 예시:
//   Conv0 -> Conv1
//   tile graph 생성
//   Conv0 output tile이 Conv1 input tile로 이어짐
//   tile-wise dependency / stage pipelining / overlap scheduling
//
// 제약:
// - 실제 Intel NPU ISA / firmware ABI는 비공개라고 가정
// - 본 코드는 그 바로 위 level의 "compiler contract baseline"
// ============================================================================

// ============================================================================
// 1. Basic types / helpers
// ============================================================================
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;

static inline u64 align_up(u64 x, u64 a) {
    return ((x + a - 1) / a) * a;
}

static inline std::string vec_to_string(const std::vector<i32>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        oss << v[i];
        if (i + 1 != v.size()) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

template <typename T>
static inline std::string list_to_string(const std::vector<T>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        oss << v[i];
        if (i + 1 != v.size()) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

// ============================================================================
// 2. Hardware capabilities
// ----------------------------------------------------------------------------
// compiler가 반드시 알아야 하는 contract surface
// ============================================================================
struct HardwareCaps {
    u64 local_sram_bytes = 4 * 1024 * 1024;   // total local SRAM
    u64 dma_alignment = 64;
    i32 queue_depth = 8192;

    // tile shape caps
    i32 max_tile_h = 32;
    i32 max_tile_w = 32;
    i32 max_tile_c = 128;

    // execution hints
    i32 mac_vector_width = 32;
    i32 max_dma_inflight = 4;
    bool supports_dma_compute_overlap = true;
    bool supports_double_buffering = true;

    // split SRAM into two banks for ping-pong
    u64 sram_bank_count = 2;
};

// ============================================================================
// 3. Tensor / shape / memory descriptors
// ============================================================================
enum class DataType {
    FP32,
    FP16,
    BF16,
    INT8
};

static inline u32 dtype_size(DataType dt) {
    switch (dt) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::BF16: return 2;
        case DataType::INT8: return 1;
    }
    throw std::runtime_error("Unknown dtype");
}

enum class MemorySpace {
    HOST,
    DDR,
    LOCAL_SRAM
};

struct TensorDesc {
    u32 id = 0;
    std::string name;
    DataType dtype {};
    std::vector<i32> shape;   // NCHW
    std::vector<i32> strides; // element strides
    MemorySpace mem_space {};
    u64 base_addr = 0;
    u64 byte_size = 0;

    std::string str() const {
        std::ostringstream oss;
        oss << "TensorDesc{id=" << id
            << ", name=" << name
            << ", shape=" << vec_to_string(shape)
            << ", strides=" << vec_to_string(strides)
            << ", base=0x" << std::hex << base_addr << std::dec
            << ", bytes=" << byte_size << "}";
        return oss.str();
    }
};

struct TensorTable {
    std::map<u32, TensorDesc> table;

    const TensorDesc& get(u32 id) const {
        auto it = table.find(id);
        if (it == table.end()) {
            throw std::runtime_error("Unknown tensor id: " + std::to_string(id));
        }
        return it->second;
    }
};

// ============================================================================
// 4. Tile regions
// ============================================================================
struct TileRegion {
    i32 n0 = 0, n1 = 0;
    i32 c0 = 0, c1 = 0;
    i32 h0 = 0, h1 = 0;
    i32 w0 = 0, w1 = 0;

    i32 n_extent() const { return n1 - n0; }
    i32 c_extent() const { return c1 - c0; }
    i32 h_extent() const { return h1 - h0; }
    i32 w_extent() const { return w1 - w0; }

    std::string str() const {
        std::ostringstream oss;
        oss << "{n:[" << n0 << "," << n1
            << "), c:[" << c0 << "," << c1
            << "), h:[" << h0 << "," << h1
            << "), w:[" << w0 << "," << w1 << ")}";
        return oss.str();
    }
};

// ============================================================================
// 5. Graph op descriptors
// ----------------------------------------------------------------------------
// middle layer / compiler가 graph-level lowering 이전에 다루는 op
// ============================================================================
enum class OpKind {
    CONV2D,
    RELU
};

struct Conv2DParams {
    i32 stride_h = 1;
    i32 stride_w = 1;
    i32 pad_h = 0;
    i32 pad_w = 0;
    i32 kernel_h = 3;
    i32 kernel_w = 3;
    i32 dilation_h = 1;
    i32 dilation_w = 1;
    i32 groups = 1;
};

struct GraphOpDesc {
    u32 op_id = 0;
    OpKind kind {};
    std::string name;

    u32 input_tensor_id = 0;
    u32 weight_tensor_id = 0;
    u32 bias_tensor_id = 0;
    u32 output_tensor_id = 0;

    Conv2DParams conv {};

    std::vector<u32> producer_ops;
    std::vector<u32> consumer_ops;
};

struct GraphDesc {
    std::vector<GraphOpDesc> ops;

    const GraphOpDesc& get_op(u32 id) const {
        for (const auto& op : ops) {
            if (op.op_id == id) return op;
        }
        throw std::runtime_error("Unknown op id: " + std::to_string(id));
    }
};

// ============================================================================
// 6. Local SRAM allocation
// ----------------------------------------------------------------------------
// double buffering을 위해 bank-aware allocation을 넣는다.
// bank 0 / bank 1 중 하나에 tile working set을 배치
// ============================================================================
struct LocalAlloc {
    std::string name;
    u64 bank_id = 0;
    u64 offset = 0;   // bank-local offset
    u64 bytes = 0;
    u64 alignment = 64;

    u64 end() const { return offset + bytes; }
};

struct TileMemoryPlan {
    u64 bank_id = 0;

    LocalAlloc input_tile;
    LocalAlloc weight_tile;
    LocalAlloc psum_tile;
    LocalAlloc output_tile;

    u64 total_bytes = 0;

    std::string str() const {
        std::ostringstream oss;
        oss << "TileMemoryPlan{bank=" << bank_id
            << ", input=(" << input_tile.offset << "," << input_tile.bytes << ")"
            << ", weight=(" << weight_tile.offset << "," << weight_tile.bytes << ")"
            << ", psum=(" << psum_tile.offset << "," << psum_tile.bytes << ")"
            << ", output=(" << output_tile.offset << "," << output_tile.bytes << ")"
            << ", total=" << total_bytes
            << "}";
        return oss.str();
    }
};

// ============================================================================
// 7. Tile task / compiler contract
// ----------------------------------------------------------------------------
// graph-level tile task:
// - 어느 op의 몇 번째 tile인지
// - dependency / barrier / memory plan / bank selection까지 포함
// ============================================================================
struct TileTask {
    u32 global_tile_id = 0;
    u32 op_id = 0;
    i32 stage_id = 0;
    i32 seq_no = 0;
    i32 pipeline_phase = 0; // used for overlap / ping-pong
};

struct ConvTileContract {
    TileTask task;

    u32 input_tensor_id = 0;
    u32 weight_tensor_id = 0;
    u32 bias_tensor_id = 0;
    u32 output_tensor_id = 0;

    Conv2DParams params;

    TileRegion input_region;
    TileRegion weight_region;
    TileRegion output_region;

    TileMemoryPlan mem_plan;

    std::vector<u32> wait_barriers;
    std::vector<u32> signal_barriers;

    std::vector<u32> producer_tile_ids;
    std::vector<u32> consumer_tile_ids;

    std::string str() const {
        std::ostringstream oss;
        oss << "ConvTileContract{tile=" << task.global_tile_id
            << ", op=" << task.op_id
            << ", seq=" << task.seq_no
            << ", phase=" << task.pipeline_phase
            << ", in=" << input_region.str()
            << ", out=" << output_region.str()
            << ", bank=" << mem_plan.bank_id
            << ", waits=" << list_to_string(wait_barriers)
            << ", signals=" << list_to_string(signal_barriers)
            << "}";
        return oss.str();
    }
};

// ============================================================================
// 8. Runtime descriptors
// ----------------------------------------------------------------------------
// compiler contract -> raw runtime command stream
// ============================================================================
enum class DescriptorKind {
    DMA_LOAD,
    DMA_STORE,
    COMPUTE_CONV,
    BARRIER_WAIT,
    BARRIER_SIGNAL
};

struct DMADescriptor {
    u32 desc_id = 0;
    u64 src_addr = 0;
    u64 dst_addr = 0;
    u64 bytes = 0;
    u64 alignment = 64;
    bool to_local = true;
    u64 bank_id = 0;
    u32 tile_id = 0;
    std::string tag;
};

struct ComputeConvDescriptor {
    u32 desc_id = 0;
    u32 op_id = 0;
    u32 tile_id = 0;

    u64 bank_id = 0;
    u64 input_offset = 0;
    u64 weight_offset = 0;
    u64 psum_offset = 0;
    u64 output_offset = 0;

    i32 n = 1;
    i32 ic = 0;
    i32 oc = 0;
    i32 oh = 0;
    i32 ow = 0;
    i32 kh = 0;
    i32 kw = 0;

    Conv2DParams params {};
    std::string tag;
};

struct BarrierWaitDescriptor {
    u32 desc_id = 0;
    u32 barrier_id = 0;
    u32 tile_id = 0;
};

struct BarrierSignalDescriptor {
    u32 desc_id = 0;
    u32 barrier_id = 0;
    u32 tile_id = 0;
};

struct RuntimeDescriptor {
    DescriptorKind kind {};
    std::optional<DMADescriptor> dma;
    std::optional<ComputeConvDescriptor> conv;
    std::optional<BarrierWaitDescriptor> wait;
    std::optional<BarrierSignalDescriptor> signal;
};

// ============================================================================
// 9. Runtime queue / state
// ============================================================================
struct BarrierState {
    bool signaled = false;
};

struct LocalSRAMBank {
    std::vector<u8> bytes;
    explicit LocalSRAMBank(u64 sz) : bytes(sz, 0) {}
};

struct RuntimeStats {
    u64 dma_load_bytes = 0;
    u64 dma_store_bytes = 0;
    u64 compute_tiles = 0;
    u64 barrier_waits = 0;
    u64 barrier_signals = 0;

    u64 overlapped_triplets = 0; // load(next)+compute(curr)+store(prev)
    u64 scheduled_tiles = 0;
};

class DescriptorQueue {
public:
    explicit DescriptorQueue(size_t cap) : capacity_(cap) {}

    void push(const RuntimeDescriptor& d) {
        if (q_.size() >= capacity_) {
            throw std::runtime_error("DescriptorQueue overflow");
        }
        q_.push_back(d);
    }

    bool empty() const { return q_.empty(); }
    size_t size() const { return q_.size(); }

    RuntimeDescriptor pop() {
        if (q_.empty()) {
            throw std::runtime_error("DescriptorQueue underflow");
        }
        RuntimeDescriptor d = q_.front();
        q_.pop_front();
        return d;
    }

private:
    std::deque<RuntimeDescriptor> q_;
    size_t capacity_;
};

// ============================================================================
// 10. Contract verifier
// ============================================================================
class ContractVerifier {
public:
    static void verify_sram_plan(const TileMemoryPlan& p, const HardwareCaps& hw) {
        const u64 bank_size = hw.local_sram_bytes / hw.sram_bank_count;
        std::array<LocalAlloc, 4> allocs = {p.input_tile, p.weight_tile, p.psum_tile, p.output_tile};

        for (const auto& a : allocs) {
            if (a.offset % a.alignment != 0) {
                throw std::runtime_error("SRAM alloc alignment violation: " + a.name);
            }
            if (a.end() > bank_size) {
                throw std::runtime_error("SRAM alloc out-of-bounds in bank: " + a.name);
            }
            if (a.bank_id != p.bank_id) {
                throw std::runtime_error("SRAM alloc bank mismatch: " + a.name);
            }
        }

        for (size_t i = 0; i < allocs.size(); ++i) {
            for (size_t j = i + 1; j < allocs.size(); ++j) {
                bool overlap = !(allocs[i].end() <= allocs[j].offset || allocs[j].end() <= allocs[i].offset);
                if (overlap) {
                    throw std::runtime_error("SRAM alloc overlap: " + allocs[i].name + " <-> " + allocs[j].name);
                }
            }
        }

        if (p.total_bytes > bank_size) {
            throw std::runtime_error("TileMemoryPlan exceeds bank SRAM cap");
        }
    }

    static void verify_conv_tile_shape(const ConvTileContract& c, const HardwareCaps& hw) {
        const i32 oh = c.output_region.h_extent();
        const i32 ow = c.output_region.w_extent();
        const i32 oc = c.output_region.c_extent();

        if (oh <= 0 || ow <= 0 || oc <= 0) {
            throw std::runtime_error("Invalid output tile extent");
        }
        if (oh > hw.max_tile_h || ow > hw.max_tile_w || oc > hw.max_tile_c) {
            throw std::runtime_error("Output tile exceeds hardware tile cap");
        }
    }

    static void verify_barrier_lists(const ConvTileContract& c) {
        std::set<u32> s;
        for (u32 b : c.wait_barriers) {
            if (!s.insert(b).second) {
                throw std::runtime_error("Duplicate barrier in wait list");
            }
        }
        s.clear();
        for (u32 b : c.signal_barriers) {
            if (!s.insert(b).second) {
                throw std::runtime_error("Duplicate barrier in signal list");
            }
        }
    }

    static void verify_conv_contract(const ConvTileContract& c, const HardwareCaps& hw) {
        verify_conv_tile_shape(c, hw);
        verify_sram_plan(c.mem_plan, hw);
        verify_barrier_lists(c);
    }
};

// ============================================================================
// 11. Tile memory planner
// ----------------------------------------------------------------------------
// phase parity에 따라 bank 0/1 할당
// ============================================================================
class TileMemoryPlanner {
public:
    static TileMemoryPlan make_plan(const ConvTileContract& proto, const HardwareCaps& hw, u64 bank_id) {
        TileMemoryPlan p;
        p.bank_id = bank_id;

        const u64 input_elems =
            static_cast<u64>(proto.input_region.n_extent()) *
            static_cast<u64>(proto.input_region.c_extent()) *
            static_cast<u64>(proto.input_region.h_extent()) *
            static_cast<u64>(proto.input_region.w_extent());

        const u64 weight_elems =
            static_cast<u64>(proto.output_region.c_extent()) *
            static_cast<u64>(proto.input_region.c_extent()) *
            static_cast<u64>(proto.params.kernel_h) *
            static_cast<u64>(proto.params.kernel_w);

        const u64 output_elems =
            static_cast<u64>(proto.output_region.n_extent()) *
            static_cast<u64>(proto.output_region.c_extent()) *
            static_cast<u64>(proto.output_region.h_extent()) *
            static_cast<u64>(proto.output_region.w_extent());

        const u64 act_bytes = 2;  // FP16 baseline
        const u64 acc_bytes = 4;  // FP32 accumulate

        u64 off = 0;

        p.input_tile  = {"input_tile",  bank_id, align_up(off, 64), input_elems * act_bytes, 64};
        off = p.input_tile.end();

        p.weight_tile = {"weight_tile", bank_id, align_up(off, 64), weight_elems * act_bytes, 64};
        off = p.weight_tile.end();

        p.psum_tile   = {"psum_tile",   bank_id, align_up(off, 64), output_elems * acc_bytes, 64};
        off = p.psum_tile.end();

        p.output_tile = {"output_tile", bank_id, align_up(off, 64), output_elems * act_bytes, 64};
        off = p.output_tile.end();

        p.total_bytes = off;

        ContractVerifier::verify_sram_plan(p, hw);
        return p;
    }
};

// ============================================================================
// 12. Graph-level tile scheduler
// ----------------------------------------------------------------------------
// 예시: Conv0 -> Conv1
//
// 전략:
// - 각 op를 output tile 기준으로 분할
// - tile dependency graph 생성
// - downstream op tile은 upstream output tile barrier를 wait
// - phase parity = seq_no % 2 로 bank A/B 선택
// - 단순 wavefront-style ordering
// ============================================================================
struct ConvProblem {
    i32 N = 1;
    i32 IC = 0;
    i32 IH = 0;
    i32 IW = 0;
    i32 OC = 0;
    i32 OH = 0;
    i32 OW = 0;
    Conv2DParams p {};
};

class GraphTileScheduler {
public:
    static std::vector<ConvTileContract> schedule_two_conv_chain(
        const GraphDesc& graph,
        const TensorTable& tensors,
        const HardwareCaps& hw) {

        // graph[0]=conv0, graph[1]=conv1 가정
        if (graph.ops.size() != 2) {
            throw std::runtime_error("This baseline demo expects exactly 2 conv ops");
        }
        const auto& op0 = graph.ops[0];
        const auto& op1 = graph.ops[1];
        if (op0.kind != OpKind::CONV2D || op1.kind != OpKind::CONV2D) {
            throw std::runtime_error("Only Conv2D ops are supported in this baseline");
        }

        auto prob0 = make_problem(op0, tensors);
        auto prob1 = make_problem(op1, tensors);

        std::vector<ConvTileContract> conv0_tiles = schedule_conv_tiles(op0, prob0, hw, /*stage=*/0, /*tile_id_base=*/1);
        std::vector<ConvTileContract> conv1_tiles = schedule_conv_tiles(op1, prob1, hw, /*stage=*/1, /*tile_id_base=*/static_cast<u32>(conv0_tiles.size() + 1));

        // tile dependency:
        // conv1 tile waits on conv0 tile(s) that produce required input channel/spatial region
        build_simple_chain_dependencies(conv0_tiles, conv1_tiles);

        // concatenate in topological stage order
        std::vector<ConvTileContract> all;
        all.reserve(conv0_tiles.size() + conv1_tiles.size());
        all.insert(all.end(), conv0_tiles.begin(), conv0_tiles.end());
        all.insert(all.end(), conv1_tiles.begin(), conv1_tiles.end());
        return all;
    }

private:
    static ConvProblem make_problem(const GraphOpDesc& op, const TensorTable& tensors) {
        const auto& in  = tensors.get(op.input_tensor_id);
        const auto& out = tensors.get(op.output_tensor_id);

        if (in.shape.size() != 4 || out.shape.size() != 4) {
            throw std::runtime_error("Expected 4D NCHW tensors");
        }

        ConvProblem p;
        p.N  = in.shape[0];
        p.IC = in.shape[1];
        p.IH = in.shape[2];
        p.IW = in.shape[3];
        p.OC = out.shape[1];
        p.OH = out.shape[2];
        p.OW = out.shape[3];
        p.p  = op.conv;
        return p;
    }

    static std::vector<ConvTileContract> schedule_conv_tiles(
        const GraphOpDesc& op,
        const ConvProblem& prob,
        const HardwareCaps& hw,
        i32 stage_id,
        u32 tile_id_base) {

        std::vector<ConvTileContract> tiles;
        i32 tile_h = std::min(hw.max_tile_h, prob.OH);
        i32 tile_w = std::min(hw.max_tile_w, prob.OW);
        i32 tile_oc = std::min(hw.max_tile_c, prob.OC);

        u32 barrier_counter = tile_id_base * 1000; // deterministic namespace
        i32 seq = 0;
        u32 next_tile_id = tile_id_base;

        for (i32 oc0 = 0; oc0 < prob.OC; oc0 += tile_oc) {
            i32 oc1 = std::min(oc0 + tile_oc, prob.OC);

            for (i32 oh0 = 0; oh0 < prob.OH; oh0 += tile_h) {
                i32 oh1 = std::min(oh0 + tile_h, prob.OH);

                for (i32 ow0 = 0; ow0 < prob.OW; ow0 += tile_w) {
                    i32 ow1 = std::min(ow0 + tile_w, prob.OW);

                    ConvTileContract c;
                    c.task.global_tile_id = next_tile_id++;
                    c.task.op_id = op.op_id;
                    c.task.stage_id = stage_id;
                    c.task.seq_no = seq;
                    c.task.pipeline_phase = seq % 2;

                    c.input_tensor_id = op.input_tensor_id;
                    c.weight_tensor_id = op.weight_tensor_id;
                    c.bias_tensor_id = op.bias_tensor_id;
                    c.output_tensor_id = op.output_tensor_id;
                    c.params = op.conv;

                    c.output_region = TileRegion{
                        0, prob.N,
                        oc0, oc1,
                        oh0, oh1,
                        ow0, ow1
                    };

                    const i32 in_h0 = std::max(0, oh0 * prob.p.stride_h - prob.p.pad_h);
                    const i32 in_w0 = std::max(0, ow0 * prob.p.stride_w - prob.p.pad_w);
                    const i32 in_h1 = std::min(prob.IH,
                        (oh1 - 1) * prob.p.stride_h - prob.p.pad_h + prob.p.kernel_h);
                    const i32 in_w1 = std::min(prob.IW,
                        (ow1 - 1) * prob.p.stride_w - prob.p.pad_w + prob.p.kernel_w);

                    c.input_region = TileRegion{
                        0, prob.N,
                        0, prob.IC,
                        in_h0, in_h1,
                        in_w0, in_w1
                    };

                    c.weight_region = TileRegion{
                        0, 1,
                        oc0, oc1,
                        0, prob.IC,
                        0, prob.p.kernel_h * prob.p.kernel_w
                    };

                    // double buffering bank selection by phase parity
                    c.mem_plan = TileMemoryPlanner::make_plan(c, hw, static_cast<u64>(c.task.pipeline_phase % static_cast<i32>(hw.sram_bank_count)));

                    // local completion barrier for this tile
                    c.signal_barriers.push_back(barrier_counter++);

                    ContractVerifier::verify_conv_contract(c, hw);
                    tiles.push_back(c);
                    seq++;
                }
            }
        }

        return tiles;
    }

    static bool overlaps_spatial(const TileRegion& a, const TileRegion& b) {
        bool h_overlap = !(a.h1 <= b.h0 || b.h1 <= a.h0);
        bool w_overlap = !(a.w1 <= b.w0 || b.w1 <= a.w0);
        return h_overlap && w_overlap;
    }

    static void build_simple_chain_dependencies(
        std::vector<ConvTileContract>& prod,
        std::vector<ConvTileContract>& cons) {

        // conv1 input은 conv0 output.
        // spatial overlap + channel compatibility 기반으로 barrier 연결.
        // baseline이므로 conservative하게 연결한다.
        for (auto& c1 : cons) {
            for (auto& c0 : prod) {
                const bool spatial_ok = overlaps_spatial(c0.output_region, c1.input_region);

                // c1 input channel == c0 output channel space라고 가정.
                const bool channel_ok =
                    !(c0.output_region.c1 <= c1.input_region.c0 ||
                      c1.input_region.c1 <= c0.output_region.c0);

                if (spatial_ok && channel_ok) {
                    if (!c0.signal_barriers.empty()) {
                        c1.wait_barriers.push_back(c0.signal_barriers.front());
                    }
                    c1.producer_tile_ids.push_back(c0.task.global_tile_id);
                    c0.consumer_tile_ids.push_back(c1.task.global_tile_id);
                }
            }

            // dedup wait list
            std::sort(c1.wait_barriers.begin(), c1.wait_barriers.end());
            c1.wait_barriers.erase(std::unique(c1.wait_barriers.begin(), c1.wait_barriers.end()), c1.wait_barriers.end());
            ContractVerifier::verify_barrier_lists(c1);
        }
    }
};

// ============================================================================
// 13. Overlap-aware lowering
// ----------------------------------------------------------------------------
// 목표:
// - double buffering + DMA/compute overlap 을 descriptor stream으로 내리기
//
// tile stream T0, T1, T2 ... 에 대해
//   phase 0: load(T0)
//   phase 1: load(T1) + compute(T0)
//   phase 2: load(T2) + compute(T1) + store(T0)
//   ...
//
// 주의:
// - 실제 하드웨어라면 이 세 엔진이 독립 queue일 수 있다.
// - 여기서는 descriptor sequence에 annotation을 유지하고, runtime stats로 overlap intent를 남긴다.
// ============================================================================
class OverlapLowerer {
public:
    static std::vector<RuntimeDescriptor> lower_stage_stream(
        const std::vector<ConvTileContract>& tiles,
        const TensorTable& tensors) {

        std::vector<RuntimeDescriptor> out;
        u32 next_desc = 1;

        for (size_t i = 0; i < tiles.size(); ++i) {
            const ConvTileContract* prev = (i >= 1) ? &tiles[i - 1] : nullptr;
            const ConvTileContract* curr = &tiles[i];
            const ConvTileContract* next = (i + 1 < tiles.size()) ? &tiles[i + 1] : nullptr;

            // wait for current tile dependencies
            for (u32 b : curr->wait_barriers) {
                RuntimeDescriptor d;
                d.kind = DescriptorKind::BARRIER_WAIT;
                d.wait = BarrierWaitDescriptor{next_desc++, b, curr->task.global_tile_id};
                out.push_back(d);
            }

            // phase A: prefetch current if first tile
            if (i == 0) {
                emit_load_pair(out, next_desc, *curr, tensors);
            }

            // phase B: prefetch next while computing current
            if (next) {
                emit_load_pair(out, next_desc, *next, tensors);
            }

            // compute current
            emit_compute(out, next_desc, *curr);

            // phase C: store previous while current computes / next loads
            if (prev) {
                emit_store(out, next_desc, *prev, tensors);
            }

            // signal current completion
            for (u32 b : curr->signal_barriers) {
                RuntimeDescriptor d;
                d.kind = DescriptorKind::BARRIER_SIGNAL;
                d.signal = BarrierSignalDescriptor{next_desc++, b, curr->task.global_tile_id};
                out.push_back(d);
            }
        }

        // drain final store
        if (!tiles.empty()) {
            const auto& last = tiles.back();
            emit_store(out, next_desc, last, tensors);
        }

        return out;
    }

private:
    static void emit_load_pair(
        std::vector<RuntimeDescriptor>& out,
        u32& next_desc,
        const ConvTileContract& c,
        const TensorTable& tensors) {

        const auto& input = tensors.get(c.input_tensor_id);
        const auto& weight = tensors.get(c.weight_tensor_id);

        RuntimeDescriptor di;
        di.kind = DescriptorKind::DMA_LOAD;
        di.dma = DMADescriptor{
            next_desc++,
            input.base_addr, // baseline: concrete tile offset calc omitted
            c.mem_plan.input_tile.offset,
            c.mem_plan.input_tile.bytes,
            c.mem_plan.input_tile.alignment,
            true,
            c.mem_plan.bank_id,
            c.task.global_tile_id,
            "prefetch_input_tile"
        };
        out.push_back(di);

        RuntimeDescriptor dw;
        dw.kind = DescriptorKind::DMA_LOAD;
        dw.dma = DMADescriptor{
            next_desc++,
            weight.base_addr,
            c.mem_plan.weight_tile.offset,
            c.mem_plan.weight_tile.bytes,
            c.mem_plan.weight_tile.alignment,
            true,
            c.mem_plan.bank_id,
            c.task.global_tile_id,
            "prefetch_weight_tile"
        };
        out.push_back(dw);
    }

    static void emit_compute(
        std::vector<RuntimeDescriptor>& out,
        u32& next_desc,
        const ConvTileContract& c) {

        RuntimeDescriptor dc;
        dc.kind = DescriptorKind::COMPUTE_CONV;
        dc.conv = ComputeConvDescriptor{
            next_desc++,
            c.task.op_id,
            c.task.global_tile_id,
            c.mem_plan.bank_id,
            c.mem_plan.input_tile.offset,
            c.mem_plan.weight_tile.offset,
            c.mem_plan.psum_tile.offset,
            c.mem_plan.output_tile.offset,
            c.output_region.n_extent(),
            c.input_region.c_extent(),
            c.output_region.c_extent(),
            c.output_region.h_extent(),
            c.output_region.w_extent(),
            c.params.kernel_h,
            c.params.kernel_w,
            c.params,
            "compute_conv_tile"
        };
        out.push_back(dc);
    }

    static void emit_store(
        std::vector<RuntimeDescriptor>& out,
        u32& next_desc,
        const ConvTileContract& c,
        const TensorTable& tensors) {

        const auto& output = tensors.get(c.output_tensor_id);

        RuntimeDescriptor ds;
        ds.kind = DescriptorKind::DMA_STORE;
        ds.dma = DMADescriptor{
            next_desc++,
            c.mem_plan.output_tile.offset,
            output.base_addr,
            c.mem_plan.output_tile.bytes,
            c.mem_plan.output_tile.alignment,
            false,
            c.mem_plan.bank_id,
            c.task.global_tile_id,
            "store_output_tile"
        };
        out.push_back(ds);
    }
};

// ============================================================================
// 14. Raw runtime executor
// ----------------------------------------------------------------------------
// simulator-level executor
// - bank-aware SRAM
// - barrier validation
// - overlap intent accounting
// ============================================================================
class RawNPURuntime {
public:
    explicit RawNPURuntime(HardwareCaps hw)
        : hw_(hw), queue_(hw.queue_depth) {

        const u64 bank_size = hw.local_sram_bytes / hw.sram_bank_count;
        for (u64 i = 0; i < hw.sram_bank_count; ++i) {
            banks_.emplace_back(bank_size);
        }
    }

    void install_barrier(u32 id) {
        barriers_[id] = BarrierState{};
    }

    void submit(const RuntimeDescriptor& d) {
        queue_.push(d);
    }

    void run() {
        // overlap intent counting:
        // consecutive pattern of [DMA_LOAD, COMPUTE_CONV, DMA_STORE]
        // 또는 load pair / compute / store 가 반복되는 것을 추적
        DescriptorKind prev2 = DescriptorKind::BARRIER_SIGNAL;
        DescriptorKind prev1 = DescriptorKind::BARRIER_SIGNAL;

        while (!queue_.empty()) {
            RuntimeDescriptor d = queue_.pop();

            if (prev2 == DescriptorKind::DMA_LOAD &&
                prev1 == DescriptorKind::COMPUTE_CONV &&
                d.kind == DescriptorKind::DMA_STORE) {
                stats_.overlapped_triplets++;
            }

            switch (d.kind) {
                case DescriptorKind::BARRIER_WAIT:
                    exec_wait(*d.wait);
                    break;
                case DescriptorKind::DMA_LOAD:
                    exec_dma(*d.dma);
                    break;
                case DescriptorKind::COMPUTE_CONV:
                    exec_conv(*d.conv);
                    break;
                case DescriptorKind::DMA_STORE:
                    exec_dma(*d.dma);
                    break;
                case DescriptorKind::BARRIER_SIGNAL:
                    exec_signal(*d.signal);
                    break;
            }

            prev2 = prev1;
            prev1 = d.kind;
        }
    }

    const RuntimeStats& stats() const { return stats_; }

private:
    void exec_wait(const BarrierWaitDescriptor& d) {
        auto it = barriers_.find(d.barrier_id);
        if (it == barriers_.end()) {
            throw std::runtime_error("Unknown barrier in wait");
        }
        if (!it->second.signaled) {
            throw std::runtime_error(
                "Barrier wait before signal: barrier=" + std::to_string(d.barrier_id) +
                ", tile=" + std::to_string(d.tile_id));
        }
        stats_.barrier_waits++;
    }

    void exec_signal(const BarrierSignalDescriptor& d) {
        auto it = barriers_.find(d.barrier_id);
        if (it == barriers_.end()) {
            throw std::runtime_error("Unknown barrier in signal");
        }
        it->second.signaled = true;
        stats_.barrier_signals++;
    }

    void exec_dma(const DMADescriptor& d) {
        if (d.bytes == 0) {
            throw std::runtime_error("DMA bytes == 0");
        }
        if (d.bank_id >= banks_.size()) {
            throw std::runtime_error("DMA bank_id out of range");
        }

        auto& bank = banks_[d.bank_id];

        if (d.to_local) {
            if (d.dst_addr + d.bytes > bank.bytes.size()) {
                throw std::runtime_error("DMA load out-of-bounds to SRAM bank");
            }
            stats_.dma_load_bytes += d.bytes;
        } else {
            if (d.src_addr + d.bytes > bank.bytes.size()) {
                throw std::runtime_error("DMA store out-of-bounds from SRAM bank");
            }
            stats_.dma_store_bytes += d.bytes;
        }
    }

    void exec_conv(const ComputeConvDescriptor& d) {
        if (d.bank_id >= banks_.size()) {
            throw std::runtime_error("Conv bank_id out of range");
        }
        auto& bank = banks_[d.bank_id];

        if (d.input_offset >= bank.bytes.size()) {
            throw std::runtime_error("conv input_offset OOB");
        }
        if (d.weight_offset >= bank.bytes.size()) {
            throw std::runtime_error("conv weight_offset OOB");
        }
        if (d.output_offset >= bank.bytes.size()) {
            throw std::runtime_error("conv output_offset OOB");
        }
        if (d.oh <= 0 || d.ow <= 0 || d.oc <= 0) {
            throw std::runtime_error("invalid conv tile shape");
        }

        stats_.compute_tiles++;
        stats_.scheduled_tiles++;
    }

private:
    HardwareCaps hw_;
    std::vector<LocalSRAMBank> banks_;
    DescriptorQueue queue_;
    std::map<u32, BarrierState> barriers_;
    RuntimeStats stats_;
};

// ============================================================================
// 15. Example graph construction
// ----------------------------------------------------------------------------
// Conv0: 1x64x56x56 -> 1x128x56x56
// Conv1: 1x128x56x56 -> 1x128x56x56
// ============================================================================
static TensorTable make_demo_tensors() {
    TensorTable tt;

    tt.table[1] = TensorDesc{
        1, "input0", DataType::FP16,
        {1, 64, 56, 56},
        {64 * 56 * 56, 56 * 56, 56, 1},
        MemorySpace::DDR,
        0x10000000,
        static_cast<u64>(1) * 64 * 56 * 56 * 2
    };

    tt.table[2] = TensorDesc{
        2, "weight0", DataType::FP16,
        {128, 64, 3, 3},
        {64 * 3 * 3, 3 * 3, 3, 1},
        MemorySpace::DDR,
        0x20000000,
        static_cast<u64>(128) * 64 * 3 * 3 * 2
    };

    tt.table[3] = TensorDesc{
        3, "mid0", DataType::FP16,
        {1, 128, 56, 56},
        {128 * 56 * 56, 56 * 56, 56, 1},
        MemorySpace::DDR,
        0x30000000,
        static_cast<u64>(1) * 128 * 56 * 56 * 2
    };

    tt.table[4] = TensorDesc{
        4, "weight1", DataType::FP16,
        {128, 128, 3, 3},
        {128 * 3 * 3, 3 * 3, 3, 1},
        MemorySpace::DDR,
        0x40000000,
        static_cast<u64>(128) * 128 * 3 * 3 * 2
    };

    tt.table[5] = TensorDesc{
        5, "output1", DataType::FP16,
        {1, 128, 56, 56},
        {128 * 56 * 56, 56 * 56, 56, 1},
        MemorySpace::DDR,
        0x50000000,
        static_cast<u64>(1) * 128 * 56 * 56 * 2
    };

    return tt;
}

static GraphDesc make_demo_graph() {
    GraphDesc g;

    GraphOpDesc conv0;
    conv0.op_id = 10;
    conv0.kind = OpKind::CONV2D;
    conv0.name = "conv0";
    conv0.input_tensor_id = 1;
    conv0.weight_tensor_id = 2;
    conv0.output_tensor_id = 3;
    conv0.conv = Conv2DParams{1,1,1,1,3,3,1,1,1};
    conv0.consumer_ops = {11};

    GraphOpDesc conv1;
    conv1.op_id = 11;
    conv1.kind = OpKind::CONV2D;
    conv1.name = "conv1";
    conv1.input_tensor_id = 3;
    conv1.weight_tensor_id = 4;
    conv1.output_tensor_id = 5;
    conv1.conv = Conv2DParams{1,1,1,1,3,3,1,1,1};
    conv1.producer_ops = {10};

    g.ops.push_back(conv0);
    g.ops.push_back(conv1);
    return g;
}

// ============================================================================
// 16. Utility: barrier installation
// ============================================================================
static void install_all_barriers(RawNPURuntime& rt, const std::vector<ConvTileContract>& tiles) {
    std::set<u32> ids;
    for (const auto& t : tiles) {
        for (u32 b : t.wait_barriers) ids.insert(b);
        for (u32 b : t.signal_barriers) ids.insert(b);
    }
    for (u32 b : ids) {
        rt.install_barrier(b);
    }
}

// ============================================================================
// 17. Group tiles by stage
// ============================================================================
static std::map<i32, std::vector<ConvTileContract>> group_by_stage(const std::vector<ConvTileContract>& all) {
    std::map<i32, std::vector<ConvTileContract>> grouped;
    for (const auto& t : all) {
        grouped[t.task.stage_id].push_back(t);
    }
    for (auto& [stage, v] : grouped) {
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
            return a.task.seq_no < b.task.seq_no;
        });
    }
    return grouped;
}

// ============================================================================
// 18. Demo main
// ============================================================================
int main() {
    try {
        HardwareCaps hw;
        hw.local_sram_bytes = 4 * 1024 * 1024;
        hw.sram_bank_count = 2;
        hw.max_tile_h = 16;
        hw.max_tile_w = 16;
        hw.max_tile_c = 64;
        hw.queue_depth = 16384;
        hw.supports_double_buffering = true;
        hw.supports_dma_compute_overlap = true;

        TensorTable tensors = make_demo_tensors();
        GraphDesc graph = make_demo_graph();

        std::cout << "=== Graph ===\n";
        for (const auto& op : graph.ops) {
            std::cout << "op_id=" << op.op_id
                      << " name=" << op.name
                      << " input=" << op.input_tensor_id
                      << " weight=" << op.weight_tensor_id
                      << " output=" << op.output_tensor_id
                      << "\n";
        }

        // 1) graph-level tile scheduling
        auto all_tiles = GraphTileScheduler::schedule_two_conv_chain(graph, tensors, hw);
        std::cout << "\nGenerated total tile contracts: " << all_tiles.size() << "\n";
        for (size_t i = 0; i < std::min<size_t>(all_tiles.size(), 6); ++i) {
            std::cout << "  " << all_tiles[i].str() << "\n";
        }

        // 2) group per stage and lower with overlap-aware lowering
        auto stages = group_by_stage(all_tiles);

        RawNPURuntime rt(hw);
        install_all_barriers(rt, all_tiles);

        std::vector<RuntimeDescriptor> full_stream;

        for (auto& [stage_id, stage_tiles] : stages) {
            auto descs = OverlapLowerer::lower_stage_stream(stage_tiles, tensors);
            full_stream.insert(full_stream.end(), descs.begin(), descs.end());
            std::cout << "\nStage " << stage_id
                      << " tiles=" << stage_tiles.size()
                      << " lowered_descs=" << descs.size() << "\n";
        }

        // 3) submit
        for (const auto& d : full_stream) {
            rt.submit(d);
        }

        // 4) execute
        rt.run();

        // 5) stats
        const auto& st = rt.stats();
        std::cout << "\n=== Runtime Stats ===\n";
        std::cout << "DMA load bytes    : " << st.dma_load_bytes << "\n";
        std::cout << "DMA store bytes   : " << st.dma_store_bytes << "\n";
        std::cout << "Compute tiles     : " << st.compute_tiles << "\n";
        std::cout << "Barrier waits     : " << st.barrier_waits << "\n";
        std::cout << "Barrier signals   : " << st.barrier_signals << "\n";
        std::cout << "Scheduled tiles   : " << st.scheduled_tiles << "\n";
        std::cout << "Overlap triplets  : " << st.overlapped_triplets << "\n";

        std::cout << "\n=== Summary ===\n";
        std::cout << "Graph-level multi-op tile scheduler + double buffering + DMA/compute overlap baseline executed.\n";
        std::cout << "Next step 1: replace exec_dma/exec_conv with real driver ABI.\n";
        std::cout << "Next step 2: add tile reuse across producer/consumer ops to eliminate DDR store/load round-trips.\n";
        std::cout << "Next step 3: promote per-stage lowering to graph-global overlap scheduler.\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}