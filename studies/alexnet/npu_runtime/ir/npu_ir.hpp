#pragma once

#include "../intel_npu_defs.hpp"

#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace intel_npu {

struct GraphDesc;
struct TensorTable;

enum class NpuOpKind {
    CONV,
    RELU,
    CLAMP,
    MAXPOOL,
    DROPOUT,
    IDENTITY,
    FALLBACK,
    LRN,
};

enum class UnsupportedPolicy {
    DECOMPOSE,
    FALLBACK,
    REJECT,
};

struct ConvAttrs {
    i32 stride_h = 1;
    i32 stride_w = 1;
    i32 pad_h = 0;
    i32 pad_w = 0;
    i32 kernel_h = 3;
    i32 kernel_w = 3;
};

struct ClampAttrs {
    float min_value = 0.0f;
    float max_value = 0.0f;
};

struct MaxPoolAttrs {
    i32 kernel_h = 2;
    i32 kernel_w = 2;
    i32 stride_h = 2;
    i32 stride_w = 2;
    i32 pad_h = 0;
    i32 pad_w = 0;
};

struct FallbackAttrs {
    std::string original_op_kind;
    std::string reason;
};

struct NpuOpAttrs {
    ConvAttrs conv;
    ClampAttrs clamp;
    MaxPoolAttrs maxpool;
    FallbackAttrs fallback;
};

struct NpuTensorIR {
    u32 id = 0;
    std::string name;
    DataType dtype = DataType::FP16;
    std::vector<i32> shape;
    MemorySpace mem_space = MemorySpace::DDR;
    u32 producer_op = 0;
    std::vector<u32> consumer_ops;
    bool is_graph_input = false;
    bool is_graph_output = false;
    bool removed = false;
};

struct NpuOpIR {
    u32 id = 0;
    NpuOpKind kind = NpuOpKind::IDENTITY;
    std::string name;
    std::vector<u32> inputs;
    std::vector<u32> outputs;
    PostOpKind post_op = PostOpKind::NONE;
    NpuOpAttrs attrs;
    i32 topo_index = -1;
    bool removed = false;
};

struct PassLog {
    std::vector<std::string> entries;

    void append(const std::string& pass_name, const std::string& message) {
        entries.push_back(pass_name + ": " + message);
    }

    std::string str() const {
        std::ostringstream oss;
        for (const auto& entry : entries) {
            oss << entry << '\n';
        }
        return oss.str();
    }
};

struct NpuGraphIR {
    std::vector<NpuOpIR> ops;
    std::vector<NpuTensorIR> tensors;
    std::vector<u32> input_tensor_ids;
    std::vector<u32> output_tensor_ids;

    NpuOpIR* find_op(u32 id);
    const NpuOpIR* find_op(u32 id) const;
    NpuTensorIR* find_tensor(u32 id);
    const NpuTensorIR* find_tensor(u32 id) const;

    void rebuild_edges();
    void recompute_topological_order();
    void replace_all_uses(u32 old_tensor_id, u32 new_tensor_id);
    void erase_removed();
};

const char* to_string(NpuOpKind kind);
const char* to_string(UnsupportedPolicy policy);

NpuGraphIR lower_graph_desc_to_npu_ir(const GraphDesc& graph, const TensorTable& tensors);
NpuGraphIR make_compiler_demo_ir();

std::string dump_graph_ir(const NpuGraphIR& graph);
std::string dump_graph_ir_compact(const NpuGraphIR& graph);

bool run_strip_dropout_pass(NpuGraphIR& graph, PassLog* log);
bool run_fuse_activation_pass(NpuGraphIR& graph, PassLog* log);
bool run_decompose_unsupported_pass(NpuGraphIR& graph, UnsupportedPolicy policy, PassLog* log);
bool run_canonicalize_pass(NpuGraphIR& graph, PassLog* log);

}  // namespace intel_npu
