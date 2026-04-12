#include "npu_ir.hpp"

#include "../intel_npu_scheduler.hpp"

#include <algorithm>
#include <deque>
#include <set>
#include <stdexcept>

namespace intel_npu {

namespace {

u32 next_ir_id(const std::vector<NpuOpIR>& ops, const std::vector<NpuTensorIR>& tensors) {
    u32 max_id = 0;
    for (const auto& op : ops) {
        max_id = std::max(max_id, op.id);
    }
    for (const auto& tensor : tensors) {
        max_id = std::max(max_id, tensor.id);
    }
    return max_id + 1;
}

}  // namespace

const char* to_string(NpuOpKind kind) {
    switch (kind) {
        case NpuOpKind::CONV: return "CONV";
        case NpuOpKind::RELU: return "RELU";
        case NpuOpKind::CLAMP: return "CLAMP";
        case NpuOpKind::MAXPOOL: return "MAXPOOL";
        case NpuOpKind::DROPOUT: return "DROPOUT";
        case NpuOpKind::IDENTITY: return "IDENTITY";
        case NpuOpKind::FALLBACK: return "FALLBACK";
        case NpuOpKind::LRN: return "LRN";
    }
    throw std::runtime_error("Unknown NPU op kind");
}

const char* to_string(UnsupportedPolicy policy) {
    switch (policy) {
        case UnsupportedPolicy::DECOMPOSE: return "DECOMPOSE";
        case UnsupportedPolicy::FALLBACK: return "FALLBACK";
        case UnsupportedPolicy::REJECT: return "REJECT";
    }
    throw std::runtime_error("Unknown unsupported policy");
}

NpuOpIR* NpuGraphIR::find_op(u32 id) {
    for (auto& op : ops) {
        if (op.id == id) {
            return &op;
        }
    }
    return nullptr;
}

const NpuOpIR* NpuGraphIR::find_op(u32 id) const {
    for (const auto& op : ops) {
        if (op.id == id) {
            return &op;
        }
    }
    return nullptr;
}

NpuTensorIR* NpuGraphIR::find_tensor(u32 id) {
    for (auto& tensor : tensors) {
        if (tensor.id == id) {
            return &tensor;
        }
    }
    return nullptr;
}

const NpuTensorIR* NpuGraphIR::find_tensor(u32 id) const {
    for (const auto& tensor : tensors) {
        if (tensor.id == id) {
            return &tensor;
        }
    }
    return nullptr;
}

void NpuGraphIR::rebuild_edges() {
    for (auto& tensor : tensors) {
        tensor.producer_op = 0;
        tensor.consumer_ops.clear();
    }

    for (const auto& op : ops) {
        if (op.removed) {
            continue;
        }
        for (u32 input_id : op.inputs) {
            if (auto* tensor = find_tensor(input_id)) {
                tensor->consumer_ops.push_back(op.id);
            }
        }
        for (u32 output_id : op.outputs) {
            if (auto* tensor = find_tensor(output_id)) {
                tensor->producer_op = op.id;
            }
        }
    }

    for (auto& tensor : tensors) {
        std::sort(tensor.consumer_ops.begin(), tensor.consumer_ops.end());
        tensor.consumer_ops.erase(std::unique(tensor.consumer_ops.begin(), tensor.consumer_ops.end()), tensor.consumer_ops.end());
        tensor.is_graph_input = std::find(input_tensor_ids.begin(), input_tensor_ids.end(), tensor.id) != input_tensor_ids.end();
        tensor.is_graph_output = std::find(output_tensor_ids.begin(), output_tensor_ids.end(), tensor.id) != output_tensor_ids.end();
    }
}

void NpuGraphIR::recompute_topological_order() {
    rebuild_edges();

    std::vector<u32> indegree(ops.size(), 0);
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (ops[i].removed) {
            continue;
        }
        std::set<u32> deps;
        for (u32 input_id : ops[i].inputs) {
            const auto* tensor = find_tensor(input_id);
            if (tensor != nullptr && tensor->producer_op != 0) {
                deps.insert(tensor->producer_op);
            }
        }
        indegree[i] = static_cast<u32>(deps.size());
    }

    std::deque<std::size_t> ready;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (!ops[i].removed && indegree[i] == 0) {
            ready.push_back(i);
        }
    }

    i32 topo = 0;
    while (!ready.empty()) {
        const std::size_t index = ready.front();
        ready.pop_front();
        ops[index].topo_index = topo++;

        for (u32 output_id : ops[index].outputs) {
            const auto* tensor = find_tensor(output_id);
            if (tensor == nullptr) {
                continue;
            }
            for (u32 consumer_id : tensor->consumer_ops) {
                for (std::size_t j = 0; j < ops.size(); ++j) {
                    if (ops[j].id == consumer_id && !ops[j].removed) {
                        if (indegree[j] > 0) {
                            --indegree[j];
                        }
                        if (indegree[j] == 0) {
                            ready.push_back(j);
                        }
                    }
                }
            }
        }
    }

    std::stable_sort(ops.begin(), ops.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.removed != rhs.removed) {
            return !lhs.removed;
        }
        return lhs.topo_index < rhs.topo_index;
    });
}

void NpuGraphIR::replace_all_uses(u32 old_tensor_id, u32 new_tensor_id) {
    for (auto& op : ops) {
        if (op.removed) {
            continue;
        }
        for (auto& input_id : op.inputs) {
            if (input_id == old_tensor_id) {
                input_id = new_tensor_id;
            }
        }
    }
    for (auto& output_id : output_tensor_ids) {
        if (output_id == old_tensor_id) {
            output_id = new_tensor_id;
        }
    }
}

void NpuGraphIR::erase_removed() {
    ops.erase(std::remove_if(ops.begin(), ops.end(), [](const auto& op) { return op.removed; }), ops.end());
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(), [](const auto& tensor) { return tensor.removed; }), tensors.end());
    rebuild_edges();
    recompute_topological_order();
}

NpuGraphIR lower_graph_desc_to_npu_ir(const GraphDesc& graph, const TensorTable& tensors) {
    NpuGraphIR ir;
    for (const auto& [tensor_id, desc] : tensors.table) {
        ir.tensors.push_back({tensor_id, desc.name, DataType::FP16, desc.shape, desc.mem_space});
    }

    for (const auto& op : graph.ops) {
        NpuOpIR ir_op;
        ir_op.id = op.op_id;
        ir_op.name = op.name;
        ir_op.inputs.push_back(op.input_tensor_id);
        if (op.weight_tensor_id != 0) {
            ir_op.inputs.push_back(op.weight_tensor_id);
        }
        ir_op.outputs.push_back(op.output_tensor_id);

        switch (op.kind) {
            case OpKind::CONV2D:
                ir_op.kind = NpuOpKind::CONV;
                ir_op.attrs.conv = {op.conv.stride_h, op.conv.stride_w, op.conv.pad_h, op.conv.pad_w, op.conv.kernel_h, op.conv.kernel_w};
                break;
            case OpKind::RELU:
                ir_op.kind = NpuOpKind::RELU;
                break;
            case OpKind::CLAMP:
                ir_op.kind = NpuOpKind::CLAMP;
                ir_op.attrs.clamp = {op.clamp_min, op.clamp_max};
                break;
            case OpKind::MAXPOOL:
                ir_op.kind = NpuOpKind::MAXPOOL;
                break;
            case OpKind::DROPOUT:
                ir_op.kind = NpuOpKind::DROPOUT;
                break;
            case OpKind::IDENTITY:
                ir_op.kind = NpuOpKind::IDENTITY;
                break;
            case OpKind::FALLBACK:
                ir_op.kind = NpuOpKind::FALLBACK;
                ir_op.attrs.fallback = {"GraphOpDesc", "frontend-marked-fallback"};
                break;
            case OpKind::LRN:
                ir_op.kind = NpuOpKind::LRN;
                break;
        }

        ir.ops.push_back(ir_op);
    }

    if (!graph.ops.empty()) {
        ir.input_tensor_ids.push_back(graph.ops.front().input_tensor_id);
        ir.output_tensor_ids.push_back(graph.ops.back().output_tensor_id);
    }

    ir.rebuild_edges();
    ir.recompute_topological_order();
    return ir;
}

NpuGraphIR make_compiler_demo_ir() {
    NpuGraphIR graph;
    graph.tensors = {
        {1, "input", DataType::FP16, {1, 64, 32, 32}, MemorySpace::DDR},
        {2, "weight0", DataType::FP16, {64, 64, 3, 3}, MemorySpace::DDR},
        {3, "conv0_out", DataType::FP16, {1, 64, 32, 32}, MemorySpace::DDR},
        {4, "relu0_out", DataType::FP16, {1, 64, 32, 32}, MemorySpace::DDR},
        {5, "dropout0_out", DataType::FP16, {1, 64, 32, 32}, MemorySpace::DDR},
        {6, "weight1", DataType::FP16, {64, 64, 3, 3}, MemorySpace::DDR},
        {7, "conv1_out", DataType::FP16, {1, 64, 32, 32}, MemorySpace::DDR},
        {8, "lrn_out", DataType::FP16, {1, 64, 32, 32}, MemorySpace::DDR},
    };

    NpuOpIR conv0;
    conv0.id = 10;
    conv0.kind = NpuOpKind::CONV;
    conv0.name = "conv0";
    conv0.inputs = {1, 2};
    conv0.outputs = {3};
    conv0.post_op = PostOpKind::NONE;
    conv0.attrs.conv = {1, 1, 1, 1, 3, 3};

    NpuOpIR relu0;
    relu0.id = 11;
    relu0.kind = NpuOpKind::RELU;
    relu0.name = "relu0";
    relu0.inputs = {3};
    relu0.outputs = {4};

    NpuOpIR dropout0;
    dropout0.id = 12;
    dropout0.kind = NpuOpKind::DROPOUT;
    dropout0.name = "dropout0";
    dropout0.inputs = {4};
    dropout0.outputs = {5};

    NpuOpIR conv1;
    conv1.id = 13;
    conv1.kind = NpuOpKind::CONV;
    conv1.name = "conv1";
    conv1.inputs = {5, 6};
    conv1.outputs = {7};
    conv1.post_op = PostOpKind::NONE;
    conv1.attrs.conv = {1, 1, 1, 1, 3, 3};

    NpuOpIR lrn0;
    lrn0.id = 14;
    lrn0.kind = NpuOpKind::LRN;
    lrn0.name = "lrn0";
    lrn0.inputs = {7};
    lrn0.outputs = {8};

    graph.ops = {conv0, relu0, dropout0, conv1, lrn0};
    graph.input_tensor_ids = {1};
    graph.output_tensor_ids = {8};
    graph.rebuild_edges();
    graph.recompute_topological_order();
    return graph;
}

std::string dump_graph_ir_compact(const NpuGraphIR& graph) {
    std::ostringstream oss;
    for (const auto& op : graph.ops) {
        if (op.removed) {
            continue;
        }
        oss << op.id << ":" << to_string(op.kind) << "(" << op.name << ") in=";
        for (std::size_t i = 0; i < op.inputs.size(); ++i) {
            oss << op.inputs[i] << (i + 1 == op.inputs.size() ? "" : ",");
        }
        oss << " out=";
        for (std::size_t i = 0; i < op.outputs.size(); ++i) {
            oss << op.outputs[i] << (i + 1 == op.outputs.size() ? "" : ",");
        }
        if (op.post_op != PostOpKind::NONE) {
            oss << " post_op=" << to_string(op.post_op);
        }
        oss << '\n';
    }
    return oss.str();
}

std::string dump_graph_ir(const NpuGraphIR& graph) {
    std::ostringstream oss;
    oss << "NpuGraphIR\n";
    oss << "Inputs: ";
    for (std::size_t i = 0; i < graph.input_tensor_ids.size(); ++i) {
        oss << graph.input_tensor_ids[i] << (i + 1 == graph.input_tensor_ids.size() ? "" : ", ");
    }
    oss << "\nOutputs: ";
    for (std::size_t i = 0; i < graph.output_tensor_ids.size(); ++i) {
        oss << graph.output_tensor_ids[i] << (i + 1 == graph.output_tensor_ids.size() ? "" : ", ");
    }
    oss << "\nOps:\n" << dump_graph_ir_compact(graph);
    oss << "Tensors:\n";
    for (const auto& tensor : graph.tensors) {
        if (tensor.removed) {
            continue;
        }
        oss << "  tensor " << tensor.id << " " << tensor.name << " producer=" << tensor.producer_op << " consumers=";
        for (std::size_t i = 0; i < tensor.consumer_ops.size(); ++i) {
            oss << tensor.consumer_ops[i] << (i + 1 == tensor.consumer_ops.size() ? "" : ",");
        }
        oss << '\n';
    }
    return oss.str();
}

}  // namespace intel_npu

