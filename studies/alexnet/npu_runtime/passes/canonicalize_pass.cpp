#include "../ir/npu_ir.hpp"

#include <algorithm>
#include <set>

namespace intel_npu {

namespace {

void remove_identity_ops(NpuGraphIR& graph, PassLog* log, bool& changed) {
    graph.rebuild_edges();
    for (auto& op : graph.ops) {
        if (op.removed || op.kind != NpuOpKind::IDENTITY || op.inputs.size() != 1 || op.outputs.size() != 1) {
            continue;
        }

        const u32 input_tensor_id = op.inputs.front();
        const u32 output_tensor_id = op.outputs.front();
        if (input_tensor_id == output_tensor_id) {
            op.removed = true;
            changed = true;
            continue;
        }

        graph.replace_all_uses(output_tensor_id, input_tensor_id);
        if (auto* output_tensor = graph.find_tensor(output_tensor_id)) {
            output_tensor->removed = true;
        }
        op.removed = true;
        changed = true;
        if (log != nullptr) {
            log->append("CanonicalizePass", "removed identity op " + std::to_string(op.id));
        }
        graph.rebuild_edges();
    }
}

std::set<u32> collect_live_ops(const NpuGraphIR& graph) {
    std::set<u32> live;
    std::vector<u32> stack = graph.output_tensor_ids;
    while (!stack.empty()) {
        const u32 tensor_id = stack.back();
        stack.pop_back();
        const auto* tensor = graph.find_tensor(tensor_id);
        if (tensor == nullptr || tensor->producer_op == 0) {
            continue;
        }
        if (!live.insert(tensor->producer_op).second) {
            continue;
        }
        const auto* producer = graph.find_op(tensor->producer_op);
        if (producer == nullptr) {
            continue;
        }
        for (u32 input_id : producer->inputs) {
            stack.push_back(input_id);
        }
    }
    return live;
}

}  // namespace

bool run_canonicalize_pass(NpuGraphIR& graph, PassLog* log) {
    bool changed = false;
    remove_identity_ops(graph, log, changed);

    graph.rebuild_edges();
    const auto live_ops = collect_live_ops(graph);
    for (auto& op : graph.ops) {
        if (op.removed) {
            changed = true;
            continue;
        }
        if (live_ops.count(op.id) == 0 && op.kind != NpuOpKind::FALLBACK) {
            op.removed = true;
            changed = true;
            if (log != nullptr) {
                log->append("CanonicalizePass", "removed dead op " + std::to_string(op.id));
            }
        }
    }

    graph.rebuild_edges();
    for (auto& tensor : graph.tensors) {
        if (tensor.removed) {
            changed = true;
            continue;
        }
        const bool is_live = tensor.is_graph_input || tensor.is_graph_output || tensor.producer_op != 0 || !tensor.consumer_ops.empty();
        if (!is_live) {
            tensor.removed = true;
            changed = true;
        }
    }

    if (changed) {
        graph.erase_removed();
    } else {
        graph.rebuild_edges();
        graph.recompute_topological_order();
    }
    return changed;
}

}  // namespace intel_npu
