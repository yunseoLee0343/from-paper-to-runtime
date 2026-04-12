#include "../ir/npu_ir.hpp"

namespace intel_npu {

bool run_fuse_activation_pass(NpuGraphIR& graph, PassLog* log) {
    bool changed = false;
    graph.rebuild_edges();
    graph.recompute_topological_order();

    for (auto& producer : graph.ops) {
        if (producer.removed || producer.kind != NpuOpKind::CONV || producer.post_op != PostOpKind::NONE) {
            continue;
        }
        if (producer.outputs.size() != 1) {
            continue;
        }

        auto* producer_output = graph.find_tensor(producer.outputs.front());
        if (producer_output == nullptr || producer_output->consumer_ops.size() != 1) {
            continue;
        }

        auto* consumer = graph.find_op(producer_output->consumer_ops.front());
        if (consumer == nullptr || consumer->removed || consumer->inputs.size() != 1 || consumer->inputs.front() != producer_output->id) {
            continue;
        }

        if (consumer->kind == NpuOpKind::RELU) {
            producer.post_op = PostOpKind::RELU;
        } else if (consumer->kind == NpuOpKind::CLAMP) {
            producer.post_op = PostOpKind::CLAMP;
            producer.attrs.clamp = consumer->attrs.clamp;
        } else {
            continue;
        }

        consumer->kind = NpuOpKind::IDENTITY;
        changed = true;
        if (log != nullptr) {
            log->append(
                "FuseActivationPass",
                "producer op " + std::to_string(producer.id) + " fused " + consumer->name + " as post_op=" + to_string(producer.post_op));
        }
    }

    if (changed) {
        graph.rebuild_edges();
        graph.recompute_topological_order();
    }
    return changed;
}

}  // namespace intel_npu
