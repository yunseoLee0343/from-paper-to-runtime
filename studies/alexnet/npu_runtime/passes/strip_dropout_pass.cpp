#include "../ir/npu_ir.hpp"

namespace intel_npu {

bool run_strip_dropout_pass(NpuGraphIR& graph, PassLog* log) {
    bool changed = false;
    for (auto& op : graph.ops) {
        if (op.removed || op.kind != NpuOpKind::DROPOUT) {
            continue;
        }
        op.kind = NpuOpKind::IDENTITY;
        changed = true;
        if (log != nullptr) {
            log->append("StripDropoutPass", "op " + std::to_string(op.id) + " " + op.name + " -> identity");
        }
    }
    if (changed) {
        graph.rebuild_edges();
        graph.recompute_topological_order();
    }
    return changed;
}

}  // namespace intel_npu
