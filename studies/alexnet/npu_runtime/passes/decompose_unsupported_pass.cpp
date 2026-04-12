#include "../ir/npu_ir.hpp"

#include <stdexcept>

namespace intel_npu {

bool run_decompose_unsupported_pass(NpuGraphIR& graph, UnsupportedPolicy policy, PassLog* log) {
    bool changed = false;
    for (auto& op : graph.ops) {
        if (op.removed || op.kind != NpuOpKind::LRN) {
            continue;
        }

        if (policy == UnsupportedPolicy::REJECT) {
            throw std::runtime_error(
                "DecomposeUnsupportedPass rejected unsupported op id=" + std::to_string(op.id) +
                " kind=" + std::string(to_string(op.kind)));
        }

        op.kind = NpuOpKind::FALLBACK;
        op.attrs.fallback.original_op_kind = "LRN";
        op.attrs.fallback.reason = policy == UnsupportedPolicy::DECOMPOSE ? "decompose-todo" : "fallback-policy";
        changed = true;
        if (log != nullptr) {
            log->append(
                "DecomposeUnsupportedPass",
                "op " + std::to_string(op.id) + " -> fallback reason=" + op.attrs.fallback.reason +
                " policy=" + to_string(policy));
        }
    }

    if (changed) {
        graph.rebuild_edges();
        graph.recompute_topological_order();
    }
    return changed;
}

}  // namespace intel_npu
