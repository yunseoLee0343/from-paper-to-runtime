#include "ir/npu_ir.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace intel_npu;

int main() {
    {
        PassLog log;
        auto ir = make_compiler_demo_ir();
        const bool stripped = run_strip_dropout_pass(ir, &log);
        assert(stripped);
        bool saw_identity_dropout = false;
        for (const auto& op : ir.ops) {
            if (op.name == "dropout0" && op.kind == NpuOpKind::IDENTITY) {
                saw_identity_dropout = true;
            }
        }
        assert(saw_identity_dropout);
    }

    {
        PassLog log;
        auto ir = make_compiler_demo_ir();
        run_strip_dropout_pass(ir, &log);
        const bool fused = run_fuse_activation_pass(ir, &log);
        assert(fused);
        const auto* conv0 = ir.find_op(10);
        const auto* relu0 = ir.find_op(11);
        assert(conv0 != nullptr && conv0->post_op == PostOpKind::RELU);
        assert(relu0 != nullptr && relu0->kind == NpuOpKind::IDENTITY);
    }

    {
        PassLog log;
        auto ir = make_compiler_demo_ir();
        const bool changed = run_decompose_unsupported_pass(ir, UnsupportedPolicy::FALLBACK, &log);
        assert(changed);
        const auto* lrn0 = ir.find_op(14);
        assert(lrn0 != nullptr && lrn0->kind == NpuOpKind::FALLBACK);
        assert(lrn0->attrs.fallback.reason == "fallback-policy");
    }

    {
        PassLog log;
        auto ir = make_compiler_demo_ir();
        const bool changed = run_decompose_unsupported_pass(ir, UnsupportedPolicy::DECOMPOSE, &log);
        assert(changed);
        const auto* lrn0 = ir.find_op(14);
        assert(lrn0 != nullptr && lrn0->kind == NpuOpKind::FALLBACK);
        assert(lrn0->attrs.fallback.reason == "decompose-todo");
    }

    {
        auto ir = make_compiler_demo_ir();
        bool threw = false;
        try {
            run_decompose_unsupported_pass(ir, UnsupportedPolicy::REJECT, nullptr);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
    }

    {
        PassLog log;
        auto ir = make_compiler_demo_ir();
        run_strip_dropout_pass(ir, &log);
        run_fuse_activation_pass(ir, &log);
        run_decompose_unsupported_pass(ir, UnsupportedPolicy::FALLBACK, &log);
        const bool canonicalized = run_canonicalize_pass(ir, &log);
        assert(canonicalized);
        for (const auto& op : ir.ops) {
            assert(op.kind != NpuOpKind::IDENTITY);
        }
        const auto* conv0 = ir.find_op(10);
        assert(conv0 != nullptr && conv0->post_op == PostOpKind::RELU);
    }

    std::cout << "Intel NPU compiler pass tests passed\n";
    return 0;
}
