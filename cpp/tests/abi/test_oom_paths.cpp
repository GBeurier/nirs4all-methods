// SPDX-License-Identifier: CeCILL-2.1
//
// Lightweight churn / leak-shape tests for allocation paths. The real OOM
// injection lives behind PLS4ALL_ENABLE_FAULT_INJECTION (a future PR);
// here we only exercise the success paths and verify NULL-output handling
// — sanitizer-enabled CI catches actual leaks.

#include "pls4all/p4a.h"
#include "harness.hpp"

TEST(oom, context_create_null_out_does_not_leak) {
    CHECK_EQ(p4a_context_create(nullptr), P4A_ERR_NULL_POINTER);
}

TEST(oom, config_create_null_out_does_not_leak) {
    CHECK_EQ(p4a_config_create(nullptr), P4A_ERR_NULL_POINTER);
}

TEST(oom, operator_bank_create_null_out) {
    CHECK_EQ(p4a_operator_bank_create(nullptr), P4A_ERR_NULL_POINTER);
}

TEST(oom, pipeline_create_null_out) {
    CHECK_EQ(p4a_pipeline_create(nullptr), P4A_ERR_NULL_POINTER);
}

TEST(oom, gating_strategy_create_null_out) {
    CHECK_EQ(p4a_gating_strategy_create(nullptr, P4A_GATING_HARD),
              P4A_ERR_NULL_POINTER);
}

TEST(oom, clone_null_dst_pointer) {
    p4a_config_t* src = nullptr;
    p4a_config_create(&src);
    CHECK_EQ(p4a_config_clone(src, nullptr), P4A_ERR_NULL_POINTER);
    p4a_config_destroy(src);
}

TEST(oom, 5k_operator_bank_adds_under_sanitizer) {
    p4a_operator_bank_t* bank = nullptr;
    p4a_operator_bank_create(&bank);
    for (int i = 0; i < 5000; ++i) {
        double p[2] = {static_cast<double>(i), 1.0};
        CHECK_EQ(p4a_operator_bank_add(bank, P4A_OP_SAVGOL_SMOOTH, p, 2),
                  P4A_OK);
    }
    int32_t sz = 0;
    p4a_operator_bank_size(bank, &sz);
    CHECK_EQ(sz, 5000);
    p4a_operator_bank_destroy(bank);
}

TEST(oom, 1k_pipeline_clones) {
    p4a_pipeline_t* base = nullptr;
    p4a_pipeline_create(&base);
    p4a_pipeline_add_operator(base, P4A_OP_SNV, nullptr, 0);
    p4a_pipeline_add_operator(base, P4A_OP_MSC, nullptr, 0);
    for (int i = 0; i < 1000; ++i) {
        p4a_pipeline_t* c = nullptr;
        CHECK_EQ(p4a_pipeline_clone(base, &c), P4A_OK);
        p4a_pipeline_destroy(c);
    }
    p4a_pipeline_destroy(base);
}
