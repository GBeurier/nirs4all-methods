// SPDX-License-Identifier: CECILL-2.1
//
// Lightweight churn / leak-shape tests for allocation paths. The real OOM
// injection lives behind PLS4ALL_ENABLE_FAULT_INJECTION (a future PR);
// here we only exercise the success paths and verify NULL-output handling
// — sanitizer-enabled CI catches actual leaks.

#include "n4m/n4m.h"
#include "harness.hpp"

TEST(oom, context_create_null_out_does_not_leak) {
    CHECK_EQ(n4m_context_create(nullptr), N4M_ERR_NULL_POINTER);
}

TEST(oom, config_create_null_out_does_not_leak) {
    CHECK_EQ(n4m_config_create(nullptr), N4M_ERR_NULL_POINTER);
}

TEST(oom, operator_bank_create_null_out) {
    CHECK_EQ(n4m_operator_bank_create(nullptr), N4M_ERR_NULL_POINTER);
}

TEST(oom, pipeline_create_null_out) {
    CHECK_EQ(n4m_pipeline_create(nullptr), N4M_ERR_NULL_POINTER);
}

TEST(oom, gating_strategy_create_null_out) {
    CHECK_EQ(n4m_gating_strategy_create(nullptr, N4M_GATING_HARD),
              N4M_ERR_NULL_POINTER);
}

TEST(oom, clone_null_dst_pointer) {
    n4m_config_t* src = nullptr;
    n4m_config_create(&src);
    CHECK_EQ(n4m_config_clone(src, nullptr), N4M_ERR_NULL_POINTER);
    n4m_config_destroy(src);
}

TEST(oom, 5k_operator_bank_adds_under_sanitizer) {
    n4m_operator_bank_t* bank = nullptr;
    n4m_operator_bank_create(&bank);
    for (int i = 0; i < 5000; ++i) {
        double p[2] = {static_cast<double>(i), 1.0};
        CHECK_EQ(n4m_operator_bank_add(bank, N4M_OP_SAVGOL_SMOOTH, p, 2),
                  N4M_OK);
    }
    int32_t sz = 0;
    n4m_operator_bank_size(bank, &sz);
    CHECK_EQ(sz, 5000);
    n4m_operator_bank_destroy(bank);
}

TEST(oom, 1k_pipeline_clones) {
    n4m_pipeline_t* base = nullptr;
    n4m_pipeline_create(&base);
    n4m_pipeline_add_operator(base, N4M_OP_SNV, nullptr, 0);
    n4m_pipeline_add_operator(base, N4M_OP_MSC, nullptr, 0);
    for (int i = 0; i < 1000; ++i) {
        n4m_pipeline_t* c = nullptr;
        CHECK_EQ(n4m_pipeline_clone(base, &c), N4M_OK);
        n4m_pipeline_destroy(c);
    }
    n4m_pipeline_destroy(base);
}
