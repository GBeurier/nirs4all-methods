// SPDX-License-Identifier: CECILL-2.1

#include "n4m/n4m.h"
#include "harness.hpp"

TEST(lifecycle, context_create_and_destroy) {
    n4m_context_t* ctx = nullptr;
    CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
    CHECK_NE(ctx, nullptr);
    n4m_context_destroy(ctx);
}

TEST(lifecycle, context_create_null_out_pointer) {
    CHECK_EQ(n4m_context_create(nullptr), N4M_ERR_NULL_POINTER);
}

TEST(lifecycle, context_destroy_null_is_safe) {
    // Documented as a no-op; this test just verifies "doesn't crash".
    n4m_context_destroy(nullptr);
    CHECK(true);
}

TEST(lifecycle, config_create_and_destroy) {
    n4m_config_t* cfg = nullptr;
    CHECK_EQ(n4m_config_create(&cfg), N4M_OK);
    CHECK_NE(cfg, nullptr);
    n4m_config_destroy(cfg);
}

TEST(lifecycle, config_destroy_null_is_safe) {
    n4m_config_destroy(nullptr);
    CHECK(true);
}

TEST(lifecycle, churn_10k_create_destroy_pairs) {
    for (int i = 0; i < 10000; ++i) {
        n4m_context_t* ctx = nullptr;
        CHECK_EQ(n4m_context_create(&ctx), N4M_OK);
        n4m_context_destroy(ctx);
    }
}

TEST(lifecycle, operator_bank_create_destroy) {
    n4m_operator_bank_t* bank = nullptr;
    CHECK_EQ(n4m_operator_bank_create(&bank), N4M_OK);
    CHECK_NE(bank, nullptr);
    n4m_operator_bank_destroy(bank);
    n4m_operator_bank_destroy(nullptr);  // NULL-safe
}

TEST(lifecycle, gating_strategy_create_destroy) {
    n4m_gating_strategy_t* gs = nullptr;
    CHECK_EQ(n4m_gating_strategy_create(&gs, N4M_GATING_SOFT), N4M_OK);
    CHECK_NE(gs, nullptr);
    n4m_gating_strategy_destroy(gs);
    n4m_gating_strategy_destroy(nullptr);
}

TEST(lifecycle, pipeline_create_destroy_clone) {
    n4m_pipeline_t* pipe = nullptr;
    CHECK_EQ(n4m_pipeline_create(&pipe), N4M_OK);
    n4m_pipeline_t* clone = nullptr;
    CHECK_EQ(n4m_pipeline_clone(pipe, &clone), N4M_OK);
    CHECK_NE(clone, pipe);
    n4m_pipeline_destroy(clone);
    n4m_pipeline_destroy(pipe);
    n4m_pipeline_destroy(nullptr);
}
