// SPDX-License-Identifier: CECILL-2.1

#include "pls4all/p4a.h"
#include "harness.hpp"

TEST(lifecycle, context_create_and_destroy) {
    p4a_context_t* ctx = nullptr;
    CHECK_EQ(p4a_context_create(&ctx), P4A_OK);
    CHECK_NE(ctx, nullptr);
    p4a_context_destroy(ctx);
}

TEST(lifecycle, context_create_null_out_pointer) {
    CHECK_EQ(p4a_context_create(nullptr), P4A_ERR_NULL_POINTER);
}

TEST(lifecycle, context_destroy_null_is_safe) {
    // Documented as a no-op; this test just verifies "doesn't crash".
    p4a_context_destroy(nullptr);
    CHECK(true);
}

TEST(lifecycle, config_create_and_destroy) {
    p4a_config_t* cfg = nullptr;
    CHECK_EQ(p4a_config_create(&cfg), P4A_OK);
    CHECK_NE(cfg, nullptr);
    p4a_config_destroy(cfg);
}

TEST(lifecycle, config_destroy_null_is_safe) {
    p4a_config_destroy(nullptr);
    CHECK(true);
}

TEST(lifecycle, churn_10k_create_destroy_pairs) {
    for (int i = 0; i < 10000; ++i) {
        p4a_context_t* ctx = nullptr;
        CHECK_EQ(p4a_context_create(&ctx), P4A_OK);
        p4a_context_destroy(ctx);
    }
}

TEST(lifecycle, operator_bank_create_destroy) {
    p4a_operator_bank_t* bank = nullptr;
    CHECK_EQ(p4a_operator_bank_create(&bank), P4A_OK);
    CHECK_NE(bank, nullptr);
    p4a_operator_bank_destroy(bank);
    p4a_operator_bank_destroy(nullptr);  // NULL-safe
}

TEST(lifecycle, gating_strategy_create_destroy) {
    p4a_gating_strategy_t* gs = nullptr;
    CHECK_EQ(p4a_gating_strategy_create(&gs, P4A_GATING_SOFT), P4A_OK);
    CHECK_NE(gs, nullptr);
    p4a_gating_strategy_destroy(gs);
    p4a_gating_strategy_destroy(nullptr);
}

TEST(lifecycle, pipeline_create_destroy_clone) {
    p4a_pipeline_t* pipe = nullptr;
    CHECK_EQ(p4a_pipeline_create(&pipe), P4A_OK);
    p4a_pipeline_t* clone = nullptr;
    CHECK_EQ(p4a_pipeline_clone(pipe, &clone), P4A_OK);
    CHECK_NE(clone, pipe);
    p4a_pipeline_destroy(clone);
    p4a_pipeline_destroy(pipe);
    p4a_pipeline_destroy(nullptr);
}
