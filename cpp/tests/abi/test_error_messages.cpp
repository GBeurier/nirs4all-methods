// SPDX-License-Identifier: CECILL-2.1

#include <cstring>
#include <cstdio>

#include "n4m/n4m.h"
#include "harness.hpp"

TEST(error_messages, fresh_context_has_empty_last_error) {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    const char* err = n4m_context_last_error(ctx);
    CHECK_NE(err, nullptr);
    CHECK_STR_EQ(err, "");
    n4m_context_destroy(ctx);
}

TEST(error_messages, last_error_writes_after_failed_setter) {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    CHECK_EQ(n4m_context_set_backend(ctx, N4M_BACKEND_OPENCL),
              N4M_ERR_BACKEND_UNAVAILABLE);
    const char* err = n4m_context_last_error(ctx);
    CHECK_NE(err, nullptr);
    CHECK(err[0] != '\0');
    CHECK_STR_CONTAINS(err, "not compiled");
    n4m_context_destroy(ctx);
}

TEST(error_messages, overwrite_not_append) {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_context_set_backend(ctx, N4M_BACKEND_OPENCL);   // sets one message
    const char* first = n4m_context_last_error(ctx);
    char first_copy[256];
    std::snprintf(first_copy, sizeof(first_copy), "%s", first);

    // Pick a backend that is guaranteed unavailable regardless of build
    // configuration. N4M_BACKEND_BLAS may be compiled in (Phase 43) and
    // would silently succeed without populating an error.
    n4m_context_set_backend(ctx, N4M_BACKEND_METAL);  // sets another
    const char* second = n4m_context_last_error(ctx);
    CHECK_STR_CONTAINS(second, "not compiled");
    // The two messages differ in the backend number, so they should not
    // be exactly equal byte-for-byte.
    CHECK(std::strcmp(first_copy, second) != 0);
    n4m_context_destroy(ctx);
}

TEST(error_messages, clear_resets_to_empty_string) {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    n4m_context_set_backend(ctx, N4M_BACKEND_OPENCL);
    CHECK(n4m_context_last_error(ctx)[0] != '\0');
    n4m_context_clear_error(ctx);
    CHECK_STR_EQ(n4m_context_last_error(ctx), "");
    n4m_context_destroy(ctx);
}

TEST(error_messages, null_ctx_returns_empty_string_not_null) {
    const char* err = n4m_context_last_error(nullptr);
    CHECK_NE(err, nullptr);
    CHECK_STR_EQ(err, "");
}

TEST(error_messages, clear_null_ctx_is_safe) {
    n4m_context_clear_error(nullptr);
    CHECK(true);
}

TEST(error_messages, getter_null_out_writes_to_last_error) {
    n4m_context_t* ctx = nullptr;
    n4m_context_create(&ctx);
    CHECK_EQ(n4m_context_get_seed(ctx, nullptr), N4M_ERR_NULL_POINTER);
    const char* err = n4m_context_last_error(ctx);
    CHECK_STR_CONTAINS(err, "null out pointer");
    n4m_context_destroy(ctx);
}
