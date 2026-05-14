// SPDX-License-Identifier: CeCILL-2.1

#include <cstring>
#include <cstdio>

#include "pls4all/p4a.h"
#include "harness.hpp"

TEST(error_messages, fresh_context_has_empty_last_error) {
    p4a_context_t* ctx = nullptr;
    p4a_context_create(&ctx);
    const char* err = p4a_context_last_error(ctx);
    CHECK_NE(err, nullptr);
    CHECK_STR_EQ(err, "");
    p4a_context_destroy(ctx);
}

TEST(error_messages, last_error_writes_after_failed_setter) {
    p4a_context_t* ctx = nullptr;
    p4a_context_create(&ctx);
    CHECK_EQ(p4a_context_set_backend(ctx, P4A_BACKEND_CUDA),
              P4A_ERR_BACKEND_UNAVAILABLE);
    const char* err = p4a_context_last_error(ctx);
    CHECK_NE(err, nullptr);
    CHECK(err[0] != '\0');
    CHECK_STR_CONTAINS(err, "not compiled");
    p4a_context_destroy(ctx);
}

TEST(error_messages, overwrite_not_append) {
    p4a_context_t* ctx = nullptr;
    p4a_context_create(&ctx);
    p4a_context_set_backend(ctx, P4A_BACKEND_CUDA);   // sets one message
    const char* first = p4a_context_last_error(ctx);
    char first_copy[256];
    std::snprintf(first_copy, sizeof(first_copy), "%s", first);

    p4a_context_set_backend(ctx, P4A_BACKEND_BLAS);   // sets another
    const char* second = p4a_context_last_error(ctx);
    CHECK_STR_CONTAINS(second, "not compiled");
    // The two messages differ in the backend number, so they should not
    // be exactly equal byte-for-byte.
    CHECK(std::strcmp(first_copy, second) != 0);
    p4a_context_destroy(ctx);
}

TEST(error_messages, clear_resets_to_empty_string) {
    p4a_context_t* ctx = nullptr;
    p4a_context_create(&ctx);
    p4a_context_set_backend(ctx, P4A_BACKEND_CUDA);
    CHECK(p4a_context_last_error(ctx)[0] != '\0');
    p4a_context_clear_error(ctx);
    CHECK_STR_EQ(p4a_context_last_error(ctx), "");
    p4a_context_destroy(ctx);
}

TEST(error_messages, null_ctx_returns_empty_string_not_null) {
    const char* err = p4a_context_last_error(nullptr);
    CHECK_NE(err, nullptr);
    CHECK_STR_EQ(err, "");
}

TEST(error_messages, clear_null_ctx_is_safe) {
    p4a_context_clear_error(nullptr);
    CHECK(true);
}

TEST(error_messages, getter_null_out_writes_to_last_error) {
    p4a_context_t* ctx = nullptr;
    p4a_context_create(&ctx);
    CHECK_EQ(p4a_context_get_seed(ctx, nullptr), P4A_ERR_NULL_POINTER);
    const char* err = p4a_context_last_error(ctx);
    CHECK_STR_CONTAINS(err, "null out pointer");
    p4a_context_destroy(ctx);
}
