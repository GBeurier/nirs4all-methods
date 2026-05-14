// SPDX-License-Identifier: CeCILL-2.1

#include "pls4all/p4a.h"
#include "harness.hpp"

namespace {

#if defined(__GNUC__) || defined(__clang__)
#  define P4A_TEST_NO_UBSAN __attribute__((no_sanitize("undefined")))
#else
#  define P4A_TEST_NO_UBSAN
#endif

P4A_TEST_NO_UBSAN const char* status_to_string_raw(int raw) {
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#endif
    const auto status = static_cast<p4a_status_t>(raw);
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif
    return p4a_status_to_string(status);
}

#undef P4A_TEST_NO_UBSAN

}  // namespace

TEST(status, every_status_has_a_string) {
    const p4a_status_t codes[] = {
        P4A_OK, P4A_ERR_INVALID_ARGUMENT, P4A_ERR_NULL_POINTER,
        P4A_ERR_SHAPE_MISMATCH, P4A_ERR_DTYPE_MISMATCH, P4A_ERR_STRIDE_INVALID,
        P4A_ERR_NOT_FITTED, P4A_ERR_NUMERICAL_FAILURE, P4A_ERR_CONVERGENCE_FAILED,
        P4A_ERR_OUT_OF_MEMORY, P4A_ERR_UNSUPPORTED, P4A_ERR_NOT_IMPLEMENTED,
        P4A_ERR_ABI_MISMATCH, P4A_ERR_IO, P4A_ERR_CORRUPT_BUFFER,
        P4A_ERR_VERSION_INCOMPATIBLE, P4A_ERR_BACKEND_UNAVAILABLE,
        P4A_ERR_CANCELLED, P4A_ERR_INTERNAL,
    };
    for (auto c : codes) {
        const char* s = p4a_status_to_string(c);
        CHECK_NE(s, nullptr);
        CHECK(s[0] != '\0');
    }
}

TEST(status, pinned_strings) {
    CHECK_STR_EQ(p4a_status_to_string(P4A_OK),                  "ok");
    CHECK_STR_EQ(p4a_status_to_string(P4A_ERR_INVALID_ARGUMENT),"invalid argument");
    CHECK_STR_EQ(p4a_status_to_string(P4A_ERR_NULL_POINTER),    "null pointer");
    CHECK_STR_EQ(p4a_status_to_string(P4A_ERR_NOT_IMPLEMENTED), "not implemented");
    CHECK_STR_EQ(p4a_status_to_string(P4A_ERR_OUT_OF_MEMORY),   "out of memory");
}

TEST(status, out_of_range_returns_unknown) {
    CHECK_STR_EQ(status_to_string_raw(9999), "unknown status");
}

TEST(status, dtype_strings_pinned) {
    CHECK_STR_EQ(p4a_dtype_to_string(P4A_DTYPE_F64), "f64");
    CHECK_STR_EQ(p4a_dtype_to_string(P4A_DTYPE_F32), "f32");
    CHECK_STR_EQ(p4a_dtype_to_string(P4A_DTYPE_I32), "i32");
    CHECK_STR_EQ(p4a_dtype_to_string(P4A_DTYPE_I64), "i64");
    CHECK_STR_EQ(p4a_dtype_to_string(P4A_DTYPE_UNKNOWN), "unknown");
}

TEST(status, backend_strings_pinned) {
    CHECK_STR_EQ(p4a_backend_to_string(P4A_BACKEND_AUTO),          "auto");
    CHECK_STR_EQ(p4a_backend_to_string(P4A_BACKEND_REFERENCE_CPU), "reference_cpu");
    CHECK_STR_EQ(p4a_backend_to_string(P4A_BACKEND_BLAS),          "blas");
    CHECK_STR_EQ(p4a_backend_to_string(P4A_BACKEND_CUDA),          "cuda");
}
