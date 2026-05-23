// SPDX-License-Identifier: CECILL-2.1

#include "n4m/n4m.h"
#include "harness.hpp"

namespace {

#if defined(__GNUC__) || defined(__clang__)
#  define N4M_TEST_NO_UBSAN __attribute__((no_sanitize("undefined")))
#else
#  define N4M_TEST_NO_UBSAN
#endif

N4M_TEST_NO_UBSAN const char* status_to_string_raw(int raw) {
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#endif
    const auto status = static_cast<n4m_status_t>(raw);
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif
    return n4m_status_to_string(status);
}

#undef N4M_TEST_NO_UBSAN

}  // namespace

TEST(status, every_status_has_a_string) {
    const n4m_status_t codes[] = {
        N4M_OK, N4M_ERR_INVALID_ARGUMENT, N4M_ERR_NULL_POINTER,
        N4M_ERR_SHAPE_MISMATCH, N4M_ERR_DTYPE_MISMATCH, N4M_ERR_STRIDE_INVALID,
        N4M_ERR_NOT_FITTED, N4M_ERR_NUMERICAL_FAILURE, N4M_ERR_CONVERGENCE_FAILED,
        N4M_ERR_OUT_OF_MEMORY, N4M_ERR_UNSUPPORTED, N4M_ERR_NOT_IMPLEMENTED,
        N4M_ERR_ABI_MISMATCH, N4M_ERR_IO, N4M_ERR_CORRUPT_BUFFER,
        N4M_ERR_VERSION_INCOMPATIBLE, N4M_ERR_BACKEND_UNAVAILABLE,
        N4M_ERR_CANCELLED, N4M_ERR_INTERNAL,
    };
    for (auto c : codes) {
        const char* s = n4m_status_to_string(c);
        CHECK_NE(s, nullptr);
        CHECK(s[0] != '\0');
    }
}

TEST(status, pinned_strings) {
    CHECK_STR_EQ(n4m_status_to_string(N4M_OK),                  "ok");
    CHECK_STR_EQ(n4m_status_to_string(N4M_ERR_INVALID_ARGUMENT),"invalid argument");
    CHECK_STR_EQ(n4m_status_to_string(N4M_ERR_NULL_POINTER),    "null pointer");
    CHECK_STR_EQ(n4m_status_to_string(N4M_ERR_NOT_IMPLEMENTED), "not implemented");
    CHECK_STR_EQ(n4m_status_to_string(N4M_ERR_OUT_OF_MEMORY),   "out of memory");
}

TEST(status, out_of_range_returns_unknown) {
    CHECK_STR_EQ(status_to_string_raw(9999), "unknown status");
}

TEST(status, dtype_strings_pinned) {
    CHECK_STR_EQ(n4m_dtype_to_string(N4M_DTYPE_F64), "f64");
    CHECK_STR_EQ(n4m_dtype_to_string(N4M_DTYPE_F32), "f32");
    CHECK_STR_EQ(n4m_dtype_to_string(N4M_DTYPE_I32), "i32");
    CHECK_STR_EQ(n4m_dtype_to_string(N4M_DTYPE_I64), "i64");
    CHECK_STR_EQ(n4m_dtype_to_string(N4M_DTYPE_UNKNOWN), "unknown");
}

TEST(status, backend_strings_pinned) {
    CHECK_STR_EQ(n4m_backend_to_string(N4M_BACKEND_AUTO),          "auto");
    CHECK_STR_EQ(n4m_backend_to_string(N4M_BACKEND_REFERENCE_CPU), "reference_cpu");
    CHECK_STR_EQ(n4m_backend_to_string(N4M_BACKEND_BLAS),          "blas");
    CHECK_STR_EQ(n4m_backend_to_string(N4M_BACKEND_CUDA),          "cuda");
}
