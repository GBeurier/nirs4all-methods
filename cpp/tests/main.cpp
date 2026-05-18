// SPDX-License-Identifier: CECILL-2.1
//
// chemometrics4all_tests — Phase 0 smoke harness. Each subsequent phase
// appends additional Runner::run() calls below.

#include "harness.hpp"

// Phase 1 — PCG64 RNG parity tests live in test_rng_pcg64.cpp; the entry
// point below is declared there and links via the same executable.
void register_rng_pcg64_tests(c4a_testing::Runner& r);

// Phase 2 — stateless preprocessing parity tests live in
// test_preprocessing_stateless.cpp.
void register_preprocessing_stateless_tests(c4a_testing::Runner& r);

// Phase 3 — stateful preprocessing parity tests live in
// test_preprocessing_stateful.cpp.
void register_preprocessing_stateful_tests(c4a_testing::Runner& r);

// Phase 4 — derivative & smoothing parity tests live in
// test_preprocessing_smoothing.cpp.
void register_preprocessing_smoothing_tests(c4a_testing::Runner& r);

// Phase 5a — baseline correction parity tests live in
// test_preprocessing_baselines.cpp.
void register_preprocessing_baselines_tests(c4a_testing::Runner& r);

namespace {

void test_version_string_nonempty() {
    const char* v = c4a_get_version_string();
    C4A_TEST_REQUIRE(v != nullptr);
    C4A_TEST_REQUIRE(v[0] != '\0');
}

void test_abi_version_compatible_with_header() {
    C4A_TEST_REQUIRE(c4a_get_abi_version_major() == C4A_ABI_VERSION_MAJOR);
    C4A_TEST_REQUIRE(c4a_get_abi_version_minor() == C4A_ABI_VERSION_MINOR);
    C4A_TEST_REQUIRE(c4a_get_abi_version_patch() == C4A_ABI_VERSION_PATCH);
    C4A_TEST_REQUIRE(c4a_check_abi_compatibility(C4A_ABI_VERSION_MAJOR,
                                                  C4A_ABI_VERSION_MINOR) == C4A_OK);
}

void test_status_strings_nonnull() {
    C4A_TEST_REQUIRE(c4a_status_to_string(C4A_OK) != nullptr);
    C4A_TEST_REQUIRE(c4a_status_to_string(C4A_ERR_NULL_POINTER) != nullptr);
    C4A_TEST_REQUIRE(c4a_status_to_string(C4A_ERR_INVALID_ARGUMENT) != nullptr);
}

void test_context_lifecycle() {
    c4a_context_t* ctx = nullptr;
    C4A_TEST_REQUIRE(c4a_context_create(&ctx) == C4A_OK);
    C4A_TEST_REQUIRE(ctx != nullptr);
    uint64_t seed = 0;
    C4A_TEST_REQUIRE(c4a_context_set_seed(ctx, 12345) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_context_get_seed(ctx, &seed) == C4A_OK);
    C4A_TEST_REQUIRE(seed == 12345);
    c4a_context_destroy(ctx);
}

void test_matrix_view_rowmajor() {
    double data[6] = {1, 2, 3, 4, 5, 6};
    c4a_matrix_view_t v;
    C4A_TEST_REQUIRE(c4a_matrix_view_init_rowmajor(&v, data, 2, 3, C4A_DTYPE_F64) == C4A_OK);
    C4A_TEST_REQUIRE(c4a_matrix_view_validate(&v) == C4A_OK);
    C4A_TEST_REQUIRE(v.rows == 2);
    C4A_TEST_REQUIRE(v.cols == 3);
    C4A_TEST_REQUIRE(v.dtype == C4A_DTYPE_F64);
}

void test_dtype_sizes() {
    C4A_TEST_REQUIRE(c4a_dtype_size(C4A_DTYPE_F64) == 8);
    C4A_TEST_REQUIRE(c4a_dtype_size(C4A_DTYPE_F32) == 4);
    C4A_TEST_REQUIRE(c4a_dtype_size(C4A_DTYPE_I64) == 8);
    C4A_TEST_REQUIRE(c4a_dtype_size(C4A_DTYPE_I32) == 4);
}

void test_backend_reference_cpu_available() {
    C4A_TEST_REQUIRE(c4a_backend_is_available(C4A_BACKEND_REFERENCE_CPU) == 1);
}

}  // namespace

int main() {
    c4a_testing::Runner r("chemometrics4all");
    r.run("version_string_nonempty",            test_version_string_nonempty);
    r.run("abi_version_compatible_with_header", test_abi_version_compatible_with_header);
    r.run("status_strings_nonnull",             test_status_strings_nonnull);
    r.run("context_lifecycle",                  test_context_lifecycle);
    r.run("matrix_view_rowmajor",               test_matrix_view_rowmajor);
    r.run("dtype_sizes",                        test_dtype_sizes);
    r.run("backend_reference_cpu_available",    test_backend_reference_cpu_available);
    register_rng_pcg64_tests(r);
    register_preprocessing_stateless_tests(r);
    register_preprocessing_stateful_tests(r);
    register_preprocessing_smoothing_tests(r);
    register_preprocessing_baselines_tests(r);
    return r.finalize();
}
