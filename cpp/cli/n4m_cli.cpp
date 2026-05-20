// SPDX-License-Identifier: CECILL-2.1
//
// n4m_cli — tiny introspection tool. Phase 0 (bootstrap) supports:
//   --version      print "nirs4all-methods <project> (ABI <abi>)"
//   --abi-info     print ABI / backend / dtype metadata
//   --selfcheck    exercise context + matrix_view lifecycle, exit non-zero on
//                  any ABI inconsistency
//
// Subsequent phases extend --selfcheck and add a --bench subcommand.
// No third-party CLI parser is used — keep runtime dependencies aligned with
// libn4m's zero-mandatory-deps promise.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "n4m/n4m.h"

namespace {

int cmd_version() {
    printf("nirs4all-methods %s (ABI %u.%u.%u)\n",
           n4m_get_version_string(),
           n4m_get_abi_version_major(),
           n4m_get_abi_version_minor(),
           n4m_get_abi_version_patch());
    return 0;
}

int cmd_abi_info() {
    printf("=== nirs4all-methods ABI info ===\n");
    printf("project_version : %s\n",   n4m_get_version_string());
    printf("abi_version     : %u.%u.%u (int=%u)\n",
           n4m_get_abi_version_major(),
           n4m_get_abi_version_minor(),
           n4m_get_abi_version_patch(),
           n4m_get_abi_version_int());
    printf("build_info      : %s\n",   n4m_get_build_info());
    printf("git_revision    : %s\n",   n4m_get_git_revision());

    printf("--- backends ---\n");
    n4m_backend_t backends[] = {
        N4M_BACKEND_AUTO,          N4M_BACKEND_REFERENCE_CPU,
        N4M_BACKEND_NATIVE_CPU,    N4M_BACKEND_BLAS,
        N4M_BACKEND_OPENMP,        N4M_BACKEND_CUDA,
        N4M_BACKEND_OPENCL,        N4M_BACKEND_METAL,
    };
    for (size_t i = 0; i < sizeof(backends)/sizeof(backends[0]); ++i) {
        printf("  %-16s available=%d\n",
               n4m_backend_to_string(backends[i]),
               n4m_backend_is_available(backends[i]));
    }

    printf("--- dtypes ---\n");
    n4m_dtype_t dtypes[] = {
        N4M_DTYPE_F64, N4M_DTYPE_F32, N4M_DTYPE_I32, N4M_DTYPE_I64,
    };
    for (size_t i = 0; i < sizeof(dtypes)/sizeof(dtypes[0]); ++i) {
        printf("  %-8s size=%zu bytes\n",
               n4m_dtype_to_string(dtypes[i]),
               n4m_dtype_size(dtypes[i]));
    }
    return 0;
}

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "selfcheck FAIL: %s (line %d)\n", #cond, __LINE__); \
        return 1; \
    } \
} while (0)

int cmd_selfcheck() {
    // ABI compatibility self-check.
    CHECK(n4m_check_abi_compatibility(N4M_ABI_VERSION_MAJOR,
                                       N4M_ABI_VERSION_MINOR) == N4M_OK);

    // Context lifecycle.
    n4m_context_t* ctx = nullptr;
    CHECK(n4m_context_create(&ctx) == N4M_OK);
    CHECK(ctx != nullptr);

    uint64_t seed = 0;
    CHECK(n4m_context_set_seed(ctx, 42) == N4M_OK);
    CHECK(n4m_context_get_seed(ctx, &seed) == N4M_OK);
    CHECK(seed == 42);

    n4m_backend_t backend = N4M_BACKEND_AUTO;
    CHECK(n4m_context_set_backend(ctx, N4M_BACKEND_REFERENCE_CPU) == N4M_OK);
    CHECK(n4m_context_get_backend(ctx, &backend) == N4M_OK);
    CHECK(backend == N4M_BACKEND_REFERENCE_CPU);

    int32_t nt = 0;
    CHECK(n4m_context_set_num_threads(ctx, 1) == N4M_OK);
    CHECK(n4m_context_get_num_threads(ctx, &nt) == N4M_OK);
    CHECK(nt == 1);

    // Matrix view validation: 3x4 row-major double matrix.
    double data[3 * 4] = {0};
    n4m_matrix_view_t view;
    CHECK(n4m_matrix_view_init_rowmajor(&view, data, 3, 4, N4M_DTYPE_F64) == N4M_OK);
    CHECK(n4m_matrix_view_validate(&view) == N4M_OK);

    // Reject negative dims.
    n4m_matrix_view_t bad;
    CHECK(n4m_matrix_view_init_rowmajor(&bad, data, -1, 4, N4M_DTYPE_F64) == N4M_ERR_INVALID_ARGUMENT);

    n4m_context_destroy(ctx);

    printf("selfcheck OK (nirs4all-methods %s, ABI %u.%u.%u)\n",
           n4m_get_version_string(),
           n4m_get_abi_version_major(),
           n4m_get_abi_version_minor(),
           n4m_get_abi_version_patch());
    return 0;
}

#undef CHECK

int print_usage(FILE* out) {
    fprintf(out,
        "Usage: n4m_cli <subcommand>\n"
        "\n"
        "Subcommands:\n"
        "  --version      print library version + ABI version\n"
        "  --abi-info     print ABI / backend / dtype metadata\n"
        "  --selfcheck    run ABI smoke tests\n"
        "  --help, -h     print this message\n"
        "\n");
    return out == stderr ? 2 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return print_usage(stderr);
    }
    const char* cmd = argv[1];
    if (strcmp(cmd, "--version") == 0)     return cmd_version();
    if (strcmp(cmd, "--abi-info") == 0)    return cmd_abi_info();
    if (strcmp(cmd, "--selfcheck") == 0)   return cmd_selfcheck();
    if (strcmp(cmd, "--help") == 0 ||
        strcmp(cmd, "-h")     == 0)        return print_usage(stdout);
    fprintf(stderr, "Unknown subcommand: %s\n", cmd);
    return print_usage(stderr);
}
