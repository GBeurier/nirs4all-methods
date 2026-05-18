// SPDX-License-Identifier: CECILL-2.1
//
// chemometrics4all_cli — tiny introspection tool. Phase 0 (bootstrap) supports:
//   --version      print "chemometrics4all <project> (ABI <abi>)"
//   --abi-info     print ABI / backend / dtype metadata
//   --selfcheck    exercise context + matrix_view lifecycle, exit non-zero on
//                  any ABI inconsistency
//
// Subsequent phases extend --selfcheck and add a --bench subcommand.
// No third-party CLI parser is used — keep runtime dependencies aligned with
// libc4a's zero-mandatory-deps promise.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chemometrics4all/c4a.h"

namespace {

int cmd_version() {
    printf("chemometrics4all %s (ABI %u.%u.%u)\n",
           c4a_get_version_string(),
           c4a_get_abi_version_major(),
           c4a_get_abi_version_minor(),
           c4a_get_abi_version_patch());
    return 0;
}

int cmd_abi_info() {
    printf("=== chemometrics4all ABI info ===\n");
    printf("project_version : %s\n",   c4a_get_version_string());
    printf("abi_version     : %u.%u.%u (int=%u)\n",
           c4a_get_abi_version_major(),
           c4a_get_abi_version_minor(),
           c4a_get_abi_version_patch(),
           c4a_get_abi_version_int());
    printf("build_info      : %s\n",   c4a_get_build_info());
    printf("git_revision    : %s\n",   c4a_get_git_revision());

    printf("--- backends ---\n");
    c4a_backend_t backends[] = {
        C4A_BACKEND_AUTO,          C4A_BACKEND_REFERENCE_CPU,
        C4A_BACKEND_NATIVE_CPU,    C4A_BACKEND_BLAS,
        C4A_BACKEND_OPENMP,        C4A_BACKEND_CUDA,
        C4A_BACKEND_OPENCL,        C4A_BACKEND_METAL,
    };
    for (size_t i = 0; i < sizeof(backends)/sizeof(backends[0]); ++i) {
        printf("  %-16s available=%d\n",
               c4a_backend_to_string(backends[i]),
               c4a_backend_is_available(backends[i]));
    }

    printf("--- dtypes ---\n");
    c4a_dtype_t dtypes[] = {
        C4A_DTYPE_F64, C4A_DTYPE_F32, C4A_DTYPE_I32, C4A_DTYPE_I64,
    };
    for (size_t i = 0; i < sizeof(dtypes)/sizeof(dtypes[0]); ++i) {
        printf("  %-8s size=%zu bytes\n",
               c4a_dtype_to_string(dtypes[i]),
               c4a_dtype_size(dtypes[i]));
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
    CHECK(c4a_check_abi_compatibility(C4A_ABI_VERSION_MAJOR,
                                       C4A_ABI_VERSION_MINOR) == C4A_OK);

    // Context lifecycle.
    c4a_context_t* ctx = nullptr;
    CHECK(c4a_context_create(&ctx) == C4A_OK);
    CHECK(ctx != nullptr);

    uint64_t seed = 0;
    CHECK(c4a_context_set_seed(ctx, 42) == C4A_OK);
    CHECK(c4a_context_get_seed(ctx, &seed) == C4A_OK);
    CHECK(seed == 42);

    c4a_backend_t backend = C4A_BACKEND_AUTO;
    CHECK(c4a_context_set_backend(ctx, C4A_BACKEND_REFERENCE_CPU) == C4A_OK);
    CHECK(c4a_context_get_backend(ctx, &backend) == C4A_OK);
    CHECK(backend == C4A_BACKEND_REFERENCE_CPU);

    int32_t nt = 0;
    CHECK(c4a_context_set_num_threads(ctx, 1) == C4A_OK);
    CHECK(c4a_context_get_num_threads(ctx, &nt) == C4A_OK);
    CHECK(nt == 1);

    // Matrix view validation: 3x4 row-major double matrix.
    double data[3 * 4] = {0};
    c4a_matrix_view_t view;
    CHECK(c4a_matrix_view_init_rowmajor(&view, data, 3, 4, C4A_DTYPE_F64) == C4A_OK);
    CHECK(c4a_matrix_view_validate(&view) == C4A_OK);

    // Reject negative dims.
    c4a_matrix_view_t bad;
    CHECK(c4a_matrix_view_init_rowmajor(&bad, data, -1, 4, C4A_DTYPE_F64) == C4A_ERR_INVALID_ARGUMENT);

    c4a_context_destroy(ctx);

    printf("selfcheck OK (chemometrics4all %s, ABI %u.%u.%u)\n",
           c4a_get_version_string(),
           c4a_get_abi_version_major(),
           c4a_get_abi_version_minor(),
           c4a_get_abi_version_patch());
    return 0;
}

#undef CHECK

int print_usage(FILE* out) {
    fprintf(out,
        "Usage: chemometrics4all_cli <subcommand>\n"
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
