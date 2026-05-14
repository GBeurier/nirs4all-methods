// SPDX-License-Identifier: CeCILL-2.1
//
// pls4all_cli — tiny introspection tool. Supported subcommands:
//   --version      print "pls4all <project> (ABI <abi> · git <rev>)"
//   --abi-info     print all ABI / backend / dtype metadata
//   --selfcheck    exercise context + config + matrix_view lifecycle and
//                   exit non-zero on the first failure
//
// No third-party CLI parser is used — Phase 0 keeps the executable's
// runtime dependencies aligned with libp4a's zero-mandatory-deps promise.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pls4all/p4a.h"

namespace {

int cmd_version() {
    printf("pls4all %s\n", p4a_get_version_string());
    const char* rev = p4a_get_git_revision();
    if (rev && rev[0] != '\0') {
        printf("git: %s\n", rev);
    }
    return 0;
}

int cmd_abi_info() {
    printf("abi-version: %u.%u.%u (int %u)\n",
        p4a_get_abi_version_major(),
        p4a_get_abi_version_minor(),
        p4a_get_abi_version_patch(),
        p4a_get_abi_version_int());
    printf("project-version: %s\n", p4a_get_version_string());
    const char* build = p4a_get_build_info();
    if (build && build[0] != '\0') {
        printf("build-info: %s\n", build);
    }

    printf("backends:\n");
    const p4a_backend_t backends[] = {
        P4A_BACKEND_AUTO, P4A_BACKEND_REFERENCE_CPU, P4A_BACKEND_NATIVE_CPU,
        P4A_BACKEND_BLAS, P4A_BACKEND_OPENMP, P4A_BACKEND_CUDA,
        P4A_BACKEND_OPENCL, P4A_BACKEND_METAL,
    };
    for (size_t i = 0; i < sizeof(backends) / sizeof(backends[0]); ++i) {
        printf("  %s: %s\n", p4a_backend_to_string(backends[i]),
               p4a_backend_is_available(backends[i]) ? "available" : "unavailable");
    }

    printf("dtypes:\n");
    const p4a_dtype_t dtypes[] = {
        P4A_DTYPE_F64, P4A_DTYPE_F32, P4A_DTYPE_I32, P4A_DTYPE_I64,
    };
    for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); ++i) {
        printf("  %s: %zu bytes\n", p4a_dtype_to_string(dtypes[i]),
               p4a_dtype_size(dtypes[i]));
    }
    return 0;
}

int cmd_selfcheck() {
    int failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "selfcheck FAIL: %s\n", #cond); \
        ++failed; \
    } \
} while (0)

    // ABI compatibility
    CHECK(p4a_check_abi_compatibility(p4a_get_abi_version_major(),
                                       p4a_get_abi_version_minor()) == P4A_OK);
    CHECK(p4a_check_abi_compatibility(p4a_get_abi_version_major() + 1, 0) ==
          P4A_ERR_ABI_MISMATCH);

    // Status string
    CHECK(strcmp(p4a_status_to_string(P4A_OK), "ok") == 0);
    CHECK(strcmp(p4a_status_to_string(P4A_ERR_NOT_IMPLEMENTED), "not implemented") == 0);

    // Dtype
    CHECK(p4a_dtype_size(P4A_DTYPE_F64) == 8);
    CHECK(p4a_dtype_size(P4A_DTYPE_F32) == 4);

    // Context lifecycle
    p4a_context_t* ctx = NULL;
    CHECK(p4a_context_create(&ctx) == P4A_OK);
    CHECK(ctx != NULL);
    CHECK(p4a_context_set_seed(ctx, 42) == P4A_OK);
    uint64_t seed = 0;
    CHECK(p4a_context_get_seed(ctx, &seed) == P4A_OK);
    CHECK(seed == 42);
    CHECK(p4a_context_set_backend(ctx, P4A_BACKEND_CUDA) == P4A_ERR_BACKEND_UNAVAILABLE);

    // Config lifecycle + setters
    p4a_config_t* cfg = NULL;
    CHECK(p4a_config_create(&cfg) == P4A_OK);
    CHECK(p4a_config_set_n_components(cfg, 5) == P4A_OK);
    CHECK(p4a_config_set_n_components(cfg, -1) == P4A_ERR_INVALID_ARGUMENT);
    int32_t nc = 0;
    CHECK(p4a_config_get_n_components(cfg, &nc) == P4A_OK);
    CHECK(nc == 5);

    // Matrix view + validate
    double buf[12];
    p4a_matrix_view_t mv;
    CHECK(p4a_matrix_view_init_rowmajor(&mv, buf, 3, 4, P4A_DTYPE_F64) == P4A_OK);
    CHECK(p4a_matrix_view_validate(&mv) == P4A_OK);

    // Model fit must return NOT_IMPLEMENTED at Phase 0
    p4a_model_t* model = NULL;
    CHECK(p4a_model_fit(ctx, cfg, &mv, &mv, &model) == P4A_ERR_NOT_IMPLEMENTED);
    CHECK(model == NULL);

    p4a_config_destroy(cfg);
    p4a_context_destroy(ctx);
#undef CHECK

    if (failed == 0) {
        printf("selfcheck OK\n");
        return 0;
    }
    fprintf(stderr, "selfcheck failed: %d check(s)\n", failed);
    return 1;
}

int usage(const char* argv0) {
    fprintf(stderr,
        "Usage: %s {--version | --abi-info | --selfcheck}\n", argv0);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return usage(argv[0]);
    if (strcmp(argv[1], "--version")  == 0) return cmd_version();
    if (strcmp(argv[1], "--abi-info") == 0) return cmd_abi_info();
    if (strcmp(argv[1], "--selfcheck")== 0) return cmd_selfcheck();
    return usage(argv[0]);
}
