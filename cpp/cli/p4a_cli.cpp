// SPDX-License-Identifier: CeCILL-2.1
//
// pls4all_cli — tiny introspection tool. Supported subcommands:
//   --version      print "pls4all <project> (ABI <abi> · git <rev>)"
//   --abi-info     print all ABI / backend / dtype metadata
//   --selfcheck    exercise context + config + matrix_view lifecycle and
//                   exit non-zero on failed ABI / model smoke checks
//
// No third-party CLI parser is used — keep the executable's
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
        printf("  %s: %llu bytes\n", p4a_dtype_to_string(dtypes[i]),
               static_cast<unsigned long long>(p4a_dtype_size(dtypes[i])));
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
    p4a_context_t* ctx = nullptr;
    CHECK(p4a_context_create(&ctx) == P4A_OK);
    CHECK(ctx != nullptr);
    CHECK(p4a_context_set_seed(ctx, 42) == P4A_OK);
    uint64_t seed = 0;
    CHECK(p4a_context_get_seed(ctx, &seed) == P4A_OK);
    CHECK(seed == 42);
    CHECK(p4a_context_set_backend(ctx, P4A_BACKEND_CUDA) == P4A_ERR_BACKEND_UNAVAILABLE);

    // Config lifecycle + setters
    p4a_config_t* cfg = nullptr;
    CHECK(p4a_config_create(&cfg) == P4A_OK);
    CHECK(p4a_config_set_n_components(cfg, 2) == P4A_OK);
    CHECK(p4a_config_set_n_components(cfg, -1) == P4A_ERR_INVALID_ARGUMENT);
    int32_t nc = 0;
    CHECK(p4a_config_get_n_components(cfg, &nc) == P4A_OK);
    CHECK(nc == 2);

    // Matrix view + validate
    double xbuf[12] = {
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
        2.0, 2.0, 2.0,
        3.0, 5.0, 4.0,
    };
    double ybuf[4] = {0.1, 0.9, 6.2, 11.9};
    p4a_matrix_view_t X;
    p4a_matrix_view_t Y;
    CHECK(p4a_matrix_view_init_rowmajor(&X, xbuf, 4, 3, P4A_DTYPE_F64) == P4A_OK);
    CHECK(p4a_matrix_view_init_rowmajor(&Y, ybuf, 4, 1, P4A_DTYPE_F64) == P4A_OK);
    CHECK(p4a_matrix_view_validate(&X) == P4A_OK);
    CHECK(p4a_matrix_view_validate(&Y) == P4A_OK);

    // Model smoke: NIPALS fit, predict, transform and export.
    p4a_model_t* model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &model) == P4A_OK);
    CHECK(model != nullptr);
    p4a_array_t* pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, model, &X, &pred) == P4A_OK);
    int64_t rows = 0;
    int64_t cols = 0;
    CHECK(p4a_array_shape(pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(pred);

    p4a_array_t* scores = nullptr;
    CHECK(p4a_model_transform_alloc(ctx, model, &X, &scores) == P4A_OK);
    CHECK(p4a_array_shape(scores, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 2);
    p4a_array_free(scores);

    size_t export_size = 0;
    CHECK(p4a_model_export_size(model, &export_size) == P4A_OK);
    CHECK(export_size > 64);
    p4a_model_destroy(model);

    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_SIMPLS) == P4A_OK);
    p4a_model_t* simpls_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &simpls_model) == P4A_OK);
    CHECK(simpls_model != nullptr);
    p4a_array_t* simpls_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, simpls_model, &X, &simpls_pred) == P4A_OK);
    CHECK(p4a_array_shape(simpls_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(simpls_pred);
    p4a_model_destroy(simpls_model);

    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_KERNEL_ALGORITHM) == P4A_OK);
    p4a_model_t* kernel_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &kernel_model) == P4A_OK);
    CHECK(kernel_model != nullptr);
    p4a_array_t* kernel_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, kernel_model, &X, &kernel_pred) == P4A_OK);
    CHECK(p4a_array_shape(kernel_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(kernel_pred);
    p4a_model_destroy(kernel_model);

    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_SVD) == P4A_OK);
    p4a_model_t* svd_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &svd_model) == P4A_OK);
    CHECK(svd_model != nullptr);
    p4a_array_t* svd_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, svd_model, &X, &svd_pred) == P4A_OK);
    CHECK(p4a_array_shape(svd_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(svd_pred);
    p4a_model_destroy(svd_model);

    CHECK(p4a_config_set_algorithm(cfg, P4A_ALGO_PCR) == P4A_OK);
    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_SVD) == P4A_OK);
    p4a_model_t* pcr_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &pcr_model) == P4A_OK);
    CHECK(pcr_model != nullptr);
    p4a_array_t* pcr_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, pcr_model, &X, &pcr_pred) == P4A_OK);
    CHECK(p4a_array_shape(pcr_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(pcr_pred);
    p4a_model_destroy(pcr_model);

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
