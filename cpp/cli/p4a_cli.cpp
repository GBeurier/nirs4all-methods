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

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>

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
    double y_multi_buf[12] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
    };
    double x_msc_buf[20] = {
        1.35, 2.45, 4.65, 7.95, 12.35,
        0.47, 1.29, 2.93, 5.39, 8.67,
        2.25, 3.60, 6.30, 10.35, 15.75,
        1.07, 1.69, 2.93, 4.79, 7.27,
    };
    p4a_matrix_view_t X;
    p4a_matrix_view_t Y;
    p4a_matrix_view_t Y_multi;
    p4a_matrix_view_t X_msc;
    CHECK(p4a_matrix_view_init_rowmajor(&X, xbuf, 4, 3, P4A_DTYPE_F64) == P4A_OK);
    CHECK(p4a_matrix_view_init_rowmajor(&Y, ybuf, 4, 1, P4A_DTYPE_F64) == P4A_OK);
    CHECK(p4a_matrix_view_init_rowmajor(&Y_multi, y_multi_buf, 4, 3, P4A_DTYPE_F64) == P4A_OK);
    CHECK(p4a_matrix_view_init_rowmajor(&X_msc, x_msc_buf, 4, 5, P4A_DTYPE_F64) == P4A_OK);
    CHECK(p4a_matrix_view_validate(&X) == P4A_OK);
    CHECK(p4a_matrix_view_validate(&Y) == P4A_OK);
    CHECK(p4a_matrix_view_validate(&Y_multi) == P4A_OK);
    CHECK(p4a_matrix_view_validate(&X_msc) == P4A_OK);

    // Pipeline smoke: fit/transform for the Phase 3a preprocessing subset.
    p4a_pipeline_t* pipe = nullptr;
    CHECK(p4a_pipeline_create(&pipe) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(pipe, P4A_OP_CENTER, nullptr, 0) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(pipe, P4A_OP_AUTOSCALE, nullptr, 0) == P4A_OK);
    CHECK(p4a_pipeline_fit(ctx, pipe, &X, nullptr) == P4A_OK);
    p4a_array_t* x_preprocessed = nullptr;
    CHECK(p4a_pipeline_transform_alloc(ctx, pipe, &X, &x_preprocessed) == P4A_OK);
    int64_t pipe_rows = 0;
    int64_t pipe_cols = 0;
    CHECK(p4a_array_shape(x_preprocessed, &pipe_rows, &pipe_cols) == P4A_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 3);
    p4a_array_free(x_preprocessed);
    p4a_pipeline_destroy(pipe);

    p4a_pipeline_t* msc_pipe = nullptr;
    CHECK(p4a_pipeline_create(&msc_pipe) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(msc_pipe, P4A_OP_MSC, nullptr, 0) == P4A_OK);
    CHECK(p4a_pipeline_fit(ctx, msc_pipe, &X_msc, nullptr) == P4A_OK);
    p4a_array_t* x_msc = nullptr;
    CHECK(p4a_pipeline_transform_alloc(ctx, msc_pipe, &X_msc, &x_msc) == P4A_OK);
    CHECK(p4a_array_shape(x_msc, &pipe_rows, &pipe_cols) == P4A_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    p4a_array_free(x_msc);
    p4a_pipeline_destroy(msc_pipe);

    p4a_pipeline_t* emsc_pipe = nullptr;
    const double emsc_degree[] = {1.0};
    CHECK(p4a_pipeline_create(&emsc_pipe) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(emsc_pipe, P4A_OP_EMSC, emsc_degree, 1) == P4A_OK);
    CHECK(p4a_pipeline_fit(ctx, emsc_pipe, &X_msc, nullptr) == P4A_OK);
    p4a_array_t* x_emsc = nullptr;
    CHECK(p4a_pipeline_transform_alloc(ctx, emsc_pipe, &X_msc, &x_emsc) == P4A_OK);
    CHECK(p4a_array_shape(x_emsc, &pipe_rows, &pipe_cols) == P4A_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    p4a_array_free(x_emsc);
    p4a_pipeline_destroy(emsc_pipe);

    p4a_pipeline_t* detrend_pipe = nullptr;
    const double detrend_degree[] = {2.0};
    CHECK(p4a_pipeline_create(&detrend_pipe) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(detrend_pipe, P4A_OP_DETREND_POLY,
                                    detrend_degree, 1) == P4A_OK);
    CHECK(p4a_pipeline_fit(ctx, detrend_pipe, &X_msc, nullptr) == P4A_OK);
    p4a_array_t* x_detrended = nullptr;
    CHECK(p4a_pipeline_transform_alloc(ctx, detrend_pipe, &X_msc, &x_detrended) == P4A_OK);
    CHECK(p4a_array_shape(x_detrended, &pipe_rows, &pipe_cols) == P4A_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    p4a_array_free(x_detrended);
    p4a_pipeline_destroy(detrend_pipe);

    p4a_pipeline_t* savgol_pipe = nullptr;
    const double savgol_smooth_params[] = {5.0, 2.0};
    const double savgol_derivative_params[] = {5.0, 2.0, 1.0, 1.0};
    CHECK(p4a_pipeline_create(&savgol_pipe) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(savgol_pipe, P4A_OP_SAVGOL_SMOOTH,
                                    savgol_smooth_params, 2) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(savgol_pipe, P4A_OP_SAVGOL_DERIVATIVE,
                                    savgol_derivative_params, 4) == P4A_OK);
    CHECK(p4a_pipeline_fit(ctx, savgol_pipe, &X_msc, nullptr) == P4A_OK);
    p4a_array_t* x_savgol = nullptr;
    CHECK(p4a_pipeline_transform_alloc(ctx, savgol_pipe, &X_msc, &x_savgol) == P4A_OK);
    CHECK(p4a_array_shape(x_savgol, &pipe_rows, &pipe_cols) == P4A_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    p4a_array_free(x_savgol);
    p4a_pipeline_destroy(savgol_pipe);

    p4a_pipeline_t* asls_pipe = nullptr;
    const double asls_params[] = {10000.0, 0.01, 5.0};
    CHECK(p4a_pipeline_create(&asls_pipe) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(asls_pipe, P4A_OP_ASLS_BASELINE,
                                    asls_params, 3) == P4A_OK);
    CHECK(p4a_pipeline_fit(ctx, asls_pipe, &X_msc, nullptr) == P4A_OK);
    p4a_array_t* x_asls = nullptr;
    CHECK(p4a_pipeline_transform_alloc(ctx, asls_pipe, &X_msc, &x_asls) == P4A_OK);
    CHECK(p4a_array_shape(x_asls, &pipe_rows, &pipe_cols) == P4A_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    p4a_array_free(x_asls);
    p4a_pipeline_destroy(asls_pipe);

    p4a_pipeline_t* norris_pipe = nullptr;
    const double norris_params[] = {5.0, 3.0, 1.0};
    CHECK(p4a_pipeline_create(&norris_pipe) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(norris_pipe, P4A_OP_NORRIS_WILLIAMS,
                                    norris_params, 3) == P4A_OK);
    CHECK(p4a_pipeline_fit(ctx, norris_pipe, &X_msc, nullptr) == P4A_OK);
    p4a_array_t* x_norris = nullptr;
    CHECK(p4a_pipeline_transform_alloc(ctx, norris_pipe, &X_msc, &x_norris) == P4A_OK);
    CHECK(p4a_array_shape(x_norris, &pipe_rows, &pipe_cols) == P4A_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    p4a_array_free(x_norris);
    p4a_pipeline_destroy(norris_pipe);

    p4a_pipeline_t* wavelet_pipe = nullptr;
    const double wavelet_params[] = {1.0, 0.01};
    CHECK(p4a_pipeline_create(&wavelet_pipe) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(wavelet_pipe, P4A_OP_WAVELET_DENOISE,
                                    wavelet_params, 2) == P4A_OK);
    CHECK(p4a_pipeline_fit(ctx, wavelet_pipe, &X_msc, nullptr) == P4A_OK);
    p4a_array_t* x_wavelet = nullptr;
    CHECK(p4a_pipeline_transform_alloc(ctx, wavelet_pipe, &X_msc, &x_wavelet) == P4A_OK);
    CHECK(p4a_array_shape(x_wavelet, &pipe_rows, &pipe_cols) == P4A_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    p4a_array_free(x_wavelet);
    p4a_pipeline_destroy(wavelet_pipe);

    p4a_pipeline_t* osc_pipe = nullptr;
    const double osc_params[] = {100.0, 1e-10};
    CHECK(p4a_pipeline_create(&osc_pipe) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(osc_pipe, P4A_OP_OSC,
                                    osc_params, 2) == P4A_OK);
    CHECK(p4a_pipeline_fit(ctx, osc_pipe, &X_msc, &Y) == P4A_OK);
    p4a_array_t* x_osc = nullptr;
    CHECK(p4a_pipeline_transform_alloc(ctx, osc_pipe, &X_msc, &x_osc) == P4A_OK);
    CHECK(p4a_array_shape(x_osc, &pipe_rows, &pipe_cols) == P4A_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    p4a_array_free(x_osc);
    p4a_pipeline_destroy(osc_pipe);

    p4a_pipeline_t* epo_pipe = nullptr;
    const double epo_params[] = {100.0, 1e-10};
    CHECK(p4a_pipeline_create(&epo_pipe) == P4A_OK);
    CHECK(p4a_pipeline_add_operator(epo_pipe, P4A_OP_EPO,
                                    epo_params, 2) == P4A_OK);
    CHECK(p4a_pipeline_fit(ctx, epo_pipe, &X_msc, &Y) == P4A_OK);
    p4a_array_t* x_epo = nullptr;
    CHECK(p4a_pipeline_transform_alloc(ctx, epo_pipe, &X_msc, &x_epo) == P4A_OK);
    CHECK(p4a_array_shape(x_epo, &pipe_rows, &pipe_cols) == P4A_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    p4a_array_free(x_epo);
    p4a_pipeline_destroy(epo_pipe);

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

    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_ORTHOGONAL_SCORES) == P4A_OK);
    p4a_model_t* oscores_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &oscores_model) == P4A_OK);
    CHECK(oscores_model != nullptr);
    p4a_array_t* oscores_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, oscores_model, &X, &oscores_pred) == P4A_OK);
    CHECK(p4a_array_shape(oscores_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(oscores_pred);
    p4a_model_destroy(oscores_model);

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

    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_WIDE_KERNEL) == P4A_OK);
    p4a_model_t* wide_kernel_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &wide_kernel_model) == P4A_OK);
    CHECK(wide_kernel_model != nullptr);
    p4a_array_t* wide_kernel_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, wide_kernel_model, &X, &wide_kernel_pred) == P4A_OK);
    CHECK(p4a_array_shape(wide_kernel_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(wide_kernel_pred);
    p4a_model_destroy(wide_kernel_model);

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

    CHECK(p4a_config_set_algorithm(cfg, P4A_ALGO_PLS_SVD) == P4A_OK);
    CHECK(p4a_config_set_deflation(cfg, P4A_DEFLATION_CANONICAL) == P4A_OK);
    CHECK(p4a_config_set_n_components(cfg, 2) == P4A_OK);
    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_SVD) == P4A_OK);
    p4a_model_t* pls_svd_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y_multi, &pls_svd_model) == P4A_OK);
    CHECK(pls_svd_model != nullptr);
    p4a_array_t* pls_svd_scores = nullptr;
    CHECK(p4a_model_transform_alloc(ctx, pls_svd_model, &X, &pls_svd_scores) == P4A_OK);
    CHECK(p4a_array_shape(pls_svd_scores, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 2);
    p4a_array_free(pls_svd_scores);
    p4a_model_destroy(pls_svd_model);

    CHECK(p4a_config_set_algorithm(cfg, P4A_ALGO_PLS_REGRESSION) == P4A_OK);
    CHECK(p4a_config_set_deflation(cfg, P4A_DEFLATION_REGRESSION) == P4A_OK);
    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_POWER) == P4A_OK);
    p4a_model_t* power_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &power_model) == P4A_OK);
    CHECK(power_model != nullptr);
    p4a_array_t* power_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, power_model, &X, &power_pred) == P4A_OK);
    CHECK(p4a_array_shape(power_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(power_pred);
    p4a_model_destroy(power_model);

    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_RANDOMIZED_SVD) == P4A_OK);
    p4a_model_t* randomized_svd_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &randomized_svd_model) == P4A_OK);
    CHECK(randomized_svd_model != nullptr);
    p4a_array_t* randomized_svd_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, randomized_svd_model, &X, &randomized_svd_pred) == P4A_OK);
    CHECK(p4a_array_shape(randomized_svd_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(randomized_svd_pred);
    p4a_model_destroy(randomized_svd_model);

    CHECK(p4a_config_set_algorithm(cfg, P4A_ALGO_PLS_DA) == P4A_OK);
    CHECK(p4a_config_set_deflation(cfg, P4A_DEFLATION_REGRESSION) == P4A_OK);
    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_NIPALS) == P4A_OK);
    p4a_model_t* pls_da_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &pls_da_model) == P4A_OK);
    CHECK(pls_da_model != nullptr);
    p4a_array_t* pls_da_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, pls_da_model, &X, &pls_da_pred) == P4A_OK);
    CHECK(p4a_array_shape(pls_da_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(pls_da_pred);
    p4a_model_destroy(pls_da_model);

    CHECK(p4a_config_set_algorithm(cfg, P4A_ALGO_OPLS) == P4A_OK);
    CHECK(p4a_config_set_deflation(cfg, P4A_DEFLATION_ORTHOGONAL) == P4A_OK);
    CHECK(p4a_config_set_n_components(cfg, 2) == P4A_OK);
    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_NIPALS) == P4A_OK);
    p4a_model_t* opls_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &opls_model) == P4A_OK);
    CHECK(opls_model != nullptr);
    p4a_array_t* opls_scores = nullptr;
    CHECK(p4a_model_transform_alloc(ctx, opls_model, &X, &opls_scores) == P4A_OK);
    CHECK(p4a_array_shape(opls_scores, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 2);
    p4a_array_free(opls_scores);
    p4a_model_destroy(opls_model);

    CHECK(p4a_config_set_algorithm(cfg, P4A_ALGO_OPLS_DA) == P4A_OK);
    CHECK(p4a_config_set_deflation(cfg, P4A_DEFLATION_ORTHOGONAL) == P4A_OK);
    CHECK(p4a_config_set_n_components(cfg, 2) == P4A_OK);
    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_NIPALS) == P4A_OK);
    p4a_model_t* opls_da_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &opls_da_model) == P4A_OK);
    CHECK(opls_da_model != nullptr);
    p4a_array_t* opls_da_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, opls_da_model, &X, &opls_da_pred) == P4A_OK);
    CHECK(p4a_array_shape(opls_da_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(opls_da_pred);
    p4a_model_destroy(opls_da_model);

    p4a_model_t* opls_da_multi_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y_multi, &opls_da_multi_model) == P4A_OK);
    CHECK(opls_da_multi_model != nullptr);
    p4a_array_t* opls_da_multi_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, opls_da_multi_model, &X, &opls_da_multi_pred) == P4A_OK);
    CHECK(p4a_array_shape(opls_da_multi_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 3);
    p4a_array_free(opls_da_multi_pred);
    p4a_model_destroy(opls_da_multi_model);

    CHECK(p4a_config_set_algorithm(cfg, P4A_ALGO_PLS_CANONICAL) == P4A_OK);
    CHECK(p4a_config_set_deflation(cfg, P4A_DEFLATION_CANONICAL) == P4A_OK);
    CHECK(p4a_config_set_n_components(cfg, 1) == P4A_OK);
    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_NIPALS) == P4A_OK);
    p4a_model_t* canonical_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &canonical_model) == P4A_OK);
    CHECK(canonical_model != nullptr);
    p4a_array_t* canonical_pred = nullptr;
    CHECK(p4a_model_predict_alloc(ctx, canonical_model, &X, &canonical_pred) == P4A_OK);
    CHECK(p4a_array_shape(canonical_pred, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(canonical_pred);
    p4a_model_destroy(canonical_model);

    CHECK(p4a_config_set_solver(cfg, P4A_SOLVER_SVD) == P4A_OK);
    p4a_model_t* canonical_svd_model = nullptr;
    CHECK(p4a_model_fit(ctx, cfg, &X, &Y, &canonical_svd_model) == P4A_OK);
    CHECK(canonical_svd_model != nullptr);
    p4a_array_t* canonical_svd_scores = nullptr;
    CHECK(p4a_model_transform_alloc(ctx, canonical_svd_model, &X, &canonical_svd_scores) == P4A_OK);
    CHECK(p4a_array_shape(canonical_svd_scores, &rows, &cols) == P4A_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    p4a_array_free(canonical_svd_scores);
    p4a_model_destroy(canonical_svd_model);

    CHECK(p4a_config_set_algorithm(cfg, P4A_ALGO_PCR) == P4A_OK);
    CHECK(p4a_config_set_deflation(cfg, P4A_DEFLATION_REGRESSION) == P4A_OK);
    CHECK(p4a_config_set_n_components(cfg, 2) == P4A_OK);
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

// SplitMix64 — minimal deterministic PRNG so the bench command does not
// pull in <random>. Matches the seeding contract documented for the bench
// runner.
struct SplitMix64 {
    uint64_t state;
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double next_double() {
        // Convert top 53 bits to [0, 1).
        return static_cast<double>(next() >> 11) * (1.0 / (1ULL << 53));
    }
    double standard_normal() {
        // Box-Muller. One sample at a time; we only need ~n*p samples for
        // benchmark fixtures and speed is not the concern here.
        double u1 = next_double();
        double u2 = next_double();
        if (u1 < 1e-300) u1 = 1e-300;
        const double radius = sqrt(-2.0 * log(u1));
        const double theta = 6.28318530717958647692 * u2;
        return radius * cos(theta);
    }
};

struct BenchOptions {
    const char* algo = nullptr;
    int64_t n_samples = 0;
    int64_t n_features = 0;
    int32_t n_components = 0;
    int32_t repeats = 5;
    uint64_t seed = 42;
};

bool resolve_algo(const char* name, p4a_algorithm_t& algo,
                  p4a_solver_t& solver, p4a_deflation_t& deflation) {
    deflation = P4A_DEFLATION_REGRESSION;
    if (strcmp(name, "pls_nipals") == 0) {
        algo = P4A_ALGO_PLS_REGRESSION;
        solver = P4A_SOLVER_NIPALS;
        return true;
    }
    if (strcmp(name, "pls_orthogonal_scores") == 0) {
        algo = P4A_ALGO_PLS_REGRESSION;
        solver = P4A_SOLVER_ORTHOGONAL_SCORES;
        return true;
    }
    if (strcmp(name, "pls_simpls") == 0) {
        algo = P4A_ALGO_PLS_REGRESSION;
        solver = P4A_SOLVER_SIMPLS;
        return true;
    }
    if (strcmp(name, "pls_kernel_algorithm") == 0) {
        algo = P4A_ALGO_PLS_REGRESSION;
        solver = P4A_SOLVER_KERNEL_ALGORITHM;
        return true;
    }
    if (strcmp(name, "pls_wide_kernel") == 0) {
        algo = P4A_ALGO_PLS_REGRESSION;
        solver = P4A_SOLVER_WIDE_KERNEL;
        return true;
    }
    if (strcmp(name, "pls_svd") == 0) {
        algo = P4A_ALGO_PLS_REGRESSION;
        solver = P4A_SOLVER_SVD;
        return true;
    }
    if (strcmp(name, "pls_power") == 0) {
        algo = P4A_ALGO_PLS_REGRESSION;
        solver = P4A_SOLVER_POWER;
        return true;
    }
    if (strcmp(name, "pls_randomized_svd") == 0) {
        algo = P4A_ALGO_PLS_REGRESSION;
        solver = P4A_SOLVER_RANDOMIZED_SVD;
        return true;
    }
    if (strcmp(name, "pcr_svd") == 0) {
        algo = P4A_ALGO_PCR;
        solver = P4A_SOLVER_SVD;
        return true;
    }
    return false;
}

int cmd_bench(int argc, char** argv) {
    BenchOptions opt;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
            opt.algo = argv[++i];
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            opt.n_samples = strtoll(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--features") == 0 && i + 1 < argc) {
            opt.n_features = strtoll(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--components") == 0 && i + 1 < argc) {
            opt.n_components = static_cast<int32_t>(strtol(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            opt.repeats = static_cast<int32_t>(strtol(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            opt.seed = static_cast<uint64_t>(strtoull(argv[++i], nullptr, 10));
        } else {
            fprintf(stderr, "unknown --bench argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (opt.algo == nullptr || opt.n_samples <= 0 || opt.n_features <= 0) {
        fprintf(stderr,
            "Usage: pls4all_cli --bench --algo <name> --samples N --features P "
            "[--components K] [--repeats R] [--seed S]\n"
            "Available algos: pls_nipals, pls_orthogonal_scores, pls_simpls,\n"
            "  pls_kernel_algorithm, pls_wide_kernel, pls_svd, pls_power,\n"
            "  pls_randomized_svd, pcr_svd\n");
        return 2;
    }
    if (opt.n_components <= 0) {
        const int64_t cap = (opt.n_samples - 1 < opt.n_features)
            ? (opt.n_samples - 1) : opt.n_features;
        opt.n_components = static_cast<int32_t>((cap < 10) ? cap : 10);
    }
    if (opt.repeats <= 0) opt.repeats = 1;

    p4a_algorithm_t algo = P4A_ALGO_PLS_REGRESSION;
    p4a_solver_t solver = P4A_SOLVER_SIMPLS;
    p4a_deflation_t deflation = P4A_DEFLATION_REGRESSION;
    if (!resolve_algo(opt.algo, algo, solver, deflation)) {
        fprintf(stderr, "unknown algo: %s\n", opt.algo);
        return 2;
    }

    const size_t n = static_cast<size_t>(opt.n_samples);
    const size_t p = static_cast<size_t>(opt.n_features);
    const size_t total = n * p;
    double* xbuf = static_cast<double*>(malloc(total * sizeof(double)));
    double* ybuf = static_cast<double*>(malloc(n * sizeof(double)));
    double* predbuf = static_cast<double*>(malloc(n * sizeof(double)));
    if (xbuf == nullptr || ybuf == nullptr || predbuf == nullptr) {
        free(xbuf); free(ybuf); free(predbuf);
        fprintf(stderr, "out of memory generating %zux%zu dataset\n", n, p);
        return 1;
    }

    // Synthesize a latent-factor dataset deterministically.
    {
        SplitMix64 rng_t{opt.seed};
        SplitMix64 rng_w{opt.seed + 1ULL};
        SplitMix64 rng_b{opt.seed + 2ULL};
        SplitMix64 rng_n{opt.seed + 3ULL};
        const int32_t k = opt.n_components;
        double* T = static_cast<double*>(malloc(n * static_cast<size_t>(k) * sizeof(double)));
        double* W = static_cast<double*>(malloc(p * static_cast<size_t>(k) * sizeof(double)));
        double* bvec = static_cast<double*>(malloc(static_cast<size_t>(k) * sizeof(double)));
        if (T == nullptr || W == nullptr || bvec == nullptr) {
            free(T); free(W); free(bvec);
            free(xbuf); free(ybuf); free(predbuf);
            fprintf(stderr, "out of memory generating latent factors\n");
            return 1;
        }
        const size_t k_sz = static_cast<size_t>(k);
        for (size_t i = 0; i < n * k_sz; ++i) T[i] = rng_t.standard_normal();
        for (size_t i = 0; i < p * k_sz; ++i) W[i] = rng_w.standard_normal();
        for (size_t i = 0; i < k_sz; ++i) bvec[i] = rng_b.standard_normal();
        for (size_t row = 0; row < n; ++row) {
            for (size_t col = 0; col < p; ++col) {
                double value = 0.05 * rng_n.standard_normal();
                for (size_t c = 0; c < k_sz; ++c) {
                    value += T[row * k_sz + c] * W[col * k_sz + c];
                }
                xbuf[row * p + col] = value;
            }
            double yval = 0.02 * rng_n.standard_normal();
            for (size_t c = 0; c < k_sz; ++c) {
                yval += T[row * k_sz + c] * bvec[c];
            }
            ybuf[row] = yval;
        }
        free(T); free(W); free(bvec);
    }

    // Build matrix views.
    p4a_matrix_view_t X;
    p4a_matrix_view_t Y;
    p4a_matrix_view_init_rowmajor(&X, xbuf, static_cast<int64_t>(n),
                                  static_cast<int64_t>(p), P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&Y, ybuf, static_cast<int64_t>(n), 1,
                                  P4A_DTYPE_F64);
    p4a_matrix_view_t Pred;
    p4a_matrix_view_init_rowmajor(&Pred, predbuf, static_cast<int64_t>(n), 1,
                                  P4A_DTYPE_F64);

    p4a_context_t* ctx = nullptr;
    if (p4a_context_create(&ctx) != P4A_OK) {
        free(xbuf); free(ybuf); free(predbuf);
        return 1;
    }
    p4a_config_t* cfg = nullptr;
    if (p4a_config_create(&cfg) != P4A_OK) {
        p4a_context_destroy(ctx);
        free(xbuf); free(ybuf); free(predbuf);
        return 1;
    }
    p4a_config_set_algorithm(cfg, algo);
    p4a_config_set_solver(cfg, solver);
    p4a_config_set_deflation(cfg, deflation);
    p4a_config_set_n_components(cfg, opt.n_components);
    p4a_config_set_center_x(cfg, 1);
    p4a_config_set_scale_x(cfg, 0);
    p4a_config_set_center_y(cfg, 1);
    p4a_config_set_scale_y(cfg, 0);
    p4a_config_set_store_scores(cfg, 0);

    // Run repeats and report median + min.
    double* fit_ms = static_cast<double*>(malloc(static_cast<size_t>(opt.repeats) * sizeof(double)));
    double* predict_ms = static_cast<double*>(malloc(static_cast<size_t>(opt.repeats) * sizeof(double)));
    if (fit_ms == nullptr || predict_ms == nullptr) {
        free(fit_ms); free(predict_ms);
        p4a_config_destroy(cfg);
        p4a_context_destroy(ctx);
        free(xbuf); free(ybuf); free(predbuf);
        return 1;
    }
    p4a_status_t last_status = P4A_OK;
    double last_rmse = 0.0;
    for (int32_t r = 0; r < opt.repeats; ++r) {
        const auto fit_t0 = std::chrono::steady_clock::now();
        p4a_model_t* model = nullptr;
        last_status = p4a_model_fit(ctx, cfg, &X, &Y, &model);
        const auto fit_t1 = std::chrono::steady_clock::now();
        if (last_status != P4A_OK) {
            fit_ms[r] = NAN; predict_ms[r] = NAN;
            continue;
        }
        last_status = p4a_model_predict(ctx, model, &X, &Pred);
        const auto pred_t1 = std::chrono::steady_clock::now();
        fit_ms[r] = std::chrono::duration<double, std::milli>(fit_t1 - fit_t0).count();
        predict_ms[r] = std::chrono::duration<double, std::milli>(pred_t1 - fit_t1).count();
        if (r == opt.repeats - 1) {
            double sumsq = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double diff = predbuf[i] - ybuf[i];
                sumsq += diff * diff;
            }
            last_rmse = sqrt(sumsq / static_cast<double>(n));
        }
        p4a_model_destroy(model);
    }

    // Median + min helper.
    auto median = [](double* arr, int32_t count) {
        // Tiny insertion sort — count is small.
        for (int32_t i = 1; i < count; ++i) {
            double cur = arr[i]; int32_t j = i - 1;
            while (j >= 0 && arr[j] > cur) { arr[j + 1] = arr[j]; --j; }
            arr[j + 1] = cur;
        }
        if (count == 0) return 0.0;
        return (count % 2 == 1) ? arr[count / 2]
            : 0.5 * (arr[count / 2 - 1] + arr[count / 2]);
    };
    auto min_of = [](const double* arr, int32_t count) {
        if (count == 0) return 0.0;
        double m = arr[0];
        for (int32_t i = 1; i < count; ++i) if (arr[i] < m) m = arr[i];
        return m;
    };
    const double fit_min = min_of(fit_ms, opt.repeats);
    const double predict_min = min_of(predict_ms, opt.repeats);
    const double fit_med = median(fit_ms, opt.repeats);
    const double predict_med = median(predict_ms, opt.repeats);

    // Print one CSV row to stdout. Header is also printed for ergonomic
    // human reads; the bench matrix runner can grep for "DATA," lines.
    printf("HEAD,algo,n_samples,n_features,n_components,repeats,seed,"
           "status,fit_ms_median,predict_ms_median,fit_ms_min,predict_ms_min,"
           "rmse_last\n");
    printf("DATA,%s,%lld,%lld,%d,%d,%llu,%s,%.6f,%.6f,%.6f,%.6f,%.6e\n",
           opt.algo,
           static_cast<long long>(opt.n_samples),
           static_cast<long long>(opt.n_features),
           static_cast<int>(opt.n_components),
           static_cast<int>(opt.repeats),
           static_cast<unsigned long long>(opt.seed),
           (last_status == P4A_OK) ? "ok" : p4a_status_to_string(last_status),
           fit_med, predict_med, fit_min, predict_min, last_rmse);

    free(fit_ms);
    free(predict_ms);
    p4a_config_destroy(cfg);
    p4a_context_destroy(ctx);
    free(xbuf); free(ybuf); free(predbuf);
    return (last_status == P4A_OK) ? 0 : 1;
}

int usage(const char* argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --version\n"
        "  %s --abi-info\n"
        "  %s --selfcheck\n"
        "  %s --bench --algo NAME --samples N --features P "
        "[--components K] [--repeats R] [--seed S]\n",
        argv0, argv0, argv0, argv0);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage(argv[0]);
    if (strcmp(argv[1], "--version")  == 0) return cmd_version();
    if (strcmp(argv[1], "--abi-info") == 0) return cmd_abi_info();
    if (strcmp(argv[1], "--selfcheck")== 0) return cmd_selfcheck();
    if (strcmp(argv[1], "--bench")    == 0) return cmd_bench(argc, argv);
    return usage(argv[0]);
}
