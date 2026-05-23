// SPDX-License-Identifier: CECILL-2.1
//
// n4m_cli — tiny introspection tool. Supported subcommands:
//   --version      print "pls4all <project> (ABI <abi> · git <rev>)"
//   --abi-info     print all ABI / backend / dtype metadata
//   --selfcheck    exercise context + config + matrix_view lifecycle and
//                   exit non-zero on failed ABI / model smoke checks
//
// No third-party CLI parser is used — keep the executable's
// runtime dependencies aligned with libn4m's zero-mandatory-deps promise.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <limits>

#include "n4m/n4m.h"

namespace {

int cmd_version() {
    printf("pls4all %s\n", n4m_get_version_string());
    const char* rev = n4m_get_git_revision();
    if (rev && rev[0] != '\0') {
        printf("git: %s\n", rev);
    }
    return 0;
}

int cmd_abi_info() {
    printf("abi-version: %u.%u.%u (int %u)\n",
        n4m_get_abi_version_major(),
        n4m_get_abi_version_minor(),
        n4m_get_abi_version_patch(),
        n4m_get_abi_version_int());
    printf("project-version: %s\n", n4m_get_version_string());
    const char* build = n4m_get_build_info();
    if (build && build[0] != '\0') {
        printf("build-info: %s\n", build);
    }

    printf("backends:\n");
    const n4m_backend_t backends[] = {
        N4M_BACKEND_AUTO, N4M_BACKEND_REFERENCE_CPU, N4M_BACKEND_NATIVE_CPU,
        N4M_BACKEND_BLAS, N4M_BACKEND_OPENMP, N4M_BACKEND_CUDA,
        N4M_BACKEND_OPENCL, N4M_BACKEND_METAL,
    };
    for (size_t i = 0; i < sizeof(backends) / sizeof(backends[0]); ++i) {
        printf("  %s: %s\n", n4m_backend_to_string(backends[i]),
               n4m_backend_is_available(backends[i]) ? "available" : "unavailable");
    }

    printf("dtypes:\n");
    const n4m_dtype_t dtypes[] = {
        N4M_DTYPE_F64, N4M_DTYPE_F32, N4M_DTYPE_I32, N4M_DTYPE_I64,
    };
    for (size_t i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); ++i) {
        printf("  %s: %llu bytes\n", n4m_dtype_to_string(dtypes[i]),
               static_cast<unsigned long long>(n4m_dtype_size(dtypes[i])));
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
    CHECK(n4m_check_abi_compatibility(n4m_get_abi_version_major(),
                                       n4m_get_abi_version_minor()) == N4M_OK);
    CHECK(n4m_check_abi_compatibility(n4m_get_abi_version_major() + 1, 0) ==
          N4M_ERR_ABI_MISMATCH);

    // Status string
    CHECK(strcmp(n4m_status_to_string(N4M_OK), "ok") == 0);
    CHECK(strcmp(n4m_status_to_string(N4M_ERR_NOT_IMPLEMENTED), "not implemented") == 0);

    // Dtype
    CHECK(n4m_dtype_size(N4M_DTYPE_F64) == 8);
    CHECK(n4m_dtype_size(N4M_DTYPE_F32) == 4);

    // Context lifecycle
    n4m_context_t* ctx = nullptr;
    CHECK(n4m_context_create(&ctx) == N4M_OK);
    CHECK(ctx != nullptr);
    CHECK(n4m_context_set_seed(ctx, 42) == N4M_OK);
    uint64_t seed = 0;
    CHECK(n4m_context_get_seed(ctx, &seed) == N4M_OK);
    CHECK(seed == 42);
    {
        const n4m_status_t cuda_status =
            n4m_context_set_backend(ctx, N4M_BACKEND_CUDA);
#if defined(N4M_USE_CUDA)
        // CUDA may be compiled in but the runtime may have no GPU.
        CHECK(cuda_status == N4M_OK ||
              cuda_status == N4M_ERR_BACKEND_UNAVAILABLE);
#else
        CHECK(cuda_status == N4M_ERR_BACKEND_UNAVAILABLE);
#endif
    }

    // Config lifecycle + setters
    n4m_config_t* cfg = nullptr;
    CHECK(n4m_config_create(&cfg) == N4M_OK);
    CHECK(n4m_config_set_n_components(cfg, 2) == N4M_OK);
    CHECK(n4m_config_set_n_components(cfg, -1) == N4M_ERR_INVALID_ARGUMENT);
    int32_t nc = 0;
    CHECK(n4m_config_get_n_components(cfg, &nc) == N4M_OK);
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
    n4m_matrix_view_t X;
    n4m_matrix_view_t Y;
    n4m_matrix_view_t Y_multi;
    n4m_matrix_view_t X_msc;
    CHECK(n4m_matrix_view_init_rowmajor(&X, xbuf, 4, 3, N4M_DTYPE_F64) == N4M_OK);
    CHECK(n4m_matrix_view_init_rowmajor(&Y, ybuf, 4, 1, N4M_DTYPE_F64) == N4M_OK);
    CHECK(n4m_matrix_view_init_rowmajor(&Y_multi, y_multi_buf, 4, 3, N4M_DTYPE_F64) == N4M_OK);
    CHECK(n4m_matrix_view_init_rowmajor(&X_msc, x_msc_buf, 4, 5, N4M_DTYPE_F64) == N4M_OK);
    CHECK(n4m_matrix_view_validate(&X) == N4M_OK);
    CHECK(n4m_matrix_view_validate(&Y) == N4M_OK);
    CHECK(n4m_matrix_view_validate(&Y_multi) == N4M_OK);
    CHECK(n4m_matrix_view_validate(&X_msc) == N4M_OK);

    // Pipeline smoke: fit/transform for the Phase 3a preprocessing subset.
    n4m_pipeline_t* pipe = nullptr;
    CHECK(n4m_pipeline_create(&pipe) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(pipe, N4M_OP_CENTER, nullptr, 0) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(pipe, N4M_OP_AUTOSCALE, nullptr, 0) == N4M_OK);
    CHECK(n4m_pipeline_fit(ctx, pipe, &X, nullptr) == N4M_OK);
    n4m_array_t* x_preprocessed = nullptr;
    CHECK(n4m_pipeline_transform_alloc(ctx, pipe, &X, &x_preprocessed) == N4M_OK);
    int64_t pipe_rows = 0;
    int64_t pipe_cols = 0;
    CHECK(n4m_array_shape(x_preprocessed, &pipe_rows, &pipe_cols) == N4M_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 3);
    n4m_array_free(x_preprocessed);
    n4m_pipeline_destroy(pipe);

    n4m_pipeline_t* msc_pipe = nullptr;
    CHECK(n4m_pipeline_create(&msc_pipe) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(msc_pipe, N4M_OP_MSC, nullptr, 0) == N4M_OK);
    CHECK(n4m_pipeline_fit(ctx, msc_pipe, &X_msc, nullptr) == N4M_OK);
    n4m_array_t* x_msc = nullptr;
    CHECK(n4m_pipeline_transform_alloc(ctx, msc_pipe, &X_msc, &x_msc) == N4M_OK);
    CHECK(n4m_array_shape(x_msc, &pipe_rows, &pipe_cols) == N4M_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    n4m_array_free(x_msc);
    n4m_pipeline_destroy(msc_pipe);

    n4m_pipeline_t* emsc_pipe = nullptr;
    const double emsc_degree[] = {1.0};
    CHECK(n4m_pipeline_create(&emsc_pipe) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(emsc_pipe, N4M_OP_EMSC, emsc_degree, 1) == N4M_OK);
    CHECK(n4m_pipeline_fit(ctx, emsc_pipe, &X_msc, nullptr) == N4M_OK);
    n4m_array_t* x_emsc = nullptr;
    CHECK(n4m_pipeline_transform_alloc(ctx, emsc_pipe, &X_msc, &x_emsc) == N4M_OK);
    CHECK(n4m_array_shape(x_emsc, &pipe_rows, &pipe_cols) == N4M_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    n4m_array_free(x_emsc);
    n4m_pipeline_destroy(emsc_pipe);

    n4m_pipeline_t* detrend_pipe = nullptr;
    const double detrend_degree[] = {2.0};
    CHECK(n4m_pipeline_create(&detrend_pipe) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(detrend_pipe, N4M_OP_DETREND_POLY,
                                    detrend_degree, 1) == N4M_OK);
    CHECK(n4m_pipeline_fit(ctx, detrend_pipe, &X_msc, nullptr) == N4M_OK);
    n4m_array_t* x_detrended = nullptr;
    CHECK(n4m_pipeline_transform_alloc(ctx, detrend_pipe, &X_msc, &x_detrended) == N4M_OK);
    CHECK(n4m_array_shape(x_detrended, &pipe_rows, &pipe_cols) == N4M_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    n4m_array_free(x_detrended);
    n4m_pipeline_destroy(detrend_pipe);

    n4m_pipeline_t* savgol_pipe = nullptr;
    const double savgol_smooth_params[] = {5.0, 2.0};
    const double savgol_derivative_params[] = {5.0, 2.0, 1.0, 1.0};
    CHECK(n4m_pipeline_create(&savgol_pipe) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(savgol_pipe, N4M_OP_SAVGOL_SMOOTH,
                                    savgol_smooth_params, 2) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(savgol_pipe, N4M_OP_SAVGOL_DERIVATIVE,
                                    savgol_derivative_params, 4) == N4M_OK);
    CHECK(n4m_pipeline_fit(ctx, savgol_pipe, &X_msc, nullptr) == N4M_OK);
    n4m_array_t* x_savgol = nullptr;
    CHECK(n4m_pipeline_transform_alloc(ctx, savgol_pipe, &X_msc, &x_savgol) == N4M_OK);
    CHECK(n4m_array_shape(x_savgol, &pipe_rows, &pipe_cols) == N4M_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    n4m_array_free(x_savgol);
    n4m_pipeline_destroy(savgol_pipe);

    n4m_pipeline_t* asls_pipe = nullptr;
    const double asls_params[] = {10000.0, 0.01, 5.0};
    CHECK(n4m_pipeline_create(&asls_pipe) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(asls_pipe, N4M_OP_ASLS_BASELINE,
                                    asls_params, 3) == N4M_OK);
    CHECK(n4m_pipeline_fit(ctx, asls_pipe, &X_msc, nullptr) == N4M_OK);
    n4m_array_t* x_asls = nullptr;
    CHECK(n4m_pipeline_transform_alloc(ctx, asls_pipe, &X_msc, &x_asls) == N4M_OK);
    CHECK(n4m_array_shape(x_asls, &pipe_rows, &pipe_cols) == N4M_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    n4m_array_free(x_asls);
    n4m_pipeline_destroy(asls_pipe);

    n4m_pipeline_t* norris_pipe = nullptr;
    const double norris_params[] = {5.0, 3.0, 1.0};
    CHECK(n4m_pipeline_create(&norris_pipe) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(norris_pipe, N4M_OP_NORRIS_WILLIAMS,
                                    norris_params, 3) == N4M_OK);
    CHECK(n4m_pipeline_fit(ctx, norris_pipe, &X_msc, nullptr) == N4M_OK);
    n4m_array_t* x_norris = nullptr;
    CHECK(n4m_pipeline_transform_alloc(ctx, norris_pipe, &X_msc, &x_norris) == N4M_OK);
    CHECK(n4m_array_shape(x_norris, &pipe_rows, &pipe_cols) == N4M_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    n4m_array_free(x_norris);
    n4m_pipeline_destroy(norris_pipe);

    n4m_pipeline_t* wavelet_pipe = nullptr;
    const double wavelet_params[] = {1.0, 0.01};
    CHECK(n4m_pipeline_create(&wavelet_pipe) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(wavelet_pipe, N4M_OP_WAVELET_DENOISE,
                                    wavelet_params, 2) == N4M_OK);
    CHECK(n4m_pipeline_fit(ctx, wavelet_pipe, &X_msc, nullptr) == N4M_OK);
    n4m_array_t* x_wavelet = nullptr;
    CHECK(n4m_pipeline_transform_alloc(ctx, wavelet_pipe, &X_msc, &x_wavelet) == N4M_OK);
    CHECK(n4m_array_shape(x_wavelet, &pipe_rows, &pipe_cols) == N4M_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    n4m_array_free(x_wavelet);
    n4m_pipeline_destroy(wavelet_pipe);

    n4m_pipeline_t* osc_pipe = nullptr;
    const double osc_params[] = {100.0, 1e-10};
    CHECK(n4m_pipeline_create(&osc_pipe) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(osc_pipe, N4M_OP_OSC,
                                    osc_params, 2) == N4M_OK);
    CHECK(n4m_pipeline_fit(ctx, osc_pipe, &X_msc, &Y) == N4M_OK);
    n4m_array_t* x_osc = nullptr;
    CHECK(n4m_pipeline_transform_alloc(ctx, osc_pipe, &X_msc, &x_osc) == N4M_OK);
    CHECK(n4m_array_shape(x_osc, &pipe_rows, &pipe_cols) == N4M_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    n4m_array_free(x_osc);
    n4m_pipeline_destroy(osc_pipe);

    n4m_pipeline_t* epo_pipe = nullptr;
    const double epo_params[] = {100.0, 1e-10};
    CHECK(n4m_pipeline_create(&epo_pipe) == N4M_OK);
    CHECK(n4m_pipeline_add_operator(epo_pipe, N4M_OP_EPO,
                                    epo_params, 2) == N4M_OK);
    CHECK(n4m_pipeline_fit(ctx, epo_pipe, &X_msc, &Y) == N4M_OK);
    n4m_array_t* x_epo = nullptr;
    CHECK(n4m_pipeline_transform_alloc(ctx, epo_pipe, &X_msc, &x_epo) == N4M_OK);
    CHECK(n4m_array_shape(x_epo, &pipe_rows, &pipe_cols) == N4M_OK);
    CHECK(pipe_rows == 4);
    CHECK(pipe_cols == 5);
    n4m_array_free(x_epo);
    n4m_pipeline_destroy(epo_pipe);

    // Model smoke: NIPALS fit, predict, transform and export.
    n4m_model_t* model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &model) == N4M_OK);
    CHECK(model != nullptr);
    n4m_array_t* pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, model, &X, &pred) == N4M_OK);
    int64_t rows = 0;
    int64_t cols = 0;
    CHECK(n4m_array_shape(pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(pred);

    n4m_array_t* scores = nullptr;
    CHECK(n4m_model_transform_alloc(ctx, model, &X, &scores) == N4M_OK);
    CHECK(n4m_array_shape(scores, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 2);
    n4m_array_free(scores);

    size_t export_size = 0;
    CHECK(n4m_model_export_size(model, &export_size) == N4M_OK);
    CHECK(export_size > 64);
    n4m_model_destroy(model);

    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_ORTHOGONAL_SCORES) == N4M_OK);
    n4m_model_t* oscores_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &oscores_model) == N4M_OK);
    CHECK(oscores_model != nullptr);
    n4m_array_t* oscores_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, oscores_model, &X, &oscores_pred) == N4M_OK);
    CHECK(n4m_array_shape(oscores_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(oscores_pred);
    n4m_model_destroy(oscores_model);

    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS) == N4M_OK);
    n4m_model_t* simpls_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &simpls_model) == N4M_OK);
    CHECK(simpls_model != nullptr);
    n4m_array_t* simpls_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, simpls_model, &X, &simpls_pred) == N4M_OK);
    CHECK(n4m_array_shape(simpls_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(simpls_pred);
    n4m_model_destroy(simpls_model);

    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_KERNEL_ALGORITHM) == N4M_OK);
    n4m_model_t* kernel_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &kernel_model) == N4M_OK);
    CHECK(kernel_model != nullptr);
    n4m_array_t* kernel_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, kernel_model, &X, &kernel_pred) == N4M_OK);
    CHECK(n4m_array_shape(kernel_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(kernel_pred);
    n4m_model_destroy(kernel_model);

    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_WIDE_KERNEL) == N4M_OK);
    n4m_model_t* wide_kernel_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &wide_kernel_model) == N4M_OK);
    CHECK(wide_kernel_model != nullptr);
    n4m_array_t* wide_kernel_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, wide_kernel_model, &X, &wide_kernel_pred) == N4M_OK);
    CHECK(n4m_array_shape(wide_kernel_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(wide_kernel_pred);
    n4m_model_destroy(wide_kernel_model);

    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_SVD) == N4M_OK);
    n4m_model_t* svd_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &svd_model) == N4M_OK);
    CHECK(svd_model != nullptr);
    n4m_array_t* svd_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, svd_model, &X, &svd_pred) == N4M_OK);
    CHECK(n4m_array_shape(svd_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(svd_pred);
    n4m_model_destroy(svd_model);

    CHECK(n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_SVD) == N4M_OK);
    CHECK(n4m_config_set_deflation(cfg, N4M_DEFLATION_CANONICAL) == N4M_OK);
    CHECK(n4m_config_set_n_components(cfg, 2) == N4M_OK);
    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_SVD) == N4M_OK);
    n4m_model_t* pls_svd_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y_multi, &pls_svd_model) == N4M_OK);
    CHECK(pls_svd_model != nullptr);
    n4m_array_t* pls_svd_scores = nullptr;
    CHECK(n4m_model_transform_alloc(ctx, pls_svd_model, &X, &pls_svd_scores) == N4M_OK);
    CHECK(n4m_array_shape(pls_svd_scores, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 2);
    n4m_array_free(pls_svd_scores);
    n4m_model_destroy(pls_svd_model);

    CHECK(n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION) == N4M_OK);
    CHECK(n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION) == N4M_OK);
    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_POWER) == N4M_OK);
    n4m_model_t* power_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &power_model) == N4M_OK);
    CHECK(power_model != nullptr);
    n4m_array_t* power_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, power_model, &X, &power_pred) == N4M_OK);
    CHECK(n4m_array_shape(power_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(power_pred);
    n4m_model_destroy(power_model);

    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_RANDOMIZED_SVD) == N4M_OK);
    n4m_model_t* randomized_svd_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &randomized_svd_model) == N4M_OK);
    CHECK(randomized_svd_model != nullptr);
    n4m_array_t* randomized_svd_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, randomized_svd_model, &X, &randomized_svd_pred) == N4M_OK);
    CHECK(n4m_array_shape(randomized_svd_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(randomized_svd_pred);
    n4m_model_destroy(randomized_svd_model);

    CHECK(n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_DA) == N4M_OK);
    CHECK(n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION) == N4M_OK);
    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS) == N4M_OK);
    n4m_model_t* pls_da_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &pls_da_model) == N4M_OK);
    CHECK(pls_da_model != nullptr);
    n4m_array_t* pls_da_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, pls_da_model, &X, &pls_da_pred) == N4M_OK);
    CHECK(n4m_array_shape(pls_da_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(pls_da_pred);
    n4m_model_destroy(pls_da_model);

    CHECK(n4m_config_set_algorithm(cfg, N4M_ALGO_OPLS) == N4M_OK);
    CHECK(n4m_config_set_deflation(cfg, N4M_DEFLATION_ORTHOGONAL) == N4M_OK);
    CHECK(n4m_config_set_n_components(cfg, 2) == N4M_OK);
    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS) == N4M_OK);
    n4m_model_t* opls_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &opls_model) == N4M_OK);
    CHECK(opls_model != nullptr);
    n4m_array_t* opls_scores = nullptr;
    CHECK(n4m_model_transform_alloc(ctx, opls_model, &X, &opls_scores) == N4M_OK);
    CHECK(n4m_array_shape(opls_scores, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 2);
    n4m_array_free(opls_scores);
    n4m_model_destroy(opls_model);

    CHECK(n4m_config_set_algorithm(cfg, N4M_ALGO_OPLS_DA) == N4M_OK);
    CHECK(n4m_config_set_deflation(cfg, N4M_DEFLATION_ORTHOGONAL) == N4M_OK);
    CHECK(n4m_config_set_n_components(cfg, 2) == N4M_OK);
    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS) == N4M_OK);
    n4m_model_t* opls_da_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &opls_da_model) == N4M_OK);
    CHECK(opls_da_model != nullptr);
    n4m_array_t* opls_da_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, opls_da_model, &X, &opls_da_pred) == N4M_OK);
    CHECK(n4m_array_shape(opls_da_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(opls_da_pred);
    n4m_model_destroy(opls_da_model);

    n4m_model_t* opls_da_multi_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y_multi, &opls_da_multi_model) == N4M_OK);
    CHECK(opls_da_multi_model != nullptr);
    n4m_array_t* opls_da_multi_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, opls_da_multi_model, &X, &opls_da_multi_pred) == N4M_OK);
    CHECK(n4m_array_shape(opls_da_multi_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 3);
    n4m_array_free(opls_da_multi_pred);
    n4m_model_destroy(opls_da_multi_model);

    CHECK(n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_CANONICAL) == N4M_OK);
    CHECK(n4m_config_set_deflation(cfg, N4M_DEFLATION_CANONICAL) == N4M_OK);
    CHECK(n4m_config_set_n_components(cfg, 1) == N4M_OK);
    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS) == N4M_OK);
    n4m_model_t* canonical_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &canonical_model) == N4M_OK);
    CHECK(canonical_model != nullptr);
    n4m_array_t* canonical_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, canonical_model, &X, &canonical_pred) == N4M_OK);
    CHECK(n4m_array_shape(canonical_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(canonical_pred);
    n4m_model_destroy(canonical_model);

    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_SVD) == N4M_OK);
    n4m_model_t* canonical_svd_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &canonical_svd_model) == N4M_OK);
    CHECK(canonical_svd_model != nullptr);
    n4m_array_t* canonical_svd_scores = nullptr;
    CHECK(n4m_model_transform_alloc(ctx, canonical_svd_model, &X, &canonical_svd_scores) == N4M_OK);
    CHECK(n4m_array_shape(canonical_svd_scores, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(canonical_svd_scores);
    n4m_model_destroy(canonical_svd_model);

    CHECK(n4m_config_set_algorithm(cfg, N4M_ALGO_PCR) == N4M_OK);
    CHECK(n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION) == N4M_OK);
    CHECK(n4m_config_set_n_components(cfg, 2) == N4M_OK);
    CHECK(n4m_config_set_solver(cfg, N4M_SOLVER_SVD) == N4M_OK);
    n4m_model_t* pcr_model = nullptr;
    CHECK(n4m_model_fit(ctx, cfg, &X, &Y, &pcr_model) == N4M_OK);
    CHECK(pcr_model != nullptr);
    n4m_array_t* pcr_pred = nullptr;
    CHECK(n4m_model_predict_alloc(ctx, pcr_model, &X, &pcr_pred) == N4M_OK);
    CHECK(n4m_array_shape(pcr_pred, &rows, &cols) == N4M_OK);
    CHECK(rows == 4);
    CHECK(cols == 1);
    n4m_array_free(pcr_pred);
    n4m_model_destroy(pcr_model);

    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
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

bool resolve_algo(const char* name, n4m_algorithm_t& algo,
                  n4m_solver_t& solver, n4m_deflation_t& deflation) {
    deflation = N4M_DEFLATION_REGRESSION;
    if (strcmp(name, "pls_nipals") == 0) {
        algo = N4M_ALGO_PLS_REGRESSION;
        solver = N4M_SOLVER_NIPALS;
        return true;
    }
    if (strcmp(name, "pls_orthogonal_scores") == 0) {
        algo = N4M_ALGO_PLS_REGRESSION;
        solver = N4M_SOLVER_ORTHOGONAL_SCORES;
        return true;
    }
    if (strcmp(name, "pls_simpls") == 0) {
        algo = N4M_ALGO_PLS_REGRESSION;
        solver = N4M_SOLVER_SIMPLS;
        return true;
    }
    if (strcmp(name, "pls_kernel_algorithm") == 0) {
        algo = N4M_ALGO_PLS_REGRESSION;
        solver = N4M_SOLVER_KERNEL_ALGORITHM;
        return true;
    }
    if (strcmp(name, "pls_wide_kernel") == 0) {
        algo = N4M_ALGO_PLS_REGRESSION;
        solver = N4M_SOLVER_WIDE_KERNEL;
        return true;
    }
    if (strcmp(name, "pls_svd") == 0) {
        algo = N4M_ALGO_PLS_REGRESSION;
        solver = N4M_SOLVER_SVD;
        return true;
    }
    if (strcmp(name, "pls_power") == 0) {
        algo = N4M_ALGO_PLS_REGRESSION;
        solver = N4M_SOLVER_POWER;
        return true;
    }
    if (strcmp(name, "pls_randomized_svd") == 0) {
        algo = N4M_ALGO_PLS_REGRESSION;
        solver = N4M_SOLVER_RANDOMIZED_SVD;
        return true;
    }
    if (strcmp(name, "pcr_svd") == 0) {
        algo = N4M_ALGO_PCR;
        solver = N4M_SOLVER_SVD;
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
            "Usage: n4m_cli --bench --algo <name> --samples N --features P "
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

    n4m_algorithm_t algo = N4M_ALGO_PLS_REGRESSION;
    n4m_solver_t solver = N4M_SOLVER_SIMPLS;
    n4m_deflation_t deflation = N4M_DEFLATION_REGRESSION;
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
    n4m_matrix_view_t X;
    n4m_matrix_view_t Y;
    n4m_matrix_view_init_rowmajor(&X, xbuf, static_cast<int64_t>(n),
                                  static_cast<int64_t>(p), N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&Y, ybuf, static_cast<int64_t>(n), 1,
                                  N4M_DTYPE_F64);
    n4m_matrix_view_t Pred;
    n4m_matrix_view_init_rowmajor(&Pred, predbuf, static_cast<int64_t>(n), 1,
                                  N4M_DTYPE_F64);

    n4m_context_t* ctx = nullptr;
    if (n4m_context_create(&ctx) != N4M_OK) {
        free(xbuf); free(ybuf); free(predbuf);
        return 1;
    }
    n4m_config_t* cfg = nullptr;
    if (n4m_config_create(&cfg) != N4M_OK) {
        n4m_context_destroy(ctx);
        free(xbuf); free(ybuf); free(predbuf);
        return 1;
    }
    n4m_config_set_algorithm(cfg, algo);
    n4m_config_set_solver(cfg, solver);
    n4m_config_set_deflation(cfg, deflation);
    n4m_config_set_n_components(cfg, opt.n_components);
    n4m_config_set_center_x(cfg, 1);
    n4m_config_set_scale_x(cfg, 0);
    n4m_config_set_center_y(cfg, 1);
    n4m_config_set_scale_y(cfg, 0);
    n4m_config_set_store_scores(cfg, 0);

    // Run repeats and report median + min.
    double* fit_ms = static_cast<double*>(malloc(static_cast<size_t>(opt.repeats) * sizeof(double)));
    double* predict_ms = static_cast<double*>(malloc(static_cast<size_t>(opt.repeats) * sizeof(double)));
    if (fit_ms == nullptr || predict_ms == nullptr) {
        free(fit_ms); free(predict_ms);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
        free(xbuf); free(ybuf); free(predbuf);
        return 1;
    }
    n4m_status_t last_status = N4M_OK;
    double last_rmse = 0.0;
    for (int32_t r = 0; r < opt.repeats; ++r) {
        const auto fit_t0 = std::chrono::steady_clock::now();
        n4m_model_t* model = nullptr;
        last_status = n4m_model_fit(ctx, cfg, &X, &Y, &model);
        const auto fit_t1 = std::chrono::steady_clock::now();
        if (last_status != N4M_OK) {
            // NAN from <math.h> is a float literal on some platforms;
            // assign explicit double NaN to avoid -Werror=double-promotion.
            const double nan_d = std::numeric_limits<double>::quiet_NaN();
            fit_ms[r] = nan_d; predict_ms[r] = nan_d;
            continue;
        }
        last_status = n4m_model_predict(ctx, model, &X, &Pred);
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
        n4m_model_destroy(model);
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
           (last_status == N4M_OK) ? "ok" : n4m_status_to_string(last_status),
           fit_med, predict_med, fit_min, predict_min, last_rmse);

    free(fit_ms);
    free(predict_ms);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
    free(xbuf); free(ybuf); free(predbuf);
    return (last_status == N4M_OK) ? 0 : 1;
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
