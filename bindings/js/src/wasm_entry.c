/* SPDX-License-Identifier: CECILL-2.1
 *
 * Emscripten entry point + production helpers.
 *
 * Pulls in the full public ABI header so all n4m_* symbols stay
 * reachable. Adds a small set of higher-level `n4m_wasm_*` raw-double-
 * pointer convenience helpers used by the PLS fast-path in model.ts.
 *
 * Historical note: JS-built `n4m_matrix_view_t*` parameters reaching the
 * deep entrypoints USED to return garbage. That was NOT an Emscripten
 * codegen bug — it was the TS ccall layer passing JS `number` for the
 * int64_t rows/cols args of n4m_matrix_view_init_rowmajor while the module
 * is built with `-s WASM_BIGINT=1` (i64 slots must be marshalled as 'i64'
 * with BigInt values). Once ffi.ts/makeMatrixView passes BigInt dims, the
 * full view-pointer path (n4m_model_fit, the method_result producers, etc.)
 * is byte-correct from JS. These raw helpers remain only as a thin PLS
 * convenience surface; they are not required for the generic method path.
 */

#include "n4m/n4m.h"
/* Internal MSC engine — lets the generic preprocessing dispatcher round-trip the
 * fitted reference spectrum (get/set_reference) for predict-later, which the
 * public MSC ABI does not expose. Internal header, not part of the public ABI. */
#include "../../../cpp/src/core/preprocessing/scatter/msc.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Touch every accessor on the matrix-view struct so Emscripten preserves
 * the in-line `n4m_matrix_view_init_*` helpers from the header. */
__attribute__((used)) static int n4m_wasm_keep_alive(void) {
    n4m_matrix_view_t v = {0};
    n4m_matrix_view_init_rowmajor(&v, NULL, 0, 0, N4M_DTYPE_F64);
    n4m_matrix_view_init_colmajor(&v, NULL, 0, 0, N4M_DTYPE_F64);
    return (int)(v.dtype + v.rows + v.cols);
}

/* Fit a PLS regression on X (n × p, row-major) predicting Y (n × q,
 * row-major). The function expects raw double pointers (JS-allocated
 * buffers in WASM linear memory) and writes:
 *
 *   coefficients_out (p * q doubles, row-major)
 *   x_mean_out       (p doubles)
 *   y_mean_out       (q doubles)
 *   predictions_out  (n * q doubles, row-major) — optional (NULL skips)
 *
 * Returns 0 on success, a N4M_ERR_* on failure. center_x / center_y
 * are always 1; solver is SIMPLS.
 *
 * Rationale: see the file-level comment — the matrix-view-arg path
 * has an Emscripten codegen issue specific to this build; this helper
 * funnels JS callers through the working raw-data-pointer path.
 */
__attribute__((used))
int n4m_wasm_pls_fit_legacy(const double *x, const double *y,
                      int n, int p, int q, int n_components,
                      double *coefficients_out,
                      double *x_mean_out,
                      double *y_mean_out,
                      double *predictions_out) {
    if (x == NULL || y == NULL || coefficients_out == NULL ||
        x_mean_out == NULL || y_mean_out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n < 2 || p < 1 || q < 1 || n_components < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }

    n4m_matrix_view_t xv, yv;
    n4m_matrix_view_init_rowmajor(&xv, (void *)x, n, p, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&yv, (void *)y, n, q, N4M_DTYPE_F64);

    n4m_context_t *ctx = NULL;
    n4m_status_t s = n4m_context_create(&ctx);
    if (s != N4M_OK) return s;
    n4m_config_t *cfg = NULL;
    s = n4m_config_create(&cfg);
    if (s != N4M_OK) { n4m_context_destroy(ctx); return s; }

    n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION);
    n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
    n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION);
    n4m_config_set_n_components(cfg, n_components);
    n4m_config_set_center_x(cfg, 1);
    n4m_config_set_center_y(cfg, 1);

    n4m_model_t *m = NULL;
    s = n4m_model_fit(ctx, cfg, &xv, &yv, &m);
    n4m_config_destroy(cfg);
    if (s != N4M_OK) { n4m_context_destroy(ctx); return s; }

    /* Copy out coefficients / x_mean / y_mean. */
    n4m_array_t *arr = NULL;
    if (n4m_model_get_array(ctx, m, N4M_MODEL_COEFFICIENTS, &arr) == N4M_OK) {
        n4m_matrix_view_t v;
        n4m_array_view(arr, &v);
        memcpy(coefficients_out, v.data,
               (size_t)p * (size_t)q * sizeof(double));
        n4m_array_free(arr);
    }
    if (n4m_model_get_array(ctx, m, N4M_MODEL_X_MEAN, &arr) == N4M_OK) {
        n4m_matrix_view_t v;
        n4m_array_view(arr, &v);
        memcpy(x_mean_out, v.data, (size_t)p * sizeof(double));
        n4m_array_free(arr);
    }
    if (n4m_model_get_array(ctx, m, N4M_MODEL_Y_MEAN, &arr) == N4M_OK) {
        n4m_matrix_view_t v;
        n4m_array_view(arr, &v);
        memcpy(y_mean_out, v.data, (size_t)q * sizeof(double));
        n4m_array_free(arr);
    }

    if (predictions_out != NULL) {
        n4m_matrix_view_t pv;
        n4m_matrix_view_init_rowmajor(&pv, predictions_out, n, q,
                                        N4M_DTYPE_F64);
        n4m_model_predict(ctx, m, &xv, &pv);
    }

    n4m_model_destroy(m);
    n4m_context_destroy(ctx);
    return N4M_OK;
}

/* Predict from coefficients + means. Useful when the model was fit
 * elsewhere and only its coefficient triple was transported. */
__attribute__((used))
int n4m_wasm_pls_predict_from_coeffs(const double *x_new,
                                       int n_new, int p, int q,
                                       const double *coefficients,
                                       const double *x_mean,
                                       const double *y_mean,
                                       double *predictions_out) {
    if (x_new == NULL || coefficients == NULL || x_mean == NULL ||
        y_mean == NULL || predictions_out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    for (int i = 0; i < n_new; ++i) {
        for (int t = 0; t < q; ++t) {
            double s = y_mean[t];
            for (int f = 0; f < p; ++f) {
                s += (x_new[i * p + f] - x_mean[f]) *
                     coefficients[f * q + t];
            }
            predictions_out[i * q + t] = s;
        }
    }
    return N4M_OK;
}

/* ============================================================================
 * Generic MODEL dispatcher (raw-double-pointer surface for JS)
 * ----------------------------------------------------------------------------
 * Fit any coefficient-based libn4m model by its catalog `type` token + a numeric
 * params vector, returning the regression coefficient triple (coefficients,
 * x_mean, y_mean) plus an optional intercept and in-sample predictions. The
 * numerics stay 100% in libn4m; this is only marshalling.
 *
 *   Tier A  — the Config/algorithm family (PLS, PCR, PLSCanonical, PLSSVD,
 *             PLSDA) routed through n4m_model_fit + n4m_model_get_array.
 *   Tier B  — the standalone coeff fits (Ridge, RidgePLS, ContinuumRegression,
 *             RobustPLS, CPPLS, SparseSIMPLS, GroupSparsePLS, FusedSparsePLS,
 *             BaggingPLS, BoostingPLS, RandomSubspacePLS) routed through their
 *             own n4m_*_fit + n4m_method_result_get_double_matrix.
 *
 * Non-coeff / unknown tokens return N4M_ERR_NOT_IMPLEMENTED.
 * ==========================================================================*/
enum n4m_wasm_model_kind {
    MK_NONE = 0,
    /* Tier A — algorithm-enum family */
    MK_PLS = 1, MK_PCR, MK_PLS_CANONICAL, MK_PLS_SVD, MK_PLS_DA,
    /* Tier B — standalone coeff fits */
    MK_RIDGE, MK_RIDGE_PLS, MK_CONTINUUM, MK_ROBUST_PLS, MK_CPPLS,
    MK_SPARSE_SIMPLS, MK_GROUP_SPARSE_PLS, MK_FUSED_SPARSE_PLS,
    MK_BAGGING_PLS, MK_BOOSTING_PLS, MK_RANDOM_SUBSPACE_PLS
};

static int model_kind_for(const char* model) {
    /* Tier A */
    if (strcmp(model, "PLS") == 0 || strcmp(model, "PLSRegression") == 0) return MK_PLS;
    if (strcmp(model, "PCR") == 0) return MK_PCR;
    if (strcmp(model, "PLSCanonical") == 0) return MK_PLS_CANONICAL;
    if (strcmp(model, "PLSSVD") == 0) return MK_PLS_SVD;
    if (strcmp(model, "PLSDA") == 0) return MK_PLS_DA;
    /* Tier B */
    if (strcmp(model, "Ridge") == 0) return MK_RIDGE;
    if (strcmp(model, "RidgePLS") == 0) return MK_RIDGE_PLS;
    if (strcmp(model, "ContinuumRegression") == 0) return MK_CONTINUUM;
    if (strcmp(model, "RobustPLS") == 0) return MK_ROBUST_PLS;
    if (strcmp(model, "CPPLS") == 0) return MK_CPPLS;
    if (strcmp(model, "SparseSIMPLS") == 0) return MK_SPARSE_SIMPLS;
    if (strcmp(model, "GroupSparsePLS") == 0) return MK_GROUP_SPARSE_PLS;
    if (strcmp(model, "FusedSparsePLS") == 0) return MK_FUSED_SPARSE_PLS;
    if (strcmp(model, "BaggingPLS") == 0) return MK_BAGGING_PLS;
    if (strcmp(model, "BoostingPLS") == 0) return MK_BOOSTING_PLS;
    if (strcmp(model, "RandomSubspacePLS") == 0) return MK_RANDOM_SUBSPACE_PLS;
    return MK_NONE;
}

/* Tier A — build a Config for the algorithm-enum family, fit via the model API,
 * and copy out the coefficient triple (+ intercept). */
static int n4m_wasm_model_fit_tier_a(
        int kind, const double* x, const double* y,
        int n, int p, int q, int n_components,
        double* coefficients_out, double* x_mean_out, double* y_mean_out,
        double* intercept_out, double* predictions_out) {
    n4m_matrix_view_t xv, yv;
    n4m_matrix_view_init_rowmajor(&xv, (void*)x, n, p, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&yv, (void*)y, n, q, N4M_DTYPE_F64);

    n4m_context_t* ctx = NULL;
    n4m_status_t s = n4m_context_create(&ctx);
    if (s != N4M_OK) return s;
    n4m_config_t* cfg = NULL;
    s = n4m_config_create(&cfg);
    if (s != N4M_OK) { n4m_context_destroy(ctx); return s; }

    switch (kind) {
        case MK_PLS:
            n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION);
            n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
            n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION);
            break;
        case MK_PCR:
            n4m_config_set_algorithm(cfg, N4M_ALGO_PCR);
            n4m_config_set_solver(cfg, N4M_SOLVER_SVD);
            n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION);
            break;
        case MK_PLS_CANONICAL:
            n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_CANONICAL);
            n4m_config_set_solver(cfg, N4M_SOLVER_NIPALS);
            n4m_config_set_deflation(cfg, N4M_DEFLATION_CANONICAL);
            break;
        case MK_PLS_SVD:
            n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_SVD);
            n4m_config_set_solver(cfg, N4M_SOLVER_SVD);
            n4m_config_set_deflation(cfg, N4M_DEFLATION_CANONICAL);
            break;
        case MK_PLS_DA:
            n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_DA);
            n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
            n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION);
            break;
        default:
            n4m_config_destroy(cfg);
            n4m_context_destroy(ctx);
            return N4M_ERR_NOT_IMPLEMENTED;
    }
    n4m_config_set_n_components(cfg, n_components);
    n4m_config_set_center_x(cfg, 1);
    n4m_config_set_center_y(cfg, 1);

    n4m_model_t* m = NULL;
    s = n4m_model_fit(ctx, cfg, &xv, &yv, &m);
    n4m_config_destroy(cfg);
    if (s != N4M_OK) { n4m_context_destroy(ctx); return s; }

    n4m_array_t* arr = NULL;
    if (n4m_model_get_array(ctx, m, N4M_MODEL_COEFFICIENTS, &arr) == N4M_OK) {
        n4m_matrix_view_t v;
        n4m_array_view(arr, &v);
        memcpy(coefficients_out, v.data, (size_t)p * (size_t)q * sizeof(double));
        n4m_array_free(arr);
    }
    if (n4m_model_get_array(ctx, m, N4M_MODEL_X_MEAN, &arr) == N4M_OK) {
        n4m_matrix_view_t v;
        n4m_array_view(arr, &v);
        memcpy(x_mean_out, v.data, (size_t)p * sizeof(double));
        n4m_array_free(arr);
    }
    if (n4m_model_get_array(ctx, m, N4M_MODEL_Y_MEAN, &arr) == N4M_OK) {
        n4m_matrix_view_t v;
        n4m_array_view(arr, &v);
        memcpy(y_mean_out, v.data, (size_t)q * sizeof(double));
        n4m_array_free(arr);
    }
    if (intercept_out != NULL) {
        for (int t = 0; t < q; ++t) intercept_out[t] = 0.0;
        if (n4m_model_get_array(ctx, m, N4M_MODEL_INTERCEPT, &arr) == N4M_OK) {
            n4m_matrix_view_t v;
            n4m_array_view(arr, &v);
            memcpy(intercept_out, v.data, (size_t)q * sizeof(double));
            n4m_array_free(arr);
        }
    }
    if (predictions_out != NULL) {
        n4m_matrix_view_t pv;
        n4m_matrix_view_init_rowmajor(&pv, predictions_out, n, q, N4M_DTYPE_F64);
        n4m_model_predict(ctx, m, &xv, &pv);
    }
    n4m_model_destroy(m);
    n4m_context_destroy(ctx);
    return N4M_OK;
}

/* Copy a named double matrix from a method-result into a caller buffer of
 * exactly `expect` doubles (silently leaves the buffer untouched if absent). */
static void copy_result_matrix(const n4m_method_result_t* r, const char* name,
                               double* out, size_t expect) {
    if (out == NULL) return;
    const double* data = NULL;
    int64_t rows = 0, cols = 0;
    if (n4m_method_result_get_double_matrix(r, name, &data, &rows, &cols) == N4M_OK
        && data != NULL && (size_t)(rows * cols) == expect) {
        memcpy(out, data, expect * sizeof(double));
    }
}

/* Tier B — call the standalone fit for `kind`, then read the coeff triple
 * (+ intercept for Ridge) out of the returned method-result. */
static int n4m_wasm_model_fit_tier_b(
        int kind, const double* params, int n_params,
        const double* x, const double* y,
        int n, int p, int q,
        double* coefficients_out, double* x_mean_out, double* y_mean_out,
        double* intercept_out, double* predictions_out) {
    n4m_matrix_view_t xv, yv;
    n4m_matrix_view_init_rowmajor(&xv, (void*)x, n, p, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&yv, (void*)y, n, q, N4M_DTYPE_F64);

    n4m_context_t* ctx = NULL;
    n4m_status_t s = n4m_context_create(&ctx);
    if (s != N4M_OK) return s;
    n4m_config_t* cfg = NULL;
    s = n4m_config_create(&cfg);
    if (s != N4M_OK) { n4m_context_destroy(ctx); return s; }

    n4m_method_result_t* res = NULL;
    switch (kind) {
        case MK_RIDGE: {
            double lambda = n_params >= 1 ? params[0] : 1.0;
            /* n4m_ridge_fit reads cfg.ridge_lambda when lambdas==NULL; pass the
             * single override array so the param vector is authoritative. */
            s = n4m_ridge_fit(ctx, cfg, &xv, &yv, &lambda, 1, &res);
            break;
        }
        case MK_RIDGE_PLS: {
            double ridge_lambda = n_params >= 1 ? params[0] : 1.0;
            s = n4m_ridge_pls_fit(ctx, cfg, &xv, &yv, ridge_lambda, &res);
            break;
        }
        case MK_CONTINUUM: {
            double tau = n_params >= 1 ? params[0] : 0.5;
            s = n4m_continuum_regression_fit(ctx, cfg, &xv, &yv, tau, &res);
            break;
        }
        case MK_ROBUST_PLS: {
            double huber_k = n_params >= 1 ? params[0] : 1.345;
            int max_irls = n_params >= 2 ? (int)params[1] : 5;
            s = n4m_robust_pls_fit(ctx, cfg, &xv, &yv, huber_k, max_irls, &res);
            break;
        }
        case MK_CPPLS: {
            double gamma = n_params >= 1 ? params[0] : 0.5;
            s = n4m_cppls_fit(ctx, cfg, &xv, &yv, gamma, &res);
            break;
        }
        case MK_SPARSE_SIMPLS: {
            double sparsity = n_params >= 1 ? params[0] : 0.0;
            s = n4m_sparse_simpls_fit(ctx, cfg, &xv, &yv, sparsity, &res);
            break;
        }
        case MK_GROUP_SPARSE_PLS: {
            double group_lambda = n_params >= 1 ? params[0] : 0.0;
            /* No group ids over this scalar surface — one group covers all
             * features (group_assignment all-zero). */
            int32_t* groups = (int32_t*)calloc((size_t)(p > 0 ? p : 1),
                                                sizeof(int32_t));
            if (groups == NULL) { s = N4M_ERR_OUT_OF_MEMORY; break; }
            s = n4m_group_sparse_pls_fit(ctx, cfg, &xv, &yv, groups, p,
                                         group_lambda, &res);
            free(groups);
            break;
        }
        case MK_FUSED_SPARSE_PLS: {
            double l1 = n_params >= 1 ? params[0] : 0.0;
            double fusion = n_params >= 2 ? params[1] : 0.0;
            s = n4m_fused_sparse_pls_fit(ctx, cfg, &xv, &yv, l1, fusion, &res);
            break;
        }
        case MK_BAGGING_PLS: {
            int n_estimators = n_params >= 1 ? (int)params[0] : 10;
            uint64_t seed = n_params >= 2 ? (uint64_t)params[1] : 0;
            s = n4m_bagging_pls_fit(ctx, cfg, &xv, &yv, n_estimators, seed, &res);
            break;
        }
        case MK_BOOSTING_PLS: {
            int n_estimators = n_params >= 1 ? (int)params[0] : 10;
            double lr = n_params >= 2 ? params[1] : 0.1;
            s = n4m_boosting_pls_fit(ctx, cfg, &xv, &yv, n_estimators, lr, &res);
            break;
        }
        case MK_RANDOM_SUBSPACE_PLS: {
            int n_estimators = n_params >= 1 ? (int)params[0] : 10;
            int feats = n_params >= 2 ? (int)params[1] : (p > 1 ? p / 2 : 1);
            uint64_t seed = n_params >= 3 ? (uint64_t)params[2] : 0;
            s = n4m_random_subspace_pls_fit(ctx, cfg, &xv, &yv, n_estimators,
                                            feats, seed, &res);
            break;
        }
        default:
            s = N4M_ERR_NOT_IMPLEMENTED;
            break;
    }
    n4m_config_destroy(cfg);
    if (s != N4M_OK || res == NULL) {
        if (res != NULL) n4m_method_result_destroy(res);
        n4m_context_destroy(ctx);
        return s != N4M_OK ? s : N4M_ERR_INVALID_ARGUMENT;
    }

    copy_result_matrix(res, "coefficients", coefficients_out,
                       (size_t)p * (size_t)q);
    copy_result_matrix(res, "x_mean", x_mean_out, (size_t)p);
    copy_result_matrix(res, "y_mean", y_mean_out, (size_t)q);
    if (intercept_out != NULL) {
        for (int t = 0; t < q; ++t) intercept_out[t] = 0.0;
        copy_result_matrix(res, "intercept", intercept_out, (size_t)q);
    }
    if (predictions_out != NULL) {
        copy_result_matrix(res, "predictions", predictions_out,
                           (size_t)n * (size_t)q);
    }
    n4m_method_result_destroy(res);
    n4m_context_destroy(ctx);
    return N4M_OK;
}

/* Fit any coeff-based model by token. Writes coefficients_out (p*q row-major),
 * x_mean_out (p), y_mean_out (q), and — when non-NULL — intercept_out (q) and
 * predictions_out (n*q). Returns N4M_ERR_NOT_IMPLEMENTED for unknown tokens. */
__attribute__((used))
int n4m_wasm_model_fit(const char* model,
                       const double* params, int n_params,
                       const double* x, const double* y,
                       int n, int p, int q, int n_components,
                       double* coefficients_out,
                       double* x_mean_out,
                       double* y_mean_out,
                       double* intercept_out,
                       double* predictions_out) {
    if (model == NULL || x == NULL || y == NULL || coefficients_out == NULL ||
        x_mean_out == NULL || y_mean_out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n_params < 0 || (n_params > 0 && params == NULL)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n < 2 || p < 1 || q < 1 || n_components < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    int kind = model_kind_for(model);
    if (kind == MK_NONE) return N4M_ERR_NOT_IMPLEMENTED;
    if (kind <= MK_PLS_DA) {
        return n4m_wasm_model_fit_tier_a(kind, x, y, n, p, q, n_components,
                                         coefficients_out, x_mean_out,
                                         y_mean_out, intercept_out,
                                         predictions_out);
    }
    return n4m_wasm_model_fit_tier_b(kind, params, n_params, x, y, n, p, q,
                                     coefficients_out, x_mean_out, y_mean_out,
                                     intercept_out, predictions_out);
}

/* Predict from a fitted coefficient triple.
 *
 * Two equivalent affine conventions are supported via the `intercept` arg:
 *   intercept == NULL : centred form, pred = y_mean + (x - x_mean).B
 *                       (every libn4m coeff model exposes this triple, and its
 *                        own in-sample `predictions` use exactly this form).
 *   intercept != NULL : explicit-intercept form, pred = intercept + x.B
 *                       (x_mean / y_mean ignored; for models that fold the
 *                        constant into `intercept`, e.g. sklearn-style Ridge).
 *
 * Because libn4m's Ridge folds y_mean - x_mean.B into `intercept`, the two
 * forms agree numerically; callers pick whichever triple they round-tripped. */
__attribute__((used))
int n4m_wasm_model_predict_from_coeffs(const double* coefficients,
                                       const double* x_mean,
                                       const double* y_mean,
                                       const double* intercept,
                                       const double* x_new,
                                       int n_new, int p, int q,
                                       double* predictions_out) {
    if (coefficients == NULL || x_new == NULL || predictions_out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (intercept == NULL && (x_mean == NULL || y_mean == NULL)) {
        return N4M_ERR_NULL_POINTER;
    }
    for (int i = 0; i < n_new; ++i) {
        for (int t = 0; t < q; ++t) {
            double s;
            if (intercept != NULL) {
                s = intercept[t];
                for (int f = 0; f < p; ++f) {
                    s += x_new[i * p + f] * coefficients[f * q + t];
                }
            } else {
                s = y_mean[t];
                for (int f = 0; f < p; ++f) {
                    s += (x_new[i * p + f] - x_mean[f]) * coefficients[f * q + t];
                }
            }
            predictions_out[i * q + t] = s;
        }
    }
    return N4M_OK;
}

/* ============================================================================
 * Generic preprocessing dispatcher (raw-double-pointer surface for JS)
 * ----------------------------------------------------------------------------
 * One handle wraps any libn4m preprocessing operator. The numerics stay 100%
 * in libn4m; this is only marshalling so the JS shell can fit/transform any
 * operator by id without a hand-written wrapper per operator. All operators in
 * this set are shape-preserving (out is n*p), which keeps the JS side trivial.
 * Stateful operators (MSC) expose their fitted state as plain doubles via
 * get/set_state so it round-trips into a saved model (.n4a) for predict-later.
 * ==========================================================================*/
enum n4m_wasm_pp_kind {
    PPK_SNV = 1, PPK_MSC, PPK_SAVGOL, PPK_DERIV1, PPK_DERIV2,
    PPK_DETREND, PPK_GAUSSIAN,
    /* A2a — baseline correctors (stateless) */
    PPK_ASLS, PPK_AIRPLS, PPK_ARPLS, PPK_MODPOLY, PPK_IMODPOLY,
    PPK_SNIP, PPK_ROLLING_BALL, PPK_IASLS, PPK_BEADS,
    /* A2b — signal conversions (stateless) */
    PPK_TO_ABS, PPK_FROM_ABS, PPK_PCT_TO_FRAC, PPK_FRAC_TO_PCT, PPK_KUBELKA_MUNK,
    /* A2c — scatter / scaling / derivative (stateless) */
    PPK_RNV, PPK_LSNV, PPK_AREA, PPK_NORRIS_WILLIAMS, PPK_LOG, PPK_WAVELET_DENOISE
};

struct n4m_wasm_pp_s {
    int kind;
    void* h;        /* public op handle, or n4m_pp_msc_state_t* for MSC */
};

static int pp_kind_for(const char* op, const double* params, int n_params) {
    if (strcmp(op, "StandardNormalVariate") == 0) return PPK_SNV;
    if (strcmp(op, "MSC") == 0) return PPK_MSC;
    if (strcmp(op, "SavitzkyGolay") == 0) return PPK_SAVGOL;
    if (strcmp(op, "Derivative") == 0)
        return (n_params >= 1 && params[0] >= 2.0) ? PPK_DERIV2 : PPK_DERIV1;
    if (strcmp(op, "Detrend") == 0) return PPK_DETREND;
    if (strcmp(op, "GaussianFilter") == 0) return PPK_GAUSSIAN;
    /* A2a — baseline correctors */
    if (strcmp(op, "AsLS") == 0) return PPK_ASLS;
    if (strcmp(op, "AirPLS") == 0) return PPK_AIRPLS;
    if (strcmp(op, "ArPLS") == 0) return PPK_ARPLS;
    if (strcmp(op, "ModPoly") == 0) return PPK_MODPOLY;
    if (strcmp(op, "IModPoly") == 0) return PPK_IMODPOLY;
    if (strcmp(op, "SNIP") == 0) return PPK_SNIP;
    if (strcmp(op, "RollingBall") == 0) return PPK_ROLLING_BALL;
    if (strcmp(op, "IAsLS") == 0) return PPK_IASLS;
    if (strcmp(op, "BEADS") == 0) return PPK_BEADS;
    /* A2b — signal conversions */
    if (strcmp(op, "ToAbsorbance") == 0) return PPK_TO_ABS;
    if (strcmp(op, "FromAbsorbance") == 0) return PPK_FROM_ABS;
    if (strcmp(op, "PercentToFraction") == 0) return PPK_PCT_TO_FRAC;
    if (strcmp(op, "FractionToPercent") == 0) return PPK_FRAC_TO_PCT;
    if (strcmp(op, "KubelkaMunk") == 0) return PPK_KUBELKA_MUNK;
    /* A2c — scatter / scaling / derivative */
    if (strcmp(op, "RobustNormalVariate") == 0) return PPK_RNV;
    if (strcmp(op, "LocalSNV") == 0) return PPK_LSNV;
    if (strcmp(op, "AreaNormalization") == 0) return PPK_AREA;
    if (strcmp(op, "NorrisWilliams") == 0) return PPK_NORRIS_WILLIAMS;
    if (strcmp(op, "LogTransform") == 0) return PPK_LOG;
    if (strcmp(op, "WaveletDenoise") == 0) return PPK_WAVELET_DENOISE;
    /* Note: n4m_pp_normalize is COLUMN-wise L2 (batch-dependent), not the
     * per-spectrum normalization a catalog node would want — intentionally
     * not exposed here until a row-wise L2 operator lands in libn4m. */
    return 0;
}

/* Create an operator from its catalog `type` token + numeric params.
 * Returns an opaque handle (JS keeps it as an int) or NULL on failure. */
__attribute__((used))
void* n4m_wasm_pp_create(const char* op, const double* params, int n_params) {
    if (op == NULL) return NULL;
    if (n_params < 0 || (n_params > 0 && params == NULL)) return NULL;
    int kind = pp_kind_for(op, params, n_params);
    if (kind == 0) return NULL;

    struct n4m_wasm_pp_s* w =
        (struct n4m_wasm_pp_s*)malloc(sizeof(struct n4m_wasm_pp_s));
    if (w == NULL) return NULL;
    w->kind = kind;
    w->h = NULL;

    n4m_status_t s = N4M_OK;
    switch (kind) {
        case PPK_SNV: {
            n4m_pp_snv_handle_t* h = NULL;
            s = n4m_pp_snv_create(&h, 1, 1, 0);
            w->h = h;
            break;
        }
        case PPK_MSC: {
            w->h = n4m_pp_msc_state_new();
            if (w->h == NULL) s = N4M_ERR_OUT_OF_MEMORY;
            break;
        }
        case PPK_SAVGOL: {
            int window = n_params >= 1 ? (int)params[0] : 11;
            int poly = n_params >= 2 ? (int)params[1] : 2;
            int deriv = n_params >= 3 ? (int)params[2] : 0;
            n4m_pp_savgol_handle_t* h = NULL;
            s = n4m_pp_savgol_create(&h, window, poly, deriv, 1.0,
                                     N4M_PP_SAVGOL_MIRROR, 0.0);
            w->h = h;
            break;
        }
        case PPK_DERIV1: {
            n4m_pp_first_derivative_handle_t* h = NULL;
            s = n4m_pp_first_derivative_create(&h, 1.0, 2);
            w->h = h;
            break;
        }
        case PPK_DERIV2: {
            n4m_pp_second_derivative_handle_t* h = NULL;
            s = n4m_pp_second_derivative_create(&h, 1.0, 2);
            w->h = h;
            break;
        }
        case PPK_DETREND: {
            int poly = n_params >= 1 ? (int)params[0] : 1;
            n4m_pp_detrend_handle_t* h = NULL;
            s = n4m_pp_detrend_create(&h, poly);
            w->h = h;
            break;
        }
        case PPK_GAUSSIAN: {
            double sigma = n_params >= 1 ? params[0] : 2.0;
            n4m_pp_gaussian_handle_t* h = NULL;
            s = n4m_pp_gaussian_create(&h, sigma, 0,
                                       N4M_PP_GAUSSIAN_REFLECT, 0.0, 4.0);
            w->h = h;
            break;
        }
        /* ---- A2a baseline correctors ---- */
        case PPK_ASLS: {
            double lam = n_params >= 1 ? params[0] : 1e6;
            double pp = n_params >= 2 ? params[1] : 1e-2;
            int max_it = n_params >= 3 ? (int)params[2] : 50;
            double tol = n_params >= 4 ? params[3] : 1e-3;
            n4m_pp_asls_handle_t* h = NULL;
            s = n4m_pp_asls_create(&h, lam, pp, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_AIRPLS: {
            double lam = n_params >= 1 ? params[0] : 1e6;
            int max_it = n_params >= 2 ? (int)params[1] : 50;
            double tol = n_params >= 3 ? params[2] : 1e-3;
            n4m_pp_airpls_handle_t* h = NULL;
            s = n4m_pp_airpls_create(&h, lam, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_ARPLS: {
            double lam = n_params >= 1 ? params[0] : 1e5;
            int max_it = n_params >= 2 ? (int)params[1] : 50;
            double tol = n_params >= 3 ? params[2] : 1e-3;
            n4m_pp_arpls_handle_t* h = NULL;
            s = n4m_pp_arpls_create(&h, lam, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_MODPOLY: {
            int poly = n_params >= 1 ? (int)params[0] : 2;
            int max_it = n_params >= 2 ? (int)params[1] : 250;
            double tol = n_params >= 3 ? params[2] : 1e-3;
            n4m_pp_modpoly_handle_t* h = NULL;
            s = n4m_pp_modpoly_create(&h, poly, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_IMODPOLY: {
            int poly = n_params >= 1 ? (int)params[0] : 2;
            int max_it = n_params >= 2 ? (int)params[1] : 250;
            double tol = n_params >= 3 ? params[2] : 1e-3;
            n4m_pp_imodpoly_handle_t* h = NULL;
            s = n4m_pp_imodpoly_create(&h, poly, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_SNIP: {
            int mhw = n_params >= 1 ? (int)params[0] : 20;
            n4m_pp_snip_handle_t* h = NULL;
            s = n4m_pp_snip_create(&h, mhw);
            w->h = h;
            break;
        }
        case PPK_ROLLING_BALL: {
            int hw = n_params >= 1 ? (int)params[0] : 20;
            int shw = n_params >= 2 ? (int)params[1] : 0;
            n4m_pp_rolling_ball_handle_t* h = NULL;
            s = n4m_pp_rolling_ball_create(&h, hw, shw);
            w->h = h;
            break;
        }
        case PPK_IASLS: {
            double lam = n_params >= 1 ? params[0] : 1e6;
            double pp = n_params >= 2 ? params[1] : 1e-2;
            int poly = n_params >= 3 ? (int)params[2] : 2;
            int max_it = n_params >= 4 ? (int)params[3] : 50;
            double tol = n_params >= 5 ? params[4] : 1e-3;
            n4m_pp_iasls_handle_t* h = NULL;
            s = n4m_pp_iasls_create(&h, lam, pp, poly, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_BEADS: {
            double lam0 = n_params >= 1 ? params[0] : 1e2;
            double lam1 = n_params >= 2 ? params[1] : 0.5;
            double lam2 = n_params >= 3 ? params[2] : 0.5;
            int max_it = n_params >= 4 ? (int)params[3] : 50;
            double tol = n_params >= 5 ? params[4] : 1e-3;
            n4m_pp_beads_handle_t* h = NULL;
            s = n4m_pp_beads_create(&h, lam0, lam1, lam2, max_it, tol);
            w->h = h;
            break;
        }
        /* ---- A2b signal conversions ---- */
        case PPK_TO_ABS: {
            int is_percent = n_params >= 1 ? (int)params[0] : 0;
            double eps = n_params >= 2 ? params[1] : 1e-8;
            int clip_neg = n_params >= 3 ? (int)params[2] : 1;
            n4m_pp_to_absorbance_handle_t* h = NULL;
            s = n4m_pp_to_absorbance_create(&h, is_percent, eps, clip_neg);
            w->h = h;
            break;
        }
        case PPK_FROM_ABS: {
            int is_percent = n_params >= 1 ? (int)params[0] : 0;
            n4m_pp_from_absorbance_handle_t* h = NULL;
            s = n4m_pp_from_absorbance_create(&h, is_percent);
            w->h = h;
            break;
        }
        case PPK_PCT_TO_FRAC: {
            n4m_pp_pct_to_frac_handle_t* h = NULL;
            s = n4m_pp_pct_to_frac_create(&h);
            w->h = h;
            break;
        }
        case PPK_FRAC_TO_PCT: {
            n4m_pp_frac_to_pct_handle_t* h = NULL;
            s = n4m_pp_frac_to_pct_create(&h);
            w->h = h;
            break;
        }
        case PPK_KUBELKA_MUNK: {
            int is_percent = n_params >= 1 ? (int)params[0] : 0;
            double eps = n_params >= 2 ? params[1] : 1e-8;
            n4m_pp_kubelka_munk_handle_t* h = NULL;
            s = n4m_pp_kubelka_munk_create(&h, is_percent, eps);
            w->h = h;
            break;
        }
        /* ---- A2c scatter / scaling / derivative ---- */
        case PPK_RNV: {
            int with_center = n_params >= 1 ? (int)params[0] : 1;
            int with_scale = n_params >= 2 ? (int)params[1] : 1;
            double k = n_params >= 3 ? params[2] : 1.4826;
            n4m_pp_rnv_handle_t* h = NULL;
            s = n4m_pp_rnv_create(&h, with_center, with_scale, k);
            w->h = h;
            break;
        }
        case PPK_LSNV: {
            int window = n_params >= 1 ? (int)params[0] : 11;
            int pad_mode = n_params >= 2 ? (int)params[1] : 0;
            double constant = n_params >= 3 ? params[2] : 0.0;
            n4m_pp_lsnv_handle_t* h = NULL;
            s = n4m_pp_lsnv_create(&h, window, pad_mode, constant);
            w->h = h;
            break;
        }
        case PPK_AREA: {
            int method = n_params >= 1 ? (int)params[0] : 1;
            n4m_pp_area_handle_t* h = NULL;
            s = n4m_pp_area_create(&h, method);
            w->h = h;
            break;
        }
        case PPK_NORRIS_WILLIAMS: {
            int segment = n_params >= 1 ? (int)params[0] : 5;
            int gap = n_params >= 2 ? (int)params[1] : 3;
            int order = n_params >= 3 ? (int)params[2] : 1;
            double delta = n_params >= 4 ? params[3] : 1.0;
            n4m_pp_norris_williams_handle_t* h = NULL;
            s = n4m_pp_norris_williams_create(&h, segment, gap, order, delta);
            w->h = h;
            break;
        }
        case PPK_LOG: {
            /* Stateless flavour only (auto_offset = 0); base 0 selects natural log. */
            double base = n_params >= 1 ? params[0] : 0.0;
            double offset = n_params >= 2 ? params[1] : 0.0;
            double min_value = n_params >= 3 ? params[2] : 1e-8;
            n4m_pp_log_handle_t* h = NULL;
            s = n4m_pp_log_create(&h, base, offset, 0, min_value);
            w->h = h;
            break;
        }
        case PPK_WAVELET_DENOISE: {
            int family = n_params >= 1 ? (int)params[0] : 0;
            int boundary = n_params >= 2 ? (int)params[1] : 0;
            int level = n_params >= 3 ? (int)params[2] : 3;
            int thr = n_params >= 4 ? (int)params[3] : 0;
            int noise = n_params >= 5 ? (int)params[4] : 0;
            n4m_pp_wavelet_denoise_handle_t* h = NULL;
            s = n4m_pp_wavelet_denoise_create(
                &h, (n4m_pp_wavelet_family_t)family,
                (n4m_pp_wavelet_boundary_t)boundary, level,
                (n4m_pp_wavelet_threshold_t)thr,
                (n4m_pp_wavelet_noise_t)noise);
            w->h = h;
            break;
        }
        default: s = N4M_ERR_INVALID_ARGUMENT; break;
    }
    if (s != N4M_OK || w->h == NULL) { free(w); return NULL; }
    return w;
}

/* Fit a stateful operator on training data (no-op for stateless operators). */
__attribute__((used))
int n4m_wasm_pp_fit(void* handle, const double* X, int n, int p) {
    if (handle == NULL || X == NULL) return N4M_ERR_NULL_POINTER;
    struct n4m_wasm_pp_s* w = (struct n4m_wasm_pp_s*)handle;
    if (w->kind == PPK_MSC) {
        return n4m_pp_msc_state_fit((n4m_pp_msc_state_t*)w->h, X, n, p);
    }
    return N4M_OK; /* stateless */
}

/* Transform X (n*p, row-major) into out (n*p). Shape-preserving. */
__attribute__((used))
int n4m_wasm_pp_transform(void* handle, const double* X, int n, int p,
                          double* out) {
    if (handle == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (n < 0 || p < 0) return N4M_ERR_INVALID_ARGUMENT;
    struct n4m_wasm_pp_s* w = (struct n4m_wasm_pp_s*)handle;
    if (w->kind == PPK_MSC) {
        return n4m_pp_msc_state_apply((n4m_pp_msc_state_t*)w->h, X, n, p, out);
    }
    n4m_matrix_view_t xv, ov;
    n4m_status_t vi = n4m_matrix_view_init_rowmajor(&xv, (void*)X, n, p, N4M_DTYPE_F64);
    if (vi != N4M_OK) return vi;
    vi = n4m_matrix_view_init_rowmajor(&ov, out, n, p, N4M_DTYPE_F64);
    if (vi != N4M_OK) return vi;
    switch (w->kind) {
        case PPK_SNV:
            return n4m_pp_snv_transform((const n4m_pp_snv_handle_t*)w->h, xv, ov);
        case PPK_SAVGOL:
            return n4m_pp_savgol_transform((const n4m_pp_savgol_handle_t*)w->h, xv, ov);
        case PPK_DERIV1:
            return n4m_pp_first_derivative_transform((const n4m_pp_first_derivative_handle_t*)w->h, xv, ov);
        case PPK_DERIV2:
            return n4m_pp_second_derivative_transform((const n4m_pp_second_derivative_handle_t*)w->h, xv, ov);
        case PPK_DETREND:
            return n4m_pp_detrend_transform((const n4m_pp_detrend_handle_t*)w->h, xv, ov);
        case PPK_GAUSSIAN:
            return n4m_pp_gaussian_transform((const n4m_pp_gaussian_handle_t*)w->h, xv, ov);
        /* ---- A2a baseline correctors ---- */
        case PPK_ASLS:
            return n4m_pp_asls_transform((const n4m_pp_asls_handle_t*)w->h, xv, ov);
        case PPK_AIRPLS:
            return n4m_pp_airpls_transform((const n4m_pp_airpls_handle_t*)w->h, xv, ov);
        case PPK_ARPLS:
            return n4m_pp_arpls_transform((const n4m_pp_arpls_handle_t*)w->h, xv, ov);
        case PPK_MODPOLY:
            return n4m_pp_modpoly_transform((const n4m_pp_modpoly_handle_t*)w->h, xv, ov);
        case PPK_IMODPOLY:
            return n4m_pp_imodpoly_transform((const n4m_pp_imodpoly_handle_t*)w->h, xv, ov);
        case PPK_SNIP:
            return n4m_pp_snip_transform((const n4m_pp_snip_handle_t*)w->h, xv, ov);
        case PPK_ROLLING_BALL:
            return n4m_pp_rolling_ball_transform((const n4m_pp_rolling_ball_handle_t*)w->h, xv, ov);
        case PPK_IASLS:
            return n4m_pp_iasls_transform((const n4m_pp_iasls_handle_t*)w->h, xv, ov);
        case PPK_BEADS:
            return n4m_pp_beads_transform((const n4m_pp_beads_handle_t*)w->h, xv, ov);
        /* ---- A2b signal conversions ---- */
        case PPK_TO_ABS:
            return n4m_pp_to_absorbance_transform((const n4m_pp_to_absorbance_handle_t*)w->h, xv, ov);
        case PPK_FROM_ABS:
            return n4m_pp_from_absorbance_transform((const n4m_pp_from_absorbance_handle_t*)w->h, xv, ov);
        case PPK_PCT_TO_FRAC:
            return n4m_pp_pct_to_frac_transform((const n4m_pp_pct_to_frac_handle_t*)w->h, xv, ov);
        case PPK_FRAC_TO_PCT:
            return n4m_pp_frac_to_pct_transform((const n4m_pp_frac_to_pct_handle_t*)w->h, xv, ov);
        case PPK_KUBELKA_MUNK:
            return n4m_pp_kubelka_munk_transform((const n4m_pp_kubelka_munk_handle_t*)w->h, xv, ov);
        /* ---- A2c scatter / scaling / derivative ---- */
        case PPK_RNV:
            return n4m_pp_rnv_transform((const n4m_pp_rnv_handle_t*)w->h, xv, ov);
        case PPK_LSNV:
            return n4m_pp_lsnv_transform((const n4m_pp_lsnv_handle_t*)w->h, xv, ov);
        case PPK_AREA:
            return n4m_pp_area_transform((const n4m_pp_area_handle_t*)w->h, xv, ov);
        case PPK_NORRIS_WILLIAMS:
            return n4m_pp_norris_williams_transform((const n4m_pp_norris_williams_handle_t*)w->h, xv, ov);
        case PPK_LOG:
            return n4m_pp_log_transform((const n4m_pp_log_handle_t*)w->h, xv, ov);
        case PPK_WAVELET_DENOISE:
            return n4m_pp_wavelet_denoise_transform((const n4m_pp_wavelet_denoise_handle_t*)w->h, xv, ov);
        default: return N4M_ERR_INVALID_ARGUMENT;
    }
}

/* Length (in doubles) of the serializable fitted state (0 for stateless ops). */
__attribute__((used))
int n4m_wasm_pp_state_len(void* handle) {
    if (handle == NULL) return 0;
    struct n4m_wasm_pp_s* w = (struct n4m_wasm_pp_s*)handle;
    if (w->kind == PPK_MSC) {
        return (int)n4m_pp_msc_state_n_features((const n4m_pp_msc_state_t*)w->h);
    }
    return 0;
}

/* Copy the fitted state into out (length == n4m_wasm_pp_state_len). */
__attribute__((used))
int n4m_wasm_pp_get_state(void* handle, double* out) {
    if (handle == NULL || out == NULL) return N4M_ERR_NULL_POINTER;
    struct n4m_wasm_pp_s* w = (struct n4m_wasm_pp_s*)handle;
    if (w->kind == PPK_MSC) {
        const n4m_pp_msc_state_t* st = (const n4m_pp_msc_state_t*)w->h;
        return n4m_pp_msc_state_get_reference(st, out,
                                              n4m_pp_msc_state_n_features(st));
    }
    return N4M_OK; /* stateless: nothing to copy */
}

/* Restore a fitted state from serialized doubles (predict-later). */
__attribute__((used))
int n4m_wasm_pp_set_state(void* handle, const double* state, int len) {
    if (handle == NULL) return N4M_ERR_NULL_POINTER;
    struct n4m_wasm_pp_s* w = (struct n4m_wasm_pp_s*)handle;
    if (w->kind == PPK_MSC) {
        return n4m_pp_msc_state_set_reference((n4m_pp_msc_state_t*)w->h,
                                              state, len);
    }
    return N4M_OK; /* stateless: nothing to restore */
}

/* Destroy an operator handle created by n4m_wasm_pp_create. */
__attribute__((used))
void n4m_wasm_pp_destroy(void* handle) {
    if (handle == NULL) return;
    struct n4m_wasm_pp_s* w = (struct n4m_wasm_pp_s*)handle;
    switch (w->kind) {
        case PPK_SNV:      n4m_pp_snv_destroy((n4m_pp_snv_handle_t*)w->h); break;
        case PPK_MSC:      n4m_pp_msc_state_free((n4m_pp_msc_state_t*)w->h); break;
        case PPK_SAVGOL:   n4m_pp_savgol_destroy((n4m_pp_savgol_handle_t*)w->h); break;
        case PPK_DERIV1:   n4m_pp_first_derivative_destroy((n4m_pp_first_derivative_handle_t*)w->h); break;
        case PPK_DERIV2:   n4m_pp_second_derivative_destroy((n4m_pp_second_derivative_handle_t*)w->h); break;
        case PPK_DETREND:  n4m_pp_detrend_destroy((n4m_pp_detrend_handle_t*)w->h); break;
        case PPK_GAUSSIAN: n4m_pp_gaussian_destroy((n4m_pp_gaussian_handle_t*)w->h); break;
        /* ---- A2a baseline correctors ---- */
        case PPK_ASLS:         n4m_pp_asls_destroy((n4m_pp_asls_handle_t*)w->h); break;
        case PPK_AIRPLS:       n4m_pp_airpls_destroy((n4m_pp_airpls_handle_t*)w->h); break;
        case PPK_ARPLS:        n4m_pp_arpls_destroy((n4m_pp_arpls_handle_t*)w->h); break;
        case PPK_MODPOLY:      n4m_pp_modpoly_destroy((n4m_pp_modpoly_handle_t*)w->h); break;
        case PPK_IMODPOLY:     n4m_pp_imodpoly_destroy((n4m_pp_imodpoly_handle_t*)w->h); break;
        case PPK_SNIP:         n4m_pp_snip_destroy((n4m_pp_snip_handle_t*)w->h); break;
        case PPK_ROLLING_BALL: n4m_pp_rolling_ball_destroy((n4m_pp_rolling_ball_handle_t*)w->h); break;
        case PPK_IASLS:        n4m_pp_iasls_destroy((n4m_pp_iasls_handle_t*)w->h); break;
        case PPK_BEADS:        n4m_pp_beads_destroy((n4m_pp_beads_handle_t*)w->h); break;
        /* ---- A2b signal conversions ---- */
        case PPK_TO_ABS:       n4m_pp_to_absorbance_destroy((n4m_pp_to_absorbance_handle_t*)w->h); break;
        case PPK_FROM_ABS:     n4m_pp_from_absorbance_destroy((n4m_pp_from_absorbance_handle_t*)w->h); break;
        case PPK_PCT_TO_FRAC:  n4m_pp_pct_to_frac_destroy((n4m_pp_pct_to_frac_handle_t*)w->h); break;
        case PPK_FRAC_TO_PCT:  n4m_pp_frac_to_pct_destroy((n4m_pp_frac_to_pct_handle_t*)w->h); break;
        case PPK_KUBELKA_MUNK: n4m_pp_kubelka_munk_destroy((n4m_pp_kubelka_munk_handle_t*)w->h); break;
        /* ---- A2c scatter / scaling / derivative ---- */
        case PPK_RNV:             n4m_pp_rnv_destroy((n4m_pp_rnv_handle_t*)w->h); break;
        case PPK_LSNV:            n4m_pp_lsnv_destroy((n4m_pp_lsnv_handle_t*)w->h); break;
        case PPK_AREA:            n4m_pp_area_destroy((n4m_pp_area_handle_t*)w->h); break;
        case PPK_NORRIS_WILLIAMS: n4m_pp_norris_williams_destroy((n4m_pp_norris_williams_handle_t*)w->h); break;
        case PPK_LOG:             n4m_pp_log_destroy((n4m_pp_log_handle_t*)w->h); break;
        case PPK_WAVELET_DENOISE: n4m_pp_wavelet_denoise_destroy((n4m_pp_wavelet_denoise_handle_t*)w->h); break;
        default: break;
    }
    free(w);
}
