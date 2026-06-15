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
    MK_BAGGING_PLS, MK_BOOSTING_PLS, MK_RANDOM_SUBSPACE_PLS,
    /* Tier B extension — additional coeff-triple fits */
    MK_MIR_PLS, MK_MB_PLS, MK_MISSING_NIPALS,
    /* Tier B extension 2 — ECR (alpha PCR↔PLS) + O2PLS (orthogonal PLS) */
    MK_ECR, MK_O2PLS
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
    if (strcmp(model, "MIRPLS") == 0) return MK_MIR_PLS;
    if (strcmp(model, "MBPLS") == 0) return MK_MB_PLS;
    if (strcmp(model, "MissingAwareNIPALS") == 0) return MK_MISSING_NIPALS;
    if (strcmp(model, "ECR") == 0) return MK_ECR;
    if (strcmp(model, "O2PLS") == 0) return MK_O2PLS;
    return MK_NONE;
}

/* Tier A — build a Config for the algorithm-enum family, fit via the model API,
 * and copy out the centred coefficient triple.
 *
 * The PLS/PCR family does NOT have a genuine affine intercept: its prediction
 * is the centred form y_mean + (x - x_mean).B. (libn4m's N4M_MODEL_INTERCEPT
 * for these models just aliases y_mean, which is NOT a constant term usable as
 * intercept + x.B — exposing it as an intercept would double-count the
 * centring.) So `has_intercept_out` is set to 0 and intercept_out is zero-filled
 * for shape only; callers must predict via the centred form. */
static int n4m_wasm_model_fit_tier_a(
        int kind, const double* x, const double* y,
        int n, int p, int q, int n_components,
        double* coefficients_out, double* x_mean_out, double* y_mean_out,
        double* intercept_out, int* has_intercept_out,
        double* predictions_out) {
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
    /* PLS/PCR family: centred form only — no genuine affine intercept. */
    if (intercept_out != NULL) {
        for (int t = 0; t < q; ++t) intercept_out[t] = 0.0;
    }
    if (has_intercept_out != NULL) *has_intercept_out = 0;
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
 * exactly `expect` doubles. Returns 1 when the matrix was present and copied,
 * 0 otherwise (leaving the buffer untouched). */
static int copy_result_matrix(const n4m_method_result_t* r, const char* name,
                              double* out, size_t expect) {
    if (out == NULL) return 0;
    const double* data = NULL;
    int64_t rows = 0, cols = 0;
    if (n4m_method_result_get_double_matrix(r, name, &data, &rows, &cols) == N4M_OK
        && data != NULL && (size_t)(rows * cols) == expect) {
        memcpy(out, data, expect * sizeof(double));
        return 1;
    }
    return 0;
}

/* Tier B — call the standalone fit for `kind`, then read the coeff triple
 * (+ intercept for Ridge) out of the returned method-result.
 *
 * Every PLS-based Tier-B fit (RidgePLS, ContinuumRegression, RobustPLS, CPPLS,
 * SparseSIMPLS, GroupSparsePLS, FusedSparsePLS, BaggingPLS, BoostingPLS,
 * RandomSubspacePLS) reads the latent-component count from `cfg.n_components`
 * (a fresh config defaults to 2), so the selected `n_components` is applied to
 * the config before the fit. Pure Ridge (MK_RIDGE) is a closed-form L2 solve
 * with no latent components — it legitimately ignores `n_components`. */
static int n4m_wasm_model_fit_tier_b(
        int kind, const double* params, int n_params,
        const double* x, const double* y,
        int n, int p, int q, int n_components,
        double* coefficients_out, double* x_mean_out, double* y_mean_out,
        double* intercept_out, int* has_intercept_out,
        double* predictions_out) {
    n4m_matrix_view_t xv, yv;
    n4m_matrix_view_init_rowmajor(&xv, (void*)x, n, p, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&yv, (void*)y, n, q, N4M_DTYPE_F64);

    n4m_context_t* ctx = NULL;
    n4m_status_t s = n4m_context_create(&ctx);
    if (s != N4M_OK) return s;
    n4m_config_t* cfg = NULL;
    s = n4m_config_create(&cfg);
    if (s != N4M_OK) { n4m_context_destroy(ctx); return s; }

    /* Apply the requested latent-component count to every PLS-based Tier-B
     * fit. MK_RIDGE is a pure (component-free) closed-form solve and keeps the
     * config default. */
    if (kind != MK_RIDGE) {
        s = n4m_config_set_n_components(cfg, n_components);
        if (s != N4M_OK) {
            n4m_config_destroy(cfg);
            n4m_context_destroy(ctx);
            return s;
        }
    }

    n4m_method_result_t* res = NULL;
    switch (kind) {
        case MK_RIDGE: {
            double lambda = n_params >= 1 ? params[0] : 1.0;
            /* n4m_estimators_ridge_fit reads cfg.ridge_lambda when lambdas==NULL; pass the
             * single override array so the param vector is authoritative. */
            s = n4m_estimators_ridge_fit(ctx, cfg, &xv, &yv, &lambda, 1, &res);
            break;
        }
        case MK_RIDGE_PLS: {
            double ridge_lambda = n_params >= 1 ? params[0] : 1.0;
            s = n4m_estimators_ridge_pls_fit(ctx, cfg, &xv, &yv, ridge_lambda, &res);
            break;
        }
        case MK_CONTINUUM: {
            double tau = n_params >= 1 ? params[0] : 0.5;
            s = n4m_estimators_continuum_regression_fit(ctx, cfg, &xv, &yv, tau, &res);
            break;
        }
        case MK_ROBUST_PLS: {
            double huber_k = n_params >= 1 ? params[0] : 1.345;
            int max_irls = n_params >= 2 ? (int)params[1] : 5;
            s = n4m_estimators_robust_pls_fit(ctx, cfg, &xv, &yv, huber_k, max_irls, &res);
            break;
        }
        case MK_CPPLS: {
            double gamma = n_params >= 1 ? params[0] : 0.5;
            s = n4m_estimators_cppls_fit(ctx, cfg, &xv, &yv, gamma, &res);
            break;
        }
        case MK_SPARSE_SIMPLS: {
            double sparsity = n_params >= 1 ? params[0] : 0.0;
            s = n4m_estimators_sparse_simpls_fit(ctx, cfg, &xv, &yv, sparsity, &res);
            break;
        }
        case MK_GROUP_SPARSE_PLS: {
            double group_lambda = n_params >= 1 ? params[0] : 0.0;
            /* No group ids over this scalar surface — one group covers all
             * features (group_assignment all-zero). */
            int32_t* groups = (int32_t*)calloc((size_t)(p > 0 ? p : 1),
                                                sizeof(int32_t));
            if (groups == NULL) { s = N4M_ERR_OUT_OF_MEMORY; break; }
            s = n4m_estimators_group_sparse_pls_fit(ctx, cfg, &xv, &yv, groups, p,
                                         group_lambda, &res);
            free(groups);
            break;
        }
        case MK_FUSED_SPARSE_PLS: {
            double l1 = n_params >= 1 ? params[0] : 0.0;
            double fusion = n_params >= 2 ? params[1] : 0.0;
            s = n4m_estimators_fused_sparse_pls_fit(ctx, cfg, &xv, &yv, l1, fusion, &res);
            break;
        }
        case MK_BAGGING_PLS: {
            int n_estimators = n_params >= 1 ? (int)params[0] : 10;
            uint64_t seed = n_params >= 2 ? (uint64_t)params[1] : 0;
            s = n4m_ensemble_bagging_pls_fit(ctx, cfg, &xv, &yv, n_estimators, seed, &res);
            break;
        }
        case MK_BOOSTING_PLS: {
            int n_estimators = n_params >= 1 ? (int)params[0] : 10;
            double lr = n_params >= 2 ? params[1] : 0.1;
            s = n4m_ensemble_boosting_pls_fit(ctx, cfg, &xv, &yv, n_estimators, lr, &res);
            break;
        }
        case MK_RANDOM_SUBSPACE_PLS: {
            int n_estimators = n_params >= 1 ? (int)params[0] : 10;
            int feats = n_params >= 2 ? (int)params[1] : (p > 1 ? p / 2 : 1);
            uint64_t seed = n_params >= 3 ? (uint64_t)params[2] : 0;
            s = n4m_ensemble_random_subspace_pls_fit(ctx, cfg, &xv, &yv, n_estimators,
                                            feats, seed, &res);
            break;
        }
        case MK_MIR_PLS: {
            /* Multiple-Inverse-Regression PLS: SIMPLS on (Y, X) then
             * pseudo-inverse → X→Y coefficients. No extra params; centred
             * coefficient triple (coefficients/x_mean/y_mean). */
            s = n4m_estimators_mir_pls_fit(ctx, cfg, &xv, &yv, &res);
            break;
        }
        case MK_MB_PLS: {
            /* Block-weighted multi-block PLS over a SINGLE block (all p
             * features). It returns input-space coefficients PLUS a genuine
             * affine intercept (and x_scale), so it predicts on RAW X via
             * intercept + x.B — the same explicit-intercept path as Ridge. */
            int64_t block_sizes[1];
            block_sizes[0] = (int64_t)p;
            s = n4m_estimators_mb_pls_fit(ctx, cfg, &xv, &yv, block_sizes, 1, &res);
            break;
        }
        case MK_MISSING_NIPALS: {
            /* Missing-aware NIPALS PLS: same centred coefficient triple as a
             * regular SIMPLS PLS but tolerant of NaN entries in X. */
            s = n4m_estimators_missing_aware_nipals_fit(ctx, cfg, &xv, &yv, &res);
            break;
        }
        case MK_ECR: {
            /* Elastic Component Regression: alpha interpolates PCR (0) ↔ PLS (1).
             * Emits the centred coefficient triple (coefficients/x_mean/y_mean). */
            double alpha = n_params >= 1 ? params[0] : 0.5;
            s = n4m_estimators_ecr_fit(ctx, cfg, &xv, &yv, alpha, &res);
            break;
        }
        case MK_O2PLS: {
            /* Bidirectional orthogonal PLS (Trygg & Wold). Takes its own
             * component counts (NOT cfg.n_components); emits the centred
             * coefficient triple. params = [n_predictive, n_x_orth, n_y_orth]. */
            int32_t n_pred = n_params >= 1 ? (int32_t)params[0] : 2;
            int32_t n_xo = n_params >= 2 ? (int32_t)params[1] : 1;
            int32_t n_yo = n_params >= 3 ? (int32_t)params[2] : 1;
            if (n_pred < 1) n_pred = 1;
            if (n_xo < 0) n_xo = 0;
            if (n_yo < 0) n_yo = 0;
            s = n4m_estimators_o2pls_fit(ctx, cfg, &xv, &yv, n_pred, n_xo, n_yo, &res);
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
    /* Only the standalone fits that emit a genuine affine "intercept" matrix
     * (currently just Ridge: intercept = y_mean - x_mean.B_descaled) set the
     * has_intercept flag. The PLS-based Tier-B fits expose only the centred
     * triple, so intercept_out is zero-filled for shape and the flag stays 0. */
    int has_intercept = 0;
    if (intercept_out != NULL) {
        for (int t = 0; t < q; ++t) intercept_out[t] = 0.0;
        has_intercept = copy_result_matrix(res, "intercept", intercept_out,
                                           (size_t)q);
    } else {
        const double* idata = NULL;
        int64_t irows = 0, icols = 0;
        has_intercept = (n4m_method_result_get_double_matrix(
                             res, "intercept", &idata, &irows, &icols) == N4M_OK
                         && idata != NULL && (size_t)(irows * icols) == (size_t)q);
    }
    if (has_intercept_out != NULL) *has_intercept_out = has_intercept;
    if (predictions_out != NULL) {
        copy_result_matrix(res, "predictions", predictions_out,
                           (size_t)n * (size_t)q);
    }
    n4m_method_result_destroy(res);
    n4m_context_destroy(ctx);
    return N4M_OK;
}

/* Fit any coeff-based model by token. Writes coefficients_out (p*q row-major),
 * x_mean_out (p), y_mean_out (q), and — when non-NULL — intercept_out (q),
 * has_intercept_out (1 when the model produced a genuine affine intercept,
 * 0 for the centred PLS/PCR family), and predictions_out (n*q).
 *
 * Intercept contract: only models with a real constant term (currently Ridge)
 * set has_intercept_out=1 and a meaningful intercept_out. The PLS/PCR family
 * (Tier A) and the PLS-based Tier-B fits predict via the centred form
 * y_mean + (x - x_mean).B and have NO affine intercept — their intercept_out is
 * zero-filled and has_intercept_out=0, so a caller must not add it to x.B.
 *
 * Returns N4M_ERR_NOT_IMPLEMENTED for unknown tokens. */
__attribute__((used))
int n4m_wasm_model_fit(const char* model,
                       const double* params, int n_params,
                       const double* x, const double* y,
                       int n, int p, int q, int n_components,
                       double* coefficients_out,
                       double* x_mean_out,
                       double* y_mean_out,
                       double* intercept_out,
                       int* has_intercept_out,
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
    if (has_intercept_out != NULL) *has_intercept_out = 0;
    int kind = model_kind_for(model);
    if (kind == MK_NONE) return N4M_ERR_NOT_IMPLEMENTED;
    if (kind <= MK_PLS_DA) {
        return n4m_wasm_model_fit_tier_a(kind, x, y, n, p, q, n_components,
                                         coefficients_out, x_mean_out,
                                         y_mean_out, intercept_out,
                                         has_intercept_out, predictions_out);
    }
    return n4m_wasm_model_fit_tier_b(kind, params, n_params, x, y, n, p, q,
                                     n_components, coefficients_out, x_mean_out,
                                     y_mean_out, intercept_out,
                                     has_intercept_out, predictions_out);
}

/* Predict from a fitted coefficient triple.
 *
 * Two prediction conventions are supported via the `intercept` arg:
 *   intercept == NULL : centred form, pred = y_mean + (x - x_mean).B
 *                       This is the ONLY correct form for the PLS/PCR family and
 *                       the PLS-based Tier-B fits: they have no affine intercept
 *                       and their own in-sample `predictions` use exactly this.
 *   intercept != NULL : explicit-intercept form, pred = intercept + x.B
 *                       (x_mean / y_mean ignored). Use this ONLY for models that
 *                       reported has_intercept=1 (currently Ridge), where
 *                       intercept already folds the centring (and any feature
 *                       scaling) into a genuine constant term.
 *
 * For Ridge the two forms agree numerically because its intercept is
 * y_mean - x_mean.B (on the de-scaled coefficient scale); for the centred-only
 * models passing a non-NULL intercept would double-count the constant, so the
 * generic JS path always uses the centred form (intercept = NULL). */
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
 * AOM-PLS / POP-PLS — operator-adaptive PLS (raw-double-pointer surface for JS)
 * ----------------------------------------------------------------------------
 * Both screen a bank of strict-linear preprocessing operators by internal
 * k-fold CV and fit SIMPLS, then export INPUT-SPACE coefficients so the model
 * predicts on RAW X via the affine form  y = intercept + X.B  (no replay of the
 * selected operator). AOM-PLS (global) picks ONE operator for the whole model;
 * POP-PLS (per-component) picks one operator PER latent component. The numerics
 * stay 100% in libn4m (n4m_model_selection_aom_pls_select / n4m_model_selection_pop_pls_select);
 * this is only bank/plan construction + marshalling.
 *
 * Default bank (when operator_kinds == NULL): a small set of STRICT-linear
 * operators that all accept default (empty) params —
 *   IDENTITY, DETREND_POLY, SAVGOL_SMOOTH, SAVGOL_DERIVATIVE, FINITE_DIFFERENCE.
 * Non-strict operators (e.g. SNV/MSC) are rejected by the selector, so they are
 * intentionally NOT in the default set. A caller-supplied operator_kinds array
 * (n4m_operator_kind_t values) overrides the default bank; every kind is added
 * with default params.
 *
 * The validation plan is k contiguous balanced folds over the n rows (the same
 * deterministic partition the C++ AOM tests use); `seed` is accepted for API
 * symmetry with the other stochastic fits but the contiguous fold partition is
 * deterministic so it does not consume the seed.
 * ==========================================================================*/

/* Shared setup for the AOM/POP selectors: validate + clamp max_components/k,
 * build the SIMPLS-regression config, build the operator bank (custom or
 * default strict set), and build a k contiguous-fold validation plan. On
 * success all four handles are owned by the caller, which MUST destroy them
 * (ctx, cfg, bank, plan) regardless of the later selector outcome. On failure
 * every handle is destroyed here and all out-params are set to NULL. */
static int n4m_wasm_aom_setup(int n, int p, int q,
                              int* max_components, int n_folds,
                              const int* operator_kinds, int n_ops,
                              n4m_matrix_view_t* xv, n4m_matrix_view_t* yv,
                              const double* x, const double* y,
                              n4m_context_t** out_ctx, n4m_config_t** out_cfg,
                              n4m_operator_bank_t** out_bank,
                              n4m_validation_plan_t** out_plan) {
    *out_ctx = NULL; *out_cfg = NULL; *out_bank = NULL; *out_plan = NULL;
    if (n < 4 || p < 1 || q < 1 || *max_components < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (n_ops < 0 || (n_ops > 0 && operator_kinds == NULL)) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    /* The selector requires max_components < n_rows and <= n_features. */
    if (*max_components > p) *max_components = p;
    if (*max_components >= n) *max_components = n - 1;
    if (*max_components < 1) return N4M_ERR_INVALID_ARGUMENT;
    /* k-fold: at least 2 folds, at most n. */
    int k = n_folds;
    if (k < 2) k = 2;
    if (k > n) k = n;

    n4m_matrix_view_init_rowmajor(xv, (void*)x, n, p, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(yv, (void*)y, n, q, N4M_DTYPE_F64);

    n4m_context_t* ctx = NULL;
    n4m_status_t s = n4m_context_create(&ctx);
    if (s != N4M_OK) return s;

    n4m_config_t* cfg = NULL;
    s = n4m_config_create(&cfg);
    if (s != N4M_OK) { n4m_context_destroy(ctx); return s; }
    /* Phase 6 AOM kernel: strict-linear operators + SIMPLS regression. */
    n4m_config_set_algorithm(cfg, N4M_ALGO_PLS_REGRESSION);
    n4m_config_set_solver(cfg, N4M_SOLVER_SIMPLS);
    n4m_config_set_deflation(cfg, N4M_DEFLATION_REGRESSION);
    n4m_config_set_scale_x(cfg, 0);

    /* ---- Operator bank ---- */
    n4m_operator_bank_t* bank = NULL;
    s = n4m_operator_bank_create(&bank);
    if (s != N4M_OK) {
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
        return s;
    }
    if (n_ops > 0) {
        for (int i = 0; i < n_ops && s == N4M_OK; ++i) {
            s = n4m_operator_bank_add(
                bank, (n4m_operator_kind_t)operator_kinds[i], NULL, 0);
        }
    } else {
        static const n4m_operator_kind_t kDefault[] = {
            N4M_OP_IDENTITY,
            N4M_OP_DETREND_POLY,
            N4M_OP_SAVGOL_SMOOTH,
            N4M_OP_SAVGOL_DERIVATIVE,
            N4M_OP_FINITE_DIFFERENCE,
        };
        const int n_default = (int)(sizeof(kDefault) / sizeof(kDefault[0]));
        for (int i = 0; i < n_default && s == N4M_OK; ++i) {
            s = n4m_operator_bank_add(bank, kDefault[i], NULL, 0);
        }
    }
    if (s != N4M_OK) {
        n4m_operator_bank_destroy(bank);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
        return s;
    }

    /* ---- Validation plan: k contiguous balanced folds over n rows ---- */
    n4m_validation_plan_t* plan = NULL;
    s = n4m_validation_plan_create(&plan);
    if (s == N4M_OK) s = n4m_validation_plan_set_n_samples(plan, n);
    int64_t* train = NULL;
    int64_t* test = NULL;
    if (s == N4M_OK) {
        train = (int64_t*)malloc((size_t)n * sizeof(int64_t));
        test = (int64_t*)malloc((size_t)n * sizeof(int64_t));
        if (train == NULL || test == NULL) s = N4M_ERR_OUT_OF_MEMORY;
    }
    for (int fold = 0; fold < k && s == N4M_OK; ++fold) {
        int n_train = 0, n_test = 0;
        for (int row = 0; row < n; ++row) {
            /* contiguous balanced partition: row -> fold (row * k) / n */
            if ((row * k) / n == fold) {
                test[n_test++] = row;
            } else {
                train[n_train++] = row;
            }
        }
        if (n_train < 1 || n_test < 1) { s = N4M_ERR_INVALID_ARGUMENT; break; }
        s = n4m_validation_plan_add_fold(plan, train, n_train, test, n_test);
    }
    free(train);
    free(test);
    if (s != N4M_OK) {
        if (plan != NULL) n4m_validation_plan_destroy(plan);
        n4m_operator_bank_destroy(bank);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
        return s;
    }

    *out_ctx = ctx;
    *out_cfg = cfg;
    *out_bank = bank;
    *out_plan = plan;
    return N4M_OK;
}

/* AOM-PLS (global): one operator for the whole model.
 *
 * Writes (all row-major, F64):
 *   coefficients_out (p * q)  input-space coefficients B
 *   intercept_out    (q)      input-space intercept
 *   selected_op_out  (int32)  bank index of the winning operator
 *   best_score_out   (double) best internal-CV score of the winner
 * Returns 0 on success, a N4M_ERR_* otherwise. On any failure every handle is
 * destroyed (no leak) and the out buffers are left untouched. */
__attribute__((used))
int n4m_wasm_aom_fit(const double* x, const double* y,
                     int n, int p, int q,
                     int max_components, int n_folds, int seed,
                     const int* operator_kinds, int n_ops,
                     double* coefficients_out,
                     double* intercept_out,
                     int* selected_op_out,
                     double* best_score_out) {
    (void)seed; /* contiguous-fold partition is deterministic */
    if (x == NULL || y == NULL || coefficients_out == NULL ||
        intercept_out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    n4m_matrix_view_t xv, yv;
    n4m_context_t* ctx = NULL;
    n4m_config_t* cfg = NULL;
    n4m_operator_bank_t* bank = NULL;
    n4m_validation_plan_t* plan = NULL;
    int mc = max_components;
    n4m_status_t s = n4m_wasm_aom_setup(n, p, q, &mc, n_folds, operator_kinds,
                                        n_ops, &xv, &yv, x, y, &ctx, &cfg,
                                        &bank, &plan);
    if (s != N4M_OK) return s;

    /* ---- Run the global AOM selector ---- */
    n4m_aom_global_result_t* res = NULL;
    s = n4m_model_selection_aom_pls_select(ctx, cfg, bank, &xv, &yv, plan, mc, &res);
    if (s != N4M_OK || res == NULL) {
        if (res != NULL) n4m_model_selection_aom_pls_result_destroy(res);
        n4m_validation_plan_destroy(plan);
        n4m_operator_bank_destroy(bank);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
        return s != N4M_OK ? s : N4M_ERR_INVALID_ARGUMENT;
    }

    /* ---- Copy out input-space coefficients + intercept + selection ---- */
    s = N4M_OK;
    const double* coeff = NULL;
    int64_t crows = 0, ccols = 0;
    if (n4m_model_selection_aom_pls_result_get_input_coefficients(
            res, &coeff, &crows, &ccols) == N4M_OK &&
        coeff != NULL && crows == p && ccols == q) {
        memcpy(coefficients_out, coeff, (size_t)p * (size_t)q * sizeof(double));
    } else {
        s = N4M_ERR_INVALID_ARGUMENT;
    }
    const double* inter = NULL;
    int64_t irows = 0, icols = 0;
    if (s == N4M_OK &&
        n4m_model_selection_aom_pls_result_get_intercept(res, &inter, &irows, &icols) ==
            N4M_OK &&
        inter != NULL && irows == 1 && icols == q) {
        memcpy(intercept_out, inter, (size_t)q * sizeof(double));
    } else if (s == N4M_OK) {
        s = N4M_ERR_INVALID_ARGUMENT;
    }
    if (s == N4M_OK && selected_op_out != NULL) {
        int32_t sel = -1;
        n4m_model_selection_aom_pls_result_get_selected_operator_index(res, &sel);
        *selected_op_out = (int)sel;
    }
    if (s == N4M_OK && best_score_out != NULL) {
        double sc = 0.0;
        n4m_model_selection_aom_pls_result_get_best_score(res, &sc);
        *best_score_out = sc;
    }

    n4m_model_selection_aom_pls_result_destroy(res);
    n4m_validation_plan_destroy(plan);
    n4m_operator_bank_destroy(bank);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
    return s;
}

/* POP-PLS (per-component): one operator picked PER latent component.
 *
 * Same bank/plan construction as AOM; runs n4m_model_selection_pop_pls_select, then
 * exports INPUT-SPACE coefficients + intercept (predict on RAW X via the same
 * affine path) plus the per-component selected-operator list (bank index of the
 * operator picked at each of the selected n_components components).
 *
 * Writes (all row-major, F64):
 *   coefficients_out (p * q)  input-space coefficients B
 *   intercept_out    (q)      input-space intercept
 *   selected_ops_out (int32 * max_components) per-component bank indices; only
 *                    the first *n_selected_out entries are meaningful (the
 *                    selected prefix length). Pass NULL to skip.
 *   n_selected_out   (int32)  number of selected components (= meaningful
 *                    entries in selected_ops_out). Pass NULL to skip.
 *   best_score_out   (double) best internal-CV prefix score. Pass NULL to skip.
 * Returns 0 on success, a N4M_ERR_* otherwise. On any failure every handle is
 * destroyed (no leak) and the out buffers are left untouched. */
__attribute__((used))
int n4m_wasm_pop_fit(const double* x, const double* y,
                     int n, int p, int q,
                     int max_components, int n_folds, int seed,
                     const int* operator_kinds, int n_ops,
                     double* coefficients_out,
                     double* intercept_out,
                     int* selected_ops_out,
                     int* n_selected_out,
                     double* best_score_out) {
    (void)seed; /* contiguous-fold partition is deterministic */
    if (x == NULL || y == NULL || coefficients_out == NULL ||
        intercept_out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    n4m_matrix_view_t xv, yv;
    n4m_context_t* ctx = NULL;
    n4m_config_t* cfg = NULL;
    n4m_operator_bank_t* bank = NULL;
    n4m_validation_plan_t* plan = NULL;
    int mc = max_components;
    n4m_status_t s = n4m_wasm_aom_setup(n, p, q, &mc, n_folds, operator_kinds,
                                        n_ops, &xv, &yv, x, y, &ctx, &cfg,
                                        &bank, &plan);
    if (s != N4M_OK) return s;

    /* ---- Run the per-component (POP) AOM selector ---- */
    n4m_aom_per_component_result_t* res = NULL;
    s = n4m_model_selection_pop_pls_select(ctx, cfg, bank, &xv, &yv, plan, mc, &res);
    if (s != N4M_OK || res == NULL) {
        if (res != NULL) n4m_model_selection_pop_pls_result_destroy(res);
        n4m_validation_plan_destroy(plan);
        n4m_operator_bank_destroy(bank);
        n4m_config_destroy(cfg);
        n4m_context_destroy(ctx);
        return s != N4M_OK ? s : N4M_ERR_INVALID_ARGUMENT;
    }

    /* ---- Copy out input-space coefficients + intercept ---- */
    s = N4M_OK;
    const double* coeff = NULL;
    int64_t crows = 0, ccols = 0;
    if (n4m_model_selection_pop_pls_result_get_input_coefficients(
            res, &coeff, &crows, &ccols) == N4M_OK &&
        coeff != NULL && crows == p && ccols == q) {
        memcpy(coefficients_out, coeff, (size_t)p * (size_t)q * sizeof(double));
    } else {
        s = N4M_ERR_INVALID_ARGUMENT;
    }
    const double* inter = NULL;
    int64_t irows = 0, icols = 0;
    if (s == N4M_OK &&
        n4m_model_selection_pop_pls_result_get_intercept(res, &inter, &irows,
                                                   &icols) == N4M_OK &&
        inter != NULL && irows == 1 && icols == q) {
        memcpy(intercept_out, inter, (size_t)q * sizeof(double));
    } else if (s == N4M_OK) {
        s = N4M_ERR_INVALID_ARGUMENT;
    }

    /* ---- Per-component selected operators (one bank index per selected
     * component) + selected component count + best prefix score ---- */
    int32_t n_selected = 0;
    if (s == N4M_OK) {
        n4m_model_selection_pop_pls_result_get_selected_n_components(res, &n_selected);
        if (n_selected < 0) n_selected = 0;
        if (n_selected > mc) n_selected = mc;
    }
    if (s == N4M_OK && selected_ops_out != NULL) {
        for (int k = 0; k < mc; ++k) selected_ops_out[k] = -1;
        const int32_t* sel = NULL;
        int32_t sel_len = 0;
        if (n4m_model_selection_pop_pls_result_get_selected_operator_indices(
                res, &sel, &sel_len) == N4M_OK && sel != NULL) {
            int copy = (int)(sel_len < n_selected ? sel_len : n_selected);
            for (int k = 0; k < copy; ++k) selected_ops_out[k] = (int)sel[k];
        }
    }
    if (s == N4M_OK && n_selected_out != NULL) {
        *n_selected_out = (int)n_selected;
    }
    if (s == N4M_OK && best_score_out != NULL) {
        double sc = 0.0;
        n4m_model_selection_pop_pls_result_get_best_score(res, &sc);
        *best_score_out = sc;
    }

    n4m_model_selection_pop_pls_result_destroy(res);
    n4m_validation_plan_destroy(plan);
    n4m_operator_bank_destroy(bank);
    n4m_config_destroy(cfg);
    n4m_context_destroy(ctx);
    return s;
}

/* ============================================================================
 * AOM-family ensembles (raw-double-pointer surface for JS)
 * ----------------------------------------------------------------------------
 * Two more members of the AOM family beyond AOM-PLS / POP-PLS. Both build their
 * own strict-linear operator/chain bank internally (selected by `profile`:
 * 0 = compact, 1 = wide), run an internal CV over `cv` contiguous balanced folds
 * (we hand them an explicit per-sample fold_ids vector = (i*cv)/n, matching the
 * AOM/POP partition), and return INPUT-SPACE coefficients + a genuine affine
 * intercept — so JS predicts on RAW X via the explicit-intercept path (like
 * AOM-PLS / Ridge). Numerics stay 100% in libn4m.
 * ==========================================================================*/

/* Build the contiguous balanced per-sample fold-id vector (i*k)/n. Caller frees.
 * Returns NULL on OOM. */
static int32_t* n4m_wasm_make_fold_ids(int n, int k) {
    int32_t* ids = (int32_t*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    if (ids == NULL) return NULL;
    for (int i = 0; i < n; ++i) ids[i] = (int32_t)(((int64_t)i * (int64_t)k) / (int64_t)n);
    return ids;
}

/* AOM Ridge simplex blender (n4m_ensemble_aom_ridge_blender_fit). OOF-blends a
 * set of (strict-linear chain, ridge-λ) candidates and returns the weighted
 * final input-space coefficients (p*q) + intercept (q). `ridge_lambdas` may be
 * NULL to use a default log grid. */
__attribute__((used))
int n4m_wasm_aom_ridge_fit(const double* x, const double* y,
                           int n, int p, int q,
                           int profile, int cv,
                           const double* ridge_lambdas, int n_lambdas,
                           double regularizer,
                           double* coefficients_out, double* intercept_out) {
    if (x == NULL || y == NULL || coefficients_out == NULL || intercept_out == NULL)
        return N4M_ERR_NULL_POINTER;
    if (n < 2 || p < 1 || q < 1) return N4M_ERR_INVALID_ARGUMENT;
    int k = cv; if (k < 2) k = 2; if (k > n) k = n;

    n4m_matrix_view_t xv, yv;
    n4m_matrix_view_init_rowmajor(&xv, (void*)x, n, p, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&yv, (void*)y, n, q, N4M_DTYPE_F64);

    n4m_context_t* ctx = NULL;
    n4m_status_t s = n4m_context_create(&ctx);
    if (s != N4M_OK) return s;
    n4m_config_t* cfg = NULL;
    s = n4m_config_create(&cfg);
    if (s != N4M_OK) { n4m_context_destroy(ctx); return s; }

    int32_t* fold_ids = n4m_wasm_make_fold_ids(n, k);
    if (fold_ids == NULL) {
        n4m_config_destroy(cfg); n4m_context_destroy(ctx);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    static const double kDefaultLambdas[] = {0.01, 0.1, 1.0, 10.0, 100.0};
    const double* lambdas = (ridge_lambdas != NULL && n_lambdas > 0) ? ridge_lambdas : kDefaultLambdas;
    int64_t nl = (ridge_lambdas != NULL && n_lambdas > 0)
                     ? (int64_t)n_lambdas
                     : (int64_t)(sizeof(kDefaultLambdas) / sizeof(kDefaultLambdas[0]));

    n4m_method_result_t* res = NULL;
    s = n4m_ensemble_aom_ridge_blender_fit(ctx, cfg, &xv, &yv,
                                           (int32_t)profile, (int32_t)k,
                                           fold_ids, (int64_t)n,
                                           lambdas, nl, regularizer, &res);
    free(fold_ids);
    n4m_config_destroy(cfg);
    if (s != N4M_OK || res == NULL) {
        if (res != NULL) n4m_method_result_destroy(res);
        n4m_context_destroy(ctx);
        return s != N4M_OK ? s : N4M_ERR_INVALID_ARGUMENT;
    }
    int ok_c = copy_result_matrix(res, "input_coefficients",
                                  coefficients_out, (size_t)p * (size_t)q);
    int ok_i = copy_result_matrix(res, "intercept", intercept_out, (size_t)q);
    n4m_method_result_destroy(res);
    n4m_context_destroy(ctx);
    return (ok_c && ok_i) ? N4M_OK : N4M_ERR_INVALID_ARGUMENT;
}

/* AOM operator PLS score stack with Ridge head
 * (n4m_ensemble_aom_operator_pls_stack_fit). CV-selects (n_components, alpha)
 * over the component grid [1..max_components] and `alphas`, refits the Ridge
 * head, and folds the stack into input-space coefficients (p*1) + intercept (1).
 * SINGLE-TARGET only (q must be 1). `alphas` may be NULL for a default grid. */
__attribute__((used))
int n4m_wasm_aom_stack_fit(const double* x, const double* y,
                           int n, int p, int q,
                           int profile, int cv, int max_components,
                           const double* alphas, int n_alphas,
                           double std_penalty, double gap_penalty,
                           double* coefficients_out, double* intercept_out) {
    if (x == NULL || y == NULL || coefficients_out == NULL || intercept_out == NULL)
        return N4M_ERR_NULL_POINTER;
    if (n < 2 || p < 1 || q != 1) return N4M_ERR_INVALID_ARGUMENT; /* single-target */
    int k = cv; if (k < 2) k = 2; if (k > n) k = n;
    int mc = max_components;
    if (mc < 1) mc = 1;
    if (mc > p) mc = p;
    if (mc >= n) mc = n - 1;
    if (mc < 1) mc = 1;

    n4m_matrix_view_t xv, yv;
    n4m_matrix_view_init_rowmajor(&xv, (void*)x, n, p, N4M_DTYPE_F64);
    n4m_matrix_view_init_rowmajor(&yv, (void*)y, n, q, N4M_DTYPE_F64);

    n4m_context_t* ctx = NULL;
    n4m_status_t s = n4m_context_create(&ctx);
    if (s != N4M_OK) return s;
    n4m_config_t* cfg = NULL;
    s = n4m_config_create(&cfg);
    if (s != N4M_OK) { n4m_context_destroy(ctx); return s; }

    int32_t* fold_ids = n4m_wasm_make_fold_ids(n, k);
    int32_t* comps = (int32_t*)malloc((size_t)mc * sizeof(int32_t));
    if (fold_ids == NULL || comps == NULL) {
        free(fold_ids); free(comps);
        n4m_config_destroy(cfg); n4m_context_destroy(ctx);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < mc; ++i) comps[i] = (int32_t)(i + 1);
    static const double kDefaultAlphas[] = {0.01, 0.1, 1.0, 10.0, 100.0};
    const double* al = (alphas != NULL && n_alphas > 0) ? alphas : kDefaultAlphas;
    int64_t na = (alphas != NULL && n_alphas > 0)
                     ? (int64_t)n_alphas
                     : (int64_t)(sizeof(kDefaultAlphas) / sizeof(kDefaultAlphas[0]));

    n4m_method_result_t* res = NULL;
    s = n4m_ensemble_aom_operator_pls_stack_fit(ctx, cfg, &xv, &yv,
                                                (int32_t)profile, (int32_t)k,
                                                fold_ids, (int64_t)n,
                                                comps, (int64_t)mc, al, na,
                                                std_penalty, gap_penalty, &res);
    free(fold_ids);
    free(comps);
    n4m_config_destroy(cfg);
    if (s != N4M_OK || res == NULL) {
        if (res != NULL) n4m_method_result_destroy(res);
        n4m_context_destroy(ctx);
        return s != N4M_OK ? s : N4M_ERR_INVALID_ARGUMENT;
    }
    int ok_c = copy_result_matrix(res, "input_coefficients",
                                  coefficients_out, (size_t)p);
    int ok_i = copy_result_matrix(res, "input_intercept", intercept_out, 1);
    n4m_method_result_destroy(res);
    n4m_context_destroy(ctx);
    return (ok_c && ok_i) ? N4M_OK : N4M_ERR_INVALID_ARGUMENT;
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
    /* Note: n4m_transform_normalize is COLUMN-wise L2 (batch-dependent), not the
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
            s = n4m_transform_snv_create(&h, 1, 1, 0);
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
            n4m_pp_savgol_mode_t mode = n_params >= 4
                ? (n4m_pp_savgol_mode_t)((int)params[3])
                : N4M_PP_SAVGOL_MIRROR;
            double cval = n_params >= 5 ? params[4] : 0.0;
            n4m_pp_savgol_handle_t* h = NULL;
            s = n4m_transform_savitzky_golay_create(&h, window, poly, deriv, 1.0,
                                     mode, cval);
            w->h = h;
            break;
        }
        case PPK_DERIV1: {
            n4m_pp_first_derivative_handle_t* h = NULL;
            s = n4m_transform_first_derivative_create(&h, 1.0, 2);
            w->h = h;
            break;
        }
        case PPK_DERIV2: {
            n4m_pp_second_derivative_handle_t* h = NULL;
            s = n4m_transform_second_derivative_create(&h, 1.0, 2);
            w->h = h;
            break;
        }
        case PPK_DETREND: {
            int poly = n_params >= 1 ? (int)params[0] : 1;
            n4m_pp_detrend_handle_t* h = NULL;
            s = n4m_transform_detrend_create(&h, poly);
            w->h = h;
            break;
        }
        case PPK_GAUSSIAN: {
            double sigma = n_params >= 1 ? params[0] : 2.0;
            n4m_pp_gaussian_handle_t* h = NULL;
            s = n4m_transform_gaussian_create(&h, sigma, 0,
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
            s = n4m_transform_asls_create(&h, lam, pp, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_AIRPLS: {
            double lam = n_params >= 1 ? params[0] : 1e6;
            int max_it = n_params >= 2 ? (int)params[1] : 50;
            double tol = n_params >= 3 ? params[2] : 1e-3;
            n4m_pp_airpls_handle_t* h = NULL;
            s = n4m_transform_airpls_create(&h, lam, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_ARPLS: {
            double lam = n_params >= 1 ? params[0] : 1e5;
            int max_it = n_params >= 2 ? (int)params[1] : 50;
            double tol = n_params >= 3 ? params[2] : 1e-3;
            n4m_pp_arpls_handle_t* h = NULL;
            s = n4m_transform_arpls_create(&h, lam, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_MODPOLY: {
            int poly = n_params >= 1 ? (int)params[0] : 2;
            int max_it = n_params >= 2 ? (int)params[1] : 250;
            double tol = n_params >= 3 ? params[2] : 1e-3;
            n4m_pp_modpoly_handle_t* h = NULL;
            s = n4m_transform_modpoly_create(&h, poly, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_IMODPOLY: {
            int poly = n_params >= 1 ? (int)params[0] : 2;
            int max_it = n_params >= 2 ? (int)params[1] : 250;
            double tol = n_params >= 3 ? params[2] : 1e-3;
            n4m_pp_imodpoly_handle_t* h = NULL;
            s = n4m_transform_imodpoly_create(&h, poly, max_it, tol);
            w->h = h;
            break;
        }
        case PPK_SNIP: {
            int mhw = n_params >= 1 ? (int)params[0] : 20;
            n4m_pp_snip_handle_t* h = NULL;
            s = n4m_transform_snip_create(&h, mhw);
            w->h = h;
            break;
        }
        case PPK_ROLLING_BALL: {
            int hw = n_params >= 1 ? (int)params[0] : 20;
            int shw = n_params >= 2 ? (int)params[1] : 0;
            n4m_pp_rolling_ball_handle_t* h = NULL;
            s = n4m_transform_rolling_ball_create(&h, hw, shw);
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
            s = n4m_transform_iasls_create(&h, lam, pp, poly, max_it, tol);
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
            s = n4m_transform_beads_create(&h, lam0, lam1, lam2, max_it, tol);
            w->h = h;
            break;
        }
        /* ---- A2b signal conversions ---- */
        case PPK_TO_ABS: {
            int is_percent = n_params >= 1 ? (int)params[0] : 0;
            double eps = n_params >= 2 ? params[1] : 1e-8;
            int clip_neg = n_params >= 3 ? (int)params[2] : 1;
            n4m_pp_to_absorbance_handle_t* h = NULL;
            s = n4m_transform_to_absorbance_create(&h, is_percent, eps, clip_neg);
            w->h = h;
            break;
        }
        case PPK_FROM_ABS: {
            int is_percent = n_params >= 1 ? (int)params[0] : 0;
            n4m_pp_from_absorbance_handle_t* h = NULL;
            s = n4m_transform_from_absorbance_create(&h, is_percent);
            w->h = h;
            break;
        }
        case PPK_PCT_TO_FRAC: {
            n4m_pp_pct_to_frac_handle_t* h = NULL;
            s = n4m_transform_percent_to_fraction_create(&h);
            w->h = h;
            break;
        }
        case PPK_FRAC_TO_PCT: {
            n4m_pp_frac_to_pct_handle_t* h = NULL;
            s = n4m_transform_fraction_to_percent_create(&h);
            w->h = h;
            break;
        }
        case PPK_KUBELKA_MUNK: {
            int is_percent = n_params >= 1 ? (int)params[0] : 0;
            double eps = n_params >= 2 ? params[1] : 1e-8;
            n4m_pp_kubelka_munk_handle_t* h = NULL;
            s = n4m_transform_kubelka_munk_create(&h, is_percent, eps);
            w->h = h;
            break;
        }
        /* ---- A2c scatter / scaling / derivative ---- */
        case PPK_RNV: {
            int with_center = n_params >= 1 ? (int)params[0] : 1;
            int with_scale = n_params >= 2 ? (int)params[1] : 1;
            double k = n_params >= 3 ? params[2] : 1.4826;
            n4m_pp_rnv_handle_t* h = NULL;
            s = n4m_transform_robust_snv_create(&h, with_center, with_scale, k);
            w->h = h;
            break;
        }
        case PPK_LSNV: {
            int window = n_params >= 1 ? (int)params[0] : 11;
            int pad_mode = n_params >= 2 ? (int)params[1] : 0;
            double constant = n_params >= 3 ? params[2] : 0.0;
            n4m_pp_lsnv_handle_t* h = NULL;
            s = n4m_transform_local_snv_create(&h, window, pad_mode, constant);
            w->h = h;
            break;
        }
        case PPK_AREA: {
            int method = n_params >= 1 ? (int)params[0] : 1;
            n4m_pp_area_handle_t* h = NULL;
            s = n4m_transform_area_normalization_create(&h, method);
            w->h = h;
            break;
        }
        case PPK_NORRIS_WILLIAMS: {
            int segment = n_params >= 1 ? (int)params[0] : 5;
            int gap = n_params >= 2 ? (int)params[1] : 3;
            int order = n_params >= 3 ? (int)params[2] : 1;
            double delta = n_params >= 4 ? params[3] : 1.0;
            n4m_pp_norris_williams_handle_t* h = NULL;
            s = n4m_transform_norris_williams_create(&h, segment, gap, order, delta);
            w->h = h;
            break;
        }
        case PPK_LOG: {
            /* Stateless flavour only (auto_offset = 0); base 0 selects natural log. */
            double base = n_params >= 1 ? params[0] : 0.0;
            double offset = n_params >= 2 ? params[1] : 0.0;
            double min_value = n_params >= 3 ? params[2] : 1e-8;
            n4m_pp_log_handle_t* h = NULL;
            s = n4m_transform_log_transform_create(&h, base, offset, 0, min_value);
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
            s = n4m_transform_wavelet_denoise_create(
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
            return n4m_transform_snv_transform((const n4m_pp_snv_handle_t*)w->h, xv, ov);
        case PPK_SAVGOL:
            return n4m_transform_savitzky_golay_transform((const n4m_pp_savgol_handle_t*)w->h, xv, ov);
        case PPK_DERIV1:
            return n4m_transform_first_derivative_transform((const n4m_pp_first_derivative_handle_t*)w->h, xv, ov);
        case PPK_DERIV2:
            return n4m_transform_second_derivative_transform((const n4m_pp_second_derivative_handle_t*)w->h, xv, ov);
        case PPK_DETREND:
            return n4m_transform_detrend_transform((const n4m_pp_detrend_handle_t*)w->h, xv, ov);
        case PPK_GAUSSIAN:
            return n4m_transform_gaussian_transform((const n4m_pp_gaussian_handle_t*)w->h, xv, ov);
        /* ---- A2a baseline correctors ---- */
        case PPK_ASLS:
            return n4m_transform_asls_transform((const n4m_pp_asls_handle_t*)w->h, xv, ov);
        case PPK_AIRPLS:
            return n4m_transform_airpls_transform((const n4m_pp_airpls_handle_t*)w->h, xv, ov);
        case PPK_ARPLS:
            return n4m_transform_arpls_transform((const n4m_pp_arpls_handle_t*)w->h, xv, ov);
        case PPK_MODPOLY:
            return n4m_transform_modpoly_transform((const n4m_pp_modpoly_handle_t*)w->h, xv, ov);
        case PPK_IMODPOLY:
            return n4m_transform_imodpoly_transform((const n4m_pp_imodpoly_handle_t*)w->h, xv, ov);
        case PPK_SNIP:
            return n4m_transform_snip_transform((const n4m_pp_snip_handle_t*)w->h, xv, ov);
        case PPK_ROLLING_BALL:
            return n4m_transform_rolling_ball_transform((const n4m_pp_rolling_ball_handle_t*)w->h, xv, ov);
        case PPK_IASLS:
            return n4m_transform_iasls_transform((const n4m_pp_iasls_handle_t*)w->h, xv, ov);
        case PPK_BEADS:
            return n4m_transform_beads_transform((const n4m_pp_beads_handle_t*)w->h, xv, ov);
        /* ---- A2b signal conversions ---- */
        case PPK_TO_ABS:
            return n4m_transform_to_absorbance_transform((const n4m_pp_to_absorbance_handle_t*)w->h, xv, ov);
        case PPK_FROM_ABS:
            return n4m_transform_from_absorbance_transform((const n4m_pp_from_absorbance_handle_t*)w->h, xv, ov);
        case PPK_PCT_TO_FRAC:
            return n4m_transform_percent_to_fraction_transform((const n4m_pp_pct_to_frac_handle_t*)w->h, xv, ov);
        case PPK_FRAC_TO_PCT:
            return n4m_transform_fraction_to_percent_transform((const n4m_pp_frac_to_pct_handle_t*)w->h, xv, ov);
        case PPK_KUBELKA_MUNK:
            return n4m_transform_kubelka_munk_transform((const n4m_pp_kubelka_munk_handle_t*)w->h, xv, ov);
        /* ---- A2c scatter / scaling / derivative ---- */
        case PPK_RNV:
            return n4m_transform_robust_snv_transform((const n4m_pp_rnv_handle_t*)w->h, xv, ov);
        case PPK_LSNV:
            return n4m_transform_local_snv_transform((const n4m_pp_lsnv_handle_t*)w->h, xv, ov);
        case PPK_AREA:
            return n4m_transform_area_normalization_transform((const n4m_pp_area_handle_t*)w->h, xv, ov);
        case PPK_NORRIS_WILLIAMS:
            return n4m_transform_norris_williams_transform((const n4m_pp_norris_williams_handle_t*)w->h, xv, ov);
        case PPK_LOG:
            return n4m_transform_log_transform_transform((const n4m_pp_log_handle_t*)w->h, xv, ov);
        case PPK_WAVELET_DENOISE:
            return n4m_transform_wavelet_denoise_transform((const n4m_pp_wavelet_denoise_handle_t*)w->h, xv, ov);
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
        case PPK_SNV:      n4m_transform_snv_destroy((n4m_pp_snv_handle_t*)w->h); break;
        case PPK_MSC:      n4m_pp_msc_state_free((n4m_pp_msc_state_t*)w->h); break;
        case PPK_SAVGOL:   n4m_transform_savitzky_golay_destroy((n4m_pp_savgol_handle_t*)w->h); break;
        case PPK_DERIV1:   n4m_transform_first_derivative_destroy((n4m_pp_first_derivative_handle_t*)w->h); break;
        case PPK_DERIV2:   n4m_transform_second_derivative_destroy((n4m_pp_second_derivative_handle_t*)w->h); break;
        case PPK_DETREND:  n4m_transform_detrend_destroy((n4m_pp_detrend_handle_t*)w->h); break;
        case PPK_GAUSSIAN: n4m_transform_gaussian_destroy((n4m_pp_gaussian_handle_t*)w->h); break;
        /* ---- A2a baseline correctors ---- */
        case PPK_ASLS:         n4m_transform_asls_destroy((n4m_pp_asls_handle_t*)w->h); break;
        case PPK_AIRPLS:       n4m_transform_airpls_destroy((n4m_pp_airpls_handle_t*)w->h); break;
        case PPK_ARPLS:        n4m_transform_arpls_destroy((n4m_pp_arpls_handle_t*)w->h); break;
        case PPK_MODPOLY:      n4m_transform_modpoly_destroy((n4m_pp_modpoly_handle_t*)w->h); break;
        case PPK_IMODPOLY:     n4m_transform_imodpoly_destroy((n4m_pp_imodpoly_handle_t*)w->h); break;
        case PPK_SNIP:         n4m_transform_snip_destroy((n4m_pp_snip_handle_t*)w->h); break;
        case PPK_ROLLING_BALL: n4m_transform_rolling_ball_destroy((n4m_pp_rolling_ball_handle_t*)w->h); break;
        case PPK_IASLS:        n4m_transform_iasls_destroy((n4m_pp_iasls_handle_t*)w->h); break;
        case PPK_BEADS:        n4m_transform_beads_destroy((n4m_pp_beads_handle_t*)w->h); break;
        /* ---- A2b signal conversions ---- */
        case PPK_TO_ABS:       n4m_transform_to_absorbance_destroy((n4m_pp_to_absorbance_handle_t*)w->h); break;
        case PPK_FROM_ABS:     n4m_transform_from_absorbance_destroy((n4m_pp_from_absorbance_handle_t*)w->h); break;
        case PPK_PCT_TO_FRAC:  n4m_transform_percent_to_fraction_destroy((n4m_pp_pct_to_frac_handle_t*)w->h); break;
        case PPK_FRAC_TO_PCT:  n4m_transform_fraction_to_percent_destroy((n4m_pp_frac_to_pct_handle_t*)w->h); break;
        case PPK_KUBELKA_MUNK: n4m_transform_kubelka_munk_destroy((n4m_pp_kubelka_munk_handle_t*)w->h); break;
        /* ---- A2c scatter / scaling / derivative ---- */
        case PPK_RNV:             n4m_transform_robust_snv_destroy((n4m_pp_rnv_handle_t*)w->h); break;
        case PPK_LSNV:            n4m_transform_local_snv_destroy((n4m_pp_lsnv_handle_t*)w->h); break;
        case PPK_AREA:            n4m_transform_area_normalization_destroy((n4m_pp_area_handle_t*)w->h); break;
        case PPK_NORRIS_WILLIAMS: n4m_transform_norris_williams_destroy((n4m_pp_norris_williams_handle_t*)w->h); break;
        case PPK_LOG:             n4m_transform_log_transform_destroy((n4m_pp_log_handle_t*)w->h); break;
        case PPK_WAVELET_DENOISE: n4m_transform_wavelet_denoise_destroy((n4m_pp_wavelet_denoise_handle_t*)w->h); break;
        default: break;
    }
    free(w);
}

/* ============================================================================
 * Splitters (raw-double-pointer surface for JS)
 * ----------------------------------------------------------------------------
 * Compute a single train/test split over the n rows and fill out_test_mask[n]
 * with 1 (test) / 0 (train). The numerics stay 100% in libn4m (the n4m_split_*
 * splitters); this is only marshalling + mask conversion.
 *
 *   kind 0 = KennardStone   (X only)        — n4m_split_kennard_stone_*
 *   kind 1 = SPXY           (X and Y)       — n4m_split_spxy_*
 *   kind 2 = KMeans         (X; seed,p0=max_iter) — n4m_split_kmeans_*
 *   kind 3 = KBinsStratified(Y; seed,p0=n_bins,p1=strategy) — n4m_split_kbins_*
 *
 * test_size is a fraction in (0, 1). `seed` is used by the stochastic splitters
 * (KMeans, KBins); KS/SPXY are deterministic and ignore it. p0/p1 are generic
 * extra integer params (see per-kind notes). y may be NULL for kinds that do not
 * use it (KS, KMeans); SPXY and KBins require it. Returns 0 on success, a
 * N4M_ERR_* otherwise; on failure out_test_mask is left untouched.
 * ==========================================================================*/
static n4m_status_t n4m_wasm_run_split_result(int kind, double test_size, unsigned int seed,
                                              int p0, int p1,
                                              const double* x, const double* y,
                                              int n, int p, int q,
                                              n4m_split_result_t* out_res) {
    if (out_res == NULL) return N4M_ERR_NULL_POINTER;
    /* KBins (3) and SystematicCircular (5) split on Y only — they need no X. */
    if (x == NULL && kind != 3 && kind != 5) return N4M_ERR_NULL_POINTER;
    if ((kind == 1 || kind == 3 || kind == 5) && y == NULL) return N4M_ERR_NULL_POINTER;
    if (n < 2 || (x != NULL && p < 1)) return N4M_ERR_INVALID_ARGUMENT;
    if (!(test_size > 0.0 && test_size < 1.0)) return N4M_ERR_INVALID_ARGUMENT;

    n4m_matrix_view_t xv, yv;
    if (x != NULL) n4m_matrix_view_init_rowmajor(&xv, (void*)x, n, p, N4M_DTYPE_F64);
    if (y != NULL) n4m_matrix_view_init_rowmajor(&yv, (void*)y, n, q < 1 ? 1 : q, N4M_DTYPE_F64);

    *out_res = (n4m_split_result_t){0};
    n4m_status_t s = N4M_ERR_NOT_IMPLEMENTED;
    switch (kind) {
        case 0: { /* KennardStone */
            n4m_split_kennard_stone_handle_t* h = NULL;
            s = n4m_model_selection_kennard_stone_create(&h, test_size);
            if (s == N4M_OK) s = n4m_model_selection_kennard_stone_split(h, xv, out_res);
            n4m_model_selection_kennard_stone_destroy(h);
            break;
        }
        case 1: { /* SPXY (joint X-Y) */
            n4m_split_spxy_handle_t* h = NULL;
            s = n4m_model_selection_spxy_create(&h, test_size);
            if (s == N4M_OK) s = n4m_model_selection_spxy_split(h, xv, yv, out_res);
            n4m_model_selection_spxy_destroy(h);
            break;
        }
        case 2: { /* KMeans (k-means++) — p0 = max_iter */
            int max_iter = p0 > 0 ? p0 : 100;
            n4m_split_kmeans_handle_t* h = NULL;
            s = n4m_model_selection_kmeans_create(&h, test_size, (uint64_t)seed, max_iter);
            if (s == N4M_OK) s = n4m_model_selection_kmeans_split(h, xv, out_res);
            n4m_model_selection_kmeans_destroy(h);
            break;
        }
        case 3: { /* KBinsStratified (Y) — p0 = n_bins, p1 = strategy */
            int n_bins = p0 > 1 ? p0 : 5;
            int strategy = p1; /* 0 uniform, 1 quantile */
            n4m_split_kbins_stratified_handle_t* h = NULL;
            s = n4m_model_selection_kbins_stratified_create(&h, test_size, (uint64_t)seed,
                                                  n_bins, strategy);
            if (s == N4M_OK) s = n4m_model_selection_kbins_stratified_split(h, yv, out_res);
            n4m_model_selection_kbins_stratified_destroy(h);
            break;
        }
        case 4: { /* SPlit / data twinning (X-based, deterministic) */
            n4m_split_split_splitter_handle_t* h = NULL;
            s = n4m_model_selection_data_twinning_create(&h, test_size, (uint64_t)seed);
            if (s == N4M_OK) s = n4m_model_selection_data_twinning_split(h, xv, out_res);
            n4m_model_selection_data_twinning_destroy(h);
            break;
        }
        case 5: { /* SystematicCircular (Y-based) */
            n4m_split_systematic_circular_handle_t* h = NULL;
            s = n4m_model_selection_systematic_circular_create(&h, test_size, (uint64_t)seed);
            if (s == N4M_OK) s = n4m_model_selection_systematic_circular_split(h, yv, out_res);
            n4m_model_selection_systematic_circular_destroy(h);
            break;
        }
        default:
            return N4M_ERR_INVALID_ARGUMENT;
    }
    return s;
}

__attribute__((used))
int n4m_wasm_split(int kind, double test_size, unsigned int seed,
                   int p0, int p1,
                   const double* x, const double* y,
                   int n, int p, int q,
                   int* out_test_mask) {
    if (out_test_mask == NULL) return N4M_ERR_NULL_POINTER;

    n4m_split_result_t res = {0};
    n4m_status_t s = n4m_wasm_run_split_result(kind, test_size, seed, p0, p1,
                                               x, y, n, p, q, &res);
    if (s != N4M_OK) { n4m_split_result_destroy(&res); return s; }

    /* Fill the mask: default train (0), mark test rows (1). */
    for (int i = 0; i < n; ++i) out_test_mask[i] = 0;
    if (res.test_idx != NULL) {
        for (int64_t t = 0; t < res.n_test; ++t) {
            int64_t row = res.test_idx[t];
            if (row >= 0 && row < (int64_t)n) out_test_mask[(int)row] = 1;
        }
    }
    n4m_split_result_destroy(&res);
    return N4M_OK;
}

__attribute__((used))
int n4m_wasm_split_indices(int kind, double test_size, unsigned int seed,
                           int p0, int p1,
                           const double* x, const double* y,
                           int n, int p, int q,
                           int* out_train_idx, int* out_n_train,
                           int* out_test_idx, int* out_n_test) {
    if (out_train_idx == NULL || out_n_train == NULL ||
        out_test_idx == NULL || out_n_test == NULL) {
        return N4M_ERR_NULL_POINTER;
    }

    n4m_split_result_t res = {0};
    n4m_status_t s = n4m_wasm_run_split_result(kind, test_size, seed, p0, p1,
                                               x, y, n, p, q, &res);
    if (s != N4M_OK) { n4m_split_result_destroy(&res); return s; }

    *out_n_train = (int)res.n_train;
    *out_n_test = (int)res.n_test;
    for (int64_t i = 0; i < res.n_train; ++i) out_train_idx[i] = (int)res.train_idx[i];
    for (int64_t i = 0; i < res.n_test; ++i) out_test_idx[i] = (int)res.test_idx[i];

    n4m_split_result_destroy(&res);
    return N4M_OK;
}
