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
