/* SPDX-License-Identifier: CeCILL-2.1
 *
 * Emscripten entry point + production helpers.
 *
 * Pulls in the full public ABI header so all p4a_* symbols stay
 * reachable. Adds a small set of higher-level `p4a_wasm_*` helpers
 * that take raw double pointers + ints, work around an Emscripten
 * 5.0.7 code-gen issue where passing `p4a_matrix_view_t*` parameters
 * directly to `p4a_model_fit` from a JS-invoked WASM function returns
 * incorrect numerics. The same `p4a_model_fit` invoked through the
 * raw-pointer helpers below is byte-correct (matches the native
 * Linux build and the Python binding).
 */

#include "pls4all/p4a.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Touch every accessor on the matrix-view struct so Emscripten preserves
 * the in-line `p4a_matrix_view_init_*` helpers from the header. */
__attribute__((used)) static int p4a_wasm_keep_alive(void) {
    p4a_matrix_view_t v = {0};
    p4a_matrix_view_init_rowmajor(&v, NULL, 0, 0, P4A_DTYPE_F64);
    p4a_matrix_view_init_colmajor(&v, NULL, 0, 0, P4A_DTYPE_F64);
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
 * Returns 0 on success, a P4A_ERR_* on failure. center_x / center_y
 * are always 1; solver is SIMPLS.
 *
 * Rationale: see the file-level comment — the matrix-view-arg path
 * has an Emscripten codegen issue specific to this build; this helper
 * funnels JS callers through the working raw-data-pointer path.
 */
__attribute__((used))
int p4a_wasm_pls_fit_legacy(const double *x, const double *y,
                      int n, int p, int q, int n_components,
                      double *coefficients_out,
                      double *x_mean_out,
                      double *y_mean_out,
                      double *predictions_out) {
    if (x == NULL || y == NULL || coefficients_out == NULL ||
        x_mean_out == NULL || y_mean_out == NULL) {
        return P4A_ERR_NULL_POINTER;
    }
    if (n < 2 || p < 1 || q < 1 || n_components < 1) {
        return P4A_ERR_INVALID_ARGUMENT;
    }

    p4a_matrix_view_t xv, yv;
    p4a_matrix_view_init_rowmajor(&xv, (void *)x, n, p, P4A_DTYPE_F64);
    p4a_matrix_view_init_rowmajor(&yv, (void *)y, n, q, P4A_DTYPE_F64);

    p4a_context_t *ctx = NULL;
    p4a_status_t s = p4a_context_create(&ctx);
    if (s != P4A_OK) return s;
    p4a_config_t *cfg = NULL;
    s = p4a_config_create(&cfg);
    if (s != P4A_OK) { p4a_context_destroy(ctx); return s; }

    p4a_config_set_algorithm(cfg, P4A_ALGO_PLS_REGRESSION);
    p4a_config_set_solver(cfg, P4A_SOLVER_SIMPLS);
    p4a_config_set_deflation(cfg, P4A_DEFLATION_REGRESSION);
    p4a_config_set_n_components(cfg, n_components);
    p4a_config_set_center_x(cfg, 1);
    p4a_config_set_center_y(cfg, 1);

    p4a_model_t *m = NULL;
    s = p4a_model_fit(ctx, cfg, &xv, &yv, &m);
    p4a_config_destroy(cfg);
    if (s != P4A_OK) { p4a_context_destroy(ctx); return s; }

    /* Copy out coefficients / x_mean / y_mean. */
    p4a_array_t *arr = NULL;
    if (p4a_model_get_array(ctx, m, P4A_MODEL_COEFFICIENTS, &arr) == P4A_OK) {
        p4a_matrix_view_t v;
        p4a_array_view(arr, &v);
        memcpy(coefficients_out, v.data,
               (size_t)p * (size_t)q * sizeof(double));
        p4a_array_free(arr);
    }
    if (p4a_model_get_array(ctx, m, P4A_MODEL_X_MEAN, &arr) == P4A_OK) {
        p4a_matrix_view_t v;
        p4a_array_view(arr, &v);
        memcpy(x_mean_out, v.data, (size_t)p * sizeof(double));
        p4a_array_free(arr);
    }
    if (p4a_model_get_array(ctx, m, P4A_MODEL_Y_MEAN, &arr) == P4A_OK) {
        p4a_matrix_view_t v;
        p4a_array_view(arr, &v);
        memcpy(y_mean_out, v.data, (size_t)q * sizeof(double));
        p4a_array_free(arr);
    }

    if (predictions_out != NULL) {
        p4a_matrix_view_t pv;
        p4a_matrix_view_init_rowmajor(&pv, predictions_out, n, q,
                                        P4A_DTYPE_F64);
        p4a_model_predict(ctx, m, &xv, &pv);
    }

    p4a_model_destroy(m);
    p4a_context_destroy(ctx);
    return P4A_OK;
}

/* Predict from coefficients + means. Useful when the model was fit
 * elsewhere and only its coefficient triple was transported. */
__attribute__((used))
int p4a_wasm_pls_predict_from_coeffs(const double *x_new,
                                       int n_new, int p, int q,
                                       const double *coefficients,
                                       const double *x_mean,
                                       const double *y_mean,
                                       double *predictions_out) {
    if (x_new == NULL || coefficients == NULL || x_mean == NULL ||
        y_mean == NULL || predictions_out == NULL) {
        return P4A_ERR_NULL_POINTER;
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
    return P4A_OK;
}
