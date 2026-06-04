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
    PPK_DETREND, PPK_GAUSSIAN
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
        default: break;
    }
    free(w);
}
