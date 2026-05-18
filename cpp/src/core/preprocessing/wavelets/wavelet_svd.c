/* SPDX-License-Identifier: CECILL-2.1 */

#include "wavelet_svd.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/svd.h"

struct c4a_pp_wavelet_svd_state_t {
    c4a_wavelet_family_t family;
    c4a_wavelet_mode_t   mode;
    int32_t              max_level;
    double               n_components_param;

    int      fitted;
    int32_t  actual_level;
    int64_t  flat_dim;
    int64_t  n_components;
    int64_t  n_features_in;
    double*  components;   /* k' x flat_dim */
};

c4a_pp_wavelet_svd_state_t* c4a_pp_wavelet_svd_state_new(
    c4a_wavelet_family_t family,
    c4a_wavelet_mode_t   mode,
    int32_t              max_level,
    double               n_components_param) {
    if (max_level < 0) return NULL;
    if (!(n_components_param > 0.0)) return NULL;
    c4a_pp_wavelet_svd_state_t* s =
        (c4a_pp_wavelet_svd_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->family             = family;
    s->mode               = mode;
    s->max_level          = max_level;
    s->n_components_param = n_components_param;
    s->fitted             = 0;
    s->actual_level       = 0;
    s->flat_dim           = 0;
    s->n_components       = 0;
    s->n_features_in      = 0;
    s->components         = NULL;
    return s;
}

void c4a_pp_wavelet_svd_state_free(c4a_pp_wavelet_svd_state_t* state) {
    if (state == NULL) return;
    free(state->components);
    free(state);
}

int c4a_pp_wavelet_svd_state_is_fitted(
    const c4a_pp_wavelet_svd_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

int64_t c4a_pp_wavelet_svd_state_n_components(
    const c4a_pp_wavelet_svd_state_t* state) {
    return (state != NULL && state->fitted) ? state->n_components : 0;
}

/* Truncated-SVD k-selection: count >= 1 OR variance ratio in (0, 1). */
static int64_t svd_select_k(double param, int64_t k_full,
                             const double* var_ratio) {
    if (param >= 1.0) {
        int64_t k = (int64_t)param;
        if (k < 1) k = 1;
        if (k > k_full) k = k_full;
        return k;
    }
    double cum = 0.0;
    for (int64_t i = 0; i < k_full; ++i) {
        cum += var_ratio[i];
        if (cum >= param) return i + 1;
    }
    return k_full;
}

c4a_status_t c4a_pp_wavelet_svd_state_fit(
    c4a_pp_wavelet_svd_state_t* state,
    const double* X, int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const int32_t max_lvl = c4a_wavelet_dwt_max_level(cols, state->family);
    const int32_t level   = (state->max_level < max_lvl) ? state->max_level
                                                          : max_lvl;
    int64_t flat_dim = 0;
    int64_t* coef_lengths = (int64_t*)malloc((size_t)(level + 1) * sizeof(int64_t));
    int64_t* offsets      = (int64_t*)malloc((size_t)(level + 1) * sizeof(int64_t));
    if (coef_lengths == NULL || offsets == NULL) {
        free(coef_lengths); free(offsets);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    c4a_status_t st = c4a_wavelet_wavedec_lengths(
        cols, state->family, state->mode, level, &flat_dim, coef_lengths);
    if (st != C4A_OK) {
        free(coef_lengths); free(offsets);
        return st;
    }
    offsets[0] = 0;
    {
        int64_t acc = coef_lengths[0];
        for (int32_t k = 1; k <= level; ++k) {
            offsets[k] = acc;
            acc += coef_lengths[k];
        }
    }
    if (flat_dim < 1) {
        free(coef_lengths); free(offsets);
        return C4A_ERR_INVALID_ARGUMENT;
    }

    double* F = (double*)malloc((size_t)rows * (size_t)flat_dim * sizeof(double));
    if (F == NULL) {
        free(coef_lengths); free(offsets);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        st = c4a_wavelet_wavedec(row, cols, state->family, state->mode,
                                  level, F + (size_t)i * (size_t)flat_dim,
                                  coef_lengths, offsets, flat_dim);
        if (st != C4A_OK) {
            free(F); free(coef_lengths); free(offsets);
            return st;
        }
    }
    free(coef_lengths); free(offsets);

    const int64_t m      = rows;
    const int64_t n      = flat_dim;
    const int64_t k_full = (m < n) ? m : n;
    double* U  = (double*)malloc((size_t)m * (size_t)k_full * sizeof(double));
    double* S  = (double*)malloc((size_t)k_full * sizeof(double));
    double* Vt = (double*)malloc((size_t)k_full * (size_t)n * sizeof(double));
    if (U == NULL || S == NULL || Vt == NULL) {
        free(U); free(S); free(Vt); free(F);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    st = c4a_svd_compact(F, m, n, U, S, Vt);
    free(F);
    if (st != C4A_OK) {
        free(U); free(S); free(Vt);
        return st;
    }
    double* evar      = (double*)malloc((size_t)k_full * sizeof(double));
    double* var_ratio = (double*)malloc((size_t)k_full * sizeof(double));
    if (evar == NULL || var_ratio == NULL) {
        free(evar); free(var_ratio);
        free(U); free(S); free(Vt);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    /* sklearn TruncatedSVD ranks by squared singular values directly
     * (no centering, so the "total variance" equals sum(S^2)). */
    double total = 0.0;
    for (int64_t i = 0; i < k_full; ++i) {
        evar[i] = S[i] * S[i];
        total  += evar[i];
    }
    if (total > 0.0) {
        const double inv = 1.0 / total;
        for (int64_t i = 0; i < k_full; ++i) var_ratio[i] = evar[i] * inv;
    } else {
        for (int64_t i = 0; i < k_full; ++i) var_ratio[i] = 0.0;
    }
    const int64_t k_kept =
        svd_select_k(state->n_components_param, k_full, var_ratio);
    double* components =
        (double*)malloc((size_t)k_kept * (size_t)n * sizeof(double));
    if (components == NULL) {
        free(evar); free(var_ratio);
        free(U); free(S); free(Vt);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    memcpy(components, Vt, (size_t)k_kept * (size_t)n * sizeof(double));
    free(evar); free(var_ratio);
    free(U); free(S); free(Vt);

    free(state->components);
    state->actual_level   = level;
    state->flat_dim       = n;
    state->n_components   = k_kept;
    state->n_features_in  = cols;
    state->components     = components;
    state->fitted         = 1;
    return C4A_OK;
}

c4a_status_t c4a_pp_wavelet_svd_state_apply(
    const c4a_pp_wavelet_svd_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (!state->fitted) return C4A_ERR_NOT_FITTED;
    if (rows < 0 || cols < 1 || out_cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->n_features_in) return C4A_ERR_SHAPE_MISMATCH;
    if (out_cols != state->n_components) return C4A_ERR_SHAPE_MISMATCH;
    if (rows == 0) return C4A_OK;

    const int32_t level = state->actual_level;
    int64_t* coef_lengths = (int64_t*)malloc((size_t)(level + 1) * sizeof(int64_t));
    int64_t* offsets      = (int64_t*)malloc((size_t)(level + 1) * sizeof(int64_t));
    if (coef_lengths == NULL || offsets == NULL) {
        free(coef_lengths); free(offsets);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    int64_t total_flat = 0;
    c4a_status_t st = c4a_wavelet_wavedec_lengths(
        cols, state->family, state->mode, level, &total_flat, coef_lengths);
    if (st != C4A_OK || total_flat != state->flat_dim) {
        free(coef_lengths); free(offsets);
        return (st != C4A_OK) ? st : C4A_ERR_SHAPE_MISMATCH;
    }
    offsets[0] = 0;
    {
        int64_t acc = coef_lengths[0];
        for (int32_t k = 1; k <= level; ++k) {
            offsets[k] = acc;
            acc += coef_lengths[k];
        }
    }
    double* flat = (double*)malloc((size_t)state->flat_dim * sizeof(double));
    if (flat == NULL) {
        free(coef_lengths); free(offsets);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        st = c4a_wavelet_wavedec(row, cols, state->family, state->mode,
                                  level, flat, coef_lengths, offsets,
                                  state->flat_dim);
        if (st != C4A_OK) {
            free(flat); free(coef_lengths); free(offsets);
            return st;
        }
        double* orow = out + (size_t)i * (size_t)out_cols;
        for (int64_t k = 0; k < state->n_components; ++k) {
            const double* comp_k =
                state->components + (size_t)k * (size_t)state->flat_dim;
            double acc = 0.0;
            for (int64_t j = 0; j < state->flat_dim; ++j) {
                acc += flat[j] * comp_k[j];
            }
            orow[k] = acc;
        }
    }
    free(flat); free(coef_lengths); free(offsets);
    return C4A_OK;
}
