/* SPDX-License-Identifier: CECILL-2.1 */

#include "wavelet_features.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct c4a_pp_wavelet_features_state_t {
    c4a_wavelet_family_t family;
    c4a_wavelet_mode_t   mode;
    int32_t              max_level;
};

c4a_pp_wavelet_features_state_t* c4a_pp_wavelet_features_state_new(
    c4a_wavelet_family_t family,
    c4a_wavelet_mode_t   mode,
    int32_t              max_level) {
    if (max_level < 0) return NULL;
    c4a_pp_wavelet_features_state_t* s =
        (c4a_pp_wavelet_features_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->family    = family;
    s->mode      = mode;
    s->max_level = max_level;
    return s;
}

void c4a_pp_wavelet_features_state_free(c4a_pp_wavelet_features_state_t* state) {
    free(state);
}

int32_t c4a_pp_wavelet_features_actual_level(
    const c4a_pp_wavelet_features_state_t* state, int64_t cols) {
    if (state == NULL || cols < 1) return 0;
    const int32_t mx = c4a_wavelet_dwt_max_level(cols, state->family);
    return (state->max_level < mx) ? state->max_level : mx;
}

int64_t c4a_pp_wavelet_features_output_size(
    const c4a_pp_wavelet_features_state_t* state, int64_t cols) {
    const int32_t lvl =
        c4a_pp_wavelet_features_actual_level(state, cols);
    return 4 * (int64_t)(lvl + 1);
}

/* Compute the four band statistics into out_stats (length 4):
 *   [mean, std, energy, entropy] */
static void band_stats(const double* x, int64_t n, double* out_stats) {
    if (n <= 0) {
        out_stats[0] = 0.0;
        out_stats[1] = 0.0;
        out_stats[2] = 0.0;
        out_stats[3] = 0.0;
        return;
    }
    double sum_x   = 0.0;
    double sum_x2  = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum_x  += x[i];
        sum_x2 += x[i] * x[i];
    }
    const double mean = sum_x / (double)n;
    /* Population std (ddof=0). */
    double sum_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = x[i] - mean;
        sum_sq += d * d;
    }
    const double std_pop = sqrt(sum_sq / (double)n);
    const double energy  = sum_x2;
    /* Shannon entropy of |x|^2 normalised to a probability distribution.
     * If the total energy is zero, return 0. */
    double entropy = 0.0;
    if (energy > 0.0) {
        const double inv_total = 1.0 / energy;
        for (int64_t i = 0; i < n; ++i) {
            const double p = (x[i] * x[i]) * inv_total;
            if (p > 0.0) {
                entropy -= p * log(p);
            }
        }
    }
    out_stats[0] = mean;
    out_stats[1] = std_pop;
    out_stats[2] = energy;
    out_stats[3] = entropy;
}

c4a_status_t c4a_pp_wavelet_features_state_apply(
    const c4a_pp_wavelet_features_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const int32_t level =
        c4a_pp_wavelet_features_actual_level(state, cols);
    const int64_t expected_cols = 4 * (int64_t)(level + 1);
    if (out_cols != expected_cols) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0) return C4A_OK;

    int64_t* coef_lengths = (int64_t*)malloc((size_t)(level + 1) * sizeof(int64_t));
    int64_t* offsets      = (int64_t*)malloc((size_t)(level + 1) * sizeof(int64_t));
    if (coef_lengths == NULL || offsets == NULL) {
        free(coef_lengths); free(offsets);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    int64_t total = 0;
    c4a_status_t st = c4a_wavelet_wavedec_lengths(
        cols, state->family, state->mode, level, &total, coef_lengths);
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
    double* coeffs = (double*)malloc((size_t)total * sizeof(double));
    if (coeffs == NULL) {
        free(coef_lengths); free(offsets);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        st = c4a_wavelet_wavedec(row, cols, state->family, state->mode,
                                  level, coeffs, coef_lengths, offsets, total);
        if (st != C4A_OK) {
            free(coeffs);
            free(coef_lengths); free(offsets);
            return st;
        }
        double* o = out + (size_t)i * (size_t)out_cols;
        for (int32_t k = 0; k <= level; ++k) {
            band_stats(coeffs + offsets[k], coef_lengths[k], o + 4 * k);
        }
    }
    free(coeffs);
    free(coef_lengths); free(offsets);
    return C4A_OK;
}
