/* SPDX-License-Identifier: CECILL-2.1 */

#include "wavelet_features.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct n4m_pp_wavelet_features_state_t {
    n4m_wavelet_family_t              family;
    n4m_wavelet_mode_t                mode;
    int32_t                           max_level;
    n4m_pp_wavelet_features_entropy_t entropy_mode;
};

n4m_pp_wavelet_features_state_t* n4m_pp_wavelet_features_state_new(
    n4m_wavelet_family_t family,
    n4m_wavelet_mode_t   mode,
    int32_t              max_level) {
    return n4m_pp_wavelet_features_state_new_ex(
        family, mode, max_level, N4M_PP_WAVELET_FEATURES_ENTROPY_ENERGY);
}

n4m_pp_wavelet_features_state_t* n4m_pp_wavelet_features_state_new_ex(
    n4m_wavelet_family_t family,
    n4m_wavelet_mode_t   mode,
    int32_t              max_level,
    n4m_pp_wavelet_features_entropy_t entropy_mode) {
    if (max_level < 0 ||
        (entropy_mode != N4M_PP_WAVELET_FEATURES_ENTROPY_ENERGY &&
         entropy_mode != N4M_PP_WAVELET_FEATURES_ENTROPY_HISTOGRAM)) {
        return NULL;
    }
    n4m_pp_wavelet_features_state_t* s =
        (n4m_pp_wavelet_features_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->family       = family;
    s->mode         = mode;
    s->max_level    = max_level;
    s->entropy_mode = entropy_mode;
    return s;
}

void n4m_pp_wavelet_features_state_free(n4m_pp_wavelet_features_state_t* state) {
    free(state);
}

int32_t n4m_pp_wavelet_features_actual_level(
    const n4m_pp_wavelet_features_state_t* state, int64_t cols) {
    if (state == NULL || cols < 1) return 0;
    const int32_t mx = n4m_wavelet_dwt_max_level(cols, state->family);
    return (state->max_level < mx) ? state->max_level : mx;
}

int64_t n4m_pp_wavelet_features_output_size(
    const n4m_pp_wavelet_features_state_t* state, int64_t cols) {
    const int32_t lvl =
        n4m_pp_wavelet_features_actual_level(state, cols);
    return 4 * (int64_t)(lvl + 1);
}

static double entropy_energy(const double* x, int64_t n, double energy) {
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
    return entropy;
}

static double entropy_histogram_10(const double* x, int64_t n) {
    double xmin = x[0];
    double xmax = x[0];
    for (int64_t i = 1; i < n; ++i) {
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
    }
    if (!(xmax > xmin)) {
        return 0.0;
    }

    double counts[10];
    for (int32_t b = 0; b < 10; ++b) {
        counts[b] = 0.0;
    }
    const double scale = 10.0 / (xmax - xmin);
    for (int64_t i = 0; i < n; ++i) {
        int32_t b = (int32_t)((x[i] - xmin) * scale);
        if (b < 0) b = 0;
        if (b > 9) b = 9;
        counts[b] += 1.0;
    }

    double entropy = 0.0;
    const double inv_n = 1.0 / (double)n;
    for (int32_t b = 0; b < 10; ++b) {
        const double p = counts[b] * inv_n;
        if (p > 0.0) {
            entropy -= p * log(p);
        }
    }
    return entropy;
}

/* Compute the four band statistics into out_stats (length 4):
 *   [mean, std, energy, entropy] */
static void band_stats(const double* x, int64_t n,
                       n4m_pp_wavelet_features_entropy_t entropy_mode,
                       double* out_stats) {
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
    const double entropy =
        (entropy_mode == N4M_PP_WAVELET_FEATURES_ENTROPY_HISTOGRAM)
            ? entropy_histogram_10(x, n)
            : entropy_energy(x, n, energy);
    out_stats[0] = mean;
    out_stats[1] = std_pop;
    out_stats[2] = energy;
    out_stats[3] = entropy;
}

n4m_status_t n4m_pp_wavelet_features_state_apply(
    const n4m_pp_wavelet_features_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    int64_t out_cols, double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const int32_t level =
        n4m_pp_wavelet_features_actual_level(state, cols);
    const int64_t expected_cols = 4 * (int64_t)(level + 1);
    if (out_cols != expected_cols) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0) return N4M_OK;

    int64_t* coef_lengths = (int64_t*)malloc((size_t)(level + 1) * sizeof(int64_t));
    int64_t* offsets      = (int64_t*)malloc((size_t)(level + 1) * sizeof(int64_t));
    if (coef_lengths == NULL || offsets == NULL) {
        free(coef_lengths); free(offsets);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    int64_t total = 0;
    n4m_status_t st = n4m_wavelet_wavedec_lengths(
        cols, state->family, state->mode, level, &total, coef_lengths);
    if (st != N4M_OK) {
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
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        st = n4m_wavelet_wavedec(row, cols, state->family, state->mode,
                                  level, coeffs, coef_lengths, offsets, total);
        if (st != N4M_OK) {
            free(coeffs);
            free(coef_lengths); free(offsets);
            return st;
        }
        double* o = out + (size_t)i * (size_t)out_cols;
        for (int32_t k = 0; k <= level; ++k) {
            band_stats(coeffs + offsets[k], coef_lengths[k],
                       state->entropy_mode, o + 4 * k);
        }
    }
    free(coeffs);
    free(coef_lengths); free(offsets);
    return N4M_OK;
}
