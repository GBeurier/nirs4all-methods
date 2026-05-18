/* SPDX-License-Identifier: CECILL-2.1 */

#include "wavelet_denoise.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct c4a_pp_wavelet_denoise_state_t {
    c4a_wavelet_family_t              family;
    c4a_wavelet_mode_t                mode;
    int32_t                            level;
    c4a_pp_wavelet_threshold_mode_t   threshold_mode;
    c4a_pp_wavelet_noise_estimator_t  noise_estimator;
};

c4a_pp_wavelet_denoise_state_t* c4a_pp_wavelet_denoise_state_new(
    c4a_wavelet_family_t              family,
    c4a_wavelet_mode_t                mode,
    int32_t                            level,
    c4a_pp_wavelet_threshold_mode_t   threshold_mode,
    c4a_pp_wavelet_noise_estimator_t  noise_estimator) {
    if (level < 0) return NULL;
    c4a_pp_wavelet_denoise_state_t* s =
        (c4a_pp_wavelet_denoise_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->family          = family;
    s->mode            = mode;
    s->level           = level;
    s->threshold_mode  = threshold_mode;
    s->noise_estimator = noise_estimator;
    return s;
}

void c4a_pp_wavelet_denoise_state_free(c4a_pp_wavelet_denoise_state_t* state) {
    free(state);
}

/* Quickselect median in-place (NumPy default behaviour: even n returns
 * the average of the two middle elements; odd n returns the middle). */
static int cmp_double(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

static double median_inplace(double* buf, int64_t n) {
    if (n <= 0) return 0.0;
    qsort(buf, (size_t)n, sizeof(double), cmp_double);
    if (n & 1) return buf[n / 2];
    return 0.5 * (buf[n / 2 - 1] + buf[n / 2]);
}

static void apply_threshold(double* buf, int64_t n, double thr,
                             c4a_pp_wavelet_threshold_mode_t mode) {
    if (mode == C4A_PP_WD_THRESHOLD_SOFT) {
        for (int64_t i = 0; i < n; ++i) {
            const double v = buf[i];
            const double a = fabs(v);
            if (a <= thr) {
                buf[i] = 0.0;
            } else {
                /* Sign preserved; (a - thr) magnitude. */
                buf[i] = (v > 0.0 ? 1.0 : -1.0) * (a - thr);
            }
        }
    } else {  /* hard */
        for (int64_t i = 0; i < n; ++i) {
            if (fabs(buf[i]) <= thr) buf[i] = 0.0;
        }
    }
}

c4a_status_t c4a_pp_wavelet_denoise_state_apply(
    const c4a_pp_wavelet_denoise_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0) return C4A_OK;

    const int32_t max_level =
        c4a_wavelet_dwt_max_level(cols, state->family);
    int32_t level = state->level;
    if (level > max_level) level = max_level;
    if (level < 1) {
        /* No detail levels -> just copy input. */
        memcpy(out, X, (size_t)rows * (size_t)cols * sizeof(double));
        return C4A_OK;
    }

    /* Compute per-row buffer sizes. */
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
    /* Scratch for median (finest detail length). */
    double* mscratch =
        (double*)malloc((size_t)coef_lengths[level] * sizeof(double));
    if (coeffs == NULL || mscratch == NULL) {
        free(coeffs); free(mscratch);
        free(coef_lengths); free(offsets);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    const double universal_factor = sqrt(2.0 * log((double)cols));

    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        double*       row_out = out + (size_t)i * (size_t)cols;
        st = c4a_wavelet_wavedec(row, cols, state->family, state->mode,
                                  level, coeffs, coef_lengths, offsets,
                                  total);
        if (st != C4A_OK) {
            free(coeffs); free(mscratch);
            free(coef_lengths); free(offsets);
            return st;
        }
        /* Finest detail occupies coeffs[offsets[level] : offsets[level] +
         * coef_lengths[level]]. */
        const int64_t fd_len = coef_lengths[level];
        const double* fd     = coeffs + offsets[level];
        double sigma;
        if (state->noise_estimator == C4A_PP_WD_NOISE_MEDIAN) {
            for (int64_t j = 0; j < fd_len; ++j) mscratch[j] = fabs(fd[j]);
            sigma = median_inplace(mscratch, fd_len) / 0.6745;
        } else {  /* std (ddof=0, matching np.std default) */
            double mean = 0.0;
            for (int64_t j = 0; j < fd_len; ++j) mean += fd[j];
            mean /= (double)fd_len;
            double var = 0.0;
            for (int64_t j = 0; j < fd_len; ++j) {
                const double d = fd[j] - mean;
                var += d * d;
            }
            var /= (double)fd_len;
            sigma = sqrt(var);
        }
        const double thr = sigma * universal_factor;
        for (int32_t k = 1; k <= level; ++k) {
            apply_threshold(coeffs + offsets[k], coef_lengths[k], thr,
                             state->threshold_mode);
        }
        /* Reconstruct directly into row_out. */
        st = c4a_wavelet_waverec(coeffs, coef_lengths, offsets, level,
                                  state->family, state->mode, cols, row_out);
        if (st != C4A_OK) {
            free(coeffs); free(mscratch);
            free(coef_lengths); free(offsets);
            return st;
        }
    }

    free(coeffs); free(mscratch);
    free(coef_lengths); free(offsets);
    return C4A_OK;
}
