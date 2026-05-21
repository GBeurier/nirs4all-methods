/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * BatchEffectAugmenter — per-row or batch-wide additive offset, slope and
 * multiplicative gain.
 */

#include "batch_effect.h"

#include <stdlib.h>

#include "core/augmentation/aug_rng_utils.h"

struct n4m_aug_batch_effect_state_t {
    double  offset_std;
    double  slope_std;
    double  gain_std;
    int32_t variation_scope;
};

n4m_aug_batch_effect_state_t* n4m_aug_batch_effect_state_new(
    double offset_std, double slope_std, double gain_std,
    int32_t variation_scope) {
    if (offset_std < 0.0 || slope_std < 0.0 || gain_std < 0.0) {
        return NULL;
    }
    if (variation_scope < 0 || variation_scope > 1) {
        return NULL;
    }
    n4m_aug_batch_effect_state_t* s =
        (n4m_aug_batch_effect_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->offset_std       = offset_std;
    s->slope_std        = slope_std;
    s->gain_std         = gain_std;
    s->variation_scope  = variation_scope;
    return s;
}

void n4m_aug_batch_effect_state_free(n4m_aug_batch_effect_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_batch_effect_apply_impl(
    const n4m_aug_batch_effect_state_t* state,
    n4m_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out) {
    if (state == NULL || rng == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }

    /* Build the normalised x axis. */
    double* x = (double*)malloc((size_t)cols * sizeof(double));
    if (x == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    if (wavelengths != NULL) {
        double wl_min = wavelengths[0], wl_max = wavelengths[0], wl_sum = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            if (wavelengths[j] < wl_min) wl_min = wavelengths[j];
            if (wavelengths[j] > wl_max) wl_max = wavelengths[j];
            wl_sum += wavelengths[j];
        }
        const double wl_mean = wl_sum / (double)cols;
        const double wl_range = wl_max - wl_min + 1e-10;
        for (int64_t j = 0; j < cols; ++j) {
            x[j] = (wavelengths[j] - wl_mean) / wl_range;
        }
    } else {
        /* arange(cols) — mean = (cols - 1) / 2, range = cols - 1 + 1e-10 */
        const double mean = (double)(cols - 1) * 0.5;
        const double rng_x = (double)(cols - 1) + 1e-10;
        for (int64_t j = 0; j < cols; ++j) {
            x[j] = ((double)j - mean) / rng_x;
        }
    }

    if (state->variation_scope == 1) {
        /* Batch mode — one (offset, slope, gain) for all rows. */
        const double offset = n4m_aug_rng_standard_normal(rng) * state->offset_std;
        const double slope  = n4m_aug_rng_standard_normal(rng) * state->slope_std;
        const double gain   = n4m_aug_rng_standard_normal(rng) * state->gain_std + 1.0;
        for (int64_t i = 0; i < rows; ++i) {
            const double* row_in  = X + (size_t)i * (size_t)cols;
            double*       row_out = out + (size_t)i * (size_t)cols;
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = row_in[j] * gain + offset + slope * x[j];
            }
        }
    } else {
        /* Per-sample mode — n_samples * 3 normal draws. nirs4all order:
         *   offsets (n_samples), slopes (n_samples), gains (n_samples). */
        double* offsets = (double*)malloc((size_t)rows * sizeof(double));
        double* slopes  = (double*)malloc((size_t)rows * sizeof(double));
        double* gains   = (double*)malloc((size_t)rows * sizeof(double));
        if (offsets == NULL || slopes == NULL || gains == NULL) {
            free(x); free(offsets); free(slopes); free(gains);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        n4m_aug_rng_normal_fill(rng, 0.0, state->offset_std, offsets, (size_t)rows);
        n4m_aug_rng_normal_fill(rng, 0.0, state->slope_std,  slopes,  (size_t)rows);
        n4m_aug_rng_normal_fill(rng, 1.0, state->gain_std,   gains,   (size_t)rows);
        for (int64_t i = 0; i < rows; ++i) {
            const double offset_i = offsets[i];
            const double slope_i  = slopes[i];
            const double gain_i   = gains[i];
            const double* row_in  = X + (size_t)i * (size_t)cols;
            double*       row_out = out + (size_t)i * (size_t)cols;
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = row_in[j] * gain_i + offset_i + slope_i * x[j];
            }
        }
        free(offsets);
        free(slopes);
        free(gains);
    }
    free(x);
    return N4M_OK;
}
