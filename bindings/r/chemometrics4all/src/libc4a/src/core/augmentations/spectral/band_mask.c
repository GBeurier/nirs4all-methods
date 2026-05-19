/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * BandMasking — implementation.
 *
 * The Python reference draws (1) n_bands_per_sample of length n_samples,
 * (2) centers (n_samples, max_bands), (3) widths (n_samples, max_bands)
 * in that order, then walks the rows. We mirror the order exactly.
 */

#include "band_mask.h"

#include <stdlib.h>
#include <string.h>

#include "core/augmentations/spectral/spectral_common.h"

struct c4a_aug_band_mask_state_t {
    c4a_rng_pcg64*           rng;     /* not owned */
    int32_t                  nb_lo;
    int32_t                  nb_hi;
    int32_t                  bw_lo;
    int32_t                  bw_hi;
    c4a_aug_band_mask_mode_t mode;
};

c4a_aug_band_mask_state_t* c4a_aug_band_mask_state_new(
    c4a_rng_pcg64* rng,
    int32_t n_bands_lo, int32_t n_bands_hi,
    int32_t bw_lo, int32_t bw_hi,
    c4a_aug_band_mask_mode_t mode) {
    if (rng == NULL) return NULL;
    if (n_bands_lo < 0 || n_bands_hi < n_bands_lo) return NULL;
    if (bw_hi <= bw_lo || bw_lo < 0) return NULL;
    if (mode != C4A_AUG_BAND_MASK_ZERO && mode != C4A_AUG_BAND_MASK_INTERP) {
        return NULL;
    }
    c4a_aug_band_mask_state_t* s =
        (c4a_aug_band_mask_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng   = rng;
    s->nb_lo = n_bands_lo;
    s->nb_hi = n_bands_hi;
    s->bw_lo = bw_lo;
    s->bw_hi = bw_hi;
    s->mode  = mode;
    return s;
}

void c4a_aug_band_mask_state_free(c4a_aug_band_mask_state_t* state) {
    free(state);
}

c4a_status_t c4a_aug_band_mask_state_apply(
    c4a_aug_band_mask_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (X != out) {
        memcpy(out, X, (size_t)rows * (size_t)cols * sizeof(double));
    }
    if (rows == 0 || cols == 0) return C4A_OK;

    const int32_t max_bands = state->nb_hi;
    const size_t  total     = (size_t)rows * (size_t)max_bands;

    int64_t* n_per_sample = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    int64_t* centers = NULL;
    int64_t* widths  = NULL;
    if (max_bands > 0) {
        centers = (int64_t*)malloc(total * sizeof(int64_t));
        widths  = (int64_t*)malloc(total * sizeof(int64_t));
    }
    if (n_per_sample == NULL ||
        (max_bands > 0 && (centers == NULL || widths == NULL))) {
        free(n_per_sample); free(centers); free(widths);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    /* Order mirrors python: n_per_sample first, then centers (full), then
     * widths (full). Each block resets its own 32-bit buffer because
     * NumPy emits a fresh `integers` call per array. */
    c4a_aug_randint_state_t rs;
    c4a_aug_randint_state_reset(&rs, state->rng);
    for (int64_t i = 0; i < rows; ++i) {
        n_per_sample[i] = c4a_aug_randint(&rs,
                                           state->nb_lo, state->nb_hi + 1);
    }
    c4a_aug_randint_state_reset(&rs, state->rng);
    for (size_t k = 0; k < total; ++k) {
        centers[k] = c4a_aug_randint(&rs, 0, cols);
    }
    c4a_aug_randint_state_reset(&rs, state->rng);
    for (size_t k = 0; k < total; ++k) {
        widths[k] = c4a_aug_randint(&rs, state->bw_lo, state->bw_hi);
    }

    for (int64_t i = 0; i < rows; ++i) {
        double* row = out + (size_t)i * (size_t)cols;
        const int64_t nb_i = n_per_sample[i];
        for (int64_t b = 0; b < nb_i; ++b) {
            const size_t idx = (size_t)i * (size_t)max_bands + (size_t)b;
            const int64_t center = centers[idx];
            const int64_t width  = widths[idx];
            int64_t start = center - width / 2;
            int64_t end   = center + width / 2;
            if (start < 0)    start = 0;
            if (end   > cols) end   = cols;
            if (start >= end) continue;
            if (state->mode == C4A_AUG_BAND_MASK_ZERO) {
                for (int64_t j = start; j < end; ++j) row[j] = 0.0;
            } else {
                /* Linear ramp matching nirs4all's BandMasking interp branch:
                 *   val_start = row[start-1] if start > 0 else row[start]
                 *   val_end   = row[end]     if end   < cols else row[end-1]
                 *   for x_local in range(end - start):
                 *     slope = (val_end - val_start) / (end - start + 1)
                 *     row[start + x_local] = val_start + slope * (x_local + 1)
                 */
                const double val_start =
                    (start > 0)    ? row[start - 1] : row[start];
                const double val_end =
                    (end   < cols) ? row[end]       : row[end - 1];
                const int64_t denom = end - start + 1;
                const double slope =
                    (denom > 0) ? (val_end - val_start) / (double)denom : 0.0;
                for (int64_t j = start; j < end; ++j) {
                    const int64_t local = j - start;
                    row[j] = val_start + slope * (double)(local + 1);
                }
            }
        }
    }
    free(n_per_sample);
    free(centers);
    free(widths);
    return C4A_OK;
}
