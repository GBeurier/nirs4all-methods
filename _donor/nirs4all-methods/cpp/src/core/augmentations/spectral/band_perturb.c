/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * BandPerturbation — implementation.
 *
 * Mirrors the per-sample variation_scope branch of
 * `nirs4all.operators.augmentation.spectral.BandPerturbation`. The four
 * random parameter blocks (centers, widths, gains, offsets) are drawn in
 * separate row-major sweeps so the PCG64 stream order matches NumPy's
 * `size=(n_samples, n_bands)` allocation pattern across four successive
 * `rng.integers` / `rng.uniform` calls.
 */

#include "band_perturb.h"

#include <stdlib.h>
#include <string.h>

#include "core/augmentations/spectral/spectral_common.h"

struct n4m_aug_band_perturb_state_t {
    n4m_rng_pcg64* rng;     /* not owned */
    int32_t        n_bands;
    int32_t        bw_lo;
    int32_t        bw_hi;
    double         gain_lo;
    double         gain_hi;
    double         offset_lo;
    double         offset_hi;
};

n4m_aug_band_perturb_state_t* n4m_aug_band_perturb_state_new(
    n4m_rng_pcg64* rng,
    int32_t n_bands,
    int32_t bw_lo, int32_t bw_hi,
    double gain_lo, double gain_hi,
    double offset_lo, double offset_hi) {
    if (rng == NULL) return NULL;
    if (n_bands < 0) return NULL;
    if (bw_hi <= bw_lo) return NULL;
    if (bw_lo < 0) return NULL;
    if (gain_hi < gain_lo) return NULL;
    if (offset_hi < offset_lo) return NULL;

    n4m_aug_band_perturb_state_t* s =
        (n4m_aug_band_perturb_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng       = rng;
    s->n_bands   = n_bands;
    s->bw_lo     = bw_lo;
    s->bw_hi     = bw_hi;
    s->gain_lo   = gain_lo;
    s->gain_hi   = gain_hi;
    s->offset_lo = offset_lo;
    s->offset_hi = offset_hi;
    return s;
}

void n4m_aug_band_perturb_state_free(n4m_aug_band_perturb_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_band_perturb_state_apply(
    n4m_aug_band_perturb_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (X != out) {
        memcpy(out, X, (size_t)rows * (size_t)cols * sizeof(double));
    }
    if (rows == 0 || cols == 0 || state->n_bands == 0) return N4M_OK;

    const int32_t nb = state->n_bands;
    const size_t total = (size_t)rows * (size_t)nb;
    int64_t* centers = (int64_t*)malloc(total * sizeof(int64_t));
    int64_t* widths  = (int64_t*)malloc(total * sizeof(int64_t));
    double*  gains   = (double*)malloc(total * sizeof(double));
    double*  offsets = (double*)malloc(total * sizeof(double));
    if (centers == NULL || widths == NULL || gains == NULL || offsets == NULL) {
        free(centers); free(widths); free(gains); free(offsets);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    /* Block 1: centers (NumPy buffers 32-bit halves; resetting once per
     * `integers` call discards leftover halves to match NumPy's per-call
     * buffer reset for `_rand_int64`). */
    n4m_aug_randint_state_t rs;
    n4m_aug_randint_state_reset(&rs, state->rng);
    for (size_t k = 0; k < total; ++k) {
        centers[k] = n4m_aug_randint(&rs, 0, cols);
    }
    /* Block 2: widths. */
    n4m_aug_randint_state_reset(&rs, state->rng);
    for (size_t k = 0; k < total; ++k) {
        widths[k] = n4m_aug_randint(&rs, state->bw_lo, state->bw_hi);
    }
    /* Block 3: gains. */
    for (size_t k = 0; k < total; ++k) {
        gains[k] = n4m_aug_uniform(state->rng, state->gain_lo, state->gain_hi);
    }
    /* Block 4: offsets. */
    for (size_t k = 0; k < total; ++k) {
        offsets[k] = n4m_aug_uniform(state->rng,
                                      state->offset_lo, state->offset_hi);
    }
    for (int64_t i = 0; i < rows; ++i) {
        double* row = out + (size_t)i * (size_t)cols;
        for (int32_t b = 0; b < nb; ++b) {
            const size_t idx = (size_t)i * (size_t)nb + (size_t)b;
            const int64_t center = centers[idx];
            const int64_t width  = widths[idx];
            int64_t start = center - width / 2;
            int64_t end   = center + width / 2;
            if (start < 0) start = 0;
            if (end   > cols) end = cols;
            if (start >= end) continue;
            const double gain   = gains[idx];
            const double offset = offsets[idx];
            for (int64_t j = start; j < end; ++j) {
                row[j] = row[j] * gain + offset;
            }
        }
    }
    free(centers);
    free(widths);
    free(gains);
    free(offsets);
    return N4M_OK;
}
