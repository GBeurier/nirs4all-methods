/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * DeadBandAugmenter — random dead spectral bands replaced by noise.
 *
 * The reference uses rng.random() for the per-sample probability check and
 * rng.integers(lo, hi+1) for the inclusive integer sample. The noise is
 * drawn via rng.normal(0, noise_std, size=...).
 */

#include "dead_band.h"

#include <stdlib.h>
#include <string.h>

#include "core/augmentations/aug_rng_utils.h"
#include "core/augmentations/spectral/spectral_common.h"

struct c4a_aug_dead_band_state_t {
    int32_t n_bands;
    int32_t width_low;
    int32_t width_high;
    double  noise_std;
    double  probability;
    int32_t variation_scope;
};

c4a_aug_dead_band_state_t* c4a_aug_dead_band_state_new(
    int32_t n_bands,
    int32_t width_low, int32_t width_high,
    double  noise_std, double probability,
    int32_t variation_scope) {
    if (n_bands < 0 || width_low < 1 || width_high < width_low) {
        return NULL;
    }
    if (noise_std < 0.0 || probability < 0.0 || probability > 1.0) {
        return NULL;
    }
    if (variation_scope < 0 || variation_scope > 1) {
        return NULL;
    }
    c4a_aug_dead_band_state_t* s =
        (c4a_aug_dead_band_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->n_bands         = n_bands;
    s->width_low       = width_low;
    s->width_high      = width_high;
    s->noise_std       = noise_std;
    s->probability     = probability;
    s->variation_scope = variation_scope;
    return s;
}

void c4a_aug_dead_band_state_free(c4a_aug_dead_band_state_t* state) {
    free(state);
}

c4a_status_t c4a_aug_dead_band_apply_impl(
    const c4a_aug_dead_band_state_t* state,
    c4a_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || rng == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }

    /* First copy X to out. */
    if (out != X) {
        memcpy(out, X, (size_t)rows * (size_t)cols * sizeof(double));
    }
    if (state->n_bands == 0) {
        return C4A_OK;
    }

    c4a_aug_randint_state_t randint_state;
    c4a_aug_randint_state_reset(&randint_state, rng);
    if (state->variation_scope == 1) {
        /* Batch mode — one probability check, shared dead bands across rows
         * with per-row noise. */
        const double u = c4a_aug_rng_next_double(rng);
        if (u >= state->probability) {
            return C4A_OK;
        }
        for (int32_t b = 0; b < state->n_bands; ++b) {
            const int32_t width = (int32_t)c4a_aug_randint(
                &randint_state, state->width_low, (int64_t)state->width_high + 1);
            const int64_t start_max = (cols - width > 0) ? (cols - width) : 1;
            const int32_t start = (int32_t)c4a_aug_randint(
                &randint_state, 0, start_max);
            const int32_t end = (start + width > (int32_t)cols)
                                ? (int32_t)cols : start + width;
            const int32_t len = end - start;
            /* size=(rows, len) noise fill — row-major, advancing rng. */
            double* tmp = (double*)malloc((size_t)rows * (size_t)len * sizeof(double));
            if (tmp == NULL) return C4A_ERR_OUT_OF_MEMORY;
            c4a_aug_rng_normal_fill(rng, 0.0, state->noise_std, tmp,
                                    (size_t)rows * (size_t)len);
            for (int64_t i = 0; i < rows; ++i) {
                double* row_out = out + (size_t)i * (size_t)cols;
                for (int32_t j = 0; j < len; ++j) {
                    row_out[start + j] =
                        tmp[(size_t)i * (size_t)len + (size_t)j];
                }
            }
            free(tmp);
        }
    } else {
        /* Per-sample mode — independent random per row. */
        for (int64_t i = 0; i < rows; ++i) {
            const double u = c4a_aug_rng_next_double(rng);
            if (u >= state->probability) {
                continue;
            }
            double* row_out = out + (size_t)i * (size_t)cols;
            for (int32_t b = 0; b < state->n_bands; ++b) {
                const int32_t width = (int32_t)c4a_aug_randint(
                    &randint_state, state->width_low, (int64_t)state->width_high + 1);
                const int64_t start_max = (cols - width > 0) ? (cols - width) : 1;
                const int32_t start = (int32_t)c4a_aug_randint(
                    &randint_state, 0, start_max);
                const int32_t end = (start + width > (int32_t)cols)
                                    ? (int32_t)cols : start + width;
                const int32_t len = end - start;
                double* tmp = (double*)malloc((size_t)len * sizeof(double));
                if (tmp == NULL) return C4A_ERR_OUT_OF_MEMORY;
                c4a_aug_rng_normal_fill(rng, 0.0, state->noise_std, tmp,
                                        (size_t)len);
                for (int32_t j = 0; j < len; ++j) {
                    row_out[start + j] = tmp[j];
                }
                free(tmp);
            }
        }
    }
    return C4A_OK;
}
