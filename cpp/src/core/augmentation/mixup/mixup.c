/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * MixupAugmenter — within-batch convex combination of pairs of spectra.
 *
 * For an input X of shape (n_samples, n_features) the algorithm draws:
 *   indices : permutation of [0, n_samples)         via Fisher-Yates
 *   lam[i]  : Beta(alpha, alpha) for i in [0, n_samples)
 * and writes
 *   out[i]  = lam[i] * X[i] + (1 - lam[i]) * X[indices[i]]
 *
 * The output has the SAME shape as the input — every output row is a mix
 * of two input rows from the same batch.
 */

#include "mixup.h"

#include <stdlib.h>

#include "core/augmentation/aug_rng_utils.h"
#include "core/augmentation/spectral/spectral_common.h"

struct n4m_aug_mixup_state_t {
    double alpha;
};

n4m_aug_mixup_state_t* n4m_aug_mixup_state_new(double alpha) {
    if (!(alpha > 0.0)) {
        return NULL;
    }
    n4m_aug_mixup_state_t* s =
        (n4m_aug_mixup_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->alpha = alpha;
    return s;
}

void n4m_aug_mixup_state_free(n4m_aug_mixup_state_t* s) {
    free(s);
}

static uint32_t mixup_next_uint32(n4m_aug_randint_state_t* state) {
    if (state->buf_has_high) {
        state->buf_has_high = 0;
        return (uint32_t)(state->buf_value >> 32);
    }
    state->buf_value = n4m_pcg64_engine_next_uint64(state->rng);
    state->buf_has_high = 1;
    return (uint32_t)(state->buf_value & 0xFFFFFFFFu);
}

static int64_t mixup_permutation_index(n4m_aug_randint_state_t* state,
                                       int64_t high) {
    uint32_t mask = (uint32_t)(high - 1);
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    for (;;) {
        const uint32_t r = mixup_next_uint32(state) & mask;
        if ((int64_t)r < high) {
            return (int64_t)r;
        }
    }
}

n4m_status_t n4m_aug_mixup_apply_impl(const n4m_aug_mixup_state_t* state,
                                      n4m_rng_pcg64* rng,
                                      const double* X, int64_t rows, int64_t cols,
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

    int64_t* indices = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    double*  lam     = (double*)malloc((size_t)rows * sizeof(double));
    if (indices == NULL || lam == NULL) {
        free(indices); free(lam);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    /* Match nirs4all reference draw order: permutation first, then beta. */
    for (int64_t i = 0; i < rows; ++i) {
        indices[i] = i;
    }
    n4m_aug_randint_state_t randint_state;
    n4m_aug_randint_state_reset(&randint_state, rng);
    for (int64_t i = rows - 1; i > 0; --i) {
        const int64_t j = mixup_permutation_index(&randint_state, i + 1);
        const int64_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
    for (int64_t i = 0; i < rows; ++i) {
        lam[i] = n4m_aug_rng_beta(rng, state->alpha, state->alpha);
    }

    /* Two-source convex combination. The result must be computed into a
     * separate buffer first to avoid clobbering rows that are referenced
     * by indices[*] later in the loop when out == X. */
    if (out != X) {
        for (int64_t i = 0; i < rows; ++i) {
            const double l = lam[i];
            const double cl = 1.0 - l;
            const double* a = X + (size_t)i * (size_t)cols;
            const double* b = X + (size_t)indices[i] * (size_t)cols;
            double*       o = out + (size_t)i * (size_t)cols;
            for (int64_t j = 0; j < cols; ++j) {
                o[j] = l * a[j] + cl * b[j];
            }
        }
    } else {
        /* in-place: stage to a tmp buffer */
        double* tmp = (double*)malloc((size_t)rows * (size_t)cols * sizeof(double));
        if (tmp == NULL) {
            free(indices); free(lam);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        for (int64_t i = 0; i < rows; ++i) {
            const double l = lam[i];
            const double cl = 1.0 - l;
            const double* a = X + (size_t)i * (size_t)cols;
            const double* b = X + (size_t)indices[i] * (size_t)cols;
            double*       t = tmp + (size_t)i * (size_t)cols;
            for (int64_t j = 0; j < cols; ++j) {
                t[j] = l * a[j] + cl * b[j];
            }
        }
        for (size_t k = 0, n = (size_t)rows * (size_t)cols; k < n; ++k) {
            out[k] = tmp[k];
        }
        free(tmp);
    }

    free(indices);
    free(lam);
    return N4M_OK;
}
