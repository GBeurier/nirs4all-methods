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

#include "core/augmentations/aug_rng_utils.h"

struct c4a_aug_mixup_state_t {
    double alpha;
};

c4a_aug_mixup_state_t* c4a_aug_mixup_state_new(double alpha) {
    if (!(alpha > 0.0)) {
        return NULL;
    }
    c4a_aug_mixup_state_t* s =
        (c4a_aug_mixup_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->alpha = alpha;
    return s;
}

void c4a_aug_mixup_state_free(c4a_aug_mixup_state_t* s) {
    free(s);
}

c4a_status_t c4a_aug_mixup_apply_impl(const c4a_aug_mixup_state_t* state,
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

    int64_t* indices = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    double*  lam     = (double*)malloc((size_t)rows * sizeof(double));
    if (indices == NULL || lam == NULL) {
        free(indices); free(lam);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    /* Match nirs4all reference draw order: permutation first, then beta. */
    c4a_aug_rng_permutation(rng, indices, rows);
    for (int64_t i = 0; i < rows; ++i) {
        lam[i] = c4a_aug_rng_beta(rng, state->alpha, state->alpha);
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
            return C4A_ERR_OUT_OF_MEMORY;
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
    return C4A_OK;
}
