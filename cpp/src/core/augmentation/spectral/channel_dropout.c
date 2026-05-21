/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ChannelDropout — implementation.
 *
 * The full dropout mask is generated in a single row-major sweep before
 * any interpolation work runs, so the PCG64 stream order matches NumPy's
 * `rng.random(size=(rows, cols)) < dropout_prob`.
 */

#include "channel_dropout.h"

#include <stdlib.h>
#include <string.h>

#include "core/augmentation/spectral/spectral_common.h"

struct n4m_aug_channel_dropout_state_t {
    n4m_rng_pcg64*                  rng;
    double                          p;
    n4m_aug_channel_dropout_mode_t  mode;
};

n4m_aug_channel_dropout_state_t* n4m_aug_channel_dropout_state_new(
    n4m_rng_pcg64* rng,
    double dropout_prob,
    n4m_aug_channel_dropout_mode_t mode) {
    if (rng == NULL) return NULL;
    if (!(dropout_prob >= 0.0 && dropout_prob <= 1.0)) return NULL;
    if (mode != N4M_AUG_CHANNEL_DROPOUT_ZERO &&
        mode != N4M_AUG_CHANNEL_DROPOUT_INTERP) {
        return NULL;
    }
    n4m_aug_channel_dropout_state_t* s =
        (n4m_aug_channel_dropout_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng  = rng;
    s->p    = dropout_prob;
    s->mode = mode;
    return s;
}

void n4m_aug_channel_dropout_state_free(n4m_aug_channel_dropout_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_channel_dropout_state_apply(
    n4m_aug_channel_dropout_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (X != out) {
        memcpy(out, X, (size_t)rows * (size_t)cols * sizeof(double));
    }
    if (rows == 0 || cols == 0) return N4M_OK;

    /* Generate the (rows, cols) mask via PCG64 next_double in row-major
     * order — matches numpy's rng.random(size=(rows, cols)). */
    uint8_t* mask = (uint8_t*)malloc((size_t)rows * (size_t)cols);
    if (mask == NULL) return N4M_ERR_OUT_OF_MEMORY;
    for (size_t k = 0; k < (size_t)rows * (size_t)cols; ++k) {
        const double u = n4m_pcg64_engine_next_double(state->rng);
        mask[k] = (u < state->p) ? 1u : 0u;
    }

    if (state->mode == N4M_AUG_CHANNEL_DROPOUT_ZERO) {
        for (int64_t i = 0; i < rows; ++i) {
            double*       row  = out + (size_t)i * (size_t)cols;
            const uint8_t* mrow = mask + (size_t)i * (size_t)cols;
            for (int64_t j = 0; j < cols; ++j) {
                if (mrow[j]) row[j] = 0.0;
            }
        }
        free(mask);
        return N4M_OK;
    }

    /* INTERP mode — scratch buffers for kept indices / values per row. */
    double*  x_kept = (double*)malloc((size_t)cols * sizeof(double));
    double*  y_kept = (double*)malloc((size_t)cols * sizeof(double));
    double*  x_drop = (double*)malloc((size_t)cols * sizeof(double));
    double*  y_drop = (double*)malloc((size_t)cols * sizeof(double));
    if (x_kept == NULL || y_kept == NULL || x_drop == NULL || y_drop == NULL) {
        free(mask); free(x_kept); free(y_kept); free(x_drop); free(y_drop);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double*  src  = X + (size_t)i * (size_t)cols;
        double*        row  = out + (size_t)i * (size_t)cols;
        const uint8_t* mrow = mask + (size_t)i * (size_t)cols;
        int64_t n_kept = 0, n_drop = 0;
        for (int64_t j = 0; j < cols; ++j) {
            if (mrow[j]) {
                x_drop[n_drop++] = (double)j;
            } else {
                x_kept[n_kept] = (double)j;
                y_kept[n_kept] = src[j];
                ++n_kept;
            }
        }
        if (n_drop == 0) continue;
        if (n_kept == 0) continue;  /* all-dropped row — leave as copy */
        n4m_aug_interp_linear(x_drop, n_drop, x_kept, y_kept, n_kept, y_drop);
        for (int64_t k = 0; k < n_drop; ++k) {
            row[(int64_t)x_drop[k]] = y_drop[k];
        }
    }
    free(x_kept); free(y_kept); free(x_drop); free(y_drop);
    free(mask);
    return N4M_OK;
}
