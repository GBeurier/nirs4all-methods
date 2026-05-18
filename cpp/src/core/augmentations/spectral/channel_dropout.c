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

#include "core/augmentations/spectral/spectral_common.h"

struct c4a_aug_channel_dropout_state_t {
    c4a_rng_pcg64*                  rng;
    double                          p;
    c4a_aug_channel_dropout_mode_t  mode;
};

c4a_aug_channel_dropout_state_t* c4a_aug_channel_dropout_state_new(
    c4a_rng_pcg64* rng,
    double dropout_prob,
    c4a_aug_channel_dropout_mode_t mode) {
    if (rng == NULL) return NULL;
    if (!(dropout_prob >= 0.0 && dropout_prob <= 1.0)) return NULL;
    if (mode != C4A_AUG_CHANNEL_DROPOUT_ZERO &&
        mode != C4A_AUG_CHANNEL_DROPOUT_INTERP) {
        return NULL;
    }
    c4a_aug_channel_dropout_state_t* s =
        (c4a_aug_channel_dropout_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng  = rng;
    s->p    = dropout_prob;
    s->mode = mode;
    return s;
}

void c4a_aug_channel_dropout_state_free(c4a_aug_channel_dropout_state_t* state) {
    free(state);
}

c4a_status_t c4a_aug_channel_dropout_state_apply(
    c4a_aug_channel_dropout_state_t* state,
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

    /* Generate the (rows, cols) mask via PCG64 next_double in row-major
     * order — matches numpy's rng.random(size=(rows, cols)). */
    uint8_t* mask = (uint8_t*)malloc((size_t)rows * (size_t)cols);
    if (mask == NULL) return C4A_ERR_OUT_OF_MEMORY;
    for (size_t k = 0; k < (size_t)rows * (size_t)cols; ++k) {
        const double u = c4a_pcg64_engine_next_double(state->rng);
        mask[k] = (u < state->p) ? 1u : 0u;
    }

    if (state->mode == C4A_AUG_CHANNEL_DROPOUT_ZERO) {
        for (int64_t i = 0; i < rows; ++i) {
            double*       row  = out + (size_t)i * (size_t)cols;
            const uint8_t* mrow = mask + (size_t)i * (size_t)cols;
            for (int64_t j = 0; j < cols; ++j) {
                if (mrow[j]) row[j] = 0.0;
            }
        }
        free(mask);
        return C4A_OK;
    }

    /* INTERP mode — scratch buffers for kept indices / values per row. */
    double*  x_kept = (double*)malloc((size_t)cols * sizeof(double));
    double*  y_kept = (double*)malloc((size_t)cols * sizeof(double));
    double*  x_drop = (double*)malloc((size_t)cols * sizeof(double));
    double*  y_drop = (double*)malloc((size_t)cols * sizeof(double));
    if (x_kept == NULL || y_kept == NULL || x_drop == NULL || y_drop == NULL) {
        free(mask); free(x_kept); free(y_kept); free(x_drop); free(y_drop);
        return C4A_ERR_OUT_OF_MEMORY;
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
        c4a_aug_interp_linear(x_drop, n_drop, x_kept, y_kept, n_kept, y_drop);
        for (int64_t k = 0; k < n_drop; ++k) {
            row[(int64_t)x_drop[k]] = y_drop[k];
        }
    }
    free(x_kept); free(y_kept); free(x_drop); free(y_drop);
    free(mask);
    return C4A_OK;
}
