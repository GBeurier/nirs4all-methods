/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * LinearBaselineDrift — reference implementation.
 *
 * RNG draw order matches `linear_drift.h`:
 *   1. `rows` offset uniforms in [offset_lo, offset_hi)
 *   2. `rows` slope  uniforms in [slope_lo,  slope_hi )
 *
 * The wavelength axis is taken as `arange(n_features)` (the implicit branch
 * in `nirs4all.spectral.LinearBaselineDrift._transform_impl`); the centered
 * axis is `arange - mean(arange) = arange - (n_features - 1)/2`. The mean
 * is computed exactly as `(cols - 1) / 2.0` to avoid any drift from
 * accumulating arange in a loop.
 */

#include "linear_drift.h"

#include <stdlib.h>

struct c4a_aug_linear_drift_state_t {
    c4a_rng_pcg64* rng;   /* borrowed; not owned. */
    double offset_min;
    double offset_max;
    double slope_min;
    double slope_max;
};

c4a_aug_linear_drift_state_t* c4a_aug_linear_drift_state_new(
    c4a_rng_pcg64* rng,
    double offset_min, double offset_max,
    double slope_min,  double slope_max) {
    c4a_aug_linear_drift_state_t* s =
        (c4a_aug_linear_drift_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng        = rng;
    s->offset_min = offset_min;
    s->offset_max = offset_max;
    s->slope_min  = slope_min;
    s->slope_max  = slope_max;
    return s;
}

void c4a_aug_linear_drift_state_free(c4a_aug_linear_drift_state_t* state) {
    free(state);
}

c4a_status_t c4a_aug_linear_drift_state_apply(
    c4a_aug_linear_drift_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || state->rng == NULL) return C4A_ERR_NULL_POINTER;
    if (X == NULL || out == NULL)             return C4A_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0)                 return C4A_ERR_INVALID_ARGUMENT;
    if (state->offset_max < state->offset_min ||
        state->slope_max  < state->slope_min) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) return C4A_OK;

    double* offsets = (double*)malloc((size_t)rows * sizeof(double));
    if (offsets == NULL) return C4A_ERR_OUT_OF_MEMORY;
    const double off_span = state->offset_max - state->offset_min;
    for (int64_t i = 0; i < rows; ++i) {
        const double u = c4a_pcg64_engine_next_double(state->rng);
        offsets[(size_t)i] = state->offset_min + off_span * u;
    }

    double* slopes = (double*)malloc((size_t)rows * sizeof(double));
    if (slopes == NULL) {
        free(offsets);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    const double slope_span = state->slope_max - state->slope_min;
    for (int64_t i = 0; i < rows; ++i) {
        const double u = c4a_pcg64_engine_next_double(state->rng);
        slopes[(size_t)i] = state->slope_min + slope_span * u;
    }

    /* lambdas_centered = j - (n_features - 1)/2 */
    const double mean_j = (double)(cols - 1) * 0.5;
    for (int64_t i = 0; i < rows; ++i) {
        const double a  = offsets[(size_t)i];
        const double b  = slopes[(size_t)i];
        const size_t off = (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            const double lc = (double)j - mean_j;
            out[off + (size_t)j] = X[off + (size_t)j] + a + b * lc;
        }
    }

    free(slopes);
    free(offsets);
    return C4A_OK;
}
