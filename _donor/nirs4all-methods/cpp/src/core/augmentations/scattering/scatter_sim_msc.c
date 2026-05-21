/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ScatterSimulationMSC — simple linear x_aug = a + b * x with per-sample
 * uniform random (a, b) draws.
 */

#include "scatter_sim_msc.h"

#include <stdlib.h>

#include "core/augmentations/aug_rng_utils.h"

struct n4m_aug_scatter_sim_state_t {
    double a_low;
    double a_high;
    double b_low;
    double b_high;
};

n4m_aug_scatter_sim_state_t* n4m_aug_scatter_sim_state_new(
    double a_low, double a_high, double b_low, double b_high) {
    if (a_high < a_low || b_high < b_low) {
        return NULL;
    }
    n4m_aug_scatter_sim_state_t* s =
        (n4m_aug_scatter_sim_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->a_low  = a_low;
    s->a_high = a_high;
    s->b_low  = b_low;
    s->b_high = b_high;
    return s;
}

void n4m_aug_scatter_sim_state_free(n4m_aug_scatter_sim_state_t* state) {
    free(state);
}

n4m_status_t n4m_aug_scatter_sim_apply_impl(
    const n4m_aug_scatter_sim_state_t* state,
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

    /* nirs4all draws a vector of `a` values (length rows), then a vector of
     * `b` values (length rows). Mirror that draw order exactly. */
    double* a = (double*)malloc((size_t)rows * sizeof(double));
    double* b = (double*)malloc((size_t)rows * sizeof(double));
    if (a == NULL || b == NULL) {
        free(a); free(b);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    n4m_aug_rng_uniform_fill(rng, state->a_low, state->a_high, a, (size_t)rows);
    n4m_aug_rng_uniform_fill(rng, state->b_low, state->b_high, b, (size_t)rows);

    for (int64_t i = 0; i < rows; ++i) {
        const double ai = a[i];
        const double bi = b[i];
        const double* row_in  = X + (size_t)i * (size_t)cols;
        double*       row_out = out + (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            row_out[j] = ai + bi * row_in[j];
        }
    }
    free(a);
    free(b);
    return N4M_OK;
}
