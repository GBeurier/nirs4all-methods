/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Multiplicative Scatter Correction using the conventional row-wise contract:
 * each spectrum is regressed against a fitted reference spectrum, then
 * corrected by subtracting the row offset and dividing by the row slope.
 */

#include "msc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct n4m_pp_msc_state_t {
    int      fitted;
    int64_t  cols;
    double*  reference;       /* reference spectrum, length cols */
    double   reference_mean;
    double   reference_den;
    int64_t  last_rows;
    double*  last_offsets;    /* row offsets from last transform */
    double*  last_slopes;     /* row slopes from last transform */
};

static void clear_last_transform(n4m_pp_msc_state_t* state) {
    if (state == NULL) return;
    free(state->last_offsets);
    free(state->last_slopes);
    state->last_offsets = NULL;
    state->last_slopes = NULL;
    state->last_rows = 0;
}

n4m_pp_msc_state_t* n4m_pp_msc_state_new(void) {
    n4m_pp_msc_state_t* s = (n4m_pp_msc_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->fitted = 0;
    s->cols = 0;
    s->reference = NULL;
    s->reference_mean = 0.0;
    s->reference_den = 0.0;
    s->last_rows = 0;
    s->last_offsets = NULL;
    s->last_slopes = NULL;
    return s;
}

void n4m_pp_msc_state_free(n4m_pp_msc_state_t* state) {
    if (state == NULL) return;
    free(state->reference);
    clear_last_transform(state);
    free(state);
}

int n4m_pp_msc_state_is_fitted(const n4m_pp_msc_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

n4m_status_t n4m_pp_msc_state_fit(n4m_pp_msc_state_t* state,
                                   const double* X,
                                   int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 2) {
        return N4M_ERR_INVALID_ARGUMENT;
    }

    double* reference = (double*)calloc((size_t)cols, sizeof(double));
    if (reference == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* row = X + (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            reference[j] += row[j];
        }
    }
    for (int64_t j = 0; j < cols; ++j) {
        reference[j] /= (double)rows;
    }

    double ref_mean = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
        ref_mean += reference[j];
    }
    ref_mean /= (double)cols;

    double ref_den = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
        const double centered = reference[j] - ref_mean;
        ref_den += centered * centered;
    }
    if (ref_den <= 0.0) {
        free(reference);
        return N4M_ERR_NUMERICAL_FAILURE;
    }

    free(state->reference);
    clear_last_transform(state);
    state->reference = reference;
    state->reference_mean = ref_mean;
    state->reference_den = ref_den;
    state->cols = cols;
    state->fitted = 1;
    return N4M_OK;
}

n4m_status_t n4m_pp_msc_state_apply(n4m_pp_msc_state_t* state,
                                     const double* X,
                                     int64_t rows, int64_t cols,
                                     double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return N4M_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->cols) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || cols == 0) {
        clear_last_transform(state);
        return N4M_OK;
    }

    double* offsets = (double*)malloc((size_t)rows * sizeof(double));
    double* slopes = (double*)malloc((size_t)rows * sizeof(double));
    if (offsets == NULL || slopes == NULL) {
        free(offsets);
        free(slopes);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const size_t base = (size_t)i * (size_t)cols;
        double row_mean = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            row_mean += X[base + (size_t)j];
        }
        row_mean /= (double)cols;

        double numer = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            numer += (X[base + (size_t)j] - row_mean)
                   * (state->reference[j] - state->reference_mean);
        }
        const double slope = numer / state->reference_den;
        if (slope == 0.0 || !isfinite(slope)) {
            free(offsets);
            free(slopes);
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        const double offset = row_mean - slope * state->reference_mean;
        offsets[i] = offset;
        slopes[i] = slope;
        for (int64_t j = 0; j < cols; ++j) {
            out[base + (size_t)j] = (X[base + (size_t)j] - offset) / slope;
        }
    }

    clear_last_transform(state);
    state->last_offsets = offsets;
    state->last_slopes = slopes;
    state->last_rows = rows;
    return N4M_OK;
}

n4m_status_t n4m_pp_msc_state_inverse_apply(const n4m_pp_msc_state_t* state,
                                             const double* X,
                                             int64_t rows, int64_t cols,
                                             double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return N4M_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->cols || rows != state->last_rows ||
        state->last_offsets == NULL || state->last_slopes == NULL) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const size_t base = (size_t)i * (size_t)cols;
        const double offset = state->last_offsets[i];
        const double slope = state->last_slopes[i];
        for (int64_t j = 0; j < cols; ++j) {
            out[base + (size_t)j] = X[base + (size_t)j] * slope + offset;
        }
    }
    return N4M_OK;
}
