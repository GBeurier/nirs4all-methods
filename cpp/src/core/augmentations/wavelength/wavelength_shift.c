/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * WavelengthShift — implementation.
 *
 * Sample n_samples uniform draws via PCG64 first (NumPy's
 * `rng.uniform(lo, hi, size=n)` recipe), then run linear interpolation for
 * each row. The wavelength grid is captured at `_new` so different rows
 * share the same x-axis allocation.
 */

#include "wavelength_shift.h"

#include <stdlib.h>
#include <string.h>

#include "core/augmentations/spectral/spectral_common.h"

struct c4a_aug_wavelength_shift_state_t {
    c4a_rng_pcg64* rng;          /* not owned */
    double         shift_lo;
    double         shift_hi;
    double*        wavelengths;  /* NULL == use unit grid */
    int64_t        n_wavelengths;
};

c4a_aug_wavelength_shift_state_t* c4a_aug_wavelength_shift_state_new(
    c4a_rng_pcg64* rng,
    double shift_lo, double shift_hi,
    const double* wavelengths, int64_t n_wavelengths) {
    if (rng == NULL) return NULL;
    if (!(shift_hi >= shift_lo)) return NULL;
    if (wavelengths != NULL && n_wavelengths <= 0) return NULL;

    c4a_aug_wavelength_shift_state_t* s =
        (c4a_aug_wavelength_shift_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng        = rng;
    s->shift_lo   = shift_lo;
    s->shift_hi   = shift_hi;
    s->wavelengths = NULL;
    s->n_wavelengths = 0;
    if (wavelengths != NULL) {
        s->wavelengths = (double*)malloc((size_t)n_wavelengths * sizeof(double));
        if (s->wavelengths == NULL) {
            free(s);
            return NULL;
        }
        memcpy(s->wavelengths, wavelengths,
               (size_t)n_wavelengths * sizeof(double));
        s->n_wavelengths = n_wavelengths;
    }
    return s;
}

void c4a_aug_wavelength_shift_state_free(
    c4a_aug_wavelength_shift_state_t* state) {
    if (state == NULL) return;
    free(state->wavelengths);
    free(state);
}

c4a_status_t c4a_aug_wavelength_shift_state_apply(
    c4a_aug_wavelength_shift_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) return C4A_OK;

    /* Resolve / cache the wavelength axis. */
    double* lambdas = state->wavelengths;
    double* allocated_grid = NULL;
    int64_t n_lambdas = state->n_wavelengths;
    if (lambdas == NULL) {
        allocated_grid = (double*)malloc((size_t)cols * sizeof(double));
        if (allocated_grid == NULL) return C4A_ERR_OUT_OF_MEMORY;
        for (int64_t j = 0; j < cols; ++j) {
            allocated_grid[j] = (double)j;
        }
        lambdas = allocated_grid;
        n_lambdas = cols;
    }
    if (n_lambdas != cols) {
        free(allocated_grid);
        return C4A_ERR_SHAPE_MISMATCH;
    }

    /* Pre-generate all per-sample shifts (NumPy: rng.uniform(lo, hi, size=n)). */
    double* shifts = (double*)malloc((size_t)rows * sizeof(double));
    double* query  = (double*)malloc((size_t)cols * sizeof(double));
    if (shifts == NULL || query == NULL) {
        free(shifts); free(query); free(allocated_grid);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        shifts[i] = c4a_aug_uniform(state->rng, state->shift_lo, state->shift_hi);
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double s = shifts[i];
        for (int64_t j = 0; j < cols; ++j) {
            query[j] = lambdas[j] - s;
        }
        c4a_aug_interp_linear(query, cols,
                               lambdas, X + (size_t)i * (size_t)cols, cols,
                               out + (size_t)i * (size_t)cols);
    }
    free(shifts);
    free(query);
    free(allocated_grid);
    return C4A_OK;
}
