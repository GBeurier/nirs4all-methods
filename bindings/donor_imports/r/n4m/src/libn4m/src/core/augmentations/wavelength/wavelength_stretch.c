/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * WavelengthStretch — implementation.
 *
 * Stretch factors come from `rng.uniform(stretch_lo, stretch_hi, size=n)`;
 * the stream order matches NumPy when factors are drawn before the loop.
 */

#include "wavelength_stretch.h"

#include <stdlib.h>
#include <string.h>

#include "core/augmentations/spectral/spectral_common.h"

struct n4m_aug_wavelength_stretch_state_t {
    n4m_rng_pcg64* rng;          /* not owned */
    double         stretch_lo;
    double         stretch_hi;
    double*        wavelengths;  /* NULL == use unit grid */
    int64_t        n_wavelengths;
};

n4m_aug_wavelength_stretch_state_t* n4m_aug_wavelength_stretch_state_new(
    n4m_rng_pcg64* rng,
    double stretch_lo, double stretch_hi,
    const double* wavelengths, int64_t n_wavelengths) {
    if (rng == NULL) return NULL;
    if (!(stretch_hi >= stretch_lo)) return NULL;
    /* Zero factor would divide by zero in the query. Reject. */
    if (!(stretch_lo > 0.0)) return NULL;
    if (wavelengths != NULL && n_wavelengths <= 0) return NULL;

    n4m_aug_wavelength_stretch_state_t* s =
        (n4m_aug_wavelength_stretch_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng           = rng;
    s->stretch_lo    = stretch_lo;
    s->stretch_hi    = stretch_hi;
    s->wavelengths   = NULL;
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

void n4m_aug_wavelength_stretch_state_free(
    n4m_aug_wavelength_stretch_state_t* state) {
    if (state == NULL) return;
    free(state->wavelengths);
    free(state);
}

n4m_status_t n4m_aug_wavelength_stretch_state_apply(
    n4m_aug_wavelength_stretch_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) return N4M_OK;

    /* Resolve / cache the wavelength axis. */
    double* lambdas = state->wavelengths;
    double* allocated_grid = NULL;
    int64_t n_lambdas = state->n_wavelengths;
    if (lambdas == NULL) {
        allocated_grid = (double*)malloc((size_t)cols * sizeof(double));
        if (allocated_grid == NULL) return N4M_ERR_OUT_OF_MEMORY;
        for (int64_t j = 0; j < cols; ++j) {
            allocated_grid[j] = (double)j;
        }
        lambdas = allocated_grid;
        n_lambdas = cols;
    }
    if (n_lambdas != cols) {
        free(allocated_grid);
        return N4M_ERR_SHAPE_MISMATCH;
    }

    /* Mean of lambdas (numpy: np.mean). */
    double sum = 0.0;
    for (int64_t j = 0; j < cols; ++j) sum += lambdas[j];
    const double center = sum / (double)cols;

    /* Pre-generate factors (numpy: rng.uniform(lo, hi, size=n_samples)). */
    double* factors = (double*)malloc((size_t)rows * sizeof(double));
    double* query   = (double*)malloc((size_t)cols * sizeof(double));
    if (factors == NULL || query == NULL) {
        free(factors); free(query); free(allocated_grid);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        factors[i] = n4m_aug_uniform(state->rng,
                                      state->stretch_lo, state->stretch_hi);
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double f = factors[i];
        for (int64_t j = 0; j < cols; ++j) {
            query[j] = center + (lambdas[j] - center) / f;
        }
        n4m_aug_interp_linear(query, cols,
                               lambdas, X + (size_t)i * (size_t)cols, cols,
                               out + (size_t)i * (size_t)cols);
    }
    free(factors);
    free(query);
    free(allocated_grid);
    return N4M_OK;
}
