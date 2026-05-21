/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * LocalWavelengthWarp — implementation (linear-interp variant).
 *
 * Random draws: a flat (n_samples, n_control_points) block of
 * `rng.uniform(-max_shift, +max_shift)` calls, generated in row-major order
 * to match NumPy's `size=(n_samples, n_control_points)` allocation pattern.
 */

#include "local_warp.h"

#include <stdlib.h>
#include <string.h>

#include "core/augmentation/spectral/spectral_common.h"

struct n4m_aug_local_warp_state_t {
    n4m_rng_pcg64* rng;          /* not owned */
    int32_t        n_control;
    double         max_shift;
    double*        wavelengths;  /* NULL == unit grid */
    int64_t        n_wavelengths;
};

n4m_aug_local_warp_state_t* n4m_aug_local_warp_state_new(
    n4m_rng_pcg64* rng,
    int32_t n_control_points,
    double max_shift,
    const double* wavelengths, int64_t n_wavelengths) {
    if (rng == NULL) return NULL;
    if (n_control_points < 2) return NULL;
    if (!(max_shift >= 0.0)) return NULL;
    if (wavelengths != NULL && n_wavelengths <= 0) return NULL;

    n4m_aug_local_warp_state_t* s =
        (n4m_aug_local_warp_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng           = rng;
    s->n_control     = n_control_points;
    s->max_shift     = max_shift;
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

void n4m_aug_local_warp_state_free(
    n4m_aug_local_warp_state_t* state) {
    if (state == NULL) return;
    free(state->wavelengths);
    free(state);
}

n4m_status_t n4m_aug_local_warp_state_apply(
    n4m_aug_local_warp_state_t* state,
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

    const int32_t k = state->n_control;
    /* Build control x-coordinates: np.linspace(lambdas[0], lambdas[-1], k). */
    double* ctrl_x = (double*)malloc((size_t)k * sizeof(double));
    double* ctrl_y = (double*)malloc((size_t)rows * (size_t)k * sizeof(double));
    double* shift  = (double*)malloc((size_t)cols * sizeof(double));
    double* query  = (double*)malloc((size_t)cols * sizeof(double));
    if (ctrl_x == NULL || ctrl_y == NULL || shift == NULL || query == NULL) {
        free(ctrl_x); free(ctrl_y); free(shift); free(query); free(allocated_grid);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    {
        const double x0 = lambdas[0];
        const double x1 = lambdas[cols - 1];
        if (k == 1) {
            ctrl_x[0] = x0;
        } else {
            const double step = (x1 - x0) / (double)(k - 1);
            for (int32_t c = 0; c < k; ++c) {
                ctrl_x[c] = x0 + step * (double)c;
            }
        }
    }
    /* Pre-generate all control-y shifts: rng.uniform(-mx, mx,
     * size=(rows, k)) in row-major order. */
    const double lo = -state->max_shift;
    const double hi =  state->max_shift;
    for (int64_t i = 0; i < rows; ++i) {
        for (int32_t c = 0; c < k; ++c) {
            ctrl_y[(size_t)i * (size_t)k + (size_t)c] =
                n4m_aug_uniform(state->rng, lo, hi);
        }
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* row_ctrl_y = ctrl_y + (size_t)i * (size_t)k;
        /* shift[j] = np.interp(lambdas[j], ctrl_x, ctrl_y_i). */
        n4m_aug_interp_linear(lambdas, cols, ctrl_x, row_ctrl_y, k, shift);
        for (int64_t j = 0; j < cols; ++j) {
            query[j] = lambdas[j] - shift[j];
        }
        n4m_aug_interp_linear(query, cols,
                               lambdas, X + (size_t)i * (size_t)cols, cols,
                               out + (size_t)i * (size_t)cols);
    }
    free(ctrl_x);
    free(ctrl_y);
    free(shift);
    free(query);
    free(allocated_grid);
    return N4M_OK;
}
