/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * InstrumentalBroadeningAugmenter — Gaussian convolution with FWHM-derived
 * sigma. Uses the existing c4a_pp_gaussian engine for the convolution.
 */

#include "instrument_broaden.h"

#include <math.h>
#include <stdlib.h>

#include "core/augmentations/aug_rng_utils.h"
#include "core/preprocessing/smoothing/gaussian.h"

struct c4a_aug_instrument_broaden_state_t {
    double  fwhm;
    int     use_fwhm_range;
    double  fwhm_low;
    double  fwhm_high;
    int32_t variation_scope;
};

c4a_aug_instrument_broaden_state_t* c4a_aug_instrument_broaden_state_new(
    double fwhm,
    int    use_fwhm_range, double fwhm_low, double fwhm_high,
    int32_t variation_scope) {
    if (!(fwhm > 0.0)) {
        return NULL;
    }
    if (use_fwhm_range && (fwhm_low <= 0.0 || fwhm_high < fwhm_low)) {
        return NULL;
    }
    if (variation_scope < 0 || variation_scope > 1) {
        return NULL;
    }
    c4a_aug_instrument_broaden_state_t* s =
        (c4a_aug_instrument_broaden_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->fwhm            = fwhm;
    s->use_fwhm_range  = use_fwhm_range ? 1 : 0;
    s->fwhm_low        = fwhm_low;
    s->fwhm_high       = fwhm_high;
    s->variation_scope = variation_scope;
    return s;
}

void c4a_aug_instrument_broaden_state_free(
    c4a_aug_instrument_broaden_state_t* state) {
    free(state);
}

/* numpy.median(diff(wavelengths)) — diff has length cols-1. */
static double median_diff(const double* wl, int64_t cols) {
    if (cols < 2) {
        return 1.0;
    }
    const int64_t n = cols - 1;
    double* d = (double*)malloc((size_t)n * sizeof(double));
    if (d == NULL) {
        return 1.0;
    }
    for (int64_t j = 0; j < n; ++j) {
        d[j] = wl[j + 1] - wl[j];
    }
    /* in-place insertion sort — fine for the sizes we expect (cols ~ 200). */
    for (int64_t i = 1; i < n; ++i) {
        const double k = d[i];
        int64_t j = i - 1;
        while (j >= 0 && d[j] > k) {
            d[j + 1] = d[j];
            --j;
        }
        d[j + 1] = k;
    }
    double med;
    if (n % 2 == 1) {
        med = d[n / 2];
    } else {
        med = 0.5 * (d[n / 2 - 1] + d[n / 2]);
    }
    free(d);
    return med;
}

/* Convolve a single row with a Gaussian of given sigma, scipy.ndimage-
 * compatible (reflect mode, truncate=4.0). Allocates and frees its own
 * Gaussian state on each call — this is a cold path. */
static c4a_status_t convolve_row(const double* row_in, int64_t cols,
                                  double sigma, double* row_out) {
    /* scipy.ndimage.gaussian_filter1d defaults: mode=reflect, truncate=4.0. */
    c4a_pp_gaussian_state_t* g = c4a_pp_gaussian_state_new(
        sigma, 0, C4A_PP_GAUSSIAN_REFLECT, 0.0, 4.0);
    if (g == NULL) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const c4a_status_t st = c4a_pp_gaussian_state_apply(g, row_in, 1, cols, row_out);
    c4a_pp_gaussian_state_free(g);
    return st;
}

static const double FWHM_TO_SIGMA = 0.4246609001440095;
/* 1 / (2 * sqrt(2 * ln 2)) — precomputed at compile time below. */

c4a_status_t c4a_aug_instrument_broaden_apply_impl(
    const c4a_aug_instrument_broaden_state_t* state,
    c4a_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
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
    /* Recompute the conversion factor exactly the way numpy does:
     *   2 * sqrt(2 * log(2))   (natural log). */
    const double k = 2.0 * sqrt(2.0 * log(2.0));
    (void)FWHM_TO_SIGMA;
    const double wl_step = (wavelengths != NULL) ? median_diff(wavelengths, cols) : 1.0;

    if (state->use_fwhm_range) {
        if (state->variation_scope == 1) {
            /* single FWHM for all rows */
            const double fwhm = c4a_aug_rng_uniform(rng,
                state->fwhm_low, state->fwhm_high);
            const double sigma_pts = fwhm / k / wl_step;
            for (int64_t i = 0; i < rows; ++i) {
                const c4a_status_t st = convolve_row(
                    X + (size_t)i * (size_t)cols, cols, sigma_pts,
                    out + (size_t)i * (size_t)cols);
                if (st != C4A_OK) return st;
            }
        } else {
            /* per-sample FWHM */
            double* fwhms = (double*)malloc((size_t)rows * sizeof(double));
            if (fwhms == NULL) return C4A_ERR_OUT_OF_MEMORY;
            c4a_aug_rng_uniform_fill(rng,
                state->fwhm_low, state->fwhm_high, fwhms, (size_t)rows);
            for (int64_t i = 0; i < rows; ++i) {
                const double sigma_pts = fwhms[i] / k / wl_step;
                const c4a_status_t st = convolve_row(
                    X + (size_t)i * (size_t)cols, cols, sigma_pts,
                    out + (size_t)i * (size_t)cols);
                if (st != C4A_OK) { free(fwhms); return st; }
            }
            free(fwhms);
        }
    } else {
        /* Fixed FWHM — no RNG draws. */
        const double sigma_pts = state->fwhm / k / wl_step;
        for (int64_t i = 0; i < rows; ++i) {
            const c4a_status_t st = convolve_row(
                X + (size_t)i * (size_t)cols, cols, sigma_pts,
                out + (size_t)i * (size_t)cols);
            if (st != C4A_OK) return st;
        }
    }
    return C4A_OK;
}
