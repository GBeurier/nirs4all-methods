/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * EMSCDistortionAugmenter — per-sample EMSC-style distortion.
 */

#include "emsc_distort.h"

#include <math.h>
#include <stdlib.h>

#include "core/augmentations/aug_rng_utils.h"

struct n4m_aug_emsc_distort_state_t {
    double  mult_low;
    double  mult_high;
    double  add_low;
    double  add_high;
    int32_t polynomial_order;
    double  polynomial_strength;
    double  correlation;
};

n4m_aug_emsc_distort_state_t* n4m_aug_emsc_distort_state_new(
    double mult_low, double mult_high,
    double add_low,  double add_high,
    int32_t polynomial_order,
    double  polynomial_strength,
    double  correlation) {
    if (mult_high < mult_low || add_high < add_low || polynomial_order < 0) {
        return NULL;
    }
    if (!(correlation >= -1.0 && correlation <= 1.0)) {
        return NULL;
    }
    n4m_aug_emsc_distort_state_t* s =
        (n4m_aug_emsc_distort_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->mult_low            = mult_low;
    s->mult_high           = mult_high;
    s->add_low             = add_low;
    s->add_high            = add_high;
    s->polynomial_order    = polynomial_order;
    s->polynomial_strength = polynomial_strength;
    s->correlation         = correlation;
    return s;
}

void n4m_aug_emsc_distort_state_free(n4m_aug_emsc_distort_state_t* state) {
    free(state);
}

static double clip(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

n4m_status_t n4m_aug_emsc_distort_apply_impl(
    const n4m_aug_emsc_distort_state_t* state,
    n4m_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out) {
    if (state == NULL || rng == NULL || X == NULL || out == NULL
        || wavelengths == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }

    /* Pre-compute wl_norm in [-1, 1]. */
    double wl_min = wavelengths[0];
    double wl_max = wavelengths[0];
    for (int64_t j = 1; j < cols; ++j) {
        if (wavelengths[j] < wl_min) wl_min = wavelengths[j];
        if (wavelengths[j] > wl_max) wl_max = wavelengths[j];
    }
    const double wl_span = wl_max - wl_min;
    double* wl_norm = (double*)malloc((size_t)cols * sizeof(double));
    if (wl_norm == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    if (wl_span <= 0.0) {
        for (int64_t j = 0; j < cols; ++j) wl_norm[j] = 0.0;
    } else {
        for (int64_t j = 0; j < cols; ++j) {
            wl_norm[j] = 2.0 * (wavelengths[j] - wl_min) / wl_span - 1.0;
        }
    }

    const double b_mean = (state->mult_low + state->mult_high) * 0.5;
    const double b_std  = (state->mult_high - state->mult_low) * 0.25;
    const double a_mean = (state->add_low + state->add_high) * 0.5;
    const double a_std  = (state->add_high - state->add_low) * 0.25;
    const double a_dev_scale = sqrt(1.0 - state->correlation * state->correlation);

    /* Allocate polynomial-order coefficients buffer. */
    double* coef = NULL;
    if (state->polynomial_order > 0) {
        coef = (double*)malloc((size_t)state->polynomial_order * sizeof(double));
        if (coef == NULL) {
            free(wl_norm);
            return N4M_ERR_OUT_OF_MEMORY;
        }
    }

    for (int64_t i = 0; i < rows; ++i) {
        /* Sample b — clipped normal. */
        double b_raw = n4m_aug_rng_standard_normal(rng) * b_std + b_mean;
        const double b = clip(b_raw, state->mult_low, state->mult_high);

        /* b_deviation = (b - 1) / b_std (using post-clip b). */
        const double b_dev = (b_std > 0.0) ? (b - 1.0) / b_std : 0.0;
        const double a_mean_correlated = a_mean
            - state->correlation * a_std * b_dev;
        double a_raw = n4m_aug_rng_standard_normal(rng) * (a_std * a_dev_scale)
                       + a_mean_correlated;
        const double a = clip(a_raw, state->add_low, state->add_high);

        /* Sample polynomial coefficients. */
        for (int32_t k = 1; k <= state->polynomial_order; ++k) {
            const double coef_std = state->polynomial_strength
                                    / sqrt((double)k);
            coef[k - 1] = n4m_aug_rng_standard_normal(rng) * coef_std;
        }

        const double* row_in  = X + (size_t)i * (size_t)cols;
        double*       row_out = out + (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            double r = a + b * row_in[j];
            if (state->polynomial_order > 0) {
                double wl_pow = 1.0;
                for (int32_t k = 0; k < state->polynomial_order; ++k) {
                    wl_pow *= wl_norm[j];
                    r += coef[k] * wl_pow;
                }
            }
            row_out[j] = r;
        }
    }

    free(coef);
    free(wl_norm);
    return N4M_OK;
}
