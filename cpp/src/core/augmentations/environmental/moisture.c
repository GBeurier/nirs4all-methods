/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * MoistureAugmenter — water-band specific shifts and intensity adjustments.
 */

#include "moisture.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/augmentations/aug_rng_utils.h"
#include "core/augmentations/environmental/aug_interp.h"

struct c4a_aug_moisture_state_t {
    double water_activity_delta;
    int    use_aw_range;
    double aw_low;
    double aw_high;
    double reference_water_activity;
    double free_water_fraction;
    double bound_water_shift;
    double moisture_content;
    int    enable_shift;
    int    enable_intensity;
};

c4a_aug_moisture_state_t* c4a_aug_moisture_state_new(
    double water_activity_delta,
    int    use_aw_range, double aw_low, double aw_high,
    double reference_water_activity,
    double free_water_fraction,
    double bound_water_shift,
    double moisture_content,
    int    enable_shift, int enable_intensity) {
    if (use_aw_range && aw_high < aw_low) {
        return NULL;
    }
    c4a_aug_moisture_state_t* s =
        (c4a_aug_moisture_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->water_activity_delta     = water_activity_delta;
    s->use_aw_range             = use_aw_range ? 1 : 0;
    s->aw_low                   = aw_low;
    s->aw_high                  = aw_high;
    s->reference_water_activity = reference_water_activity;
    s->free_water_fraction      = free_water_fraction;
    s->bound_water_shift        = bound_water_shift;
    s->moisture_content         = moisture_content;
    s->enable_shift             = enable_shift ? 1 : 0;
    s->enable_intensity         = enable_intensity ? 1 : 0;
    return s;
}

void c4a_aug_moisture_state_free(c4a_aug_moisture_state_t* state) {
    free(state);
}

static double clipd(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void gaussian_region(const double* wl, int64_t cols,
                             double center, double width, double* out) {
    for (int64_t j = 0; j < cols; ++j) {
        const double z = (wl[j] - center) / width;
        out[j] = exp(-0.5 * z * z);
    }
}

static void shift_water_band(const double* row, const double* wl, int64_t cols,
                              double center, double width, double shift,
                              double* row_out) {
    if (fabs(shift) < 0.01) {
        memcpy(row_out, row, (size_t)cols * sizeof(double));
        return;
    }
    double* weights    = (double*)malloc((size_t)cols * sizeof(double));
    double* shifted_wl = (double*)malloc((size_t)cols * sizeof(double));
    gaussian_region(wl, cols, center, width, weights);
    for (int64_t j = 0; j < cols; ++j) shifted_wl[j] = wl[j] + shift * weights[j];
    c4a_aug_np_interp(wl, cols, shifted_wl, row, cols, row_out);
    free(weights); free(shifted_wl);
}

c4a_status_t c4a_aug_moisture_apply_impl(
    const c4a_aug_moisture_state_t* state,
    c4a_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out) {
    if (state == NULL || rng == NULL || X == NULL || out == NULL
        || wavelengths == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }

    double* aw_deltas = (double*)malloc((size_t)rows * sizeof(double));
    if (aw_deltas == NULL) return C4A_ERR_OUT_OF_MEMORY;
    if (state->use_aw_range) {
        c4a_aug_rng_uniform_fill(rng, state->aw_low, state->aw_high,
                                 aw_deltas, (size_t)rows);
    } else {
        for (int64_t i = 0; i < rows; ++i) {
            aw_deltas[i] = state->water_activity_delta;
        }
    }

    if (out != X) {
        memcpy(out, X, (size_t)rows * (size_t)cols * sizeof(double));
    }

    double* tmp   = (double*)malloc((size_t)cols * sizeof(double));
    double* wreg1 = (double*)malloc((size_t)cols * sizeof(double));
    double* wreg2 = (double*)malloc((size_t)cols * sizeof(double));
    if (tmp == NULL || wreg1 == NULL || wreg2 == NULL) {
        free(aw_deltas); free(tmp); free(wreg1); free(wreg2);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    if (state->enable_intensity) {
        gaussian_region(wavelengths, cols, 1435.0, 40.0, wreg1);
        gaussian_region(wavelengths, cols, 1930.0, 35.0, wreg2);
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double aw = clipd(state->reference_water_activity + aw_deltas[i],
                                0.0, 1.0);
        const double sigmoid_factor = 1.0 / (1.0 + exp(-8.0 * (aw - 0.5)));
        const double free_fraction  = state->free_water_fraction * sigmoid_factor;
        const double bound_fraction = 1.0 - free_fraction;

        double* row = out + (size_t)i * (size_t)cols;

        if (state->enable_shift) {
            shift_water_band(row, wavelengths, cols, 1435.0, 50.0,
                             state->bound_water_shift * bound_fraction, tmp);
            memcpy(row, tmp, (size_t)cols * sizeof(double));
            shift_water_band(row, wavelengths, cols, 1930.0, 40.0,
                             state->bound_water_shift * 0.8 * bound_fraction, tmp);
            memcpy(row, tmp, (size_t)cols * sizeof(double));
        }
        if (state->enable_intensity) {
            const double enhancement = state->moisture_content / 0.10 - 1.0;
            double sum = 0.0;
            for (int64_t j = 0; j < cols; ++j) sum += row[j];
            const double abs_mean = fabs(sum / (double)cols);
            for (int64_t j = 0; j < cols; ++j) {
                row[j] += enhancement * 0.1 * (wreg1[j] + wreg2[j]) * abs_mean;
            }
        }
    }
    free(aw_deltas);
    free(tmp);
    free(wreg1);
    free(wreg2);
    return C4A_OK;
}
