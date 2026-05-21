/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * TemperatureAugmenter — region-specific NIR temperature effects.
 *
 * The six TEMPERATURE_REGION_PARAMS literal-based regions are encoded as
 * (wl_min, wl_max, shift_per_degree, intensity_per_degree,
 *  broadening_per_degree) tuples baked into this implementation.
 */

#include "temperature.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/augmentations/aug_rng_utils.h"
#include "core/augmentations/environmental/aug_interp.h"
#include "core/preprocessing/smoothing/gaussian.h"

/* TEMPERATURE_REGION_PARAMS — must match nirs4all.environmental in name
 * order (Python insertion-ordered dict iteration). */
typedef struct region_params {
    double wl_min;
    double wl_max;
    double shift_per_degree;
    double intensity_per_degree;
    double broadening_per_degree;
} region_params;

static const region_params kTempRegions[6] = {
    /* oh_1st_overtone */    {1400.0, 1520.0, -0.30, -0.002,  0.001},
    /* oh_combination */     {1900.0, 2000.0, -0.40, -0.003,  0.0012},
    /* ch_1st_overtone */    {1650.0, 1780.0, -0.05, -0.0005, 0.0008},
    /* nh_1st_overtone */    {1490.0, 1560.0, -0.20, -0.0015, 0.001},
    /* water_free */         {1380.0, 1420.0, -0.10,  0.003,  0.0008},
    /* water_bound */        {1440.0, 1500.0, -0.35, -0.004,  0.0015},
};

struct n4m_aug_temperature_state_t {
    double  temperature_delta;
    int     use_temp_range;
    double  temp_low;
    double  temp_high;
    int     enable_shift;
    int     enable_intensity;
    int     enable_broadening;
    int     region_specific;
};

n4m_aug_temperature_state_t* n4m_aug_temperature_state_new(
    double temperature_delta,
    int   use_temp_range, double temp_low, double temp_high,
    int   enable_shift, int enable_intensity, int enable_broadening,
    int   region_specific) {
    if (use_temp_range && temp_high < temp_low) {
        return NULL;
    }
    n4m_aug_temperature_state_t* s =
        (n4m_aug_temperature_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->temperature_delta = temperature_delta;
    s->use_temp_range    = use_temp_range ? 1 : 0;
    s->temp_low          = temp_low;
    s->temp_high         = temp_high;
    s->enable_shift      = enable_shift ? 1 : 0;
    s->enable_intensity  = enable_intensity ? 1 : 0;
    s->enable_broadening = enable_broadening ? 1 : 0;
    s->region_specific   = region_specific ? 1 : 0;
    return s;
}

void n4m_aug_temperature_state_free(n4m_aug_temperature_state_t* state) {
    free(state);
}

/* Build region weights via sigmoid edges. */
static void region_weights(const double* wl, int64_t cols,
                            double wl_min, double wl_max, double edge_width,
                            double* w) {
    const double k = 10.0 / edge_width;
    for (int64_t j = 0; j < cols; ++j) {
        const double rising  = 1.0 / (1.0 + exp(-k * (wl[j] - wl_min)));
        const double falling = 1.0 / (1.0 + exp( k * (wl[j] - wl_max)));
        w[j] = rising * falling;
    }
}

/* Apply weighted wavelength shift via np.interp. */
static void apply_wl_shift(const double* row, const double* wl, int64_t cols,
                            double shift, const double* weights,
                            double* row_out) {
    double* shifted_wl = (double*)malloc((size_t)cols * sizeof(double));
    for (int64_t j = 0; j < cols; ++j) {
        shifted_wl[j] = wl[j] + shift * weights[j];
    }
    n4m_aug_np_interp(wl, cols, shifted_wl, row, cols, row_out);
    free(shifted_wl);
}

/* Apply weighted broadening = row * (1 - w) + gaussian_filter1d(row, sigma) * w. */
static n4m_status_t apply_broaden(const double* row, int64_t cols,
                                   double sigma, const double* weights,
                                   double* row_out) {
    if (sigma < 0.1) {
        memcpy(row_out, row, (size_t)cols * sizeof(double));
        return N4M_OK;
    }
    n4m_pp_gaussian_state_t* g = n4m_pp_gaussian_state_new(
        sigma, 0, N4M_PP_GAUSSIAN_REFLECT, 0.0, 4.0);
    if (g == NULL) return N4M_ERR_INVALID_ARGUMENT;
    double* tmp = (double*)malloc((size_t)cols * sizeof(double));
    if (tmp == NULL) {
        n4m_pp_gaussian_state_free(g);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    n4m_pp_gaussian_state_apply(g, row, 1, cols, tmp);
    for (int64_t j = 0; j < cols; ++j) {
        row_out[j] = row[j] * (1.0 - weights[j]) + tmp[j] * weights[j];
    }
    n4m_pp_gaussian_state_free(g);
    free(tmp);
    return N4M_OK;
}

n4m_status_t n4m_aug_temperature_apply_impl(
    const n4m_aug_temperature_state_t* state,
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

    /* Draw per-sample delta_t. */
    double* deltas = (double*)malloc((size_t)rows * sizeof(double));
    if (deltas == NULL) return N4M_ERR_OUT_OF_MEMORY;
    if (state->use_temp_range) {
        n4m_aug_rng_uniform_fill(rng, state->temp_low, state->temp_high,
                                 deltas, (size_t)rows);
    } else {
        for (int64_t i = 0; i < rows; ++i) deltas[i] = state->temperature_delta;
    }

    /* Copy X into out first. */
    if (out != X) {
        memcpy(out, X, (size_t)rows * (size_t)cols * sizeof(double));
    }

    double* weights = (double*)malloc((size_t)cols * sizeof(double));
    double* tmp     = (double*)malloc((size_t)cols * sizeof(double));
    if (weights == NULL || tmp == NULL) {
        free(deltas); free(weights); free(tmp);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double dt = deltas[i];
        if (fabs(dt) < 0.01) continue;

        double* row = out + (size_t)i * (size_t)cols;
        if (state->region_specific) {
            for (int r = 0; r < 6; ++r) {
                const region_params* p = &kTempRegions[r];
                /* If no wavelength in region — skip. */
                int any = 0;
                for (int64_t j = 0; j < cols; ++j) {
                    if (wavelengths[j] >= p->wl_min &&
                        wavelengths[j] <= p->wl_max) {
                        any = 1; break;
                    }
                }
                if (!any) continue;

                region_weights(wavelengths, cols, p->wl_min, p->wl_max, 20.0, weights);

                if (state->enable_shift) {
                    const double shift = p->shift_per_degree * dt;
                    apply_wl_shift(row, wavelengths, cols, shift, weights, tmp);
                    memcpy(row, tmp, (size_t)cols * sizeof(double));
                }
                if (state->enable_intensity) {
                    const double factor = 1.0 + p->intensity_per_degree * dt;
                    for (int64_t j = 0; j < cols; ++j) {
                        row[j] = row[j] * (1.0 + (factor - 1.0) * weights[j]);
                    }
                }
                if (state->enable_broadening && p->broadening_per_degree != 0.0) {
                    const double factor = 1.0 + p->broadening_per_degree * dt;
                    if (factor > 1.0) {
                        const double sigma = (factor - 1.0) * 3.0;
                        apply_broaden(row, cols, sigma, weights, tmp);
                        memcpy(row, tmp, (size_t)cols * sizeof(double));
                    }
                }
            }
        } else {
            /* Uniform mode: average parameters. */
            double avg_shift = 0.0, avg_intensity = 0.0, avg_broadening = 0.0;
            for (int r = 0; r < 6; ++r) {
                avg_shift      += kTempRegions[r].shift_per_degree;
                avg_intensity  += kTempRegions[r].intensity_per_degree;
                avg_broadening += kTempRegions[r].broadening_per_degree;
            }
            avg_shift      /= 6.0;
            avg_intensity  /= 6.0;
            avg_broadening /= 6.0;

            if (state->enable_shift) {
                const double shift = avg_shift * dt;
                /* np.interp(wl, wl + shift, row) — uniform shift. */
                double* shifted_wl = (double*)malloc((size_t)cols * sizeof(double));
                for (int64_t j = 0; j < cols; ++j) shifted_wl[j] = wavelengths[j] + shift;
                n4m_aug_np_interp(wavelengths, cols, shifted_wl, row, cols, tmp);
                free(shifted_wl);
                memcpy(row, tmp, (size_t)cols * sizeof(double));
            }
            if (state->enable_intensity) {
                const double factor = 1.0 + avg_intensity * dt;
                for (int64_t j = 0; j < cols; ++j) {
                    row[j] *= factor;
                }
            }
            if (state->enable_broadening) {
                const double factor = 1.0 + avg_broadening * dt;
                if (factor > 1.0) {
                    const double sigma = (factor - 1.0) * 5.0;
                    if (sigma > 0.1) {
                        n4m_pp_gaussian_state_t* g = n4m_pp_gaussian_state_new(
                            sigma, 0, N4M_PP_GAUSSIAN_REFLECT, 0.0, 4.0);
                        if (g != NULL) {
                            n4m_pp_gaussian_state_apply(g, row, 1, cols, tmp);
                            memcpy(row, tmp, (size_t)cols * sizeof(double));
                            n4m_pp_gaussian_state_free(g);
                        }
                    }
                }
            }
        }
    }

    free(deltas);
    free(weights);
    free(tmp);
    return N4M_OK;
}
