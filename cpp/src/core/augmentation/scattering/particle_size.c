/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * ParticleSizeAugmenter — apply wavelength-dependent scattering distortions
 * driven by per-sample particle size draws.
 *
 * The scatter_noise step uses a Gaussian-smoothed normal noise pattern of
 * sigma=3 (matching scipy.ndimage.gaussian_filter1d with truncate=4.0,
 * the SciPy default). We reuse the existing n4m_pp_gaussian engine to get
 * bit-exact kernel and convolution semantics.
 */

#include "particle_size.h"

#include <math.h>
#include <stdlib.h>

#include "core/augmentation/aug_rng_utils.h"
#include "core/preprocessing/smoothing/gaussian.h"

struct n4m_aug_particle_size_state_t {
    double mean_size_um;
    double size_variation_um;
    int    use_size_range;
    double size_range_low_um;
    double size_range_high_um;
    double reference_size_um;
    double wavelength_exponent;
    double size_effect_strength;
    int    include_path_length;
    double path_length_sensitivity;
};

n4m_aug_particle_size_state_t* n4m_aug_particle_size_state_new(
    double mean_size_um, double size_variation_um,
    int   use_size_range, double size_range_low_um, double size_range_high_um,
    double reference_size_um, double wavelength_exponent,
    double size_effect_strength,
    int   include_path_length, double path_length_sensitivity) {
    if (!(reference_size_um > 0.0)) {
        return NULL;
    }
    if (use_size_range && !(size_range_high_um >= size_range_low_um)) {
        return NULL;
    }
    n4m_aug_particle_size_state_t* s =
        (n4m_aug_particle_size_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->mean_size_um            = mean_size_um;
    s->size_variation_um       = size_variation_um;
    s->use_size_range          = use_size_range ? 1 : 0;
    s->size_range_low_um       = size_range_low_um;
    s->size_range_high_um      = size_range_high_um;
    s->reference_size_um       = reference_size_um;
    s->wavelength_exponent     = wavelength_exponent;
    s->size_effect_strength    = size_effect_strength;
    s->include_path_length     = include_path_length ? 1 : 0;
    s->path_length_sensitivity = path_length_sensitivity;
    return s;
}

void n4m_aug_particle_size_state_free(n4m_aug_particle_size_state_t* state) {
    free(state);
}

static double clip(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

n4m_status_t n4m_aug_particle_size_apply_impl(
    const n4m_aug_particle_size_state_t* state,
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

    /* Draw n particle sizes. */
    double* sizes = (double*)malloc((size_t)rows * sizeof(double));
    if (sizes == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    if (state->use_size_range) {
        n4m_aug_rng_uniform_fill(rng,
            state->size_range_low_um, state->size_range_high_um,
            sizes, (size_t)rows);
    } else {
        n4m_aug_rng_normal_fill(rng,
            state->mean_size_um, state->size_variation_um,
            sizes, (size_t)rows);
        for (int64_t i = 0; i < rows; ++i) {
            sizes[i] = clip(sizes[i], 5.0, 500.0);
        }
    }

    /* Build wl_factor[j] = clip(wl[j]/1500, 0.1, 10)^(-exponent). */
    double* wl_factor = (double*)malloc((size_t)cols * sizeof(double));
    if (wl_factor == NULL) {
        free(sizes);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t j = 0; j < cols; ++j) {
        const double wl_safe = (wavelengths[j] < 1.0) ? 1.0 : wavelengths[j];
        const double wl_norm = clip(wl_safe / 1500.0, 0.1, 10.0);
        wl_factor[j] = pow(wl_norm, -state->wavelength_exponent);
    }

    /* Scatter noise: sigma=3 Gaussian smoothing of a normal noise vector
     * per row. We use the existing n4m_pp_gaussian engine with default
     * scipy parameters (reflect mode, truncate=4.0). */
    n4m_pp_gaussian_state_t* gauss = n4m_pp_gaussian_state_new(
        3.0, 0, N4M_PP_GAUSSIAN_REFLECT, 0.0, 4.0);
    if (gauss == NULL) {
        free(sizes); free(wl_factor);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    double* noise_in  = (double*)malloc((size_t)cols * sizeof(double));
    double* noise_out = (double*)malloc((size_t)cols * sizeof(double));
    if (noise_in == NULL || noise_out == NULL) {
        free(sizes); free(wl_factor);
        n4m_pp_gaussian_state_free(gauss);
        free(noise_in); free(noise_out);
        return N4M_ERR_OUT_OF_MEMORY;
    }

    const double noise_std = 0.005 * state->size_effect_strength;
    /* For each sample, compute the baseline, apply path length, generate
     * noise (consuming the rng), smooth and add. */
    for (int64_t i = 0; i < rows; ++i) {
        const double size_ratio = clip(sizes[i] / state->reference_size_um,
                                       0.1, 10.0);
        const double size_factor = pow(size_ratio, -0.5);
        const double s0 = state->size_effect_strength * (size_factor - 1.0);

        /* baseline[j] = s0 * wl_factor[j], centered. */
        double sum = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            sum += s0 * wl_factor[j];
        }
        const double mean = sum / (double)cols;

        const double* row_in  = X + (size_t)i * (size_t)cols;
        double*       row_out = out + (size_t)i * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            row_out[j] = row_in[j] + (s0 * wl_factor[j] - mean);
        }

        /* Path length factor. */
        if (state->include_path_length) {
            double pf = 1.0 + state->path_length_sensitivity
                              * log(sizes[i] / state->reference_size_um);
            pf = clip(pf, 0.7, 1.5);
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] *= pf;
            }
        }

        /* Generate per-sample noise vector, smooth, then add. nirs4all
         * uses np.random.default_rng(random_state) created *inside*
         * transform — so each apply() call re-seeds and produces the
         * same sequence. We follow the rng handle stream instead, which
         * is the n4m contract. */
        n4m_aug_rng_normal_fill(rng, 0.0, noise_std,
                                noise_in, (size_t)cols);
        n4m_pp_gaussian_state_apply(gauss, noise_in, 1, cols, noise_out);
        for (int64_t j = 0; j < cols; ++j) {
            row_out[j] += noise_out[j];
        }
    }

    n4m_pp_gaussian_state_free(gauss);
    free(noise_in);
    free(noise_out);
    free(sizes);
    free(wl_factor);
    return N4M_OK;
}
