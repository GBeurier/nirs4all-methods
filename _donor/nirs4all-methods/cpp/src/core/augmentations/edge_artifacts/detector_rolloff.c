/* SPDX-License-Identifier: CECILL-2.1 */
#include "detector_rolloff.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/rng_pcg64.h"

struct n4m_aug_detector_rolloff_state_t {
    int32_t detector_model;
    double  effect_strength;
    double  noise_amplification;
    int     include_baseline_distortion;
};

typedef struct {
    double opt_min;
    double opt_max;
    double roll_off_rate;
    double min_sensitivity;
} detector_model_params_t;

static int detector_model_get(int32_t id, detector_model_params_t* out) {
    switch (id) {
        case N4M_AUG_DETECTOR_INGAAS_STANDARD:
            out->opt_min = 1000.0;
            out->opt_max = 1600.0;
            out->roll_off_rate   = 0.008;
            out->min_sensitivity = 0.30;
            return 0;
        case N4M_AUG_DETECTOR_INGAAS_EXTENDED:
            out->opt_min = 1100.0;
            out->opt_max = 2200.0;
            out->roll_off_rate   = 0.005;
            out->min_sensitivity = 0.20;
            return 0;
        case N4M_AUG_DETECTOR_PBS:
            out->opt_min = 1000.0;
            out->opt_max = 2800.0;
            out->roll_off_rate   = 0.004;
            out->min_sensitivity = 0.25;
            return 0;
        case N4M_AUG_DETECTOR_SILICON_CCD:
            out->opt_min = 400.0;
            out->opt_max = 900.0;
            out->roll_off_rate   = 0.015;
            out->min_sensitivity = 0.10;
            return 0;
        case N4M_AUG_DETECTOR_GENERIC_NIR:
            out->opt_min = 900.0;
            out->opt_max = 1700.0;
            out->roll_off_rate   = 0.006;
            out->min_sensitivity = 0.35;
            return 0;
        default:
            return 1;
    }
}

n4m_aug_detector_rolloff_state_t* n4m_aug_detector_rolloff_state_new(
    int32_t detector_model,
    double effect_strength,
    double noise_amplification,
    int include_baseline_distortion) {
    detector_model_params_t tmp;
    if (detector_model_get(detector_model, &tmp) != 0) return NULL;
    n4m_aug_detector_rolloff_state_t* s = (n4m_aug_detector_rolloff_state_t*)
        calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->detector_model              = detector_model;
    s->effect_strength             = effect_strength;
    s->noise_amplification         = noise_amplification;
    s->include_baseline_distortion = include_baseline_distortion ? 1 : 0;
    return s;
}

void n4m_aug_detector_rolloff_state_free(
    n4m_aug_detector_rolloff_state_t* state) {
    free(state);
}

static double clamp_double(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

n4m_status_t n4m_aug_detector_rolloff_state_apply(
    const n4m_aug_detector_rolloff_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out) {
    if (state == NULL || X == NULL || wavelengths == NULL || out == NULL ||
        rng_void == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return N4M_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return N4M_OK;
    }
    detector_model_params_t mp;
    if (detector_model_get(state->detector_model, &mp) != 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    n4m_rng_pcg64* rng = (n4m_rng_pcg64*)rng_void;
    /* Compute sensitivity curve over wavelengths. */
    double* sens = (double*)malloc((size_t)cols * sizeof(double));
    if (sens == NULL) return N4M_ERR_OUT_OF_MEMORY;
    for (int64_t j = 0; j < cols; ++j) {
        double s = 1.0;
        const double wl = wavelengths[j];
        if (wl < mp.opt_min) {
            const double d = mp.opt_min - wl;
            const double decay = exp(-mp.roll_off_rate *
                                      state->effect_strength * d);
            s = mp.min_sensitivity + (1.0 - mp.min_sensitivity) * decay;
        } else if (wl > mp.opt_max) {
            const double d = wl - mp.opt_max;
            const double decay = exp(-mp.roll_off_rate *
                                      state->effect_strength * d);
            s = mp.min_sensitivity + (1.0 - mp.min_sensitivity) * decay;
        }
        sens[j] = s;
    }
    /* Pre-compute baseline distortion term once. */
    double* baseline = NULL;
    if (state->include_baseline_distortion) {
        baseline = (double*)malloc((size_t)cols * sizeof(double));
        if (baseline == NULL) { free(sens); return N4M_ERR_OUT_OF_MEMORY; }
        for (int64_t j = 0; j < cols; ++j) {
            baseline[j] = (1.0 - sens[j]) * 0.01 * state->effect_strength;
        }
    }
    /* For each sample apply: noise then baseline distortion. */
    double* noise_buf = NULL;
    if (state->noise_amplification > 0.0) {
        noise_buf = (double*)malloc((size_t)cols * sizeof(double));
        if (noise_buf == NULL) {
            free(sens); free(baseline);
            return N4M_ERR_OUT_OF_MEMORY;
        }
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        /* Start from input. */
        if (orow != xrow) {
            memcpy(orow, xrow, (size_t)cols * sizeof(double));
        }
        if (state->noise_amplification > 0.0) {
            n4m_pcg64_engine_standard_normal_fill(rng, noise_buf,
                                                    (size_t)cols);
            for (int64_t j = 0; j < cols; ++j) {
                const double s_clip = clamp_double(sens[j], 0.1, 1.0);
                const double factor = (1.0 / s_clip - 1.0) *
                                       state->noise_amplification;
                orow[j] += noise_buf[j] * factor;
            }
        }
        if (state->include_baseline_distortion) {
            for (int64_t j = 0; j < cols; ++j) orow[j] += baseline[j];
        }
    }
    free(sens); free(baseline); free(noise_buf);
    return N4M_OK;
}
