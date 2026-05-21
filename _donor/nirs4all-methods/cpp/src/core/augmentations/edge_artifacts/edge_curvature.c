/* SPDX-License-Identifier: CECILL-2.1 */
#include "edge_curvature.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/augmentations/spectral/spectral_common.h"
#include "../../common/rng_pcg64.h"

struct n4m_aug_edge_curvature_state_t {
    double  curvature_strength;
    int32_t curvature_type;
    double  asymmetry;
    double  edge_focus;
};

n4m_aug_edge_curvature_state_t* n4m_aug_edge_curvature_state_new(
    double curvature_strength, int32_t curvature_type,
    double asymmetry, double edge_focus) {
    if (curvature_type < 0 || curvature_type > 3) return NULL;
    if (edge_focus < 0.0 || edge_focus > 1.0) return NULL;
    if (asymmetry < -1.0 || asymmetry > 1.0) return NULL;
    n4m_aug_edge_curvature_state_t* s = (n4m_aug_edge_curvature_state_t*)
        calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->curvature_strength = curvature_strength;
    s->curvature_type     = curvature_type;
    s->asymmetry          = asymmetry;
    s->edge_focus         = edge_focus;
    return s;
}

void n4m_aug_edge_curvature_state_free(
    n4m_aug_edge_curvature_state_t* state) {
    free(state);
}

static double clamp_d(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

n4m_status_t n4m_aug_edge_curvature_state_apply(
    const n4m_aug_edge_curvature_state_t* state,
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
    n4m_rng_pcg64* rng = (n4m_rng_pcg64*)rng_void;
    /* Normalised wavelength axis. */
    double* wl_norm = (double*)malloc((size_t)cols * sizeof(double));
    if (wl_norm == NULL) return N4M_ERR_OUT_OF_MEMORY;
    double wl_min = wavelengths[0], wl_max = wavelengths[0];
    for (int64_t j = 1; j < cols; ++j) {
        if (wavelengths[j] < wl_min) wl_min = wavelengths[j];
        if (wavelengths[j] > wl_max) wl_max = wavelengths[j];
    }
    const double wl_span = wl_max - wl_min;
    if (wl_span <= 0.0) {
        for (int64_t j = 0; j < cols; ++j) wl_norm[j] = 0.5;
    } else {
        for (int64_t j = 0; j < cols; ++j) {
            wl_norm[j] = (wavelengths[j] - wl_min) / wl_span;
        }
    }
    const double focus_power = 2.0 + 2.0 * state->edge_focus;
    double* curv = (double*)malloc((size_t)cols * sizeof(double));
    if (curv == NULL) { free(wl_norm); return N4M_ERR_OUT_OF_MEMORY; }
    n4m_aug_randint_state_t randint_state;
    n4m_aug_randint_state_reset(&randint_state, rng);
    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        /* Determine ctype for this sample. */
        int32_t ctype = state->curvature_type;
        if (ctype == N4M_AUG_EDGE_CURVATURE_RANDOM) {
            ctype = (int32_t)n4m_aug_randint(&randint_state, 0, 3) + 1;
        }
        if (ctype == N4M_AUG_EDGE_CURVATURE_SMILE) {
            for (int64_t j = 0; j < cols; ++j) {
                const double xc = wl_norm[j] - 0.5;
                curv[j] = pow(fabs(xc), focus_power) * 2.0 *
                          state->curvature_strength;
            }
        } else if (ctype == N4M_AUG_EDGE_CURVATURE_FROWN) {
            for (int64_t j = 0; j < cols; ++j) {
                const double xc = wl_norm[j] - 0.5;
                curv[j] = -pow(fabs(xc), focus_power) * 2.0 *
                           state->curvature_strength;
            }
        } else {
            /* Asymmetric: jitter asymmetry by U(-0.2, 0.2). */
            const double jitter = n4m_aug_uniform(rng, -0.2, 0.2);
            double asym = state->asymmetry + jitter;
            asym = clamp_d(asym, -1.0, 1.0);
            const double left  = 1.0 + asym;
            const double right = 1.0 - asym;
            for (int64_t j = 0; j < cols; ++j) {
                const double xc = wl_norm[j] - 0.5;
                if (wl_norm[j] < 0.5) {
                    curv[j] = pow(fabs(xc), focus_power) * left;
                } else {
                    curv[j] = pow(fabs(xc), focus_power) * right;
                }
            }
            const int64_t sign_idx = n4m_aug_randint(&randint_state, 0, 2);
            const double sign = (sign_idx == 0) ? -1.0 : 1.0;
            for (int64_t j = 0; j < cols; ++j) {
                curv[j] = sign * curv[j] * state->curvature_strength;
            }
        }
        /* Small random variation multiplier. */
        double gauss;
        n4m_pcg64_engine_standard_normal_fill(rng, &gauss, 1);
        const double scale = 1.0 + 0.1 * gauss;
        for (int64_t j = 0; j < cols; ++j) {
            orow[j] = xrow[j] + curv[j] * scale;
        }
    }
    free(wl_norm); free(curv);
    return N4M_OK;
}
