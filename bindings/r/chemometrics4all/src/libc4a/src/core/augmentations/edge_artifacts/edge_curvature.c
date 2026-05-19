/* SPDX-License-Identifier: CECILL-2.1 */
#include "edge_curvature.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/rng_pcg64.h"

struct c4a_aug_edge_curvature_state_t {
    double  curvature_strength;
    int32_t curvature_type;
    double  asymmetry;
    double  edge_focus;
};

c4a_aug_edge_curvature_state_t* c4a_aug_edge_curvature_state_new(
    double curvature_strength, int32_t curvature_type,
    double asymmetry, double edge_focus) {
    if (curvature_type < 0 || curvature_type > 3) return NULL;
    if (edge_focus < 0.0 || edge_focus > 1.0) return NULL;
    if (asymmetry < -1.0 || asymmetry > 1.0) return NULL;
    c4a_aug_edge_curvature_state_t* s = (c4a_aug_edge_curvature_state_t*)
        calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->curvature_strength = curvature_strength;
    s->curvature_type     = curvature_type;
    s->asymmetry          = asymmetry;
    s->edge_focus         = edge_focus;
    return s;
}

void c4a_aug_edge_curvature_state_free(
    c4a_aug_edge_curvature_state_t* state) {
    free(state);
}

static double clamp_d(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

c4a_status_t c4a_aug_edge_curvature_state_apply(
    const c4a_aug_edge_curvature_state_t* state,
    void* rng_void,
    const double* X, int64_t rows, int64_t cols,
    const double* wavelengths,
    double* out) {
    if (state == NULL || X == NULL || wavelengths == NULL || out == NULL ||
        rng_void == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (rows == 0 || cols == 0) {
        if (out != X) memcpy(out, X, (size_t)(rows * cols) * sizeof(double));
        return C4A_OK;
    }
    c4a_rng_pcg64* rng = (c4a_rng_pcg64*)rng_void;
    /* Normalised wavelength axis. */
    double* wl_norm = (double*)malloc((size_t)cols * sizeof(double));
    if (wl_norm == NULL) return C4A_ERR_OUT_OF_MEMORY;
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
    if (curv == NULL) { free(wl_norm); return C4A_ERR_OUT_OF_MEMORY; }
    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        /* Determine ctype for this sample. */
        int32_t ctype = state->curvature_type;
        if (ctype == C4A_AUG_EDGE_CURVATURE_RANDOM) {
            const double u = c4a_pcg64_engine_next_double(rng);
            if (u < 1.0 / 3.0) ctype = C4A_AUG_EDGE_CURVATURE_SMILE;
            else if (u < 2.0 / 3.0) ctype = C4A_AUG_EDGE_CURVATURE_FROWN;
            else ctype = C4A_AUG_EDGE_CURVATURE_ASYMMETRIC;
        }
        if (ctype == C4A_AUG_EDGE_CURVATURE_SMILE) {
            for (int64_t j = 0; j < cols; ++j) {
                const double xc = wl_norm[j] - 0.5;
                curv[j] = pow(fabs(xc), focus_power) * 2.0 *
                          state->curvature_strength;
            }
        } else if (ctype == C4A_AUG_EDGE_CURVATURE_FROWN) {
            for (int64_t j = 0; j < cols; ++j) {
                const double xc = wl_norm[j] - 0.5;
                curv[j] = -pow(fabs(xc), focus_power) * 2.0 *
                           state->curvature_strength;
            }
        } else {
            /* Asymmetric: jitter asymmetry by U(-0.2, 0.2). */
            const double jitter = (c4a_pcg64_engine_next_double(rng) - 0.5)
                                   * 0.4;
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
            const double u2 = c4a_pcg64_engine_next_double(rng);
            const double sign = (u2 < 0.5) ? -1.0 : 1.0;
            for (int64_t j = 0; j < cols; ++j) {
                curv[j] = sign * curv[j] * state->curvature_strength;
            }
        }
        /* Small random variation multiplier. */
        double gauss;
        c4a_pcg64_engine_standard_normal_fill(rng, &gauss, 1);
        const double scale = 1.0 + 0.1 * gauss;
        for (int64_t j = 0; j < cols; ++j) {
            orow[j] = xrow[j] + curv[j] * scale;
        }
    }
    free(wl_norm); free(curv);
    return C4A_OK;
}
