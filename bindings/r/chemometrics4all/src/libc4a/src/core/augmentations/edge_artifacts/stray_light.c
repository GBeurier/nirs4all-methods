/* SPDX-License-Identifier: CECILL-2.1 */
#include "stray_light.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/rng_pcg64.h"

struct c4a_aug_stray_light_state_t {
    double stray_light_fraction;
    double edge_enhancement;
    double edge_width;
    int    include_peak_truncation;
};

c4a_aug_stray_light_state_t* c4a_aug_stray_light_state_new(
    double stray_light_fraction, double edge_enhancement,
    double edge_width, int include_peak_truncation) {
    if (stray_light_fraction < 0.0) return NULL;
    if (edge_width < 0.0 || edge_width > 0.5) return NULL;
    c4a_aug_stray_light_state_t* s = (c4a_aug_stray_light_state_t*)
        calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->stray_light_fraction    = stray_light_fraction;
    s->edge_enhancement        = edge_enhancement;
    s->edge_width              = edge_width;
    s->include_peak_truncation = include_peak_truncation ? 1 : 0;
    return s;
}

void c4a_aug_stray_light_state_free(c4a_aug_stray_light_state_t* state) {
    free(state);
}

static double clamp_d(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

c4a_status_t c4a_aug_stray_light_state_apply(
    const c4a_aug_stray_light_state_t* state,
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
    /* Compute base stray-light profile. */
    double* profile = (double*)malloc((size_t)cols * sizeof(double));
    if (profile == NULL) return C4A_ERR_OUT_OF_MEMORY;
    for (int64_t j = 0; j < cols; ++j) profile[j] = state->stray_light_fraction;
    const int64_t edge_points = (int64_t)((double)cols * state->edge_width);
    if (edge_points > 0) {
        /* Left edge: x in [0, 5], right edge: x in [5, 0] over edge_points. */
        for (int64_t j = 0; j < edge_points; ++j) {
            const double xv = (edge_points <= 1) ? 0.0 :
                              5.0 * (double)j / (double)(edge_points - 1);
            const double lw = 1.0 + (state->edge_enhancement - 1.0) /
                                      (1.0 + exp(xv - 2.5));
            profile[j] *= lw;
        }
        for (int64_t j = 0; j < edge_points; ++j) {
            const double xv = (edge_points <= 1) ? 5.0 :
                              5.0 - 5.0 * (double)j / (double)(edge_points - 1);
            const double rw = 1.0 + (state->edge_enhancement - 1.0) /
                                      (1.0 + exp(xv - 2.5));
            profile[cols - edge_points + j] *= rw;
        }
    }
    /* Per-sample work. */
    double* noise_buf = (double*)malloc((size_t)cols * sizeof(double));
    double* stray_buf = (double*)malloc((size_t)cols * sizeof(double));
    if (noise_buf == NULL || stray_buf == NULL) {
        free(profile); free(noise_buf); free(stray_buf);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        c4a_pcg64_engine_standard_normal_fill(rng, noise_buf, (size_t)cols);
        for (int64_t j = 0; j < cols; ++j) {
            double s = profile[j] * (1.0 + 0.1 * noise_buf[j]);
            s = clamp_d(s, 0.0, 0.1);
            stray_buf[j] = s;
        }
        for (int64_t j = 0; j < cols; ++j) {
            const double A = xrow[j];
            double T = pow(10.0, -A);
            T = clamp_d(T, 1e-10, 1.0);
            const double s  = stray_buf[j];
            const double To = (T + s) / (1.0 + s);
            double Aout = -log10(To);
            if (state->include_peak_truncation && A > 1.5) {
                const double over = clamp_d(A - 1.5, 0.0, 2.0);
                const double factor = 1.0 - 0.05 * over;
                Aout *= factor;
            }
            orow[j] = Aout;
        }
    }
    free(profile); free(noise_buf); free(stray_buf);
    return C4A_OK;
}
