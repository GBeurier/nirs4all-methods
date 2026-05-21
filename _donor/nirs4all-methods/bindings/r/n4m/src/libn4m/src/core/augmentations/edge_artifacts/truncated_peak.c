/* SPDX-License-Identifier: CECILL-2.1 */
#include "truncated_peak.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/rng_pcg64.h"

struct n4m_aug_truncated_peak_state_t {
    double peak_probability;
    double amplitude_min;
    double amplitude_max;
    double width_min;
    double width_max;
    int    left_edge;
    int    right_edge;
};

n4m_aug_truncated_peak_state_t* n4m_aug_truncated_peak_state_new(
    double peak_probability,
    double amplitude_min, double amplitude_max,
    double width_min, double width_max,
    int left_edge, int right_edge) {
    if (peak_probability < 0.0 || peak_probability > 1.0) return NULL;
    if (amplitude_min > amplitude_max) return NULL;
    if (width_min <= 0.0 || width_min > width_max) return NULL;
    n4m_aug_truncated_peak_state_t* s = (n4m_aug_truncated_peak_state_t*)
        calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    s->peak_probability = peak_probability;
    s->amplitude_min    = amplitude_min;
    s->amplitude_max    = amplitude_max;
    s->width_min        = width_min;
    s->width_max        = width_max;
    s->left_edge        = left_edge  ? 1 : 0;
    s->right_edge       = right_edge ? 1 : 0;
    return s;
}

void n4m_aug_truncated_peak_state_free(
    n4m_aug_truncated_peak_state_t* state) {
    free(state);
}

static double uniform(n4m_rng_pcg64* rng, double lo, double hi) {
    return lo + (hi - lo) * n4m_pcg64_engine_next_double(rng);
}

n4m_status_t n4m_aug_truncated_peak_state_apply(
    const n4m_aug_truncated_peak_state_t* state,
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
    double wl_min = wavelengths[0], wl_max = wavelengths[0];
    for (int64_t j = 1; j < cols; ++j) {
        if (wavelengths[j] < wl_min) wl_min = wavelengths[j];
        if (wavelengths[j] > wl_max) wl_max = wavelengths[j];
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* xrow = X + i * cols;
        double* orow = out + i * cols;
        if (orow != xrow) {
            memcpy(orow, xrow, (size_t)cols * sizeof(double));
        }
        if (state->left_edge) {
            const double r = n4m_pcg64_engine_next_double(rng);
            if (r < state->peak_probability) {
                const double amp   = uniform(rng, state->amplitude_min,
                                              state->amplitude_max);
                const double width = uniform(rng, state->width_min,
                                              state->width_max);
                const double off   = width * uniform(rng, 0.5, 1.5);
                const double centre = wl_min - off;
                for (int64_t j = 0; j < cols; ++j) {
                    const double d = (wavelengths[j] - centre) / width;
                    orow[j] += amp * exp(-0.5 * d * d);
                }
            }
        }
        if (state->right_edge) {
            const double r = n4m_pcg64_engine_next_double(rng);
            if (r < state->peak_probability) {
                const double amp   = uniform(rng, state->amplitude_min,
                                              state->amplitude_max);
                const double width = uniform(rng, state->width_min,
                                              state->width_max);
                const double off   = width * uniform(rng, 0.5, 1.5);
                const double centre = wl_max + off;
                for (int64_t j = 0; j < cols; ++j) {
                    const double d = (wavelengths[j] - centre) / width;
                    orow[j] += amp * exp(-0.5 * d * d);
                }
            }
        }
    }
    return N4M_OK;
}
