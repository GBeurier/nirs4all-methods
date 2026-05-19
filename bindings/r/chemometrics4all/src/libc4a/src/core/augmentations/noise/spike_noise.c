/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SpikeNoise — reference implementation.
 *
 * Draw order matches the docstring in `spike_noise.h`:
 *   1. `rows` count draws (per-sample n_spikes pick)
 *   2. `rows * n_spikes_max` index draws (candidate spike positions)
 *   3. `rows * n_spikes_max` amplitude draws (spike heights)
 *
 * Each draw uses `c4a_pcg64_engine_next_double` (uniform in [0, 1)) — so the
 * PCG64 stream is consumed in a deterministic, NumPy-comparable order.
 */

#include "spike_noise.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct c4a_aug_spike_noise_state_t {
    c4a_rng_pcg64* rng;   /* borrowed; not owned. */
    int32_t n_spikes_min;
    int32_t n_spikes_max;
    double  amplitude_min;
    double  amplitude_max;
};

static int cmp_int64(const void* a, const void* b) {
    const int64_t va = *(const int64_t*)a;
    const int64_t vb = *(const int64_t*)b;
    if (va < vb) return -1;
    if (va > vb) return  1;
    return 0;
}

c4a_aug_spike_noise_state_t* c4a_aug_spike_noise_state_new(
    c4a_rng_pcg64* rng,
    int32_t n_spikes_min,
    int32_t n_spikes_max,
    double  amplitude_min,
    double  amplitude_max) {
    c4a_aug_spike_noise_state_t* s =
        (c4a_aug_spike_noise_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng           = rng;
    s->n_spikes_min  = n_spikes_min;
    s->n_spikes_max  = n_spikes_max;
    s->amplitude_min = amplitude_min;
    s->amplitude_max = amplitude_max;
    return s;
}

void c4a_aug_spike_noise_state_free(c4a_aug_spike_noise_state_t* state) {
    free(state);
}

c4a_status_t c4a_aug_spike_noise_state_apply(
    c4a_aug_spike_noise_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out) {
    if (state == NULL || state->rng == NULL) return C4A_ERR_NULL_POINTER;
    if (X == NULL || out == NULL)             return C4A_ERR_NULL_POINTER;
    if (rows < 0 || cols < 0)                 return C4A_ERR_INVALID_ARGUMENT;
    if (state->n_spikes_min < 0 ||
        state->n_spikes_max < state->n_spikes_min) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (state->amplitude_max < state->amplitude_min) {
        return C4A_ERR_INVALID_ARGUMENT;
    }

    /* Always copy X -> out first (we add spikes in place to `out`). */
    if (rows == 0 || cols == 0) return C4A_OK;
    const size_t n_total = (size_t)rows * (size_t)cols;
    memcpy(out, X, n_total * sizeof(double));

    const int32_t n_min = state->n_spikes_min;
    const int32_t n_max = state->n_spikes_max;
    const double  amp_lo = state->amplitude_min;
    const double  amp_hi = state->amplitude_max;
    const double  amp_span = amp_hi - amp_lo;
    const int32_t n_choice = n_max - n_min + 1;

    /* 1. Per-row spike count. */
    int32_t* n_spikes_per_sample =
        (int32_t*)malloc((size_t)rows * sizeof(int32_t));
    if (n_spikes_per_sample == NULL) return C4A_ERR_OUT_OF_MEMORY;
    for (int64_t i = 0; i < rows; ++i) {
        const double u = c4a_pcg64_engine_next_double(state->rng);
        int32_t k = n_min + (int32_t)floor(u * (double)n_choice);
        if (k > n_max) k = n_max;            /* defence vs. floating edges */
        if (k < n_min) k = n_min;
        n_spikes_per_sample[(size_t)i] = k;
    }

    /* 2. Candidate indices (rows * n_max draws). */
    int64_t* all_indices = NULL;
    if (n_max > 0) {
        all_indices = (int64_t*)malloc(
            (size_t)rows * (size_t)n_max * sizeof(int64_t));
        if (all_indices == NULL) {
            free(n_spikes_per_sample);
            return C4A_ERR_OUT_OF_MEMORY;
        }
    }
    for (int64_t i = 0; i < rows; ++i) {
        for (int32_t k = 0; k < n_max; ++k) {
            const double u = c4a_pcg64_engine_next_double(state->rng);
            int64_t idx = (int64_t)floor(u * (double)cols);
            if (idx >= cols) idx = cols - 1;  /* defence vs. u == nextafter(1.0) */
            if (idx < 0)     idx = 0;
            all_indices[(size_t)i * (size_t)n_max + (size_t)k] = idx;
        }
    }

    /* 3. Amplitudes (rows * n_max draws). */
    double* all_amps = NULL;
    if (n_max > 0) {
        all_amps = (double*)malloc(
            (size_t)rows * (size_t)n_max * sizeof(double));
        if (all_amps == NULL) {
            free(all_indices);
            free(n_spikes_per_sample);
            return C4A_ERR_OUT_OF_MEMORY;
        }
    }
    for (int64_t i = 0; i < rows; ++i) {
        for (int32_t k = 0; k < n_max; ++k) {
            const double u = c4a_pcg64_engine_next_double(state->rng);
            all_amps[(size_t)i * (size_t)n_max + (size_t)k] =
                amp_lo + amp_span * u;
        }
    }

    /* 4. Apply: per-row sort+unique then add amplitudes[0..len_uniq). */
    int64_t* tmp_idx = NULL;
    if (n_max > 0) {
        tmp_idx = (int64_t*)malloc((size_t)n_max * sizeof(int64_t));
        if (tmp_idx == NULL) {
            free(all_amps);
            free(all_indices);
            free(n_spikes_per_sample);
            return C4A_ERR_OUT_OF_MEMORY;
        }
    }
    for (int64_t i = 0; i < rows; ++i) {
        const int32_t n_this = n_spikes_per_sample[(size_t)i];
        if (n_this <= 0) continue;
        /* Copy + sort. */
        memcpy(tmp_idx,
               all_indices + (size_t)i * (size_t)n_max,
               (size_t)n_this * sizeof(int64_t));
        qsort(tmp_idx, (size_t)n_this, sizeof(int64_t), cmp_int64);
        /* Dedupe (sorted -> linear sweep). */
        size_t len_uniq = 0;
        for (int32_t k = 0; k < n_this; ++k) {
            if (k == 0 || tmp_idx[(size_t)k] != tmp_idx[len_uniq - 1]) {
                tmp_idx[len_uniq++] = tmp_idx[(size_t)k];
            }
        }
        /* Add amplitudes[0..len_uniq) at unique sorted positions. */
        const double* amps_row = all_amps + (size_t)i * (size_t)n_max;
        for (size_t k = 0; k < len_uniq; ++k) {
            const int64_t pos = tmp_idx[k];
            out[(size_t)i * (size_t)cols + (size_t)pos] += amps_row[k];
        }
    }

    free(tmp_idx);
    free(all_amps);
    free(all_indices);
    free(n_spikes_per_sample);
    return C4A_OK;
}
