/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * LocalClipping — implementation.
 *
 * Stream order: centers (n_samples * n_regions) then widths
 * (n_samples * n_regions). The Python reference draws both with
 * `size=(n_samples, n_regions)`.
 *
 * Percentile interpolation matches numpy 1.26.4's `method='linear'`:
 *
 *   frac_idx = (pct / 100) * (n - 1)
 *   lo       = floor(frac_idx)
 *   hi       = ceil(frac_idx)
 *   result   = sorted[lo] + (frac_idx - lo) * (sorted[hi] - sorted[lo])
 */

#include "local_clip.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/augmentations/spectral/spectral_common.h"

struct c4a_aug_local_clip_state_t {
    c4a_rng_pcg64* rng;
    int32_t        n_regions;
    int32_t        width_lo;
    int32_t        width_hi;
};

c4a_aug_local_clip_state_t* c4a_aug_local_clip_state_new(
    c4a_rng_pcg64* rng,
    int32_t n_regions,
    int32_t width_lo, int32_t width_hi) {
    if (rng == NULL) return NULL;
    if (n_regions < 0) return NULL;
    if (width_hi <= width_lo || width_lo < 0) return NULL;
    c4a_aug_local_clip_state_t* s =
        (c4a_aug_local_clip_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->rng       = rng;
    s->n_regions = n_regions;
    s->width_lo  = width_lo;
    s->width_hi  = width_hi;
    return s;
}

void c4a_aug_local_clip_state_free(c4a_aug_local_clip_state_t* state) {
    free(state);
}

/* qsort comparator for ascending doubles. */
static int compare_double_asc(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

static double percentile_linear(const double* sorted, int64_t n, double pct) {
    if (n <= 1) return sorted[0];
    const double frac_idx = (pct / 100.0) * (double)(n - 1);
    int64_t lo = (int64_t)floor(frac_idx);
    int64_t hi = (int64_t)ceil(frac_idx);
    if (lo < 0) lo = 0;
    if (hi >= n) hi = n - 1;
    if (lo == hi) return sorted[lo];
    const double w = frac_idx - (double)lo;
    return sorted[lo] + w * (sorted[hi] - sorted[lo]);
}

c4a_status_t c4a_aug_local_clip_state_apply(
    c4a_aug_local_clip_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) return C4A_ERR_INVALID_ARGUMENT;
    if (X != out) {
        memcpy(out, X, (size_t)rows * (size_t)cols * sizeof(double));
    }
    if (rows == 0 || cols == 0 || state->n_regions == 0) return C4A_OK;

    const int32_t nr = state->n_regions;
    const size_t total = (size_t)rows * (size_t)nr;
    int64_t* centers = (int64_t*)malloc(total * sizeof(int64_t));
    int64_t* widths  = (int64_t*)malloc(total * sizeof(int64_t));
    double*  scratch = (double*)malloc((size_t)cols * sizeof(double));
    if (centers == NULL || widths == NULL || scratch == NULL) {
        free(centers); free(widths); free(scratch);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    c4a_aug_randint_state_t rs;
    c4a_aug_randint_state_reset(&rs, state->rng);
    for (size_t k = 0; k < total; ++k) {
        centers[k] = c4a_aug_randint(&rs, 0, cols);
    }
    c4a_aug_randint_state_reset(&rs, state->rng);
    for (size_t k = 0; k < total; ++k) {
        widths[k] = c4a_aug_randint(&rs, state->width_lo, state->width_hi);
    }
    for (int64_t i = 0; i < rows; ++i) {
        double* row = out + (size_t)i * (size_t)cols;
        for (int32_t r = 0; r < nr; ++r) {
            const size_t idx = (size_t)i * (size_t)nr + (size_t)r;
            const int64_t center = centers[idx];
            const int64_t width  = widths[idx];
            int64_t start = center - width / 2;
            int64_t end   = center + width / 2;
            if (start < 0)   start = 0;
            if (end   > cols) end  = cols;
            if (start >= end) continue;
            const int64_t n_segment = end - start;
            memcpy(scratch, row + start, (size_t)n_segment * sizeof(double));
            qsort(scratch, (size_t)n_segment, sizeof(double),
                  compare_double_asc);
            const double limit = percentile_linear(scratch, n_segment, 90.0);
            for (int64_t j = start; j < end; ++j) {
                if (row[j] > limit) row[j] = limit;
            }
        }
    }
    free(centers);
    free(widths);
    free(scratch);
    return C4A_OK;
}
