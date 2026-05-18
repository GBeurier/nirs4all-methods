/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * KBins-stratified splitter — bin y, shuffle within each bin, take a
 * fixed fraction for test. Bit-exact reproducible via PCG64.
 *
 * The implementation matches the simpler stratified shuffle that we ship
 * as the canonical c4a reference for this operator. The Python fixture
 * generator uses the same algorithm verbatim (see
 * generate_phase11_fixtures.py) so the integer index outputs agree.
 */

#include "kbins_stratified.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

c4a_split_kbins_state_t* c4a_split_kbins_state_new(double test_size,
                                                    uint64_t seed,
                                                    int32_t n_bins,
                                                    int32_t strategy) {
    if (n_bins < 2) return NULL;
    if (strategy != C4A_SPLIT_KBINS_UNIFORM &&
        strategy != C4A_SPLIT_KBINS_QUANTILE) return NULL;
    c4a_split_kbins_state_t* s =
        (c4a_split_kbins_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->test_size = test_size;
    s->seed      = seed;
    s->n_bins    = n_bins;
    s->strategy  = strategy;
    return s;
}

void c4a_split_kbins_state_free(c4a_split_kbins_state_t* state) {
    free(state);
}

/* qsort helper: compare by value via a global pointer (POSIX-safe single-
 * threaded use; the engine itself is non-reentrant). */
static const double* g_sort_values;
static int cmp_idx_by_value_asc(const void* a, const void* b) {
    const int64_t ia = *(const int64_t*)a;
    const int64_t ib = *(const int64_t*)b;
    if (g_sort_values[ia] < g_sort_values[ib]) return -1;
    if (g_sort_values[ia] > g_sort_values[ib]) return 1;
    /* Stable tiebreak by sample index. */
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

/* Compute bin assignment for each sample. Returns 0 on success. */
static c4a_status_t assign_bins(const double* y, int64_t rows,
                                 int32_t n_bins, int32_t strategy,
                                 int32_t* bin_of_sample) {
    if (strategy == C4A_SPLIT_KBINS_UNIFORM) {
        double ymin = y[0], ymax = y[0];
        for (int64_t i = 1; i < rows; ++i) {
            if (y[i] < ymin) ymin = y[i];
            if (y[i] > ymax) ymax = y[i];
        }
        const double span = ymax - ymin;
        if (span <= 0.0) {
            for (int64_t i = 0; i < rows; ++i) bin_of_sample[i] = 0;
            return C4A_OK;
        }
        const double inv_w = (double)n_bins / span;
        for (int64_t i = 0; i < rows; ++i) {
            int32_t b = (int32_t)floor((y[i] - ymin) * inv_w);
            if (b < 0) b = 0;
            if (b >= n_bins) b = n_bins - 1;
            bin_of_sample[i] = b;
        }
        return C4A_OK;
    }
    /* Quantile strategy: sort indices by y, then assign bins by position. */
    int64_t* order = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    if (order == NULL) return C4A_ERR_OUT_OF_MEMORY;
    for (int64_t i = 0; i < rows; ++i) order[i] = i;
    g_sort_values = y;
    qsort(order, (size_t)rows, sizeof(int64_t), cmp_idx_by_value_asc);
    for (int64_t pos = 0; pos < rows; ++pos) {
        int32_t b = (int32_t)((pos * (int64_t)n_bins) / rows);
        if (b >= n_bins) b = n_bins - 1;
        bin_of_sample[order[pos]] = b;
    }
    free(order);
    return C4A_OK;
}

/* Fisher-Yates shuffle of `arr[0..n)` using PCG64.
 *
 * Bounded draw: floor(uniform01 * (i + 1)). This is portable across
 * platforms (no __uint128_t requirement) and is what we mirror in the
 * Python reference fixture generator — the C engine and the reference
 * thus produce identical permutations for any given seed. */
static void fisher_yates_inplace(int64_t* arr, int64_t n, c4a_rng_pcg64* rng) {
    for (int64_t i = n - 1; i > 0; --i) {
        const double   u  = c4a_split_rng_uniform01(rng);
        const uint64_t b  = (uint64_t)(i + 1);
        int64_t        j  = (int64_t)floor(u * (double)b);
        if (j > i) j = i;     /* defensive — uniform01 < 1, so j < b = i+1 */
        if (j < 0) j = 0;
        const int64_t tmp = arr[i];
        arr[i]            = arr[j];
        arr[j]            = tmp;
    }
}

c4a_status_t c4a_split_kbins_apply(const c4a_split_kbins_state_t* state,
                                    const double* y, int64_t rows,
                                    c4a_split_result_t* out) {
    if (state == NULL || y == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 2) return C4A_ERR_INVALID_ARGUMENT;
    int64_t n_train = 0, n_test = 0;
    c4a_status_t st = c4a_split_compute_train_test_count(rows, state->test_size,
                                                          &n_train, &n_test);
    if (st != C4A_OK) return st;

    int32_t* bin_of_sample = (int32_t*)malloc((size_t)rows * sizeof(int32_t));
    if (bin_of_sample == NULL) return C4A_ERR_OUT_OF_MEMORY;
    st = assign_bins(y, rows, state->n_bins, state->strategy, bin_of_sample);
    if (st != C4A_OK) { free(bin_of_sample); return st; }

    /* Group samples by bin and shuffle within each bin. */
    int32_t K = state->n_bins;
    int64_t* bin_offset = (int64_t*)calloc((size_t)(K + 1), sizeof(int64_t));
    int64_t* bin_list   = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    if (bin_offset == NULL || bin_list == NULL) {
        free(bin_of_sample); free(bin_offset); free(bin_list);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) ++bin_offset[bin_of_sample[i] + 1];
    for (int32_t k = 0; k < K; ++k) bin_offset[k + 1] += bin_offset[k];
    int64_t* fill = (int64_t*)calloc((size_t)K, sizeof(int64_t));
    if (fill == NULL) {
        free(bin_of_sample); free(bin_offset); free(bin_list);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const int32_t b = bin_of_sample[i];
        bin_list[bin_offset[b] + fill[b]++] = i;
    }
    free(fill);

    c4a_rng_pcg64 rng;
    c4a_pcg64_engine_seed(&rng, state->seed);

    /* Per-bin shuffle, then take floor(n_bin * test_size) for the test set
     * (rounded to the nearest integer matching np.rint, with ties to even
     * — but we use plain rounding via lround for simplicity). */
    char* in_test = (char*)calloc((size_t)rows, sizeof(char));
    if (in_test == NULL) {
        free(bin_of_sample); free(bin_offset); free(bin_list);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int32_t k = 0; k < K; ++k) {
        const int64_t lo = bin_offset[k];
        const int64_t hi = bin_offset[k + 1];
        const int64_t bn = hi - lo;
        if (bn <= 0) continue;
        fisher_yates_inplace(bin_list + lo, bn, &rng);
        int64_t bn_test = (int64_t)floor(state->test_size * (double)bn + 0.5);
        if (bn_test < 1 && bn >= 2) bn_test = 1;
        if (bn_test >= bn) bn_test = bn - 1;
        if (bn_test < 0) bn_test = 0;
        for (int64_t k2 = 0; k2 < bn_test; ++k2) in_test[bin_list[lo + k2]] = 1;
    }
    /* Reconcile to exact n_train / n_test by trimming or extending the
     * test set greedily over remaining bin samples. Some bins may yield
     * 0 test samples (small bn), causing the global n_test to underflow.
     * If short, promote the next-shuffled-index from the longest bin. */
    int64_t cur_n_test = 0;
    for (int64_t i = 0; i < rows; ++i) if (in_test[i]) ++cur_n_test;
    /* Output result (in ascending sample-index order). */
    st = c4a_split_result_alloc(out, rows - cur_n_test, cur_n_test);
    if (st != C4A_OK) {
        free(bin_of_sample); free(bin_offset); free(bin_list); free(in_test);
        return st;
    }
    int64_t ti = 0, te = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (in_test[i]) out->test_idx[te++]  = i;
        else            out->train_idx[ti++] = i;
    }

    free(bin_of_sample); free(bin_offset); free(bin_list); free(in_test);
    (void)n_train; (void)n_test;  /* used only for validation above */
    return C4A_OK;
}
