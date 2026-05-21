/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * BinnedStratifiedGroupKFold splitter — bin y, assign each group a bin
 * label, then K-fold stratified across groups.
 *
 * The algorithm is the simplified equivalent of sklearn's
 * StratifiedGroupKFold composed with KBinsDiscretizer. The Python
 * fixture generator uses the same algorithm so the integer index outputs
 * are byte-equal across the C and Python implementations.
 */

#include "binned_strat_group_kfold.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

n4m_split_bsgk_state_t* n4m_split_bsgk_state_new(int32_t n_splits,
                                                   int32_t n_bins,
                                                   int32_t strategy,
                                                   int32_t shuffle,
                                                   uint64_t seed) {
    if (n_splits < 2 || n_bins < 2) return NULL;
    if (strategy != N4M_SPLIT_KBINS_UNIFORM &&
        strategy != N4M_SPLIT_KBINS_QUANTILE) return NULL;
    n4m_split_bsgk_state_t* s =
        (n4m_split_bsgk_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->n_splits = n_splits;
    s->n_bins   = n_bins;
    s->strategy = strategy;
    s->shuffle  = shuffle;
    s->seed     = seed;
    return s;
}

void n4m_split_bsgk_state_free(n4m_split_bsgk_state_t* state) {
    free(state);
}

/* Bin assignment (uniform / quantile). Shared with KBinsStratifiedSplitter
 * but inlined locally so the two engines stay decoupled. */
static int cmp_idx_by_val_asc;
static const double* g_y_ref;
static int cmp_idx_by_y_asc(const void* a, const void* b) {
    (void)cmp_idx_by_val_asc;
    const int64_t ia = *(const int64_t*)a;
    const int64_t ib = *(const int64_t*)b;
    if (g_y_ref[ia] < g_y_ref[ib]) return -1;
    if (g_y_ref[ia] > g_y_ref[ib]) return 1;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

static void bin_y_into(int32_t n_bins, int32_t strategy,
                       const double* y, int64_t rows,
                       int32_t* bin_of_sample) {
    if (strategy == N4M_SPLIT_KBINS_UNIFORM) {
        double ymin = y[0], ymax = y[0];
        for (int64_t i = 1; i < rows; ++i) {
            if (y[i] < ymin) ymin = y[i];
            if (y[i] > ymax) ymax = y[i];
        }
        const double span = ymax - ymin;
        if (span <= 0.0) {
            for (int64_t i = 0; i < rows; ++i) bin_of_sample[i] = 0;
            return;
        }
        const double inv_w = (double)n_bins / span;
        for (int64_t i = 0; i < rows; ++i) {
            int32_t b = (int32_t)floor((y[i] - ymin) * inv_w);
            if (b < 0) b = 0;
            if (b >= n_bins) b = n_bins - 1;
            bin_of_sample[i] = b;
        }
        return;
    }
    /* Quantile. */
    int64_t* order = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    if (order == NULL) {
        for (int64_t i = 0; i < rows; ++i) bin_of_sample[i] = 0;
        return;
    }
    for (int64_t i = 0; i < rows; ++i) order[i] = i;
    g_y_ref = y;
    qsort(order, (size_t)rows, sizeof(int64_t), cmp_idx_by_y_asc);
    for (int64_t pos = 0; pos < rows; ++pos) {
        int32_t b = (int32_t)((pos * (int64_t)n_bins) / rows);
        if (b >= n_bins) b = n_bins - 1;
        bin_of_sample[order[pos]] = b;
    }
    free(order);
}

/* Fisher-Yates on int64_t. Same draws as kbins_stratified. */
static void fisher_yates(int64_t* arr, int64_t n, n4m_rng_pcg64* rng) {
    for (int64_t i = n - 1; i > 0; --i) {
        const double   u = n4m_split_rng_uniform01(rng);
        const uint64_t b = (uint64_t)(i + 1);
        int64_t        j = (int64_t)floor(u * (double)b);
        if (j > i) j = i;
        if (j < 0) j = 0;
        const int64_t tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

n4m_status_t n4m_split_bsgk_assign(const n4m_split_bsgk_state_t* state,
                                    const double* y, int64_t rows,
                                    const int64_t* groups,
                                    int32_t* fold_assignment) {
    if (state == NULL || y == NULL || groups == NULL ||
        fold_assignment == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 2) return N4M_ERR_INVALID_ARGUMENT;
    const int32_t K = state->n_splits;
    const int32_t B = state->n_bins;

    /* Enumerate unique groups in first-seen order. */
    int64_t* sample_to_group = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    int64_t* unique_groups   = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    int32_t* bin_of_sample   = (int32_t*)malloc((size_t)rows * sizeof(int32_t));
    if (sample_to_group == NULL || unique_groups == NULL ||
        bin_of_sample == NULL) {
        free(sample_to_group); free(unique_groups); free(bin_of_sample);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    int64_t n_groups = 0;
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t g = groups[i];
        int64_t pos = -1;
        for (int64_t k = 0; k < n_groups; ++k) {
            if (unique_groups[k] == g) { pos = k; break; }
        }
        if (pos < 0) { pos = n_groups++; unique_groups[pos] = g; }
        sample_to_group[i] = pos;
    }
    if (n_groups < K) {
        free(sample_to_group); free(unique_groups); free(bin_of_sample);
        return N4M_ERR_INVALID_ARGUMENT;
    }

    bin_y_into(B, state->strategy, y, rows, bin_of_sample);

    /* Group bin label = bin of the group's first sample (deterministic). */
    int32_t* group_bin = (int32_t*)malloc((size_t)n_groups * sizeof(int32_t));
    char*    seen = (char*)calloc((size_t)n_groups, sizeof(char));
    if (group_bin == NULL || seen == NULL) {
        free(sample_to_group); free(unique_groups); free(bin_of_sample);
        free(group_bin); free(seen);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t g = sample_to_group[i];
        if (!seen[g]) {
            group_bin[g] = bin_of_sample[i];
            seen[g] = 1;
        }
    }
    free(seen);

    /* Build per-bin lists of groups in first-seen order. */
    int64_t* bin_offset = (int64_t*)calloc((size_t)(B + 1), sizeof(int64_t));
    int64_t* bin_groups = (int64_t*)malloc((size_t)n_groups * sizeof(int64_t));
    if (bin_offset == NULL || bin_groups == NULL) {
        free(sample_to_group); free(unique_groups); free(bin_of_sample);
        free(group_bin); free(bin_offset); free(bin_groups);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t g = 0; g < n_groups; ++g) ++bin_offset[group_bin[g] + 1];
    for (int32_t b = 0; b < B; ++b) bin_offset[b + 1] += bin_offset[b];
    int64_t* fill = (int64_t*)calloc((size_t)B, sizeof(int64_t));
    if (fill == NULL) {
        free(sample_to_group); free(unique_groups); free(bin_of_sample);
        free(group_bin); free(bin_offset); free(bin_groups);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t g = 0; g < n_groups; ++g) {
        const int32_t b = group_bin[g];
        bin_groups[bin_offset[b] + fill[b]++] = g;
    }
    free(fill);

    /* Optionally shuffle groups within each bin. */
    if (state->shuffle) {
        n4m_rng_pcg64 rng;
        n4m_pcg64_engine_seed(&rng, state->seed);
        for (int32_t b = 0; b < B; ++b) {
            const int64_t lo = bin_offset[b];
            const int64_t hi = bin_offset[b + 1];
            const int64_t bn = hi - lo;
            if (bn > 1) fisher_yates(bin_groups + lo, bn, &rng);
        }
    }

    /* Round-robin assign groups to folds within each bin stratum. */
    int32_t* group_fold = (int32_t*)malloc((size_t)n_groups * sizeof(int32_t));
    if (group_fold == NULL) {
        free(sample_to_group); free(unique_groups); free(bin_of_sample);
        free(group_bin); free(bin_offset); free(bin_groups);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int32_t b = 0; b < B; ++b) {
        const int64_t lo = bin_offset[b];
        const int64_t hi = bin_offset[b + 1];
        for (int64_t p = lo; p < hi; ++p) {
            const int64_t g = bin_groups[p];
            group_fold[g] = (int32_t)((p - lo) % K);
        }
    }

    /* Expand per-sample. */
    for (int64_t i = 0; i < rows; ++i) {
        fold_assignment[i] = group_fold[sample_to_group[i]];
    }

    free(sample_to_group); free(unique_groups); free(bin_of_sample);
    free(group_bin); free(bin_offset); free(bin_groups); free(group_fold);
    return N4M_OK;
}

n4m_status_t n4m_split_bsgk_extract(const int32_t* fold_assignment,
                                     int64_t rows, int32_t fold_idx,
                                     n4m_split_result_t* out) {
    if (fold_assignment == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 1 || fold_idx < 0) return N4M_ERR_INVALID_ARGUMENT;
    int64_t n_test = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (fold_assignment[i] == fold_idx) ++n_test;
    }
    int64_t n_train = rows - n_test;
    if (n_train < 1 || n_test < 1) return N4M_ERR_INVALID_ARGUMENT;
    n4m_status_t st = n4m_split_result_alloc(out, n_train, n_test);
    if (st != N4M_OK) return st;
    int64_t ti = 0, te = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (fold_assignment[i] == fold_idx) out->test_idx[te++]  = i;
        else                                 out->train_idx[ti++] = i;
    }
    return N4M_OK;
}
