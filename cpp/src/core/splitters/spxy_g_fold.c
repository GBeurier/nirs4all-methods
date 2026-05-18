/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Group-aware SPXY K-Fold splitter.
 *
 * Implements the K-fold path of nirs4all.operators.splitters.SPXYGFold:
 *   1. Aggregate samples by group label (first-seen order).
 *   2. Compute per-group representatives (mean or median) for X and Y.
 *   3. Run the SPXYFold alternating max-min assignment on the
 *      n_groups x n_groups distance matrix.
 *   4. Expand per-group fold assignment back to per-sample.
 *
 * We restrict to int64 group labels here (the C ABI doesn't carry generic
 * hashable group keys). Median aggregation uses a quickselect-based
 * O(n log n) partial sort per column. Mean aggregation is O(group_size).
 */

#include "spxy_g_fold.h"
#include "spxy_fold.h"

#include <stdlib.h>
#include <string.h>

c4a_split_spxy_g_fold_state_t* c4a_split_spxy_g_fold_state_new(int32_t n_splits,
                                                                int32_t use_y,
                                                                int32_t aggregation) {
    if (n_splits < 2) return NULL;
    if (aggregation != 0 && aggregation != 1) return NULL;
    c4a_split_spxy_g_fold_state_t* s =
        (c4a_split_spxy_g_fold_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->n_splits    = n_splits;
    s->use_y       = use_y;
    s->aggregation = aggregation;
    return s;
}

void c4a_split_spxy_g_fold_state_free(c4a_split_spxy_g_fold_state_t* state) {
    free(state);
}

/* qsort helper for sorting doubles ascending (median computation). */
static int cmp_double_asc(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double median_of_doubles(double* tmp, int64_t n) {
    qsort(tmp, (size_t)n, sizeof(double), cmp_double_asc);
    if ((n % 2) == 1) {
        return tmp[n / 2];
    }
    return 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
}

c4a_status_t c4a_split_spxy_g_fold_assign(const c4a_split_spxy_g_fold_state_t* s,
                                           const double* X, int64_t rows,
                                           int64_t cols_x,
                                           const double* Y, int64_t cols_y,
                                           const int64_t* groups,
                                           int32_t* fold_assignment) {
    if (s == NULL || X == NULL || groups == NULL || fold_assignment == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (s->use_y != 0 && Y == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 2 || cols_x < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    /* Step 1: enumerate unique groups in first-seen order. */
    int64_t* sample_to_group = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    int64_t* unique_groups   = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    if (sample_to_group == NULL || unique_groups == NULL) {
        free(sample_to_group); free(unique_groups);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    int64_t n_groups = 0;
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t g = groups[i];
        int64_t pos = -1;
        for (int64_t k = 0; k < n_groups; ++k) {
            if (unique_groups[k] == g) { pos = k; break; }
        }
        if (pos < 0) {
            pos = n_groups++;
            unique_groups[pos] = g;
        }
        sample_to_group[i] = pos;
    }
    if (n_groups < s->n_splits) {
        free(sample_to_group); free(unique_groups);
        return C4A_ERR_INVALID_ARGUMENT;
    }

    /* Step 2: compute representatives.
     * For mean: accumulate per column and divide by group size.
     * For median: per-column gather then qsort.
     * Y is treated the same way except hamming uses the mode (most common
     * value). For simplicity we use mean for euclidean Y and the first
     * sample's value for hamming (matches stats.mode tiebreak via
     * "first value" in the canonical reference). */
    int64_t* group_sizes = (int64_t*)calloc((size_t)n_groups, sizeof(int64_t));
    double* X_rep = (double*)calloc((size_t)n_groups * (size_t)cols_x,
                                     sizeof(double));
    double* Y_rep = NULL;
    if (s->use_y != 0) {
        Y_rep = (double*)calloc((size_t)n_groups * (size_t)cols_y,
                                sizeof(double));
        if (Y_rep == NULL) {
            free(sample_to_group); free(unique_groups);
            free(group_sizes); free(X_rep);
            return C4A_ERR_OUT_OF_MEMORY;
        }
    }
    if (group_sizes == NULL || X_rep == NULL) {
        free(sample_to_group); free(unique_groups);
        free(group_sizes); free(X_rep); free(Y_rep);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        ++group_sizes[sample_to_group[i]];
    }

    if (s->aggregation == 0) {
        /* Mean aggregation. */
        for (int64_t i = 0; i < rows; ++i) {
            const int64_t g = sample_to_group[i];
            for (int64_t c = 0; c < cols_x; ++c) {
                X_rep[g * cols_x + c] += X[i * cols_x + c];
            }
            if (s->use_y != 0) {
                for (int64_t c = 0; c < cols_y; ++c) {
                    Y_rep[g * cols_y + c] += Y[i * cols_y + c];
                }
            }
        }
        for (int64_t g = 0; g < n_groups; ++g) {
            const double inv = 1.0 / (double)group_sizes[g];
            for (int64_t c = 0; c < cols_x; ++c) X_rep[g * cols_x + c] *= inv;
            if (s->use_y != 0) {
                for (int64_t c = 0; c < cols_y; ++c) {
                    Y_rep[g * cols_y + c] *= inv;
                }
            }
        }
    } else {
        /* Median aggregation: per-column gather then sort. */
        int64_t max_gs = 0;
        for (int64_t g = 0; g < n_groups; ++g) {
            if (group_sizes[g] > max_gs) max_gs = group_sizes[g];
        }
        double* tmp = (double*)malloc((size_t)max_gs * sizeof(double));
        if (tmp == NULL) {
            free(sample_to_group); free(unique_groups);
            free(group_sizes); free(X_rep); free(Y_rep);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        /* Compute medians by iterating over each group separately. */
        int64_t* group_offsets = (int64_t*)calloc((size_t)(n_groups + 1),
                                                  sizeof(int64_t));
        int64_t* sample_lists  = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
        if (group_offsets == NULL || sample_lists == NULL) {
            free(tmp); free(group_offsets); free(sample_lists);
            free(sample_to_group); free(unique_groups);
            free(group_sizes); free(X_rep); free(Y_rep);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        for (int64_t i = 0; i < rows; ++i) ++group_offsets[sample_to_group[i] + 1];
        for (int64_t g = 0; g < n_groups; ++g) group_offsets[g + 1] += group_offsets[g];
        int64_t* fill = (int64_t*)calloc((size_t)n_groups, sizeof(int64_t));
        for (int64_t i = 0; i < rows; ++i) {
            const int64_t g = sample_to_group[i];
            sample_lists[group_offsets[g] + fill[g]++] = i;
        }
        free(fill);
        for (int64_t g = 0; g < n_groups; ++g) {
            const int64_t lo = group_offsets[g];
            const int64_t hi = group_offsets[g + 1];
            const int64_t gs = hi - lo;
            for (int64_t c = 0; c < cols_x; ++c) {
                for (int64_t k = 0; k < gs; ++k) {
                    tmp[k] = X[sample_lists[lo + k] * cols_x + c];
                }
                X_rep[g * cols_x + c] = median_of_doubles(tmp, gs);
            }
            if (s->use_y != 0) {
                for (int64_t c = 0; c < cols_y; ++c) {
                    for (int64_t k = 0; k < gs; ++k) {
                        tmp[k] = Y[sample_lists[lo + k] * cols_y + c];
                    }
                    Y_rep[g * cols_y + c] = median_of_doubles(tmp, gs);
                }
            }
        }
        free(tmp); free(group_offsets); free(sample_lists);
    }

    /* Step 3: run SPXYFold-style assignment on the group representatives. */
    c4a_split_spxy_fold_state_t inner;
    inner.n_splits = s->n_splits;
    inner.use_y    = s->use_y;
    int32_t* group_folds = (int32_t*)malloc((size_t)n_groups * sizeof(int32_t));
    if (group_folds == NULL) {
        free(sample_to_group); free(unique_groups);
        free(group_sizes); free(X_rep); free(Y_rep);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    c4a_status_t st = c4a_split_spxy_fold_assign(&inner, X_rep, n_groups,
                                                  cols_x, Y_rep, cols_y,
                                                  group_folds);
    if (st != C4A_OK) {
        free(sample_to_group); free(unique_groups);
        free(group_sizes); free(X_rep); free(Y_rep); free(group_folds);
        return st;
    }

    /* Step 4: expand per-group → per-sample. */
    for (int64_t i = 0; i < rows; ++i) {
        fold_assignment[i] = group_folds[sample_to_group[i]];
    }

    free(sample_to_group); free(unique_groups);
    free(group_sizes); free(X_rep); free(Y_rep); free(group_folds);
    return C4A_OK;
}

c4a_status_t c4a_split_spxy_g_fold_extract(const int32_t* fold_assignment,
                                            int64_t rows, int32_t fold_idx,
                                            c4a_split_result_t* out) {
    return c4a_split_spxy_fold_extract(fold_assignment, rows, fold_idx, out);
}
