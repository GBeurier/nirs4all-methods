/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * KBins-stratified splitter — bin y, then delegate the split contract to the
 * same deterministic sequence used by sklearn StratifiedShuffleSplit.
 *
 * nirs4all.KBinsStratifiedSplitter is implemented as KBinsDiscretizer followed
 * by sklearn.model_selection.StratifiedShuffleSplit(random_state=seed). For
 * parity, this file mirrors the relevant sklearn path:
 *   - sort class members by bin label using stable mergesort semantics;
 *   - allocate train/test counts with sklearn.utils.extmath._approximate_mode;
 *   - shuffle each class and the final train/test vectors with NumPy legacy
 *     RandomState(MT19937), not n4m's PCG64 engine.
 */

#include "kbins_stratified.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define N4M_MT19937_N 624
#define N4M_MT19937_M 397

typedef struct n4m_kbins_mt19937_t {
    uint32_t state[N4M_MT19937_N];
    int32_t pos;
} n4m_kbins_mt19937_t;

static void mt19937_seed(n4m_kbins_mt19937_t* rng, uint32_t seed) {
    rng->state[0] = seed;
    for (int32_t i = 1; i < N4M_MT19937_N; ++i) {
        const uint32_t prev = rng->state[i - 1];
        rng->state[i] = (uint32_t)(1812433253UL * (prev ^ (prev >> 30)) +
                                   (uint32_t)i);
    }
    rng->pos = N4M_MT19937_N;
}

static void mt19937_twist(n4m_kbins_mt19937_t* rng) {
    static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
    int32_t kk;
    for (kk = 0; kk < N4M_MT19937_N - N4M_MT19937_M; ++kk) {
        const uint32_t y = (rng->state[kk] & 0x80000000U) |
                           (rng->state[kk + 1] & 0x7fffffffU);
        rng->state[kk] = rng->state[kk + N4M_MT19937_M] ^
                         (y >> 1) ^ mag01[y & 0x1U];
    }
    for (; kk < N4M_MT19937_N - 1; ++kk) {
        const uint32_t y = (rng->state[kk] & 0x80000000U) |
                           (rng->state[kk + 1] & 0x7fffffffU);
        rng->state[kk] = rng->state[kk + (N4M_MT19937_M - N4M_MT19937_N)] ^
                         (y >> 1) ^ mag01[y & 0x1U];
    }
    {
        const uint32_t y = (rng->state[N4M_MT19937_N - 1] & 0x80000000U) |
                           (rng->state[0] & 0x7fffffffU);
        rng->state[N4M_MT19937_N - 1] = rng->state[N4M_MT19937_M - 1] ^
                                        (y >> 1) ^ mag01[y & 0x1U];
    }
    rng->pos = 0;
}

static uint32_t mt19937_next_uint32(n4m_kbins_mt19937_t* rng) {
    if (rng->pos >= N4M_MT19937_N) {
        mt19937_twist(rng);
    }
    uint32_t y = rng->state[rng->pos++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

static uint32_t mt19937_interval(n4m_kbins_mt19937_t* rng, uint32_t max) {
    uint32_t mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    uint32_t value;
    do {
        value = mt19937_next_uint32(rng) & mask;
    } while (value > max);
    return value;
}

static void mt19937_shuffle_i64(n4m_kbins_mt19937_t* rng,
                                int64_t* values, int64_t n) {
    for (int64_t i = n - 1; i > 0; --i) {
        const int64_t j = (int64_t)mt19937_interval(rng, (uint32_t)i);
        const int64_t tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
    }
}

static void mt19937_permutation_i64(n4m_kbins_mt19937_t* rng,
                                    int64_t* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = i;
    }
    mt19937_shuffle_i64(rng, out, n);
}

n4m_split_kbins_state_t* n4m_split_kbins_state_new(double test_size,
                                                    uint64_t seed,
                                                    int32_t n_bins,
                                                    int32_t strategy) {
    if (n_bins < 2) return NULL;
    if (strategy != N4M_SPLIT_KBINS_UNIFORM &&
        strategy != N4M_SPLIT_KBINS_QUANTILE) return NULL;
    n4m_split_kbins_state_t* s =
        (n4m_split_kbins_state_t*)malloc(sizeof(*s));
    if (s == NULL) return NULL;
    s->test_size = test_size;
    s->seed      = seed;
    s->n_bins    = n_bins;
    s->strategy  = strategy;
    return s;
}

void n4m_split_kbins_state_free(n4m_split_kbins_state_t* state) {
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

static double percentile_linear_sorted(const double* sorted, int64_t n,
                                       double percentile) {
    if (percentile <= 0.0) return sorted[0];
    if (percentile >= 100.0) return sorted[n - 1];
    const double pos = (percentile / 100.0) * (double)(n - 1);
    const int64_t lo = (int64_t)floor(pos);
    const int64_t hi = lo + 1 < n ? lo + 1 : lo;
    const double frac = pos - (double)lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/* Compute bin assignment for each sample. Returns 0 on success. */
static n4m_status_t assign_bins(const double* y, int64_t rows,
                                 int32_t n_bins, int32_t strategy,
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
            return N4M_OK;
        }
        const double inv_w = (double)n_bins / span;
        for (int64_t i = 0; i < rows; ++i) {
            int32_t b = (int32_t)floor((y[i] - ymin) * inv_w);
            if (b < 0) b = 0;
            if (b >= n_bins) b = n_bins - 1;
            bin_of_sample[i] = b;
        }
        return N4M_OK;
    }
    /* Quantile strategy: mirror sklearn.KBinsDiscretizer(strategy="quantile")
     * for one feature: linear percentiles, drop edges whose width is <= 1e-8,
     * then np.searchsorted(edges[1:-1], y, side="right"). */
    int64_t* order = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    if (order == NULL) return N4M_ERR_OUT_OF_MEMORY;
    for (int64_t i = 0; i < rows; ++i) order[i] = i;
    g_sort_values = y;
    qsort(order, (size_t)rows, sizeof(int64_t), cmp_idx_by_value_asc);

    double* sorted = (double*)malloc((size_t)rows * sizeof(double));
    double* edges = (double*)malloc((size_t)(n_bins + 1) * sizeof(double));
    if (sorted == NULL || edges == NULL) {
        free(order);
        free(sorted);
        free(edges);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        sorted[i] = y[order[i]];
    }
    if (sorted[0] == sorted[rows - 1]) {
        for (int64_t i = 0; i < rows; ++i) bin_of_sample[i] = 0;
        free(order);
        free(sorted);
        free(edges);
        return N4M_OK;
    }

    int32_t edge_count = 0;
    for (int32_t e = 0; e <= n_bins; ++e) {
        const double pct = 100.0 * (double)e / (double)n_bins;
        const double edge = percentile_linear_sorted(sorted, rows, pct);
        if (edge_count == 0 || edge - edges[edge_count - 1] > 1.0e-8) {
            edges[edge_count++] = edge;
        }
    }
    if (edge_count < 2) {
        for (int64_t i = 0; i < rows; ++i) bin_of_sample[i] = 0;
        free(order);
        free(sorted);
        free(edges);
        return N4M_OK;
    }

    const int32_t actual_bins = edge_count - 1;
    for (int64_t i = 0; i < rows; ++i) {
        int32_t b = 0;
        while (b < actual_bins - 1 && y[i] >= edges[b + 1]) {
            ++b;
        }
        bin_of_sample[i] = b;
    }
    free(order);
    free(sorted);
    free(edges);
    return N4M_OK;
}

static n4m_status_t approximate_mode(const int64_t* class_counts,
                                      int32_t n_classes,
                                      int64_t n_draws,
                                      n4m_kbins_mt19937_t* rng,
                                      int64_t* out) {
    int64_t total = 0;
    for (int32_t i = 0; i < n_classes; ++i) {
        total += class_counts[i];
    }
    if (total <= 0 || n_draws < 0 || n_draws > total) {
        return N4M_ERR_INVALID_ARGUMENT;
    }

    double* remainder = (double*)malloc((size_t)n_classes * sizeof(double));
    if (remainder == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }

    int64_t floor_sum = 0;
    for (int32_t i = 0; i < n_classes; ++i) {
        const double continuous =
            ((double)class_counts[i] / (double)total) * (double)n_draws;
        const double floored = floor(continuous);
        out[i] = (int64_t)floored;
        remainder[i] = continuous - floored;
        floor_sum += out[i];
    }

    int64_t need_to_add = n_draws - floor_sum;
    while (need_to_add > 0) {
        double best = -1.0;
        for (int32_t i = 0; i < n_classes; ++i) {
            if (remainder[i] > best) best = remainder[i];
        }
        if (best < 0.0) break;

        int32_t n_tied = 0;
        for (int32_t i = 0; i < n_classes; ++i) {
            if (remainder[i] == best) ++n_tied;
        }
        int64_t* tied = (int64_t*)malloc((size_t)n_tied * sizeof(int64_t));
        int64_t* perm = (int64_t*)malloc((size_t)n_tied * sizeof(int64_t));
        if (tied == NULL || perm == NULL) {
            free(tied);
            free(perm);
            free(remainder);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        int32_t pos = 0;
        for (int32_t i = 0; i < n_classes; ++i) {
            if (remainder[i] == best) tied[pos++] = i;
        }

        mt19937_permutation_i64(rng, perm, n_tied);
        const int64_t add_now =
            need_to_add < (int64_t)n_tied ? need_to_add : (int64_t)n_tied;
        for (int64_t j = 0; j < add_now; ++j) {
            out[tied[perm[j]]] += 1;
        }
        for (int32_t i = 0; i < n_tied; ++i) {
            remainder[tied[i]] = -1.0;
        }
        free(tied);
        free(perm);
        need_to_add -= add_now;
    }

    free(remainder);
    return N4M_OK;
}

n4m_status_t n4m_split_kbins_apply(const n4m_split_kbins_state_t* state,
                                    const double* y, int64_t rows,
                                    n4m_split_result_t* out) {
    if (state == NULL || y == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 2) return N4M_ERR_INVALID_ARGUMENT;
    int64_t n_train = 0, n_test = 0;
    n4m_status_t st = n4m_split_compute_train_test_count(rows, state->test_size,
                                                          &n_train, &n_test);
    if (st != N4M_OK) return st;

    int32_t* bin_of_sample = (int32_t*)malloc((size_t)rows * sizeof(int32_t));
    if (bin_of_sample == NULL) return N4M_ERR_OUT_OF_MEMORY;
    st = assign_bins(y, rows, state->n_bins, state->strategy, bin_of_sample);
    if (st != N4M_OK) { free(bin_of_sample); return st; }

    /* Group samples by non-empty bin. This is equivalent to
     * np.argsort(y_indices, kind="mergesort") split by class counts because
     * we scan samples in ascending index order while filling each class. */
    int32_t K = state->n_bins;
    int64_t* bin_offset = (int64_t*)calloc((size_t)(K + 1), sizeof(int64_t));
    int64_t* bin_list   = (int64_t*)malloc((size_t)rows * sizeof(int64_t));
    if (bin_offset == NULL || bin_list == NULL) {
        free(bin_of_sample); free(bin_offset); free(bin_list);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) ++bin_offset[bin_of_sample[i] + 1];
    for (int32_t k = 0; k < K; ++k) bin_offset[k + 1] += bin_offset[k];
    int64_t* fill = (int64_t*)calloc((size_t)K, sizeof(int64_t));
    if (fill == NULL) {
        free(bin_of_sample); free(bin_offset); free(bin_list);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const int32_t b = bin_of_sample[i];
        bin_list[bin_offset[b] + fill[b]++] = i;
    }
    free(fill);

    int32_t n_classes = 0;
    for (int32_t k = 0; k < K; ++k) {
        if (bin_offset[k + 1] > bin_offset[k]) ++n_classes;
    }
    int64_t* class_counts = (int64_t*)malloc((size_t)n_classes * sizeof(int64_t));
    int64_t* class_starts = (int64_t*)malloc((size_t)n_classes * sizeof(int64_t));
    int64_t* n_i = (int64_t*)malloc((size_t)n_classes * sizeof(int64_t));
    int64_t* t_i = (int64_t*)malloc((size_t)n_classes * sizeof(int64_t));
    if (class_counts == NULL || class_starts == NULL || n_i == NULL ||
        t_i == NULL) {
        free(bin_of_sample); free(bin_offset); free(bin_list);
        free(class_counts); free(class_starts); free(n_i); free(t_i);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    int32_t cls = 0;
    for (int32_t k = 0; k < K; ++k) {
        const int64_t lo = bin_offset[k];
        const int64_t hi = bin_offset[k + 1];
        if (hi <= lo) continue;
        class_starts[cls] = lo;
        class_counts[cls] = hi - lo;
        if (class_counts[cls] < 2) {
            free(bin_of_sample); free(bin_offset); free(bin_list);
            free(class_counts); free(class_starts); free(n_i); free(t_i);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        ++cls;
    }
    if (n_train < n_classes || n_test < n_classes) {
        free(bin_of_sample); free(bin_offset); free(bin_list);
        free(class_counts); free(class_starts); free(n_i); free(t_i);
        return N4M_ERR_INVALID_ARGUMENT;
    }

    n4m_kbins_mt19937_t rng;
    mt19937_seed(&rng, (uint32_t)state->seed);

    st = approximate_mode(class_counts, n_classes, n_train, &rng, n_i);
    if (st != N4M_OK) {
        free(bin_of_sample); free(bin_offset); free(bin_list);
        free(class_counts); free(class_starts); free(n_i); free(t_i);
        return st;
    }
    for (int32_t i = 0; i < n_classes; ++i) {
        t_i[i] = class_counts[i] - n_i[i];
    }
    st = approximate_mode(t_i, n_classes, n_test, &rng, t_i);
    if (st != N4M_OK) {
        free(bin_of_sample); free(bin_offset); free(bin_list);
        free(class_counts); free(class_starts); free(n_i); free(t_i);
        return st;
    }

    st = n4m_split_result_alloc(out, n_train, n_test);
    if (st != N4M_OK) {
        free(bin_of_sample); free(bin_offset); free(bin_list);
        free(class_counts); free(class_starts); free(n_i); free(t_i);
        return st;
    }
    int64_t train_pos = 0;
    int64_t test_pos = 0;
    for (int32_t i = 0; i < n_classes; ++i) {
        int64_t* perm = (int64_t*)malloc((size_t)class_counts[i] * sizeof(int64_t));
        if (perm == NULL) {
            n4m_split_result_free(out);
            free(bin_of_sample); free(bin_offset); free(bin_list);
            free(class_counts); free(class_starts); free(n_i); free(t_i);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        mt19937_permutation_i64(&rng, perm, class_counts[i]);
        for (int64_t j = 0; j < n_i[i]; ++j) {
            out->train_idx[train_pos++] =
                bin_list[class_starts[i] + perm[j]];
        }
        for (int64_t j = 0; j < t_i[i]; ++j) {
            out->test_idx[test_pos++] =
                bin_list[class_starts[i] + perm[n_i[i] + j]];
        }
        free(perm);
    }
    mt19937_shuffle_i64(&rng, out->train_idx, n_train);
    mt19937_shuffle_i64(&rng, out->test_idx, n_test);

    free(bin_of_sample); free(bin_offset); free(bin_list);
    free(class_counts); free(class_starts); free(n_i); free(t_i);
    return N4M_OK;
}
