/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Robust Standard Normal Variate (RobustSNV / RSNV).
 *
 * Two passes per row: first compute the median of the row, subtract it,
 * then compute the median of the absolute deviations (MAD). Scale = k * MAD;
 * if the scale is exactly zero (constant row) we divide by 1.0 instead, which
 * matches nirs4all's `scale[scale == 0] = 1.0` guard.
 *
 * The median routine partitions a per-row scratch buffer in place with
 * Hoare's Quickselect, then for even-length rows averages the two central
 * order statistics — bit-exact with `numpy.median` for any reasonable input
 * because all order statistics are computed by direct comparison + averaging.
 */

#include "robust_snv.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct n4m_pp_rnv_state_t {
    int    with_center;
    int    with_scale;
    double k;
};

n4m_pp_rnv_state_t* n4m_pp_rnv_state_new(int with_center, int with_scale, double k) {
    n4m_pp_rnv_state_t* s = (n4m_pp_rnv_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->with_center = with_center ? 1 : 0;
    s->with_scale  = with_scale  ? 1 : 0;
    s->k           = k;
    return s;
}

void n4m_pp_rnv_state_free(n4m_pp_rnv_state_t* state) {
    free(state);
}

static inline void swap_double(double* a, double* b) {
    double t = *a;
    *a = *b;
    *b = t;
}

/* Quickselect on buf[lo..hi] (inclusive): rearrange so the k-th order
 * statistic ends up at buf[k]. Uses Lomuto partitioning with median-of-three
 * pivot selection — simple, predictable, and avoids the edge cases of
 * Hoare's "stash pivot at hi-1" trick on ranges of <= 3 elements.
 *
 * Invariant post-partition: buf[lo..p-1] <= buf[p] <= buf[p+1..hi]
 * (with <= on the left, < on the right matched to the median-of-3
 * pre-ordering). */
static void quickselect(double* buf, int64_t lo, int64_t hi, int64_t k) {
    while (lo < hi) {
        if (hi - lo == 1) {
            if (buf[lo] > buf[hi]) swap_double(&buf[lo], &buf[hi]);
            return;
        }
        /* Median-of-three pivot: order buf[lo] <= buf[mid] <= buf[hi]; then
         * move the median (buf[mid]) to position `lo + 1` so the partition
         * loop starts on a non-pivot element. */
        const int64_t mid = lo + (hi - lo) / 2;
        if (buf[lo] > buf[mid]) swap_double(&buf[lo], &buf[mid]);
        if (buf[lo] > buf[hi])  swap_double(&buf[lo], &buf[hi]);
        if (buf[mid] > buf[hi]) swap_double(&buf[mid], &buf[hi]);
        swap_double(&buf[mid], &buf[lo + 1]);
        const double pivot = buf[lo + 1];

        /* Lomuto-style partition: i sweeps from lo+2 to hi, swapping each
         * element < pivot down. The pivot finally lands at position `store`. */
        int64_t store = lo + 1;
        for (int64_t i = lo + 2; i <= hi; ++i) {
            if (buf[i] < pivot) {
                ++store;
                swap_double(&buf[i], &buf[store]);
            }
        }
        /* Place the pivot in its final slot. */
        swap_double(&buf[lo + 1], &buf[store]);

        if (store == k) return;
        if (store > k) {
            hi = store - 1;
        } else {
            lo = store + 1;
        }
    }
}

/* Median of buf[0..n) — destructive: buf is reordered in place.
 * Mirrors numpy.median: for even n, returns the mean of the two middle order
 * statistics; for odd n, returns the central order statistic. */
static double median_destructive(double* buf, int64_t n) {
    if (n <= 0) {
        return 0.0;
    }
    if (n == 1) {
        return buf[0];
    }
    if ((n & 1) == 1) {
        const int64_t mid = n / 2;
        quickselect(buf, 0, n - 1, mid);
        return buf[mid];
    }
    const int64_t hi_idx = n / 2;
    /* Place the upper-middle order statistic at hi_idx; everything below
     * is <= buf[hi_idx]; the lower-middle is then the max of buf[0..hi_idx). */
    quickselect(buf, 0, n - 1, hi_idx);
    double upper = buf[hi_idx];
    double lower = buf[0];
    for (int64_t i = 1; i < hi_idx; ++i) {
        if (buf[i] > lower) {
            lower = buf[i];
        }
    }
    return (lower + upper) * 0.5;
}

n4m_status_t n4m_pp_rnv_apply(const n4m_pp_rnv_state_t* state,
                              const double* X, int64_t rows, int64_t cols,
                              double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }

    double* scratch = (double*)malloc((size_t)cols * sizeof(double));
    if (scratch == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t i = 0; i < rows; ++i) {
        const double* row_in  = X + (size_t)i * (size_t)cols;
        double*       row_out = out + (size_t)i * (size_t)cols;

        /* Compute the median only if centering is enabled. nirs4all evaluates
         * the two `if` blocks sequentially: centering subtracts the median
         * first; scaling then takes the MAD of the (possibly-centered) data.
         * When `with_center=False`, MAD is computed on raw |X|. */
        double med = 0.0;
        if (state->with_center) {
            memcpy(scratch, row_in, (size_t)cols * sizeof(double));
            med = median_destructive(scratch, cols);
        }

        double scale = 1.0;
        if (state->with_scale) {
            if (state->with_center) {
                for (int64_t j = 0; j < cols; ++j) {
                    scratch[j] = fabs(row_in[j] - med);
                }
            } else {
                for (int64_t j = 0; j < cols; ++j) {
                    scratch[j] = fabs(row_in[j]);
                }
            }
            const double mad = median_destructive(scratch, cols);
            const double s   = state->k * mad;
            if (s != 0.0) {
                scale = s;
            }
        }

        if (state->with_center && state->with_scale) {
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = (row_in[j] - med) / scale;
            }
        } else if (state->with_center) {
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = row_in[j] - med;
            }
        } else if (state->with_scale) {
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = row_in[j] / scale;
            }
        } else {
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = row_in[j];
            }
        }
    }

    free(scratch);
    return N4M_OK;
}
