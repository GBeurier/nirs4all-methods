/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * RangeDiscretizer — `np.digitize(X, bins, right=False)` over a fixed edge
 * vector. Stateless.
 *
 * Reference: nirs4all.operators.transforms.targets.RangeDiscretizer.
 *
 * `np.digitize(arr, bins, right=False)`: for each x in arr returns the
 * smallest i such that x < bins[i], or len(bins) if no such i exists. This
 * is equivalent to `searchsorted(bins, x, side='right')`. We implement it
 * with a binary search.
 */

#include "range_discretizer.h"

#include <stdlib.h>
#include <string.h>

struct n4m_pp_range_disc_state_t {
    int64_t n_edges;
    double* edges;
};

n4m_pp_range_disc_state_t* n4m_pp_range_disc_state_new(const double* bins,
                                                        int64_t n_edges) {
    if (n_edges < 0) {
        return NULL;
    }
    if (n_edges > 0 && bins == NULL) {
        return NULL;
    }
    /* Verify strictly ascending. */
    for (int64_t i = 1; i < n_edges; ++i) {
        if (!(bins[i - 1] < bins[i])) {
            return NULL;
        }
    }
    n4m_pp_range_disc_state_t* s =
        (n4m_pp_range_disc_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->n_edges = n_edges;
    if (n_edges == 0) {
        s->edges = NULL;
    } else {
        s->edges = (double*)malloc((size_t)n_edges * sizeof(double));
        if (s->edges == NULL) {
            free(s);
            return NULL;
        }
        memcpy(s->edges, bins, (size_t)n_edges * sizeof(double));
    }
    return s;
}

void n4m_pp_range_disc_state_free(n4m_pp_range_disc_state_t* state) {
    if (state == NULL) return;
    free(state->edges);
    free(state);
}

/* Digitize with right=False: returns i such that bins[i-1] <= x < bins[i].
 * Equivalent to searchsorted(bins, x, side='right'). */
static int32_t digitize_left_inclusive(const double* edges, int64_t n,
                                        double x) {
    int64_t lo = 0;
    int64_t hi = n;
    while (lo < hi) {
        const int64_t mid = (lo + hi) >> 1;
        if (edges[mid] <= x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return (int32_t)lo;
}

n4m_status_t n4m_pp_range_disc_apply(const n4m_pp_range_disc_state_t* state,
                                      const double* X,
                                      int64_t rows, int64_t cols,
                                      int32_t* out) {
    if (state == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows > 0 && cols > 0 && X == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    const int64_t total = rows * cols;
    if (total == 0) {
        return N4M_OK;
    }
    for (int64_t k = 0; k < total; ++k) {
        out[k] = digitize_left_inclusive(state->edges, state->n_edges, X[k]);
    }
    return N4M_OK;
}
