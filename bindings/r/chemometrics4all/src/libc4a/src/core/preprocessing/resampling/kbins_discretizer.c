/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * IntegerKBinsDiscretizer — sklearn.preprocessing.KBinsDiscretizer with
 * `encode='ordinal'` and integer output, fit/transform contract.
 *
 * Reference:
 *   - sklearn.preprocessing._discretization.KBinsDiscretizer.{fit, transform}
 *   - nirs4all.operators.transforms.targets.IntegerKBinsDiscretizer
 *
 * Per-column algorithm (uniform):
 *   col_min, col_max = X[:, j].min(), X[:, j].max()
 *   edges = linspace(col_min, col_max, n_bins + 1)
 *
 * Per-column algorithm (quantile):
 *   pcts  = linspace(0, 100, n_bins + 1)
 *   edges = np.percentile(X[:, j], pcts)  # linear interpolation
 *
 * After computing edges, sklearn drops entries with width <= 1e-8 (per
 * column). We mirror this dedup step so the inner-edge searchsorted is
 * stable across degenerate inputs (e.g. a constant column).
 *
 * Transform per column j:
 *   out[:, j] = searchsorted(inner_edges_j, X[:, j], side='right')
 * with `inner_edges_j = edges_j[1:-1]` (length n_bins_j - 1).
 */

#include "kbins_discretizer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define C4A_KBINS_WIDTH_FLOOR 1e-8

struct c4a_pp_kbins_disc_state_t {
    int32_t  n_bins;
    int32_t  strategy;   /* 0 uniform, 1 quantile */
    int      fitted;
    int64_t  cols;
    /* For each column j of the fit matrix, `edges[j]` is a length-`n_edges[j]`
     * array of bin edges (including the two outer endpoints). The inner-edge
     * search uses edges[j][1 .. n_edges[j]-2]. */
    int32_t* n_edges;       /* per-column edge count, length cols */
    double** edges;         /* per-column edge arrays, length cols */
};

c4a_pp_kbins_disc_state_t* c4a_pp_kbins_disc_state_new(int32_t n_bins,
                                                        int32_t strategy) {
    if (n_bins < 2) {
        return NULL;
    }
    if (strategy != 0 && strategy != 1) {
        return NULL;
    }
    c4a_pp_kbins_disc_state_t* s =
        (c4a_pp_kbins_disc_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->n_bins   = n_bins;
    s->strategy = strategy;
    s->fitted   = 0;
    s->cols     = 0;
    s->n_edges  = NULL;
    s->edges    = NULL;
    return s;
}

static void free_edges(c4a_pp_kbins_disc_state_t* s) {
    if (s->edges != NULL) {
        for (int64_t j = 0; j < s->cols; ++j) {
            free(s->edges[j]);
        }
        free(s->edges);
        s->edges = NULL;
    }
    free(s->n_edges);
    s->n_edges = NULL;
}

void c4a_pp_kbins_disc_state_free(c4a_pp_kbins_disc_state_t* state) {
    if (state == NULL) return;
    free_edges(state);
    free(state);
}

int c4a_pp_kbins_disc_state_is_fitted(const c4a_pp_kbins_disc_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

/* Dedup edges in-place: drop edges[i] when (edges[i] - edges[i-1]) <= 1e-8.
 * Always keeps edges[0]. Returns the new length. */
static int32_t dedup_edges(double* edges, int32_t n) {
    if (n <= 1) return n;
    int32_t w = 1;
    for (int32_t r = 1; r < n; ++r) {
        if ((edges[r] - edges[w - 1]) > C4A_KBINS_WIDTH_FLOOR) {
            edges[w++] = edges[r];
        }
    }
    return w;
}

/* numpy.percentile(arr, q) with linear interpolation (default). q is in
 * [0, 100]. arr is sorted ascending in-place by the caller. */
static double percentile_linear(const double* sorted, int64_t n, double q) {
    if (n == 1) return sorted[0];
    /* Index in [0, n-1] using q/100. */
    const double t = (q / 100.0) * (double)(n - 1);
    int64_t i = (int64_t)floor(t);
    if (i < 0) i = 0;
    if (i >= n - 1) {
        return sorted[n - 1];
    }
    const double frac = t - (double)i;
    return sorted[i] + frac * (sorted[i + 1] - sorted[i]);
}

static int compare_double(const void* a, const void* b) {
    const double da = *(const double*)a;
    const double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

c4a_status_t c4a_pp_kbins_disc_state_fit(c4a_pp_kbins_disc_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols) {
    if (state == NULL || X == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    /* Tear down any prior fit (refit semantics). */
    free_edges(state);
    state->cols    = cols;
    state->n_edges = (int32_t*)malloc((size_t)cols * sizeof(int32_t));
    state->edges   = (double**)calloc((size_t)cols, sizeof(double*));
    if (state->n_edges == NULL || state->edges == NULL) {
        free_edges(state);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    const int32_t n_bins = state->n_bins;
    const int32_t n_init = n_bins + 1;
    double* col_buf = (double*)malloc((size_t)rows * sizeof(double));
    if (col_buf == NULL) {
        free_edges(state);
        return C4A_ERR_OUT_OF_MEMORY;
    }
    for (int64_t j = 0; j < cols; ++j) {
        double* e = (double*)malloc((size_t)n_init * sizeof(double));
        if (e == NULL) {
            free(col_buf);
            free_edges(state);
            return C4A_ERR_OUT_OF_MEMORY;
        }
        if (state->strategy == 0) {
            /* Uniform: linspace(col_min, col_max, n_bins + 1). */
            double col_min = X[(size_t)j];
            double col_max = col_min;
            for (int64_t i = 1; i < rows; ++i) {
                const double v = X[(size_t)i * (size_t)cols + (size_t)j];
                if (v < col_min) col_min = v;
                if (v > col_max) col_max = v;
            }
            if (n_init == 1) {
                e[0] = col_min;
            } else {
                const double step = (col_max - col_min) / (double)(n_init - 1);
                e[0] = col_min;
                for (int32_t k = 1; k < n_init - 1; ++k) {
                    e[k] = col_min + step * (double)k;
                }
                e[n_init - 1] = col_max;
            }
        } else {
            /* Quantile: sort the column then linear-interp percentiles. */
            for (int64_t i = 0; i < rows; ++i) {
                col_buf[i] = X[(size_t)i * (size_t)cols + (size_t)j];
            }
            qsort(col_buf, (size_t)rows, sizeof(double), compare_double);
            for (int32_t k = 0; k < n_init; ++k) {
                const double q = 100.0 * (double)k / (double)(n_init - 1);
                e[k] = percentile_linear(col_buf, rows, q);
            }
        }
        /* sklearn drops bins narrower than 1e-8 (per column). */
        const int32_t kept = dedup_edges(e, n_init);
        state->edges[j]   = e;
        state->n_edges[j] = kept;
    }
    free(col_buf);
    state->fitted = 1;
    return C4A_OK;
}

/* searchsorted(edges_inner, x, side='right') — leftmost i s.t. edges[i] > x. */
static int32_t searchsorted_right(const double* edges, int32_t n, double x) {
    int32_t lo = 0;
    int32_t hi = n;
    while (lo < hi) {
        const int32_t mid = (lo + hi) >> 1;
        if (edges[mid] <= x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

c4a_status_t c4a_pp_kbins_disc_state_apply(
    const c4a_pp_kbins_disc_state_t* state,
    const double* X,
    int64_t rows, int64_t cols,
    int32_t* out) {
    if (state == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return C4A_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->cols) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (rows > 0 && cols > 0 && X == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (rows == 0 || cols == 0) {
        return C4A_OK;
    }
    for (int64_t j = 0; j < cols; ++j) {
        const double* col_edges = state->edges[j];
        const int32_t n_e       = state->n_edges[j];
        /* Inner edges = col_edges[1 .. n_e - 1) — length n_e - 2. When n_e
         * <= 2 (degenerate column with 1 or 2 edges left after dedup), all
         * outputs collapse to 0. */
        const double*  inner = (n_e > 2) ? (col_edges + 1) : NULL;
        const int32_t  n_in  = (n_e > 2) ? (n_e - 2) : 0;
        for (int64_t i = 0; i < rows; ++i) {
            const double v =
                X[(size_t)i * (size_t)cols + (size_t)j];
            out[(size_t)i * (size_t)cols + (size_t)j] =
                searchsorted_right(inner, n_in, v);
        }
    }
    return C4A_OK;
}
