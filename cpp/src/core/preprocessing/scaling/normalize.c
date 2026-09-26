/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Normalize (column-wise) reference implementation.
 *
 * Linalg-norm mode replicates np.linalg.norm with axis=0 (Frobenius / L2
 * column norm) by accumulating x*x left-to-right per column and taking
 * sqrt at the end — same order as numpy's reduction.
 *
 * User-defined-range mode computes per-column min/max via standard
 * left-to-right scan; matches numpy.min/max on the same memory order.
 */

#include "normalize.h"

#include <math.h>
#include <stdlib.h>

#define N4M_PP_NORMALIZE_STACK_COLS 4096

#if defined(__GNUC__) || defined(__clang__)
#define N4M_RESTRICT __restrict__
#else
#define N4M_RESTRICT
#endif

struct n4m_pp_normalize_state_t {
    double  feature_min;
    double  feature_max;
    int     user_defined;
    int     fitted;
    int64_t cols;
    double* stat_lo;  /* range mode: column min; unused (NULL) in norm mode */
    double* stat_hi;  /* range mode: column max; norm mode: column L2 norm */
    double* factor;   /* derived: span / (max - min), or 1 / norm */
};

/* nirs4all: `user_defined = feature_range[0] != -1 or feature_range[1] != 1` */
static int is_user_defined(double feature_min, double feature_max) {
    return (feature_min != -1.0) || (feature_max != 1.0);
}

/* numpy.min / numpy.max scan left-to-right within each column. This
 * row-major traversal preserves that per-column order while avoiding
 * repeated long-stride passes over X. */
static void column_min_max(const double* X, int64_t rows, size_t n_cols,
                           double* col_min, double* col_max) {
    for (size_t j = 0; j < n_cols; ++j) {
        col_min[j] = X[j];
        col_max[j] = X[j];
    }
    for (int64_t i = 1; i < rows; ++i) {
        const double* N4M_RESTRICT row = X + (size_t)i * n_cols;
        for (size_t j = 0; j < n_cols; ++j) {
            const double v = row[j];
            if (v < col_min[j]) col_min[j] = v;
            if (v > col_max[j]) col_max[j] = v;
        }
    }
}

/* np.linalg.norm(X, axis=0) = sqrt(sum(X[:, j]^2)). For every column the
 * additions happen in row order, matching numpy's reduction order. */
static void column_norms(const double* X, int64_t rows, size_t n_cols,
                         double* norm) {
    for (size_t j = 0; j < n_cols; ++j) {
        norm[j] = 0.0;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* N4M_RESTRICT row = X + (size_t)i * n_cols;
        for (size_t j = 0; j < n_cols; ++j) {
            const double v = row[j];
            norm[j] += v * v;
        }
    }
    for (size_t j = 0; j < n_cols; ++j) {
        norm[j] = sqrt(norm[j]);
    }
}

/* Per-column factors from the statistics: `f = (imax - imin) / (max - min)`
 * in range mode, `1 / norm` in norm mode (a single division per column). */
static void column_factors(int user_defined, double span, const double* lo,
                           const double* hi, size_t n_cols, double* factor) {
    for (size_t j = 0; j < n_cols; ++j) {
        factor[j] = user_defined ? span / (hi[j] - lo[j]) : 1.0 / hi[j];
    }
}

/* nirs4all: `X = imin + f * (X - min_)` (range) or `X / norm` (norm). We
 * replicate this exact arithmetic tree (one multiplication per element by
 * the per-column factor) so the rounding sequence matches. */
static void apply_factors(int user_defined, double imin, const double* lo,
                          const double* factor, const double* X, int64_t rows,
                          size_t n_cols, double* out) {
    for (int64_t i = 0; i < rows; ++i) {
        const double* N4M_RESTRICT row_in = X + (size_t)i * n_cols;
        double* N4M_RESTRICT row_out = out + (size_t)i * n_cols;
        if (user_defined) {
            for (size_t j = 0; j < n_cols; ++j) {
                row_out[j] = imin + factor[j] * (row_in[j] - lo[j]);
            }
        } else {
            for (size_t j = 0; j < n_cols; ++j) {
                row_out[j] = row_in[j] * factor[j];
            }
        }
    }
}

static void clear_fit(n4m_pp_normalize_state_t* s) {
    free(s->stat_lo);
    free(s->stat_hi);
    free(s->factor);
    s->stat_lo = NULL;
    s->stat_hi = NULL;
    s->factor  = NULL;
    s->fitted  = 0;
    s->cols    = 0;
}

n4m_pp_normalize_state_t* n4m_pp_normalize_state_new(double feature_min,
                                                     double feature_max) {
    n4m_pp_normalize_state_t* s =
        (n4m_pp_normalize_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->feature_min  = feature_min;
    s->feature_max  = feature_max;
    s->user_defined = is_user_defined(feature_min, feature_max);
    return s;
}

void n4m_pp_normalize_state_free(n4m_pp_normalize_state_t* state) {
    if (state == NULL) return;
    clear_fit(state);
    free(state);
}

/* Takes ownership of `lo` / `hi` (lo is NULL in norm mode). */
static n4m_status_t install_stats(n4m_pp_normalize_state_t* s, int64_t cols,
                                  double* lo, double* hi) {
    double* factor = (double*)malloc((size_t)cols * sizeof(double));
    if (factor == NULL) {
        free(lo);
        free(hi);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    column_factors(s->user_defined, s->feature_max - s->feature_min, lo, hi,
                   (size_t)cols, factor);
    clear_fit(s);
    s->stat_lo = lo;
    s->stat_hi = hi;
    s->factor  = factor;
    s->cols    = cols;
    s->fitted  = 1;
    return N4M_OK;
}

n4m_status_t n4m_pp_normalize_state_fit(n4m_pp_normalize_state_t* state,
                                        const double* X, int64_t rows,
                                        int64_t cols) {
    if (state == NULL || X == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const size_t n_cols = (size_t)cols;
    double* lo = NULL;
    double* hi = (double*)malloc(n_cols * sizeof(double));
    if (state->user_defined) {
        lo = (double*)malloc(n_cols * sizeof(double));
    }
    if (hi == NULL || (state->user_defined && lo == NULL)) {
        free(lo);
        free(hi);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    if (state->user_defined) {
        column_min_max(X, rows, n_cols, lo, hi);
    } else {
        column_norms(X, rows, n_cols, hi);
    }
    return install_stats(state, cols, lo, hi);
}

n4m_status_t n4m_pp_normalize_apply(const n4m_pp_normalize_state_t* state,
                                    const double* X, int64_t rows, int64_t cols,
                                    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return N4M_ERR_NOT_FITTED;
    }
    if (rows < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->cols) {
        return N4M_ERR_SHAPE_MISMATCH;
    }
    apply_factors(state->user_defined, state->feature_min, state->stat_lo,
                  state->factor, X, rows, (size_t)cols, out);
    return N4M_OK;
}

n4m_status_t n4m_pp_normalize_apply_params(double feature_min,
                                           double feature_max,
                                           const double* X, int64_t rows,
                                           int64_t cols, double* out) {
    if (X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }

    const size_t n_cols = (size_t)cols;
    const int user_defined = is_user_defined(feature_min, feature_max);
    double lo_stack[N4M_PP_NORMALIZE_STACK_COLS];
    double hi_stack[N4M_PP_NORMALIZE_STACK_COLS];
    double* lo = lo_stack;
    double* hi = hi_stack;
    int heap = 0;

    if (n_cols > N4M_PP_NORMALIZE_STACK_COLS) {
        lo = (double*)malloc(n_cols * sizeof(*lo));
        hi = (double*)malloc(n_cols * sizeof(*hi));
        if (lo == NULL || hi == NULL) {
            free(lo);
            free(hi);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        heap = 1;
    }
    /* The factors overwrite the column max / norm, which apply no longer
     * needs. */
    if (user_defined) {
        column_min_max(X, rows, n_cols, lo, hi);
    } else {
        column_norms(X, rows, n_cols, hi);
    }
    column_factors(user_defined, feature_max - feature_min, lo, hi, n_cols, hi);
    apply_factors(user_defined, feature_min, lo, hi, X, rows, n_cols, out);
    if (heap) {
        free(lo);
        free(hi);
    }
    return N4M_OK;
}

n4m_status_t n4m_pp_normalize_state_save(const n4m_pp_normalize_state_t* state,
                                         n4m_state_writer_t* w) {
    if (!state->fitted) return N4M_ERR_NOT_FITTED;
    if (state->user_defined) {
        n4m_state_write_f64_array(w, state->stat_lo, state->cols);
    }
    n4m_state_write_f64_array(w, state->stat_hi, state->cols);
    return N4M_OK;
}

n4m_status_t n4m_pp_normalize_state_load(n4m_pp_normalize_state_t* state,
                                         n4m_state_reader_t* r, int64_t n_features) {
    double* lo = NULL;
    double* hi = NULL;
    n4m_status_t st = N4M_OK;
    if (state->user_defined) {
        st = n4m_state_read_f64_array_new(r, n_features, &lo);
    }
    if (st == N4M_OK) st = n4m_state_read_f64_array_new(r, n_features, &hi);
    if (st != N4M_OK) {
        free(lo);
        return st;
    }
    return install_stats(state, n_features, lo, hi);
}
