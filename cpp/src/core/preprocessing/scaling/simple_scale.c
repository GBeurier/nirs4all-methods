/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SimpleScale — per-column min-max scaling to [0, 1].
 *
 * Computes nirs4all's `X = (X - min_) / (max_ - min_)` as a multiplication
 * by the per-column reciprocal `1 / (max_ - min_)`.
 */

#include "simple_scale.h"

#include <stdlib.h>

#define N4M_PP_SIMPLE_SCALE_STACK_COLS 4096

#if defined(__GNUC__) || defined(__clang__)
#define N4M_RESTRICT __restrict__
#else
#define N4M_RESTRICT
#endif

struct n4m_pp_simple_scale_state_t {
    int _reserved;
};

n4m_pp_simple_scale_state_t* n4m_pp_simple_scale_state_new(void) {
    static n4m_pp_simple_scale_state_t state = {0};
    return &state;
}

void n4m_pp_simple_scale_state_free(n4m_pp_simple_scale_state_t* state) {
    (void)state;
}

/* Preserve the per-column left-to-right min/max order (numpy.min / max)
 * while reading the matrix row-major. */
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

/* `inv_range[j] = 1 / (max - min)`; may alias `col_max`. */
static void column_inv_range(const double* col_min, const double* col_max,
                             size_t n_cols, double* inv_range) {
    for (size_t j = 0; j < n_cols; ++j) {
        inv_range[j] = 1.0 / (col_max[j] - col_min[j]);
    }
}

static void apply_min_range(const double* col_min, const double* inv_range,
                            const double* X, int64_t rows, size_t n_cols,
                            double* out) {
    for (int64_t i = 0; i < rows; ++i) {
        const double* N4M_RESTRICT row_in = X + (size_t)i * n_cols;
        double* N4M_RESTRICT row_out = out + (size_t)i * n_cols;
        for (size_t j = 0; j < n_cols; ++j) {
            row_out[j] = (row_in[j] - col_min[j]) * inv_range[j];
        }
    }
}

n4m_status_t n4m_pp_simple_scale_apply(const n4m_pp_simple_scale_state_t* state,
                                       const double* X,
                                       int64_t rows, int64_t cols,
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

    const size_t n_cols = (size_t)cols;
    double min_stack[N4M_PP_SIMPLE_SCALE_STACK_COLS];
    double range_stack[N4M_PP_SIMPLE_SCALE_STACK_COLS];
    double* col_min = min_stack;
    double* col_range = range_stack;
    int heap = 0;

    if (n_cols > N4M_PP_SIMPLE_SCALE_STACK_COLS) {
        col_min = (double*)malloc(n_cols * sizeof(*col_min));
        col_range = (double*)malloc(n_cols * sizeof(*col_range));
        if (col_min == NULL || col_range == NULL) {
            free(col_min);
            free(col_range);
            return N4M_ERR_OUT_OF_MEMORY;
        }
        heap = 1;
    }

    column_min_max(X, rows, n_cols, col_min, col_range);
    column_inv_range(col_min, col_range, n_cols, col_range);
    apply_min_range(col_min, col_range, X, rows, n_cols, out);
    if (heap) {
        free(col_min);
        free(col_range);
    }
    return N4M_OK;
}

struct n4m_pp_simple_scale_fit_state_t {
    int     fitted;
    int64_t cols;
    double* col_min;
    double* col_max;
    double* inv_range;  /* derived: 1 / (max - min) */
};

static void fit_state_clear(n4m_pp_simple_scale_fit_state_t* s) {
    free(s->col_min);
    free(s->col_max);
    free(s->inv_range);
    s->col_min   = NULL;
    s->col_max   = NULL;
    s->inv_range = NULL;
    s->fitted    = 0;
    s->cols      = 0;
}

n4m_pp_simple_scale_fit_state_t* n4m_pp_simple_scale_fit_state_new(void) {
    return (n4m_pp_simple_scale_fit_state_t*)calloc(
        1, sizeof(n4m_pp_simple_scale_fit_state_t));
}

void n4m_pp_simple_scale_fit_state_free(n4m_pp_simple_scale_fit_state_t* state) {
    if (state == NULL) return;
    fit_state_clear(state);
    free(state);
}

/* Takes ownership of `col_min` / `col_max`. */
static n4m_status_t fit_state_install(n4m_pp_simple_scale_fit_state_t* s,
                                      int64_t cols, double* col_min,
                                      double* col_max) {
    double* inv_range = (double*)malloc((size_t)cols * sizeof(double));
    if (inv_range == NULL) {
        free(col_min);
        free(col_max);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    column_inv_range(col_min, col_max, (size_t)cols, inv_range);
    fit_state_clear(s);
    s->col_min   = col_min;
    s->col_max   = col_max;
    s->inv_range = inv_range;
    s->cols      = cols;
    s->fitted    = 1;
    return N4M_OK;
}

n4m_status_t n4m_pp_simple_scale_fit_state_fit(n4m_pp_simple_scale_fit_state_t* state,
                                               const double* X, int64_t rows,
                                               int64_t cols) {
    if (state == NULL || X == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 1 || cols < 1) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    double* col_min = (double*)malloc((size_t)cols * sizeof(double));
    double* col_max = (double*)malloc((size_t)cols * sizeof(double));
    if (col_min == NULL || col_max == NULL) {
        free(col_min);
        free(col_max);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    column_min_max(X, rows, (size_t)cols, col_min, col_max);
    return fit_state_install(state, cols, col_min, col_max);
}

n4m_status_t n4m_pp_simple_scale_fit_state_apply(
    const n4m_pp_simple_scale_fit_state_t* state, const double* X, int64_t rows,
    int64_t cols, double* out) {
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
    apply_min_range(state->col_min, state->inv_range, X, rows, (size_t)cols, out);
    return N4M_OK;
}

n4m_status_t n4m_pp_simple_scale_fit_state_save(
    const n4m_pp_simple_scale_fit_state_t* state, n4m_state_writer_t* w) {
    if (!state->fitted) return N4M_ERR_NOT_FITTED;
    n4m_state_write_f64_array(w, state->col_min, state->cols);
    n4m_state_write_f64_array(w, state->col_max, state->cols);
    return N4M_OK;
}

n4m_status_t n4m_pp_simple_scale_fit_state_load(
    n4m_pp_simple_scale_fit_state_t* state, n4m_state_reader_t* r, int64_t n_features) {
    double* col_min = NULL;
    double* col_max = NULL;
    n4m_status_t st = n4m_state_read_f64_array_new(r, n_features, &col_min);
    if (st == N4M_OK) st = n4m_state_read_f64_array_new(r, n_features, &col_max);
    if (st != N4M_OK) {
        free(col_min);
        return st;
    }
    return fit_state_install(state, n_features, col_min, col_max);
}
