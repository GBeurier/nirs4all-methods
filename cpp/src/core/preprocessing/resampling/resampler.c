/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Resampler — wavelength-grid interpolation between a fitted source axis and
 * a configured target axis.
 *
 * Reference: nirs4all.operators.transforms.resampler.Resampler.
 *
 * The operator caches:
 *   - source wavelengths (post-crop), strictly ascending;
 *   - optional crop mask (indices into the raw input);
 *   - per-target-query segment index (`seg[k]`) so the per-row hot path is
 *     just bracketing-table lookups + arithmetic;
 *   - for `method == cubic`, the source-segment widths `h[i]` and pre-
 *     allocated Thomas-algorithm scratch buffers — the actual tridiagonal
 *     system is re-derived and solved per row because its RHS depends on y.
 *
 * Per-row apply:
 *   - linear : slope * (q - x_lo) + y_lo
 *   - nearest: y[round_to_nearest_index(q)] (scipy interp1d 'nearest', uses
 *              midpoints as bin boundaries, `side='left'` rounding).
 *   - cubic  : not-a-knot CubicSpline (scipy default for kind='cubic').
 *
 * Bounds handling:
 *   - `bounds_error == 1` returns C4A_ERR_NUMERICAL_FAILURE on any
 *     out-of-range query.
 *   - `bounds_error == 0` and `extrapolate == 0`: out-of-range queries are
 *     filled with `fill_value`.
 *   - `bounds_error == 0` and `extrapolate == 1`: out-of-range queries are
 *     evaluated by the interpolation method (linear extrapolates the end
 *     slope; nearest clamps to the boundary y; cubic extrapolates the end
 *     cubic segment).
 */

#include "resampler.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct c4a_pp_resampler_state_t {
    /* Configuration */
    int32_t  method;          /* 0 linear, 1 nearest, 2 cubic */
    int32_t  use_crop;
    double   crop_min;
    double   crop_max;
    double   fill_value;
    int32_t  bounds_error;
    int32_t  extrapolate;
    int64_t  n_target;        /* -1 == identity */
    double*  target_wl;       /* length n_target (NULL when identity) */

    /* Fitted state */
    int      fitted;
    int64_t  n_raw;           /* raw source length (pre-crop) */
    int64_t  n_source;        /* post-crop source length */
    double*  source_wl;       /* length n_source (post-crop axis) */
    int64_t* crop_idx;        /* length n_source, indices into raw input (or NULL when no crop) */

    /* Per-query bracketing (length n_target; absent in identity) */
    int64_t* seg_idx;
    int32_t* out_of_lo;
    int32_t* out_of_hi;

    /* Cubic spline precomputed quantities (method == 2 only) */
    double*  h;             /* length n_source - 1 (source segment widths) */
    /* Per-row scratch — owned to avoid mallocs in the hot path.
     * Thomas-algorithm needs two length-n_interior auxiliaries (diag + super)
     * plus the RHS and the M output. We pack them in a single allocation:
     *   cs_buf layout (length 4 * n_source):
     *     [0 .. n_source)               — RHS storage
     *     [n_source .. 2*n_source)      — M output
     *     [2*n_source .. 3*n_source)    — Thomas reduced diagonal
     *     [3*n_source .. 4*n_source)    — Thomas reduced super-diagonal
     */
    double*  cs_buf;
};

static void state_clear_fit(c4a_pp_resampler_state_t* s) {
    free(s->source_wl);    s->source_wl = NULL;
    free(s->crop_idx);     s->crop_idx  = NULL;
    free(s->seg_idx);      s->seg_idx   = NULL;
    free(s->out_of_lo);    s->out_of_lo = NULL;
    free(s->out_of_hi);    s->out_of_hi = NULL;
    free(s->h);            s->h         = NULL;
    free(s->cs_buf);       s->cs_buf    = NULL;
    s->fitted   = 0;
    s->n_raw    = 0;
    s->n_source = 0;
}

c4a_pp_resampler_state_t* c4a_pp_resampler_state_new(
    const double* target_wl, int64_t n_target,
    int32_t method,
    double crop_min, double crop_max, int use_crop,
    double fill_value, int bounds_error, int extrapolate) {
    if (method != 0 && method != 1 && method != 2) {
        return NULL;
    }
    if (n_target != -1) {
        if (n_target < 1 || target_wl == NULL) {
            return NULL;
        }
        for (int64_t i = 1; i < n_target; ++i) {
            if (!(target_wl[i - 1] < target_wl[i])) {
                return NULL;
            }
        }
    }
    if (use_crop && !(crop_min <= crop_max)) {
        return NULL;
    }
    c4a_pp_resampler_state_t* s =
        (c4a_pp_resampler_state_t*)calloc(1, sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->method        = method;
    s->use_crop      = use_crop ? 1 : 0;
    s->crop_min      = crop_min;
    s->crop_max      = crop_max;
    s->fill_value    = fill_value;
    s->bounds_error  = bounds_error ? 1 : 0;
    s->extrapolate   = extrapolate ? 1 : 0;
    s->n_target      = n_target;
    if (n_target != -1) {
        s->target_wl = (double*)malloc((size_t)n_target * sizeof(double));
        if (s->target_wl == NULL) {
            free(s);
            return NULL;
        }
        memcpy(s->target_wl, target_wl, (size_t)n_target * sizeof(double));
    }
    return s;
}

void c4a_pp_resampler_state_free(c4a_pp_resampler_state_t* state) {
    if (state == NULL) return;
    state_clear_fit(state);
    free(state->target_wl);
    free(state);
}

int c4a_pp_resampler_state_is_fitted(const c4a_pp_resampler_state_t* state) {
    return (state != NULL && state->fitted) ? 1 : 0;
}

c4a_status_t c4a_pp_resampler_state_fit(c4a_pp_resampler_state_t* state,
                                         const double* source_wl,
                                         int64_t n_source) {
    if (state == NULL || source_wl == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n_source < 2) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    for (int64_t i = 1; i < n_source; ++i) {
        if (!(source_wl[i - 1] < source_wl[i])) {
            return C4A_ERR_INVALID_ARGUMENT;
        }
    }
    state_clear_fit(state);
    state->n_raw = n_source;
    /* Apply crop if requested. */
    int64_t n_eff = n_source;
    if (state->use_crop) {
        int64_t kept = 0;
        int64_t* idx = (int64_t*)malloc((size_t)n_source * sizeof(int64_t));
        if (idx == NULL) return C4A_ERR_OUT_OF_MEMORY;
        for (int64_t i = 0; i < n_source; ++i) {
            if (source_wl[i] >= state->crop_min &&
                source_wl[i] <= state->crop_max) {
                idx[kept++] = i;
            }
        }
        if (kept == 0) {
            free(idx);
            return C4A_ERR_INVALID_ARGUMENT;
        }
        state->crop_idx = idx;
        n_eff = kept;
    }
    state->n_source  = n_eff;
    state->source_wl = (double*)malloc((size_t)n_eff * sizeof(double));
    if (state->source_wl == NULL) return C4A_ERR_OUT_OF_MEMORY;
    if (state->use_crop) {
        for (int64_t i = 0; i < n_eff; ++i) {
            state->source_wl[i] = source_wl[state->crop_idx[i]];
        }
    } else {
        memcpy(state->source_wl, source_wl,
               (size_t)n_eff * sizeof(double));
    }
    /* Cubic requires at least 4 source points for the not-a-knot system. */
    if (state->method == 2 && n_eff < 4) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    /* Linear/nearest need at least 2 source points (already verified). */
    /* Pre-bracket every target query. */
    const int64_t n_target = state->n_target;
    if (n_target != -1) {
        state->seg_idx   = (int64_t*)malloc((size_t)n_target * sizeof(int64_t));
        state->out_of_lo = (int32_t*)calloc((size_t)n_target, sizeof(int32_t));
        state->out_of_hi = (int32_t*)calloc((size_t)n_target, sizeof(int32_t));
        if (state->seg_idx == NULL || state->out_of_lo == NULL ||
            state->out_of_hi == NULL) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
        const double src_lo = state->source_wl[0];
        const double src_hi = state->source_wl[n_eff - 1];
        for (int64_t k = 0; k < n_target; ++k) {
            const double q = state->target_wl[k];
            if (q < src_lo) {
                state->out_of_lo[k] = 1;
                state->seg_idx[k] = 0;
                continue;
            }
            if (q > src_hi) {
                state->out_of_hi[k] = 1;
                state->seg_idx[k] = n_eff - 2;
                continue;
            }
            /* Locate segment [source_wl[i], source_wl[i+1]] containing q.
             * For q == src_hi we fall into segment n_eff - 2. */
            int64_t lo = 0;
            int64_t hi = n_eff - 1;
            while (lo + 1 < hi) {
                const int64_t mid = (lo + hi) >> 1;
                if (state->source_wl[mid] <= q) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            state->seg_idx[k] = lo;
        }
    }
    /* Cubic-specific precomputations. */
    if (state->method == 2) {
        state->h = (double*)malloc((size_t)(n_eff - 1) * sizeof(double));
        if (state->h == NULL) return C4A_ERR_OUT_OF_MEMORY;
        for (int64_t i = 0; i < n_eff - 1; ++i) {
            state->h[i] = state->source_wl[i + 1] - state->source_wl[i];
        }
        state->cs_buf = (double*)malloc(4 * (size_t)n_eff * sizeof(double));
        if (state->cs_buf == NULL) return C4A_ERR_OUT_OF_MEMORY;
    }
    state->fitted = 1;
    return C4A_OK;
}

int64_t c4a_pp_resampler_state_input_cols(const c4a_pp_resampler_state_t* s) {
    if (s == NULL || !s->fitted) return 0;
    return s->n_raw;
}

int64_t c4a_pp_resampler_state_output_cols(const c4a_pp_resampler_state_t* s) {
    if (s == NULL || !s->fitted) return 0;
    if (s->n_target == -1) {
        return s->n_source;
    }
    return s->n_target;
}

/* Solve the not-a-knot cubic-spline tridiagonal system A * M = b on the
 * length-n source axis with `h[i] = source[i+1] - source[i]`. The standard
 * 3-banded boundary rows are eliminated up-front so the interior system in
 * M[1..n-2] is genuinely tridiagonal and amenable to the Thomas algorithm.
 *
 * Row 0 (not-a-knot):    h[1]*M[0] - (h[0]+h[1])*M[1] + h[0]*M[2] = 0
 *   => M[0] = ((h[0]+h[1])*M[1] - h[0]*M[2]) / h[1]
 * Substitute into the interior row at i = 1:
 *   h[0]*M[0] + 2*(h[0]+h[1])*M[1] + h[1]*M[2] = rhs[1]
 *   becomes
 *   D_1 * M[1] + S_1 * M[2] = rhs[1]
 * with
 *   D_1 = (h[0]+h[1]) * (h[0] + 2*h[1]) / h[1]
 *   S_1 = (h[1]^2 - h[0]^2) / h[1]
 *
 * Symmetric elimination on row n-1 (relating M[n-3], M[n-2], M[n-1]).
 *
 * Interior rows i = 2..n-3 keep the standard:
 *   A[i] = h[i-1],  D[i] = 2*(h[i-1]+h[i]),  S[i] = h[i],  RHS[i] = rhs[i]
 *
 * After Thomas, recover M[0] and M[n-1] from the BC expressions.
 */
static void solve_not_a_knot(const double* h,
                              const double* rhs,
                              int64_t n,
                              double* M,
                              double* diag_buf,
                              double* super_buf,
                              double* rb_buf) {
    /* `n_interior = n - 2` unknowns in the reduced tridiagonal. */
    const int64_t ni = n - 2;
    /* Build the reduced tridiagonal coefficients. */
    /* Row 1 (interior index 0): */
    {
        const double h0 = h[0];
        const double h1 = h[1];
        diag_buf[0]  = (h0 + h1) * (h0 + 2.0 * h1) / h1;
        super_buf[0] = (h1 * h1 - h0 * h0) / h1;
        rb_buf[0]    = rhs[1];
    }
    /* Rows 2..n-3 (interior indices 1..ni-2): standard tridiagonal. */
    for (int64_t k = 1; k < ni - 1; ++k) {
        const int64_t i = k + 1;
        diag_buf[k]  = 2.0 * (h[i - 1] + h[i]);
        super_buf[k] = h[i];
        rb_buf[k]    = rhs[i];
    }
    /* Row n-2 (interior index ni - 1): mirror not-a-knot elimination. */
    {
        const double h_l = h[n - 3];
        const double h_r = h[n - 2];
        diag_buf[ni - 1] = (h_l + h_r) * (h_r + 2.0 * h_l) / h_l;
        /* super_buf[ni - 1] is unused (last row). */
        rb_buf[ni - 1]   = rhs[n - 2];
    }
    /* Thomas forward sweep. Sub-diagonal A[k]:
     *   k = 0: doesn't exist (top of the interior system).
     *   k = 1..ni - 2: A[k] = h[k] (i.e., h[i-1] for original row i = k + 1).
     *   k = ni - 1: corner-eliminated A_{n-2} = (h[n-3]^2 - h[n-2]^2)/h[n-3].
     */
    for (int64_t k = 1; k < ni; ++k) {
        double a_k;
        if (k == ni - 1) {
            const double h_l = h[n - 3];
            const double h_r = h[n - 2];
            a_k = (h_l * h_l - h_r * h_r) / h_l;
        } else {
            a_k = h[k];   /* h[(k+1)-1] = h[k] for original row index k+1 */
        }
        const double m = a_k / diag_buf[k - 1];
        diag_buf[k] -= m * super_buf[k - 1];
        rb_buf[k]   -= m * rb_buf[k - 1];
    }
    /* Back-substitution into M[1..n-2]. */
    M[n - 2] = rb_buf[ni - 1] / diag_buf[ni - 1];
    for (int64_t k = ni - 2; k >= 0; --k) {
        const int64_t i = k + 1;
        M[i] = (rb_buf[k] - super_buf[k] * M[i + 1]) / diag_buf[k];
    }
    /* Recover boundary M[0] and M[n-1] from the not-a-knot BC equations. */
    M[0] = ((h[0] + h[1]) * M[1] - h[0] * M[2]) / h[1];
    M[n - 1] = ((h[n - 2] + h[n - 3]) * M[n - 2] - h[n - 2] * M[n - 3]) / h[n - 3];
}

static void cubic_evaluate_row(const c4a_pp_resampler_state_t* s,
                                const double* y,
                                double* out) {
    const int64_t n = s->n_source;
    /* Layout of cs_buf: [rhs | M | diag | super] each of length n. */
    double* cs_rhs   = s->cs_buf;
    double* cs_m     = s->cs_buf + n;
    double* cs_diag  = s->cs_buf + 2 * n;
    double* cs_super = s->cs_buf + 3 * n;
    /* RHS only depends on y. */
    cs_rhs[0]     = 0.0;
    cs_rhs[n - 1] = 0.0;
    for (int64_t i = 1; i < n - 1; ++i) {
        cs_rhs[i] = 6.0 * (
            (y[i + 1] - y[i]) / s->h[i] -
            (y[i]     - y[i - 1]) / s->h[i - 1]);
    }
    solve_not_a_knot(s->h, cs_rhs, n, cs_m, cs_diag, cs_super, cs_rhs);
    /* Note: solve_not_a_knot uses cs_rhs as the reduced-RHS scratch and
     * overwrites it in place — we no longer need the original RHS once
     * the M vector is solved. */
    const int64_t n_target = s->n_target;
    for (int64_t k = 0; k < n_target; ++k) {
        if (!s->extrapolate && (s->out_of_lo[k] || s->out_of_hi[k])) {
            out[k] = s->fill_value;
            continue;
        }
        const int64_t i = s->seg_idx[k];
        const double hi = s->h[i];
        const double dx = s->target_wl[k] - s->source_wl[i];
        const double a = y[i];
        const double b = (y[i + 1] - y[i]) / hi - hi * (2.0 * cs_m[i] + cs_m[i + 1]) / 6.0;
        const double c = cs_m[i] * 0.5;
        const double d = (cs_m[i + 1] - cs_m[i]) / (6.0 * hi);
        out[k] = a + dx * (b + dx * (c + dx * d));
    }
}

static void linear_evaluate_row(const c4a_pp_resampler_state_t* s,
                                 const double* y,
                                 double* out) {
    const int64_t n_target = s->n_target;
    for (int64_t k = 0; k < n_target; ++k) {
        if (!s->extrapolate && (s->out_of_lo[k] || s->out_of_hi[k])) {
            out[k] = s->fill_value;
            continue;
        }
        const int64_t i = s->seg_idx[k];
        const double x_lo = s->source_wl[i];
        const double x_hi = s->source_wl[i + 1];
        const double y_lo = y[i];
        const double y_hi = y[i + 1];
        const double slope = (y_hi - y_lo) / (x_hi - x_lo);
        out[k] = slope * (s->target_wl[k] - x_lo) + y_lo;
    }
}

static void nearest_evaluate_row(const c4a_pp_resampler_state_t* s,
                                  const double* y,
                                  double* out) {
    /* scipy interp1d 'nearest' uses midpoints between successive source
     * wavelengths as bin boundaries (`x_bds = self.x / 2.0;
     * x_bds = x_bds[1:] + x_bds[:-1]` — division-before-addition, matters
     * for float-overflow correctness even though it doesn't affect IEEE
     * double values from finite inputs). Boundary search uses side='left'. */
    const int64_t n = s->n_source;
    const int64_t n_target = s->n_target;
    for (int64_t k = 0; k < n_target; ++k) {
        if (!s->extrapolate && (s->out_of_lo[k] || s->out_of_hi[k])) {
            out[k] = s->fill_value;
            continue;
        }
        const double q = s->target_wl[k];
        int64_t lo = 0;
        int64_t hi = n - 1;
        while (lo < hi) {
            const int64_t mid = (lo + hi) >> 1;
            const double bd = s->source_wl[mid] / 2.0 + s->source_wl[mid + 1] / 2.0;
            if (bd < q) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < 0) lo = 0;
        if (lo > n - 1) lo = n - 1;
        out[k] = y[lo];
    }
}

c4a_status_t c4a_pp_resampler_state_apply(
    const c4a_pp_resampler_state_t* state,
    const double* X,
    int64_t rows, int64_t cols,
    int64_t out_cols,
    double* out) {
    if (state == NULL || X == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (!state->fitted) {
        return C4A_ERR_NOT_FITTED;
    }
    if (rows < 0 || cols < 0 || out_cols < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (cols != state->n_raw) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    const int64_t expected_out = (state->n_target == -1) ? state->n_source
                                                          : state->n_target;
    if (out_cols != expected_out) {
        return C4A_ERR_SHAPE_MISMATCH;
    }
    if (rows == 0 || out_cols == 0) {
        return C4A_OK;
    }
    if (state->bounds_error && state->n_target != -1) {
        for (int64_t k = 0; k < state->n_target; ++k) {
            if (state->out_of_lo[k] || state->out_of_hi[k]) {
                return C4A_ERR_NUMERICAL_FAILURE;
            }
        }
    }
    /* Identity / passthrough. */
    if (state->n_target == -1) {
        for (int64_t i = 0; i < rows; ++i) {
            const double* row = X + (size_t)i * (size_t)cols;
            double* dst = out + (size_t)i * (size_t)out_cols;
            if (state->use_crop) {
                for (int64_t k = 0; k < out_cols; ++k) {
                    dst[k] = row[state->crop_idx[k]];
                }
            } else {
                memcpy(dst, row, (size_t)out_cols * sizeof(double));
            }
        }
        return C4A_OK;
    }
    double* row_scratch = NULL;
    if (state->use_crop) {
        row_scratch = (double*)malloc((size_t)state->n_source * sizeof(double));
        if (row_scratch == NULL) {
            return C4A_ERR_OUT_OF_MEMORY;
        }
    }
    for (int64_t i = 0; i < rows; ++i) {
        const double* raw = X + (size_t)i * (size_t)cols;
        const double* row;
        if (state->use_crop) {
            for (int64_t k = 0; k < state->n_source; ++k) {
                row_scratch[k] = raw[state->crop_idx[k]];
            }
            row = row_scratch;
        } else {
            row = raw;
        }
        double* dst = out + (size_t)i * (size_t)out_cols;
        switch (state->method) {
            case 0:  linear_evaluate_row(state, row, dst); break;
            case 1:  nearest_evaluate_row(state, row, dst); break;
            case 2:  cubic_evaluate_row(state, row, dst); break;
            default: /* unreachable */ break;
        }
    }
    free(row_scratch);
    return C4A_OK;
}
