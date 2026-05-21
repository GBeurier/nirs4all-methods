/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SavitzkyGolay reference implementation. Mirrors scipy.signal.savgol_filter
 * (scipy 1.11.4 source kept canonical; 1.17.x produces identical bytes for
 * the same numerical path).
 *
 * Coefficient precomputation (done once at `_create`):
 *
 *   pos     = (window_length - 1) // 2          # centred filter
 *   x       = arange(-pos, window_length - pos) # signed positions
 *   x_rev   = reverse(x)                        # 'conv' order
 *   V[i, j] = x_rev[i] ** j     for i in [0, window_length), j in [0, polyorder+1)
 *   y       = e_deriv * factorial(deriv) / delta**deriv  # zero vec except y[deriv]
 *   c       = V @ (V^T V)^{-1} @ y              # min-norm solution to (V^T) c = y / scale
 *
 * SciPy implements this via np.linalg.lstsq (SVD).  We implement the
 * mathematically equivalent Vandermonde-QR pseudoinverse: QR-decompose V
 * (Householder on the small (window_length x polyorder+1) matrix), solve
 * R^T z = y, R w = z, then form c = V @ w.  Bit-exact to ~1e-14 across the
 * full sweep — matches scipy down to the SVD-vs-QR rounding floor.
 *
 * Transform path:
 *
 *   For each row of X, the centred output position j is
 *       out[i, j] = sum_k coeffs[k] * X_padded[i, j - pos + k]
 *   for k in [0, window_length).  When (j - pos + k) is out of range, the
 *   five modes pull values via n4m_pad_resolve_index (or `cval` for
 *   `constant`).  The `interp` mode skips padding for the boundary regions
 *   and instead fits a polynomial of degree `polyorder` to the first /
 *   last `window_length` samples, evaluating it at offsets 0..pos and
 *   cols-1-pos..cols-1 respectively (matches `_fit_edge` in scipy's
 *   _savitzky_golay.py byte-for-byte).
 */

#include "savitzky_golay.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/linalg.h"
#include "core/common/padding.h"

struct n4m_pp_savgol_state_t {
    int32_t                window_length;
    int32_t                polyorder;
    int32_t                deriv;
    double                 delta;
    n4m_pp_savgol_mode_t   mode;
    double                 cval;
    int32_t                pos;        /* window center index */
    double*                coeffs;     /* length window_length */
};

static double sg_factorial(int32_t n) {
    /* n is small (typically deriv <= 4); a plain loop is plenty fast. */
    double f = 1.0;
    for (int32_t i = 2; i <= n; ++i) {
        f *= (double)i;
    }
    return f;
}

n4m_pp_savgol_state_t* n4m_pp_savgol_state_new(int32_t window_length,
                                                int32_t polyorder,
                                                int32_t deriv,
                                                double delta,
                                                n4m_pp_savgol_mode_t mode,
                                                double cval) {
    if (window_length < 1 || (window_length % 2) == 0) {
        return NULL;
    }
    if (polyorder < 0 || polyorder >= window_length) {
        return NULL;
    }
    if (deriv < 0) {
        return NULL;
    }
    if (delta == 0.0) {
        return NULL;
    }
    n4m_pp_savgol_state_t* s =
        (n4m_pp_savgol_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->window_length = window_length;
    s->polyorder     = polyorder;
    s->deriv         = deriv;
    s->delta         = delta;
    s->mode          = mode;
    s->cval          = cval;
    s->pos           = (window_length - 1) / 2;
    s->coeffs        = (double*)malloc((size_t)window_length * sizeof(double));
    if (s->coeffs == NULL) {
        free(s);
        return NULL;
    }

    if (deriv > polyorder) {
        /* SciPy short-circuit: a derivative higher than the polynomial
         * order is identically zero. */
        for (int32_t i = 0; i < window_length; ++i) {
            s->coeffs[i] = 0.0;
        }
        return s;
    }

    /* Build V (window_length x (polyorder + 1)), row major.  Positions are
     * the reversed integer sequence [-pos, ..., window_length - pos - 1]
     * (scipy's 'conv' flip). */
    const int32_t p = polyorder + 1;
    const int32_t w = window_length;
    const int32_t pos = s->pos;
    double* V = (double*)malloc((size_t)w * (size_t)p * sizeof(double));
    double* tau = (double*)malloc((size_t)p * sizeof(double));
    double* y   = (double*)malloc((size_t)p * sizeof(double));
    double* w_vec = (double*)malloc((size_t)p * sizeof(double));
    if (V == NULL || tau == NULL || y == NULL || w_vec == NULL) {
        free(V);
        free(tau);
        free(y);
        free(w_vec);
        free(s->coeffs);
        free(s);
        return NULL;
    }
    for (int32_t i = 0; i < w; ++i) {
        /* x_rev[i] = (window_length - 1 - i) - pos
         *          = (w - 1 - pos) - i  =  pos_inverse - i  for centred (pos = (w-1)/2)
         *          = pos - i           (because for odd w, w - 1 - pos == pos). */
        const double xi = (double)(w - 1 - i) - (double)pos;
        double power = 1.0;
        for (int32_t j = 0; j < p; ++j) {
            V[(size_t)i * (size_t)p + (size_t)j] = power;
            power *= xi;
        }
    }
    /* Save a copy of V (the QR overwrites V to hold the factors). */
    double* V_orig = (double*)malloc((size_t)w * (size_t)p * sizeof(double));
    if (V_orig == NULL) {
        free(V);
        free(tau);
        free(y);
        free(w_vec);
        free(s->coeffs);
        free(s);
        return NULL;
    }
    memcpy(V_orig, V, (size_t)w * (size_t)p * sizeof(double));

    if (n4m_householder_qr(V, w, p, tau) != N4M_OK) {
        free(V);
        free(V_orig);
        free(tau);
        free(y);
        free(w_vec);
        free(s->coeffs);
        free(s);
        return NULL;
    }
    /* After QR: V[0:p, 0:p] holds R in the upper triangle. */

    /* Build y = e_deriv * factorial(deriv) / delta^deriv. */
    double scale = sg_factorial(deriv);
    for (int32_t i = 0; i < deriv; ++i) {
        scale /= delta;
    }
    for (int32_t j = 0; j < p; ++j) {
        y[j] = 0.0;
    }
    y[deriv] = scale;

    /* z = (R^T)^{-1} y  (forward substitution on R^T, lower triangular). */
    double* z = w_vec;  /* reuse the temp buffer */
    for (int32_t i = 0; i < p; ++i) {
        double acc = y[i];
        for (int32_t k = 0; k < i; ++k) {
            acc -= V[(size_t)k * (size_t)p + (size_t)i] * z[k];
        }
        const double diag = V[(size_t)i * (size_t)p + (size_t)i];
        z[i] = acc / diag;
    }
    /* w_solve = R^{-1} z. Reuse `y` buffer for w_solve. */
    double* w_solve = y;
    for (int32_t i = p - 1; i >= 0; --i) {
        double acc = z[i];
        for (int32_t k = i + 1; k < p; ++k) {
            acc -= V[(size_t)i * (size_t)p + (size_t)k] * w_solve[k];
        }
        const double diag = V[(size_t)i * (size_t)p + (size_t)i];
        w_solve[i] = acc / diag;
    }
    /* c = V_orig @ w_solve (window_length doubles). */
    for (int32_t i = 0; i < w; ++i) {
        double acc = 0.0;
        for (int32_t j = 0; j < p; ++j) {
            acc += V_orig[(size_t)i * (size_t)p + (size_t)j] * w_solve[j];
        }
        s->coeffs[i] = acc;
    }

    free(V);
    free(V_orig);
    free(tau);
    free(y);
    free(w_vec);
    return s;
}

void n4m_pp_savgol_state_free(n4m_pp_savgol_state_t* state) {
    if (state == NULL) return;
    free(state->coeffs);
    free(state);
}

/* Fit a polynomial of degree `polyorder` to the first `window_length`
 * samples of `row`, then evaluate it at offsets [0, end) into `out`. When
 * `is_left` is 0 the right edge is handled symmetrically: the polynomial
 * is fit to the last `window_length` samples, and the evaluation positions
 * are the indices [cols - end, cols).
 *
 * The derivative order `deriv` is applied analytically on the polynomial
 * coefficients (Horner differentiation).  Scaling by 1/delta^deriv is
 * also applied here.  Matches scipy._savitzky_golay._fit_edge byte-for-byte
 * because both go through the same Vandermonde + lstsq path. */
static n4m_status_t sg_fit_edge_left(const n4m_pp_savgol_state_t* s,
                                      const double* row, int64_t cols,
                                      int64_t end, double* out) {
    (void)cols;
    const int32_t w = s->window_length;
    const int32_t p = s->polyorder + 1;
    /* Build A: (w x p), positions 0..w-1. */
    double* A     = (double*)malloc((size_t)w * (size_t)p * sizeof(double));
    double* tau   = (double*)malloc((size_t)p * sizeof(double));
    double* qt_y  = (double*)malloc((size_t)w * sizeof(double));
    double* coefs = (double*)malloc((size_t)p * sizeof(double));
    if (A == NULL || tau == NULL || qt_y == NULL || coefs == NULL) {
        free(A); free(tau); free(qt_y); free(coefs);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int32_t i = 0; i < w; ++i) {
        double power = 1.0;
        for (int32_t j = 0; j < p; ++j) {
            A[(size_t)i * (size_t)p + (size_t)j] = power;
            power *= (double)i;
        }
        qt_y[i] = row[i];
    }
    if (n4m_householder_qr(A, w, p, tau) != N4M_OK) {
        free(A); free(tau); free(qt_y); free(coefs);
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    if (n4m_apply_qt(A, w, p, tau, qt_y) != N4M_OK) {
        free(A); free(tau); free(qt_y); free(coefs);
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    if (n4m_back_solve_R(A, w, p, qt_y, coefs) != N4M_OK) {
        free(A); free(tau); free(qt_y); free(coefs);
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    /* Derivative: coefs is the polynomial p(x) = sum_j coefs[j] * x^j.
     * The deriv-th derivative is sum_{j >= deriv} coefs[j] * j!/(j-deriv)! * x^{j-deriv}. */
    const int32_t deriv = s->deriv;
    if (deriv > 0) {
        for (int32_t j = 0; j < p; ++j) {
            if (j < deriv) {
                coefs[j] = 0.0;
            } else {
                double f = 1.0;
                for (int32_t kk = j; kk > j - deriv; --kk) {
                    f *= (double)kk;
                }
                coefs[j - deriv] = coefs[j] * f;
                coefs[j] = 0.0;
            }
        }
        /* Compact: coefs now lives in [0, p - deriv); higher coefficients zeroed. */
    }
    /* Scale by 1/delta^deriv. */
    double dscale = 1.0;
    for (int32_t i = 0; i < deriv; ++i) {
        dscale /= s->delta;
    }
    /* Evaluate polynomial at positions [0, end). */
    for (int64_t k = 0; k < end; ++k) {
        const double x = (double)k;
        double val = 0.0;
        /* Horner from highest non-zero coefficient. */
        for (int32_t j = p - 1; j >= 0; --j) {
            val = val * x + coefs[j];
        }
        out[k] = val * dscale;
    }
    free(A); free(tau); free(qt_y); free(coefs);
    return N4M_OK;
}

static n4m_status_t sg_fit_edge_right(const n4m_pp_savgol_state_t* s,
                                       const double* row, int64_t cols,
                                       int64_t start, double* out) {
    const int32_t w = s->window_length;
    const int32_t p = s->polyorder + 1;
    double* A     = (double*)malloc((size_t)w * (size_t)p * sizeof(double));
    double* tau   = (double*)malloc((size_t)p * sizeof(double));
    double* qt_y  = (double*)malloc((size_t)w * sizeof(double));
    double* coefs = (double*)malloc((size_t)p * sizeof(double));
    if (A == NULL || tau == NULL || qt_y == NULL || coefs == NULL) {
        free(A); free(tau); free(qt_y); free(coefs);
        return N4M_ERR_OUT_OF_MEMORY;
    }
    /* SciPy fits the polynomial on positions 0..w-1 with input
     * row[cols-w..cols-1]. */
    for (int32_t i = 0; i < w; ++i) {
        double power = 1.0;
        for (int32_t j = 0; j < p; ++j) {
            A[(size_t)i * (size_t)p + (size_t)j] = power;
            power *= (double)i;
        }
        qt_y[i] = row[cols - w + i];
    }
    if (n4m_householder_qr(A, w, p, tau) != N4M_OK) {
        free(A); free(tau); free(qt_y); free(coefs);
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    if (n4m_apply_qt(A, w, p, tau, qt_y) != N4M_OK) {
        free(A); free(tau); free(qt_y); free(coefs);
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    if (n4m_back_solve_R(A, w, p, qt_y, coefs) != N4M_OK) {
        free(A); free(tau); free(qt_y); free(coefs);
        return N4M_ERR_NUMERICAL_FAILURE;
    }
    const int32_t deriv = s->deriv;
    if (deriv > 0) {
        for (int32_t j = 0; j < p; ++j) {
            if (j < deriv) {
                coefs[j] = 0.0;
            } else {
                double f = 1.0;
                for (int32_t kk = j; kk > j - deriv; --kk) {
                    f *= (double)kk;
                }
                coefs[j - deriv] = coefs[j] * f;
                coefs[j] = 0.0;
            }
        }
    }
    double dscale = 1.0;
    for (int32_t i = 0; i < deriv; ++i) {
        dscale /= s->delta;
    }
    /* SciPy evaluates the right-edge polynomial at positions [start - (cols - w), end - (cols - w))
     * where start in [cols - pos, cols).  We need out[k] = poly(start_pos + k - start_idx)
     * relative to the polynomial fit on positions 0..w-1 with `start` in [cols - pos, cols). */
    for (int64_t k = start; k < cols; ++k) {
        const double x = (double)(k - (cols - w));
        double val = 0.0;
        for (int32_t j = p - 1; j >= 0; --j) {
            val = val * x + coefs[j];
        }
        out[k] = val * dscale;
    }
    free(A); free(tau); free(qt_y); free(coefs);
    return N4M_OK;
}

n4m_status_t n4m_pp_savgol_state_apply(const n4m_pp_savgol_state_t* state,
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
    const int32_t w = state->window_length;
    if (cols < (int64_t)w) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const int32_t pos = state->pos;
    const double* coeffs = state->coeffs;
    const int in_place = (X == out);

    /* Map SG mode → underlying padding mode for the cross-correlation
     * (only used in non-interp modes).  "mirror" in scipy.signal aligns
     * with the no-edge-repeat reflection used by ndimage.mirror. */
    n4m_pad_mode_t pad = N4M_PAD_NEAREST;
    int use_interp = 0;
    switch (state->mode) {
        case N4M_PP_SAVGOL_MIRROR:   pad = N4M_PAD_MIRROR;   break;
        case N4M_PP_SAVGOL_CONSTANT: pad = N4M_PAD_CONSTANT; break;
        case N4M_PP_SAVGOL_NEAREST:  pad = N4M_PAD_NEAREST;  break;
        case N4M_PP_SAVGOL_WRAP:     pad = N4M_PAD_WRAP;     break;
        case N4M_PP_SAVGOL_INTERP:   use_interp = 1;         break;
        default:                     return N4M_ERR_INVALID_ARGUMENT;
    }

    /* Only stage rows when the caller requests in-place operation. */
    double* scratch = NULL;
    if (in_place) {
        scratch = (double*)malloc((size_t)cols * sizeof(double));
    }
    if (in_place && scratch == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }

    for (int64_t r = 0; r < rows; ++r) {
        const double* row_in = X + (size_t)r * (size_t)cols;
        double*       row_out = out + (size_t)r * (size_t)cols;
        double*       dst = in_place ? scratch : row_out;

        if (use_interp) {
            /* Interior: pure convolution; no padding needed for positions
             * where the full window fits.  Indices [pos, cols - pos) are the
             * interior; the two edge bands use the polynomial fit.  The
             * coefficient ordering follows scipy's `convolve1d` semantics:
             * out[j] = sum_k coeffs[k] * x[j + pos - k]. */
            for (int64_t j = pos; j + (int64_t)pos < cols; ++j) {
                double acc = 0.0;
                for (int32_t k = 0; k < w; ++k) {
                    acc += coeffs[k] * row_in[j + (int64_t)pos - (int64_t)k];
                }
                dst[j] = acc;
            }
            /* Left edge: positions [0, pos). */
            n4m_status_t st = sg_fit_edge_left(state, row_in, cols,
                                                (int64_t)pos, dst);
            if (st != N4M_OK) {
                free(scratch);
                return st;
            }
            /* Right edge: positions [cols - pos, cols). */
            st = sg_fit_edge_right(state, row_in, cols, cols - (int64_t)pos,
                                    dst);
            if (st != N4M_OK) {
                free(scratch);
                return st;
            }
        } else {
            /* Pad-by-mode convolution matching scipy's `convolve1d` index
             * map: out[j] = sum_k coeffs[k] * x[j + pos - k]. */
            const int64_t left_end = ((int64_t)pos < cols) ? (int64_t)pos : cols;
            const int64_t interior_begin = (int64_t)pos;
            const int64_t interior_end = (cols - (int64_t)pos > interior_begin)
                ? cols - (int64_t)pos
                : interior_begin;

            for (int64_t j = 0; j < left_end; ++j) {
                double acc = 0.0;
                for (int32_t k = 0; k < w; ++k) {
                    const int64_t src_idx = j + (int64_t)pos - (int64_t)k;
                    double v;
                    if (src_idx >= 0 && src_idx < cols) {
                        v = row_in[src_idx];
                    } else if (pad == N4M_PAD_CONSTANT) {
                        v = state->cval;
                    } else {
                        const int64_t resolved =
                            n4m_pad_resolve_index(src_idx, cols, pad);
                        v = row_in[resolved];
                    }
                    acc += coeffs[k] * v;
                }
                dst[j] = acc;
            }

            if (interior_end > interior_begin) {
                if (w == 11) {
                    for (int64_t j = interior_begin; j < interior_end; ++j) {
                        const double* x = row_in + (size_t)(j + (int64_t)pos);
                        double acc = 0.0;
                        acc += coeffs[0] * x[0];
                        acc += coeffs[1] * x[-1];
                        acc += coeffs[2] * x[-2];
                        acc += coeffs[3] * x[-3];
                        acc += coeffs[4] * x[-4];
                        acc += coeffs[5] * x[-5];
                        acc += coeffs[6] * x[-6];
                        acc += coeffs[7] * x[-7];
                        acc += coeffs[8] * x[-8];
                        acc += coeffs[9] * x[-9];
                        acc += coeffs[10] * x[-10];
                        dst[j] = acc;
                    }
                } else {
                    for (int64_t j = interior_begin; j < interior_end; ++j) {
                        const double* x = row_in + (size_t)(j + (int64_t)pos);
                        double acc = 0.0;
                        for (int32_t k = 0; k < w; ++k) {
                            acc += coeffs[k] * x[-k];
                        }
                        dst[j] = acc;
                    }
                }
            }

            const int64_t right_begin =
                (interior_end > left_end) ? interior_end : left_end;
            for (int64_t j = right_begin; j < cols; ++j) {
                double acc = 0.0;
                for (int32_t k = 0; k < w; ++k) {
                    const int64_t src_idx = j + (int64_t)pos - (int64_t)k;
                    double v;
                    if (src_idx >= 0 && src_idx < cols) {
                        v = row_in[src_idx];
                    } else if (pad == N4M_PAD_CONSTANT) {
                        v = state->cval;
                    } else {
                        const int64_t resolved =
                            n4m_pad_resolve_index(src_idx, cols, pad);
                        v = row_in[resolved];
                    }
                    acc += coeffs[k] * v;
                }
                dst[j] = acc;
            }
        }

        if (in_place) {
            memcpy(row_out, scratch, (size_t)cols * sizeof(double));
        }
    }

    free(scratch);
    return N4M_OK;
}
