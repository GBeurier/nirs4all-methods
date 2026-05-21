/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Local Standard Normal Variate (LSNV) reference implementation.
 *
 * The Python reference computes:
 *   Xp     = np.pad(X, ((0,0), (half, half)), mode=...)
 *   csum   = np.pad(np.cumsum(Xp, axis=1), ((0,0), (1,0)), mode='constant')
 *   mov_mean  = (csum[:, w:] - csum[:, :-w]) / w
 *   Xp2    = Xp * Xp
 *   csum2  = np.pad(np.cumsum(Xp2, axis=1), ...)
 *   mov_mean2 = (csum2[:, w:] - csum2[:, :-w]) / w
 *   mov_var = max(mov_mean2 - mov_mean^2, 0)
 *   mov_std = sqrt(mov_var); mov_std[mov_std == 0] = 1.0
 *   X_norm  = (X - mov_mean) / mov_std
 *
 * The C engine mirrors this row-by-row using `m + 2*half` scratch buffers
 * for the padded row, the cumsum of the padded row, and the cumsum of its
 * square. Each output position j (0 <= j < cols) uses
 *   mov_sum  = csum[j + w] - csum[j]
 *   mov_sq   = csum2[j + w] - csum2[j]
 * which matches NumPy's slicing exactly because indexing into the
 * one-step-shifted cumsum gives the same per-element accumulation order.
 */

#include "local_snv.h"

#include <math.h>
#include <stdlib.h>

struct n4m_pp_lsnv_state_t {
    int32_t                 window;
    n4m_pp_lsnv_pad_mode_t  pad_mode;
    double                  constant_value;
};

n4m_pp_lsnv_state_t* n4m_pp_lsnv_state_new(int32_t window,
                                           n4m_pp_lsnv_pad_mode_t pad_mode,
                                           double constant_value) {
    if (window < 3 || (window % 2) == 0) {
        return NULL;
    }
    if (pad_mode != N4M_PP_LSNV_PAD_REFLECT &&
        pad_mode != N4M_PP_LSNV_PAD_EDGE &&
        pad_mode != N4M_PP_LSNV_PAD_CONSTANT) {
        return NULL;
    }
    n4m_pp_lsnv_state_t* s = (n4m_pp_lsnv_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->window         = window;
    s->pad_mode       = pad_mode;
    s->constant_value = constant_value;
    return s;
}

void n4m_pp_lsnv_state_free(n4m_pp_lsnv_state_t* state) {
    free(state);
}

/* Reflect padding for index k in the padded row, where the unpadded row
 * spans [half, half + m). NumPy's "reflect" mode mirrors WITHOUT repeating
 * the edge element, and iterates if the pad width exceeds m-1. The
 * effective period is 2*(m-1) for m > 1; for m == 1 the only value is
 * row[0].
 *
 * Bringing the (possibly negative) offset `off = k - half` into the canonical
 * [0, m) range:
 *   - take `off mod period` (Euclidean, always positive)
 *   - if the result is >= m, reflect it back via `period - off_mod` */
static inline double lsnv_reflect(const double* row, int64_t m, int64_t k,
                                  int64_t half) {
    if (m == 1) {
        return row[0];
    }
    const int64_t period = 2 * (m - 1);
    int64_t off = k - half;
    /* Positive remainder modulo `period`. */
    off = ((off % period) + period) % period;
    if (off >= m) {
        off = period - off;
    }
    return row[off];
}

static inline double lsnv_edge(const double* row, int64_t m, int64_t k,
                               int64_t half) {
    if (k < half) {
        return row[0];
    }
    if (k >= half + m) {
        return row[m - 1];
    }
    return row[k - half];
}

n4m_status_t n4m_pp_lsnv_apply(const n4m_pp_lsnv_state_t* state,
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
    const int64_t w     = (int64_t)state->window;
    const int64_t half  = w / 2;
    if (cols < 1) {
        return N4M_OK;
    }
    const int64_t padded = cols + 2 * half;
    /* Scratch:
     *   padded_row:       padded doubles
     *   cs0:              padded + 1 doubles (cumsum prepended with 0)
     *   cs1:              padded + 1 doubles (cumsum of x^2 prepended with 0)
     */
    const size_t scratch_n = (size_t)padded + (size_t)(padded + 1) * 2;
    double* scratch = (double*)malloc(scratch_n * sizeof(double));
    if (scratch == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    double* pad_row = scratch;
    double* cs0     = scratch + padded;
    double* cs1     = scratch + padded + (padded + 1);

    const double w_d = (double)w;

    for (int64_t i = 0; i < rows; ++i) {
        const double* row_in  = X + (size_t)i * (size_t)cols;
        double*       row_out = out + (size_t)i * (size_t)cols;

        /* Build the padded row according to the requested boundary mode. */
        switch (state->pad_mode) {
            case N4M_PP_LSNV_PAD_REFLECT:
                for (int64_t k = 0; k < padded; ++k) {
                    pad_row[k] = lsnv_reflect(row_in, cols, k, half);
                }
                break;
            case N4M_PP_LSNV_PAD_EDGE:
                for (int64_t k = 0; k < padded; ++k) {
                    pad_row[k] = lsnv_edge(row_in, cols, k, half);
                }
                break;
            case N4M_PP_LSNV_PAD_CONSTANT:
            default:
                for (int64_t k = 0; k < half; ++k) {
                    pad_row[k] = state->constant_value;
                }
                for (int64_t k = 0; k < cols; ++k) {
                    pad_row[half + k] = row_in[k];
                }
                for (int64_t k = 0; k < half; ++k) {
                    pad_row[half + cols + k] = state->constant_value;
                }
                break;
        }

        /* Cumulative sums with a leading zero element so a window of width w
         * starting at position j has sum cs0[j + w] - cs0[j]. */
        cs0[0] = 0.0;
        cs1[0] = 0.0;
        for (int64_t k = 0; k < padded; ++k) {
            const double v = pad_row[k];
            cs0[k + 1] = cs0[k] + v;
            cs1[k + 1] = cs1[k] + v * v;
        }

        for (int64_t j = 0; j < cols; ++j) {
            /* Replicate numpy's `(csum[w:] - csum[:-w]) / w` element-wise:
             * the division must be done per element, not via a precomputed
             * reciprocal multiplied in — those differ in IEEE 754 rounding
             * on certain inputs and the parity test catches that. */
            const double mov_mean  = (cs0[j + w] - cs0[j]) / w_d;
            const double mov_mean2 = (cs1[j + w] - cs1[j]) / w_d;
            double mov_var = mov_mean2 - mov_mean * mov_mean;
            if (mov_var < 0.0) {
                mov_var = 0.0;
            }
            double mov_std = sqrt(mov_var);
            if (mov_std == 0.0) {
                mov_std = 1.0;
            }
            row_out[j] = (row_in[j] - mov_mean) / mov_std;
        }
    }

    free(scratch);
    return N4M_OK;
}
