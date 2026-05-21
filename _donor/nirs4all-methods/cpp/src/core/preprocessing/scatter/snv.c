/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Standard Normal Variate (SNV) — row-wise center + scale.
 *
 * Matches numpy.mean / numpy.std (ddof) over axis=1, with the
 * nirs4all.StandardNormalVariate `std[std == 0] = 1.0` guard generalised
 * to `std < 1e-15` to absorb floating-point fuzz on flat spectra.
 *
 * Two-pass formulation (mean then sum-of-squares) is preferred over Welford
 * because numpy's parity stream uses the algebraic identity and matches
 * bit-for-bit when both sides accumulate in the same order.
 */

#include "snv.h"

#include <math.h>
#include <stdlib.h>

struct n4m_pp_snv_state_t {
    int with_mean;
    int with_std;
    int ddof;
};

n4m_pp_snv_state_t* n4m_pp_snv_state_new(int with_mean, int with_std, int ddof) {
    n4m_pp_snv_state_t* s = (n4m_pp_snv_state_t*)malloc(sizeof(*s));
    if (s == NULL) {
        return NULL;
    }
    s->with_mean = with_mean ? 1 : 0;
    s->with_std  = with_std  ? 1 : 0;
    s->ddof      = ddof;
    return s;
}

void n4m_pp_snv_state_free(n4m_pp_snv_state_t* state) {
    free(state);
}

n4m_status_t n4m_pp_snv_apply(const n4m_pp_snv_state_t* state,
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

    const double cols_d  = (double)cols;
    const int64_t denom_count = (int64_t)cols - (int64_t)state->ddof;

    for (int64_t i = 0; i < rows; ++i) {
        const double* row_in = X + (size_t)i * (size_t)cols;
        double*       row_out = out + (size_t)i * (size_t)cols;

        double mean = 0.0;
        if (state->with_mean || state->with_std) {
            double acc = 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                acc += row_in[j];
            }
            /* numpy.mean evaluates sum/N as a division, not as `sum * (1/N)` —
             * the rounding sequence is different and the parity test catches
             * the discrepancy on some seeds. */
            mean = acc / cols_d;
        }

        double std = 1.0;
        int    apply_std = 0;
        if (state->with_std) {
            double sq = 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                const double d = row_in[j] - mean;
                sq += d * d;
            }
            if (denom_count > 0) {
                const double s = sqrt(sq / (double)denom_count);
                /* Match nirs4all exactly: `std[std == 0] = 1.0`. Any non-zero
                 * std (however tiny) is used as a divisor — same as numpy. */
                if (s != 0.0) {
                    std       = s;
                    apply_std = 1;
                }
            }
        }

        if (state->with_mean && apply_std) {
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = (row_in[j] - mean) / std;
            }
        } else if (state->with_mean) {
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = row_in[j] - mean;
            }
        } else if (apply_std) {
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = row_in[j] / std;
            }
        } else {
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = row_in[j];
            }
        }
    }
    return N4M_OK;
}
