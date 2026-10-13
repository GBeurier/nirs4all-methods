/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Standard Normal Variate (SNV) — row-wise center + scale.
 *
 * Matches numpy.mean / numpy.std (ddof) over axis=1, with the
 * nirs4all.StandardNormalVariate `std[std == 0] = 1.0` guard.
 *
 * Two-pass formulation (mean then sum-of-squares) is preferred over Welford
 * because numpy's parity stream uses the algebraic identity and matches
 * bit-for-bit when both sides accumulate in the same order.
 *
 * Numerical guards (Methods 1.3.7): rows with non-finite values propagate them
 * through the plain two-pass formulas as before; for finite rows, constant
 * rows map to exact zeros when centred (numpy would standardize the rounding
 * residue of its mean, e.g. eleven 0.1 -> eleven 1.0); rows whose sums or
 * squared deviations overflow or underflow are recomputed on rescaled values;
 * a non-finite result returns N4M_ERR_NUMERICAL_FAILURE.
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

/* Row-wise mean and standard deviation that cannot overflow or underflow.
 * `x` is finite. The fast path keeps numpy's two-pass order (sum/N, then
 * sum of squared deviations) so ordinary rows stay bit-identical to numpy;
 * rows whose sums overflow or whose squared deviations leave the normal
 * range are recomputed on values divided by max|x| (resp. max|x - mean|).
 * Returns 0 on success, -1 when the result is not finite. */
static int snv_row_moments(const double* x, int64_t cols, int64_t denom_count,
                           int need_std, double* mean_out, double* std_out) {
    const double n = (double)cols;
    double acc = 0.0;
    for (int64_t j = 0; j < cols; ++j) acc += x[j];
    double mean = acc / n;
    if (!isfinite(mean)) {
        double scale = 0.0;
        for (int64_t j = 0; j < cols; ++j) scale = fmax(scale, fabs(x[j]));
        double scaled = 0.0;
        for (int64_t j = 0; j < cols; ++j) scaled += x[j] / scale;
        mean = (scaled / n) * scale;
    }
    *mean_out = mean;
    *std_out = 0.0;
    if (!need_std || denom_count <= 0) return isfinite(mean) ? 0 : -1;
    /* Squares below 2^-900 or above DBL_MAX lose the deviations; rescale. */
    double sq = 0.0;
    int rescale = 0;
    for (int64_t j = 0; j < cols; ++j) {
        const double d = x[j] - mean;
        if (!isfinite(d)) { rescale = 1; break; }
        sq += d * d;
    }
    /* The row is not constant, so a zero sum means every square underflowed. */
    if (!rescale && (!isfinite(sq) || sq < 0x1p-900)) rescale = 1;
    if (!rescale) {
        *std_out = sqrt(sq / (double)denom_count);
        return isfinite(*std_out) ? 0 : -1;
    }
    double scale = 0.0;
    for (int64_t j = 0; j < cols; ++j)
        scale = fmax(scale, fabs(x[j] * 0.5 - mean * 0.5));
    if (scale == 0.0) return 0;
    double ssq = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
        const double t = (x[j] * 0.5 - mean * 0.5) / scale;
        ssq += t * t;
    }
    /* std = 2 * scale * sqrt(ssq / denom); scale <= DBL_MAX / 2 keeps it finite
     * unless the true deviation itself exceeds DBL_MAX. */
    *std_out = 2.0 * scale * sqrt(ssq / (double)denom_count);
    return isfinite(*std_out) ? 0 : -1;
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

    const int64_t denom_count = (int64_t)cols - (int64_t)state->ddof;

    for (int64_t i = 0; i < rows; ++i) {
        const double* row_in = X + (size_t)i * (size_t)cols;
        double*       row_out = out + (size_t)i * (size_t)cols;

        int finite = 1;
        for (int64_t j = 0; j < cols && finite; ++j) finite = isfinite(row_in[j]);
        if (!finite) {
            /* Unchanged contract: non-finite values propagate (numpy-like). */
            double acc = 0.0, sq = 0.0;
            for (int64_t j = 0; j < cols; ++j) acc += row_in[j];
            const double mean = acc / (double)cols;
            for (int64_t j = 0; j < cols; ++j) sq += (row_in[j] - mean) * (row_in[j] - mean);
            const double sd = denom_count > 0 ? sqrt(sq / (double)denom_count) : 0.0;
            const int scale = state->with_std && sd != 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                double v = state->with_mean ? row_in[j] - mean : row_in[j];
                row_out[j] = scale ? v / sd : v;
            }
            continue;
        }
        /* Constant rows (every value bitwise equal up to the sign of zero)
         * have no shape: the mean is the value itself and the SD is zero, so
         * centring gives exact zeros and no scaling is applied. Rounding
         * residue of a computed mean is never standardized into signal. */
        int constant = 1;
        for (int64_t j = 1; j < cols && constant; ++j) {
            constant = row_in[j] == row_in[0];
        }
        if (constant) {
            for (int64_t j = 0; j < cols; ++j) {
                row_out[j] = state->with_mean ? 0.0 : row_in[j];
            }
            continue;
        }

        double mean = 0.0, std = 0.0;
        if (state->with_mean || state->with_std) {
            if (snv_row_moments(row_in, cols, denom_count, state->with_std, &mean, &std) != 0) {
                return N4M_ERR_NUMERICAL_FAILURE;
            }
        }
        /* nirs4all `std[std == 0] = 1.0`: a zero SD leaves the row unscaled. */
        const int apply_std = state->with_std && std != 0.0;

        for (int64_t j = 0; j < cols; ++j) {
            double v = row_in[j];
            if (state->with_mean && apply_std) {
                /* (x - mean) / std without overflow of the difference. */
                const double d = v - mean;
                v = isfinite(d) ? d / std : 2.0 * ((v * 0.5 - mean * 0.5) / std);
            } else if (state->with_mean) {
                v = v - mean;
            } else if (apply_std) {
                v = v / std;
            }
            if (!isfinite(v)) return N4M_ERR_NUMERICAL_FAILURE;
            row_out[j] = v;
        }
    }
    return N4M_OK;
}
