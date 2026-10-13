/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Standard Normal Variate (SNV) — row-wise center + scale.
 *
 * Mean and standard deviation (ddof) over each row, as numpy.mean / numpy.std
 * over axis=1, with the nirs4all.StandardNormalVariate `std[std == 0] = 1.0`
 * guard. Agreement with numpy is within rounding (parity tolerance 1e-12),
 * not bit-for-bit: the moments are computed more accurately.
 *
 * Numerical scheme (ABI 2.20), for a finite row x of n values:
 *   - constant rows (all values equal) map to exact zeros when centred and
 *     are never scaled;
 *   - y = x * 2^-e with e = ilogb(max|x|), an exact power-of-two rescaling
 *     (|y| < 2), so no sum overflows and subnormal rows are not lost (values
 *     below 2^-1074 relative to max|x| are not representable after scaling);
 *   - the mean is a double-double m_hi + m_lo from a compensated (Neumaier)
 *     sum, the exact fma division remainder and one refinement pass, so a
 *     large offset (1e16, 1e300) does not leak into the deviations;
 *   - deviations d = (y - m_hi) - m_lo are rescaled again by 2^-ilogb(max|d|)
 *     before squaring, so squares cannot underflow;
 *   - centred and scaled output is t / sd_t, never reconstructed at the
 *     original scale; other modes rescale back and must stay finite.
 * A non-finite result for a finite row returns N4M_ERR_NUMERICAL_FAILURE.
 * Rows containing NaN/Inf propagate them through the plain formulas.
 *
 * First-order rounding bound (u = 2^-53) on every entry of a centred and
 * scaled row, reported by n4m_pp_snv_apply_bounded:
 *   B = (3 u A + n u^2 (A + |m|)) / sd_y + 8 u max|z|,
 * with A = mean|y - m|, sd_y the SD in y units and z the output row.
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

/* Neumaier compensated summation. */
typedef struct { double sum, comp; } snv_acc_t;

static void snv_acc_add(snv_acc_t* a, double v) {
    const double t = a->sum + v;
    if (fabs(a->sum) >= fabs(v)) {
        a->comp += (a->sum - t) + v;
    } else {
        a->comp += (v - t) + a->sum;
    }
    a->sum = t;
}

/* Plain two-pass formulas for rows holding NaN/Inf (unchanged contract). */
static void snv_row_nonfinite(const n4m_pp_snv_state_t* state, const double* x,
                              int64_t cols, int64_t denom_count, double* out) {
    double acc = 0.0, sq = 0.0;
    for (int64_t j = 0; j < cols; ++j) acc += x[j];
    const double mean = acc / (double)cols;
    for (int64_t j = 0; j < cols; ++j) sq += (x[j] - mean) * (x[j] - mean);
    const double sd = denom_count > 0 ? sqrt(sq / (double)denom_count) : 0.0;
    const int scale = state->with_std && sd != 0.0;
    for (int64_t j = 0; j < cols; ++j) {
        const double v = state->with_mean ? x[j] - mean : x[j];
        out[j] = scale ? v / sd : v;
    }
}

/* One finite, non-constant row. `bound` (may be NULL) receives the rounding
 * bound of the centred-and-scaled output, NaN for other modes. */
static n4m_status_t snv_row_finite(const n4m_pp_snv_state_t* state, const double* x,
                                   int64_t cols, int64_t denom_count, double* out,
                                   double* bound) {
    const double n = (double)cols;
    const double u = 0x1p-53;
    double amax = 0.0;
    for (int64_t j = 0; j < cols; ++j) amax = fmax(amax, fabs(x[j]));
    const int e = ilogb(amax);
    /* Mean as a double-double m_hi + m_lo: the remainder of a correctly
     * rounded division is exact (fma), then one refinement pass. */
    snv_acc_t s0 = {0.0, 0.0};
    for (int64_t j = 0; j < cols; ++j) snv_acc_add(&s0, ldexp(x[j], -e));
    const double m_hi = s0.sum / n;
    double m_lo = (fma(-m_hi, n, s0.sum) + s0.comp) / n;
    snv_acc_t s1 = {0.0, 0.0};
    for (int64_t j = 0; j < cols; ++j) snv_acc_add(&s1, (ldexp(x[j], -e) - m_hi) - m_lo);
    m_lo += (s1.sum + s1.comp) / n;
    double abs_dev = 0.0, dmax = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
        const double d = fabs((ldexp(x[j], -e) - m_hi) - m_lo);
        abs_dev += d;
        dmax = fmax(dmax, d);
    }
    if (bound) *bound = NAN;
    const int scale = state->with_std && denom_count > 0 && dmax > 0.0;
    if (!scale) {
        /* Centre only (or no SD available): x - mean at the original scale. */
        for (int64_t j = 0; j < cols; ++j) {
            const double v = state->with_mean
                ? ldexp((ldexp(x[j], -e) - m_hi) - m_lo, e) : x[j];
            if (!isfinite(v)) return N4M_ERR_NUMERICAL_FAILURE;
            out[j] = v;
        }
        return N4M_OK;
    }
    /* Deviations rescaled by 2^-k (k = ilogb(max|d|)) before squaring. */
    const int k = ilogb(dmax);
    snv_acc_t ss = {0.0, 0.0};
    for (int64_t j = 0; j < cols; ++j) {
        const double t = ldexp((ldexp(x[j], -e) - m_hi) - m_lo, -k);
        snv_acc_add(&ss, t * t);
    }
    const double sd_t = sqrt((ss.sum + ss.comp) / (double)denom_count);
    if (!(sd_t > 0.0) || !isfinite(sd_t)) return N4M_ERR_NUMERICAL_FAILURE;
    double zmax = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
        const double y = ldexp(x[j], -e);
        double v;
        if (state->with_mean) {
            v = ldexp((y - m_hi) - m_lo, -k) / sd_t;
        } else {
            /* x / sd = (y / sd_t) * 2^-k; may legitimately overflow. */
            v = ldexp(y / sd_t, -k);
        }
        if (!isfinite(v)) return N4M_ERR_NUMERICAL_FAILURE;
        zmax = fmax(zmax, fabs(v));
        out[j] = v;
    }
    if (bound && state->with_mean) {
        /* Mean error <= 3u A + n u^2 (A + |m|); deviation and SD roundings
         * add at most 8u |z|. A = mean|y - m|, sd_y = sd_t 2^k. */
        const double a = abs_dev / n, sd_y = ldexp(sd_t, k);
        *bound = (3.0 * u * a + n * u * u * (a + fabs(m_hi))) / sd_y + 8.0 * u * zmax;
    }
    return N4M_OK;
}

n4m_status_t n4m_pp_snv_apply_bounded(const n4m_pp_snv_state_t* state,
                                      const double* X, int64_t rows, int64_t cols,
                                      double* out, double* row_bounds) {
    if (state == NULL || X == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    const int64_t denom_count = (int64_t)cols - (int64_t)state->ddof;
    for (int64_t i = 0; i < rows; ++i) {
        const double* row_in = X + (size_t)i * (size_t)cols;
        double*       row_out = out + (size_t)i * (size_t)cols;
        double* bound = row_bounds ? row_bounds + i : NULL;
        if (bound) *bound = NAN;
        if (cols == 0) continue;
        int finite = 1;
        for (int64_t j = 0; j < cols && finite; ++j) finite = isfinite(row_in[j]);
        if (!finite) {
            snv_row_nonfinite(state, row_in, cols, denom_count, row_out);
            continue;
        }
        int constant = 1;
        for (int64_t j = 1; j < cols && constant; ++j) constant = row_in[j] == row_in[0];
        if (constant) {
            for (int64_t j = 0; j < cols; ++j) row_out[j] = state->with_mean ? 0.0 : row_in[j];
            if (bound && state->with_mean) *bound = 0.0;
            continue;
        }
        if (!state->with_mean && !state->with_std) {
            for (int64_t j = 0; j < cols; ++j) row_out[j] = row_in[j];
            continue;
        }
        const n4m_status_t st = snv_row_finite(state, row_in, cols, denom_count, row_out, bound);
        if (st != N4M_OK) return st;
    }
    return N4M_OK;
}

n4m_status_t n4m_pp_snv_apply(const n4m_pp_snv_state_t* state,
                              const double* X, int64_t rows, int64_t cols,
                              double* out) {
    return n4m_pp_snv_apply_bounded(state, X, rows, cols, out, NULL);
}
