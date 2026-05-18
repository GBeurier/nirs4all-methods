/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Filter banks + single/multi-level DWT and IDWT kernels for the Phase 6
 * wavelets operators.  Coefficients are vendored from PyWavelets 1.6.0;
 * algorithms mirror ``pywt.dwt`` / ``pywt.wavedec`` / ``pywt.idwt`` /
 * ``pywt.waverec`` for the {haar, db4, sym4, coif1} families and the
 * {periodization, symmetric, zero} boundary modes.
 *
 * Output-length conventions:
 *
 *   periodization : ceil(n / 2)
 *                    (PyWavelets pads odd-length inputs internally with one
 *                    extra copy of the trailing sample before periodising.
 *                    We do the same: ``n_eff = n + (n & 1)``.)
 *   symmetric     : floor((n + L - 1) / 2)
 *   zero          : same as symmetric (the extension shape changes but the
 *                    output length is identical).
 *
 * Single-level DWT — closed form (no scratch allocation):
 *
 *   periodization: out[k] = sum_{i=0}^{L-1} h_rev[i] * x_eff[(2*k + i - (L/2 - 1)) mod n_eff]
 *                  where x_eff = x for even n, and x with x[n-1] appended for odd n.
 *   symmetric:    extend x by L-1 with edge-repeat reflection on each side,
 *                  then out[k] = sum_{i=0}^{L-1} h[i] * x_ext[2*k + 2*(L-1) - i - 1]
 *                  (equivalent to ``np.convolve(x_ext, h)[L::2][:out_len]``).
 *   zero:         identical to symmetric but with zero padding instead of
 *                  edge-repeat reflection.  Reduces to
 *                  ``np.convolve(x, h)[1::2][:out_len]``.
 *
 * IDWT (inverse):
 *
 *   Upsample cA / cD by inserting zeros at odd indices (2*m taps); convolve
 *   with the reconstruction low / high filters; sum.  The full convolution
 *   has length 2*m + L - 1.  For symmetric / zero modes we trim ``[L-2 :
 *   L-2 + n_out]``.  For periodization we fold the convolution modulo 2*m
 *   and roll by ``L/2 - 1`` before taking the first ``n_out`` samples.
 */

#include "wavelet_kernels.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------- Filter banks ------------------------------ */

static const double k_haar_dec_lo[2] = {
     0.7071067811865476,  0.7071067811865476
};
static const double k_haar_dec_hi[2] = {
    -0.7071067811865476,  0.7071067811865476
};
static const double k_haar_rec_lo[2] = {
     0.7071067811865476,  0.7071067811865476
};
static const double k_haar_rec_hi[2] = {
     0.7071067811865476, -0.7071067811865476
};

/* db4 (Daubechies-4, order 8). Vendored from PyWavelets 1.6.0. */
static const double k_db4_dec_lo[8] = {
    -0.010597401785069032,  0.0328830116668852,
     0.030841381835560764, -0.18703481171909309,
    -0.027983769416859854,  0.6308807679298589,
     0.7148465705529157,    0.2303778133088965
};
static const double k_db4_dec_hi[8] = {
    -0.2303778133088965,    0.7148465705529157,
    -0.6308807679298589,   -0.027983769416859854,
     0.18703481171909309,   0.030841381835560764,
    -0.0328830116668852,   -0.010597401785069032
};
static const double k_db4_rec_lo[8] = {
     0.2303778133088965,    0.7148465705529157,
     0.6308807679298589,   -0.027983769416859854,
    -0.18703481171909309,   0.030841381835560764,
     0.0328830116668852,   -0.010597401785069032
};
static const double k_db4_rec_hi[8] = {
    -0.010597401785069032, -0.0328830116668852,
     0.030841381835560764,  0.18703481171909309,
    -0.027983769416859854, -0.6308807679298589,
     0.7148465705529157,   -0.2303778133088965
};

/* sym4 (Symlet-4, order 8).  Vendored from PyWavelets 1.6.0. */
static const double k_sym4_dec_lo[8] = {
    -0.07576571478927333,  -0.02963552764599851,
     0.49761866763201545,   0.8037387518059161,
     0.29785779560527736,  -0.09921954357684722,
    -0.012603967262037833,  0.0322231006040427
};
static const double k_sym4_dec_hi[8] = {
    -0.0322231006040427,   -0.012603967262037833,
     0.09921954357684722,   0.29785779560527736,
    -0.8037387518059161,    0.49761866763201545,
     0.02963552764599851,  -0.07576571478927333
};
static const double k_sym4_rec_lo[8] = {
     0.0322231006040427,   -0.012603967262037833,
    -0.09921954357684722,   0.29785779560527736,
     0.8037387518059161,    0.49761866763201545,
    -0.02963552764599851,  -0.07576571478927333
};
static const double k_sym4_rec_hi[8] = {
    -0.07576571478927333,   0.02963552764599851,
     0.49761866763201545,  -0.8037387518059161,
     0.29785779560527736,   0.09921954357684722,
    -0.012603967262037833, -0.0322231006040427
};

/* coif1 (Coiflet-1, order 6).  Vendored from PyWavelets 1.6.0. */
static const double k_coif1_dec_lo[6] = {
    -0.015655728135791993, -0.07273261951252645,
     0.3848648468648578,    0.8525720202116004,
     0.3378976624574818,   -0.07273261951252645
};
static const double k_coif1_dec_hi[6] = {
     0.07273261951252645,   0.3378976624574818,
    -0.8525720202116004,    0.3848648468648578,
     0.07273261951252645,  -0.015655728135791993
};
static const double k_coif1_rec_lo[6] = {
    -0.07273261951252645,   0.3378976624574818,
     0.8525720202116004,    0.3848648468648578,
    -0.07273261951252645,  -0.015655728135791993
};
static const double k_coif1_rec_hi[6] = {
    -0.015655728135791993,  0.07273261951252645,
     0.3848648468648578,   -0.8525720202116004,
     0.3378976624574818,    0.07273261951252645
};

/* ------------------------ Public helper functions ------------------------ */

int c4a_wavelet_family_from_name(const char* name, c4a_wavelet_family_t* out) {
    if (name == NULL || out == NULL) return -1;
    if (strcmp(name, "haar")  == 0) { *out = C4A_WAVELET_HAAR;  return 0; }
    if (strcmp(name, "db1")   == 0) { *out = C4A_WAVELET_HAAR;  return 0; }
    if (strcmp(name, "db4")   == 0) { *out = C4A_WAVELET_DB4;   return 0; }
    if (strcmp(name, "sym4")  == 0) { *out = C4A_WAVELET_SYM4;  return 0; }
    if (strcmp(name, "coif1") == 0) { *out = C4A_WAVELET_COIF1; return 0; }
    return -1;
}

int c4a_wavelet_mode_from_name(const char* name, c4a_wavelet_mode_t* out) {
    if (name == NULL || out == NULL) return -1;
    if (strcmp(name, "periodization") == 0) { *out = C4A_WAVELET_MODE_PERIODIZATION; return 0; }
    if (strcmp(name, "symmetric")     == 0) { *out = C4A_WAVELET_MODE_SYMMETRIC;     return 0; }
    if (strcmp(name, "zero")          == 0) { *out = C4A_WAVELET_MODE_ZERO;          return 0; }
    return -1;
}

int32_t c4a_wavelet_filter_length(c4a_wavelet_family_t family) {
    switch (family) {
        case C4A_WAVELET_HAAR:  return 2;
        case C4A_WAVELET_DB4:   return 8;
        case C4A_WAVELET_SYM4:  return 8;
        case C4A_WAVELET_COIF1: return 6;
    }
    return 0;
}

void c4a_wavelet_dec_filters(c4a_wavelet_family_t family,
                              double* out_lo, double* out_hi) {
    const int32_t L = c4a_wavelet_filter_length(family);
    const double* lo = NULL;
    const double* hi = NULL;
    switch (family) {
        case C4A_WAVELET_HAAR:  lo = k_haar_dec_lo;  hi = k_haar_dec_hi;  break;
        case C4A_WAVELET_DB4:   lo = k_db4_dec_lo;   hi = k_db4_dec_hi;   break;
        case C4A_WAVELET_SYM4:  lo = k_sym4_dec_lo;  hi = k_sym4_dec_hi;  break;
        case C4A_WAVELET_COIF1: lo = k_coif1_dec_lo; hi = k_coif1_dec_hi; break;
    }
    if (out_lo && lo) memcpy(out_lo, lo, (size_t)L * sizeof(double));
    if (out_hi && hi) memcpy(out_hi, hi, (size_t)L * sizeof(double));
}

void c4a_wavelet_rec_filters(c4a_wavelet_family_t family,
                              double* out_lo, double* out_hi) {
    const int32_t L = c4a_wavelet_filter_length(family);
    const double* lo = NULL;
    const double* hi = NULL;
    switch (family) {
        case C4A_WAVELET_HAAR:  lo = k_haar_rec_lo;  hi = k_haar_rec_hi;  break;
        case C4A_WAVELET_DB4:   lo = k_db4_rec_lo;   hi = k_db4_rec_hi;   break;
        case C4A_WAVELET_SYM4:  lo = k_sym4_rec_lo;  hi = k_sym4_rec_hi;  break;
        case C4A_WAVELET_COIF1: lo = k_coif1_rec_lo; hi = k_coif1_rec_hi; break;
    }
    if (out_lo && lo) memcpy(out_lo, lo, (size_t)L * sizeof(double));
    if (out_hi && hi) memcpy(out_hi, hi, (size_t)L * sizeof(double));
}

int64_t c4a_wavelet_dwt_output_length(int64_t n,
                                       c4a_wavelet_family_t family,
                                       c4a_wavelet_mode_t mode) {
    if (n <= 0) return 0;
    const int32_t L = c4a_wavelet_filter_length(family);
    if (L <= 0) return 0;
    if (mode == C4A_WAVELET_MODE_PERIODIZATION) {
        return (n + 1) / 2;   /* ceil(n / 2) */
    }
    return (n + L - 1) / 2;   /* floor((n + L - 1) / 2) */
}

int32_t c4a_wavelet_dwt_max_level(int64_t n, c4a_wavelet_family_t family) {
    const int32_t L = c4a_wavelet_filter_length(family);
    if (n <= 0 || L <= 1) return 0;
    /* pywt: max_level = floor(log2(n / (L - 1))).  For n < L - 1 this is
     * negative -> clamped to 0. */
    if (n < (int64_t)(L - 1)) return 0;
    int32_t lvl = 0;
    int64_t cur = n;
    const int64_t threshold = (int64_t)(L - 1);
    while ((cur / 2) >= threshold) {
        cur /= 2;
        ++lvl;
    }
    return lvl;
}

/* ---------------- Mode-specific single-level DWT cores -------------------
 *
 * Each helper writes ``out_len`` doubles to ``out_cA`` / ``out_cD`` (or
 * just to a single output buffer when used inside multi-level loops).
 */

static int64_t periodization_index(int64_t k, int64_t n_eff) {
    /* Euclidean modulus — works for any sign of k. */
    int64_t r = k % n_eff;
    return (r < 0) ? r + n_eff : r;
}

static void dwt_periodization(const double* in, int64_t n,
                               const double* h_lo, const double* h_hi,
                               int32_t L,
                               double* out_cA, double* out_cD,
                               double* x_eff_scratch) {
    /* When n is odd, pywt appends the last sample so n_eff = n + 1.  When
     * n is even, x_eff aliases ``in`` directly. */
    const int64_t n_eff = n + (n & 1);
    const double* x_eff = NULL;
    if ((n & 1) == 0) {
        x_eff = in;
    } else {
        if (x_eff_scratch == NULL) return;
        memcpy(x_eff_scratch, in, (size_t)n * sizeof(double));
        x_eff_scratch[n] = in[n - 1];
        x_eff = x_eff_scratch;
    }
    const int64_t out_len = n_eff / 2;
    const int32_t half_offset = L / 2 - 1;
    for (int64_t k = 0; k < out_len; ++k) {
        double accA = 0.0;
        double accD = 0.0;
        for (int32_t i = 0; i < L; ++i) {
            const int64_t idx =
                periodization_index((int64_t)(2 * k) + i - half_offset, n_eff);
            const double v = x_eff[idx];
            /* h_lo / h_hi are stored in forward order (low-pass and
             * high-pass analysis filters).  The DWT formula uses the
             * reversed filter; folding the reversal into the index gives
             * h[L - 1 - i] * x[...].  Equivalent: iterate i over [0, L)
             * and use h[L - 1 - i].  We keep the index explicit. */
            accA += h_lo[L - 1 - i] * v;
            accD += h_hi[L - 1 - i] * v;
        }
        out_cA[k] = accA;
        out_cD[k] = accD;
    }
}

static void dwt_symmetric_or_zero(const double* in, int64_t n,
                                   const double* h_lo, const double* h_hi,
                                   int32_t L,
                                   c4a_wavelet_mode_t mode,
                                   double* out_cA, double* out_cD) {
    /* Build the extended row of length n + 2*(L - 1).  For symmetric mode
     * the extension is edge-repeat reflection; for zero mode it is zeros.
     * The convolution of the forward filter with the extended row, taken
     * at positions ``L, L+2, L+4, ...``, gives the standard DWT output. */
    const int64_t ext_len = n + 2 * (int64_t)(L - 1);
    /* The convolution full length is ext_len + L - 1; we only need the
     * positions [L, L+2, ..., L + 2*(out_len-1)] which sit at offsets
     * (ext_idx, filter_idx) with ext_idx + filter_idx == sample index.
     * Each output sample touches at most L extended-row taps.  Instead of
     * materialising the extended row, we evaluate the indices on the fly.
     */
    const int64_t out_len = (n + L - 1) / 2;
    for (int64_t k = 0; k < out_len; ++k) {
        const int64_t base = (int64_t)L + 2 * k;  /* full-conv output index */
        double accA = 0.0;
        double accD = 0.0;
        for (int32_t i = 0; i < L; ++i) {
            /* extended index = base - i, accessing x_ext[ext_idx] */
            const int64_t ext_idx = base - i;
            if (ext_idx < 0 || ext_idx >= ext_len) continue;  /* leading zeros */
            double v;
            if (ext_idx < (int64_t)(L - 1)) {
                /* Left pad. */
                if (mode == C4A_WAVELET_MODE_ZERO) {
                    v = 0.0;
                } else {
                    /* Edge-repeat reflection: index in the left pad is
                     * (L-2-ext_idx); maps to in[L-2-ext_idx]. */
                    const int64_t pos = (int64_t)(L - 2) - ext_idx;
                    if (pos < 0 || pos >= n) continue;
                    v = in[pos];
                }
            } else if (ext_idx < (int64_t)(L - 1) + n) {
                v = in[ext_idx - (int64_t)(L - 1)];
            } else {
                /* Right pad. */
                if (mode == C4A_WAVELET_MODE_ZERO) {
                    v = 0.0;
                } else {
                    /* Edge-repeat reflection: position past last sample is
                     * (ext_idx - (L-1) - n), maps to in[n - 1 - q]. */
                    const int64_t q = ext_idx - (int64_t)(L - 1) - n;
                    const int64_t pos = (int64_t)(n - 1) - q;
                    if (pos < 0 || pos >= n) continue;
                    v = in[pos];
                }
            }
            accA += h_lo[i] * v;
            accD += h_hi[i] * v;
        }
        out_cA[k] = accA;
        out_cD[k] = accD;
    }
}

c4a_status_t c4a_wavelet_dwt(const double* in, int64_t n,
                              c4a_wavelet_family_t family,
                              c4a_wavelet_mode_t mode,
                              double* out_cA, double* out_cD) {
    if (in == NULL || out_cA == NULL || out_cD == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const int32_t L = c4a_wavelet_filter_length(family);
    if (L <= 0) return C4A_ERR_INVALID_ARGUMENT;
    double h_lo[C4A_WAVELET_MAX_FILTER];
    double h_hi[C4A_WAVELET_MAX_FILTER];
    c4a_wavelet_dec_filters(family, h_lo, h_hi);
    if (mode == C4A_WAVELET_MODE_PERIODIZATION) {
        double* scratch = NULL;
        double  stack_scratch[64];
        if (n & 1) {
            if (n + 1 <= (int64_t)(sizeof(stack_scratch) / sizeof(stack_scratch[0]))) {
                scratch = stack_scratch;
            } else {
                scratch = (double*)malloc((size_t)(n + 1) * sizeof(double));
                if (scratch == NULL) return C4A_ERR_OUT_OF_MEMORY;
            }
        }
        dwt_periodization(in, n, h_lo, h_hi, L, out_cA, out_cD, scratch);
        if (scratch != NULL && scratch != stack_scratch) free(scratch);
        return C4A_OK;
    }
    if (mode == C4A_WAVELET_MODE_SYMMETRIC || mode == C4A_WAVELET_MODE_ZERO) {
        dwt_symmetric_or_zero(in, n, h_lo, h_hi, L, mode, out_cA, out_cD);
        return C4A_OK;
    }
    return C4A_ERR_INVALID_ARGUMENT;
}

/* ----------------------- Single-level IDWT cores ------------------------- */

c4a_status_t c4a_wavelet_idwt(const double* cA, const double* cD, int64_t m,
                               int64_t n_out,
                               c4a_wavelet_family_t family,
                               c4a_wavelet_mode_t mode,
                               double* out) {
    if (cA == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (m < 1 || n_out < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    const int32_t L = c4a_wavelet_filter_length(family);
    if (L <= 0) return C4A_ERR_INVALID_ARGUMENT;
    double g_lo[C4A_WAVELET_MAX_FILTER];
    double g_hi[C4A_WAVELET_MAX_FILTER];
    c4a_wavelet_rec_filters(family, g_lo, g_hi);

    /* For symmetric / zero modes we need to compute the full-convolution
     * of the upsampled coefficients with the reconstruction filters and
     * trim ``[L-2 : L-2 + n_out]``.  For periodization we wrap the
     * convolution modulo 2*m and roll by L/2 - 1.
     *
     * Implementation: iterate over output positions, accumulating the
     * contribution of each upsampled tap (taps live at even positions in
     * the upsampled domain) without materialising the upsampled signal.
     */
    if (mode == C4A_WAVELET_MODE_SYMMETRIC || mode == C4A_WAVELET_MODE_ZERO) {
        /* full_conv has length 2*m + L - 1 == 2*m + L - 1, indexed
         * 0 .. 2*m + L - 2.  We need indices [L-2 .. L-2 + n_out - 1].
         * For each full-conv index ``f``, the upsample taps that contribute
         * are at ``f - i`` for i in [0, L), constrained to even,
         * non-negative, < 2*m. */
        for (int64_t out_idx = 0; out_idx < n_out; ++out_idx) {
            const int64_t f = (int64_t)(L - 2) + out_idx;
            double acc = 0.0;
            for (int32_t i = 0; i < L; ++i) {
                const int64_t up_idx = f - i;
                if (up_idx < 0 || up_idx >= 2 * m) continue;
                if ((up_idx & 1) != 0) continue;  /* zero at odd positions */
                const int64_t tap = up_idx / 2;
                const double v_cA = cA[tap];
                const double v_cD = (cD != NULL) ? cD[tap] : 0.0;
                acc += g_lo[i] * v_cA + g_hi[i] * v_cD;
            }
            out[out_idx] = acc;
        }
        return C4A_OK;
    }

    if (mode == C4A_WAVELET_MODE_PERIODIZATION) {
        const int64_t period = 2 * m;
        /* Compute the full-convolution length 2*m + L - 1, fold modulo
         * period, then roll by ``L/2 - 1`` so that the first element of
         * ``out`` corresponds to the original signal's first sample. */
        if (period <= 0) return C4A_ERR_INVALID_ARGUMENT;
        const int32_t roll = L / 2 - 1;
        double  stack_wrapped[64];
        double* wrapped = NULL;
        if (period <= (int64_t)(sizeof(stack_wrapped) / sizeof(stack_wrapped[0]))) {
            wrapped = stack_wrapped;
        } else {
            wrapped = (double*)malloc((size_t)period * sizeof(double));
            if (wrapped == NULL) return C4A_ERR_OUT_OF_MEMORY;
        }
        memset(wrapped, 0, (size_t)period * sizeof(double));
        for (int64_t f = 0; f < 2 * m + (int64_t)(L - 1); ++f) {
            double acc = 0.0;
            for (int32_t i = 0; i < L; ++i) {
                const int64_t up_idx = f - i;
                if (up_idx < 0 || up_idx >= 2 * m) continue;
                if ((up_idx & 1) != 0) continue;
                const int64_t tap = up_idx / 2;
                const double v_cA = cA[tap];
                const double v_cD = (cD != NULL) ? cD[tap] : 0.0;
                acc += g_lo[i] * v_cA + g_hi[i] * v_cD;
            }
            int64_t pos = f % period;
            if (pos < 0) pos += period;
            wrapped[pos] += acc;
        }
        for (int64_t k = 0; k < n_out; ++k) {
            int64_t src = (k + roll) % period;
            if (src < 0) src += period;
            out[k] = wrapped[src];
        }
        if (wrapped != stack_wrapped) free(wrapped);
        return C4A_OK;
    }

    return C4A_ERR_INVALID_ARGUMENT;
}

/* ---------------------- Multi-level decomposition ------------------------ */

c4a_status_t c4a_wavelet_wavedec_lengths(int64_t n,
                                          c4a_wavelet_family_t family,
                                          c4a_wavelet_mode_t mode,
                                          int32_t level,
                                          int64_t* out_total,
                                          int64_t* out_coef_lengths) {
    if (out_total == NULL || out_coef_lengths == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n < 1 || level < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    /* Compute per-level approximation lengths: a_0 = n, a_{k+1} = dwt_out(a_k). */
    int64_t cur = n;
    int64_t total = 0;
    /* The downward chain produces detail lengths d_1, d_2, ..., d_level.
     * Store them at out_coef_lengths[level], [level-1], ..., [1]. */
    for (int32_t k = 0; k < level; ++k) {
        const int64_t out_len =
            c4a_wavelet_dwt_output_length(cur, family, mode);
        out_coef_lengths[(int32_t)level - k] = out_len;
        cur = out_len;
        total += out_len;
    }
    /* Top-level approximation has the same length as the last detail. */
    out_coef_lengths[0] = cur;
    total += cur;
    *out_total = total;
    return C4A_OK;
}

c4a_status_t c4a_wavelet_wavedec(const double* in, int64_t n,
                                  c4a_wavelet_family_t family,
                                  c4a_wavelet_mode_t mode,
                                  int32_t level,
                                  double* coeffs,
                                  int64_t* coef_lengths,
                                  int64_t* offsets,
                                  int64_t coeffs_capacity) {
    if (in == NULL || coeffs == NULL || coef_lengths == NULL || offsets == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (n < 1 || level < 0) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    int64_t total = 0;
    const c4a_status_t lst =
        c4a_wavelet_wavedec_lengths(n, family, mode, level, &total, coef_lengths);
    if (lst != C4A_OK) return lst;
    if (coeffs_capacity < total) return C4A_ERR_SHAPE_MISMATCH;

    /* Allocate scratch for the running approximation. */
    double* cur = (double*)malloc((size_t)n * sizeof(double));
    if (cur == NULL) return C4A_ERR_OUT_OF_MEMORY;
    memcpy(cur, in, (size_t)n * sizeof(double));
    int64_t cur_len = n;

    /* Detail buffers go from finest to coarsest, but the output is laid
     * out coarsest-first.  We compute level by level and stage detail
     * coefficients at the right offset using the lengths table. */
    /* Compute offsets: offsets[0] = 0 -> top cA; offsets[1] -> top cD;
     * offsets[2] -> cD_{top-1}; ... offsets[level] -> cD_1. */
    offsets[0] = 0;
    int64_t acc = coef_lengths[0];
    for (int32_t k = 1; k <= level; ++k) {
        offsets[k] = acc;
        acc += coef_lengths[k];
    }

    /* Buffers for one DWT step: cA and cD of length next_len. */
    int64_t max_step = c4a_wavelet_dwt_output_length(n, family, mode);
    /* In all subsequent steps, the input length is even smaller, so
     * ``max_step`` is a safe upper bound. */
    double* next_cA = (double*)malloc((size_t)max_step * sizeof(double));
    double* next_cD = (double*)malloc((size_t)max_step * sizeof(double));
    if (next_cA == NULL || next_cD == NULL) {
        free(cur); free(next_cA); free(next_cD);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    for (int32_t k = 0; k < level; ++k) {
        const int64_t out_len =
            c4a_wavelet_dwt_output_length(cur_len, family, mode);
        const c4a_status_t st =
            c4a_wavelet_dwt(cur, cur_len, family, mode, next_cA, next_cD);
        if (st != C4A_OK) {
            free(cur); free(next_cA); free(next_cD);
            return st;
        }
        /* Detail at iteration k corresponds to detail level (level - k). */
        const int32_t out_slot = level - k;
        memcpy(coeffs + offsets[out_slot], next_cD,
               (size_t)coef_lengths[out_slot] * sizeof(double));
        /* Approximation becomes the next input. */
        memcpy(cur, next_cA, (size_t)out_len * sizeof(double));
        cur_len = out_len;
    }
    /* Final approximation -> coeffs[offsets[0] : offsets[0] + cur_len]. */
    memcpy(coeffs + offsets[0], cur, (size_t)cur_len * sizeof(double));

    free(cur);
    free(next_cA);
    free(next_cD);
    return C4A_OK;
}

c4a_status_t c4a_wavelet_waverec(const double* coeffs,
                                  const int64_t* coef_lengths,
                                  const int64_t* offsets,
                                  int32_t level,
                                  c4a_wavelet_family_t family,
                                  c4a_wavelet_mode_t mode,
                                  int64_t n_out,
                                  double* out) {
    if (coeffs == NULL || coef_lengths == NULL || offsets == NULL || out == NULL) {
        return C4A_ERR_NULL_POINTER;
    }
    if (level < 0 || n_out < 1) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    if (level == 0) {
        /* Trivial: copy approximation. */
        memcpy(out, coeffs + offsets[0], (size_t)n_out * sizeof(double));
        return C4A_OK;
    }
    /* Start with the top-level approximation, then iteratively idwt
     * against each detail level from coarsest to finest.  The
     * intermediate reconstruction length at step k follows the same
     * chain as forward DWT, but in reverse. */

    /* We need to know the intermediate signal lengths.  Forward chain
     * produced approximation lengths a_0 = n_out, a_1, ..., a_level
     * where a_k = c4a_wavelet_dwt_output_length(a_{k-1}, family, mode).
     * coef_lengths[0] = a_level (top approx + top detail length).
     * The reconstruction at step k (for k = level-1 down to 0) yields
     * length a_k. */
    int64_t  approx_lens[64];
    if (level + 1 > (int32_t)(sizeof(approx_lens) / sizeof(approx_lens[0]))) {
        return C4A_ERR_INVALID_ARGUMENT;
    }
    approx_lens[0] = n_out;
    for (int32_t k = 1; k <= level; ++k) {
        approx_lens[k] = c4a_wavelet_dwt_output_length(
            approx_lens[k - 1], family, mode);
    }
    if (approx_lens[level] != coef_lengths[0]) {
        return C4A_ERR_SHAPE_MISMATCH;
    }

    /* Allocate two scratch buffers and ping-pong. */
    const int64_t max_len = n_out;  /* upper bound at the bottom step */
    double* cur = (double*)malloc((size_t)max_len * sizeof(double));
    if (cur == NULL) return C4A_ERR_OUT_OF_MEMORY;
    int64_t cur_len = coef_lengths[0];
    memcpy(cur, coeffs + offsets[0], (size_t)cur_len * sizeof(double));

    double* next = (double*)malloc((size_t)max_len * sizeof(double));
    if (next == NULL) {
        free(cur);
        return C4A_ERR_OUT_OF_MEMORY;
    }

    /* Reconstruction iterates from the top of the tree down to the
     * leaves.  ``coef_lengths`` is laid out as
     *   [cA_top, cD_level, cD_{level-1}, ..., cD_1]
     * so at step ``step`` (1-based) we read the detail at index
     * ``step`` and aim for the approximation length at level
     * ``level - step`` (== approx_lens[level - step]). */
    for (int32_t step = 1; step <= level; ++step) {
        const int64_t det_len = coef_lengths[step];
        if (det_len != cur_len) {
            free(cur); free(next);
            return C4A_ERR_SHAPE_MISMATCH;
        }
        const int64_t target_len = approx_lens[level - step];
        const c4a_status_t st = c4a_wavelet_idwt(
            cur, coeffs + offsets[step], cur_len, target_len,
            family, mode, next);
        if (st != C4A_OK) {
            free(cur); free(next);
            return st;
        }
        memcpy(cur, next, (size_t)target_len * sizeof(double));
        cur_len = target_len;
    }
    memcpy(out, cur, (size_t)n_out * sizeof(double));
    free(cur);
    free(next);
    return C4A_OK;
}
