/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Shared helpers for Phase 16 spectral / wavelength augmenters.
 *
 * The Gaussian kernel constructor and the convolution loop match
 * nirs4all's `_get_gaussian_kernel` and `_convolve_1d_batch` byte-for-byte:
 * both build a normalized Gaussian of the requested odd length and apply
 * scipy.ndimage.convolve1d(mode='reflect'). Because the kernel is symmetric
 * for order=0, scipy's internal `weights[::-1]` reversal is a no-op and
 * convolve1d collapses to a reflect-padded correlation — implemented here
 * via the shared `n4m_pad_resolve_index` helper from padding.h.
 *
 * Uniform doubles route through PCG64's `next_double` callback (which is
 * exactly `(uint64 >> 11) * 2^-53`); NumPy's Generator.uniform(a, b) reads
 * the same callback and produces `a + (b - a) * u`. Bit-exact.
 *
 * Bounded integer draws follow NumPy 1.26.4's `random_bounded_uint64`
 * (Lemire's accept/reject with 128-bit multiplication). NumPy uses this
 * path for any `Generator.integers(low, high)` call with `endpoint=False`
 * (the default) when the range fits in uint64.
 *
 * Linear interpolation matches np.interp's edge clamping + linear bracket.
 */

#include "spectral_common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/common/padding.h"

void n4m_aug_gauss_kernel_build(double sigma, int32_t width, double* out) {
    /* Caller guarantees sigma > 0 and width >= 1 odd. */
    const int32_t half = (width - 1) / 2;
    double sum = 0.0;
    for (int32_t i = 0; i < width; ++i) {
        const double x = (double)(i - half);
        const double v = exp(-(x * x) / (2.0 * sigma * sigma));
        out[i] = v;
        sum += v;
    }
    if (sum > 0.0) {
        const double inv = 1.0 / sum;
        for (int32_t i = 0; i < width; ++i) {
            out[i] *= inv;
        }
    }
}

static n4m_status_t n4m_aug_convolve_with_mode(
    const double* X, int64_t rows, int64_t cols,
    const double* kernel, int64_t klen,
    n4m_pad_mode_t pad_mode,
    double* out) {
    if (X == NULL || kernel == NULL || out == NULL) {
        return N4M_ERR_NULL_POINTER;
    }
    if (rows < 0 || cols < 0 || klen <= 0) {
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (rows == 0 || cols == 0) {
        return N4M_OK;
    }
    const int64_t half = klen / 2;
    double* scratch = (double*)malloc((size_t)cols * sizeof(double));
    if (scratch == NULL) {
        return N4M_ERR_OUT_OF_MEMORY;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const double* row_in  = X + (size_t)r * (size_t)cols;
        double*       row_out = out + (size_t)r * (size_t)cols;
        for (int64_t j = 0; j < cols; ++j) {
            double acc = 0.0;
            /* scipy.ndimage.convolve1d reverses the kernel internally; for a
             * symmetric Gaussian that is a no-op. The straight correlation
             * is `out[j] = sum_k kernel[k] * X[j + k - half]`. */
            for (int64_t k = 0; k < klen; ++k) {
                int64_t src = j + k - half;
                double v;
                if (src >= 0 && src < cols) {
                    v = row_in[src];
                } else {
                    const int64_t resolved =
                        n4m_pad_resolve_index(src, cols, pad_mode);
                    v = row_in[resolved];
                }
                acc += kernel[k] * v;
            }
            scratch[j] = acc;
        }
        memcpy(row_out, scratch, (size_t)cols * sizeof(double));
    }
    free(scratch);
    return N4M_OK;
}

n4m_status_t n4m_aug_convolve_reflect(
    const double* X, int64_t rows, int64_t cols,
    const double* kernel, int64_t klen,
    double* out) {
    return n4m_aug_convolve_with_mode(X, rows, cols, kernel, klen,
                                      N4M_PAD_REFLECT, out);
}

n4m_status_t n4m_aug_convolve_mirror(
    const double* X, int64_t rows, int64_t cols,
    const double* kernel, int64_t klen,
    double* out) {
    return n4m_aug_convolve_with_mode(X, rows, cols, kernel, klen,
                                      N4M_PAD_MIRROR, out);
}

double n4m_aug_uniform(n4m_rng_pcg64* rng, double lo, double hi) {
    /* (uint64 >> 11) * 2^-53 — same recipe as numpy's next_double callback. */
    const double u = n4m_pcg64_engine_next_double(rng);
    return lo + (hi - lo) * u;
}

/* 128-bit multiplication helper. Returns (hi, lo) such that
 * (uint128)a * b == ((uint128)hi << 64) | lo. */
static inline void n4m_aug_umul128(uint64_t a, uint64_t b,
                                    uint64_t* hi, uint64_t* lo) {
#if defined(__SIZEOF_INT128__)
    const __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
    *lo = (uint64_t)prod;
    *hi = (uint64_t)(prod >> 64);
#else
    /* Schoolbook 64x64 -> 128 via four 32x32 products. */
    const uint64_t a_lo = a & 0xFFFFFFFFu;
    const uint64_t a_hi = a >> 32;
    const uint64_t b_lo = b & 0xFFFFFFFFu;
    const uint64_t b_hi = b >> 32;
    const uint64_t ll = a_lo * b_lo;
    const uint64_t lh = a_lo * b_hi;
    const uint64_t hl = a_hi * b_lo;
    const uint64_t hh = a_hi * b_hi;
    const uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFu) + (hl & 0xFFFFFFFFu);
    *lo = (mid << 32) | (ll & 0xFFFFFFFFu);
    *hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
#endif
}

/* Internal state for NumPy's buffered 32-bit Lemire path.
 *
 * NumPy 1.26 dispatches `rng.integers(low, high, dtype=int64, endpoint=False)`
 * to `_rand_int64`, which picks the 32-bit path whenever `rng_excl =
 * high - low` fits in uint32 (`rng_excl <= 0xFFFFFFFF`). In that path the
 * raw uint64 stream is buffered: each draw pulls a full uint64 and emits
 * TWO 32-bit halves (low half first, then high half). Lemire's accept/
 * reject runs on the 32-bit values against `rng_excl`.
 *
 * For widths > 2^32 the unbuffered 64-bit Lemire is used directly.
 *
 * The augmenter helper hides this dispatch behind a stateless `randint`
 * function; the buffer state lives in static-thread-local storage tied to
 * the supplied PCG64 handle. Each randint call extracts the next 32-bit
 * half from the buffer (refilling from the engine when empty) and runs
 * Lemire's accept/reject against rng_excl until acceptance.
 *
 * Determinism: with a fixed seed, the buffer ordering (low half then high
 * half) yields the exact NumPy stream order. The buffer state is per-call;
 * a new draw context refills from the engine, so call boundaries advance
 * the engine by the same number of uint64s as NumPy. Tests reseed the
 * engine before each apply (via `n4m_rng_pcg64_set_seed`) so buffer state
 * carryover never breaks parity.
 *
 * To keep that contract simple, the buffer is held in the PCG64 engine's
 * caller-supplied wrapper rather than as a static — but since the public
 * `n4m_rng_pcg64_state_t` is opaque we instead refill on every call and
 * consume two halves per uint64 only when the second consumer in the same
 * `randint` call needs another draw. The end-to-end effect is that for
 * any pure stream of `randint(0, n)` calls with n <= 2^32 the engine
 * advances exactly ceil(n_draws / 2) uint64s — same as NumPy.
 *
 * NOTE: NumPy's actual buffered path keeps the second half across multiple
 * `integers` calls. To reproduce that the buffer must be persistent. We
 * implement it via two engine-state fields exposed by the spectral_common
 * augmenter ABI: `n4m_aug_randint_state_t`. Each operator builds one in
 * `_apply` and threads it through the inner loop.
 */

void n4m_aug_randint_state_reset(n4m_aug_randint_state_t* state,
                                  n4m_rng_pcg64* rng) {
    state->rng         = rng;
    state->buf_value   = 0;
    state->buf_has_high = 0;
}

static uint32_t n4m_aug_next_uint32(n4m_aug_randint_state_t* state) {
    if (state->buf_has_high) {
        state->buf_has_high = 0;
        return (uint32_t)(state->buf_value >> 32);
    }
    state->buf_value = n4m_pcg64_engine_next_uint64(state->rng);
    state->buf_has_high = 1;
    return (uint32_t)(state->buf_value & 0xFFFFFFFFu);
}

int64_t n4m_aug_randint(n4m_aug_randint_state_t* state,
                         int64_t lo, int64_t hi) {
    /* NumPy 1.26 `_rand_int64` dispatch:
     *   rng_excl = hi - lo (exclusive width)
     *   if rng_excl <= 2^32: 32-bit Lemire on buffered 32-bit halves.
     *   else                 : 64-bit Lemire on raw uint64.
     */
    const uint64_t rng_excl = (uint64_t)(hi - lo);
    if (rng_excl == 0u) {
        return lo;  /* defensive: zero-width range */
    }
    if (rng_excl <= 0xFFFFFFFFu) {
        /* 32-bit Lemire with buffered halves. */
        const uint32_t excl32 = (uint32_t)rng_excl;
        uint32_t x = n4m_aug_next_uint32(state);
        uint64_t m = (uint64_t)x * (uint64_t)excl32;
        uint32_t m_hi = (uint32_t)(m >> 32);
        uint32_t m_lo = (uint32_t)(m & 0xFFFFFFFFu);
        if (m_lo < excl32) {
            const uint32_t t = ((uint32_t)0u - excl32) % excl32;
            while (m_lo < t) {
                x = n4m_aug_next_uint32(state);
                m = (uint64_t)x * (uint64_t)excl32;
                m_hi = (uint32_t)(m >> 32);
                m_lo = (uint32_t)(m & 0xFFFFFFFFu);
            }
        }
        return (int64_t)m_hi + lo;
    }
    /* 64-bit Lemire on raw uint64. */
    uint64_t x = n4m_pcg64_engine_next_uint64(state->rng);
    uint64_t m_hi = 0;
    uint64_t m_lo = 0;
    n4m_aug_umul128(x, rng_excl, &m_hi, &m_lo);
    if (m_lo < rng_excl) {
        const uint64_t t = ((uint64_t)0u - rng_excl) % rng_excl;
        while (m_lo < t) {
            x = n4m_pcg64_engine_next_uint64(state->rng);
            n4m_aug_umul128(x, rng_excl, &m_hi, &m_lo);
        }
    }
    return (int64_t)m_hi + lo;
}

void n4m_aug_interp_linear(
    const double* x_query, int64_t n_query,
    const double* x_known, const double* y_known, int64_t n_known,
    double* out) {
    /* Caller guarantees n_known >= 1. */
    if (n_known <= 0 || n_query <= 0) {
        return;
    }
    const double xf  = x_known[0];
    const double xl  = x_known[n_known - 1];
    const double yf  = y_known[0];
    const double yl  = y_known[n_known - 1];
    for (int64_t i = 0; i < n_query; ++i) {
        const double xq = x_query[i];
        if (xq <= xf) {
            out[i] = yf;
            continue;
        }
        if (xq >= xl) {
            out[i] = yl;
            continue;
        }
        /* Binary search for the upper bracketing index in [1, n_known-1]. */
        int64_t lo = 1;
        int64_t hi = n_known - 1;
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (x_known[mid] >= xq) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        const int64_t r = lo;
        const double x0 = x_known[r - 1];
        const double x1 = x_known[r];
        const double y0 = y_known[r - 1];
        const double y1 = y_known[r];
        const double dx = x1 - x0;
        const double t = (dx != 0.0) ? (xq - x0) / dx : 0.0;
        out[i] = y0 + t * (y1 - y0);
    }
}
