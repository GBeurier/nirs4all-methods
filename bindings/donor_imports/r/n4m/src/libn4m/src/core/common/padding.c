/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Implementation of n4m_pad_resolve_index. The branchy modes (reflect /
 * mirror) compute their canonical index through a single Euclidean modulus.
 *
 * `reflect` (ndimage convention, edge-repeat) — for x = [a b c d]:
 *     index sequence    : ..., a, b, c, d, d, c, b, a, a, b, c, d, ...
 *     period            : 2 * n
 *     canonical step    : 0 <= off < n  -> off
 *                          n <= off < 2n -> 2n - 1 - off
 *
 * `mirror` (signal / ndimage no-edge-repeat) — for x = [a b c d]:
 *     index sequence    : ..., d, c, b, a, b, c, d, c, b, a, b, c, ...
 *     period            : 2 * (n - 1)         (n must be >= 2; n == 1 is a
 *                          degenerate case returning 0).
 *     canonical step    : 0 <= off < n        -> off
 *                          n <= off < period   -> period - off
 *
 * The Euclidean modulus is implemented via `(k % p + p) % p` which works for
 * both positive and negative `k` (C's `%` is truncated, so we add `p` before
 * the second remainder).
 */

#include "padding.h"

static inline int64_t pad_mod(int64_t a, int64_t p) {
    /* Euclidean (always-non-negative) modulus. */
    const int64_t r = a % p;
    return (r < 0) ? r + p : r;
}

int64_t n4m_pad_resolve_index(int64_t k, int64_t n, n4m_pad_mode_t mode) {
    if (n <= 0) {
        return -1;
    }
    if (k >= 0 && k < n) {
        return k;
    }
    switch (mode) {
        case N4M_PAD_NEAREST: {
            if (k < 0)        return 0;
            return n - 1;
        }
        case N4M_PAD_WRAP: {
            return pad_mod(k, n);
        }
        case N4M_PAD_REFLECT: {
            if (n == 1) return 0;
            const int64_t period = 2 * n;
            int64_t off = pad_mod(k, period);
            if (off >= n) {
                off = 2 * n - 1 - off;
            }
            return off;
        }
        case N4M_PAD_MIRROR: {
            if (n == 1) return 0;
            const int64_t period = 2 * (n - 1);
            int64_t off = pad_mod(k, period);
            if (off >= n) {
                off = period - off;
            }
            return off;
        }
        case N4M_PAD_CONSTANT:
        default:
            return -1;
    }
}
