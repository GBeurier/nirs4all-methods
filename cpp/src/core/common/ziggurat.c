/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Ziggurat algorithm for the standard normal distribution.
 *
 * Vendored from NumPy 1.26.4's `random_standard_normal` in
 * `numpy/random/src/distributions/distributions.c` (BSD-3-Clause). The
 * 256-region Ziggurat construction was originally described by Marsaglia &
 * Tsang (2000) and refined by NumPy upstream for numerical stability in
 * the tail (the `log1p` trick to dodge `log(0)`).
 *
 * Bit-exact parity invariants matched against NumPy:
 *
 *   - We read one uint64 per inner iteration via `c4a_rng_pcg64_next_uint64`.
 *   - Bits are sliced as `idx = r & 0xff; r >>= 8; sign = r & 1;
 *                          rabs = (r >> 1) & 0x000fffffffffffff` (52 bits).
 *   - The "wedge" rejection draws an extra uniform double via
 *     `c4a_pcg64_engine_next_double` (which is `(uint64 >> 11) * 2^-53`).
 *   - The tail draws two additional uniforms via `next_double`.
 *
 * Any deviation from these draw counts/orders breaks parity. Don't refactor.
 */

#include "rng_pcg64.h"
#include "ziggurat_constants.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

void c4a_pcg64_engine_standard_normal_fill(c4a_rng_pcg64* rng, double* out,
                                            size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (;;) {
            uint64_t r    = c4a_pcg64_engine_next_uint64(rng);
            int      idx  = (int)(r & 0xffu);
            r >>= 8;
            int      sign = (int)(r & 0x1u);
            uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
            double   x    = (double)rabs * wi_double[idx];
            if (sign != 0) {
                x = -x;
            }
            if (rabs < ki_double[idx]) {
                out[i] = x;
                break;
            }
            if (idx == 0) {
                /* Tail rejection — Marsaglia (1964) tail trick, with the
                 * `1.0 - U` rewriting `log1p(-U)` to avoid `log(0)`. */
                for (;;) {
                    double xx = -ziggurat_nor_inv_r *
                                log1p(-c4a_pcg64_engine_next_double(rng));
                    double yy = -log1p(-c4a_pcg64_engine_next_double(rng));
                    if (yy + yy > xx * xx) {
                        double v = ziggurat_nor_r + xx;
                        out[i]   = ((rabs >> 8) & 0x1u) ? -v : v;
                        goto next_sample;
                    }
                }
            } else {
                /* Wedge rejection — accept if the random uniform falls
                 * inside the parabola defined by fi_double[idx-1], idx. */
                if (((fi_double[idx - 1] - fi_double[idx]) *
                         c4a_pcg64_engine_next_double(rng) +
                     fi_double[idx]) < exp(-0.5 * x * x)) {
                    out[i] = x;
                    break;
                }
            }
        }
next_sample:
        ;
    }
}
