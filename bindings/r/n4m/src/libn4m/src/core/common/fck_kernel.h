/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Fractional Convolutional Kernel (FCK) builder — produces one normalised,
 * fractional-derivative-like kernel of length `kernel_size` from the
 * (alpha, scale, sigma) triple used by the static FCK transformer.
 *
 * Algorithm matches the canonical Python reference
 *   nirs4all.operators.transforms.fck_static._build_kernel(alpha,
 *                                                          scale,
 *                                                          kernel_size,
 *                                                          sigma)
 * which constructs the kernel directly in the spatial domain:
 *
 *   half      = (kernel_size - 1) / 2
 *   idx[i]    = (i - half) * scale,        i in [0, kernel_size)
 *   gaussian  = exp(-0.5 * (idx / sigma)^2)
 *   if alpha < 0.1:
 *       h = gaussian
 *   else:
 *       eps   = 1e-8
 *       frac  = sign(idx) * (|idx| + eps)^alpha   (sign(0) == 0)
 *       h     = gaussian * frac
 *       h     = h - mean(h)
 *   norm = sum(|h|) + 1e-8
 *   h    = h / norm
 *
 * The conceptual frequency-domain formulation H(omega) ~ |omega|^alpha *
 * exp(-sigma * omega^2) motivates the kernel, but the canonical Python
 * implementation builds it directly in the spatial domain — no FFT/DFT is
 * required and parity is bit-exact when the inputs match.
 *
 * The helper writes `kernel_size` doubles into `out` and is pure C with no
 * external dependencies. Returns 0 on success, -1 on validation failure
 * (out == NULL, kernel_size not odd and >= 3, or scale <= 0).
 */
#ifndef N4M_CORE_COMMON_FCK_KERNEL_H
#define N4M_CORE_COMMON_FCK_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Build the (alpha, scale, sigma) fractional convolutional kernel into `out`.
 *
 * INTERNAL — not part of the public C ABI. The kernel builder is consumed
 * exclusively by the §22 static FCK transformer (see n4m_pp_fck_static_*);
 * callers that need a fractional kernel must go through that handle. This
 * symbol has hidden visibility and is intentionally absent from n4m.h.
 *
 * Parameters
 *   alpha       fractional order, any real (alpha < 0.1 → pure Gaussian)
 *   scale       multiplier on the index axis (must be > 0 in practice)
 *   sigma       Gaussian envelope width (clamped to 1e-6 if smaller)
 *   kernel_size length of `out` (must be > 0; odd sizes are the supported
 *               case, matching the canonical Python implementation)
 *   out         destination buffer of `kernel_size` doubles
 *
 * Returns
 *   0  on success.
 *   -1 if validation fails.
 *
 * The resulting kernel is L1-normalised: h <- h / (sum(|h|) + 1e-8). The
 * additive epsilon matches the canonical Python implementation and removes
 * the need for a separate "norm too small" guard.
 */
int n4m_fck_kernel_1d(double alpha, double scale, double sigma,
                       int32_t kernel_size, double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_COMMON_FCK_KERNEL_H */
