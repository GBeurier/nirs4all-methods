/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Fractional Convolutional Kernel (FCK) builder — single helper that produces
 * one normalised, fractional-derivative-like kernel of length `kernel_size`
 * from the (alpha, sigma) pair.
 *
 * Algorithm matches the canonical Python reference
 *   nirs4all.operators.models.sklearn.fckpls.fractional_kernel_1d(alpha,
 *                                                                  sigma,
 *                                                                  kernel_size)
 * which constructs the kernel directly in the spatial domain:
 *
 *   idx[i]    = i - (k - 1) / 2,        i in [0, k)
 *   gaussian  = exp(-0.5 * (idx / max(sigma, 1e-6))^2)
 *   if alpha < 1e-10:
 *       h = gaussian
 *   else:
 *       sign     = sign(idx)            (0 at the centre)
 *       frac     = |idx|^alpha,         set to 0 where |idx| < 1e-10
 *       h        = gaussian * frac * sign
 *   if alpha > 0.1:
 *       h        = h - mean(h)
 *   L1-normalise: h <- h / sum(|h|)       (no-op if the L1 norm is < 1e-12)
 *
 * The conceptual frequency-domain formulation H(omega) ~ |omega|^alpha *
 * exp(-sigma * omega^2) motivates the kernel, but the canonical Python
 * implementation builds it directly in the spatial domain — no FFT/DFT is
 * required and parity is bit-exact when the input matches.
 *
 * The helper writes `kernel_size` doubles into `out` and is pure C with no
 * external dependencies. Returns 0 on success, -1 on validation failure
 * (kernel_size <= 0 or out == NULL).
 */
#ifndef CHEMOMETRICS4ALL_CORE_COMMON_FCK_KERNEL_H
#define CHEMOMETRICS4ALL_CORE_COMMON_FCK_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Build the (alpha, sigma) fractional convolutional kernel into `out`.
 *
 * INTERNAL — not part of the public C ABI. The kernel builder is consumed
 * exclusively by the §22 static FCK transformer (see c4a_pp_fck_static_*);
 * callers that need a fractional kernel must go through that handle. This
 * symbol has hidden visibility and is intentionally absent from c4a.h.
 *
 * Parameters
 *   alpha       fractional order, any real (0 → pure Gaussian)
 *   sigma       Gaussian envelope width (clamped to 1e-6 if smaller)
 *   kernel_size length of `out` (must be > 0; odd sizes recommended)
 *   out         destination buffer of `kernel_size` doubles
 *
 * Returns
 *   0  on success.
 *   -1 if kernel_size <= 0 or out == NULL.
 *
 * The resulting kernel is L1-normalised (sum |h_i| == 1) unless the raw
 * L1 norm falls below 1e-12, in which case the un-normalised kernel is
 * left in place (matches the Python `if norm > 1e-12` guard).
 */
int c4a_fck_kernel_1d(double alpha, double sigma, int32_t kernel_size,
                       double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_COMMON_FCK_KERNEL_H */
