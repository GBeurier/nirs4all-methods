/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Boundary index lookup for 1-D padding used by SavitzkyGolay (scipy.signal
 * convention) and Gaussian (scipy.ndimage convention).
 *
 * Modes mirror SciPy.ndimage semantics on a length-N row:
 *
 *   reflect (ndimage)        : "edge repeat" reflection.   For N=4 the row
 *                              [a b c d] extends as [b a | a b c d | d c]
 *                              with period 2*N.  Index -1 maps to 0, -2 to 1.
 *   mirror (signal/ndimage)  : "no edge repeat" reflection.  For N=4 the row
 *                              extends as [c b | a b c d | c b] with period
 *                              2*(N-1).  Index -1 maps to 1, -2 to 2.  This
 *                              is also the SciPy `savgol_filter(mode="mirror")`
 *                              behaviour and the default for ndimage spline
 *                              filters.
 *   nearest                  : clamp to the nearest edge sample (a or d).
 *   wrap                     : periodic (modulo N).
 *   constant                 : fictitious sample equal to the requested cval.
 *                              When `constant` is selected the helper returns
 *                              -1 to signal "the caller must use cval here".
 *
 * The helper resolves an out-of-range index `k` (which may be negative or
 * >= N) to a valid index in [0, N), or -1 when the mode is `constant`. It
 * never accesses memory; callers feed it scalar indices and load row[i] (or
 * cval) themselves. The implementation is header-inlined to give the
 * compiler the chance to fold the mode dispatch out of inner loops.
 */
#ifndef CHEMOMETRICS4ALL_CORE_COMMON_PADDING_H
#define CHEMOMETRICS4ALL_CORE_COMMON_PADDING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum c4a_pad_mode_t {
    C4A_PAD_REFLECT  = 0,   /* ndimage: no-edge-repeat reflection */
    C4A_PAD_CONSTANT = 1,
    C4A_PAD_NEAREST  = 2,
    C4A_PAD_MIRROR   = 3,   /* edge-repeat reflection (savgol "mirror") */
    C4A_PAD_WRAP     = 4
} c4a_pad_mode_t;

/* Resolve an out-of-range index `k` to a valid index in [0, n) under `mode`.
 *
 * Returns:
 *   - The clamped / reflected / wrapped index in [0, n) for all modes
 *     except C4A_PAD_CONSTANT.
 *   - -1 when `mode == C4A_PAD_CONSTANT` and `k` is outside [0, n) — the
 *     caller substitutes `cval`.
 *   - `k` itself when 0 <= k < n (the in-range fast path).
 *
 * n must be > 0. */
int64_t c4a_pad_resolve_index(int64_t k, int64_t n, c4a_pad_mode_t mode);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_COMMON_PADDING_H */
