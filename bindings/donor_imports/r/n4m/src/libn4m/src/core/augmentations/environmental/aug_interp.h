/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * 1-D piecewise-linear interpolation helper matching NumPy's `np.interp`.
 *
 * Shared by the environmental augmenters (Temperature, Moisture) which
 * apply wavelength shifts via interpolation.
 *
 * Contract:
 *   - `xp` must be monotonically non-decreasing.
 *   - For `x < xp[0]` returns `fp[0]` (NumPy's left-extrapolation default).
 *   - For `x > xp[-1]` returns `fp[-1]` (NumPy's right-extrapolation default).
 *   - Inside the range, uses standard linear interpolation.
 *
 * Internal use only — not part of the public ABI.
 */
#ifndef N4M_CORE_AUG_INTERP_H
#define N4M_CORE_AUG_INTERP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void n4m_aug_np_interp(const double* x,  int64_t n_x,
                       const double* xp, const double* fp, int64_t n_xp,
                       double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_AUG_INTERP_H */
