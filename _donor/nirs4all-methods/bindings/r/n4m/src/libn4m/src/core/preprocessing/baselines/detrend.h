/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Detrend — polynomial baseline subtraction along axis=1.
 *
 * For each row x of length n:
 *   coefs    = polyfit(positions=[0..n-1], x, polyorder)
 *   baseline = polyval(coefs, positions)
 *   out      = x - baseline
 *
 * Matches numpy: np.polyfit / np.polyval at the small QR-vs-SVD rounding
 * level (the internal parity fixture uses np.polyfit which goes through np.linalg.lstsq;
 * we use Householder QR through n4m_householder_qr).
 */
#ifndef N4M_CORE_PREPROCESSING_BASELINES_DETREND_H
#define N4M_CORE_PREPROCESSING_BASELINES_DETREND_H

#include <stdint.h>

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_detrend_state_t n4m_pp_detrend_state_t;

n4m_pp_detrend_state_t* n4m_pp_detrend_state_new(int32_t polyorder);
void                     n4m_pp_detrend_state_free(n4m_pp_detrend_state_t* state);

n4m_status_t n4m_pp_detrend_state_apply(const n4m_pp_detrend_state_t* state,
                                         const double* X,
                                         int64_t rows, int64_t cols,
                                         double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PREPROCESSING_BASELINES_DETREND_H */
