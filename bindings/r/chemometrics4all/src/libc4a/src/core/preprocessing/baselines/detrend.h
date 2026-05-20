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
 * we use Householder QR through c4a_householder_qr).
 */
#ifndef CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_DETREND_H
#define CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_DETREND_H

#include <stdint.h>

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_detrend_state_t c4a_pp_detrend_state_t;

c4a_pp_detrend_state_t* c4a_pp_detrend_state_new(int32_t polyorder);
void                     c4a_pp_detrend_state_free(c4a_pp_detrend_state_t* state);

c4a_status_t c4a_pp_detrend_state_apply(const c4a_pp_detrend_state_t* state,
                                         const double* X,
                                         int64_t rows, int64_t cols,
                                         double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PREPROCESSING_BASELINES_DETREND_H */
