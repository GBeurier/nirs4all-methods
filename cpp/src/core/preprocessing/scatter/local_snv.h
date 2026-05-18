/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Local Standard Normal Variate (LSNV) — per-sample sliding window SNV
 * along the feature axis.
 *
 * Reference: nirs4all.operators.transforms.scalers.LocalStandardNormalVariate
 * which pads each row with `half = window // 2` samples (reflect, edge, or
 * constant), computes moving mean / mean-of-squares via cumsum, derives
 * mov_var = max(mov_mean2 - mov_mean^2, 0), and replaces the centre value
 * with (X - mov_mean) / mov_std (mov_std==0 → 1.0).
 *
 * The C engine reproduces the exact cumsum recurrence to match NumPy
 * bit-for-bit on the rounding sequence.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_SCATTER_LOCAL_SNV_H
#define CHEMOMETRICS4ALL_CORE_PP_SCATTER_LOCAL_SNV_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

/* c4a_pp_lsnv_pad_mode_t is declared in chemometrics4all/c4a.h (public). */

typedef struct c4a_pp_lsnv_state_t c4a_pp_lsnv_state_t;

/* `window` must be odd and >= 3. Returns NULL on allocation failure or
 * invalid parameters. */
c4a_pp_lsnv_state_t* c4a_pp_lsnv_state_new(int32_t window,
                                           c4a_pp_lsnv_pad_mode_t pad_mode,
                                           double constant_value);

void c4a_pp_lsnv_state_free(c4a_pp_lsnv_state_t* state);

c4a_status_t c4a_pp_lsnv_apply(const c4a_pp_lsnv_state_t* state,
                               const double* X, int64_t rows, int64_t cols,
                               double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SCATTER_LOCAL_SNV_H */
