/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * FractionToPercent — element-wise multiplication by 100.
 *
 * Reference: nirs4all.operators.transforms.signal_conversion.FractionToPercent
 *   X' = X * 100.0
 *
 * No tunable parameters. Placeholder state for ABI symmetry with the rest
 * of the preprocessing surface.
 */
#ifndef N4M_CORE_PP_SIGNAL_CONVERSION_FRACTION_TO_PERCENT_H
#define N4M_CORE_PP_SIGNAL_CONVERSION_FRACTION_TO_PERCENT_H

#include "n4m/n4m.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_pp_frac_to_pct_state_t n4m_pp_frac_to_pct_state_t;

n4m_pp_frac_to_pct_state_t* n4m_pp_frac_to_pct_state_new(void);

void n4m_pp_frac_to_pct_state_free(n4m_pp_frac_to_pct_state_t* state);

n4m_status_t n4m_pp_frac_to_pct_apply(
    const n4m_pp_frac_to_pct_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* N4M_CORE_PP_SIGNAL_CONVERSION_FRACTION_TO_PERCENT_H */
