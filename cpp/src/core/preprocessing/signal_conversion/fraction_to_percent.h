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
#ifndef CHEMOMETRICS4ALL_CORE_PP_SIGNAL_CONVERSION_FRACTION_TO_PERCENT_H
#define CHEMOMETRICS4ALL_CORE_PP_SIGNAL_CONVERSION_FRACTION_TO_PERCENT_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_frac_to_pct_state_t c4a_pp_frac_to_pct_state_t;

c4a_pp_frac_to_pct_state_t* c4a_pp_frac_to_pct_state_new(void);

void c4a_pp_frac_to_pct_state_free(c4a_pp_frac_to_pct_state_t* state);

c4a_status_t c4a_pp_frac_to_pct_apply(
    const c4a_pp_frac_to_pct_state_t* state,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SIGNAL_CONVERSION_FRACTION_TO_PERCENT_H */
