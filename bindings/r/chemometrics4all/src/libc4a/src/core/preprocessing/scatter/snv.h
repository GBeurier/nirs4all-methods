/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the Standard Normal Variate (SNV) row-wise
 * preprocessing operator. The public C ABI lives in c4a.h and is
 * implemented in c_api/c_api_preprocessing.cpp which wraps this engine.
 *
 * Reference: nirs4all.operators.transforms.scalers.StandardNormalVariate
 * (operating on axis=1, with optional centering and scaling, ddof for std).
 *
 * Numerical contract:
 *   - mean is the arithmetic mean across cols of each row.
 *   - std is the population (ddof=0) or sample (ddof>0) standard deviation.
 *   - When std < 1e-15 (flat row), it is treated as 1.0 to avoid div-by-0;
 *     this matches NumPy's behaviour after the nirs4all-side `std[std == 0]
 *     = 1.0` guard with a slack to tolerate floating-point fuzz.
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_SCATTER_SNV_H
#define CHEMOMETRICS4ALL_CORE_PP_SCATTER_SNV_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_snv_state_t c4a_pp_snv_state_t;

/* Allocate a new SNV state. `with_mean` / `with_std` are booleans (any
 * non-zero int means enabled); `ddof` must be 0 or positive but is not
 * upper-bounded here (the apply step skips std division when ddof >= cols). */
c4a_pp_snv_state_t* c4a_pp_snv_state_new(int with_mean, int with_std, int ddof);

/* Release a state allocated by c4a_pp_snv_state_new. NULL-safe. */
void c4a_pp_snv_state_free(c4a_pp_snv_state_t* state);

/* Apply SNV row-wise on the row-major buffer X (shape rows x cols), writing
 * the result to `out` (same shape, may equal X for in-place operation). */
c4a_status_t c4a_pp_snv_apply(const c4a_pp_snv_state_t* state,
                              const double* X, int64_t rows, int64_t cols,
                              double* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_SCATTER_SNV_H */
