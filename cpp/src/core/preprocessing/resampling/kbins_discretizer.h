/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for IntegerKBinsDiscretizer (sklearn KBinsDiscretizer
 * with integer-labelled output). The public C ABI lives in c4a.h and is
 * implemented in c_api/c_api_resampling.cpp.
 *
 * Reference: nirs4all.operators.transforms.targets.IntegerKBinsDiscretizer
 *            (which wraps sklearn.preprocessing.KBinsDiscretizer
 *             with `encode='ordinal'`).
 *
 * fit(X):  per column j, compute `bin_edges_[j]` of length n_bins + 1:
 *   strategy = uniform :  edges = linspace(col.min(), col.max(), n_bins + 1)
 *   strategy = quantile:  edges = percentile(col, linspace(0, 100, n_bins + 1))
 *                                using linear interpolation
 * After computing, sklearn drops bins whose width is <= 1e-8 (per column).
 *
 * transform(X):  per column j,
 *   out[:, j] = searchsorted(bin_edges_[j][1:-1], X[:, j], side='right')
 *   (clipped naturally to [0, n_bins_[j] - 1] since side='right' on inner
 *    edges produces exactly that range).
 *
 * Output dtype is int32 (nirs4all casts the float result back to int32).
 *
 * Strategy codes:
 *   0 — uniform
 *   1 — quantile
 */
#ifndef CHEMOMETRICS4ALL_CORE_PP_RESAMPLING_KBINS_DISCRETIZER_H
#define CHEMOMETRICS4ALL_CORE_PP_RESAMPLING_KBINS_DISCRETIZER_H

#include "chemometrics4all/c4a.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct c4a_pp_kbins_disc_state_t c4a_pp_kbins_disc_state_t;

/* Allocate an IntegerKBinsDiscretizer state. `n_bins >= 2`. `strategy`:
 *   0 — uniform
 *   1 — quantile  */
c4a_pp_kbins_disc_state_t* c4a_pp_kbins_disc_state_new(int32_t n_bins,
                                                        int32_t strategy);

void c4a_pp_kbins_disc_state_free(c4a_pp_kbins_disc_state_t* state);

int c4a_pp_kbins_disc_state_is_fitted(const c4a_pp_kbins_disc_state_t* state);

/* Fit edges from the training matrix (rows x cols). The state stores
 * `bin_edges_[j]` per column. */
c4a_status_t c4a_pp_kbins_disc_state_fit(c4a_pp_kbins_disc_state_t* state,
                                          const double* X,
                                          int64_t rows, int64_t cols);

/* Apply per-column digitization. Output is row-major int32, same shape as
 * input (rows x cols). */
c4a_status_t c4a_pp_kbins_disc_state_apply(
    const c4a_pp_kbins_disc_state_t* state,
    const double* X,
    int64_t rows, int64_t cols,
    int32_t* out);

#ifdef __cplusplus
}
#endif

#endif /* CHEMOMETRICS4ALL_CORE_PP_RESAMPLING_KBINS_DISCRETIZER_H */
