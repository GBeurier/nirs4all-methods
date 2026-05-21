/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * Internal C engine for the ScatterSimulationMSC augmenter.
 *
 * Reference: nirs4all.operators.augmentation.spectral.ScatterSimulationMSC.
 *
 * Algorithm (per sample):
 *   a ~ Uniform(a_lo, a_hi)
 *   b ~ Uniform(b_lo, b_hi)
 *   out[i, :] = a + b * X[i, :]
 *
 * The augmenter ignores wavelength input — the (a, b) pair is identical
 * regardless of the wavelength axis (the reference implementation has a
 * `reference_mode` parameter but only "self" mode is exercised here; the
 * "global_mean" branch is a no-op for this transform).
 */
#ifndef N4M_CORE_AUG_SCATTER_SIM_MSC_H
#define N4M_CORE_AUG_SCATTER_SIM_MSC_H

#include "n4m/n4m.h"

#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_scatter_sim_state_t n4m_aug_scatter_sim_state_t;

n4m_aug_scatter_sim_state_t* n4m_aug_scatter_sim_state_new(
    double a_low, double a_high, double b_low, double b_high);
void n4m_aug_scatter_sim_state_free(n4m_aug_scatter_sim_state_t* state);

n4m_status_t n4m_aug_scatter_sim_apply_impl(
    const n4m_aug_scatter_sim_state_t* state,
    n4m_rng_pcg64* rng,
    const double* X, int64_t rows, int64_t cols,
    double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_AUG_SCATTER_SIM_MSC_H */
