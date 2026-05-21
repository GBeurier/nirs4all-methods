/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * SpikeNoise — Phase 15 augmenter (n4m_aug_spike_noise_*).
 *
 * Adds a small number of spike values at random wavelengths per sample.
 *
 * Internal parity algorithm — bit-identical to the internal parity fixture
 * `n4m_parity_augmenters_ref.spike_noise`, which mirrors
 * `nirs4all.operators.augmentation.spectral.SpikeNoise` with uniform
 * `int(floor(u * range))` mapping in place of `rng.integers` (so we can
 * stay inside the PCG64 primitives available in `core/common/rng_pcg64`).
 *
 *     # per-row spike count: [n_min, n_max] inclusive
 *     count_draws[i]   = next_double()
 *     n_spikes[i]      = n_min + floor(count_draws[i] * (n_max - n_min + 1))
 *
 *     # (n_samples * n_max) candidate indices / amplitudes — row-major
 *     idx_draws[i, k]  = next_double()
 *     indices[i, k]    = floor(idx_draws[i, k] * n_features)
 *
 *     amp_draws[i, k]  = next_double()
 *     amps[i, k]       = amp_lo + (amp_hi - amp_lo) * amp_draws[i, k]
 *
 *     # collect first n_spikes[i] indices, sort+dedup via the nirs4all
 *     # `np.unique` quirk (returns sorted unique), then add the first
 *     # `len(unique)` amplitudes (NOT amplitude indexed by sample index!).
 *     uniq[i, .]       = sorted_unique(indices[i, :n_spikes[i]])
 *     out[i, uniq[i, k]] = X[i, uniq[i, k]] + amps[i, k]   # for k < len(uniq)
 *
 * All other positions are copied from X.
 *
 * RNG draw order (locked):
 *   - rows count draws,
 *   - rows * n_spikes_max index draws,
 *   - rows * n_spikes_max amplitude draws.
 */
#ifndef N4M_CORE_AUG_NOISE_SPIKE_NOISE_H
#define N4M_CORE_AUG_NOISE_SPIKE_NOISE_H

#include <stdint.h>

#include "n4m/n4m.h"
#include "core/common/rng_pcg64.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct n4m_aug_spike_noise_state_t n4m_aug_spike_noise_state_t;

n4m_aug_spike_noise_state_t* n4m_aug_spike_noise_state_new(
    n4m_rng_pcg64* rng,
    int32_t n_spikes_min,
    int32_t n_spikes_max,
    double  amplitude_min,
    double  amplitude_max);

void n4m_aug_spike_noise_state_free(n4m_aug_spike_noise_state_t* state);

n4m_status_t n4m_aug_spike_noise_state_apply(
    n4m_aug_spike_noise_state_t* state,
    const double* X, int64_t rows, int64_t cols, double* out);

#ifdef __cplusplus
}
#endif

#endif /* N4M_CORE_AUG_NOISE_SPIKE_NOISE_H */
