# SPDX-License-Identifier: CECILL-2.1
"""
SpikeNoise — frozen NumPy reference.

Adds 0..N spikes at random wavelength positions per sample. Mirrors
``nirs4all.operators.augmentation.spectral.SpikeNoise`` with the integer /
uniform draws unrolled to explicit ``floor(u * range)`` mappings so the
C implementation can match without re-implementing NumPy's Lemire
rejection for bounded integers.

RNG draw order (locked):

    1. n_samples uniforms for per-row spike count
    2. n_samples * n_spikes_max uniforms for candidate indices
    3. n_samples * n_spikes_max uniforms for candidate amplitudes

For each row we take the first ``n_spikes[i]`` indices, ``np.unique``
them (sorted unique), then add ``amplitudes[0:len_uniq]`` at those
positions. This mirrors the nirs4all quirk where the amplitude is
sliced by the unique count, not paired by index.
"""

from __future__ import annotations

import numpy as np


__all__ = ["spike_noise_apply"]


def spike_noise_apply(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    n_spikes_min: int,
    n_spikes_max: int,
    amplitude_min: float,
    amplitude_max: float,
) -> np.ndarray:
    """Apply spike noise to spectra.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features), dtype float64
    rng : np.random.Generator
    n_spikes_min, n_spikes_max : int
        Inclusive range for per-row spike count; each row draws an integer
        in ``[n_spikes_min, n_spikes_max]``.
    amplitude_min, amplitude_max : float
        Half-open range for spike amplitudes.

    Returns
    -------
    out : np.ndarray, same shape and dtype as X
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got {X.ndim}-D")
    if n_spikes_min < 0 or n_spikes_max < n_spikes_min:
        raise ValueError("require 0 <= n_spikes_min <= n_spikes_max")
    if amplitude_max < amplitude_min:
        raise ValueError("require amplitude_min <= amplitude_max")

    n_samples, n_features = X.shape
    out = X.copy()
    if n_samples == 0 or n_features == 0 or n_spikes_max == 0:
        return out

    n_choice = n_spikes_max - n_spikes_min + 1
    amp_span = amplitude_max - amplitude_min

    # 1. Per-row spike count via floor(u * n_choice) + n_min.
    count_draws = rng.random(size=n_samples)
    n_spikes_per_sample = np.floor(count_draws * n_choice).astype(np.int64)
    n_spikes_per_sample = np.minimum(n_spikes_per_sample, n_spikes_max - n_spikes_min)
    n_spikes_per_sample += n_spikes_min

    # 2. Candidate indices: floor(u * n_features).
    idx_draws = rng.random(size=(n_samples, n_spikes_max))
    all_indices = np.floor(idx_draws * n_features).astype(np.int64)
    np.minimum(all_indices, n_features - 1, out=all_indices)

    # 3. Candidate amplitudes: amp_min + u * span.
    amp_draws = rng.random(size=(n_samples, n_spikes_max))
    all_amplitudes = amplitude_min + amp_span * amp_draws

    # 4. Per-row apply: take first n_spikes_per_sample[i] entries, sort+dedup,
    #    then add amplitudes[0..len_uniq] at the sorted unique positions.
    for i in range(n_samples):
        n_this = int(n_spikes_per_sample[i])
        if n_this <= 0:
            continue
        uniq = np.unique(all_indices[i, :n_this])
        amps = all_amplitudes[i, : uniq.shape[0]]
        out[i, uniq] += amps

    return out
